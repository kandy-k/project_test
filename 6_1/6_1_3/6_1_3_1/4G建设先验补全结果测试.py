from __future__ import annotations

import sys
import csv
from dataclasses import dataclass
from pathlib import Path

try:
    import rasterio
    import numpy as np
except ImportError as exc:  # pragma: no cover - dependency check
    print(f"缺少依赖 rasterio 或 numpy，无法执行检查：{exc}", file=sys.stderr)
    sys.exit(2)


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
SPECTRUM_DIR = REPO_ROOT / "dataset" / "products" / "spectrum"
VALIDATION_DIR = REPO_ROOT / "dataset" / "validation" / "spectrum"
OUTPUT_DIR = SCRIPT_DIR / "outputs"
CORE_FILES = {
    "4G建设先验": SPECTRUM_DIR / "Global_LTE_60arcsec_GNN.tif",
    "行政区陆地掩膜": SPECTRUM_DIR / "全球国家行政区_aligned_60arcsec.tif",
}
OPTIONAL_FILES = {
    "原始4G基站栅格": SPECTRUM_DIR / "LTE_count_60arcsec.tif",
    "人口": SPECTRUM_DIR / "pop_landscan_2023_aligned_60arcsec.tif",
    "夜间灯光": SPECTRUM_DIR / "VNL_npp_2024_global_vcmslcfg_v2_c202502261200.average_masked.dat_aligned_60arcsec.tif",
    "建成环境体量": SPECTRUM_DIR / "GHS_BUILT_V_E2025_GLOBE_R2023A_4326_30ss_V1_0_aligned_60arcsec.tif",
    "城市化等级": SPECTRUM_DIR / "GHS_SMOD_E2025_GLOBE_R2023A_54009_1000_V2_0_aligned_60arcsec.tif",
}
RAW_CELL_TOWERS = VALIDATION_DIR / "cell_towers_2026-02-04-T000000.csv"
TARGET_RES_DEG = 1.0 / 60.0
TOL = 1e-9


@dataclass
class CheckResult:
    name: str
    passed: bool
    details: str


def raster_info(path: Path) -> dict[str, object]:
    with rasterio.open(path) as ds:
        return {
            "driver": ds.driver,
            "width": ds.width,
            "height": ds.height,
            "count": ds.count,
            "dtype": ds.dtypes[0],
            "crs": str(ds.crs) if ds.crs else None,
            "resolution": {"x": ds.res[0], "y": ds.res[1]},
            "bounds": {
                "left": ds.bounds.left,
                "bottom": ds.bounds.bottom,
                "right": ds.bounds.right,
                "top": ds.bounds.top,
            },
        }


def read_band(path: Path) -> tuple[np.ndarray, float | None]:
    with rasterio.open(path) as ds:
        return ds.read(1), ds.nodata


def valid_mask(arr: np.ndarray, nodata: float | None) -> np.ndarray:
    mask = np.isfinite(arr)
    if nodata is not None and np.isfinite(nodata):
        mask &= arr != nodata
    return mask


def describe(values: np.ndarray) -> dict[str, float | int | None]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "p05": None, "p50": None, "p95": None, "max": None, "mean": None}
    return {
        "count": int(values.size),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def sample_spearman(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float | None:
    try:
        from scipy.stats import spearmanr
    except ImportError:
        return None

    idx = np.flatnonzero(mask.ravel())
    if idx.size < 3:
        return None
    if idx.size > 200_000:
        rng = np.random.default_rng(42)
        idx = rng.choice(idx, 200_000, replace=False)
    rho, _ = spearmanr(x.ravel()[idx].astype(float), y.ravel()[idx].astype(float))
    return float(rho)


def positive_mask(arr: np.ndarray, nodata: float | None) -> np.ndarray:
    return valid_mask(arr, nodata) & (arr > 0)


def normalized_log_signal(arr: np.ndarray, nodata: float | None, base_mask: np.ndarray) -> np.ndarray:
    data = arr.astype(float, copy=False)
    valid = valid_mask(data, nodata)
    signal = np.where(valid & (data > 0), np.log1p(data), 0.0)
    positive_values = signal[base_mask & (signal > 0)]
    if positive_values.size == 0:
        return np.zeros_like(signal, dtype=float)
    scale = max(float(np.percentile(positive_values, 95)), 1e-12)
    return np.clip(signal / scale, 0.0, 1.0)


def build_demand_support_mask(base_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    signals: list[np.ndarray] = []
    score_layers: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    layer_names = ["人口", "夜间灯光", "建成环境体量"]

    for name in layer_names:
        path = OPTIONAL_FILES[name]
        if not path.is_file():
            rows.append({"signal": name, "pixel_count": 0, "note": "未提供，未参与补全率分母"})
            continue
        arr, nodata = read_band(path)
        signal = positive_mask(arr, nodata)
        signals.append(signal)
        score_layers.append(normalized_log_signal(arr, nodata, base_mask))
        rows.append({"signal": name, "pixel_count": int(np.sum(base_mask & signal)), "note": "正值像元参与需求支撑判断"})

    if signals:
        demand_support = base_mask & np.logical_or.reduce(signals)
    else:
        demand_support = base_mask.copy()

    if score_layers:
        demand_score = np.maximum.reduce(score_layers)
    else:
        demand_score = np.where(base_mask, 1.0, 0.0)

    rows.append({"signal": "需求支撑并集", "pixel_count": int(np.sum(demand_support)), "note": "人口、夜间灯光或建筑体量任一为正"})
    return demand_support, demand_score, rows


def calculate_completion_metrics(
    prior: np.ndarray,
    raw: np.ndarray,
    prior_valid: np.ndarray,
    raw_valid: np.ndarray,
    demand_support: np.ndarray,
    demand_score: np.ndarray,
    priority_percentile: float = 75.0,
) -> dict[str, object]:
    prior_positive = prior_valid & (prior > 0)
    raw_positive = raw_valid & (raw > 0)

    evaluation_mask = prior_valid & demand_support
    raw_zero_or_empty = evaluation_mask & ~raw_positive
    completed = raw_zero_or_empty & prior_positive

    priority_threshold = None
    priority_sparse = np.zeros_like(prior_valid, dtype=bool)
    priority_completed = np.zeros_like(prior_valid, dtype=bool)
    candidate_scores = demand_score[raw_zero_or_empty & np.isfinite(demand_score)]
    if candidate_scores.size > 0:
        priority_threshold = float(np.percentile(candidate_scores, priority_percentile))
        priority_sparse = raw_zero_or_empty & (demand_score >= priority_threshold)
        priority_completed = priority_sparse & prior_positive

    land_sparse = prior_valid & ~raw_positive
    land_completed = land_sparse & prior_positive
    raw_positive_count = int(np.sum(raw_positive))

    return {
        "raw_positive_count": raw_positive_count,
        "prior_on_raw_positive_count": int(np.sum(raw_positive & prior_positive)),
        "positive_expansion_factor": float(np.sum(prior_positive) / max(raw_positive_count, 1)),
        "demand_supported_pixel_count": int(np.sum(evaluation_mask)),
        "raw_zero_or_empty_count": int(np.sum(raw_zero_or_empty)),
        "completed_pixels_count": int(np.sum(completed)),
        "completion_ratio": float(np.sum(completed) / max(np.sum(raw_zero_or_empty), 1)),
        "priority_demand_percentile": priority_percentile,
        "priority_demand_threshold": priority_threshold,
        "priority_sparse_area_count": int(np.sum(priority_sparse)),
        "priority_completed_pixels_count": int(np.sum(priority_completed)),
        "priority_completion_ratio": float(np.sum(priority_completed) / max(np.sum(priority_sparse), 1)),
        "land_sparse_area_count": int(np.sum(land_sparse)),
        "land_sparse_completed_count": int(np.sum(land_completed)),
        "land_sparse_completion_ratio_reference": float(np.sum(land_completed) / max(np.sum(land_sparse), 1)),
    }


def same_grid(a: dict[str, object], b: dict[str, object]) -> bool:
    return (
        a["crs"] == b["crs"]
        and a["width"] == b["width"]
        and a["height"] == b["height"]
        and a["resolution"] == b["resolution"]
        and a["bounds"] == b["bounds"]
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_results_csv(path: Path, results: list[CheckResult]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "status", "details"])
        writer.writeheader()
        for result in results:
            writer.writerow({
                "name": result.name,
                "status": "通过" if result.passed else "未通过",
                "details": result.details,
            })


def write_inventory_csv(path: Path, infos: dict[str, dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "width",
                "height",
                "count",
                "dtype",
                "crs",
                "resolution_x",
                "resolution_y",
                "bounds",
            ],
        )
        writer.writeheader()
        for name, info in infos.items():
            resolution = info["resolution"]
            writer.writerow({
                "name": name,
                "width": info["width"],
                "height": info["height"],
                "count": info["count"],
                "dtype": info["dtype"],
                "crs": info["crs"],
                "resolution_x": resolution["x"],  # type: ignore[index]
                "resolution_y": resolution["y"],  # type: ignore[index]
                "bounds": info["bounds"],
            })


def write_dict_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, results: list[CheckResult], context: dict[str, object]) -> None:
    lines = [
        "4G建设先验补全结果测试摘要",
        f"检查目录：{context['spectrum_dir']}",
        f"检查结果：{sum(result.passed for result in results)}/{len(results)} 项通过",
    ]
    if "completion_metrics" in context:
        lines.extend(["", "补全指标："])
        for row in context["completion_metrics"]:  # type: ignore[union-attr]
            lines.append(f"- {row['metric']}：{row['value']} {row['unit']}")
    if "correlation_rows" in context:
        lines.extend(["", "辅助图层相关性："])
        for row in context["correlation_rows"]:  # type: ignore[union-attr]
            lines.append(f"- {row['layer']}：Spearman rho={row['spearman_rho']}")
    if "inventory_csv" in context:
        lines.append(f"栅格清单：{context['inventory_csv']}")
    if "results_csv" in context:
        lines.append(f"检查结果表：{context['results_csv']}")
    lines.extend(["", "详细结果："])
    lines.extend(f"- [{'通过' if result.passed else '未通过'}] {result.name}：{result.details}" for result in results)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(results: list[CheckResult], context: dict[str, object]) -> None:
    ensure_dir(OUTPUT_DIR)
    results_csv = OUTPUT_DIR / "prior_completion_results.csv"
    metrics_csv = OUTPUT_DIR / "prior_completion_metrics.csv"
    correlation_csv = OUTPUT_DIR / "prior_auxiliary_correlations.csv"
    summary_txt = OUTPUT_DIR / "prior_completion_summary.txt"
    write_results_csv(results_csv, results)
    context["results_csv"] = results_csv
    context["metrics_csv"] = metrics_csv
    context["correlation_csv"] = correlation_csv
    write_dict_rows(metrics_csv, context.get("completion_metrics", []), ["metric", "value", "unit", "note"])  # type: ignore[arg-type]
    write_dict_rows(correlation_csv, context.get("correlation_rows", []), ["layer", "spearman_rho", "sample_available"])  # type: ignore[arg-type]
    if "infos" in context:
        inventory_csv = OUTPUT_DIR / "spectrum_raster_inventory.csv"
        write_inventory_csv(inventory_csv, context["infos"])  # type: ignore[arg-type]
        context["inventory_csv"] = inventory_csv
    write_summary(summary_txt, results, context)
    context["summary_txt"] = summary_txt


def build_results() -> tuple[list[CheckResult], dict[str, object]]:
    results: list[CheckResult] = []
    context: dict[str, object] = {
        "spectrum_dir": SPECTRUM_DIR,
        "validation_dir": VALIDATION_DIR,
        "output_dir": OUTPUT_DIR,
    }

    results.append(CheckResult("spectrum 目录存在", SPECTRUM_DIR.is_dir(), f"目录路径：{SPECTRUM_DIR}"))

    missing = [name for name, path in CORE_FILES.items() if not path.is_file()]
    results.append(
        CheckResult(
            "4G建设先验核心数据齐全",
            not missing,
            "全部必需文件存在" if not missing else f"缺少文件：{', '.join(missing)}",
        )
    )
    if missing:
        return results, context

    infos = {name: raster_info(path) for name, path in CORE_FILES.items()}
    infos.update({name: raster_info(path) for name, path in OPTIONAL_FILES.items() if path.is_file()})
    context["infos"] = infos

    prior_info = infos["4G建设先验"]
    res = prior_info["resolution"]
    resolution_ok = abs(float(res["x"]) - TARGET_RES_DEG) <= TOL and abs(float(res["y"]) - TARGET_RES_DEG) <= TOL  # type: ignore[index]
    results.append(
        CheckResult(
            "4G建设先验分辨率为60arcsec",
            resolution_ok,
            f"resolution={res}",
        )
    )

    crs_ok = prior_info["crs"] == "EPSG:4326"
    results.append(CheckResult("坐标参考系统为EPSG:4326", crs_ok, f"CRS={prior_info['crs']}"))

    grid_problems = [name for name, info in infos.items() if not same_grid(prior_info, info)]
    results.append(
        CheckResult(
            "辅助图层与4G建设先验共格",
            not grid_problems,
            "已提供图层均与4G建设先验共格" if not grid_problems else f"不共格图层：{', '.join(grid_problems)}",
        )
    )

    prior, prior_nodata = read_band(CORE_FILES["4G建设先验"])
    admin, _ = read_band(CORE_FILES["行政区陆地掩膜"])
    land = admin > 0
    prior_valid = land & valid_mask(prior, prior_nodata)
    positive = prior_valid & (prior > 0)
    context["prior_stats"] = describe(prior[prior_valid].astype(float))
    context["land_valid_count"] = int(np.sum(prior_valid))
    context["positive_pixel_count"] = int(np.sum(positive))
    demand_support, demand_score, demand_support_rows = build_demand_support_mask(prior_valid)
    context["demand_support_rows"] = demand_support_rows
    completion_metrics = [
        {"metric": "4G建设先验陆地有效像元数", "value": context["land_valid_count"], "unit": "个", "note": "陆地掩膜内可读取像元"},
        {"metric": "4G建设先验正值像元数", "value": context["positive_pixel_count"], "unit": "个", "note": "预测基站密度大于0的像元"},
        {"metric": "4G建设先验正值覆盖率", "value": float(context["positive_pixel_count"] / max(context["land_valid_count"], 1)), "unit": "-", "note": "正值像元数/陆地有效像元数"},
    ]
    for row in demand_support_rows:
        completion_metrics.append({
            "metric": f"需求支撑像元数-{row['signal']}",
            "value": row["pixel_count"],
            "unit": "个",
            "note": row["note"],
        })
    results.append(CheckResult("4G建设先验有效像元存在", context["land_valid_count"] > 0, f"陆地有效像元数：{context['land_valid_count']}"))
    results.append(CheckResult("4G建设先验正值覆盖可记录", context["positive_pixel_count"] > 0, f"正值像元数：{context['positive_pixel_count']}"))

    raw_path = OPTIONAL_FILES["原始4G基站栅格"]
    if raw_path.is_file():
        raw, raw_nodata = read_band(raw_path)
        raw_valid = land & valid_mask(raw, raw_nodata)
        raw_completion = calculate_completion_metrics(
            prior=prior,
            raw=raw,
            prior_valid=prior_valid,
            raw_valid=raw_valid,
            demand_support=demand_support,
            demand_score=demand_score,
        )
        completion_metrics.extend([
            {"metric": "原始4G栅格正值像元数", "value": raw_completion["raw_positive_count"], "unit": "个", "note": "原始公开基站栅格中已有记录的像元"},
            {"metric": "4G先验/原始正值像元扩展倍数", "value": raw_completion["positive_expansion_factor"], "unit": "倍", "note": "GNN正值覆盖相对原始正值覆盖的扩大倍数"},
            {"metric": "原始正值区被先验保留比例", "value": float(raw_completion["prior_on_raw_positive_count"] / max(raw_completion["raw_positive_count"], 1)), "unit": "-", "note": "原始有站区域中GNN仍预测为正的比例"},
            {"metric": "需求支撑稀疏区像元数", "value": raw_completion["raw_zero_or_empty_count"], "unit": "个", "note": "人口、夜间灯光或建筑体量支撑，但原始LTE无正值的区域"},
            {"metric": "需求支撑稀疏区补全像元数", "value": raw_completion["completed_pixels_count"], "unit": "个", "note": "需求支撑稀疏区内GNN预测为正的像元"},
            {"metric": "需求支撑稀疏区补全率", "value": raw_completion["completion_ratio"], "unit": "-", "note": "补全像元数/需求支撑稀疏区像元数"},
            {"metric": "重点需求稀疏区像元数", "value": raw_completion["priority_sparse_area_count"], "unit": "个", "note": "需求强度前25%的稀疏区"},
            {"metric": "重点需求稀疏区补全像元数", "value": raw_completion["priority_completed_pixels_count"], "unit": "个", "note": "重点需求稀疏区内GNN预测为正的像元"},
            {"metric": "重点需求稀疏区补全率", "value": raw_completion["priority_completion_ratio"], "unit": "-", "note": "重点需求稀疏区补全像元数/重点需求稀疏区像元数"},
            {"metric": "全陆地缺站区参考补全率", "value": raw_completion["land_sparse_completion_ratio_reference"], "unit": "-", "note": "仅作参考，不作为主判定"},
        ])
        results.append(
            CheckResult(
                "需求支撑稀疏区补全率可计算",
                int(raw_completion["raw_zero_or_empty_count"]) > 0,
                f"稀疏区像元数：{raw_completion['raw_zero_or_empty_count']}，补全像元数：{raw_completion['completed_pixels_count']}，补全率：{raw_completion['completion_ratio']}",
            )
        )
        results.append(
            CheckResult(
                "重点需求稀疏区补全率可计算",
                int(raw_completion["priority_sparse_area_count"]) > 0,
                f"重点稀疏区像元数：{raw_completion['priority_sparse_area_count']}，补全像元数：{raw_completion['priority_completed_pixels_count']}，补全率：{raw_completion['priority_completion_ratio']}",
            )
        )
    else:
        results.append(CheckResult("原始4G栅格补全率记录", True, f"未提供原始4G栅格，跳过补全率计算：{raw_path}"))

    context["completion_metrics"] = completion_metrics

    if RAW_CELL_TOWERS.is_file():
        results.append(CheckResult("OpenCellID原始点数据存在", True, f"文件路径：{RAW_CELL_TOWERS}"))
    else:
        results.append(CheckResult("OpenCellID原始点数据记录", True, f"未提供原始CSV，跳过点数据规模统计：{RAW_CELL_TOWERS}"))

    correlation_rows = []
    for name, path in OPTIONAL_FILES.items():
        if not path.is_file() or name == "原始4G基站栅格":
            continue
        layer, layer_nodata = read_band(path)
        mask = positive & valid_mask(layer, layer_nodata)
        rho = sample_spearman(np.log1p(np.clip(prior, 0, None)), layer, mask)
        correlation_rows.append({"layer": name, "spearman_rho": rho, "sample_available": rho is not None})
    context["correlation_rows"] = correlation_rows
    if correlation_rows:
        has_corr = any(row["spearman_rho"] is not None for row in correlation_rows)
        results.append(CheckResult("4G建设先验与辅助图层空间对应关系可计算", has_corr, f"相关性结果：{correlation_rows}"))
    else:
        results.append(CheckResult("4G建设先验与辅助图层空间对应关系记录", True, "未提供人口/夜间灯光/建成环境/城市化等级辅助图层，跳过相关性计算"))

    return results, context


def print_report(results: list[CheckResult], context: dict[str, object]) -> None:
    print("4G建设先验补全结果测试报告")
    print(f"检查目录：{context['spectrum_dir']}")
    print(f"输出目录：{context['output_dir']}")
    print(f"检查结果：{sum(result.passed for result in results)}/{len(results)} 项通过")
    print()

    infos = context.get("infos")
    if infos:
        prior_info = infos["4G建设先验"]
        print("4G建设先验栅格信息：")
        print(f"- 尺寸: {prior_info['width']} x {prior_info['height']}")
        print(f"- CRS: {prior_info['crs']}")
        print(f"- 分辨率: {prior_info['resolution']}")
        print(f"- 空间范围: {prior_info['bounds']}")
        if "prior_stats" in context:
            print(f"- 分布统计: {context['prior_stats']}")
            print(f"- 正值像元数: {context['positive_pixel_count']}")
            print(f"- 辅助图层相关性: {context['correlation_rows']}")
        print()

    print("详细结果：")
    for result in results:
        status = "通过" if result.passed else "未通过"
        print(f"- [{status}] {result.name}：{result.details}")

    if "summary_txt" in context:
        print()
        print("输出文件：")
        print(f"- 结果表：{context['results_csv']}")
        print(f"- 指标表：{context['metrics_csv']}")
        print(f"- 辅助图层相关性表：{context['correlation_csv']}")
        if "inventory_csv" in context:
            print(f"- 栅格清单：{context['inventory_csv']}")
        print(f"- 摘要：{context['summary_txt']}")


def main() -> int:
    results, context = build_results()
    write_outputs(results, context)
    print_report(results, context)
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
