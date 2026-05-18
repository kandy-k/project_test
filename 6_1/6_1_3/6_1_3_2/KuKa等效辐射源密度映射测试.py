from __future__ import annotations

import sys
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import rasterio
except ImportError as exc:  # pragma: no cover - dependency check
    print(f"缺少依赖 rasterio，无法执行检查：{exc}", file=sys.stderr)
    sys.exit(2)


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
SPECTRUM_DIR = REPO_ROOT / "dataset" / "products" / "spectrum"
OUTPUT_DIR = SCRIPT_DIR / "outputs"
RATIO_FILE = SPECTRUM_DIR / "5g_density_multiplier_result_60arcsec.tif"
LTE_FILE = SPECTRUM_DIR / "Global_LTE_60arcsec_GNN.tif"
HEIGHT_FILE = SPECTRUM_DIR / "GHS_BUILT_H_FROM_VS_E2025_E2030_aligned_60arcsec.tif"
AREA_FILE = SPECTRUM_DIR / "GHS_BUILT_S_E2030_GLOBE_R2023A_4326_30ss_V1_0_aligned_60arcsec.tif"
VOLUME_FILE = SPECTRUM_DIR / "GHS_BUILT_V_E2025_GLOBE_R2023A_4326_30ss_V1_0_aligned_60arcsec.tif"
MIN_BUILDING_PIXELS = 1000


@dataclass
class CheckResult:
    name: str
    passed: bool
    details: str


def read_band(path: Path) -> tuple[np.ndarray, object, object]:
    with rasterio.open(path) as ds:
        return ds.read(1), ds.nodata, ds


def valid_mask(arr: np.ndarray, nodata: object) -> np.ndarray:
    mask = np.isfinite(arr)
    if nodata is not None and np.isfinite(float(nodata)):
        mask &= arr != float(nodata)
    return mask


def same_grid(path_a: Path, path_b: Path) -> bool:
    with rasterio.open(path_a) as a, rasterio.open(path_b) as b:
        return a.crs == b.crs and a.transform == b.transform and a.width == b.width and a.height == b.height


def stats(values: np.ndarray) -> dict[str, float | int | None]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "p50": None, "p95": None, "p99": None, "max": None, "mean": None}
    return {
        "count": int(values.size),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
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


def high_low_mean_ratio(signal: np.ndarray, values: np.ndarray, mask: np.ndarray) -> float | None:
    valid = mask & np.isfinite(signal) & np.isfinite(values) & (signal > 0)
    if int(np.sum(valid)) < 100:
        return None
    q33, q67 = np.percentile(signal[valid], [33.33, 66.67])
    low = valid & (signal <= q33)
    high = valid & (signal > q67)
    if not np.any(low) or not np.any(high):
        return None
    low_mean = float(np.mean(values[low]))
    high_mean = float(np.mean(values[high]))
    if low_mean <= 0:
        return None
    return high_mean / low_mean


def scene_rows(signal_name: str, signal: np.ndarray, ratio: np.ndarray, lte: np.ndarray, mask: np.ndarray) -> list[dict[str, object]]:
    valid = mask & np.isfinite(signal) & np.isfinite(ratio) & np.isfinite(lte) & (signal > 0)
    if int(np.sum(valid)) < 100:
        return []
    q33, q67 = np.percentile(signal[valid], [33.33, 66.67])
    groups = [
        ("low", valid & (signal <= q33)),
        ("middle", valid & (signal > q33) & (signal <= q67)),
        ("high", valid & (signal > q67)),
    ]
    rows: list[dict[str, object]] = []
    for group_name, group_mask in groups:
        if not np.any(group_mask):
            continue
        kuka_density = np.clip(lte[group_mask], 0, None) * ratio[group_mask]
        rows.append({
            "scene_signal": signal_name,
            "group": group_name,
            "pixel_count": int(np.sum(group_mask)),
            "signal_mean": float(np.mean(signal[group_mask])),
            "ratio_mean": float(np.mean(ratio[group_mask])),
            "ratio_p50": float(np.percentile(ratio[group_mask], 50)),
            "ratio_p95": float(np.percentile(ratio[group_mask], 95)),
            "kuka_density_mean": float(np.mean(kuka_density)),
        })
    return rows


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


def write_dict_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_metrics_csv(path: Path, context: dict[str, object]) -> None:
    rows = []
    if "ratio_stats" in context:
        ratio_stats = context["ratio_stats"]
        for key, value in ratio_stats.items():  # type: ignore[union-attr]
            rows.append({"metric": f"建筑区密度倍数{key}", "value": value, "unit": "倍" if key != "count" else "个"})
    if "height_stats" in context:
        height_stats = context["height_stats"]
        for key, value in height_stats.items():  # type: ignore[union-attr]
            rows.append({"metric": f"建筑区V/S高度{key}", "value": value, "unit": "m" if key != "count" else "个"})
    if "kuka_density_stats" in context:
        kuka_stats = context["kuka_density_stats"]
        for key, value in kuka_stats.items():  # type: ignore[union-attr]
            rows.append({"metric": f"KuKa等效密度{key}", "value": value, "unit": "-" if key != "count" else "个"})
    for key in (
        "ratio_vs_lte_spearman",
        "ratio_vs_height_spearman",
        "kuka_density_vs_lte_spearman",
        "height_high_low_mean_ratio",
        "kuka_density_height_high_low_ratio",
        "ratio_p99_p50",
        "kuka_density_mean",
    ):
        if key in context:
            rows.append({"metric": key, "value": context[key], "unit": "-"})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value", "unit"])
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, results: list[CheckResult], context: dict[str, object]) -> None:
    lines = [
        "KuKa等效辐射源密度映射测试摘要",
        f"密度倍数栅格：{context['ratio_file']}",
        f"V/S等效建筑高度：{context['height_file']}",
        f"检查结果：{sum(result.passed for result in results)}/{len(results)} 项通过",
    ]
    if "ratio_stats" in context:
        lines.extend([
            "",
            "核心指标：",
            f"建筑区有效像元数：{context['building_count']}",
            f"密度倍数统计：{context['ratio_stats']}",
            f"V/S高度统计：{context['height_stats']}",
            f"分场景统计表：{context.get('scene_csv')}",
        ])
    lines.extend(["", "输出文件：", f"结果表：{context['results_csv']}", f"指标表：{context['metrics_csv']}", "", "详细结果："])
    lines.extend(f"- [{'通过' if result.passed else '未通过'}] {result.name}：{result.details}" for result in results)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(results: list[CheckResult], context: dict[str, object]) -> None:
    ensure_dir(OUTPUT_DIR)
    results_csv = OUTPUT_DIR / "kuka_density_mapping_results.csv"
    metrics_csv = OUTPUT_DIR / "kuka_density_mapping_metrics.csv"
    scene_csv = OUTPUT_DIR / "kuka_density_scene_table.csv"
    summary_txt = OUTPUT_DIR / "kuka_density_mapping_summary.txt"
    context["results_csv"] = results_csv
    context["metrics_csv"] = metrics_csv
    context["scene_csv"] = scene_csv
    write_results_csv(results_csv, results)
    write_metrics_csv(metrics_csv, context)
    write_dict_rows(
        scene_csv,
        context.get("scene_rows", []),  # type: ignore[arg-type]
        ["scene_signal", "group", "pixel_count", "signal_mean", "ratio_mean", "ratio_p50", "ratio_p95", "kuka_density_mean"],
    )
    write_summary(summary_txt, results, context)
    context["summary_txt"] = summary_txt


def build_results() -> tuple[list[CheckResult], dict[str, object]]:
    results: list[CheckResult] = []
    context: dict[str, object] = {
        "ratio_file": RATIO_FILE,
        "lte_file": LTE_FILE,
        "height_file": HEIGHT_FILE,
        "area_file": AREA_FILE,
        "volume_file": VOLUME_FILE,
        "output_dir": OUTPUT_DIR,
    }

    required = {
        "5G密度倍数": RATIO_FILE,
        "4G建设先验": LTE_FILE,
        "V/S等效建筑高度": HEIGHT_FILE,
        "建筑面积": AREA_FILE,
        "建筑体积": VOLUME_FILE,
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    results.append(CheckResult("密度倍数测试数据齐全", not missing, "全部文件存在" if not missing else f"缺少文件：{', '.join(missing)}"))
    if missing:
        return results, context

    grid_ok = same_grid(RATIO_FILE, LTE_FILE) and same_grid(RATIO_FILE, HEIGHT_FILE) and same_grid(RATIO_FILE, AREA_FILE) and same_grid(RATIO_FILE, VOLUME_FILE)
    results.append(CheckResult("密度倍数与建筑环境图层共格", grid_ok, f"共格状态：{grid_ok}"))

    ratio, ratio_nodata, _ = read_band(RATIO_FILE)
    lte, lte_nodata, _ = read_band(LTE_FILE)
    height, height_nodata, _ = read_band(HEIGHT_FILE)
    area, area_nodata, _ = read_band(AREA_FILE)
    volume, volume_nodata, _ = read_band(VOLUME_FILE)

    building_mask = (
        valid_mask(ratio, ratio_nodata)
        & valid_mask(lte, lte_nodata)
        & valid_mask(height, height_nodata)
        & valid_mask(area, area_nodata)
        & valid_mask(volume, volume_nodata)
        & (ratio > 0)
        & (lte >= 0)
        & (height > 0)
        & (area > 0)
        & (volume > 0)
    )
    building_count = int(np.sum(building_mask))
    context["building_count"] = building_count
    results.append(CheckResult("建筑区有效像元数量充足", building_count >= MIN_BUILDING_PIXELS, f"有效建筑区像元数：{building_count}"))
    if building_count == 0:
        return results, context

    ratio_stats = stats(ratio[building_mask].astype(float))
    height_stats = stats(height[building_mask].astype(float))
    context["ratio_stats"] = ratio_stats
    context["height_stats"] = height_stats
    kuka_density = np.clip(lte, 0, None) * ratio
    context["kuka_density_stats"] = stats(kuka_density[building_mask].astype(float))
    context["kuka_density_mean"] = context["kuka_density_stats"]["mean"]
    context["ratio_p99_p50"] = None if not ratio_stats["p50"] else float(ratio_stats["p99"] / ratio_stats["p50"])  # type: ignore[operator]
    context["ratio_vs_lte_spearman"] = sample_spearman(ratio, np.log1p(np.clip(lte, 0, None)), building_mask)
    context["ratio_vs_height_spearman"] = sample_spearman(ratio, height, building_mask)
    context["kuka_density_vs_lte_spearman"] = sample_spearman(kuka_density, np.log1p(np.clip(lte, 0, None)), building_mask)
    context["height_high_low_mean_ratio"] = high_low_mean_ratio(height, ratio, building_mask)
    context["kuka_density_height_high_low_ratio"] = high_low_mean_ratio(height, kuka_density, building_mask)
    context["scene_rows"] = (
        scene_rows("V/S等效建筑高度", height, ratio, lte, building_mask)
        + scene_rows("建筑面积", area, ratio, lte, building_mask)
        + scene_rows("建筑体积", volume, ratio, lte, building_mask)
    )

    ratio_positive_ok = ratio_stats["p50"] is not None and float(ratio_stats["p50"]) > 0
    results.append(CheckResult("建筑区密度倍数为正", ratio_positive_ok, f"倍数P50={ratio_stats['p50']}，P95={ratio_stats['p95']}"))

    high_tail_ok = ratio_stats["p95"] is not None and ratio_stats["p50"] is not None and float(ratio_stats["p95"]) >= float(ratio_stats["p50"])
    results.append(CheckResult("密度倍数高分位不低于中位数", high_tail_ok, f"P50={ratio_stats['p50']}，P95={ratio_stats['p95']}"))

    height_reasonable = height_stats["p50"] is not None and 0 < float(height_stats["p50"]) < 100
    results.append(CheckResult("V/S等效建筑高度量级合理", height_reasonable, f"高度P50={height_stats['p50']} m，P99={height_stats['p99']} m"))
    results.append(CheckResult("密度倍数与建筑高度响应可记录", context["ratio_vs_height_spearman"] is not None, f"Spearman rho={context['ratio_vs_height_spearman']}"))
    high_low_ok = context["height_high_low_mean_ratio"] is None or float(context["height_high_low_mean_ratio"]) >= 1.0
    results.append(CheckResult("高建筑区倍数不低于低建筑区", high_low_ok, f"高/低建筑高度组平均倍数比={context['height_high_low_mean_ratio']}"))
    results.append(CheckResult("KuKa等效密度继承4G建设强度", context["kuka_density_vs_lte_spearman"] is not None and float(context["kuka_density_vs_lte_spearman"]) > 0.5, f"KuKa密度 vs 4G先验 Spearman rho={context['kuka_density_vs_lte_spearman']}"))
    results.append(CheckResult("密度倍数尾部放大受控", context["ratio_p99_p50"] is not None and float(context["ratio_p99_p50"]) < 2.0, f"P99/P50={context['ratio_p99_p50']}"))
    results.append(CheckResult("建筑环境分场景统计可输出", bool(context["scene_rows"]), f"分场景统计行数：{len(context['scene_rows'])}"))

    return results, context


def print_report(results: list[CheckResult], context: dict[str, object]) -> None:
    print("KuKa等效辐射源密度映射测试报告")
    print(f"密度倍数栅格：{context['ratio_file']}")
    print(f"V/S等效建筑高度：{context['height_file']}")
    print(f"输出目录：{context['output_dir']}")
    print(f"检查结果：{sum(result.passed for result in results)}/{len(results)} 项通过")
    print()

    if "ratio_stats" in context:
        print("建筑区统计：")
        print(f"- 有效像元数: {context['building_count']}")
        print(f"- 密度倍数统计: {context['ratio_stats']}")
        print(f"- V/S高度统计: {context['height_stats']}")
        print(f"- 倍数与4G先验Spearman: {context['ratio_vs_lte_spearman']}")
        print(f"- 倍数与V/S高度Spearman: {context['ratio_vs_height_spearman']}")
        print(f"- KuKa等效密度与4G先验Spearman: {context['kuka_density_vs_lte_spearman']}")
        print(f"- 高/低建筑高度组平均倍数比: {context['height_high_low_mean_ratio']}")
        print(f"- 高/低建筑高度组KuKa等效密度比: {context['kuka_density_height_high_low_ratio']}")
        print(f"- 密度倍数P99/P50: {context['ratio_p99_p50']}")
        print(f"- 分场景统计行数: {len(context['scene_rows'])}")
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
        print(f"- 分场景统计表：{context['scene_csv']}")
        print(f"- 摘要：{context['summary_txt']}")


def main() -> int:
    results, context = build_results()
    write_outputs(results, context)
    print_report(results, context)
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
