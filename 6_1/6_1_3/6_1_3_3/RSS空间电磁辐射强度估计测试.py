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

try:
    from scipy.stats import spearmanr
except ImportError:  # pragma: no cover - optional dependency
    spearmanr = None


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
SPECTRUM_DIR = REPO_ROOT / "dataset" / "products" / "spectrum"
OUTPUT_DIR = SCRIPT_DIR / "outputs"
RSS_FILE = SPECTRUM_DIR / "Global_5G_Radiation_Map_60arcsec.tif"
LTE_FILE = SPECTRUM_DIR / "Global_LTE_60arcsec_GNN.tif"
RATIO_FILE = SPECTRUM_DIR / "5g_density_multiplier_result_60arcsec.tif"
HEIGHT_FILE = SPECTRUM_DIR / "GHS_BUILT_H_FROM_VS_E2025_E2030_aligned_60arcsec.tif"
AREA_FILE = SPECTRUM_DIR / "GHS_BUILT_S_E2030_GLOBE_R2023A_4326_30ss_V1_0_aligned_60arcsec.tif"
ADMIN_FILE = SPECTRUM_DIR / "全球国家行政区_aligned_60arcsec.tif"
MIN_CALCULATED_PIXELS = 1000
EARTH_METERS_PER_DEGREE = 111_319.9


@dataclass
class CheckResult:
    name: str
    passed: bool
    details: str


def read_band(path: Path) -> tuple[np.ndarray, float | None]:
    with rasterio.open(path) as ds:
        return ds.read(1), ds.nodata


def same_grid(path_a: Path, path_b: Path) -> bool:
    with rasterio.open(path_a) as a, rasterio.open(path_b) as b:
        return a.crs == b.crs and a.transform == b.transform and a.width == b.width and a.height == b.height


def valid_mask(arr: np.ndarray, nodata: float | None) -> np.ndarray:
    mask = np.isfinite(arr)
    if nodata is not None and np.isfinite(nodata):
        mask &= arr != nodata
    return mask


def describe(values: np.ndarray) -> dict[str, float | int | None]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "p05": None, "p50": None, "p95": None, "min": None, "max": None}
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def grid_area_m2(path: Path) -> float:
    with rasterio.open(path) as ds:
        res_x, res_y = ds.res
        if ds.crs and ds.crs.is_geographic:
            return abs(res_x * EARTH_METERS_PER_DEGREE * res_y * EARTH_METERS_PER_DEGREE)
        return abs(res_x * res_y)


def blocking_beta(height: np.ndarray, area: np.ndarray, cell_area_m2: float) -> np.ndarray:
    coverage = np.clip(area.astype(float) / cell_area_m2, 0.0, 1.0)
    feature_width = np.minimum(height.astype(float), 40.0)
    beta = np.zeros_like(feature_width, dtype=float)
    valid = feature_width > 0.1
    beta[valid] = (4.0 * coverage[valid]) / (np.pi * feature_width[valid])
    return beta


def sample_corr(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float | None:
    if spearmanr is None:
        return None
    idx = np.flatnonzero(mask.ravel())
    if idx.size < 3:
        return None
    if idx.size > 200_000:
        rng = np.random.default_rng(42)
        idx = rng.choice(idx, 200_000, replace=False)
    rho, _ = spearmanr(x.ravel()[idx].astype(float), y.ravel()[idx].astype(float))
    return float(rho)


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
    rows = [
        {"metric": "陆地像元数", "value": context.get("land_count"), "unit": "个"},
        {"metric": "陆地有效RSS像元数", "value": context.get("rss_valid_count"), "unit": "个"},
        {"metric": "实际计算型RSS像元数", "value": context.get("calculated_count"), "unit": "个"},
        {"metric": "RSS有效覆盖率", "value": context.get("rss_valid_coverage"), "unit": "-"},
        {"metric": "计算型RSS覆盖率", "value": context.get("calculated_coverage"), "unit": "-"},
        {"metric": "RSS动态范围P95-P05", "value": context.get("dynamic_range"), "unit": "dB"},
    ]
    if "rss_stats" in context:
        for key, value in context["rss_stats"].items():  # type: ignore[union-attr]
            rows.append({"metric": f"RSS_{key}", "value": value, "unit": "dBm" if key != "count" else "个"})
    if "correlations" in context:
        for key, value in context["correlations"].items():  # type: ignore[union-attr]
            rows.append({"metric": key, "value": value, "unit": "-"})
    if "beta_stats" in context:
        for key, value in context["beta_stats"].items():  # type: ignore[union-attr]
            rows.append({"metric": f"Blocking_beta_{key}", "value": value, "unit": "-"})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value", "unit"])
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, results: list[CheckResult], context: dict[str, object]) -> None:
    lines = [
        "RSS空间电磁辐射强度估计测试摘要",
        f"RSS主结果：{context['rss_file']}",
        f"检查结果：{sum(result.passed for result in results)}/{len(results)} 项通过",
    ]
    if "rss_stats" in context:
        lines.extend([
            "",
            "核心指标：",
            f"陆地像元数：{context['land_count']}",
            f"陆地有效RSS像元数：{context['rss_valid_count']}",
            f"计算型RSS像元数：{context['calculated_count']}",
            f"RSS有效覆盖率：{context['rss_valid_coverage']}",
            f"计算型RSS覆盖率：{context['calculated_coverage']}",
            f"RSS分布：{context['rss_stats']}",
            f"RSS动态范围：{context['dynamic_range']}",
            f"环境阻塞参数beta分布：{context.get('beta_stats')}",
            f"相关性：{context['correlations']}",
            f"空间对应关系表：{context.get('correlation_csv')}",
        ])
    lines.extend(["", "输出文件：", f"结果表：{context['results_csv']}", f"指标表：{context['metrics_csv']}", "", "详细结果："])
    lines.extend(f"- [{'通过' if result.passed else '未通过'}] {result.name}：{result.details}" for result in results)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(results: list[CheckResult], context: dict[str, object]) -> None:
    ensure_dir(OUTPUT_DIR)
    results_csv = OUTPUT_DIR / "rss_intensity_results.csv"
    metrics_csv = OUTPUT_DIR / "rss_intensity_metrics.csv"
    correlation_csv = OUTPUT_DIR / "rss_spatial_correlations.csv"
    summary_txt = OUTPUT_DIR / "rss_intensity_summary.txt"
    context["results_csv"] = results_csv
    context["metrics_csv"] = metrics_csv
    context["correlation_csv"] = correlation_csv
    write_results_csv(results_csv, results)
    write_metrics_csv(metrics_csv, context)
    write_dict_rows(
        correlation_csv,
        context.get("correlation_rows", []),  # type: ignore[arg-type]
        ["layer", "spearman_rho"],
    )
    write_summary(summary_txt, results, context)
    context["summary_txt"] = summary_txt


def build_results() -> tuple[list[CheckResult], dict[str, object]]:
    results: list[CheckResult] = []
    context: dict[str, object] = {"rss_file": RSS_FILE, "spectrum_dir": SPECTRUM_DIR, "output_dir": OUTPUT_DIR}

    required = {
        "RSS主结果": RSS_FILE,
        "4G建设先验": LTE_FILE,
        "5G密度倍数": RATIO_FILE,
        "V/S等效建筑高度": HEIGHT_FILE,
        "建筑面积": AREA_FILE,
        "行政区陆地掩膜": ADMIN_FILE,
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    results.append(CheckResult("RSS测试数据齐全", not missing, "全部文件存在" if not missing else f"缺少文件：{', '.join(missing)}"))
    if missing:
        return results, context

    grid_ok = all(same_grid(RSS_FILE, path) for name, path in required.items() if name != "RSS主结果")
    results.append(CheckResult("RSS与核心输入图层共格", grid_ok, f"共格状态：{grid_ok}"))

    rss, rss_nodata = read_band(RSS_FILE)
    lte, lte_nodata = read_band(LTE_FILE)
    ratio, ratio_nodata = read_band(RATIO_FILE)
    height, height_nodata = read_band(HEIGHT_FILE)
    area, area_nodata = read_band(AREA_FILE)
    admin, _ = read_band(ADMIN_FILE)
    beta = blocking_beta(height, area, grid_area_m2(RSS_FILE))

    land = admin > 0
    land_count = int(np.sum(land))
    rss_valid = land & valid_mask(rss, rss_nodata)
    calculated = rss_valid & (rss > -199.0)
    stats = describe(rss[calculated].astype(float))
    context["rss_stats"] = stats
    context["land_count"] = land_count
    context["rss_valid_count"] = int(np.sum(rss_valid))
    context["calculated_count"] = int(np.sum(calculated))
    context["rss_valid_coverage"] = float(context["rss_valid_count"] / max(land_count, 1))
    context["calculated_coverage"] = float(context["calculated_count"] / max(land_count, 1))

    results.append(CheckResult("陆地RSS有效像元存在", context["rss_valid_count"] > 0, f"陆地有效RSS像元数：{context['rss_valid_count']}"))
    results.append(CheckResult("实际计算型RSS像元数量充足", context["calculated_count"] >= MIN_CALCULATED_PIXELS, f"高于-199 dBm像元数：{context['calculated_count']}"))
    results.append(CheckResult("RSS有效覆盖率可记录", context["rss_valid_coverage"] > 0, f"RSS有效覆盖率：{context['rss_valid_coverage']}"))
    results.append(CheckResult("计算型RSS覆盖率可记录", context["calculated_coverage"] > 0, f"计算型RSS覆盖率：{context['calculated_coverage']}"))

    dynamic_range = None if stats["p05"] is None or stats["p95"] is None else float(stats["p95"]) - float(stats["p05"])
    context["dynamic_range"] = dynamic_range
    results.append(
        CheckResult(
            "RSS动态范围可识别",
            dynamic_range is not None and dynamic_range > 0,
            "P95-P05=NA" if dynamic_range is None else f"P95-P05={dynamic_range:.4f} dB",
        )
    )

    area_valid = valid_mask(area, area_nodata)
    corr_mask = calculated & valid_mask(lte, lte_nodata) & valid_mask(ratio, ratio_nodata) & valid_mask(height, height_nodata) & area_valid
    lte_rho = sample_corr(rss, np.log1p(np.clip(lte, 0, None)), corr_mask)
    ratio_rho = sample_corr(rss, ratio, corr_mask)
    height_rho = sample_corr(rss, height, corr_mask)
    beta_rho = sample_corr(rss, beta, corr_mask)
    context["beta_stats"] = describe(beta[land & area_valid].astype(float))
    context["correlations"] = {
        "rss_vs_lte_log1p_spearman": lte_rho,
        "rss_vs_density_multiplier_spearman": ratio_rho,
        "rss_vs_vs_height_spearman": height_rho,
        "rss_vs_blocking_beta_spearman": beta_rho,
    }
    context["correlation_rows"] = [
        {"layer": "4G建设先验_log1p", "spearman_rho": lte_rho},
        {"layer": "KuKa密度倍数", "spearman_rho": ratio_rho},
        {"layer": "V/S等效建筑高度", "spearman_rho": height_rho},
        {"layer": "环境阻塞参数beta", "spearman_rho": beta_rho},
    ]

    results.append(CheckResult("RSS与解释变量相关性可计算", lte_rho is not None, f"相关性：{context['correlations']}"))
    results.append(CheckResult("环境阻塞参数可计算", context["beta_stats"]["count"] > 0, f"beta统计：{context['beta_stats']}"))
    return results, context


def print_report(results: list[CheckResult], context: dict[str, object]) -> None:
    print("RSS空间电磁辐射强度估计测试报告")
    print(f"RSS主结果：{context['rss_file']}")
    print(f"输出目录：{context['output_dir']}")
    print(f"检查结果：{sum(result.passed for result in results)}/{len(results)} 项通过")
    print()

    if "rss_stats" in context:
        print("RSS统计：")
        print(f"- 陆地像元数: {context['land_count']}")
        print(f"- 陆地有效RSS像元数: {context['rss_valid_count']}")
        print(f"- 计算型RSS像元数: {context['calculated_count']}")
        print(f"- RSS有效覆盖率: {context['rss_valid_coverage']}")
        print(f"- 计算型RSS覆盖率: {context['calculated_coverage']}")
        print(f"- RSS分布: {context['rss_stats']}")
        print(f"- 动态范围: {context['dynamic_range']}")
        print(f"- 阻塞参数beta分布: {context.get('beta_stats')}")
        print(f"- 相关性: {context['correlations']}")
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
        print(f"- 空间对应关系表：{context['correlation_csv']}")
        print(f"- 摘要：{context['summary_txt']}")


def main() -> int:
    results, context = build_results()
    write_outputs(results, context)
    print_report(results, context)
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
