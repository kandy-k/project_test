from __future__ import annotations

import sys
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize
except ImportError as exc:  # pragma: no cover - dependency check
    print(f"缺少依赖，无法执行检查：{exc}", file=sys.stderr)
    sys.exit(2)

try:
    from scipy.stats import spearmanr
except ImportError:  # pragma: no cover - optional dependency
    spearmanr = None


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
SPECTRUM_DIR = REPO_ROOT / "dataset" / "products" / "spectrum"
VALIDATION_DIR = REPO_ROOT / "dataset" / "validation" / "spectrum"
OUTPUT_DIR = SCRIPT_DIR / "outputs"
RSS_FILE = SPECTRUM_DIR / "Global_5G_Radiation_Map_60arcsec.tif"
COUNTRY_TRAFFIC_FILE = VALIDATION_DIR / "country_traffic_2024_clean.csv"
BOUNDARY_DIR = VALIDATION_DIR / "boundaries"
OOKLA_DIR = VALIDATION_DIR / "ookla"
MIN_MATCHED_COUNTRIES = 20
TOP_PERCENTS = (1, 5, 10, 20)


@dataclass
class CheckResult:
    name: str
    passed: bool
    details: str


def find_country_boundary(directory: Path) -> Path | None:
    if not directory.is_dir():
        return None
    files = [path for path in directory.rglob("*") if path.suffix.lower() in {".shp", ".gpkg", ".geojson"}]
    return sorted(files)[0] if files else None


def find_ookla_source() -> Path | None:
    search_dirs = [OOKLA_DIR, VALIDATION_DIR]
    for directory in search_dirs:
        if not directory.is_dir():
            continue
        gps_files = [path for path in directory.rglob("gps_mobile_tiles.shp") if path.is_file()]
        if gps_files:
            return sorted(gps_files)[0]
        vector_files = [path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in {".shp", ".gpkg", ".geojson"} and "ookla" in str(path).lower()]
        if vector_files:
            return sorted(vector_files)[0]
        raster_files = [path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in {".tif", ".tiff"} and "ookla" in str(path).lower()]
        if raster_files:
            return sorted(raster_files)[0]
    return None


def read_rss_and_meta(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    with rasterio.open(path) as ds:
        return ds.read(1), {
            "shape": ds.shape,
            "transform": ds.transform,
            "crs": ds.crs,
            "nodata": ds.nodata,
            "meta": ds.meta.copy(),
        }


def make_country_mask(boundary_path: Path, meta: dict[str, object]) -> tuple[np.ndarray, pd.DataFrame]:
    gdf = gpd.read_file(boundary_path)
    iso_col = "ISO_A3" if "ISO_A3" in gdf.columns else "ADM0_A3" if "ADM0_A3" in gdf.columns else None
    if iso_col is None:
        raise ValueError("国家边界文件缺少 ISO_A3 或 ADM0_A3 字段")

    gdf = gdf.dropna(subset=[iso_col]).copy()
    gdf = gdf[gdf[iso_col].astype(str).str.len() == 3].copy()
    if gdf.crs is not None and meta["crs"] is not None and gdf.crs != meta["crs"]:
        gdf = gdf.to_crs(meta["crs"])

    gdf["iso_id"] = np.arange(1, len(gdf) + 1)
    shapes = ((geom, int(value)) for geom, value in zip(gdf.geometry, gdf["iso_id"]) if geom is not None and not geom.is_empty)
    mask = rasterize(shapes, out_shape=meta["shape"], transform=meta["transform"], fill=0, dtype="int32")
    lookup = gdf[["iso_id", iso_col]].rename(columns={iso_col: "entityIso"})
    lookup["entityIso"] = lookup["entityIso"].astype(str).str.upper()
    return mask, lookup


def make_ookla_raster_like(source_path: Path, meta: dict[str, object]) -> tuple[np.ndarray, Path]:
    """把 Ookla 活跃度数据转换到 RSS 共格栅格。

    优先复用 outputs 中已经生成的缓存；输入为 Shapefile/GPKG/GeoJSON 时按 devices
    字段栅格化，输入为已共格 GeoTIFF 时直接读取。
    """

    cache_tif = OUTPUT_DIR / "ookla_devices_like_rss_60arcsec.tif"
    if cache_tif.is_file() and cache_tif.stat().st_size > 0:
        with rasterio.open(cache_tif) as ds:
            return ds.read(1), cache_tif

    if source_path.suffix.lower() in {".tif", ".tiff"}:
        with rasterio.open(source_path) as ds:
            same_grid = ds.crs == meta["crs"] and ds.transform == meta["transform"] and ds.shape == meta["shape"]
            if not same_grid:
                raise ValueError("Ookla栅格与RSS主结果不共格，请提供矢量瓦片或预先重采样到60arcsec。")
            arr = ds.read(1)
            raster_meta = meta["meta"].copy()  # type: ignore[union-attr]
            raster_meta.update(dtype="float32", count=1, nodata=0, compress="lzw")
            with rasterio.open(cache_tif, "w", **raster_meta) as dst:
                dst.write(arr.astype("float32"), 1)
            return arr, cache_tif

    gdf = gpd.read_file(source_path)
    if "devices" not in gdf.columns:
        raise ValueError("Ookla矢量数据缺少 devices 字段")
    gdf = gdf.dropna(subset=["devices"]).copy()
    if gdf.crs is not None and meta["crs"] is not None and gdf.crs != meta["crs"]:
        gdf = gdf.to_crs(meta["crs"])

    shapes = (
        (geom, float(value))
        for geom, value in zip(gdf.geometry, gdf["devices"])
        if geom is not None and not geom.is_empty
    )
    arr = rasterize(
        shapes,
        out_shape=meta["shape"],
        transform=meta["transform"],
        fill=0,
        dtype="float32",
    )
    raster_meta = meta["meta"].copy()  # type: ignore[union-attr]
    raster_meta.update(dtype="float32", count=1, nodata=0, compress="lzw")
    with rasterio.open(cache_tif, "w", **raster_meta) as dst:
        dst.write(arr, 1)
    return arr, cache_tif


def aggregate_by_country(rss: np.ndarray, country_mask: np.ndarray, nodata: float | None) -> pd.DataFrame:
    valid = (country_mask > 0) & np.isfinite(rss)
    if nodata is not None and np.isfinite(nodata):
        valid &= rss != nodata
    valid &= rss > -199.0

    iso = country_mask[valid].astype(np.int64)
    watts = np.power(10.0, (rss[valid].astype(float) - 30.0) / 10.0)
    sums = np.bincount(iso, weights=watts)
    counts = np.bincount(iso)

    rows = []
    for iso_id in np.where(counts > 0)[0]:
        if iso_id == 0:
            continue
        rows.append({"iso_id": int(iso_id), "pred_total_watts": float(sums[iso_id]), "valid_pixels": int(counts[iso_id])})
    return pd.DataFrame(rows)


def calculate_ookla_coverage(rss: np.ndarray, country_mask: np.ndarray, ookla: np.ndarray, nodata: float | None) -> list[dict[str, object]]:
    valid = (country_mask > 0) & np.isfinite(rss) & np.isfinite(ookla)
    if nodata is not None and np.isfinite(nodata):
        valid &= rss != nodata
    valid &= rss > -199.0
    active = valid & (ookla >= 1)

    rows: list[dict[str, object]] = []
    rss_values = rss[valid].astype(float)
    active_total = int(np.sum(active))
    if rss_values.size == 0:
        return rows

    for top_p in TOP_PERCENTS:
        threshold = float(np.percentile(rss_values, 100 - top_p))
        hotspot = valid & (rss >= threshold)
        hotspot_pixels = int(np.sum(hotspot))
        covered_pixels = int(np.sum(hotspot & active))
        rows.append({
            "region": "GLOBAL",
            "top_p_percent": top_p,
            "rss_threshold_dbm": threshold,
            "rss_hotspot_pixels": hotspot_pixels,
            "ookla_active_pixels": active_total,
            "covered_active_pixels": covered_pixels,
            "coverage_rate": None if active_total == 0 else float(covered_pixels / active_total),
        })
    return rows


def load_traffic(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"entityIso", "dataValue"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"国家流量表缺少字段：{', '.join(sorted(missing))}")
    df = df.dropna(subset=["entityIso", "dataValue"]).copy()
    df["entityIso"] = df["entityIso"].astype(str).str.upper().str.strip()
    df["dataValue"] = pd.to_numeric(df["dataValue"], errors="coerce")
    df = df.dropna(subset=["dataValue"])
    if "dataYear" in df.columns:
        df = df.sort_values("dataYear", ascending=False).drop_duplicates("entityIso")
    return df


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


def write_metrics_csv(path: Path, context: dict[str, object]) -> None:
    rows = [
        {"metric": "聚合国家数", "value": context.get("aggregated_country_count"), "unit": "个"},
        {"metric": "匹配国家数", "value": context.get("matched_country_count"), "unit": "个"},
        {"metric": "国家匹配覆盖率", "value": context.get("country_match_coverage"), "unit": "-"},
        {"metric": "国家尺度Spearman rho", "value": context.get("spearman_rho_raw"), "unit": "-"},
        {"metric": "国家尺度Spearman p-value", "value": context.get("spearman_p_value_raw"), "unit": "-"},
        {"metric": "国家尺度log-Spearman rho", "value": context.get("spearman_rho_log"), "unit": "-"},
        {"metric": "国家尺度log-Pearson r", "value": context.get("pearson_r_log"), "unit": "-"},
    ]
    for row in context.get("coverage_rows", []):  # type: ignore[union-attr]
        top_p = row["top_p_percent"]
        rows.append({"metric": f"Top{top_p}% RSS热点覆盖Ookla活跃像元率", "value": row["coverage_rate"], "unit": "-"})
        rows.append({"metric": f"Top{top_p}% RSS热点像元数", "value": row["rss_hotspot_pixels"], "unit": "个"})
        rows.append({"metric": f"Top{top_p}% 覆盖Ookla活跃像元数", "value": row["covered_active_pixels"], "unit": "个"})
    top10_row = context.get("top10_coverage_row")
    if top10_row:
        rows.append({"metric": "Ookla活跃像元数", "value": top10_row["ookla_active_pixels"], "unit": "个"})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value", "unit"])
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, results: list[CheckResult], context: dict[str, object]) -> None:
    lines = [
        "外部数据一致性测试摘要",
        f"RSS主结果：{context['rss_file']}",
        f"国家流量统计：{context['country_traffic_file']}",
        f"国家边界文件：{context['boundary_file']}",
        f"Ookla数据：{context.get('ookla_source')}",
        f"检查结果：{sum(result.passed for result in results)}/{len(results)} 项通过",
    ]
    if "matched_country_count" in context:
        lines.extend([
            "",
            "核心指标：",
            f"聚合国家数：{context['aggregated_country_count']}",
            f"匹配国家数：{context['matched_country_count']}",
            f"国家匹配覆盖率：{context['country_match_coverage']}",
            f"国家尺度Spearman rho：{context['spearman_rho_raw']}",
            f"国家尺度Spearman p-value：{context['spearman_p_value_raw']}",
            f"国家尺度log-Spearman rho：{context['spearman_rho_log']}",
            f"国家尺度log-Pearson r：{context['pearson_r_log']}",
        ])
    if "coverage_rows" in context:
        lines.extend(["", "Ookla热点覆盖率："])
        for row in context["coverage_rows"]:  # type: ignore[union-attr]
            rate = row["coverage_rate"]
            rate_text = "NA" if rate is None else f"{float(rate) * 100:.2f}%"
            lines.append(f"Top-{row['top_p_percent']}%：{rate_text}，活跃像元={row['ookla_active_pixels']}，覆盖像元={row['covered_active_pixels']}")
    lines.extend([
        "",
        "输出文件：",
        f"国家聚合结果表：{context.get('output_table')}",
        f"Ookla热点覆盖率表：{context.get('coverage_csv')}",
        f"Ookla共格缓存栅格：{context.get('ookla_cache_tif')}",
        f"结果表：{context['results_csv']}",
        f"指标表：{context['metrics_csv']}",
        "",
        "详细结果：",
    ])
    lines.extend(f"- [{'通过' if result.passed else '未通过'}] {result.name}：{result.details}" for result in results)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(results: list[CheckResult], context: dict[str, object]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_csv = OUTPUT_DIR / "external_consistency_results.csv"
    metrics_csv = OUTPUT_DIR / "external_consistency_metrics.csv"
    summary_txt = OUTPUT_DIR / "external_consistency_summary.txt"
    context["results_csv"] = results_csv
    context["metrics_csv"] = metrics_csv
    write_results_csv(results_csv, results)
    write_metrics_csv(metrics_csv, context)
    write_summary(summary_txt, results, context)
    context["summary_txt"] = summary_txt


def build_results() -> tuple[list[CheckResult], dict[str, object]]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: list[CheckResult] = []
    context: dict[str, object] = {
        "rss_file": RSS_FILE,
        "country_traffic_file": COUNTRY_TRAFFIC_FILE,
        "boundary_dir": BOUNDARY_DIR,
        "ookla_dir": OOKLA_DIR,
        "output_dir": OUTPUT_DIR,
    }

    boundary_file = find_country_boundary(BOUNDARY_DIR)
    ookla_source = find_ookla_source()
    context["boundary_file"] = boundary_file
    context["ookla_source"] = ookla_source
    results.append(CheckResult("RSS主结果存在", RSS_FILE.is_file(), f"文件路径：{RSS_FILE}"))
    results.append(CheckResult("国家流量统计存在", COUNTRY_TRAFFIC_FILE.is_file(), f"文件路径：{COUNTRY_TRAFFIC_FILE}"))
    results.append(CheckResult("国家边界数据存在", boundary_file is not None, f"边界文件：{boundary_file}" if boundary_file else f"边界目录：{BOUNDARY_DIR}"))
    results.append(CheckResult("Ookla测速活跃度数据存在", ookla_source is not None, f"Ookla数据：{ookla_source}" if ookla_source else f"Ookla目录：{OOKLA_DIR}"))
    if not RSS_FILE.is_file() or not COUNTRY_TRAFFIC_FILE.is_file() or boundary_file is None:
        return results, context

    rss, meta = read_rss_and_meta(RSS_FILE)
    country_mask, lookup = make_country_mask(boundary_file, meta)
    pred = aggregate_by_country(rss, country_mask, meta["nodata"])
    traffic = load_traffic(COUNTRY_TRAFFIC_FILE)
    merged = pred.merge(lookup, on="iso_id", how="left").merge(traffic[["entityIso", "dataValue"]], on="entityIso", how="inner")
    merged = merged[(merged["pred_total_watts"] > 0) & (merged["dataValue"] > 0)].copy()
    context["aggregated_country_count"] = len(pred)
    context["matched_country_count"] = len(merged)
    context["country_match_coverage"] = float(len(merged) / max(len(pred), 1))

    output_table = OUTPUT_DIR / "country_spectrum_consistency_table.csv"
    merged.to_csv(output_table, index=False, encoding="utf-8-sig")
    context["output_table"] = output_table

    results.append(CheckResult("国家尺度聚合可计算", not pred.empty, f"聚合国家数：{len(pred)}"))
    results.append(CheckResult("可匹配外部统计国家数量充足", len(merged) >= MIN_MATCHED_COUNTRIES, f"匹配国家数：{len(merged)}"))

    rho = None
    if spearmanr is not None and len(merged) >= 3:
        rho, p_value = spearmanr(merged["pred_total_watts"], merged["dataValue"])
        log_rho, log_p_value = spearmanr(np.log10(merged["pred_total_watts"]), np.log10(merged["dataValue"]))
        context["spearman_rho_raw"] = float(rho)
        context["spearman_p_value_raw"] = float(p_value)
        context["spearman_rho_log"] = float(log_rho)
        context["spearman_p_value_log"] = float(log_p_value)
        context["pearson_r_log"] = float(np.corrcoef(np.log10(merged["pred_total_watts"]), np.log10(merged["dataValue"]))[0, 1])
    else:
        context["spearman_rho_raw"] = None
        context["spearman_p_value_raw"] = None
        context["spearman_rho_log"] = None
        context["spearman_p_value_log"] = None
        context["pearson_r_log"] = None

    results.append(CheckResult("国家尺度Spearman相关可计算", context["spearman_rho_raw"] is not None, f"rho={context['spearman_rho_raw']}，p={context['spearman_p_value_raw']}"))

    if ookla_source is None:
        results.append(CheckResult("Ookla热点覆盖率可计算", False, "缺少Ookla测速活跃度数据，无法计算Top-p%热点覆盖率"))
        context["coverage_rows"] = []
        return results, context

    try:
        ookla_arr, ookla_cache_tif = make_ookla_raster_like(ookla_source, meta)
        coverage_rows = calculate_ookla_coverage(rss, country_mask, ookla_arr, meta["nodata"])
    except Exception as exc:
        results.append(CheckResult("Ookla热点覆盖率可计算", False, f"计算失败：{exc}"))
        context["coverage_rows"] = []
        return results, context

    coverage_csv = OUTPUT_DIR / "ookla_hotspot_coverage.csv"
    pd.DataFrame(coverage_rows).to_csv(coverage_csv, index=False, encoding="utf-8-sig")
    context["coverage_rows"] = coverage_rows
    context["coverage_csv"] = coverage_csv
    context["ookla_cache_tif"] = ookla_cache_tif
    top10_row = next((row for row in coverage_rows if row["top_p_percent"] == 10), None)
    context["top10_coverage_row"] = top10_row
    results.append(CheckResult("Ookla热点覆盖率可计算", bool(coverage_rows), f"输出Top-p%覆盖率 {len(coverage_rows)} 行"))
    if top10_row is not None:
        rate = top10_row["coverage_rate"]
        rate_text = "NA" if rate is None else f"{float(rate) * 100:.2f}%"
        results.append(CheckResult("Top10%热点覆盖率可记录", rate is not None, f"Top10%覆盖率：{rate_text}，Ookla活跃像元数：{top10_row['ookla_active_pixels']}"))
    return results, context


def print_report(results: list[CheckResult], context: dict[str, object]) -> None:
    print("外部数据一致性测试报告")
    print(f"RSS主结果：{context['rss_file']}")
    print(f"国家流量统计：{context['country_traffic_file']}")
    print(f"国家边界目录：{context['boundary_dir']}")
    print(f"Ookla目录：{context['ookla_dir']}")
    print(f"输出目录：{context['output_dir']}")
    print(f"检查结果：{sum(result.passed for result in results)}/{len(results)} 项通过")
    print()

    if "matched_country_count" in context:
        print("核心指标：")
        print(f"- 聚合国家数: {context['aggregated_country_count']}")
        print(f"- 匹配国家数: {context['matched_country_count']}")
        print(f"- 国家匹配覆盖率: {context['country_match_coverage']}")
        print(f"- Spearman rho: {context['spearman_rho_raw']}")
        print(f"- p-value: {context['spearman_p_value_raw']}")
        print(f"- log-Spearman rho: {context['spearman_rho_log']}")
        print(f"- log-Pearson r: {context['pearson_r_log']}")
        if "top10_coverage_row" in context and context["top10_coverage_row"]:
            row = context["top10_coverage_row"]
            print(f"- Top10% Ookla覆盖率: {row['coverage_rate']}")
        print(f"- 国家聚合结果表: {context['output_table']}")
        if "coverage_csv" in context:
            print(f"- Ookla热点覆盖率表: {context['coverage_csv']}")
        print()

    print("详细结果：")
    for result in results:
        status = "通过" if result.passed else "未通过"
        print(f"- [{status}] {result.name}：{result.details}")


def main() -> int:
    results, context = build_results()
    write_outputs(results, context)
    print_report(results, context)
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
