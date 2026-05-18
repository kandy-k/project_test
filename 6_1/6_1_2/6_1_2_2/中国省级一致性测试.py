from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from scipy.stats import pearsonr, spearmanr


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
PRED_TIF = REPO_ROOT / "dataset" / "products" / "demand" / "pred_final_pred_masked.tif"
INPUT_ROOT = REPO_ROOT / "dataset" / "validation"
BOUNDARY_DIR = INPUT_ROOT / "boundaries"
TRUTH_DIR = INPUT_ROOT / "truth" / "china_province"
OUTPUT_DIR = SCRIPT_DIR / "outputs"

BOUNDARY_SUFFIXES = {".gpkg", ".shp", ".geojson"}
TRUTH_SUFFIXES = {".csv", ".xlsx", ".xls"}
BACKGROUND_ID = 0
NODATA_FALLBACK = -9999.0

MIN_MATCHED_PROVINCES = 20
MIN_SPEARMAN_RHO = 0.60
MIN_PEARSON_LOG = 0.60

NAME_FIELD_CANDIDATES = (
    "name_zh",
    "province_name",
    "gn_name",
    "name",
    "name_en",
    "name_local",
)


@dataclass
class CheckResult:
    name: str
    passed: bool
    details: str


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def find_truth_file(directory: Path) -> Path | None:
    if not directory.is_dir():
        return None
    candidates = [path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in TRUTH_SUFFIXES]
    return sorted(candidates)[0] if candidates else None


def boundary_priority(path: Path) -> tuple[int, str]:
    text = str(path).lower()
    score = 0
    if "admin_1" in text or "adm1" in text or "province" in text:
        score -= 20
    if "admin_0" in text or "country" in text:
        score += 20
    return score, text


def find_boundary_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    files = [path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in BOUNDARY_SUFFIXES]
    return sorted(files, key=boundary_priority)


def get_nodata(ds: rasterio.io.DatasetReader, band: int = 1) -> float | None:
    if ds.nodatavals and ds.nodatavals[band - 1] is not None:
        return ds.nodatavals[band - 1]
    if ds.nodata is not None:
        return ds.nodata
    return NODATA_FALLBACK


def valid_mask(arr: np.ndarray, nodata: float | None) -> np.ndarray:
    mask = np.isfinite(arr)
    if nodata is not None:
        mask &= arr != nodata
    return mask


def normalize_province_name(name: object) -> str:
    text = str(name).strip()
    replacements = {
        "北京市": "北京",
        "天津市": "天津",
        "上海市": "上海",
        "重庆市": "重庆",
        "河北省": "河北",
        "山西省": "山西",
        "辽宁省": "辽宁",
        "吉林省": "吉林",
        "黑龙江省": "黑龙江",
        "江苏省": "江苏",
        "浙江省": "浙江",
        "安徽省": "安徽",
        "福建省": "福建",
        "江西省": "江西",
        "山东省": "山东",
        "河南省": "河南",
        "湖北省": "湖北",
        "湖南省": "湖南",
        "广东省": "广东",
        "海南省": "海南",
        "四川省": "四川",
        "贵州省": "贵州",
        "云南省": "云南",
        "陕西省": "陕西",
        "甘肃省": "甘肃",
        "青海省": "青海",
        "台湾省": "台湾",
        "内蒙古自治区": "内蒙古",
        "广西壮族自治区": "广西",
        "西藏自治区": "西藏",
        "宁夏回族自治区": "宁夏",
        "新疆维吾尔自治区": "新疆",
        "香港特别行政区": "香港",
        "澳门特别行政区": "澳门",
    }
    if text in replacements:
        return replacements[text]

    for suffix in (
        "维吾尔自治区",
        "壮族自治区",
        "回族自治区",
        "特别行政区",
        "自治区",
        "省",
        "市",
    ):
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text.replace(" ", "")


def convert_to_eb(value: object, unit: object) -> float:
    number = float(value)
    unit_text = str(unit).strip().upper()
    factors = {
        "B": 1 / (1024**6),
        "KB": 1 / (1024**5),
        "MB": 1 / (1024**4),
        "GB": 1 / (1024**3),
        "TB": 1 / (1024**2),
        "PB": 1 / 1024,
        "EB": 1.0,
        "万GB".upper(): 10_000 / (1024**3),
        "亿GB".upper(): 100_000_000 / (1024**3),
        "万TB".upper(): 10_000 / (1024**2),
        "亿TB".upper(): 100_000_000 / (1024**2),
    }
    if unit_text not in factors:
        raise ValueError(f"不支持的流量单位：{unit}")
    return number * factors[unit_text]


def load_truth_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    province_field = None
    for candidate in ("province_name", "name_zh", "name"):
        if candidate in df.columns:
            province_field = candidate
            break
    if province_field is None:
        raise ValueError("省级统计表缺少省份名称列，需要 province_name 或 name_zh。")

    df = df.copy()
    df["province_name"] = df[province_field].astype(str).str.strip()
    df["province_norm"] = df["province_name"].map(normalize_province_name)

    if "traffic_eb" in df.columns:
        df["traffic_eb"] = pd.to_numeric(df["traffic_eb"], errors="coerce")
    elif {"traffic_value", "traffic_unit"}.issubset(df.columns):
        df["traffic_eb"] = [convert_to_eb(v, u) for v, u in zip(df["traffic_value"], df["traffic_unit"])]
    else:
        raise ValueError("省级统计表缺少流量列，需要 traffic_eb 或 traffic_value + traffic_unit。")

    if df["traffic_eb"].isna().any():
        raise ValueError("省级统计表存在无法转换为数值的流量记录。")

    if "adm1_code" in df.columns:
        df["adm1_code"] = df["adm1_code"].astype(str).str.strip()

    if df["province_norm"].duplicated().any():
        duplicates = sorted(df.loc[df["province_norm"].duplicated(), "province_name"].unique())
        raise ValueError(f"省级统计表中存在重复省份记录：{duplicates}")

    return df.sort_values("province_norm").reset_index(drop=True)


def load_matching_boundary(boundary_files: list[Path], truth_df: pd.DataFrame) -> tuple[gpd.GeoDataFrame, Path, str]:
    truth_codes = set(truth_df["adm1_code"]) if "adm1_code" in truth_df.columns else set()
    truth_names = set(truth_df["province_norm"])

    best_match_count = -1
    best_payload: tuple[gpd.GeoDataFrame, Path, str] | None = None
    best_error: str | None = None

    for path in boundary_files:
        try:
            gdf = gpd.read_file(path)
            if gdf.crs is None:
                continue
            if "adm0_a3" in gdf.columns:
                gdf = gdf[gdf["adm0_a3"].astype(str) == "CHN"].copy()
            if gdf.empty:
                continue

            if truth_codes and "adm1_code" in gdf.columns:
                gdf["match_key"] = gdf["adm1_code"].astype(str).str.strip()
                matched = gdf[gdf["match_key"].isin(truth_codes)].copy()
                match_count = len(matched["match_key"].unique())
                if match_count > best_match_count:
                    best_match_count = match_count
                    best_payload = (matched, path, "adm1_code")
                continue

            candidate_fields = [field for field in NAME_FIELD_CANDIDATES if field in gdf.columns]
            if not candidate_fields:
                continue
            for field in candidate_fields:
                matched = gdf.copy()
                matched["match_key"] = matched[field].astype(str).map(normalize_province_name)
                matched = matched[matched["match_key"].isin(truth_names)].copy()
                match_count = len(matched["match_key"].unique())
                if match_count > best_match_count:
                    best_match_count = match_count
                    best_payload = (matched, path, field)
        except Exception as exc:
            best_error = str(exc)

    if best_payload is None:
        raise ValueError(f"未找到可匹配中国省级统计的边界文件。{best_error or ''}".strip())

    matched_gdf, matched_path, match_field = best_payload
    if matched_gdf.empty:
        raise ValueError("已找到边界文件，但没有匹配到任何中国省份。")

    matched_gdf = matched_gdf.dissolve(by="match_key", as_index=False)
    return matched_gdf, matched_path, match_field


def build_province_mask(template_tif: Path, gdf: gpd.GeoDataFrame, province_to_id: dict[str, int]) -> np.ndarray:
    with rasterio.open(template_tif) as ds:
        out_shape = (ds.height, ds.width)
        transform = ds.transform
        crs = ds.crs

    gdf_aligned = gdf.to_crs(crs)
    shapes = []
    for _, row in gdf_aligned.iterrows():
        key = str(row["match_key"])
        province_id = province_to_id.get(key)
        if province_id is None:
            continue
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        shapes.append((geom, int(province_id)))

    if not shapes:
        raise ValueError("没有生成可用于栅格化的省级边界。")

    return rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        fill=BACKGROUND_ID,
        all_touched=False,
        dtype="int32",
    )


def aggregate_by_mask(tif_path: Path, mask: np.ndarray, nodata_val: float | None) -> tuple[dict[int, float], dict[int, int]]:
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}

    with rasterio.open(tif_path) as ds:
        for _, win in ds.block_windows(1):
            arr = ds.read(1, window=win).astype(np.float32)
            row_off, col_off = win.row_off, win.col_off
            mask_window = mask[row_off : row_off + win.height, col_off : col_off + win.width]
            valid = (mask_window != 0) & valid_mask(arr, nodata_val)
            if not np.any(valid):
                continue

            province_ids = mask_window[valid].astype(np.int32)
            values = arr[valid].astype(np.float64)
            unique_ids, inverse = np.unique(province_ids, return_inverse=True)
            value_sums = np.bincount(inverse, weights=values)
            value_counts = np.bincount(inverse)

            for idx, province_id in enumerate(unique_ids):
                pid = int(province_id)
                sums[pid] = sums.get(pid, 0.0) + float(value_sums[idx])
                counts[pid] = counts.get(pid, 0) + int(value_counts[idx])

    return sums, counts


def padded_log_limits(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    all_values = np.concatenate([x, y])
    positive = all_values[np.isfinite(all_values) & (all_values > 0)]
    if len(positive) == 0:
        raise ValueError("没有正值可用于绘图。")

    low = max(float(np.quantile(positive, 0.08)), float(positive.min()))
    high = max(float(np.quantile(positive, 0.99)), low * 10.0)
    return 10 ** (math.log10(low) - 0.10), 10 ** (math.log10(high) + 0.10)


def save_scatter(df: pd.DataFrame, output_path: Path, spearman_value: float, pearson_value: float, wmape: float) -> None:
    plot_df = df[(df["true_eb"] > 0) & (df["pred_eb_scaled"] > 0)].copy()
    if plot_df.empty:
        return

    plot_min, plot_max = padded_log_limits(
        plot_df["true_eb"].to_numpy(dtype=float),
        plot_df["pred_eb_scaled"].to_numpy(dtype=float),
    )

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.scatter(
        plot_df["true_eb"],
        plot_df["pred_eb_scaled"],
        s=42,
        alpha=0.92,
        color="#d1495b",
        edgecolors="white",
        linewidths=0.5,
        rasterized=True,
    )
    ax.plot([plot_min, plot_max], [plot_min, plot_max], "--", color="0.35", linewidth=1.2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(plot_min, plot_max)
    ax.set_ylim(plot_min, plot_max)
    ax.set_xlabel("MIIT provincial statistic (EB)")
    ax.set_ylabel("Aggregated demand result (scaled, EB)")
    ax.grid(True, which="major", alpha=0.25)
    ax.grid(True, which="minor", alpha=0.08)
    ax.text(
        0.04,
        0.96,
        (
            f"n = {len(plot_df)}\n"
            f"Spearman rho = {spearman_value:.3f}\n"
            f"Pearson r(log1p) = {pearson_value:.3f}\n"
            f"wMAPE = {wmape:.1f}%"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="0.8"),
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_results() -> tuple[list[CheckResult], dict[str, object]]:
    results: list[CheckResult] = []
    context: dict[str, object] = {
        "pred_tif": PRED_TIF,
        "boundary_dir": BOUNDARY_DIR,
        "truth_dir": TRUTH_DIR,
        "output_dir": OUTPUT_DIR,
    }

    ensure_dir(OUTPUT_DIR)
    truth_file = find_truth_file(TRUTH_DIR)
    boundary_files = find_boundary_files(BOUNDARY_DIR)

    results.append(CheckResult("主结果栅格存在", PRED_TIF.is_file(), f"文件路径：{PRED_TIF}"))
    results.append(
        CheckResult(
            "省级边界数据存在",
            bool(boundary_files),
            f"找到 {len(boundary_files)} 个候选边界文件" if boundary_files else f"边界目录：{BOUNDARY_DIR}",
        )
    )
    results.append(
        CheckResult(
            "工信部2024省级统计存在",
            truth_file is not None,
            f"统计文件：{truth_file}" if truth_file is not None else f"统计目录：{TRUTH_DIR}",
        )
    )

    if not PRED_TIF.is_file() or not boundary_files or truth_file is None:
        return results, context

    try:
        with rasterio.open(PRED_TIF) as pred_ds:
            pred_nodata = get_nodata(pred_ds, 1)
            context["pred_meta"] = {
                "crs": str(pred_ds.crs),
                "width": pred_ds.width,
                "height": pred_ds.height,
            }
        results.append(CheckResult("主结果栅格可读取", True, f"CRS={context['pred_meta']['crs']}，size={context['pred_meta']['width']}x{context['pred_meta']['height']}"))
    except Exception as exc:
        results.append(CheckResult("主结果栅格可读取", False, f"读取失败：{exc}"))
        return results, context

    try:
        truth_df = load_truth_table(truth_file)
        context["truth_df"] = truth_df
        results.append(CheckResult("工信部省级统计可读取", True, f"省份数：{len(truth_df)}"))
    except Exception as exc:
        results.append(CheckResult("工信部省级统计可读取", False, f"读取失败：{exc}"))
        return results, context

    try:
        boundary_gdf, boundary_file, match_field = load_matching_boundary(boundary_files, truth_df)
        context["boundary_gdf"] = boundary_gdf
        context["boundary_file"] = boundary_file
        context["match_field"] = match_field
        results.append(CheckResult("省级边界可读取", True, f"边界文件：{boundary_file}；匹配字段：{match_field}；边界数：{len(boundary_gdf)}"))
    except Exception as exc:
        results.append(CheckResult("省级边界可读取", False, f"读取失败：{exc}"))
        return results, context

    truth_df = truth_df.copy()
    if "adm1_code" in truth_df.columns and match_field == "adm1_code":
        truth_df["match_key"] = truth_df["adm1_code"].astype(str).str.strip()
    else:
        truth_df["match_key"] = truth_df["province_norm"]

    missing_in_boundary = sorted(set(truth_df["match_key"]) - set(boundary_gdf["match_key"]))
    results.append(
        CheckResult(
            "省级匹配关系完整",
            not missing_in_boundary,
            "所有统计省份均匹配到边界" if not missing_in_boundary else f"边界缺少省份：{missing_in_boundary}",
        )
    )
    if missing_in_boundary:
        return results, context

    truth_df["province_id"] = np.arange(1, len(truth_df) + 1, dtype=np.int32)
    province_to_id = dict(zip(truth_df["match_key"], truth_df["province_id"]))

    try:
        province_mask = build_province_mask(PRED_TIF, boundary_gdf, province_to_id)
        pred_sums, pred_counts = aggregate_by_mask(PRED_TIF, province_mask, pred_nodata)
        results.append(CheckResult("省级聚合结果可计算", True, f"成功聚合 {len(pred_sums)} 个省份"))
    except Exception as exc:
        results.append(CheckResult("省级聚合结果可计算", False, f"计算失败：{exc}"))
        return results, context

    rows = []
    for _, row in truth_df.iterrows():
        province_id = int(row["province_id"])
        rows.append(
            {
                "province_id": province_id,
                "province_name": str(row["province_name"]),
                "match_key": str(row["match_key"]),
                "true_eb": float(row["traffic_eb"]),
                "pred_sum_raw": float(pred_sums.get(province_id, 0.0)),
                "pred_pixel_count": int(pred_counts.get(province_id, 0)),
            }
        )
    df = pd.DataFrame(rows)

    matched_count = int((df["pred_pixel_count"] > 0).sum())
    results.append(CheckResult("有效省份数量充足", matched_count >= MIN_MATCHED_PROVINCES, f"有效省份数：{matched_count}，阈值：{MIN_MATCHED_PROVINCES}"))
    if matched_count < MIN_MATCHED_PROVINCES:
        return results, context

    true_total = float(df["true_eb"].sum())
    pred_total = float(df["pred_sum_raw"].sum())
    if pred_total <= 0:
        results.append(CheckResult("量级缩放可计算", False, "聚合后的预测总量非正，无法缩放"))
        return results, context

    scale_factor = true_total / pred_total
    df["pred_eb_scaled"] = df["pred_sum_raw"] * scale_factor

    spearman_value, spearman_p = spearmanr(df["true_eb"], df["pred_sum_raw"])
    pearson_value, pearson_p = pearsonr(np.log1p(df["true_eb"]), np.log1p(df["pred_eb_scaled"]))
    wmape = float(np.sum(np.abs(df["pred_eb_scaled"] - df["true_eb"])) / np.sum(df["true_eb"]) * 100.0)
    rmse = float(np.sqrt(np.mean((df["pred_eb_scaled"] - df["true_eb"]) ** 2)))

    results.append(CheckResult("量级缩放可计算", True, f"scale_factor={scale_factor:.8f}"))
    results.append(
        CheckResult(
            "排序结构保持合理一致性",
            float(spearman_value) >= MIN_SPEARMAN_RHO,
            f"Spearman rho={spearman_value:.4f}，阈值={MIN_SPEARMAN_RHO:.2f}，p={spearman_p:.2e}",
        )
    )
    results.append(
        CheckResult(
            "量级关系保持合理一致性",
            float(pearson_value) >= MIN_PEARSON_LOG,
            f"Pearson r(log1p)={pearson_value:.4f}，阈值={MIN_PEARSON_LOG:.2f}，p={pearson_p:.2e}，wMAPE={wmape:.1f}%",
        )
    )

    output_table = OUTPUT_DIR / "china_province_consistency_table.csv"
    output_summary = OUTPUT_DIR / "china_province_consistency_summary.txt"
    output_scatter = OUTPUT_DIR / "china_province_consistency_scatter.pdf"
    df.sort_values("true_eb", ascending=False).to_csv(output_table, index=False, encoding="utf-8-sig")
    save_scatter(df, output_scatter, float(spearman_value), float(pearson_value), wmape)

    summary_lines = [
        "中国省级一致性验证摘要",
        f"主结果栅格：{PRED_TIF}",
        f"边界文件：{boundary_file}",
        f"统计文件：{truth_file}",
        f"匹配字段：{match_field}",
        f"省份数：{len(df)}",
        f"有效省份数：{matched_count}",
        f"scale_factor：{scale_factor}",
        f"Spearman rho：{spearman_value}",
        f"Spearman p-value：{spearman_p}",
        f"Pearson r(log1p)：{pearson_value}",
        f"Pearson p-value：{pearson_p}",
        f"wMAPE (%)：{wmape}",
        f"RMSE (EB)：{rmse}",
        f"输出表：{output_table}",
        f"摘要：{output_summary}",
        f"散点图：{output_scatter}",
    ]
    output_summary.write_text("\n".join(summary_lines), encoding="utf-8")

    context["output_table"] = output_table
    context["output_summary"] = output_summary
    context["output_scatter"] = output_scatter
    context["metrics"] = {
        "spearman_rho": float(spearman_value),
        "pearson_log": float(pearson_value),
        "wmape_percent": wmape,
        "rmse_eb": rmse,
    }

    return results, context


def print_report(results: list[CheckResult], context: dict[str, object]) -> None:
    passed_count = sum(result.passed for result in results)
    print("中国省级一致性测试报告")
    print(f"主结果栅格：{context['pred_tif']}")
    print(f"边界目录：{context['boundary_dir']}")
    print(f"统计目录：{context['truth_dir']}")
    print(f"输出目录：{context['output_dir']}")
    print(f"检查结果：{passed_count}/{len(results)} 项通过")
    print()

    if "metrics" in context:
        metrics = context["metrics"]
        print("核心指标：")
        print(f"- Spearman rho: {metrics['spearman_rho']:.4f}")
        print(f"- Pearson r(log1p): {metrics['pearson_log']:.4f}")
        print(f"- wMAPE: {metrics['wmape_percent']:.1f}%")
        print(f"- RMSE: {metrics['rmse_eb']:.4f} EB")
        print()

    print("详细结果：")
    for result in results:
        status = "通过" if result.passed else "未通过"
        print(f"- [{status}] {result.name}：{result.details}")

    if all(result.passed for result in results) and "output_table" in context:
        print()
        print("输出文件：")
        print(f"- 表格：{context['output_table']}")
        print(f"- 摘要：{context['output_summary']}")
        print(f"- 散点图：{context['output_scatter']}")


def main() -> int:
    results, context = build_results()
    print_report(results, context)
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
