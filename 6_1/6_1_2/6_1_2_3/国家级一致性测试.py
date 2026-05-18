from __future__ import annotations

import re
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
TRUTH_DIR = INPUT_ROOT / "truth" / "country_level"
OUTPUT_DIR = SCRIPT_DIR / "outputs"

BOUNDARY_SUFFIXES = {".gpkg", ".shp", ".geojson"}
TRUTH_SUFFIXES = {".csv", ".xlsx", ".xls"}
BACKGROUND_ID = 0
NODATA_FALLBACK = -9999.0

MIN_MATCHED_COUNTRIES = 20
MIN_VAL_COUNTRIES = 10
MIN_SPEARMAN_RHO = 0.60
MIN_PEARSON_LOG = 0.60
MIN_TOTAL_EB_FOR_SCATTER = 0.2
LOWER_QUANTILE = 0.08
UPPER_QUANTILE = 0.99
PAD_DECADES = 0.10

COUNTRY_CODE_FIELD_CANDIDATES = (
    "ADM0_A3",
    "adm0_a3",
    "ISO_A3",
    "iso_a3",
    "SOV_A3",
    "sov_a3",
    "WB_A3",
    "wb_a3",
)
COUNTRY_NAME_FIELD_CANDIDATES = (
    "ADMIN",
    "admin",
    "NAME_EN",
    "name_en",
    "NAME",
    "name",
    "NAME_LONG",
    "name_long",
    "FORMAL_EN",
    "formal_en",
    "ABBREV",
    "abbrev",
    "NAME_SORT",
    "name_sort",
    "NAME_CIAWF",
    "name_ciawf",
)

COUNTRY_ALIASES = {
    "People's Republic of Bangladesh": "BGD",
    "People’s Republic of Bangladesh": "BGD",
    "Democratic Republic of Congo": "COD",
    "Republic of Congo": "COG",
    "Democratic People's Republic of Algeria": "DZA",
    "Democratic People’s Republic of Algeria": "DZA",
    "Republic of Korea": "KOR",
    "Korea, Republic of": "KOR",
    "Democratic People's Republic of Korea": "PRK",
    "Democratic People’s Republic of Korea": "PRK",
    "Republic of Fiji Islands": "FJI",
    "Kingdom of Swaziland": "SWZ",
    "Socialist Republic of Viet Nam": "VNM",
    "Union of Myanmar": "MMR",
}

SPLIT_ORDER = ["all", "train", "val"]
SPLIT_COLORS = {"all": "#264653", "train": "#8ea3b0", "val": "#d1495b"}


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
    if "admin_0" in text or "adm0" in text or "country" in text:
        score -= 20
    if "admin_1" in text or "adm1" in text or "province" in text:
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


def normalize_country_name(name: object) -> str:
    text = str(name).strip().lower()
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
    text = re.sub(r"^[\W_]+|[\W_]+$", "", text)

    prefixes = [
        "democratic people's republic of ",
        "democratic peoples republic of ",
        "people's republic of ",
        "peoples republic of ",
        "republic of ",
        "kingdom of ",
        "commonwealth of the ",
        "commonwealth of ",
        "state of ",
        "federal republic of ",
        "federative republic of ",
        "islamic republic of ",
        "democratic republic of the ",
        "democratic republic of ",
        "socialist republic of ",
        "plurinational state of ",
        "united states of ",
        "union of ",
        "the ",
    ]
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break

    special = {
        "somalia republic": "somalia",
        "slovakia republic": "slovakia",
    }
    text = special.get(text, text)
    text = text.replace("&", "and")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def select_country_code_field(gdf: gpd.GeoDataFrame) -> str:
    for field in COUNTRY_CODE_FIELD_CANDIDATES:
        if field in gdf.columns:
            return field
    raise ValueError(f"国家边界文件缺少 ISO3 编码字段，候选字段：{COUNTRY_CODE_FIELD_CANDIDATES}")


def build_name_index(gdf: gpd.GeoDataFrame) -> dict[str, set[str]]:
    index: dict[str, set[str]] = {}
    code_field = select_country_code_field(gdf)
    for _, row in gdf.iterrows():
        iso3 = str(row[code_field]).strip().upper()
        for field in COUNTRY_NAME_FIELD_CANDIDATES:
            if field not in gdf.columns:
                continue
            value = row.get(field)
            if value is None or pd.isna(value):
                continue
            normalized = normalize_country_name(value)
            if normalized:
                index.setdefault(normalized, set()).add(iso3)
    return index


def load_truth_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    df = df.copy()

    gt_field = None
    for candidate in ("gt_total_eb", "traffic_eb", "total_eb"):
        if candidate in df.columns:
            gt_field = candidate
            break
    if gt_field is None:
        raise ValueError("国家统计表缺少真值总量列，需要 gt_total_eb、traffic_eb 或 total_eb。")

    iso3_field = None
    for candidate in ("matched_iso3", "iso3", "ISO3", "adm0_a3", "ADM0_A3"):
        if candidate in df.columns:
            iso3_field = candidate
            break

    country_name_field = None
    for candidate in ("Country_en", "country_en", "country_name", "country", "name_en", "name"):
        if candidate in df.columns:
            country_name_field = candidate
            break

    if iso3_field is None and country_name_field is None:
        raise ValueError("国家统计表至少需要 iso3 或 Country_en/country_name 列。")

    if country_name_field is None:
        df["Country_en"] = df[iso3_field].astype(str).str.strip().str.upper()
    else:
        df["Country_en"] = df[country_name_field].astype(str).str.strip()

    if "country_id" not in df.columns:
        df["country_id"] = np.arange(1, len(df) + 1, dtype=np.int32)
    if "country_name" not in df.columns:
        df["country_name"] = df["Country_en"]
    if "role" not in df.columns:
        df["role"] = "all"
    df["role"] = df["role"].astype(str).str.strip().str.lower()

    df["gt_total_eb"] = pd.to_numeric(df[gt_field], errors="coerce")
    if df["gt_total_eb"].isna().any():
        raise ValueError("国家统计表存在无法转换为数值的 gt_total_eb。")

    if iso3_field is not None:
        df["matched_iso3"] = df[iso3_field].astype(str).str.strip().str.upper()
    else:
        df["matched_iso3"] = ""

    if df["country_id"].duplicated().any():
        raise ValueError("国家统计表中的 country_id 存在重复。")

    return df.reset_index(drop=True)


def attach_iso3_codes(df_template: pd.DataFrame, gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    if (df_template["matched_iso3"] != "").all():
        return df_template.copy()

    name_index = build_name_index(gdf)
    df = df_template.copy()

    for idx, row in df.iterrows():
        if df.at[idx, "matched_iso3"]:
            continue

        country_en = str(row["Country_en"])
        if country_en in COUNTRY_ALIASES:
            df.at[idx, "matched_iso3"] = COUNTRY_ALIASES[country_en]
            continue

        normalized = normalize_country_name(country_en)
        hits = sorted(name_index.get(normalized, []))
        if len(hits) == 1:
            df.at[idx, "matched_iso3"] = hits[0]
            continue
        if len(hits) > 1:
            raise ValueError(f"国家名称匹配存在歧义：{country_en} -> {hits}")
        raise ValueError(f"无法将国家名称匹配到边界文件：{country_en}")

    if df["matched_iso3"].duplicated().any():
        dupes = df.loc[df["matched_iso3"].duplicated(keep=False), ["Country_en", "matched_iso3"]]
        raise ValueError(f"国家 ISO3 匹配存在重复：\n{dupes.to_string(index=False)}")
    return df


def load_matching_boundary(boundary_files: list[Path]) -> tuple[gpd.GeoDataFrame, Path, str]:
    last_error = None
    for path in boundary_files:
        try:
            gdf = gpd.read_file(path)
            if gdf.crs is None:
                continue
            code_field = select_country_code_field(gdf)
            return gdf, path, code_field
        except Exception as exc:  # pragma: no cover
            last_error = str(exc)
    raise ValueError(f"未找到可用的国家边界文件。{last_error or ''}".strip())


def build_country_mask(template_tif: Path, gdf_admin0: gpd.GeoDataFrame, iso3_to_country_id: dict[str, int], code_field: str) -> np.ndarray:
    with rasterio.open(template_tif) as ds:
        out_shape = (ds.height, ds.width)
        transform = ds.transform
        crs = ds.crs

    gdf_admin0 = gdf_admin0.to_crs(crs)
    shapes = []
    for _, row in gdf_admin0.iterrows():
        iso3 = str(row[code_field]).strip().upper()
        if iso3 not in iso3_to_country_id:
            continue
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        shapes.append((geom, int(iso3_to_country_id[iso3])))

    if not shapes:
        raise ValueError("国家边界没有与统计表匹配上的有效几何。")

    return rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        fill=BACKGROUND_ID,
        all_touched=False,
        dtype="int32",
    )


def aggregate_country_totals(demand_tif: Path, country_mask: np.ndarray, nodata_val: float | None) -> tuple[dict[int, float], dict[int, int], dict[int, int], dict[int, int]]:
    total_sum: dict[int, float] = {}
    total_count: dict[int, int] = {}
    total_mask_pixels: dict[int, int] = {}
    total_tiles: dict[int, int] = {}

    with rasterio.open(demand_tif) as ds:
        for _, win in ds.block_windows(1):
            arr = ds.read(1, window=win).astype(np.float32)
            row_off, col_off = win.row_off, win.col_off
            mask_window = country_mask[row_off : row_off + win.height, col_off : col_off + win.width]

            inside = mask_window != 0
            if np.any(inside):
                ids_all = mask_window[inside].astype(np.int32)
                unique_all, counts_all = np.unique(ids_all, return_counts=True)
                for country_id, count in zip(unique_all, counts_all):
                    total_mask_pixels[int(country_id)] = total_mask_pixels.get(int(country_id), 0) + int(count)

            valid = inside & valid_mask(arr, nodata_val)
            if not np.any(valid):
                continue

            ids_valid = mask_window[valid].astype(np.int32)
            values = arr[valid].astype(np.float64)
            unique_ids, inverse = np.unique(ids_valid, return_inverse=True)
            value_sums = np.bincount(inverse, weights=values)
            value_counts = np.bincount(inverse)

            for idx, country_id in enumerate(unique_ids):
                cid = int(country_id)
                total_sum[cid] = total_sum.get(cid, 0.0) + float(value_sums[idx])
                total_count[cid] = total_count.get(cid, 0) + int(value_counts[idx])
                total_tiles[cid] = total_tiles.get(cid, 0) + 1

    return total_sum, total_count, total_mask_pixels, total_tiles


def padded_log_limits(values_x: np.ndarray, values_y: np.ndarray, lower_q: float, upper_q: float, pad_decades: float) -> tuple[float, float]:
    all_values = np.concatenate([values_x, values_y])
    positive = all_values[np.isfinite(all_values) & (all_values > 0)]
    if len(positive) == 0:
        raise ValueError("没有正值可用于绘图。")

    low = np.quantile(positive, lower_q)
    high = np.quantile(positive, upper_q)
    low = max(low, positive.min())
    high = max(high, low * 10)

    low = 10 ** (np.log10(low) - pad_decades)
    high = 10 ** (np.log10(high) + pad_decades)
    return low, high


def select_label_points(val_df: pd.DataFrame) -> pd.DataFrame:
    label_rows = []

    high_over = val_df.sort_values("ratio", ascending=False).iloc[0]
    label_rows.append(high_over)

    high_under = val_df.sort_values("ratio", ascending=True).iloc[0]
    if high_under["Country_en"] not in {row["Country_en"] for row in label_rows}:
        label_rows.append(high_under)

    q1 = val_df["gt_total_eb"].quantile(0.25)
    q3 = val_df["gt_total_eb"].quantile(0.75)
    mid_df = val_df[(val_df["gt_total_eb"] >= q1) & (val_df["gt_total_eb"] <= q3)].copy()
    if mid_df.empty:
        mid_df = val_df.copy()

    mid_outlier = (
        mid_df[~mid_df["Country_en"].isin({row["Country_en"] for row in label_rows})]
        .sort_values("log_abs_err", ascending=False)
        .head(1)
    )
    if not mid_outlier.empty:
        label_rows.append(mid_outlier.iloc[0])

    return pd.DataFrame(label_rows).drop_duplicates(subset="Country_en")


def compute_split_metrics(df: pd.DataFrame, split_name: str) -> dict[str, float | int | str]:
    if split_name == "all":
        sub = df.copy()
    else:
        sub = df[df["role"] == split_name].copy()

    positive = sub[(sub["gt_total_eb"] > 0) & (sub["pred_total_eb"] > 0)].copy()
    if positive.empty:
        return {
            "split": split_name,
            "n_total": int(len(sub)),
            "n_positive": 0,
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
            "pearson_log": np.nan,
            "pearson_log_p": np.nan,
            "wmape_percent": np.nan,
            "rmse_eb": np.nan,
            "pred_total_sum": float(sub["pred_total_eb"].sum()) if "pred_total_eb" in sub.columns else np.nan,
            "gt_total_sum": float(sub["gt_total_eb"].sum()) if "gt_total_eb" in sub.columns else np.nan,
            "total_ratio": np.nan,
        }

    gt = positive["gt_total_eb"].to_numpy(dtype=float)
    pred = positive["pred_total_eb"].to_numpy(dtype=float)
    spearman_value, spearman_p = spearmanr(gt, pred)
    pearson_value, pearson_p = pearsonr(np.log1p(gt), np.log1p(pred))
    wmape = float(np.sum(np.abs(pred - gt)) / np.sum(gt) * 100.0)
    rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))
    pred_total_sum = float(positive["pred_total_eb"].sum())
    gt_total_sum = float(positive["gt_total_eb"].sum())
    total_ratio = pred_total_sum / gt_total_sum if gt_total_sum > 0 else np.nan

    return {
        "split": split_name,
        "n_total": int(len(sub)),
        "n_positive": int(len(positive)),
        "spearman_rho": float(spearman_value),
        "spearman_p": float(spearman_p),
        "pearson_log": float(pearson_value),
        "pearson_log_p": float(pearson_p),
        "wmape_percent": wmape,
        "rmse_eb": rmse,
        "pred_total_sum": pred_total_sum,
        "gt_total_sum": gt_total_sum,
        "total_ratio": float(total_ratio) if np.isfinite(total_ratio) else np.nan,
    }


def save_scatter_with_roles(df: pd.DataFrame, output_path: Path, val_metrics: dict[str, float | int | str]) -> None:
    plot_df = df.copy()
    plot_df["gt_total_eb"] = pd.to_numeric(plot_df["gt_total_eb"], errors="coerce")
    plot_df["pred_total_eb"] = pd.to_numeric(plot_df["pred_total_eb"], errors="coerce")
    plot_df = plot_df.dropna(subset=["gt_total_eb", "pred_total_eb", "role"])
    plot_df = plot_df[
        (plot_df["gt_total_eb"] >= MIN_TOTAL_EB_FOR_SCATTER)
        & (plot_df["pred_total_eb"] >= MIN_TOTAL_EB_FOR_SCATTER)
    ].copy()

    train_df = plot_df[plot_df["role"] == "train"].copy()
    val_df = plot_df[plot_df["role"] == "val"].copy()
    if val_df.empty:
        return

    val_df["log_abs_err"] = np.abs(np.log10(val_df["pred_total_eb"]) - np.log10(val_df["gt_total_eb"]))
    val_df["ratio"] = val_df["pred_total_eb"] / val_df["gt_total_eb"]
    label_df = select_label_points(val_df)

    plot_min, plot_max = padded_log_limits(
        plot_df["gt_total_eb"].to_numpy(dtype=float),
        plot_df["pred_total_eb"].to_numpy(dtype=float),
        lower_q=LOWER_QUANTILE,
        upper_q=UPPER_QUANTILE,
        pad_decades=PAD_DECADES,
    )

    fig, ax = plt.subplots(figsize=(7.2, 6.4))

    if not train_df.empty:
        ax.scatter(
            train_df["gt_total_eb"].clip(lower=plot_min, upper=plot_max),
            train_df["pred_total_eb"].clip(lower=plot_min, upper=plot_max),
            s=14,
            alpha=0.16,
            color=SPLIT_COLORS["train"],
            label="Train countries",
            rasterized=True,
        )
    ax.scatter(
        val_df["gt_total_eb"].clip(lower=plot_min, upper=plot_max),
        val_df["pred_total_eb"].clip(lower=plot_min, upper=plot_max),
        s=42,
        alpha=0.95,
        color=SPLIT_COLORS["val"],
        edgecolors="white",
        linewidths=0.5,
        label="Validation countries",
        rasterized=True,
    )

    ax.plot([plot_min, plot_max], [plot_min, plot_max], linestyle="--", linewidth=1.2, color="0.35", label="1:1 line")

    for _, row in label_df.iterrows():
        x = float(np.clip(row["gt_total_eb"], plot_min, plot_max))
        y = float(np.clip(row["pred_total_eb"], plot_min, plot_max))
        if row["ratio"] > 1.2:
            offset = (6, 6)
        elif row["ratio"] < 0.83:
            offset = (6, -10)
        else:
            offset = (-34, 6)
        ax.annotate(
            row["Country_en"],
            (x, y),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
            color="#202020",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.75, edgecolor="none"),
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(plot_min, plot_max)
    ax.set_ylim(plot_min, plot_max)
    ax.set_xlabel("Reported national total (EB)")
    ax.set_ylabel("Predicted national total (EB)")

    metrics_text = "\n".join(
        [
            f"Validation countries: n = {val_metrics['n_positive']}",
            f"Pearson r (log1p) = {val_metrics['pearson_log']:.3f}",
            f"Spearman rho = {val_metrics['spearman_rho']:.3f}",
            f"wMAPE = {val_metrics['wmape_percent']:.1f}%",
        ]
    )
    ax.text(
        0.04,
        0.96,
        metrics_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="0.8"),
    )

    ax.legend(frameon=False, loc="lower right")
    ax.grid(True, which="major", alpha=0.25)
    ax.grid(True, which="minor", alpha=0.08)

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
            "国家边界数据存在",
            bool(boundary_files),
            f"找到 {len(boundary_files)} 个候选边界文件" if boundary_files else f"边界目录：{BOUNDARY_DIR}",
        )
    )
    results.append(
        CheckResult(
            "国家统计表存在",
            truth_file is not None,
            f"统计文件：{truth_file}" if truth_file is not None else f"统计目录：{TRUTH_DIR}",
        )
    )

    if not PRED_TIF.is_file() or not boundary_files or truth_file is None:
        return results, context

    try:
        with rasterio.open(PRED_TIF) as pred_ds:
            demand_nodata = get_nodata(pred_ds, 1)
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
        results.append(CheckResult("国家统计表可读取", True, f"国家数：{len(truth_df)}"))
    except Exception as exc:
        results.append(CheckResult("国家统计表可读取", False, f"读取失败：{exc}"))
        return results, context

    try:
        gdf_admin0, boundary_file, code_field = load_matching_boundary(boundary_files)
        context["boundary_file"] = boundary_file
        context["code_field"] = code_field
        results.append(CheckResult("国家边界可读取", True, f"边界文件：{boundary_file}；编码字段：{code_field}；边界数：{len(gdf_admin0)}"))
    except Exception as exc:
        results.append(CheckResult("国家边界可读取", False, f"读取失败：{exc}"))
        return results, context

    try:
        df_eval = attach_iso3_codes(truth_df, gdf_admin0)
    except Exception as exc:
        results.append(CheckResult("国家匹配关系完整", False, f"匹配失败：{exc}"))
        return results, context
    results.append(CheckResult("国家匹配关系完整", True, f"成功匹配 {len(df_eval)} 个国家"))

    iso3_to_country_id = dict(zip(df_eval["matched_iso3"], df_eval["country_id"]))
    try:
        country_mask = build_country_mask(PRED_TIF, gdf_admin0, iso3_to_country_id, code_field)
        pred_sums, valid_counts, mask_pixels, tile_counts = aggregate_country_totals(PRED_TIF, country_mask, demand_nodata)
        results.append(CheckResult("国家聚合结果可计算", True, f"成功聚合 {len(pred_sums)} 个国家"))
    except Exception as exc:
        results.append(CheckResult("国家聚合结果可计算", False, f"计算失败：{exc}"))
        return results, context

    df_out = df_eval.copy()
    df_out["pred_total_raw_eb"] = df_out["country_id"].map(pred_sums).fillna(0.0).astype(float)
    df_out["pred_total_eb"] = df_out["pred_total_raw_eb"]
    df_out["pred_gt_ratio"] = np.where(df_out["gt_total_eb"] > 0, df_out["pred_total_eb"] / df_out["gt_total_eb"], np.nan)
    df_out["abs_error_eb"] = np.abs(df_out["pred_total_eb"] - df_out["gt_total_eb"])
    df_out["signed_error_eb"] = df_out["pred_total_eb"] - df_out["gt_total_eb"]
    df_out["rel_error"] = np.where(df_out["gt_total_eb"] > 0, df_out["signed_error_eb"] / df_out["gt_total_eb"], np.nan)
    df_out["ape_percent"] = np.where(df_out["gt_total_eb"] > 0, df_out["abs_error_eb"] / df_out["gt_total_eb"] * 100.0, np.nan)
    df_out["tile_count"] = df_out["country_id"].map(tile_counts).fillna(0).astype(int)
    df_out["mask_pixels"] = df_out["country_id"].map(mask_pixels).fillna(0).astype(int)
    df_out["valid_pixels_seen"] = df_out["country_id"].map(valid_counts).fillna(0).astype(int)

    matched_count = int((df_out["valid_pixels_seen"] > 0).sum())
    results.append(CheckResult("有效国家数量充足", matched_count >= MIN_MATCHED_COUNTRIES, f"有效国家数：{matched_count}，阈值：{MIN_MATCHED_COUNTRIES}"))
    if matched_count < MIN_MATCHED_COUNTRIES:
        return results, context

    metrics_records = [compute_split_metrics(df_out, split) for split in SPLIT_ORDER]
    metrics_df = pd.DataFrame(metrics_records)
    context["metrics_df"] = metrics_df

    val_metrics = metrics_df.set_index("split").loc["val"].to_dict() if "val" in set(metrics_df["split"]) else {}
    if not val_metrics or int(val_metrics.get("n_positive", 0)) < MIN_VAL_COUNTRIES:
        results.append(CheckResult("验证国家数量充足", False, f"验证国家有效样本数不足：{int(val_metrics.get('n_positive', 0)) if val_metrics else 0}，阈值：{MIN_VAL_COUNTRIES}"))
        return results, context
    results.append(CheckResult("验证国家数量充足", True, f"验证国家有效样本数：{int(val_metrics['n_positive'])}"))

    results.append(
        CheckResult(
            "验证国家排序结构保持合理一致性",
            float(val_metrics["spearman_rho"]) >= MIN_SPEARMAN_RHO,
            f"Spearman rho={float(val_metrics['spearman_rho']):.4f}，阈值={MIN_SPEARMAN_RHO:.2f}，p={float(val_metrics['spearman_p']):.2e}",
        )
    )
    results.append(
        CheckResult(
            "验证国家量级关系保持合理一致性",
            float(val_metrics["pearson_log"]) >= MIN_PEARSON_LOG,
            f"Pearson r(log1p)={float(val_metrics['pearson_log']):.4f}，阈值={MIN_PEARSON_LOG:.2f}，p={float(val_metrics['pearson_log_p']):.2e}，wMAPE={float(val_metrics['wmape_percent']):.1f}%",
        )
    )

    output_table = OUTPUT_DIR / "country_level_consistency_table.csv"
    output_summary = OUTPUT_DIR / "country_level_consistency_summary.txt"
    output_scatter = OUTPUT_DIR / "country_level_consistency_scatter.pdf"
    preferred_cols = [
        "country_id",
        "country_name",
        "Country_cn",
        "Country_en",
        "role",
        "gt_total_eb",
        "matched_iso3",
        "pred_total_raw_eb",
        "pred_total_eb",
        "pred_gt_ratio",
        "abs_error_eb",
        "signed_error_eb",
        "rel_error",
        "ape_percent",
        "tile_count",
        "mask_pixels",
        "valid_pixels_seen",
    ]
    cols = [col for col in preferred_cols if col in df_out.columns]
    remaining = [col for col in df_out.columns if col not in cols]
    df_out = df_out[cols + remaining]
    df_out.sort_values("gt_total_eb", ascending=False).to_csv(output_table, index=False, encoding="utf-8-sig")
    save_scatter_with_roles(df_out, output_scatter, val_metrics)

    summary_lines = [
        "国家级一致性验证摘要",
        f"主结果栅格：{PRED_TIF}",
        f"边界文件：{boundary_file}",
        f"统计文件：{truth_file}",
        f"国家数：{len(df_out)}",
        f"有效国家数：{matched_count}",
        "",
        "分组指标：",
    ]
    for split in SPLIT_ORDER:
        row = metrics_df.set_index("split").loc[split]
        summary_lines.extend(
            [
                f"[{split}] n_positive = {int(row['n_positive'])}",
                f"[{split}] Spearman rho = {row['spearman_rho']}",
                f"[{split}] Pearson r(log1p) = {row['pearson_log']}",
                f"[{split}] wMAPE (%) = {row['wmape_percent']}",
                f"[{split}] RMSE (EB) = {row['rmse_eb']}",
                f"[{split}] total_ratio = {row['total_ratio']}",
            ]
        )
    summary_lines.extend(
        [
            "",
            f"输出表：{output_table}",
            f"摘要：{output_summary}",
            f"散点图：{output_scatter}",
        ]
    )
    output_summary.write_text("\n".join(summary_lines), encoding="utf-8")

    context["output_table"] = output_table
    context["output_summary"] = output_summary
    context["output_scatter"] = output_scatter
    context["val_metrics"] = val_metrics

    return results, context


def print_report(results: list[CheckResult], context: dict[str, object]) -> None:
    passed_count = sum(result.passed for result in results)
    print("国家级一致性测试报告")
    print(f"主结果栅格：{context['pred_tif']}")
    print(f"边界目录：{context['boundary_dir']}")
    print(f"统计目录：{context['truth_dir']}")
    print(f"输出目录：{context['output_dir']}")
    print(f"检查结果：{passed_count}/{len(results)} 项通过")
    print()

    if "metrics_df" in context:
        print("分组核心指标：")
        metrics_df = context["metrics_df"].set_index("split")
        for split in SPLIT_ORDER:
            if split not in metrics_df.index:
                continue
            row = metrics_df.loc[split]
            print(
                f"- {split}: n={int(row['n_positive'])}, "
                f"Spearman={row['spearman_rho']:.4f}, "
                f"Pearson(log1p)={row['pearson_log']:.4f}, "
                f"wMAPE={row['wmape_percent']:.1f}%, "
                f"total_ratio={row['total_ratio']:.4f}"
            )
        print()

    print("详细结果：")
    for result in results:
        status = "通过" if result.passed else "未通过"
        print(f"- [{status}] {result.name}：{result.details}")

    if all(result.passed for result in results) and "output_table" in context:
        print()
        print("输出文件：")
        print(f"- 总表：{context['output_table']}")
        print(f"- 摘要：{context['output_summary']}")
        print(f"- 散点图：{context['output_scatter']}")


def main() -> int:
    results, context = build_results()
    print_report(results, context)
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
