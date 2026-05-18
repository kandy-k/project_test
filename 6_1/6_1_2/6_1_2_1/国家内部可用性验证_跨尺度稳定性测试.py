from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"

PRED_TIF = REPO_ROOT / "dataset" / "products" / "demand" / "pred_final_pred_masked.tif"
COUNTRY_INPUT_ROOT = REPO_ROOT / "dataset" / "validation"
BOUNDARY_DIR = COUNTRY_INPUT_ROOT / "boundaries"
TRUTH_DIR = COUNTRY_INPUT_ROOT / "truth" / "country_level"
COUNTRY_LEVEL_OUTPUT_TABLE = (
    REPO_ROOT / "6_1" / "6_1_2" / "6_1_2_3" / "outputs" / "country_level_consistency_table.csv"
)

BOUNDARY_SUFFIXES = {".gpkg", ".shp", ".geojson"}
TRUTH_SUFFIXES = {".csv", ".xlsx", ".xls"}
BACKGROUND_ID = 0
NODATA_FALLBACK = -9999.0

SCALE_FACTORS = [10, 100]
TOP_P_LIST = [5, 10, 20]
SAMPLE_PER_COUNTRY = 8000
RANDOM_SEED = 1234
SAVE_COARSE_NPY = True

MIN_COUNTRIES_WITH_PIXELS = 20
CORE_SCALE_FACTOR = 10
CORE_TOP_P = 10
MIN_CORE_JACCARD = 0.40
MIN_CORE_OVERLAP = 0.60

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


def select_country_code_field(gdf: gpd.GeoDataFrame) -> str:
    for field in COUNTRY_CODE_FIELD_CANDIDATES:
        if field in gdf.columns:
            return field
    raise ValueError(f"国家边界文件缺少 ISO3 编码字段，候选字段：{COUNTRY_CODE_FIELD_CANDIDATES}")


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


def load_truth_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    df = df.copy()

    iso3_field = None
    for candidate in ("matched_iso3", "iso3", "ISO3", "adm0_a3", "ADM0_A3"):
        if candidate in df.columns:
            iso3_field = candidate
            break

    name_field = None
    for candidate in ("Country_en", "country_en", "country_name", "country", "name_en", "name"):
        if candidate in df.columns:
            name_field = candidate
            break

    role_field = "role" if "role" in df.columns else None

    out = pd.DataFrame(index=df.index.copy())
    if "country_id" in df.columns:
        out["source_country_id"] = pd.to_numeric(df["country_id"], errors="coerce")
    out["Country_en"] = df[name_field].astype(str).str.strip() if name_field is not None else ""
    out["country_name"] = out["Country_en"].replace("", np.nan).fillna("UNKNOWN")
    out["role"] = df[role_field].astype(str).str.strip().str.lower() if role_field is not None else "all"

    if iso3_field is not None:
        out["matched_iso3"] = df[iso3_field].astype(str).str.strip().str.upper()
    else:
        if not COUNTRY_LEVEL_OUTPUT_TABLE.is_file():
            raise ValueError("国家统计表缺少 matched_iso3/iso3，且未找到 6.1.2.3 输出表用于补全 ISO3。")
        ref_df = pd.read_csv(COUNTRY_LEVEL_OUTPUT_TABLE)
        if "matched_iso3" not in ref_df.columns:
            raise ValueError("6.1.2.3 输出表中缺少 matched_iso3 列，无法补全国家编码。")

        ref_df = ref_df.copy()
        ref_df["matched_iso3"] = ref_df["matched_iso3"].astype(str).str.strip().str.upper()
        merged = None

        if "source_country_id" in out.columns and "country_id" in ref_df.columns:
            merged = out.merge(
                ref_df[["country_id", "matched_iso3"]].drop_duplicates(),
                left_on="source_country_id",
                right_on="country_id",
                how="left",
            )

        if merged is None or merged["matched_iso3"].isna().all():
            if "Country_en" in ref_df.columns:
                merged = out.merge(
                    ref_df[["Country_en", "matched_iso3"]].drop_duplicates(),
                    on="Country_en",
                    how="left",
                )

        if merged is None or "matched_iso3" not in merged.columns or merged["matched_iso3"].isna().any():
            sample_missing = (
                merged.loc[merged["matched_iso3"].isna(), ["country_name"]].head(10)
                if merged is not None and "matched_iso3" in merged.columns
                else out[["country_name"]].head(10)
            )
            raise ValueError(
                "无法从 6.1.2.3 输出表补全全部国家 ISO3，示例缺失：\n"
                f"{sample_missing.to_string(index=False)}"
            )
        out["matched_iso3"] = merged["matched_iso3"].astype(str).str.strip().str.upper()

    out = out[out["matched_iso3"].str.len() == 3].copy()
    out = out.drop_duplicates(subset="matched_iso3", keep="first").reset_index(drop=True)
    out["country_id"] = np.arange(1, len(out) + 1, dtype=np.int32)

    if out.empty:
        raise ValueError("国家统计表中没有可用的 ISO3 国家记录。")
    return out


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


class Reservoir:
    __slots__ = ("max_size", "rng", "buf", "n_seen")

    def __init__(self, max_size: int, rng: np.random.Generator):
        self.max_size = int(max_size)
        self.rng = rng
        self.buf = np.empty((0,), dtype=np.float32)
        self.n_seen = 0

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float32).ravel()
        if values.size == 0:
            return

        if self.buf.size < self.max_size:
            take = min(self.max_size - self.buf.size, values.size)
            if take > 0:
                self.buf = np.concatenate([self.buf, values[:take]])
                self.n_seen += take
                values = values[take:]
                if values.size == 0:
                    return

        incoming = values.size
        seen_before = self.n_seen
        draw_upper = np.arange(seen_before + 1, seen_before + incoming + 1, dtype=np.int64)
        replace_idx = self.rng.integers(0, draw_upper)
        keep = replace_idx < self.max_size
        if np.any(keep):
            self.buf[replace_idx[keep]] = values[keep]
        self.n_seen += incoming

    def threshold_for_top_p(self, p: float) -> float:
        if self.buf.size == 0:
            return np.nan
        return float(np.percentile(self.buf, 100.0 - float(p)))


def build_country_mask_cache(
    template_tif: Path,
    gdf_admin0: gpd.GeoDataFrame,
    truth_df: pd.DataFrame,
    code_field: str,
) -> tuple[Path, Path, pd.DataFrame, int]:
    mask_tif = OUTPUT_DIR / "iso_id_mask.tif"
    lookup_csv = OUTPUT_DIR / "iso_id_lookup.csv"

    if mask_tif.exists() and lookup_csv.exists():
        lookup_df = pd.read_csv(lookup_csv)
        lookup_df["country_id"] = lookup_df["country_id"].astype(int)
        return mask_tif, lookup_csv, lookup_df, int(lookup_df["country_id"].max())

    iso3_to_country_id = {
        str(row.matched_iso3).strip().upper(): int(row.country_id)
        for row in truth_df.itertuples(index=False)
    }

    with rasterio.open(template_tif) as ds:
        meta = ds.meta.copy()
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
        raise ValueError("国家边界与统计表之间没有可栅格化的匹配国家。")

    country_mask = rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        fill=BACKGROUND_ID,
        all_touched=False,
        dtype="int32",
    )

    meta.update(dtype=rasterio.int32, count=1, nodata=BACKGROUND_ID, compress="lzw")
    with rasterio.open(mask_tif, "w", **meta) as dst:
        dst.write(country_mask, 1)

    lookup_df = truth_df[["country_id", "matched_iso3", "country_name", "role"]].copy()
    lookup_df = lookup_df.sort_values("country_id").reset_index(drop=True)
    lookup_df.to_csv(lookup_csv, index=False, encoding="utf-8-sig")
    return mask_tif, lookup_csv, lookup_df, int(lookup_df["country_id"].max())


def build_coarse_mean(pred_ds: rasterio.io.DatasetReader, factor: int) -> np.ndarray:
    height = max(1, pred_ds.height // factor)
    width = max(1, pred_ds.width // factor)
    return pred_ds.read(1, out_shape=(height, width), resampling=Resampling.average).astype(np.float32)


def coarse_back_window(
    coarse: np.ndarray,
    row_off: int,
    col_off: int,
    height: int,
    width: int,
    factor: int,
) -> np.ndarray:
    rows = (np.arange(row_off, row_off + height) // factor).astype(np.int64)
    cols = (np.arange(col_off, col_off + width) // factor).astype(np.int64)
    rows = np.clip(rows, 0, coarse.shape[0] - 1)
    cols = np.clip(cols, 0, coarse.shape[1] - 1)
    return coarse[np.ix_(rows, cols)]


def first_pass_thresholds(pred_tif: Path, iso_mask_tif: Path, max_id: int):
    rng = np.random.default_rng(RANDOM_SEED)
    res_pred = [Reservoir(SAMPLE_PER_COUNTRY, rng) for _ in range(max_id + 1)]
    res_coarse = {factor: [Reservoir(SAMPLE_PER_COUNTRY, rng) for _ in range(max_id + 1)] for factor in SCALE_FACTORS}

    with rasterio.open(pred_tif) as pred_ds, rasterio.open(iso_mask_tif) as mask_ds:
        nodata = get_nodata(pred_ds)
        coarse_maps = {factor: build_coarse_mean(pred_ds, factor) for factor in SCALE_FACTORS}

        if SAVE_COARSE_NPY:
            for factor, arr in coarse_maps.items():
                np.save(OUTPUT_DIR / f"coarse_mean_factor{factor}.npy", arr)

        for _, win in pred_ds.block_windows(1):
            pred = pred_ds.read(1, window=win).astype(np.float32)
            iso = mask_ds.read(1, window=win).astype(np.int32)
            mask = (iso > 0) & valid_mask(pred, nodata)
            if not np.any(mask):
                continue

            pred_values = pred[mask]
            iso_values = iso[mask]
            unique_ids = np.unique(iso_values)

            coarse_windows = {
                factor: coarse_back_window(coarse_maps[factor], win.row_off, win.col_off, win.height, win.width, factor)[mask]
                for factor in SCALE_FACTORS
            }

            for country_id in unique_ids:
                if not (0 < country_id <= max_id):
                    continue
                country_mask = iso_values == country_id
                res_pred[country_id].update(pred_values[country_mask])
                for factor in SCALE_FACTORS:
                    res_coarse[factor][country_id].update(coarse_windows[factor][country_mask])

    thr_pred = {top_p: np.full(max_id + 1, np.nan, dtype=np.float32) for top_p in TOP_P_LIST}
    thr_coarse = {
        factor: {top_p: np.full(max_id + 1, np.nan, dtype=np.float32) for top_p in TOP_P_LIST}
        for factor in SCALE_FACTORS
    }

    for top_p in TOP_P_LIST:
        for country_id in range(1, max_id + 1):
            thr_pred[top_p][country_id] = res_pred[country_id].threshold_for_top_p(top_p)
            for factor in SCALE_FACTORS:
                thr_coarse[factor][top_p][country_id] = res_coarse[factor][country_id].threshold_for_top_p(top_p)
    return thr_pred, thr_coarse


def second_pass_counts(pred_tif: Path, iso_mask_tif: Path, max_id: int, thr_pred, thr_coarse):
    counts = {
        factor: {
            top_p: {
                "inter": np.zeros(max_id + 1, dtype=np.int64),
                "union": np.zeros(max_id + 1, dtype=np.int64),
                "hot1": np.zeros(max_id + 1, dtype=np.int64),
                "hot2": np.zeros(max_id + 1, dtype=np.int64),
                "pix": np.zeros(max_id + 1, dtype=np.int64),
            }
            for top_p in TOP_P_LIST
        }
        for factor in SCALE_FACTORS
    }

    with rasterio.open(pred_tif) as pred_ds, rasterio.open(iso_mask_tif) as mask_ds:
        nodata = get_nodata(pred_ds)
        coarse_maps = {}
        for factor in SCALE_FACTORS:
            npy_path = OUTPUT_DIR / f"coarse_mean_factor{factor}.npy"
            coarse_maps[factor] = (
                np.load(npy_path).astype(np.float32)
                if npy_path.exists()
                else build_coarse_mean(pred_ds, factor)
            )

        for _, win in pred_ds.block_windows(1):
            pred = pred_ds.read(1, window=win).astype(np.float32)
            iso = mask_ds.read(1, window=win).astype(np.int32)
            valid = (iso > 0) & valid_mask(pred, nodata)
            if not np.any(valid):
                continue

            iso_values = iso[valid]
            unique_ids, inv = np.unique(iso_values, return_inverse=True)
            pix_add = np.bincount(inv, minlength=unique_ids.size)
            for idx, country_id in enumerate(unique_ids):
                if 0 < country_id <= max_id:
                    for factor in SCALE_FACTORS:
                        for top_p in TOP_P_LIST:
                            counts[factor][top_p]["pix"][country_id] += int(pix_add[idx])

            pred_values = pred[valid]
            for factor in SCALE_FACTORS:
                coarse_back = coarse_back_window(coarse_maps[factor], win.row_off, win.col_off, win.height, win.width, factor)
                coarse_values = coarse_back[valid]

                for top_p in TOP_P_LIST:
                    hot1 = pred_values >= thr_pred[top_p][iso_values]
                    hot2 = coarse_values >= thr_coarse[factor][top_p][iso_values]
                    inter = hot1 & hot2
                    union = hot1 | hot2

                    additions = {
                        "inter": np.bincount(inv, weights=inter.astype(np.int64), minlength=unique_ids.size),
                        "union": np.bincount(inv, weights=union.astype(np.int64), minlength=unique_ids.size),
                        "hot1": np.bincount(inv, weights=hot1.astype(np.int64), minlength=unique_ids.size),
                        "hot2": np.bincount(inv, weights=hot2.astype(np.int64), minlength=unique_ids.size),
                    }
                    for idx, country_id in enumerate(unique_ids):
                        if 0 < country_id <= max_id:
                            for key, values in additions.items():
                                counts[factor][top_p][key][country_id] += int(values[idx])
    return counts


def save_metrics(lookup_df: pd.DataFrame, max_id: int, counts) -> pd.DataFrame:
    summary_rows: list[dict[str, float | int]] = []
    id_to_meta = {
        int(row.country_id): {
            "matched_iso3": str(row.matched_iso3),
            "country_name": str(row.country_name),
            "role": str(row.role),
        }
        for row in lookup_df.itertuples(index=False)
    }

    for factor in SCALE_FACTORS:
        for top_p in TOP_P_LIST:
            rows = []
            data = counts[factor][top_p]
            for country_id in range(1, max_id + 1):
                pix = int(data["pix"][country_id])
                meta = id_to_meta.get(country_id)
                if pix <= 0 or meta is None:
                    continue

                inter = int(data["inter"][country_id])
                union = int(data["union"][country_id])
                hot1 = int(data["hot1"][country_id])
                hot2 = int(data["hot2"][country_id])
                rows.append(
                    {
                        "matched_iso3": meta["matched_iso3"],
                        "country_id": country_id,
                        "country_name": meta["country_name"],
                        "role": meta["role"],
                        "scale_factor": factor,
                        "top_p_percent": top_p,
                        "jaccard": float(inter / (union + 1e-12)),
                        "overlap_inter_over_hot1": float(inter / (hot1 + 1e-12)),
                        "pix_valid": pix,
                        "hot_native_pix": hot1,
                        "hot_coarse_pix": hot2,
                        "inter_pix": inter,
                        "union_pix": union,
                    }
                )

            df = pd.DataFrame(rows)
            df.to_csv(
                OUTPUT_DIR / f"within_country_scale_factor{factor}_top{top_p}.csv",
                index=False,
                encoding="utf-8-sig",
            )
            if not df.empty:
                summary_rows.append(
                    {
                        "scale_factor": factor,
                        "top_p_percent": top_p,
                        "countries": int(df["matched_iso3"].nunique()),
                        "jaccard_mean": float(df["jaccard"].mean()),
                        "jaccard_median": float(df["jaccard"].median()),
                        "overlap_mean": float(df["overlap_inter_over_hot1"].mean()),
                        "overlap_median": float(df["overlap_inter_over_hot1"].median()),
                    }
                )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / "within_country_scale_global_summary.csv", index=False, encoding="utf-8-sig")
    return summary_df


def find_summary_row(summary_df: pd.DataFrame, scale_factor: int, top_p_percent: int) -> pd.Series:
    matched = summary_df[
        (summary_df["scale_factor"] == scale_factor)
        & (summary_df["top_p_percent"] == top_p_percent)
    ]
    if matched.empty:
        raise ValueError(f"未找到 scale_factor={scale_factor}, top_p_percent={top_p_percent} 的汇总结果。")
    return matched.iloc[0]


def build_core_metric_checks(summary_df: pd.DataFrame, countries_with_pixels: int) -> list[CheckResult]:
    row_10_5 = find_summary_row(summary_df, 10, 5)
    row_10_10 = find_summary_row(summary_df, 10, 10)
    row_10_20 = find_summary_row(summary_df, 10, 20)
    row_100_10 = find_summary_row(summary_df, 100, 10)

    return [
        CheckResult(
            "有效国家数量充足",
            countries_with_pixels >= MIN_COUNTRIES_WITH_PIXELS,
            f"有效国家数={countries_with_pixels}，阈值={MIN_COUNTRIES_WITH_PIXELS}",
        ),
        CheckResult(
            f"核心尺度 factor={CORE_SCALE_FACTOR}, top{CORE_TOP_P}% 的 Jaccard 达标",
            float(row_10_10["jaccard_mean"]) >= MIN_CORE_JACCARD,
            f"jaccard_mean={float(row_10_10['jaccard_mean']):.4f}，阈值={MIN_CORE_JACCARD:.2f}",
        ),
        CheckResult(
            f"核心尺度 factor={CORE_SCALE_FACTOR}, top{CORE_TOP_P}% 的 Overlap 达标",
            float(row_10_10["overlap_mean"]) >= MIN_CORE_OVERLAP,
            f"overlap_mean={float(row_10_10['overlap_mean']):.4f}，阈值={MIN_CORE_OVERLAP:.2f}",
        ),
        CheckResult(
            "尺度变粗后重合度下降规律合理",
            float(row_10_10["jaccard_mean"]) > float(row_100_10["jaccard_mean"])
            and float(row_10_10["overlap_mean"]) > float(row_100_10["overlap_mean"]),
            "factor10/top10 对 factor100/top10: "
            f"jaccard={float(row_10_10['jaccard_mean']):.4f}>{float(row_100_10['jaccard_mean']):.4f}, "
            f"overlap={float(row_10_10['overlap_mean']):.4f}>{float(row_100_10['overlap_mean']):.4f}",
        ),
        CheckResult(
            "热点比例扩大后稳定性提升趋势合理",
            float(row_10_20["jaccard_mean"]) > float(row_10_10["jaccard_mean"]) > float(row_10_5["jaccard_mean"])
            and float(row_10_20["overlap_mean"]) > float(row_10_10["overlap_mean"]) > float(row_10_5["overlap_mean"]),
            "factor10 下 top5/top10/top20: "
            f"jaccard={float(row_10_5['jaccard_mean']):.4f}/{float(row_10_10['jaccard_mean']):.4f}/{float(row_10_20['jaccard_mean']):.4f}, "
            f"overlap={float(row_10_5['overlap_mean']):.4f}/{float(row_10_10['overlap_mean']):.4f}/{float(row_10_20['overlap_mean']):.4f}",
        ),
    ]


def build_summary_text(
    pred_tif: Path,
    boundary_file: Path,
    truth_file: Path,
    iso_mask_tif: Path,
    lookup_csv: Path,
    summary_df: pd.DataFrame,
    results: list[CheckResult],
) -> str:
    lines = [
        "6.1.2.1 国家内部可用性验证——跨尺度稳定性测试",
        "",
        f"主结果栅格：{pred_tif}",
        f"边界文件：{boundary_file}",
        f"统计文件：{truth_file}",
        f"国家掩膜：{iso_mask_tif}",
        f"国家映射表：{lookup_csv}",
        "",
        f"尺度因子：{SCALE_FACTORS}",
        f"热点比例：{TOP_P_LIST}",
        "",
        "核心判定：",
    ]
    for item in results:
        lines.append(f"[{'通过' if item.passed else '未通过'}] {item.name}: {item.details}")

    lines.append("")
    lines.append("全局摘要：")
    if summary_df.empty:
        lines.append("无可用结果。")
    else:
        for row in summary_df.itertuples(index=False):
            lines.append(
                f"factor={int(row.scale_factor)}, top_p={int(row.top_p_percent)}%: "
                f"countries={int(row.countries)}, "
                f"jaccard_mean={float(row.jaccard_mean):.4f}, "
                f"jaccard_median={float(row.jaccard_median):.4f}, "
                f"overlap_mean={float(row.overlap_mean):.4f}, "
                f"overlap_median={float(row.overlap_median):.4f}"
            )

    lines.extend(
        [
            "",
            f"输出目录：{OUTPUT_DIR}",
            f"摘要表：{OUTPUT_DIR / 'within_country_scale_global_summary.csv'}",
        ]
    )
    return "\n".join(lines)


def print_results(results: list[CheckResult]) -> None:
    for item in results:
        print(f"[{'通过' if item.passed else '未通过'}] {item.name}: {item.details}")


def main() -> int:
    ensure_dir(OUTPUT_DIR)

    boundary_files = find_boundary_files(BOUNDARY_DIR)
    truth_file = find_truth_file(TRUTH_DIR)
    input_ready = PRED_TIF.is_file() and bool(boundary_files) and truth_file is not None

    results: list[CheckResult] = [
        CheckResult(
            "输入数据齐备",
            input_ready,
            f"pred={PRED_TIF.is_file()}，boundary_candidates={len(boundary_files)}，truth={truth_file is not None}",
        )
    ]

    if not input_ready:
        print_results(results)
        return 1

    gdf_admin0, boundary_file, code_field = load_matching_boundary(boundary_files)
    truth_df = load_truth_table(truth_file)
    iso_mask_tif, lookup_csv, lookup_df, max_id = build_country_mask_cache(PRED_TIF, gdf_admin0, truth_df, code_field)

    results.append(
        CheckResult(
            "国家边界与掩膜构建成功",
            iso_mask_tif.is_file(),
            f"边界文件={boundary_file}；编码字段={code_field}；国家数={len(lookup_df)}；掩膜={iso_mask_tif}",
        )
    )

    print("========== 6.1.2.1 国家内部可用性验证——跨尺度稳定性测试 ==========")
    print(f"Pred: {PRED_TIF}")
    print(f"Boundary: {boundary_file}")
    print(f"Truth: {truth_file}")
    print(f"ISO mask: {iso_mask_tif}")
    print("[1/3] 估计各国家 top-p 热点阈值...")
    thr_pred, thr_coarse = first_pass_thresholds(PRED_TIF, iso_mask_tif, max_id)
    print("[2/3] 统计跨尺度热点交并关系...")
    counts = second_pass_counts(PRED_TIF, iso_mask_tif, max_id, thr_pred, thr_coarse)
    print("[3/3] 保存结果...")
    summary_df = save_metrics(lookup_df, max_id, counts)

    countries_with_pixels = int(summary_df["countries"].max()) if not summary_df.empty else 0
    results.extend(build_core_metric_checks(summary_df, countries_with_pixels))

    summary_txt = OUTPUT_DIR / "within_country_scale_summary.txt"
    summary_txt.write_text(
        build_summary_text(PRED_TIF, boundary_file, truth_file, iso_mask_tif, lookup_csv, summary_df, results),
        encoding="utf-8",
    )

    passed_count = sum(1 for item in results if item.passed)
    print()
    print_results(results)
    print()
    print(f"结果：{passed_count}/{len(results)} 项通过")
    print(f"Outputs: {OUTPUT_DIR}")
    return 0 if passed_count == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
