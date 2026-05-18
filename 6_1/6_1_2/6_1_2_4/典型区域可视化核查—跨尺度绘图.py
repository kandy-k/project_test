from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.features import geometry_window
from rasterio.windows import transform as window_transform
from shapely.geometry import mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"

PRED_TIF = REPO_ROOT / "dataset" / "products" / "demand" / "pred_final_pred_masked.tif"
CROSS_SCALE_TABLE = (
    REPO_ROOT / "6_1" / "6_1_2" / "6_1_2_1" / "outputs" / "within_country_scale_factor10_top10.csv"
)
BOUNDARY_DIR = REPO_ROOT / "dataset" / "validation" / "boundaries"

BOUNDARY_SUFFIXES = {".gpkg", ".shp", ".geojson"}
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

TOPK_PER_GROUP = 6
MIN_PIX_VALID = 200_000
BOUNDARY_BUFFER_DEG = 0.5
SCALE_FACTOR = 10
TOP_P_PERCENT = 10
NODATA_FALLBACK = -9999.0
VMIN_Q = 5
VMAX_Q = 95

RAW_CMAP = "viridis"
COARSE_CMAP = "magma"
HOTSPOT_CMAP = ListedColormap(
    [
        (0, 0, 0, 0),
        (0.93, 0.32, 0.24, 0.82),
        (0.16, 0.43, 0.85, 0.82),
        (0.58, 0.15, 0.68, 0.90),
    ]
)


@dataclass
class CheckResult:
    name: str
    passed: bool
    details: str


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def compute_hotspots(
    pred_win: np.ndarray,
    coarse_win: np.ndarray,
    country_mask: np.ndarray,
    nodata: float | None,
    top_p_percent: int,
):
    valid = country_mask & valid_mask(pred_win, nodata) & np.isfinite(coarse_win)
    pix_valid = int(valid.sum())
    if pix_valid < 100:
        return None

    pred_values = pred_win[valid]
    coarse_values = coarse_win[valid]
    pred_thr = np.percentile(pred_values, 100 - top_p_percent)
    coarse_thr = np.percentile(coarse_values, 100 - top_p_percent)

    hot_pred = np.zeros_like(pred_win, dtype=bool)
    hot_coarse = np.zeros_like(pred_win, dtype=bool)
    hot_pred[valid] = pred_values >= pred_thr
    hot_coarse[valid] = coarse_values >= coarse_thr

    inter = int((hot_pred & hot_coarse).sum())
    union = int((hot_pred | hot_coarse).sum())
    hot_pred_count = int(hot_pred.sum())
    hot_coarse_count = int(hot_coarse.sum())
    jaccard = float(inter / (union + 1e-12))
    overlap = float(inter / (hot_pred_count + 1e-12))
    return {
        "hot_pred": hot_pred,
        "hot_coarse": hot_coarse,
        "pix_valid": pix_valid,
        "hot_pred_count": hot_pred_count,
        "hot_coarse_count": hot_coarse_count,
        "inter": inter,
        "union": union,
        "jaccard": jaccard,
        "overlap": overlap,
    }


def data_limits(plot_arr: np.ndarray) -> tuple[float | None, float | None]:
    values = plot_arr[np.isfinite(plot_arr)]
    if values.size == 0:
        return None, None
    vmin = float(np.percentile(values, VMIN_Q))
    vmax = float(np.percentile(values, VMAX_Q))
    if np.isclose(vmin, vmax):
        vmin -= 1e-6
        vmax += 1e-6
    return vmin, vmax


def prepare_class_array(hot_pred: np.ndarray, hot_coarse: np.ndarray) -> np.ndarray:
    cls = np.zeros_like(hot_pred, dtype=np.uint8)
    cls[hot_pred & ~hot_coarse] = 1
    cls[~hot_pred & hot_coarse] = 2
    cls[hot_pred & hot_coarse] = 3
    return cls


def plot_country_figure(
    sub_gdf: gpd.GeoDataFrame,
    iso3: str,
    group_label: str,
    pred_win: np.ndarray,
    coarse_win: np.ndarray,
    out_transform,
    country_mask: np.ndarray,
    hotspot_result: dict[str, float | int | np.ndarray],
    output_path: Path,
    nodata: float | None,
) -> None:
    group_tag = "High-stability" if "高" in group_label else "Low-stability"
    height, width = pred_win.shape
    xmin = out_transform.c
    ymax = out_transform.f
    xmax = xmin + out_transform.a * width
    ymin = ymax + out_transform.e * height
    extent = [xmin, xmax, ymin, ymax]

    pred_plot = pred_win.astype(np.float32).copy()
    coarse_plot = coarse_win.astype(np.float32).copy()
    pred_plot[~country_mask] = np.nan
    coarse_plot[~country_mask] = np.nan
    if nodata is not None:
        pred_plot[pred_plot == nodata] = np.nan

    pred_vmin, pred_vmax = data_limits(pred_plot)
    coarse_vmin, coarse_vmax = data_limits(coarse_plot)
    cls = prepare_class_array(hotspot_result["hot_pred"], hotspot_result["hot_coarse"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), dpi=220)

    ax = axes[0]
    im0 = ax.imshow(pred_plot, extent=extent, cmap=RAW_CMAP, vmin=pred_vmin, vmax=pred_vmax, interpolation="nearest")
    sub_gdf.boundary.plot(ax=ax, linewidth=1.0, color="black")
    ax.set_title(f"{iso3} | Native demand")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    im1 = ax.imshow(coarse_plot, extent=extent, cmap=COARSE_CMAP, vmin=coarse_vmin, vmax=coarse_vmax, interpolation="nearest")
    sub_gdf.boundary.plot(ax=ax, linewidth=1.0, color="black")
    ax.set_title(f"{iso3} | {SCALE_FACTOR}x coarse-back projection")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[2]
    ax.imshow(pred_plot, extent=extent, cmap=RAW_CMAP, vmin=pred_vmin, vmax=pred_vmax, interpolation="nearest")
    ax.imshow(cls, extent=extent, cmap=HOTSPOT_CMAP, interpolation="nearest")
    sub_gdf.boundary.plot(ax=ax, linewidth=1.0, color="black")
    ax.set_title(
        f"{iso3} | Hotspot overlap\n"
        f"J={hotspot_result['jaccard']:.3f}, O={hotspot_result['overlap']:.3f}, "
        f"pix={int(hotspot_result['pix_valid']):,}"
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor=(0.93, 0.32, 0.24, 0.82), edgecolor="none", label="Native hotspot"),
            Patch(facecolor=(0.16, 0.43, 0.85, 0.82), edgecolor="none", label="Coarse hotspot"),
            Patch(facecolor=(0.58, 0.15, 0.68, 0.90), edgecolor="none", label="Shared hotspot"),
        ],
        loc="lower left",
        framealpha=0.92,
    )

    fig.suptitle(
        f"6.1.2.5 Typical-region visual check | {group_tag} | top{TOP_P_PERCENT}% hotspots | factor={SCALE_FACTOR}",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def load_cross_scale_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"matched_iso3", "jaccard", "overlap_inter_over_hot1", "pix_valid"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"跨尺度结果表缺少字段：{sorted(missing)}")
    return df.copy()


def pick_countries(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    filtered = df[df["pix_valid"] >= MIN_PIX_VALID].copy()
    if filtered.empty:
        raise ValueError(f"pix_valid >= {MIN_PIX_VALID} 后没有国家可选，请调整阈值。")

    ascending = mode == "low"
    picked = filtered.sort_values(
        by=["jaccard", "overlap_inter_over_hot1", "pix_valid"],
        ascending=[ascending, ascending, False],
    ).head(TOPK_PER_GROUP)
    return picked.reset_index(drop=True)


def build_summary_text(
    boundary_file: Path,
    selected_high: pd.DataFrame,
    selected_low: pd.DataFrame,
    generated_rows: list[dict[str, object]],
) -> str:
    lines = [
        "6.1.2.5 典型区域可视化核查——跨尺度绘图",
        "",
        f"主结果栅格：{PRED_TIF}",
        f"跨尺度结果表：{CROSS_SCALE_TABLE}",
        f"边界文件：{boundary_file}",
        f"尺度因子：{SCALE_FACTOR}",
        f"热点比例：top{TOP_P_PERCENT}%",
        f"高稳定样区数量：{len(selected_high)}",
        f"低稳定样区数量：{len(selected_low)}",
        "",
        "高稳定样区：",
    ]
    for row in selected_high.itertuples(index=False):
        lines.append(
            f"{row.matched_iso3}: jaccard={float(row.jaccard):.4f}, "
            f"overlap={float(row.overlap_inter_over_hot1):.4f}, pix_valid={int(row.pix_valid)}"
        )

    lines.append("")
    lines.append("低稳定样区：")
    for row in selected_low.itertuples(index=False):
        lines.append(
            f"{row.matched_iso3}: jaccard={float(row.jaccard):.4f}, "
            f"overlap={float(row.overlap_inter_over_hot1):.4f}, pix_valid={int(row.pix_valid)}"
        )

    lines.append("")
    lines.append(f"已生成图件数量：{len(generated_rows)}")
    lines.append(f"输出目录：{OUTPUT_DIR}")
    return "\n".join(lines)


def print_results(results: list[CheckResult]) -> None:
    for item in results:
        print(f"[{'通过' if item.passed else '未通过'}] {item.name}: {item.details}")


def main() -> int:
    ensure_dir(OUTPUT_DIR)
    high_dir = OUTPUT_DIR / "high"
    low_dir = OUTPUT_DIR / "low"
    ensure_dir(high_dir)
    ensure_dir(low_dir)

    boundary_files = find_boundary_files(BOUNDARY_DIR)
    inputs_ready = PRED_TIF.is_file() and CROSS_SCALE_TABLE.is_file() and bool(boundary_files)
    results = [
        CheckResult(
            "输入数据齐备",
            inputs_ready,
            f"pred={PRED_TIF.is_file()}，cross_scale_table={CROSS_SCALE_TABLE.is_file()}，boundary_candidates={len(boundary_files)}",
        )
    ]
    if not inputs_ready:
        print_results(results)
        return 1

    gdf_admin0, boundary_file, code_field = load_matching_boundary(boundary_files)
    df_metrics = load_cross_scale_table(CROSS_SCALE_TABLE)
    high_df = pick_countries(df_metrics, "high")
    low_df = pick_countries(df_metrics, "low")

    results.append(
        CheckResult(
            "典型区域选择成功",
            len(high_df) == TOPK_PER_GROUP and len(low_df) == TOPK_PER_GROUP,
            f"high={len(high_df)}，low={len(low_df)}，目标每组={TOPK_PER_GROUP}",
        )
    )

    high_df.to_csv(high_dir / "picked_countries.csv", index=False, encoding="utf-8-sig")
    low_df.to_csv(low_dir / "picked_countries.csv", index=False, encoding="utf-8-sig")

    print("========== 6.1.2.5 典型区域可视化核查——跨尺度绘图 ==========")
    print(f"Pred: {PRED_TIF}")
    print(f"Cross-scale table: {CROSS_SCALE_TABLE}")
    print(f"Boundary: {boundary_file}")
    print("[1/2] 计算粗尺度回投并逐国绘图...")

    generated_rows: list[dict[str, object]] = []

    with rasterio.open(PRED_TIF) as ds:
        nodata = get_nodata(ds)
        gdf_use = gdf_admin0.to_crs(ds.crs)
        coarse = build_coarse_mean(ds, SCALE_FACTOR)

        for group_label, picked_df, out_dir in (
            ("高稳定样区", high_df, high_dir),
            ("低稳定样区", low_df, low_dir),
        ):
            for rank, row in enumerate(picked_df.itertuples(index=False), start=1):
                iso3 = str(row.matched_iso3).strip().upper()
                sub = gdf_use[gdf_use[code_field].astype(str).str.strip().str.upper() == iso3]
                if sub.empty:
                    continue

                geom = sub.geometry.union_all()
                geom_crop = geom.buffer(BOUNDARY_BUFFER_DEG) if BOUNDARY_BUFFER_DEG > 0 else geom

                try:
                    win = geometry_window(ds, [mapping(geom_crop)], pad_x=0, pad_y=0, north_up=True)
                except Exception:
                    continue

                pred_win = ds.read(1, window=win).astype(np.float32)
                coarse_win = coarse_back_window(coarse, win.row_off, win.col_off, win.height, win.width, SCALE_FACTOR)
                out_transform = window_transform(win, ds.transform)
                country_mask = geometry_mask(
                    [mapping(geom)],
                    out_shape=(win.height, win.width),
                    transform=out_transform,
                    invert=True,
                    all_touched=False,
                )

                hotspot_result = compute_hotspots(pred_win, coarse_win, country_mask, nodata, TOP_P_PERCENT)
                if hotspot_result is None:
                    continue

                output_name = (
                    f"{rank:02d}_{group_label}_{iso3}_J{hotspot_result['jaccard']:.3f}"
                    f"_O{hotspot_result['overlap']:.3f}.png"
                )
                output_path = out_dir / output_name
                plot_country_figure(
                    sub_gdf=sub,
                    iso3=iso3,
                    group_label=group_label,
                    pred_win=pred_win,
                    coarse_win=coarse_win,
                    out_transform=out_transform,
                    country_mask=country_mask,
                    hotspot_result=hotspot_result,
                    output_path=output_path,
                    nodata=nodata,
                )

                generated_rows.append(
                    {
                        "group_label": group_label,
                        "rank": rank,
                        "matched_iso3": iso3,
                        "jaccard": float(hotspot_result["jaccard"]),
                        "overlap": float(hotspot_result["overlap"]),
                        "pix_valid": int(hotspot_result["pix_valid"]),
                        "output_png": str(output_path),
                    }
                )

    print("[2/2] 保存图件清单与摘要...")
    generated_df = pd.DataFrame(generated_rows)
    generated_df.to_csv(OUTPUT_DIR / "typical_region_figure_manifest.csv", index=False, encoding="utf-8-sig")

    summary_txt = OUTPUT_DIR / "typical_region_visual_check_summary.txt"
    summary_txt.write_text(
        build_summary_text(boundary_file, high_df, low_df, generated_rows),
        encoding="utf-8",
    )

    expected_min = min(TOPK_PER_GROUP, len(high_df)) + min(TOPK_PER_GROUP, len(low_df))
    results.append(
        CheckResult(
            "典型区域图件生成成功",
            len(generated_rows) >= expected_min,
            f"generated={len(generated_rows)}，expected={expected_min}",
        )
    )
    results.append(
        CheckResult(
            "图件清单与摘要已生成",
            (OUTPUT_DIR / "typical_region_figure_manifest.csv").is_file() and summary_txt.is_file(),
            f"manifest={OUTPUT_DIR / 'typical_region_figure_manifest.csv'}；summary={summary_txt}",
        )
    )

    print()
    print_results(results)
    print()
    print(f"结果：{sum(item.passed for item in results)}/{len(results)} 项通过")
    print(f"Outputs: {OUTPUT_DIR}")
    return 0 if all(item.passed for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
