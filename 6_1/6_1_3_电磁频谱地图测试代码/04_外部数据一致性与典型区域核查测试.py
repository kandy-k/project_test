# -*- coding: utf-8 -*-
"""
6.1.3.4 外部数据一致性与典型区域核查测试

本脚本固定按 6 arcmin / 360 arcsec 版本开展外部一致性验证。

测试内容：
1. 国家尺度：
   将 360arcsec RSS 电磁频谱地图按国家聚合，并与 ITU 国家级移动宽带互联网
   流量数据计算 Spearman 秩相关。

2. 微观尺度：
   将 Ookla Speedtest 移动网络活跃数据栅格化到当前 360arcsec RSS 模板，
   计算 Top-p% 高频谱热点对 Ookla 活跃网格的覆盖率。

说明：
本脚本会在测试结果目录中生成与当前 RSS 栅格严格共格的国家掩膜和 Ookla 缓存，
避免误用其他分辨率版本的验证栅格。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio import features

from common import (
    NATURE_PALETTE,
    PATHS,
    add_note_box,
    get_test_out_dir,
    grid_info,
    metric_row,
    pearson_corr,
    plot_quality_dashboard,
    print_saved,
    read_band,
    set_academic_style,
    status_from_checks,
    spearman_corr,
    style_academic_axes,
    write_csv,
    write_json,
    write_metric_outputs,
    write_verdict_report,
)

TEST_OUT_DIR = get_test_out_dir("04_外部数据一致性与典型区域核查测试")


def choose_360_rss_result():
    """返回 360arcsec RSS 最终主结果文件，不回退到旧版或 60arcsec。"""

    rss_path = PATHS["rss_360_v2"]
    if not rss_path.exists() or rss_path.stat().st_size <= 0:
        raise FileNotFoundError(f"未找到 360arcsec RSS 最终主结果文件: {rss_path}")
    return rss_path


def make_country_mask_like(template_path):
    """把国家边界矢量栅格化到当前 RSS 模板。

    输出：
    - iso_id_mask: 与 RSS 完全共格的国家 ID 栅格；
    - lookup_csv: iso_id 与 ISO_A3 的对应关系。
    """

    mask_tif = TEST_OUT_DIR / f"04_iso_mask_like_{template_path.stem}.tif"
    lookup_csv = TEST_OUT_DIR / f"04_iso_lookup_like_{template_path.stem}.csv"

    if mask_tif.exists() and lookup_csv.exists():
        return mask_tif, lookup_csv

    import geopandas as gpd

    gdf = gpd.read_file(PATHS["country_shp"])
    iso_col = "ISO_A3"
    if iso_col not in gdf.columns:
        raise ValueError("全球行政边界.shp 中未找到 ISO_A3 字段。")

    gdf = gdf.dropna(subset=[iso_col]).copy()
    gdf = gdf[gdf[iso_col].astype(str).str.len() == 3].copy()
    gdf["iso_id"] = np.arange(1, len(gdf) + 1)

    with rasterio.open(template_path) as src:
        meta = src.meta.copy()
        out_shape = src.shape
        transform = src.transform

    shapes = ((geom, int(value)) for geom, value in zip(gdf.geometry, gdf["iso_id"]))
    iso_arr = features.rasterize(
        shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype="int32",
    )

    meta.update(dtype=rasterio.int32, count=1, nodata=0, compress="lzw")
    with rasterio.open(mask_tif, "w", **meta) as dst:
        dst.write(iso_arr, 1)

    lookup = gdf[["iso_id", iso_col]].rename(columns={iso_col: "entityIso"})
    lookup.to_csv(lookup_csv, index=False, encoding="utf-8-sig")

    return mask_tif, lookup_csv


def make_ookla_raster_like(template_path):
    """把 Ookla Speedtest 矢量数据栅格化到当前 RSS 模板。

    如果原项目中已有 Ookla 栅格，但分辨率不同，也不直接复用，避免 60/360 混用。
    """

    ookla_tif = TEST_OUT_DIR / f"04_ookla_devices_like_{template_path.stem}.tif"
    if ookla_tif.exists() and ookla_tif.stat().st_size > 0:
        return ookla_tif, None

    ookla_shp = PATHS["ookla_raster"].parent.parent / "1_原始数据" / "2024-01-01_performance_mobile_tiles" / "gps_mobile_tiles.shp"
    if not ookla_shp.exists():
        return None, f"未找到 Ookla Shapefile：{ookla_shp}，已跳过微观热点覆盖率计算。"

    import geopandas as gpd

    gdf = gpd.read_file(ookla_shp)
    if "devices" not in gdf.columns:
        return None, "Ookla Shapefile 中未找到 devices 字段，已跳过微观热点覆盖率计算。"
    gdf = gdf.dropna(subset=["devices"])

    with rasterio.open(template_path) as src:
        meta = src.meta.copy()
        out_shape = src.shape
        transform = src.transform
        crs = src.crs

    if gdf.crs is not None and crs is not None and gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    shapes = ((geom, float(value)) for geom, value in zip(gdf.geometry, gdf["devices"]))
    ookla_arr = features.rasterize(
        shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype="float32",
    )

    meta.update(dtype=rasterio.float32, count=1, nodata=0, compress="lzw")
    with rasterio.open(ookla_tif, "w", **meta) as dst:
        dst.write(ookla_arr, 1)

    return ookla_tif, None


def aggregate_rss_by_country(rss, iso):
    """将 RSS dBm 转为线性 Watts 后按国家求和。"""

    valid = (iso > 0) & np.isfinite(rss)
    watts = np.power(10.0, (rss[valid] - 30.0) / 10.0)
    iso_valid = iso[valid].astype(np.int64)

    sum_watts = np.bincount(iso_valid, weights=watts)
    pixel_counts = np.bincount(iso_valid)

    rows = []
    for iso_id in np.where(pixel_counts > 0)[0]:
        if iso_id == 0:
            continue
        rows.append(
            {
                "iso_id": int(iso_id),
                "pred_total_watts": float(sum_watts[iso_id]),
                "valid_pixels": int(pixel_counts[iso_id]),
            }
        )
    return pd.DataFrame(rows)


def choose_latest_itu_rows(df):
    """每个国家保留最新年份的 ITU 流量数据。"""

    df = df.dropna(subset=["entityIso", "dataValue"]).copy()
    df["entityIso"] = df["entityIso"].astype(str).str.strip().str.upper()
    if "dataYear" in df.columns:
        df = df.sort_values("dataYear", ascending=False).drop_duplicates("entityIso")
    return df


def calculate_ookla_coverage(rss, iso, ookla):
    """计算全球 Top-p% 高频谱热点对 Ookla 活跃网格的覆盖率。"""

    valid = (iso > 0) & np.isfinite(rss) & np.isfinite(ookla)
    rows = []
    for top_p in [1, 5, 10, 20]:
        rss_valid = rss[valid]
        if rss_valid.size == 0:
            continue

        threshold = np.percentile(rss_valid, 100 - top_p)
        rss_hotspot = valid & (rss >= threshold)
        ookla_active = valid & (ookla >= 1)

        covered = np.logical_and(rss_hotspot, ookla_active).sum()
        active_total = ookla_active.sum()

        rows.append(
            {
                "region": "GLOBAL",
                "top_p_percent": top_p,
                "ookla_active_pixels": int(active_total),
                "covered_active_pixels": int(covered),
                "coverage_rate": float(covered / active_total) if active_total > 0 else None,
            }
        )
    return rows


def plot_country_scatter(df_final, output_path: Path, spearman_result=None):
    """绘制国家尺度 ITU 真值与预测总辐射功率散点图。"""
    set_academic_style(plt)
    if df_final.empty:
        return

    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=300)
    ax.scatter(
        df_final["log_itu_dataValue"],
        df_final["log_pred_watts"],
        s=42,
        alpha=0.7,
        color=NATURE_PALETTE["blue"],
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )

    x = df_final["log_itu_dataValue"].to_numpy()
    y = df_final["log_pred_watts"].to_numpy()
    if len(x) >= 2:
        coef = np.polyfit(x, y, 1)
        x_line = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        y_line = coef[0] * x_line + coef[1]
        ax.plot(x_line, y_line, color=NATURE_PALETTE["orange"], linewidth=2.0, linestyle="--", zorder=4)

    ax.set_xlabel(r"Log$_{10}$(ITU mobile-broadband traffic)")
    ax.set_ylabel(r"Log$_{10}$(Predicted total RSS power / Watts)")
    ax.set_title("Macro-level Consistency: RSS Map vs. ITU Traffic", pad=10)
    style_academic_axes(ax, grid_axis="both")
    if spearman_result is not None:
        add_note_box(
            ax,
            f"Spearman rho = {spearman_result['rho']:.3f}\n"
            f"p-value = {spearman_result['p_value']:.2e}\n"
            f"N = {spearman_result['n']} countries",
            loc=(0.05, 0.95),
            fontsize=10,
        )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_ookla_coverage(rows, output_path: Path):
    """绘制 Top-p% 高频谱热点对 Ookla 活跃网格的覆盖率。"""
    set_academic_style(plt)
    if not rows:
        return

    labels = [f"Top {r['top_p_percent']}%" for r in rows]
    values = [0 if r["coverage_rate"] is None else r["coverage_rate"] * 100 for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=300)
    bars = ax.bar(labels, values, color=[NATURE_PALETTE["gray"], NATURE_PALETTE["blue"], NATURE_PALETTE["orange"], NATURE_PALETTE["green"]], edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_ylabel("Ookla active area coverage (%)")
    ax.set_ylim(0, max(5, max(values) * 1.15))
    ax.set_xlabel("Top-p% RSS hotspots")
    ax.set_title("Micro-level Consistency: RSS Hotspots vs. Ookla Active Areas", pad=10)
    style_academic_axes(ax, grid_axis="y")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.1f}%", ha="center", va="bottom", fontsize=9)
    add_note_box(ax, "Higher bars mean RSS hotspots cover more measured\nmobile-network active pixels from Ookla.", loc=(0.03, 0.95), fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    rss_path = choose_360_rss_result()
    rss, _, _, _, _ = read_band(rss_path)

    iso_mask_path, lookup_path = make_country_mask_like(rss_path)
    iso, _, _, _, _ = read_band(iso_mask_path)

    lookup = pd.read_csv(lookup_path)
    lookup["entityIso"] = lookup["entityIso"].astype(str).str.strip().str.upper()

    df_pred = aggregate_rss_by_country(rss, iso)
    df_pred = df_pred.merge(lookup, on="iso_id", how="left")

    df_itu = choose_latest_itu_rows(pd.read_csv(PATHS["itu"]))
    df_final = df_pred.merge(df_itu[["entityIso", "dataValue"]], on="entityIso", how="inner")
    df_final = df_final[(df_final["pred_total_watts"] > 0) & (df_final["dataValue"] > 0)].copy()
    df_final["log_pred_watts"] = np.log10(df_final["pred_total_watts"])
    df_final["log_itu_dataValue"] = np.log10(df_final["dataValue"])

    spearman_raw = spearman_corr(df_final["pred_total_watts"].values, df_final["dataValue"].values)
    spearman_log = spearman_corr(df_final["log_pred_watts"].values, df_final["log_itu_dataValue"].values)
    pearson_log = pearson_corr(df_final["log_pred_watts"].values, df_final["log_itu_dataValue"].values)

    ookla_path, ookla_note = make_ookla_raster_like(rss_path)
    coverage_rows = []
    if ookla_path is not None:
        ookla, _, _, _, _ = read_band(ookla_path)
        coverage_rows = calculate_ookla_coverage(rss, iso, ookla)

    country_scatter_path = TEST_OUT_DIR / "04_国家尺度ITU一致性散点图.png"
    coverage_plot_path = TEST_OUT_DIR / "04_Ookla热点覆盖率柱状图.png"
    plot_country_scatter(df_final, country_scatter_path, spearman_raw)
    plot_ookla_coverage(coverage_rows, coverage_plot_path)

    macro_rho = None if spearman_raw is None else spearman_raw["rho"]
    log_rho = None if spearman_log is None else spearman_log["rho"]
    log_pearson = None if pearson_log is None else pearson_log["r"]
    top10_row = next((r for r in coverage_rows if r["top_p_percent"] == 10), None)
    top10_coverage = None if top10_row is None else top10_row["coverage_rate"]
    verdict = status_from_checks(
        fail_checks=[
            len(df_final) < 20,
            macro_rho is None or macro_rho <= 0,
        ],
        warn_checks=[
            macro_rho is None or macro_rho < 0.50,
            top10_coverage is None,
            top10_coverage is not None and top10_coverage < 0.30,
        ],
    )
    quality_items = [
        {
            "label": "国家样本匹配",
            "score": min(len(df_final) / 100, 1.0),
            "status": "PASS" if len(df_final) >= 100 else "WARN" if len(df_final) >= 20 else "FAIL",
            "detail": f"{len(df_final)} 个国家匹配ITU统计",
        },
        {
            "label": "宏观秩相关",
            "score": max(0.0, min((macro_rho or 0.0) / 0.8, 1.0)),
            "status": "PASS" if macro_rho is not None and macro_rho >= 0.60 else "WARN" if macro_rho is not None and macro_rho > 0 else "FAIL",
            "detail": f"Spearman rho={macro_rho if macro_rho is not None else 'NA'}",
        },
        {
            "label": "对数线性一致性",
            "score": max(0.0, min((log_pearson or 0.0) / 0.8, 1.0)),
            "status": "PASS" if log_pearson is not None and log_pearson >= 0.60 else "WARN" if log_pearson is not None and log_pearson > 0 else "FAIL",
            "detail": f"log-Pearson r={log_pearson if log_pearson is not None else 'NA'}",
        },
        {
            "label": "Top10%热点覆盖",
            "score": 0.0 if top10_coverage is None else min(top10_coverage / 0.80, 1.0),
            "status": "PASS" if top10_coverage is not None and top10_coverage >= 0.60 else "WARN" if top10_coverage is not None and top10_coverage >= 0.30 else "FAIL",
            "detail": "未计算Ookla覆盖率" if top10_coverage is None else f"{top10_coverage * 100:.1f}% Ookla活跃像元被覆盖",
        },
    ]
    coverage_by_top = {r["top_p_percent"]: r["coverage_rate"] for r in coverage_rows}
    active_pixels = None if top10_row is None else top10_row["ookla_active_pixels"]
    metric_rows = [
        metric_row("6.1.3.4", "可匹配ITU国家数", len(df_final), "个", "建议>=100", "PASS" if len(df_final) >= 100 else "WARN"),
        metric_row("6.1.3.4", "国家尺度RSS总功率与ITU流量Spearman相关", macro_rho, "-", "建议>=0.60", "PASS" if macro_rho is not None and macro_rho >= 0.60 else "WARN"),
        metric_row("6.1.3.4", "对数尺度Spearman相关", log_rho, "-", "记录单调一致性", "记录"),
        metric_row("6.1.3.4", "对数尺度Pearson相关", log_pearson, "-", "建议>=0.60", "PASS" if log_pearson is not None and log_pearson >= 0.60 else "WARN"),
        metric_row("6.1.3.4", "Ookla活跃像元数", active_pixels, "个", "记录外部验证样本规模", "记录" if active_pixels is not None else "NA"),
        metric_row("6.1.3.4", "Top1% RSS热点覆盖Ookla活跃像元率", None if coverage_by_top.get(1) is None else coverage_by_top[1] * 100, "%", "热点召回参考", "记录"),
        metric_row("6.1.3.4", "Top5% RSS热点覆盖Ookla活跃像元率", None if coverage_by_top.get(5) is None else coverage_by_top[5] * 100, "%", "热点召回参考", "记录"),
        metric_row("6.1.3.4", "Top10% RSS热点覆盖Ookla活跃像元率", None if top10_coverage is None else top10_coverage * 100, "%", "建议>=60%", "PASS" if top10_coverage is not None and top10_coverage >= 0.60 else "WARN"),
        metric_row("6.1.3.4", "Top20% RSS热点覆盖Ookla活跃像元率", None if coverage_by_top.get(20) is None else coverage_by_top[20] * 100, "%", "热点召回参考", "记录"),
        metric_row("6.1.3.4", "总体判定", verdict, "-", "PASS/WARN/FAIL", verdict),
    ]
    dashboard_path = TEST_OUT_DIR / "04_测试质量总览.png"
    plot_quality_dashboard(quality_items, "External Consistency Test: Quality Overview", dashboard_path)

    country_csv = TEST_OUT_DIR / "04_外部数据一致性_国家级聚合结果.csv"
    coverage_csv = TEST_OUT_DIR / "04_外部数据一致性_Ookla热点覆盖率.csv"
    metric_csv_path = TEST_OUT_DIR / "04_外部数据一致性与典型区域核查测试_测试指标表.csv"
    metric_json_path = TEST_OUT_DIR / "04_外部数据一致性与典型区域核查测试_测试指标表.json"
    json_path = TEST_OUT_DIR / "04_外部数据一致性与典型区域核查测试_summary.json"
    txt_path = TEST_OUT_DIR / "04_外部数据一致性与典型区域核查测试_结论.txt"

    df_final.to_csv(country_csv, index=False, encoding="utf-8-sig")
    write_csv(
        coverage_csv,
        coverage_rows,
        fieldnames=[
            "region",
            "top_p_percent",
            "ookla_active_pixels",
            "covered_active_pixels",
            "coverage_rate",
        ],
    )

    summary = {
        "test_name": "外部数据一致性与典型区域核查测试",
        "data_lineage": {
            "rss_grid": grid_info(rss_path),
            "country_mask_like_rss": str(iso_mask_path),
            "iso_lookup": str(lookup_path),
            "itu_table": str(PATHS["itu"]),
            "ookla_raster_like_rss": None if ookla_path is None else str(ookla_path),
        },
        "rss_result_note": "当前使用 6_输出数据中的 Global_5G_Radiation_Map_360arcsec_v2.tif 作为 360arcsec RSS 最终主结果。",
        "matched_country_count": int(len(df_final)),
        "spearman_pred_watts_vs_itu": spearman_raw,
        "spearman_log_pred_vs_log_itu": spearman_log,
        "pearson_log_pred_vs_log_itu": pearson_log,
        "ookla_coverage_note": ookla_note,
        "ookla_coverage_rows": coverage_rows,
        "quality_assessment": {
            "verdict": verdict,
            "quality_items": quality_items,
        },
        "key_metrics_for_test_outline": metric_rows,
        "visual_outputs": {
            "country_scatter": str(country_scatter_path),
            "ookla_coverage_bar": str(coverage_plot_path),
            "quality_dashboard": str(dashboard_path),
        },
    }

    write_json(json_path, summary)
    write_metric_outputs(metric_csv_path, metric_json_path, metric_rows)
    write_verdict_report(
        txt_path,
        "外部数据一致性与典型区域核查测试结论",
        verdict,
        [
            f"当前固定使用360arcsec主线RSS结果 {rss_path.name}，可匹配ITU国家数量为 {len(df_final)}。",
            f"国家尺度Spearman rho={macro_rho if macro_rho is not None else 'NA'}，log-Spearman rho={log_rho if log_rho is not None else 'NA'}，log-Pearson r={log_pearson if log_pearson is not None else 'NA'}。",
            f"Ookla微观覆盖率状态：{ookla_note or '已完成全球Top-p%覆盖率计算。'}",
            "该项用于判断RSS频谱地图是否在宏观通信活动强度和微观测速活跃热点上保持合理一致。",
        ],
        evidence_rows=[
            "国家聚合结果已输出为CSV，散点图采用log10尺度展示ITU流量与预测总RSS功率关系。",
            "Ookla覆盖率图展示Top-1%、Top-5%、Top-10%、Top-20%高RSS热点对测速活跃像元的覆盖比例。",
        ],
        caveats=[
            "Ookla数据具有稀疏性，未测速区域不能直接解释为无通信活动，因此覆盖率更适合作为热点召回指标而非逐像元精确率。",
            "个别国家低一致性可由统计口径差异、行政边界效应或输入数据缺失解释，应结合典型区域图件复核。",
        ],
    )

    print_saved(country_csv, coverage_csv, metric_csv_path, metric_json_path, json_path, txt_path, country_scatter_path, coverage_plot_path, dashboard_path)


if __name__ == "__main__":
    main()
