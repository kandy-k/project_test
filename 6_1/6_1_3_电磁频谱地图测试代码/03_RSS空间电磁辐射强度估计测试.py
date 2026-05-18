# -*- coding: utf-8 -*-
"""
6.1.3.3 RSS 空间电磁辐射强度估计测试

本脚本固定按项目验收主线的 6 arcmin / 360 arcsec 版本执行测试。

数据链路：
1. 4G 建设先验：
   6_输出数据/Global_LTE_360arcsec_GNN.tif

2. 360arcsec Ku/Ka 密度倍数：
   4_5G基站倍数预测/5g_density_multiplier_result_360arcsec.tif

3. 建筑环境：
   aligned_6arcmin 下的建筑高度和建筑面积栅格

4. RSS 电磁频谱地图：
   6_输出数据/Global_5G_Radiation_Map_360arcsec_v2.tif

说明：
本脚本只做测试，不重新执行全图 RSS 积分。当前测试以 6_输出数据中已经交付的
Global_5G_Radiation_Map_360arcsec_v2.tif 作为 360arcsec 电磁频谱地图主结果。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio

from common import (
    NATURE_PALETTE,
    PATHS,
    add_note_box,
    describe_array,
    get_test_out_dir,
    grid_info,
    metric_row,
    pearson_corr,
    print_saved,
    read_band,
    sample_pair,
    same_grid,
    set_academic_style,
    status_from_checks,
    spearman_corr,
    style_academic_axes,
    write_csv,
    write_json,
    write_metric_outputs,
    write_verdict_report,
    plot_quality_dashboard,
)

TEST_OUT_DIR = get_test_out_dir("03_RSS空间电磁辐射强度估计测试")


def calculate_grid_area_m2(tif_path):
    """根据栅格分辨率估算单像元面积。

    当前数据使用 EPSG:4326 经纬度坐标。这里沿用生产脚本中的简化处理：
    1 度约等于 111319.9 m，因此 0.1° × 0.1° 约为 123.92 km²。
    """

    with rasterio.open(tif_path) as src:
        res_x, res_y = src.res
        crs = src.crs

    if crs and crs.is_geographic:
        return abs(res_x * 111319.9 * res_y * 111319.9)
    return abs(res_x * res_y)


def beta_from_height_area(height, building_area, grid_area_m2):
    """复现频谱强度计算中的环境阻塞参数 beta。

    beta = 4p / (pi D)
    p 为建筑覆盖率，D 为由平均建筑高度近似得到的特征尺度，并限制最大值 40 m。
    """

    height = np.asarray(height, dtype=np.float64)
    building_area = np.asarray(building_area, dtype=np.float64)

    coverage_ratio = np.clip(building_area / grid_area_m2, 0.0, 1.0)
    feature_width = np.minimum(height, 40.0)

    beta = np.zeros_like(height, dtype=np.float64)
    valid = feature_width > 0.1
    beta[valid] = (4.0 * coverage_ratio[valid]) / (np.pi * feature_width[valid])
    return beta


def choose_360_rss_result():
    """返回 360arcsec RSS 最终主结果文件。

    你当前重新跑通后的最终输出是 Global_5G_Radiation_Map_360arcsec_v2.tif，
    因此测试脚本固定使用该文件，不再回退到旧版 360arcsec 或 60arcsec 结果。
    """

    rss_path = PATHS["rss_360_v2"]
    if not rss_path.exists() or rss_path.stat().st_size <= 0:
        raise FileNotFoundError(f"未找到 360arcsec RSS 最终主结果文件: {rss_path}")
    return rss_path


def append_correlation_row(rows, layer_name, rss, layer, valid_mask, same_grid_flag):
    """抽样计算 RSS 与某个解释变量之间的相关性。"""

    if not same_grid_flag:
        rows.append(
            {
                "layer": layer_name,
                "same_grid_with_rss": False,
                "spearman_rho": None,
                "spearman_p_value": None,
                "pearson_r": None,
                "sample_count": 0,
            }
        )
        return

    rss_sample, layer_sample = sample_pair(rss, layer, mask=valid_mask, max_samples=200_000)
    sp = spearman_corr(rss_sample, layer_sample)
    pr = pearson_corr(rss_sample, layer_sample)

    rows.append(
        {
            "layer": layer_name,
            "same_grid_with_rss": True,
            "spearman_rho": None if sp is None else sp["rho"],
            "spearman_p_value": None if sp is None else sp["p_value"],
            "pearson_r": None if pr is None else pr["r"],
            "sample_count": 0 if sp is None else sp["n"],
        }
    )


def plot_rss_histogram(rss, valid_mask, output_path: Path):
    """绘制 RSS 有效像元的数值分布。"""
    set_academic_style(plt)
    vals = rss[valid_mask].astype(float)
    if vals.size == 0:
        return
    if vals.size > 300_000:
        rng = np.random.default_rng(42)
        vals = rng.choice(vals, 300_000, replace=False)

    p05, p50, p95 = np.percentile(vals, [5, 50, 95])
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=300)
    ax.hist(vals, bins=70, color=NATURE_PALETTE["blue"], edgecolor="white", linewidth=0.25, zorder=3)
    for value, label, color in [(p05, "P05", NATURE_PALETTE["gray"]), (p50, "P50", NATURE_PALETTE["orange"]), (p95, "P95", NATURE_PALETTE["red"])]:
        ax.axvline(value, color=color, linestyle="--", linewidth=2, label=f"{label}={value:.1f} dBm")
    ax.set_xlabel("RSS intensity (dBm)")
    ax.set_ylabel("Calculated grid pixels")
    ax.set_title("RSS Distribution of the 360arcsec Electromagnetic Map", pad=10)
    style_academic_axes(ax, grid_axis="y")
    ax.legend(loc="upper right")
    add_note_box(ax, "Only pixels above the land-side bottom floor are shown.\nDashed lines mark P05/P50/P95 of RSS.", loc=(0.03, 0.95), fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_rows(rows, output_path: Path):
    """绘制 RSS 与解释变量之间的 Spearman 相关性。"""
    set_academic_style(plt)
    valid_rows = [r for r in rows if r.get("spearman_rho") is not None]
    if not valid_rows:
        return

    label_map = {
        "LTE_4G_prior_log1p": "4G construction prior",
        "Building_height": "Building height",
        "Blocking_beta": "Blockage beta",
        "KuKa_density_multiplier_360arcsec": "Ku/Ka multiplier",
    }
    labels = [label_map.get(r["layer"], r["layer"]) for r in valid_rows]
    values = [r["spearman_rho"] for r in valid_rows]
    order = np.argsort(values)
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]
    colors = [NATURE_PALETTE["blue"] if v >= 0 else NATURE_PALETTE["red"] for v in values]

    fig, ax = plt.subplots(figsize=(8.8, 5), dpi=300)
    bars = ax.barh(labels, values, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    ax.axvline(0, color=NATURE_PALETTE["black"], linewidth=0.8)
    ax.set_xlabel("Spearman rho with RSS")
    ax.set_title("Physical Correspondence of RSS Map", pad=10)
    ax.grid(axis="x", color=NATURE_PALETTE["light_gray"], linestyle="-", linewidth=0.6, alpha=0.9)
    ax.set_xlim(min(-0.1, min(values) - 0.08), max(0.1, max(values) + 0.08))
    style_academic_axes(ax, grid_axis="x")
    for bar, value in zip(bars, values):
        ax.text(
            value,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=8.8,
        )
    add_note_box(ax, "This figure checks whether RSS follows the expected\n4G prior, blockage, and Ku/Ka density multiplier signals.", loc=(0.03, 0.95), fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    rss_path = choose_360_rss_result()

    lte_4g_prior_path = PATHS["lte_pred_360"]
    ratio_path = PATHS["ratio_360"]
    height_path = PATHS["height_360"]
    area_path = PATHS["area_360"]
    admin_path = PATHS["admin_360"]

    rss, _, _, _, rss_nodata = read_band(rss_path)
    lte_4g_prior, _, _, _, lte_nodata = read_band(lte_4g_prior_path)
    height, _, _, _, height_nodata = read_band(height_path)
    building_area, _, _, _, area_nodata = read_band(area_path)
    admin, _, _, _, _ = read_band(admin_path)

    ratio_exists = ratio_path.exists() and ratio_path.stat().st_size > 0
    ratio = None
    ratio_nodata = None
    if ratio_exists:
        ratio, _, _, _, ratio_nodata = read_band(ratio_path)

    grid_area_m2 = calculate_grid_area_m2(rss_path)
    beta = beta_from_height_area(height, building_area, grid_area_m2)

    land_mask = admin > 0
    rss_valid = land_mask & np.isfinite(rss)
    if rss_nodata is not None and np.isfinite(rss_nodata):
        rss_valid &= rss != rss_nodata

    # 生产脚本中，陆地上不满足计算条件的位置通常写为 -200 dBm。
    # 这里把大于 -199 dBm 的像元视为“有实际积分计算贡献”的像元。
    calculated_rss_mask = rss_valid & (rss > -199.0)

    correlation_rows = []

    append_correlation_row(
        rows=correlation_rows,
        layer_name="LTE_4G_prior_log1p",
        rss=rss,
        layer=np.log1p(np.clip(lte_4g_prior, 0, None)),
        valid_mask=calculated_rss_mask,
        same_grid_flag=same_grid(rss_path, lte_4g_prior_path),
    )

    append_correlation_row(
        rows=correlation_rows,
        layer_name="Building_height",
        rss=rss,
        layer=height,
        valid_mask=calculated_rss_mask,
        same_grid_flag=same_grid(rss_path, height_path),
    )

    append_correlation_row(
        rows=correlation_rows,
        layer_name="Blocking_beta",
        rss=rss,
        layer=beta,
        valid_mask=calculated_rss_mask,
        same_grid_flag=same_grid(rss_path, height_path) and same_grid(rss_path, area_path),
    )

    if ratio_exists:
        append_correlation_row(
            rows=correlation_rows,
            layer_name="KuKa_density_multiplier_360arcsec",
            rss=rss,
            layer=ratio,
            valid_mask=calculated_rss_mask,
            same_grid_flag=same_grid(rss_path, ratio_path),
        )

    summary = {
        "test_name": "RSS空间电磁辐射强度估计测试",
        "data_lineage": {
            "rss_grid": grid_info(rss_path),
            "lte_4g_prior_grid": grid_info(lte_4g_prior_path),
            "density_multiplier_grid": grid_info(ratio_path) if ratio_exists else None,
            "height_grid": grid_info(height_path),
            "building_area_grid": grid_info(area_path),
            "admin_mask_grid": grid_info(admin_path),
        },
        "rss_result_note": "当前使用 6_输出数据中的 Global_5G_Radiation_Map_360arcsec_v2.tif 作为 360arcsec RSS 最终主结果。",
        "grid_alignment": {
            "rss_vs_lte_4g_prior": same_grid(rss_path, lte_4g_prior_path),
            "rss_vs_density_multiplier": same_grid(rss_path, ratio_path) if ratio_exists else False,
            "rss_vs_height": same_grid(rss_path, height_path),
            "rss_vs_building_area": same_grid(rss_path, area_path),
            "rss_vs_admin_mask": same_grid(rss_path, admin_path),
        },
        "grid_area_m2": float(grid_area_m2),
        "rss_stats_on_land": describe_array(rss, rss_nodata, land_mask),
        "rss_stats_calculated_pixels": describe_array(rss, rss_nodata, calculated_rss_mask),
        "land_pixel_count": int(np.sum(land_mask)),
        "rss_valid_land_count": int(np.sum(rss_valid)),
        "calculated_rss_pixel_count": int(np.sum(calculated_rss_mask)),
        "noise_floor_count_minus_200": int(np.sum(rss_valid & (rss <= -199.0))),
        "blocking_beta_stats_on_land": describe_array(beta, None, land_mask),
        "ratio_360_exists": ratio_exists,
        "correlations": correlation_rows,
    }

    grid_ok = all(summary["grid_alignment"].values())
    calc_ratio = float(summary["calculated_rss_pixel_count"] / max(summary["rss_valid_land_count"], 1))
    corr_by_layer = {row["layer"]: row for row in correlation_rows}
    prior_rho = corr_by_layer.get("LTE_4G_prior_log1p", {}).get("spearman_rho")
    multiplier_rho = corr_by_layer.get("KuKa_density_multiplier_360arcsec", {}).get("spearman_rho")
    p05 = summary["rss_stats_calculated_pixels"]["p05"]
    p95 = summary["rss_stats_calculated_pixels"]["p95"]
    dynamic_range = None if p05 is None or p95 is None else float(p95 - p05)
    verdict = status_from_checks(
        fail_checks=[
            not grid_ok,
            summary["calculated_rss_pixel_count"] <= 0,
            p05 is None,
        ],
        warn_checks=[
            calc_ratio < 0.01,
            prior_rho is None or prior_rho < 0.10,
            dynamic_range is None or dynamic_range < 1.0,
        ],
    )
    quality_items = [
        {
            "label": "核心图层共格",
            "score": 1.0 if grid_ok else 0.0,
            "status": "PASS" if grid_ok else "FAIL",
            "detail": f"RSS与先验/倍数/建筑/掩膜对齐：{grid_ok}",
        },
        {
            "label": "有效RSS计算区",
            "score": min(calc_ratio / 0.20, 1.0),
            "status": "PASS" if calc_ratio >= 0.05 else "WARN" if calc_ratio > 0 else "FAIL",
            "detail": f"{summary['calculated_rss_pixel_count']:,} 像元；占陆地有效RSS {calc_ratio * 100:.1f}%",
        },
        {
            "label": "建设强度对应",
            "score": max(0.0, min((prior_rho or 0.0) / 0.5, 1.0)),
            "status": "PASS" if prior_rho is not None and prior_rho >= 0.30 else "WARN" if prior_rho is not None and prior_rho >= 0.10 else "FAIL",
            "detail": f"RSS vs 4G先验 rho={prior_rho if prior_rho is not None else 'NA'}",
        },
        {
            "label": "RSS动态范围",
            "score": min((dynamic_range or 0.0) / 20.0, 1.0),
            "status": "PASS" if dynamic_range is not None and dynamic_range >= 5.0 else "WARN" if dynamic_range is not None and dynamic_range >= 1.0 else "FAIL",
            "detail": f"P95-P05={dynamic_range:.2f} dB" if dynamic_range is not None else "P95-P05=NA",
        },
    ]
    if multiplier_rho is not None:
        quality_items.append(
            {
                "label": "倍数映射对应",
                "score": max(0.0, min(multiplier_rho / 0.5, 1.0)),
                "status": "PASS" if multiplier_rho >= 0.30 else "WARN" if multiplier_rho >= 0.10 else "FAIL",
                "detail": f"RSS vs Ku/Ka倍数 rho={multiplier_rho:.3f}",
            }
        )
    summary["quality_assessment"] = {
        "verdict": verdict,
        "quality_items": quality_items,
    }
    height_rho = corr_by_layer.get("Building_height", {}).get("spearman_rho")
    beta_rho = corr_by_layer.get("Blocking_beta", {}).get("spearman_rho")
    metric_rows = [
        metric_row("6.1.3.3", "RSS与核心输入图层共格", grid_ok, "-", "应为True", "PASS" if grid_ok else "FAIL"),
        metric_row("6.1.3.3", "陆地RSS有效像元数", summary["rss_valid_land_count"], "个", "记录输出覆盖规模", "记录"),
        metric_row("6.1.3.3", "实际积分计算RSS像元数", summary["calculated_rss_pixel_count"], "个", "应>0", "PASS" if summary["calculated_rss_pixel_count"] > 0 else "FAIL"),
        metric_row("6.1.3.3", "实际积分计算RSS像元占比", calc_ratio * 100, "%", "建议>=5%", "PASS" if calc_ratio >= 0.05 else "WARN"),
        metric_row("6.1.3.3", "RSS P05", summary["rss_stats_calculated_pixels"]["p05"], "dBm", "记录分布下界", "记录"),
        metric_row("6.1.3.3", "RSS P50", summary["rss_stats_calculated_pixels"]["p50"], "dBm", "记录中位强度", "记录"),
        metric_row("6.1.3.3", "RSS P95", summary["rss_stats_calculated_pixels"]["p95"], "dBm", "记录分布上界", "记录"),
        metric_row("6.1.3.3", "RSS动态范围P95-P05", dynamic_range, "dB", "建议>=5 dB", "PASS" if dynamic_range is not None and dynamic_range >= 5.0 else "WARN"),
        metric_row("6.1.3.3", "RSS与4G建设先验Spearman相关", prior_rho, "-", "建议>=0.30", "PASS" if prior_rho is not None and prior_rho >= 0.30 else "WARN"),
        metric_row("6.1.3.3", "RSS与建筑高度Spearman相关", height_rho, "-", "记录环境响应", "记录"),
        metric_row("6.1.3.3", "RSS与阻塞参数Spearman相关", beta_rho, "-", "记录环境响应", "记录"),
        metric_row("6.1.3.3", "RSS与Ku/Ka密度倍数Spearman相关", multiplier_rho, "-", "建议>=0.30", "PASS" if multiplier_rho is not None and multiplier_rho >= 0.30 else "WARN"),
        metric_row("6.1.3.3", "总体判定", verdict, "-", "PASS/WARN/FAIL", verdict),
    ]
    summary["key_metrics_for_test_outline"] = metric_rows

    rss_hist_path = TEST_OUT_DIR / "03_RSS分布直方图.png"
    corr_plot_path = TEST_OUT_DIR / "03_RSS空间对应关系.png"
    dashboard_path = TEST_OUT_DIR / "03_测试质量总览.png"
    plot_rss_histogram(rss, calculated_rss_mask, rss_hist_path)
    plot_correlation_rows(correlation_rows, corr_plot_path)
    plot_quality_dashboard(quality_items, "RSS Electromagnetic Map Test: Quality Overview", dashboard_path)

    summary["visual_outputs"] = {
        "rss_histogram": str(rss_hist_path),
        "spatial_correlations_bar": str(corr_plot_path),
        "quality_dashboard": str(dashboard_path),
    }

    json_path = TEST_OUT_DIR / "03_RSS空间电磁辐射强度估计测试_summary.json"
    csv_path = TEST_OUT_DIR / "03_RSS空间电磁辐射强度估计测试_空间对应关系.csv"
    metric_csv_path = TEST_OUT_DIR / "03_RSS空间电磁辐射强度估计测试_测试指标表.csv"
    metric_json_path = TEST_OUT_DIR / "03_RSS空间电磁辐射强度估计测试_测试指标表.json"
    txt_path = TEST_OUT_DIR / "03_RSS空间电磁辐射强度估计测试_结论.txt"

    write_json(json_path, summary)
    write_csv(csv_path, correlation_rows)
    write_metric_outputs(metric_csv_path, metric_json_path, metric_rows)
    write_verdict_report(
        txt_path,
        "RSS空间电磁辐射强度估计测试结论",
        verdict,
        [
            f"当前固定使用360arcsec主线RSS结果 {rss_path.name}，单位按dBm解释。",
            f"RSS陆地有效像元数为 {summary['rss_valid_land_count']:,}，其中高于-199 dBm的计算型像元为 {summary['calculated_rss_pixel_count']:,}。",
            f"RSS与4G建设先验的Spearman rho为 {prior_rho if prior_rho is not None else 'NA'}，用于判断高辐射区域是否跟随通信建设强度。",
            f"计算型RSS的P05/P50/P95分别为 {summary['rss_stats_calculated_pixels']['p05']}、{summary['rss_stats_calculated_pixels']['p50']}、{summary['rss_stats_calculated_pixels']['p95']} dBm。",
        ],
        evidence_rows=[
            f"RSS与核心输入图层共格状态：{grid_ok}。",
            "CSV中已输出RSS与4G建设先验、建筑高度、阻塞参数、360arcsec密度倍数的空间对应关系。",
            "该项重点检验RSS结果的物理可解释性和空间稳定性，不要求与真实逐点频谱测量值完全一致。",
        ],
        caveats=[
            "低于或等于-199 dBm的像元按生产流程中的底噪/无贡献区域解释，应结合NoData和陆地掩膜判读。",
        ],
    )

    print_saved(json_path, csv_path, metric_csv_path, metric_json_path, txt_path, rss_hist_path, corr_plot_path, dashboard_path)


if __name__ == "__main__":
    main()
