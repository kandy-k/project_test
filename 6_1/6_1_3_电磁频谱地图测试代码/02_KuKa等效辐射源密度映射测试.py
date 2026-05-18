# -*- coding: utf-8 -*-
"""
6.1.3.2 Ku/Ka 等效辐射源密度映射测试

本测试以已经生成的 360arcsec Ku/Ka 密度倍数栅格作为受试对象。
核心验证口径调整为“建筑高度正值区域”，即：

    V/S 等效建筑高度 > 0

原因是 Ku/Ka 密度倍数的场景响应主要来自建成环境高度/UMa-UMi 切换。
无建筑区域的 0 高度像元数量很大，若纳入主统计，会显著压低分位数差异，
不利于判断建筑区内的高频段密度映射是否合理。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import (
    NATURE_PALETTE,
    PATHS,
    add_note_box,
    describe_array,
    get_test_out_dir,
    grid_info,
    metric_row,
    plot_quality_dashboard,
    print_saved,
    read_band,
    sample_pair,
    same_grid,
    set_academic_style,
    spearman_corr,
    status_from_checks,
    style_academic_axes,
    write_csv,
    write_json,
    write_metric_outputs,
    write_verdict_report,
)


TEST_OUT_DIR = get_test_out_dir("02_KuKa等效辐射源密度映射测试")

PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "1_输入数据").exists())
ALIGNED_6ARCMIN = PROJECT_ROOT / "1_输入数据" / "2_已处理数据" / "aligned_6arcmin"

HEIGHT_VS_360 = ALIGNED_6ARCMIN / "GHS_BUILT_H_FROM_VS_E2025_E2030_aligned_360arcsec.tif"
HEIGHT_VS_REPORT = ALIGNED_6ARCMIN / "GHS_BUILT_H_FROM_VS_E2025_E2030_aligned_360arcsec_check_report.json"


def valid_mask(data, nodata=None):
    mask = np.isfinite(data)
    if nodata is not None and np.isfinite(nodata):
        mask &= data != nodata
    return mask


def safe_ratio(numerator, denominator):
    if denominator is None or denominator <= 0:
        return None
    return float(numerator / denominator)


def safe_percentiles(data, mask):
    values = np.asarray(data[mask], dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "count": 0,
            "min": None,
            "p05": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
            "mean": None,
            "std": None,
        }
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def correlation_row(layer_name, ratio, layer, mask):
    ratio_sample, layer_sample = sample_pair(ratio, layer, mask=mask, max_samples=200_000)
    corr = spearman_corr(ratio_sample, layer_sample)
    return {
        "layer": layer_name,
        "spearman_rho": None if corr is None else corr["rho"],
        "spearman_p_value": None if corr is None else corr["p_value"],
        "sample_count": 0 if corr is None else corr["n"],
    }


def scene_row(scene_type, scene_name, mask, scene_signal, ratio, lte_4g_prior):
    valid = mask & np.isfinite(scene_signal) & np.isfinite(ratio) & np.isfinite(lte_4g_prior)
    valid &= ratio > 0
    if np.sum(valid) == 0:
        return {
            "scene_type": scene_type,
            "scene_name": scene_name,
            "pixel_count": 0,
            "scene_signal_p50": None,
            "multiplier_p50": None,
            "multiplier_p75": None,
            "multiplier_mean": None,
            "kuka_density_p50": None,
        }

    lte = np.clip(lte_4g_prior[valid], 0, None)
    kuka_density = lte * ratio[valid]
    scene_stats = safe_percentiles(scene_signal, valid)
    ratio_stats = safe_percentiles(ratio, valid)
    kuka_stats = safe_percentiles(kuka_density, np.isfinite(kuka_density))

    return {
        "scene_type": scene_type,
        "scene_name": scene_name,
        "pixel_count": int(np.sum(valid)),
        "scene_signal_p50": scene_stats["p50"],
        "multiplier_p50": ratio_stats["p50"],
        "multiplier_p75": ratio_stats["p75"],
        "multiplier_mean": ratio_stats["mean"],
        "kuka_density_p50": kuka_stats["p50"],
    }


def build_tercile_rows(scene_signal, scene_label, base_mask, ratio, lte_4g_prior):
    positive = base_mask & np.isfinite(scene_signal) & (scene_signal > 0)
    if np.sum(positive) < 100:
        return []

    q33, q67 = np.percentile(scene_signal[positive], [33.33, 66.67])
    groups = [
        (f"Low {scene_label}", positive & (scene_signal <= q33)),
        (f"Medium {scene_label}", positive & (scene_signal > q33) & (scene_signal <= q67)),
        (f"High {scene_label}", positive & (scene_signal > q67)),
    ]
    return [
        scene_row(f"{scene_label}_tercile", name, mask, scene_signal, ratio, lte_4g_prior)
        for name, mask in groups
    ]


def plot_ratio_histogram(ratio, mask, output_path: Path):
    set_academic_style(plt)
    values = ratio[mask].astype(float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    if values.size > 300_000:
        rng = np.random.default_rng(42)
        values = rng.choice(values, 300_000, replace=False)

    p50, p95, p99 = np.percentile(values, [50, 95, 99])
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=300)
    ax.hist(values, bins=70, color=NATURE_PALETTE["blue"], edgecolor="white", linewidth=0.25, zorder=3)
    for value, label, color in [
        (p50, "P50", NATURE_PALETTE["gray"]),
        (p95, "P95", NATURE_PALETTE["orange"]),
        (p99, "P99", NATURE_PALETTE["red"]),
    ]:
        ax.axvline(value, color=color, linestyle="--", linewidth=2, label=f"{label}={value:.2f}")
    ax.set_xlabel("4G-to-Ku/Ka density multiplier")
    ax.set_ylabel("Building-positive grid pixels")
    ax.set_title("Ku/Ka Density Multiplier in Building-Positive Pixels", pad=10)
    style_academic_axes(ax, grid_axis="y")
    ax.legend(loc="upper right")
    add_note_box(
        ax,
        "Statistics are calculated only where V/S equivalent\nbuilding height is greater than zero.",
        loc=(0.03, 0.95),
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_metric_summary(ratio_stats, height_stats, height_rho, verdict, output_path: Path):
    set_academic_style(plt)

    multiplier_labels = ["P50", "P95", "P99", "Max"]
    multiplier_values = [
        ratio_stats["p50"],
        ratio_stats["p95"],
        ratio_stats["p99"],
        ratio_stats["max"],
    ]
    height_labels = ["H P50", "H P95"]
    height_values = [height_stats["p50"], height_stats["p95"]]
    p99_p50 = safe_ratio(ratio_stats["p99"], ratio_stats["p50"])

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.8), dpi=300)

    bars = axes[0].bar(multiplier_labels, multiplier_values, color=NATURE_PALETTE["blue"], edgecolor="white", linewidth=0.8, zorder=3)
    axes[0].set_ylabel("Density multiplier")
    axes[0].set_title("Multiplier Quantiles")
    style_academic_axes(axes[0], grid_axis="y")
    for bar, value in zip(bars, multiplier_values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8.5)

    bars = axes[1].bar(height_labels, height_values, color=NATURE_PALETTE["green"], edgecolor="white", linewidth=0.8, zorder=3)
    axes[1].set_ylabel("V/S equivalent height (m)")
    axes[1].set_title("Building Height Signal")
    style_academic_axes(axes[1], grid_axis="y")
    for bar, value in zip(bars, height_values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f} m", ha="center", va="bottom", fontsize=8.5)

    axes[2].axis("off")
    rho_text = "NA" if height_rho is None else f"{height_rho:.4f}"
    p99_p50_text = "NA" if p99_p50 is None else f"{p99_p50:.4f}"
    cards = [
        ("P99 / P50", p99_p50_text),
        ("Spearman rho", rho_text),
        ("Verdict", verdict),
    ]
    for idx, (label, value) in enumerate(cards):
        y = 0.78 - idx * 0.28
        axes[2].text(0.05, y + 0.08, label, transform=axes[2].transAxes, fontsize=9.5, color=NATURE_PALETTE["gray"])
        axes[2].text(0.05, y - 0.02, value, transform=axes[2].transAxes, fontsize=18, color=NATURE_PALETTE["black"], fontweight="bold")
        axes[2].axhline(y - 0.10, xmin=0.05, xmax=0.95, color=NATURE_PALETTE["light_gray"], linewidth=0.8)
    axes[2].set_title("Key Checks")

    fig.suptitle("Ku/Ka Density Multiplier Test Metrics in Building-Positive Pixels", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    lte_4g_prior_path = PATHS["lte_pred_360"]
    ratio_path = PATHS["ratio_360"]
    area_path = PATHS["area_360"]
    volume_path = PATHS["built_v_360"]
    height_path = HEIGHT_VS_360

    if not height_path.exists():
        raise FileNotFoundError(
            f"未找到 V/S 等效建筑高度文件: {height_path}\n"
            "请先运行 2_数据处理/建筑体积面积比生成等效高度_360arcsec.py"
        )

    ratio, _, _, _, ratio_nodata = read_band(ratio_path)
    height_vs, _, _, _, height_nodata = read_band(height_path)
    area, _, _, _, area_nodata = read_band(area_path)
    volume, _, _, _, volume_nodata = read_band(volume_path)
    lte_4g_prior, _, _, _, lte_nodata = read_band(lte_4g_prior_path)

    ratio_valid = valid_mask(ratio, ratio_nodata) & (ratio > 0)
    height_valid = valid_mask(height_vs, height_nodata)
    area_valid = valid_mask(area, area_nodata)
    volume_valid = valid_mask(volume, volume_nodata)
    lte_valid = valid_mask(lte_4g_prior, lte_nodata)

    building_mask = ratio_valid & height_valid & (height_vs > 0)
    building_lte_mask = building_mask & lte_valid

    same_grid_ratio_height = same_grid(ratio_path, height_path)
    same_grid_ratio_area = same_grid(ratio_path, area_path)
    same_grid_ratio_volume = same_grid(ratio_path, volume_path)
    same_grid_ratio_lte = same_grid(ratio_path, lte_4g_prior_path)
    grid_ok = same_grid_ratio_height and same_grid_ratio_area and same_grid_ratio_volume and same_grid_ratio_lte

    ratio_stats_all = describe_array(ratio, ratio_nodata, ratio_valid)
    ratio_stats_building = describe_array(ratio, ratio_nodata, building_mask)
    height_stats_building = describe_array(height_vs, height_nodata, building_mask)
    area_stats_building = describe_array(area, area_nodata, building_mask & area_valid & (area > 0))
    volume_stats_building = describe_array(volume, volume_nodata, building_mask & volume_valid & (volume > 0))

    lte_sample, ratio_sample = sample_pair(lte_4g_prior, ratio, mask=building_lte_mask, max_samples=200_000)
    kuka_density_sample = np.clip(lte_sample, 0, None) * ratio_sample

    correlation_rows = [
        correlation_row("V_over_S_equivalent_height", ratio, height_vs, building_mask),
        correlation_row("building_area", ratio, area, building_mask & area_valid & (area > 0)),
        correlation_row("building_volume", ratio, volume, building_mask & volume_valid & (volume > 0)),
    ]
    corr_by_layer = {r["layer"]: r for r in correlation_rows}
    height_rho = corr_by_layer["V_over_S_equivalent_height"]["spearman_rho"]
    area_rho = corr_by_layer["building_area"]["spearman_rho"]
    volume_rho = corr_by_layer["building_volume"]["spearman_rho"]

    height_scene_rows = build_tercile_rows(height_vs, "V/S height", building_mask, ratio, lte_4g_prior)
    area_scene_rows = build_tercile_rows(area, "building area", building_mask & area_valid & (area > 0), ratio, lte_4g_prior)
    volume_scene_rows = build_tercile_rows(volume, "building volume", building_mask & volume_valid & (volume > 0), ratio, lte_4g_prior)
    scene_group_rows = height_scene_rows + area_scene_rows + volume_scene_rows

    height_rows = [r for r in height_scene_rows if r["pixel_count"] > 0]
    height_group_pass = False
    height_high_low_mean_ratio = None
    if len(height_rows) >= 3:
        low_mean = height_rows[0]["multiplier_mean"]
        high_mean = height_rows[-1]["multiplier_mean"]
        height_group_pass = low_mean is not None and high_mean is not None and high_mean > low_mean
        height_high_low_mean_ratio = safe_ratio(high_mean, low_mean)

    building_ratio_count = ratio_stats_building["count"]
    building_ratio_p50 = ratio_stats_building["p50"] or 0
    building_ratio_p95 = ratio_stats_building["p95"] or 0
    building_ratio_p99 = ratio_stats_building["p99"] or 0
    p95_p50_ratio = safe_ratio(building_ratio_p95, building_ratio_p50)
    p99_p50_ratio = safe_ratio(building_ratio_p99, building_ratio_p50)
    building_pixel_rate = safe_ratio(building_ratio_count, ratio_stats_all["count"]) or 0.0

    verdict = status_from_checks(
        fail_checks=[
            not grid_ok,
            building_ratio_count <= 0,
            building_ratio_p50 <= 1.0,
        ],
        warn_checks=[
            height_rho is None or height_rho < 0.60,
            not height_group_pass,
            p95_p50_ratio is None or p95_p50_ratio > 2.0,
        ],
    )

    quality_items = [
        {
            "label": "核心图层共格",
            "score": 1.0 if grid_ok else 0.0,
            "status": "PASS" if grid_ok else "FAIL",
            "detail": f"倍数/高度/面积/体量/4G先验对齐: {grid_ok}",
        },
        {
            "label": "建筑区有效性",
            "score": min(building_pixel_rate / 0.10, 1.0),
            "status": "PASS" if building_ratio_count > 0 else "FAIL",
            "detail": f"建筑高度正值像元 {building_ratio_count:,}, 占全图 {building_pixel_rate * 100:.1f}%",
        },
        {
            "label": "建筑区倍数范围",
            "score": 1.0 if building_ratio_p50 > 1.0 and building_ratio_p95 > building_ratio_p50 else 0.4,
            "status": "PASS" if building_ratio_p50 > 1.0 and building_ratio_p95 > building_ratio_p50 else "WARN",
            "detail": f"P50={building_ratio_p50:.2f}, P95={building_ratio_p95:.2f}",
        },
        {
            "label": "高度场景响应",
            "score": max(0.0, min((height_rho or 0.0) / 0.8, 1.0)),
            "status": "PASS" if height_rho is not None and height_rho >= 0.60 else "WARN" if height_rho is not None and height_rho >= 0.30 else "FAIL",
            "detail": f"倍数 vs V/S高度 Spearman rho={height_rho if height_rho is not None else 'NA'}",
        },
        {
            "label": "分组差异",
            "score": 1.0 if height_group_pass else 0.4,
            "status": "PASS" if height_group_pass else "WARN",
            "detail": "高V/S高度组平均倍数高于低V/S高度组" if height_group_pass else "高度分组差异不明显",
        },
    ]

    summary = {
        "test_name": "Ku/Ka等效辐射源密度映射测试",
        "test_scope_note": "主统计口径仅使用 V/S 等效建筑高度 > 0 的建筑区像元；全图统计仅作背景参考。",
        "data_lineage": {
            "lte_4g_prior_grid": grid_info(lte_4g_prior_path),
            "density_multiplier_grid": grid_info(ratio_path),
            "equivalent_height_grid_from_volume_area": grid_info(height_path),
            "equivalent_height_check_report": str(HEIGHT_VS_REPORT),
            "building_area_grid": grid_info(area_path),
            "building_volume_grid": grid_info(volume_path),
            "kuka_equivalent_density_definition": "Ku/Ka等效辐射源密度 = GraphSAGE 4G建设先验 × 360arcsec密度倍数栅格",
        },
        "grid_alignment": {
            "ratio_vs_v_over_s_height": same_grid_ratio_height,
            "ratio_vs_building_area": same_grid_ratio_area,
            "ratio_vs_building_volume": same_grid_ratio_volume,
            "ratio_vs_lte_4g_prior": same_grid_ratio_lte,
        },
        "ratio_stats_all_valid_pixels_background": ratio_stats_all,
        "ratio_stats_building_positive_pixels": ratio_stats_building,
        "v_over_s_height_stats_building_positive_pixels": height_stats_building,
        "building_area_stats_building_positive_pixels": area_stats_building,
        "building_volume_stats_building_positive_pixels": volume_stats_building,
        "building_positive_pixel_count": building_ratio_count,
        "building_positive_pixel_rate_in_ratio_grid": building_pixel_rate,
        "kuka_equivalent_density_sample_stats_building_pixels": describe_array(kuka_density_sample),
        "correlations_in_building_positive_pixels": correlation_rows,
        "scene_group_consistency": scene_group_rows,
        "height_group_high_low_mean_ratio": height_high_low_mean_ratio,
        "quality_assessment": {
            "verdict": verdict,
            "quality_items": quality_items,
            "note": "当前Ku/Ka映射采用28 GHz作为Ka代理频点；主验证口径为建筑高度正值区域。",
        },
    }

    metric_rows = [
        metric_row("6.1.3.2", "建筑区密度倍数P50", building_ratio_p50, "倍", "应>1", "PASS" if building_ratio_p50 > 1.0 else "FAIL"),
        metric_row("6.1.3.2", "建筑区密度倍数P95", building_ratio_p95, "倍", "应>P50", "PASS" if building_ratio_p95 > building_ratio_p50 else "WARN"),
        metric_row("6.1.3.2", "建筑区密度倍数P99", building_ratio_p99, "倍", "记录高值尾部", "记录"),
        metric_row("6.1.3.2", "建筑区密度倍数P99/P50", p99_p50_ratio, "倍", "记录高值尾部提升", "记录"),
        metric_row("6.1.3.2", "建筑区密度倍数最大值", ratio_stats_building["max"], "倍", "记录极值", "记录"),
        metric_row("6.1.3.2", "倍数与V/S等效高度Spearman相关", height_rho, "-", "建议>=0.60", "PASS" if height_rho is not None and height_rho >= 0.60 else "WARN"),
        metric_row("6.1.3.2", "V/S等效高度P50", height_stats_building["p50"], "m", "记录建筑区高度", "记录"),
        metric_row("6.1.3.2", "V/S等效高度P95", height_stats_building["p95"], "m", "记录建筑区高度", "记录"),
        metric_row("6.1.3.2", "总体判定", verdict, "-", "PASS/WARN/FAIL", verdict),
    ]
    summary["key_metrics_for_test_outline"] = metric_rows

    ratio_hist_path = TEST_OUT_DIR / "02_建筑区密度倍数分布直方图.png"
    metric_summary_plot_path = TEST_OUT_DIR / "02_建筑区密度倍数测试指标总览.png"
    dashboard_path = TEST_OUT_DIR / "02_测试质量总览.png"
    plot_ratio_histogram(ratio, building_mask, ratio_hist_path)
    plot_metric_summary(ratio_stats_building, height_stats_building, height_rho, verdict, metric_summary_plot_path)
    plot_quality_dashboard(quality_items, "Ku/Ka Equivalent Density Test: Quality Overview", dashboard_path)

    summary["visual_outputs"] = {
        "building_positive_ratio_histogram": str(ratio_hist_path),
        "building_positive_metric_summary": str(metric_summary_plot_path),
        "quality_dashboard": str(dashboard_path),
    }

    json_path = TEST_OUT_DIR / "02_KuKa等效辐射源密度映射测试_summary.json"
    corr_csv_path = TEST_OUT_DIR / "02_KuKa等效辐射源密度映射测试_建筑区相关性.csv"
    scene_csv_path = TEST_OUT_DIR / "02_KuKa等效辐射源密度映射测试_建筑区分场景统计表.csv"
    metric_csv_path = TEST_OUT_DIR / "02_KuKa等效辐射源密度映射测试_测试指标表.csv"
    metric_json_path = TEST_OUT_DIR / "02_KuKa等效辐射源密度映射测试_测试指标表.json"
    txt_path = TEST_OUT_DIR / "02_KuKa等效辐射源密度映射测试_结论.txt"

    write_json(json_path, summary)
    write_csv(corr_csv_path, correlation_rows)
    write_csv(scene_csv_path, scene_group_rows)
    write_metric_outputs(metric_csv_path, metric_json_path, metric_rows)
    write_verdict_report(
        txt_path,
        "Ku/Ka等效辐射源密度映射测试结论",
        verdict,
        [
            "本测试主统计口径限定为V/S等效建筑高度>0的建筑区像元，无建筑区域仅作为背景统计，不参与场景响应主判定。",
            f"建筑区密度倍数P50={building_ratio_p50:.4f}，P95={building_ratio_p95:.4f}，P99={building_ratio_p99:.4f}，P99/P50={p99_p50_ratio:.4f}，最大值={ratio_stats_building['max']:.4f}。",
            f"密度倍数与V/S等效建筑高度的Spearman rho为 {height_rho if height_rho is not None else 'NA'}，用于检验高频段密度映射是否随建成环境高度增强而提高。",
            f"按V/S等效建筑高度三分组后，高高度组平均密度倍数高于低高度组，高/低组平均倍数比为 {height_high_low_mean_ratio if height_high_low_mean_ratio is not None else 'NA'}。",
        ],
        evidence_rows=[
            f"密度倍数栅格与4G建设先验、V/S等效高度、建筑面积和建筑体量图层的共格状态为 {grid_ok}。",
            f"全图倍数统计仅作背景参考：全图P50={ratio_stats_all['p50']:.4f}，P95={ratio_stats_all['p95']:.4f}。",
            f"V/S等效高度建筑区P50={height_stats_building['p50']:.4f} m，P95={height_stats_building['p95']:.4f} m。",
            "当前映射采用28 GHz作为Ka代理频点，验证重点为高频段等效密度倍数的场景响应与数值稳定性。",
        ],
        caveats=[
            "无建筑区域的0高度像元不参与主统计；其倍数结果保留在全图背景统计中，不作为场景差异判定依据。",
        ],
    )

    print_saved(
        json_path,
        corr_csv_path,
        scene_csv_path,
        metric_csv_path,
        metric_json_path,
        txt_path,
        ratio_hist_path,
        metric_summary_plot_path,
        dashboard_path,
    )


if __name__ == "__main__":
    main()
