# -*- coding: utf-8 -*-
"""
6.1.3.1 4G 建设先验补全结果测试

本脚本对应测试大纲中的“4G 建设先验补全结果测试”。

本项目中 4G/LTE 基站数据存在三个层级：
1. 原始点数据：
   1_输入数据/1_原始数据/cell_towers_2026-02-04-T000000.csv
   这是 OpenCellID 原始基站点表，包含 lon、lat、radio 等字段。

2. 处理后栅格数据：
   1_输入数据/2_已处理数据/aligned_6arcmin/LTE_count_360arcsec.tif
   这是将原始点数据筛选 LTE 后，按 6 arcmin 目标网格聚合得到的原始 4G 计数栅格。

3. 模型预测/补全结果：
   6_输出数据/Global_LTE_360arcsec_GNN.tif
   这是 GraphSAGE 模型在 6 arcmin 全球网格上推理得到的 4G 基站密度/数量先验。

测试重点：
- 检查三类数据是否存在，记录原始点数据规模与 LTE 样本规模；
- 检查处理后 LTE_count_360arcsec 与 GNN 预测结果是否严格共格；
- 统计“原始 LTE 栅格为 0 或无站，但 GNN 预测为正”的像元，量化补全作用；
- 检查 GNN 预测结果与人口、夜间灯光、建成环境、城市化等级等辅助图层的空间一致性；
- 输出 JSON/CSV/TXT 报告，供测试大纲“测试结果”和“测试结论”填写使用。
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import (
    NATURE_PALETTE,
    PATHS,
    add_note_box,
    compact_count,
    describe_array,
    get_test_out_dir,
    grid_info,
    metric_row,
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

TEST_OUT_DIR = get_test_out_dir("01_4G建设先验补全结果测试")


# 为了避免原始 OpenCellID CSV 很大时一次性读入内存，这里使用 csv 模块流式扫描。
# 该函数只统计行数、LTE 行数和经纬度有效性，不做任何重处理或重栅格化。
def summarize_cell_towers_csv(csv_path: Path) -> dict:
    """统计原始 OpenCellID 基站点 CSV 的基础信息。

    返回字段说明：
    - total_rows: 原始点表总记录数；
    - lte_rows: radio 字段为 LTE 的记录数；
    - lte_valid_lonlat_rows: LTE 中经纬度可解析且位于合法范围内的记录数；
    - columns: CSV 表头字段，用于证明 lon/lat/radio 等关键字段存在。
    """

    if not csv_path.exists():
        return {
            "path": str(csv_path),
            "exists": False,
            "total_rows": 0,
            "lte_rows": 0,
            "lte_valid_lonlat_rows": 0,
            "columns": [],
            "note": "原始基站 CSV 不存在，请检查路径。",
        }

    total_rows = 0
    lte_rows = 0
    lte_valid_lonlat_rows = 0

    with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []

        required_columns = {"lon", "lat", "radio"}
        missing_columns = sorted(required_columns - set(columns))
        if missing_columns:
            return {
                "path": str(csv_path),
                "exists": True,
                "total_rows": 0,
                "lte_rows": 0,
                "lte_valid_lonlat_rows": 0,
                "columns": columns,
                "missing_columns": missing_columns,
                "note": "CSV 缺少关键字段，无法统计 LTE 点。",
            }

        for row in reader:
            total_rows += 1

            # OpenCellID 中 radio 字段标记网络制式，本项目只取 LTE 作为 4G 基站。
            if str(row.get("radio", "")).strip().upper() != "LTE":
                continue

            lte_rows += 1

            # 检查经纬度是否可解析、是否处于 WGS84 合法范围。
            try:
                lon = float(row["lon"])
                lat = float(row["lat"])
            except (TypeError, ValueError):
                continue

            if -180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0:
                lte_valid_lonlat_rows += 1

    return {
        "path": str(csv_path),
        "exists": True,
        "total_rows": total_rows,
        "lte_rows": lte_rows,
        "lte_valid_lonlat_rows": lte_valid_lonlat_rows,
        "columns": columns,
        "missing_columns": [],
        "note": "原始 OpenCellID 点数据统计完成。",
    }


def build_land_and_valid_masks(pred, pred_nodata, processed_lte, processed_lte_nodata, admin):
    """构建后续统计要用的掩膜。

    admin > 0 表示陆地区域或有国家归属区域；
    pred_valid 表示 GNN 预测结果在陆地区域内有效；
    processed_lte_valid 表示处理后的 LTE_count_360arcsec 在陆地区域内有效。
    """

    land_mask = admin > 0

    pred_valid = land_mask & np.isfinite(pred)
    if pred_nodata is not None and np.isfinite(pred_nodata):
        pred_valid &= pred != pred_nodata

    processed_lte_valid = land_mask & np.isfinite(processed_lte)
    if processed_lte_nodata is not None and np.isfinite(processed_lte_nodata):
        processed_lte_valid &= processed_lte != processed_lte_nodata

    return land_mask, pred_valid, processed_lte_valid


def _finite_positive_mask(data, nodata=None):
    """返回有效且大于 0 的像元掩膜。"""

    arr = np.asarray(data)
    mask = np.isfinite(arr)
    if nodata is not None and np.isfinite(nodata):
        mask &= arr != nodata
    return mask & (arr > 0)


def _normalized_log_signal(data, nodata, base_mask):
    """把长尾人类活动图层转成 0-1 需求强度分数。"""

    arr = np.asarray(data, dtype=np.float64)
    valid = np.isfinite(arr)
    if nodata is not None and np.isfinite(nodata):
        valid &= arr != nodata
    with np.errstate(invalid="ignore"):
        signal = np.where(valid & (arr > 0), np.log1p(arr), 0.0)
    positive = signal[base_mask & (signal > 0)]
    if positive.size == 0:
        return np.zeros_like(signal, dtype=np.float64)
    scale = max(float(np.percentile(positive, 95)), 1e-12)
    return np.clip(signal / scale, 0.0, 1.0)


def build_demand_support_mask(base_mask):
    """构建有人类活动支撑的陆地区域掩膜。

    公开 LTE 缺失不一定代表模型应该补站。沙漠、荒原、无人山地等区域即使没有
    LTE 记录，也不应作为补全率分母。因此这里用人口、夜间灯光、建筑体量和
    建筑体量共同定义“存在通信需求支撑”的区域。SMOD 城市化等级容易把
    很多低密度背景陆地纳入分母，因此只作为辅助一致性检查，不放入补全率主分母。
    """

    pop, _, _, _, pop_nodata = read_band(PATHS["pop_360"])
    ntl, _, _, _, ntl_nodata = read_band(PATHS["ntl_360"])
    built_v, _, _, _, built_nodata = read_band(PATHS["built_v_360"])
    smod, _, _, _, smod_nodata = read_band(PATHS["smod_360"])

    pop_signal = _finite_positive_mask(pop, pop_nodata)
    ntl_signal = _finite_positive_mask(ntl, ntl_nodata)
    built_signal = _finite_positive_mask(built_v, built_nodata)

    smod_valid = np.isfinite(smod)
    if smod_nodata is not None and np.isfinite(smod_nodata):
        smod_valid &= smod != smod_nodata
    # GHS-SMOD 中 10 通常代表低密度/农村背景；>10 更明确表示有人类定居等级增强。
    smod_signal = smod_valid & (smod > 10)

    demand_score = np.maximum.reduce(
        [
            _normalized_log_signal(pop, pop_nodata, base_mask),
            _normalized_log_signal(ntl, ntl_nodata, base_mask),
            _normalized_log_signal(built_v, built_nodata, base_mask),
        ]
    )
    demand_support = base_mask & (pop_signal | ntl_signal | built_signal)
    rows = [
        {"signal": "population_positive", "pixel_count": int(np.sum(base_mask & pop_signal))},
        {"signal": "nighttime_light_positive", "pixel_count": int(np.sum(base_mask & ntl_signal))},
        {"signal": "building_volume_positive", "pixel_count": int(np.sum(base_mask & built_signal))},
        {
            "signal": "settlement_level_gt_10_reference_only",
            "pixel_count": int(np.sum(base_mask & smod_signal)),
        },
        {"signal": "union_demand_supported", "pixel_count": int(np.sum(demand_support))},
    ]
    return demand_support, rows, demand_score


def calculate_completion_metrics(
    pred,
    processed_lte,
    pred_valid,
    processed_lte_valid,
    demand_support_mask=None,
    demand_score=None,
    priority_percentile=75,
):
    """计算“补全效果”相关统计。

    这里的核心不是重新判断模型好坏，而是为测试大纲提供可量化证据：
    - 原始处理后 LTE 栅格中有站的像元数量；
    - GNN 预测为正的像元数量；
    - 有通信需求支撑的陆地区域内，原始 LTE 为 0、负值、NoData 或无有效观测，
      但 GNN 预测为正的像元数量。

    第三项可理解为模型对“应该有通信建设可能”的公开基站数据采样稀疏区域的
    补全覆盖，而不是把沙漠、无人区等不应建站区域纳入分母。
    同时额外输出“重点需求区”补全率：在需求支撑稀疏区内按人口/灯光/建筑体量
    综合需求分数取前 25%（默认 percentile=75）作为更接近测试判读的主指标。
    注意：当前 LTE_count_360arcsec.tif 的无站区域并不一定写成有效 0 值，
    很多区域会以 NoData/无效像元形式存在。因此“稀疏区”不能只用
    processed_lte_valid & (processed_lte <= 0)，否则会把 NoData 缺站区排除掉。
    这里默认再叠加 demand_support_mask；若未传入，则退化为仅使用 pred_valid。
    """

    processed_lte_positive = processed_lte_valid & (processed_lte > 0)
    pred_positive = pred_valid & (pred > 0)
    evaluation_mask = pred_valid if demand_support_mask is None else (pred_valid & demand_support_mask)

    processed_lte_valid_non_positive = evaluation_mask & processed_lte_valid & (processed_lte <= 0)
    processed_lte_missing_or_invalid = evaluation_mask & ~processed_lte_valid
    processed_lte_zero_or_empty = evaluation_mask & ~processed_lte_positive
    completed_pixels = processed_lte_zero_or_empty & pred_positive

    land_zero_or_empty = pred_valid & ~processed_lte_positive
    land_completed_pixels = land_zero_or_empty & pred_positive

    priority_threshold = None
    priority_sparse_area = np.zeros_like(pred_valid, dtype=bool)
    priority_completed_pixels = np.zeros_like(pred_valid, dtype=bool)
    if demand_score is not None and np.sum(processed_lte_zero_or_empty) > 0:
        scores = np.asarray(demand_score, dtype=np.float64)
        candidate_scores = scores[processed_lte_zero_or_empty & np.isfinite(scores)]
        if candidate_scores.size > 0:
            priority_threshold = float(np.percentile(candidate_scores, priority_percentile))
            priority_sparse_area = processed_lte_zero_or_empty & (scores >= priority_threshold)
            priority_completed_pixels = priority_sparse_area & pred_positive

    return {
        "processed_lte_positive_count": int(np.sum(processed_lte_positive)),
        "pred_positive_count": int(np.sum(pred_positive)),
        "completion_denominator_scope": "demand_supported_land_sparse_area",
        "demand_supported_pixel_count": int(np.sum(evaluation_mask)),
        "processed_lte_valid_non_positive_count": int(np.sum(processed_lte_valid_non_positive)),
        "processed_lte_missing_or_invalid_count": int(np.sum(processed_lte_missing_or_invalid)),
        "processed_lte_zero_or_empty_count": int(np.sum(processed_lte_zero_or_empty)),
        "completed_pixels_count": int(np.sum(completed_pixels)),
        "completed_pixels_ratio_in_lte_zero_or_empty": float(
            np.sum(completed_pixels) / max(np.sum(processed_lte_zero_or_empty), 1)
        ),
        "priority_demand_percentile": priority_percentile,
        "priority_demand_threshold": priority_threshold,
        "priority_sparse_area_count": int(np.sum(priority_sparse_area)),
        "priority_completed_pixels_count": int(np.sum(priority_completed_pixels)),
        "priority_completion_ratio": float(
            np.sum(priority_completed_pixels) / max(np.sum(priority_sparse_area), 1)
        ),
        "land_sparse_area_reference_count": int(np.sum(land_zero_or_empty)),
        "land_sparse_area_completed_count": int(np.sum(land_completed_pixels)),
        "land_sparse_area_completion_ratio_reference": float(
            np.sum(land_completed_pixels) / max(np.sum(land_zero_or_empty), 1)
        ),
    }


def calculate_neighbor_continuity(arr, mask):
    """计算相邻像元差异，用于辅助判断结果是否存在明显空间断裂。

    GraphSAGE 本身利用四邻域图结构建模，因此预测结果应具有一定空间连续性。
    这里分别统计水平和垂直相邻像元的绝对差均值与 95 分位数。
    """

    arr = np.asarray(arr, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)

    neighbor_pairs = [
        ("horizontal", arr[:, 1:], arr[:, :-1], mask[:, 1:] & mask[:, :-1]),
        ("vertical", arr[1:, :], arr[:-1, :], mask[1:, :] & mask[:-1, :]),
    ]

    metrics = {}
    for name, a, b, valid_pair_mask in neighbor_pairs:
        diff = np.abs(a[valid_pair_mask] - b[valid_pair_mask])
        if diff.size == 0:
            metrics[name] = None
            continue

        metrics[name] = {
            "pair_count": int(diff.size),
            "mean_abs_neighbor_diff": float(np.mean(diff)),
            "p95_abs_neighbor_diff": float(np.percentile(diff, 95)),
        }

    return metrics


def calculate_auxiliary_consistency(pred, pred_valid, lte_pred_path):
    """计算 GNN 预测结果与辅助图层之间的空间相关性。

    这里使用 Spearman 秩相关，不要求两个变量线性同量纲，只检查空间排序关系。
    例如：预测高值区是否大体对应人口更高、夜间灯光更强、建成环境更强的区域。
    """

    auxiliary_layers = {
        "POP_population": PATHS["pop_360"],
        "NTL_nighttime_light": PATHS["ntl_360"],
        "BUILT_V_building_volume": PATHS["built_v_360"],
        "SMOD_settlement_model": PATHS["smod_360"],
    }

    rows = []
    for layer_name, layer_path in auxiliary_layers.items():
        aux, _, _, _, aux_nodata = read_band(layer_path)

        # sample_pair 会在有效区域中最多抽样 200000 个像元，避免全球大栅格全量相关性计算过慢。
        pred_sample, aux_sample = sample_pair(pred, aux, mask=pred_valid, max_samples=200_000)
        corr = spearman_corr(pred_sample, aux_sample)

        rows.append(
            {
                "aux_layer": layer_name,
                "aux_path": str(layer_path),
                "same_grid_with_gnn_pred": same_grid(lte_pred_path, layer_path),
                "spearman_rho": None if corr is None else corr["rho"],
                "p_value": None if corr is None else corr["p_value"],
                "sample_count": 0 if corr is None else corr["n"],
                "aux_valid_count_on_land": describe_array(aux, aux_nodata)["count"],
            }
        )

    return rows


def plot_completion_metrics(completion_metrics, output_path: Path):
    """绘制补全效果柱状图。"""
    set_academic_style(plt)
    labels = [
        "Observed LTE\ncount > 0",
        "GNN predicted\ncount > 0",
        "Filled by GNN\nin demand-supported sparse land",
        "Filled by GNN\nin priority-demand sparse land",
    ]
    values = [
        completion_metrics["processed_lte_positive_count"],
        completion_metrics["pred_positive_count"],
        completion_metrics["completed_pixels_count"],
        completion_metrics["priority_completed_pixels_count"],
    ]

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    bars = ax.bar(
        labels,
        values,
        color=[NATURE_PALETTE["gray"], NATURE_PALETTE["blue"], NATURE_PALETTE["orange"], NATURE_PALETTE["green"]],
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    ax.set_ylabel("Land grid pixels")
    ax.set_title("4G Construction Prior Completion", pad=10)
    style_academic_axes(ax, grid_axis="y")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), compact_count(value), ha="center", va="bottom", fontsize=9)
    fill_ratio = completion_metrics["priority_completion_ratio"] * 100
    add_note_box(
        ax,
        "The green bar focuses on the top 25% demand-supported\n"
        "sparse land pixels by population/light/building signal.\n"
        f"Priority-demand fill ratio: {fill_ratio:.1f}%",
        loc=(0.03, 0.95),
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_auxiliary_correlations(rows, output_path: Path):
    """绘制预测结果与辅助图层的 Spearman 相关性。"""
    set_academic_style(plt)
    valid_rows = [r for r in rows if r["spearman_rho"] is not None]
    if not valid_rows:
        return

    label_map = {
        "POP_population": "Population",
        "NTL_nighttime_light": "Nighttime light",
        "BUILT_V_building_volume": "Building volume",
        "SMOD_settlement_model": "Settlement level",
    }
    labels = [label_map.get(r["aux_layer"], r["aux_layer"]) for r in valid_rows]
    values = [r["spearman_rho"] for r in valid_rows]

    order = np.argsort(values)
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]

    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)
    colors = [NATURE_PALETTE["blue"] if v >= 0 else NATURE_PALETTE["red"] for v in values]
    bars = ax.barh(labels, values, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    ax.axvline(0, color=NATURE_PALETTE["black"], linewidth=0.8)
    ax.set_xlabel("Spearman rho")
    ax.set_title("Spatial Consistency: GNN 4G Prior vs. Auxiliary Layers", pad=10)
    ax.set_xlim(min(-0.1, min(values) - 0.08), max(0.1, max(values) + 0.08))
    style_academic_axes(ax, grid_axis="x")
    for bar, value in zip(bars, values):
        ax.text(value, bar.get_y() + bar.get_height() / 2, f" {value:.3f}", va="center", ha="left" if value >= 0 else "right", fontsize=8.8)
    add_note_box(ax, "Positive rho means the 4G prior increases with\nhuman activity or built-environment intensity.", loc=(0.03, 0.96), fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    # -----------------------------
    # 1. 明确三类核心输入/输出路径
    # -----------------------------
    raw_point_csv_path = PATHS["cell_towers_csv"]
    processed_lte_count_grid_path = PATHS["lte_raw_360"]
    gnn_predicted_grid_path = PATHS["lte_pred_360"]
    admin_mask_path = PATHS["admin_360"]

    # -----------------------------
    # 2. 读取原始点数据统计信息
    # -----------------------------
    raw_point_csv_summary = summarize_cell_towers_csv(raw_point_csv_path)

    # -----------------------------
    # 3. 读取处理后 LTE 栅格、GNN 预测栅格、行政区陆地掩膜
    # -----------------------------
    pred, _, _, _, pred_nodata = read_band(gnn_predicted_grid_path)
    processed_lte, _, _, _, processed_lte_nodata = read_band(processed_lte_count_grid_path)
    admin, _, _, _, _ = read_band(admin_mask_path)

    land_mask, pred_valid, processed_lte_valid = build_land_and_valid_masks(
        pred=pred,
        pred_nodata=pred_nodata,
        processed_lte=processed_lte,
        processed_lte_nodata=processed_lte_nodata,
        admin=admin,
    )
    demand_support_mask, demand_support_rows, demand_score = build_demand_support_mask(pred_valid)

    # -----------------------------
    # 4. 计算补全效果、空间连续性和辅助图层一致性
    # -----------------------------
    completion_metrics = calculate_completion_metrics(
        pred=pred,
        processed_lte=processed_lte,
        pred_valid=pred_valid,
        processed_lte_valid=processed_lte_valid,
        demand_support_mask=demand_support_mask,
        demand_score=demand_score,
        priority_percentile=75,
    )

    continuity_metrics = calculate_neighbor_continuity(pred, pred_valid)

    auxiliary_consistency_rows = calculate_auxiliary_consistency(
        pred=pred,
        pred_valid=pred_valid,
        lte_pred_path=gnn_predicted_grid_path,
    )

    grid_ok = same_grid(processed_lte_count_grid_path, gnn_predicted_grid_path) and same_grid(
        admin_mask_path,
        gnn_predicted_grid_path,
    )
    valid_aux = [r for r in auxiliary_consistency_rows if r["spearman_rho"] is not None]
    positive_aux = [r for r in valid_aux if r["spearman_rho"] > 0.10]
    fill_ratio = completion_metrics["completed_pixels_ratio_in_lte_zero_or_empty"]
    priority_fill_ratio = completion_metrics["priority_completion_ratio"]
    valid_ratio = summary_valid_ratio = float(np.sum(pred_valid) / max(np.sum(land_mask), 1))
    verdict = status_from_checks(
        fail_checks=[
            not raw_point_csv_summary["exists"],
            not grid_ok,
            summary_valid_ratio <= 0,
        ],
        warn_checks=[
            completion_metrics["priority_completed_pixels_count"] <= 0,
            priority_fill_ratio < 0.50,
            len(positive_aux) < max(1, len(valid_aux) // 2),
        ],
    )
    quality_items = [
        {
            "label": "核心栅格共格",
            "score": 1.0 if grid_ok else 0.0,
            "status": "PASS" if grid_ok else "FAIL",
            "detail": f"LTE/GNN/Admin 对齐：{grid_ok}",
        },
        {
            "label": "陆地有效覆盖",
            "score": valid_ratio,
            "status": "PASS" if valid_ratio > 0.95 else "WARN" if valid_ratio > 0 else "FAIL",
            "detail": f"{valid_ratio * 100:.1f}% 陆地像元有有效预测",
        },
        {
            "label": "重点需求稀疏区补全",
            "score": min(priority_fill_ratio, 1.0),
            "status": "PASS" if priority_fill_ratio >= 0.60 else "WARN" if priority_fill_ratio >= 0.50 else "FAIL",
            "detail": (
                f"{completion_metrics['priority_completed_pixels_count']:,} 像元；"
                f"补全率 {priority_fill_ratio * 100:.1f}%"
            ),
        },
        {
            "label": "辅助图层一致性",
            "score": len(positive_aux) / max(len(valid_aux), 1),
            "status": "PASS" if len(positive_aux) >= 3 else "WARN" if len(positive_aux) >= 1 else "FAIL",
            "detail": f"{len(positive_aux)}/{len(valid_aux)} 个辅助层 rho > 0.10",
        },
    ]
    pop_rho = next((r["spearman_rho"] for r in auxiliary_consistency_rows if r["aux_layer"] == "POP_population"), None)
    ntl_rho = next((r["spearman_rho"] for r in auxiliary_consistency_rows if r["aux_layer"] == "NTL_nighttime_light"), None)
    built_rho = next((r["spearman_rho"] for r in auxiliary_consistency_rows if r["aux_layer"] == "BUILT_V_building_volume"), None)
    smod_rho = next((r["spearman_rho"] for r in auxiliary_consistency_rows if r["aux_layer"] == "SMOD_settlement_model"), None)
    metric_rows = [
        metric_row("6.1.3.1", "原始OpenCellID LTE记录数", raw_point_csv_summary["lte_rows"], "条", "记录测试输入规模", "记录"),
        metric_row("6.1.3.1", "经纬度有效LTE记录数", raw_point_csv_summary["lte_valid_lonlat_rows"], "条", "记录可用样本规模", "记录"),
        metric_row("6.1.3.1", "LTE计数栅格与GNN预测栅格共格", grid_ok, "-", "应为True", "PASS" if grid_ok else "FAIL"),
        metric_row("6.1.3.1", "GNN陆地有效覆盖率", valid_ratio * 100, "%", "建议>95%", "PASS" if valid_ratio > 0.95 else "WARN"),
        metric_row("6.1.3.1", "需求支撑稀疏区补全像元数", completion_metrics["completed_pixels_count"], "个", "记录补全规模", "记录"),
        metric_row("6.1.3.1", "需求支撑稀疏区补全率", fill_ratio * 100, "%", "背景参考指标", "记录"),
        metric_row("6.1.3.1", "重点需求稀疏区补全像元数", completion_metrics["priority_completed_pixels_count"], "个", "应>0", "PASS" if completion_metrics["priority_completed_pixels_count"] > 0 else "FAIL"),
        metric_row("6.1.3.1", "重点需求稀疏区补全率", priority_fill_ratio * 100, "%", "建议>=60%", "PASS" if priority_fill_ratio >= 0.60 else "WARN"),
        metric_row("6.1.3.1", "人口与4G先验Spearman相关", pop_rho, "-", "应为正相关", "PASS" if pop_rho is not None and pop_rho > 0.10 else "WARN"),
        metric_row("6.1.3.1", "夜间灯光与4G先验Spearman相关", ntl_rho, "-", "应为正相关", "PASS" if ntl_rho is not None and ntl_rho > 0.10 else "WARN"),
        metric_row("6.1.3.1", "建筑体量与4G先验Spearman相关", built_rho, "-", "应为正相关", "PASS" if built_rho is not None and built_rho > 0.10 else "WARN"),
        metric_row("6.1.3.1", "城市化等级与4G先验Spearman相关", smod_rho, "-", "应为正相关", "PASS" if smod_rho is not None and smod_rho > 0.10 else "WARN"),
        metric_row("6.1.3.1", "总体判定", verdict, "-", "PASS/WARN/FAIL", verdict),
    ]

    # -----------------------------
    # 5. 汇总 JSON 报告
    # -----------------------------
    summary = {
        "test_name": "4G建设先验补全结果测试",
        "data_lineage": {
            "raw_point_csv": raw_point_csv_summary,
            "processed_lte_count_grid": grid_info(processed_lte_count_grid_path),
            "gnn_predicted_grid": grid_info(gnn_predicted_grid_path),
            "admin_mask_grid": grid_info(admin_mask_path),
        },
        "grid_alignment": {
            "processed_lte_count_grid_vs_gnn_predicted_grid": same_grid(
                processed_lte_count_grid_path,
                gnn_predicted_grid_path,
            ),
            "admin_mask_grid_vs_gnn_predicted_grid": same_grid(
                admin_mask_path,
                gnn_predicted_grid_path,
            ),
        },
        "statistics": {
            "land_pixel_count": int(np.sum(land_mask)),
            "gnn_pred_stats_on_land": describe_array(pred, pred_nodata, land_mask),
            "processed_lte_count_stats_on_land": describe_array(
                processed_lte,
                processed_lte_nodata,
                land_mask,
            ),
        },
        "completion_metrics": completion_metrics,
        "demand_support_definition": {
            "scope": "陆地且GNN有效预测区域内，人口>0、夜间灯光>0或建筑体量>0任一条件成立；SMOD仅作辅助参考，不进入补全率主分母",
            "priority_scope": "在需求支撑稀疏区中，按人口/夜间灯光/建筑体量综合需求分数取前25%作为重点需求区主判定口径",
            "component_pixel_counts": demand_support_rows,
        },
        "neighbor_continuity": continuity_metrics,
        "auxiliary_spatial_consistency": auxiliary_consistency_rows,
        "quality_assessment": {
            "verdict": verdict,
            "quality_items": quality_items,
        },
        "key_metrics_for_test_outline": metric_rows,
    }

    # -----------------------------
    # 6. 输出测试结果文件
    # -----------------------------
    completion_plot_path = TEST_OUT_DIR / "01_补全效果柱状图.png"
    aux_corr_plot_path = TEST_OUT_DIR / "01_辅助图层相关性.png"
    dashboard_path = TEST_OUT_DIR / "01_测试质量总览.png"
    plot_completion_metrics(completion_metrics, completion_plot_path)
    plot_auxiliary_correlations(auxiliary_consistency_rows, aux_corr_plot_path)
    plot_quality_dashboard(quality_items, "4G Prior Completion Test: Quality Overview", dashboard_path)

    summary["visual_outputs"] = {
        "completion_metrics_bar": str(completion_plot_path),
        "auxiliary_correlations_bar": str(aux_corr_plot_path),
        "quality_dashboard": str(dashboard_path),
    }

    json_path = TEST_OUT_DIR / "01_4G建设先验补全结果测试_summary.json"
    csv_path = TEST_OUT_DIR / "01_4G建设先验补全结果测试_辅助图层相关性.csv"
    metric_csv_path = TEST_OUT_DIR / "01_4G建设先验补全结果测试_测试指标表.csv"
    metric_json_path = TEST_OUT_DIR / "01_4G建设先验补全结果测试_测试指标表.json"
    txt_path = TEST_OUT_DIR / "01_4G建设先验补全结果测试_结论.txt"

    write_json(json_path, summary)
    write_csv(csv_path, auxiliary_consistency_rows)
    write_metric_outputs(metric_csv_path, metric_json_path, metric_rows)

    conclusion_lines = [
        "4G建设先验补全结果测试结论建议：",
        f"1. 原始点数据为 {raw_point_csv_path.name}，LTE 记录数为 {raw_point_csv_summary['lte_rows']:,}，其中经纬度有效 LTE 记录数为 {raw_point_csv_summary['lte_valid_lonlat_rows']:,}。",
        f"2. 处理后 LTE 计数栅格为 {processed_lte_count_grid_path.name}，GNN 预测结果为 {gnn_predicted_grid_path.name}。",
        f"3. 处理后 LTE 栅格与 GNN 预测栅格是否完全共格：{summary['grid_alignment']['processed_lte_count_grid_vs_gnn_predicted_grid']}。",
        f"4. GNN 预测结果陆地有效像元数为 {summary['statistics']['gnn_pred_stats_on_land']['count']:,}。",
        f"5. 有通信需求支撑的陆地区域内，处理后 LTE 为 0 或无站、但 GNN 预测为正的像元数为 {completion_metrics['completed_pixels_count']:,}，可作为采样稀疏区域补全效果的量化记录。",
        "6. 辅助图层 Spearman 相关性已输出到 CSV，可用于判断预测高值区与人口、夜间灯光、建成环境和城市化等级的空间一致性。",
        "7. 若典型区域图件中未出现明显拼块、条带或非物理斑块，且高值区主要对应城市与人口密集区，则该项可判定通过。",
    ]
    write_verdict_report(
        txt_path,
        "4G建设先验补全结果测试结论",
        verdict,
        [
            f"GraphSAGE 4G建设先验与原始LTE栅格、行政区掩膜的共格状态为 {grid_ok}。",
            f"在重点需求稀疏区（需求强度前25%）中，GNN给出正值补全的像元为 {completion_metrics['priority_completed_pixels_count']:,} 个，补全率为 {priority_fill_ratio * 100:.1f}%。",
            f"宽口径需求支撑稀疏区补全率为 {fill_ratio * 100:.1f}%，作为背景参考。",
            f"辅助图层一致性方面，{len(positive_aux)}/{len(valid_aux)} 个辅助图层与4G先验呈明确正相关（Spearman rho > 0.10）。",
        ],
        evidence_rows=[
            f"原始OpenCellID中LTE记录数：{raw_point_csv_summary['lte_rows']:,}，经纬度有效LTE记录数：{raw_point_csv_summary['lte_valid_lonlat_rows']:,}。",
            f"GNN陆地有效像元数：{summary['statistics']['gnn_pred_stats_on_land']['count']:,}。",
            f"重点需求稀疏区候选像元数为 {completion_metrics['priority_sparse_area_count']:,}；主判定口径已排除低需求无人区。",
            f"宽口径需求支撑稀疏区由人口>0、夜间灯光>0或建筑体量>0任一条件确定；候选稀疏区像元数为 {completion_metrics['processed_lte_zero_or_empty_count']:,}。",
            f"全陆地缺站区参考补全率为 {completion_metrics['land_sparse_area_completion_ratio_reference'] * 100:.1f}%，仅作对照，不作为主判定。",
            "该项主要检验4G建设强度先验是否连续、可解释，并能补全公开基站数据采样稀疏区。",
        ],
        caveats=[
            "该测试不等同于逐点真实基站校验，重点是作为Ku/Ka频谱地图上游建设强度先验的合理性检查。",
        ],
    )

    print_saved(json_path, csv_path, metric_csv_path, metric_json_path, txt_path, completion_plot_path, aux_corr_plot_path, dashboard_path)


if __name__ == "__main__":
    main()
