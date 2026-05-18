# -*- coding: utf-8 -*-
"""
6.1.3 电磁频谱地图测试公共工具。

这些脚本只读取现有成果并生成测试报告，不重新训练模型，也不重算整幅频谱图。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio


PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "1_输入数据").exists())

# 所有测试产物集中放在代码目录之外，避免代码和结果混在一起。
RESULT_ROOT = PROJECT_ROOT / "6_1_3_电磁频谱地图测试结果"
RESULT_ROOT.mkdir(parents=True, exist_ok=True)

# 保留 OUT_DIR 名称，兼容已有脚本；新脚本优先使用 get_test_out_dir。
OUT_DIR = RESULT_ROOT

INPUT_DIR = PROJECT_ROOT / "1_输入数据"
PROCESSED_DIR = INPUT_DIR / "2_已处理数据"
TRAIN_DIR = INPUT_DIR / "3_训练数据"
OUTPUT_DIR = PROJECT_ROOT / "6_输出数据"

ALIGNED_6ARCMIN = PROCESSED_DIR / "aligned_6arcmin"
ALIGNED_1ARCMIN = PROCESSED_DIR / "aligned_1arcmin"

PATHS = {
    "cell_towers_csv": INPUT_DIR / "1_原始数据" / "cell_towers_2026-02-04-T000000.csv",
    "lte_pred_360": OUTPUT_DIR / "Global_LTE_360arcsec_GNN.tif",
    "lte_raw_360": ALIGNED_6ARCMIN / "LTE_count_360arcsec.tif",
    "admin_360": ALIGNED_6ARCMIN / "全球国家行政区_aligned_360arcsec.tif",
    "pop_360": ALIGNED_6ARCMIN / "pop_landscan_2023_aligned_360arcsec.tif",
    "ntl_360": ALIGNED_6ARCMIN / "VNL_npp_2024_global_vcmslcfg_v2_c202502261200.average_masked.dat_aligned_360arcsec.tif",
    "built_v_360": ALIGNED_6ARCMIN / "GHS_BUILT_V_E2025_GLOBE_R2023A_4326_30ss_V1_0_aligned_360arcsec.tif",
    "smod_360": ALIGNED_6ARCMIN / "GHS_SMOD_E2025_GLOBE_R2023A_54009_1000_V2_0_aligned_360arcsec.tif",
    "height_360": ALIGNED_6ARCMIN / "GHS_BUILT_H_AGBH_E2018_GLOBE_R2023A_54009_100_V1_0_aligned_360arcsec.tif",
    "area_360": ALIGNED_6ARCMIN / "GHS_BUILT_S_E2030_GLOBE_R2023A_4326_30ss_V1_0_aligned_360arcsec.tif",
    "ratio_360": PROJECT_ROOT / "4_5G基站倍数预测" / "5g_density_multiplier_result_360arcsec.tif",
    "lte_pred_60": OUTPUT_DIR / "Global_LTE_60arcsec_GNN.tif",
    "admin_60": ALIGNED_1ARCMIN / "全球国家行政区_aligned_60arcsec.tif",
    "height_60": ALIGNED_1ARCMIN / "GHS_BUILT_H_AGBH_E2018_GLOBE_R2023A_54009_100_V1_0_aligned_1km_aligned_60arcsec.tif",
    "area_60": ALIGNED_1ARCMIN / "GHS_BUILT_S_E2030_GLOBE_R2023A_54009_100_V1_0_aligned_60arcsec.tif",
    "ratio": PROJECT_ROOT / "4_5G基站倍数预测" / "5g_density_multiplier_result.tif",
    "rss_60_v2": OUTPUT_DIR / "Global_5G_Radiation_Map_60arcsec_v2.tif",
    "rss_360_v2": OUTPUT_DIR / "Global_5G_Radiation_Map_360arcsec_v2.tif",
    "rss_360": OUTPUT_DIR / "Global_5G_Radiation_Map_360arcsec.tif",
    "itu": INPUT_DIR / "1_原始数据" / "country_traffic_2024_clean.csv",
    "country_shp": INPUT_DIR / "1_原始数据" / "全球行政边界.shp",
    "ookla_raster": PROCESSED_DIR / "Ookla_Silver_Standard_Devices.tif",
}

NATURE_PALETTE = {
    "blue": "#3B6FB6",
    "orange": "#D97B40",
    "green": "#2E8B57",
    "red": "#B94A48",
    "purple": "#7E6AAD",
    "gray": "#6B7280",
    "light_gray": "#E5E7EB",
    "black": "#222222",
}

STATUS_STYLE = {
    "PASS": {"label": "通过", "color": NATURE_PALETTE["green"]},
    "WARN": {"label": "关注", "color": NATURE_PALETTE["orange"]},
    "FAIL": {"label": "未通过", "color": NATURE_PALETTE["red"]},
    "INFO": {"label": "记录", "color": NATURE_PALETTE["gray"]},
}


def existing_path(candidates: Iterable[Path]) -> Path:
    for path in candidates:
        if path.exists() and path.stat().st_size > 0:
            return path
    joined = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(f"未找到可用文件，候选路径如下:\n{joined}")


def read_band(path: Path, masked: bool = False):
    with rasterio.open(path) as src:
        data = src.read(1, masked=masked)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
    return data, profile, transform, crs, nodata


def valid_mask(data, nodata=None):
    arr = np.asarray(data)
    mask = np.isfinite(arr)
    if nodata is not None and np.isfinite(nodata):
        mask &= arr != nodata
    return mask


def finite_values(data, nodata=None):
    arr = np.asarray(data, dtype=np.float64)
    return arr[valid_mask(arr, nodata)]


def describe_array(data, nodata=None, mask=None):
    arr = np.asarray(data, dtype=np.float64)
    m = valid_mask(arr, nodata)
    if mask is not None:
        m &= mask
    vals = arr[m]
    if vals.size == 0:
        return {
            "count": 0,
            "min": None,
            "p01": None,
            "p05": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
            "mean": None,
            "std": None,
        }
    return {
        "count": int(vals.size),
        "min": float(np.nanmin(vals)),
        "p01": float(np.nanpercentile(vals, 1)),
        "p05": float(np.nanpercentile(vals, 5)),
        "p50": float(np.nanpercentile(vals, 50)),
        "p95": float(np.nanpercentile(vals, 95)),
        "p99": float(np.nanpercentile(vals, 99)),
        "max": float(np.nanmax(vals)),
        "mean": float(np.nanmean(vals)),
        "std": float(np.nanstd(vals)),
    }


def grid_info(path: Path):
    with rasterio.open(path) as src:
        return {
            "path": str(path),
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "crs": str(src.crs),
            "transform": tuple(src.transform)[:6],
            "resolution": src.res,
            "bounds": tuple(src.bounds),
            "nodata": src.nodata,
            "dtype": src.dtypes[0],
        }


def same_grid(path_a: Path, path_b: Path) -> bool:
    with rasterio.open(path_a) as a, rasterio.open(path_b) as b:
        return (
            a.crs == b.crs
            and a.transform == b.transform
            and a.width == b.width
            and a.height == b.height
        )


def spearman_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 3:
        return None
    try:
        from scipy.stats import spearmanr

        rho, p = spearmanr(x, y)
        return {"rho": float(rho), "p_value": float(p), "n": int(x.size)}
    except Exception:
        return None


def pearson_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 3:
        return None
    return {"r": float(np.corrcoef(x, y)[0, 1]), "n": int(x.size)}


def sample_pair(x, y, mask=None, max_samples=200_000, seed=42):
    x = np.asarray(x)
    y = np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y)
    if mask is not None:
        m &= mask
    idx = np.flatnonzero(m.ravel())
    if idx.size == 0:
        return np.array([]), np.array([])
    if idx.size > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(idx, max_samples, replace=False)
    return x.ravel()[idx].astype(np.float64), y.ravel()[idx].astype(np.float64)


def write_json(path: Path, obj):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def write_csv(path: Path, rows, fieldnames=None):
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metric_row(section, metric, value, unit="", criterion="", result="", note=""):
    """构造可直接写入测试大纲的量化指标行。"""

    return {
        "section": section,
        "metric": metric,
        "value": value,
        "unit": unit,
        "criterion": criterion,
        "result": result,
        "note": note,
    }


def _format_metric_value(value):
    if value is None:
        return "NA"
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return "NA"
        return f"{float(value):.6g}"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, bool):
        return str(value)
    return str(value)


def write_metric_outputs(csv_path: Path, json_path: Path, rows):
    """同时输出 CSV 和 JSON 格式的关键指标表。"""

    normalized_rows = []
    for row in rows:
        item = dict(row)
        item["value"] = _format_metric_value(item.get("value"))
        normalized_rows.append(item)
    fieldnames = ["section", "metric", "value", "unit", "criterion", "result", "note"]
    write_csv(csv_path, normalized_rows, fieldnames=fieldnames)
    write_json(json_path, normalized_rows)


def write_text(path: Path, lines):
    if isinstance(lines, str):
        text = lines
    else:
        text = "\n".join(str(line) for line in lines)
    path.write_text(text, encoding="utf-8")


def get_test_out_dir(folder_name: str) -> Path:
    """返回某个测试项自己的输出目录。"""
    path = RESULT_ROOT / folder_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_academic_style(plt):
    """统一测试图表风格，采用接近 Nature 子刊的克制学术图风格。"""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": NATURE_PALETTE["black"],
            "axes.linewidth": 0.8,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "savefig.dpi": 450,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_academic_axes(ax, grid_axis="y"):
    """给坐标轴加浅网格与精简边框。"""
    ax.set_axisbelow(True)
    if grid_axis:
        ax.grid(axis=grid_axis, color=NATURE_PALETTE["light_gray"], linestyle="-", linewidth=0.6, alpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_edgecolor(NATURE_PALETTE["black"])
        ax.spines[side].set_linewidth(0.8)


def add_note_box(ax, text, loc=(0.03, 0.97), fontsize=9.5):
    """在图内添加白底说明框。"""
    ax.text(
        loc[0],
        loc[1],
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=fontsize,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.94, edgecolor=NATURE_PALETTE["light_gray"]),
    )


def compact_count(value):
    """把像元数量格式化成更适合图中标注的字符串。"""
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.0f}"


def print_saved(*paths: Path):
    print("输出文件:")
    for path in paths:
        print(f"  - {path}")


def status_from_checks(fail_checks=None, warn_checks=None) -> str:
    """根据布尔检查项给出总体状态。True 表示对应问题存在。"""
    if any(fail_checks or []):
        return "FAIL"
    if any(warn_checks or []):
        return "WARN"
    return "PASS"


def metric_status(value, pass_value, warn_value=None, higher_is_better=True) -> str:
    """把连续指标转成 PASS/WARN/FAIL。"""
    if value is None or not np.isfinite(value):
        return "FAIL"
    if warn_value is None:
        warn_value = pass_value
    if higher_is_better:
        if value >= pass_value:
            return "PASS"
        if value >= warn_value:
            return "WARN"
        return "FAIL"
    if value <= pass_value:
        return "PASS"
    if value <= warn_value:
        return "WARN"
    return "FAIL"


def plot_quality_dashboard(items, title: str, output_path: Path):
    """绘制一页式质量判定总览图。

    items 字段：
    - label: 指标名
    - score: 0-1 的可视化进度值
    - status: PASS/WARN/FAIL/INFO
    - detail: 图中显示的关键数值或解释
    """
    import matplotlib.pyplot as plt

    set_academic_style(plt)
    if not items:
        return

    labels = [item["label"] for item in items]
    scores = [float(np.clip(item.get("score", 0.0), 0.0, 1.0)) for item in items]
    statuses = [item.get("status", "INFO") for item in items]
    details = [item.get("detail", "") for item in items]
    colors = [STATUS_STYLE.get(status, STATUS_STYLE["INFO"])["color"] for status in statuses]

    y = np.arange(len(items))
    fig_height = max(3.2, 0.58 * len(items) + 1.3)
    fig, ax = plt.subplots(figsize=(8.6, fig_height), dpi=450)
    ax.barh(y, [1.0] * len(items), color="#F3F4F6", edgecolor="none", height=0.58, zorder=1)
    ax.barh(y, scores, color=colors, edgecolor="white", linewidth=0.6, height=0.58, zorder=2)

    for i, (score, status, detail) in enumerate(zip(scores, statuses, details)):
        status_text = STATUS_STYLE.get(status, STATUS_STYLE["INFO"])["label"]
        ax.text(0.012, i, status_text, va="center", ha="left", color="white", fontsize=8.5, fontweight="bold", zorder=3)
        ax.text(1.015, i, detail, va="center", ha="left", color=NATURE_PALETTE["black"], fontsize=8.3)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.38)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Quality score")
    ax.set_title(title, pad=10)
    ax.invert_yaxis()
    style_academic_axes(ax, grid_axis="x")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_verdict_report(path: Path, title: str, verdict: str, key_findings, evidence_rows=None, caveats=None):
    """写出更适合直接放入测试大纲的结论文本。"""
    style = STATUS_STYLE.get(verdict, STATUS_STYLE["INFO"])
    lines = [
        f"{title}",
        f"总体判定：{style['label']}（{verdict}）",
        "",
        "关键结论：",
    ]
    lines.extend(f"{idx}. {line}" for idx, line in enumerate(key_findings, start=1))
    if evidence_rows:
        lines.extend(["", "关键证据："])
        lines.extend(f"- {row}" for row in evidence_rows)
    if caveats:
        lines.extend(["", "需关注事项："])
        lines.extend(f"- {row}" for row in caveats)
    write_text(path, lines)
