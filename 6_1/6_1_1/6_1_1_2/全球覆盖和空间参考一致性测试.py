from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import rasterio
except ImportError as exc:  # pragma: no cover - dependency check
    print(f"缺少依赖 rasterio，无法执行检查：{exc}", file=sys.stderr)
    sys.exit(2)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEMAND_DIR = REPO_ROOT / "dataset" / "products" / "demand"
CORE_FILE = DEMAND_DIR / "pred_final_pred_masked.tif"
METADATA_FILE = DEMAND_DIR / "metadata.json"
TOL = 1e-9


@dataclass
class CheckResult:
    name: str
    status: str
    details: str


def almost_equal(a: float, b: float, tol: float = TOL) -> bool:
    return math.isclose(a, b, rel_tol=0.0, abs_tol=tol)


def dict_float_equal(left: dict[str, float], right: dict[str, float]) -> bool:
    return all(key in right and almost_equal(float(left[key]), float(right[key])) for key in left)


def raster_summary(path: Path) -> dict[str, object]:
    with rasterio.open(path) as ds:
        crs_text = str(ds.crs) if ds.crs else None
        if ds.crs is None:
            projection_type = None
        elif ds.crs.is_geographic:
            projection_type = "geographic"
        elif ds.crs.is_projected:
            projection_type = "projected"
        else:
            projection_type = "other"

        return {
            "path": path,
            "name": path.name,
            "driver": ds.driver,
            "crs": crs_text,
            "projection_type": projection_type,
            "width": ds.width,
            "height": ds.height,
            "resolution": {"x": ds.res[0], "y": ds.res[1]},
            "bounds": {
                "left": ds.bounds.left,
                "bottom": ds.bounds.bottom,
                "right": ds.bounds.right,
                "top": ds.bounds.top,
            },
            "transform": tuple(ds.transform)[:6],
        }


def load_metadata(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def compare_core_with_metadata(core: dict[str, object], metadata: dict[str, object]) -> list[str]:
    problems: list[str] = []
    if metadata.get("crs") != core["crs"]:
        problems.append(f"crs 应为 {core['crs']}")
    if metadata.get("width") != core["width"]:
        problems.append(f"width 应为 {core['width']}")
    if metadata.get("height") != core["height"]:
        problems.append(f"height 应为 {core['height']}")
    if metadata.get("resolution") != core["resolution"]:
        problems.append("resolution 与栅格实际值不一致")
    if metadata.get("bounds") != core["bounds"]:
        problems.append("bounds 与栅格实际值不一致")
    return problems


def build_results() -> tuple[list[CheckResult], dict[str, object]]:
    results: list[CheckResult] = []
    context: dict[str, object] = {"demand_dir": DEMAND_DIR}

    results.append(CheckResult("demand 目录存在", "通过" if DEMAND_DIR.is_dir() else "未通过", f"目录路径：{DEMAND_DIR}"))
    results.append(CheckResult("核心栅格存在", "通过" if CORE_FILE.is_file() else "未通过", f"核心文件：{CORE_FILE}"))
    results.append(CheckResult("元数据文件存在", "通过" if METADATA_FILE.is_file() else "未通过", f"元数据文件：{METADATA_FILE}"))

    if not CORE_FILE.is_file():
        return results, context

    core = raster_summary(CORE_FILE)
    context["core"] = core

    lon_span = float(core["bounds"]["right"]) - float(core["bounds"]["left"])  # type: ignore[index]
    global_cover_ok = (
        almost_equal(float(core["bounds"]["left"]), -180.0)  # type: ignore[index]
        and almost_equal(float(core["bounds"]["right"]), 180.0)  # type: ignore[index]
        and almost_equal(lon_span, 360.0)
        and float(core["bounds"]["bottom"]) >= -90.0  # type: ignore[index]
        and float(core["bounds"]["top"]) <= 90.0  # type: ignore[index]
    )
    results.append(
        CheckResult(
            "覆盖全球研究范围",
            "通过" if global_cover_ok else "未通过",
            (
                f"bounds={core['bounds']}，经度跨度={lon_span}"
                if global_cover_ok
                else f"当前 bounds={core['bounds']}，未满足全球经度全覆盖或纬度范围有效性要求"
            ),
        )
    )

    crs_ok = core["crs"] is not None
    results.append(
        CheckResult(
            "坐标参考系统有效",
            "通过" if crs_ok else "未通过",
            f"CRS={core['crs']}",
        )
    )

    projection_ok = core["projection_type"] in {"geographic", "projected"}
    projection_label = "地理坐标系" if core["projection_type"] == "geographic" else "投影坐标系"
    if core["projection_type"] not in {"geographic", "projected"}:
        projection_label = str(core["projection_type"])
    results.append(
        CheckResult(
            "投影信息有效",
            "通过" if projection_ok else "未通过",
            f"projection_type={projection_label}",
        )
    )

    if METADATA_FILE.is_file():
        try:
            metadata = load_metadata(METADATA_FILE)
        except Exception as exc:
            results.append(CheckResult("空间参数与元数据一致", "未通过", f"metadata.json 解析失败：{exc}"))
        else:
            context["metadata"] = metadata
            problems = compare_core_with_metadata(core, metadata)
            results.append(
                CheckResult(
                    "空间参数与元数据一致",
                    "通过" if not problems else "未通过",
                    "CRS、分辨率、行列数和空间范围与 metadata.json 一致" if not problems else "；".join(problems),
                )
            )
    else:
        results.append(CheckResult("空间参数与元数据一致", "跳过", "缺少 metadata.json，无法执行一致性比对"))

    return results, context


def print_report(results: list[CheckResult], context: dict[str, object]) -> None:
    print("demand 全球覆盖和空间参考一致性检查报告")
    print(f"检查目录：{context['demand_dir']}")
    print(f"检查结果：{sum(result.status == '通过' for result in results)}/{len(results)} 项通过")
    print()

    core = context.get("core")
    if core:
        print("核心栅格信息：")
        print(f"- 文件名: {core['name']}")
        print(f"- CRS: {core['crs']}")
        print(f"- 投影类型: {core['projection_type']}")
        print(f"- 分辨率: {core['resolution']}")
        print(f"- 行列数: {core['width']} x {core['height']}")
        print(f"- 空间范围: {core['bounds']}")
        print()

    print("详细结果：")
    for result in results:
        print(f"- [{result.status}] {result.name}：{result.details}")


def main() -> int:
    results, context = build_results()
    print_report(results, context)
    return 0 if all(result.status != "未通过" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
