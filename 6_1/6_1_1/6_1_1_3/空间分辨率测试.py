from __future__ import annotations

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
TARGET_AREA_KM2 = 10.0
EARTH_RADIUS_KM = 6371.0088


@dataclass
class CheckResult:
    name: str
    passed: bool
    details: str


def pixel_area_km2_geographic(res_x_deg: float, res_y_deg: float, lat_center_deg: float) -> float:
    lon_span_rad = math.radians(abs(res_x_deg))
    lat_top_rad = math.radians(lat_center_deg + abs(res_y_deg) / 2.0)
    lat_bottom_rad = math.radians(lat_center_deg - abs(res_y_deg) / 2.0)
    return (EARTH_RADIUS_KM ** 2) * lon_span_rad * abs(math.sin(lat_top_rad) - math.sin(lat_bottom_rad))


def read_raster_info(path: Path) -> dict[str, object]:
    with rasterio.open(path) as ds:
        return {
            "name": path.name,
            "crs": ds.crs,
            "crs_text": str(ds.crs) if ds.crs else None,
            "width": ds.width,
            "height": ds.height,
            "bounds": {
                "left": ds.bounds.left,
                "bottom": ds.bounds.bottom,
                "right": ds.bounds.right,
                "top": ds.bounds.top,
            },
            "res_x": ds.res[0],
            "res_y": ds.res[1],
        }


def estimate_pixel_area_km2(info: dict[str, object]) -> tuple[float, str]:
    crs = info["crs"]
    res_x = float(info["res_x"])
    res_y = float(info["res_y"])

    if crs and crs.is_projected:
        area_km2 = abs(res_x * res_y) / 1_000_000.0
        method = "按投影坐标系分辨率直接换算"
        return area_km2, method

    if crs and crs.is_geographic:
        bounds = info["bounds"]
        lat_center = (float(bounds["top"]) + float(bounds["bottom"])) / 2.0  # type: ignore[index]
        area_km2 = pixel_area_km2_geographic(res_x, res_y, lat_center)
        method = f"按地理坐标系中心纬度 {lat_center:.6f}° 的球面像元面积估算"
        return area_km2, method

    raise ValueError("无法识别坐标系类型，不能估算单像元地表面积")


def build_results() -> tuple[list[CheckResult], dict[str, object]]:
    results: list[CheckResult] = []
    context: dict[str, object] = {"demand_dir": DEMAND_DIR}

    results.append(CheckResult("demand 目录存在", DEMAND_DIR.is_dir(), f"目录路径：{DEMAND_DIR}"))
    results.append(CheckResult("主结果数据存在", CORE_FILE.is_file(), f"主结果文件：{CORE_FILE}"))

    if not CORE_FILE.is_file():
        return results, context

    try:
        raster_info = read_raster_info(CORE_FILE)
    except Exception as exc:  # pragma: no cover - depends on raster file
        results.append(CheckResult("主结果数据可读取", False, f"读取失败：{exc}"))
        return results, context

    context["raster_info"] = raster_info
    results.append(
        CheckResult(
            "主结果数据可读取",
            True,
            (
                f"CRS={raster_info['crs_text']}，"
                f"分辨率=({raster_info['res_x']}, {raster_info['res_y']})，"
                f"行列数={raster_info['width']}x{raster_info['height']}"
            ),
        )
    )

    resolution_ok = float(raster_info["res_x"]) > 0 and float(raster_info["res_y"]) > 0
    results.append(
        CheckResult(
            "空间分辨率提取成功",
            resolution_ok,
            f"res_x={raster_info['res_x']}，res_y={raster_info['res_y']}",
        )
    )

    try:
        pixel_area_km2, area_method = estimate_pixel_area_km2(raster_info)
    except Exception as exc:
        results.append(CheckResult("单像元地表面积量级可计算", False, f"计算失败：{exc}"))
        return results, context

    context["pixel_area_km2"] = pixel_area_km2
    context["area_method"] = area_method
    results.append(
        CheckResult(
            "单像元地表面积量级可计算",
            True,
            f"单像元面积约为 {pixel_area_km2:.6f} km^2；{area_method}",
        )
    )

    metric_ok = pixel_area_km2 <= TARGET_AREA_KM2
    results.append(
        CheckResult(
            "满足阶段性指标要求",
            metric_ok,
            (
                f"单像元面积 {pixel_area_km2:.6f} km^2 <= {TARGET_AREA_KM2:.1f} km^2"
                if metric_ok
                else f"单像元面积 {pixel_area_km2:.6f} km^2 > {TARGET_AREA_KM2:.1f} km^2"
            ),
        )
    )

    return results, context


def print_report(results: list[CheckResult], context: dict[str, object]) -> None:
    print("demand 空间分辨率指标检查报告")
    print(f"检查目录：{context['demand_dir']}")
    print(f"检查结果：{sum(result.passed for result in results)}/{len(results)} 项通过")
    print()

    raster_info = context.get("raster_info")
    if raster_info:
        print("主结果栅格信息：")
        print(f"- 文件名: {raster_info['name']}")
        print(f"- CRS: {raster_info['crs_text']}")
        print(f"- 分辨率: ({raster_info['res_x']}, {raster_info['res_y']})")
        print(f"- 行列数: {raster_info['width']} x {raster_info['height']}")
        print(f"- 空间范围: {raster_info['bounds']}")
        print()

    if "pixel_area_km2" in context:
        print("面积估算：")
        print(f"- 单像元地表面积: {context['pixel_area_km2']:.6f} km^2")
        print(f"- 估算方法: {context['area_method']}")
        print()

    print("详细结果：")
    for result in results:
        status = "通过" if result.passed else "未通过"
        print(f"- [{status}] {result.name}：{result.details}")


def main() -> int:
    results, context = build_results()
    print_report(results, context)
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
