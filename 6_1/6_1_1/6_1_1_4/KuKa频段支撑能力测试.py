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
SPECTRUM_DIR = REPO_ROOT / "dataset" / "products" / "spectrum"
KUKA_MAIN_FILE = SPECTRUM_DIR / "Global_5G_Radiation_Map_60arcsec.tif"
README_FILE = SPECTRUM_DIR / "README.md"
TARGET_RES_DEG = 1.0 / 60.0
TOL = 1e-9


@dataclass
class CheckResult:
    name: str
    passed: bool
    details: str


def read_raster_info(path: Path) -> dict[str, object]:
    with rasterio.open(path) as ds:
        return {
            "name": path.name,
            "driver": ds.driver,
            "width": ds.width,
            "height": ds.height,
            "band_count": ds.count,
            "dtype": ds.dtypes[0],
            "crs": str(ds.crs) if ds.crs else None,
            "nodata": ds.nodata,
            "resolution": {"x": ds.res[0], "y": ds.res[1]},
            "bounds": {
                "left": ds.bounds.left,
                "bottom": ds.bounds.bottom,
                "right": ds.bounds.right,
                "top": ds.bounds.top,
            },
        }


def build_results() -> tuple[list[CheckResult], dict[str, object]]:
    results: list[CheckResult] = []
    context: dict[str, object] = {"spectrum_dir": SPECTRUM_DIR, "main_file": KUKA_MAIN_FILE}

    results.append(CheckResult("spectrum 目录存在", SPECTRUM_DIR.is_dir(), f"目录路径：{SPECTRUM_DIR}"))
    results.append(CheckResult("Ku/Ka频段主成果存在", KUKA_MAIN_FILE.is_file(), f"主成果文件：{KUKA_MAIN_FILE}"))
    if not KUKA_MAIN_FILE.is_file():
        return results, context

    results.append(CheckResult("Ku/Ka频段主成果文件非空", KUKA_MAIN_FILE.stat().st_size > 0, f"文件大小：{KUKA_MAIN_FILE.stat().st_size} 字节"))

    try:
        info = read_raster_info(KUKA_MAIN_FILE)
    except Exception as exc:  # pragma: no cover - depends on raster file
        results.append(CheckResult("Ku/Ka频段主成果可读取", False, f"读取失败：{exc}"))
        return results, context

    context["raster_info"] = info
    results.append(
        CheckResult(
            "Ku/Ka频段主成果可读取",
            True,
            f"driver={info['driver']}，size={info['width']}x{info['height']}，band_count={info['band_count']}，dtype={info['dtype']}",
        )
    )

    format_ok = KUKA_MAIN_FILE.suffix.lower() == ".tif" and info["driver"] == "GTiff"
    results.append(CheckResult("Ku/Ka频段主成果格式规范", format_ok, f"扩展名={KUKA_MAIN_FILE.suffix}，driver={info['driver']}"))

    crs_ok = info["crs"] == "EPSG:4326"
    results.append(CheckResult("坐标参考系统为EPSG:4326", crs_ok, f"CRS={info['crs']}"))

    res = info["resolution"]
    resolution_ok = math.isclose(float(res["x"]), TARGET_RES_DEG, rel_tol=0.0, abs_tol=TOL) and math.isclose(float(res["y"]), TARGET_RES_DEG, rel_tol=0.0, abs_tol=TOL)  # type: ignore[index]
    results.append(CheckResult("Ku/Ka频段主成果分辨率为60arcsec", resolution_ok, f"resolution={res}"))

    bounds = info["bounds"]
    coverage_ok = (
        math.isclose(float(bounds["left"]), -180.0, rel_tol=0.0, abs_tol=TOL)  # type: ignore[index]
        and math.isclose(float(bounds["right"]), 180.0, rel_tol=0.0, abs_tol=TOL)  # type: ignore[index]
        and float(bounds["bottom"]) >= -90.0  # type: ignore[index]
        and float(bounds["top"]) <= 90.0  # type: ignore[index]
    )
    results.append(CheckResult("Ku/Ka频段主成果覆盖全球研究范围", coverage_ok, f"bounds={bounds}"))

    readme_text = README_FILE.read_text(encoding="utf-8", errors="ignore") if README_FILE.is_file() else ""
    readme_ok = "Global_5G_Radiation_Map_60arcsec.tif" in readme_text and ("KuKa" in readme_text or "Ku/Ka" in readme_text or "5G" in readme_text)
    results.append(CheckResult("Ku/Ka频段主成果说明可追溯", readme_ok, f"说明文档：{README_FILE}" if readme_ok else "README中未检索到主成果或频段说明"))

    return results, context


def print_report(results: list[CheckResult], context: dict[str, object]) -> None:
    print("Ku/Ka频段支撑能力检查报告")
    print(f"检查目录：{context['spectrum_dir']}")
    print(f"检查结果：{sum(result.passed for result in results)}/{len(results)} 项通过")
    print()

    info = context.get("raster_info")
    if info:
        print("Ku/Ka频段主成果栅格信息：")
        print(f"- 文件名: {info['name']}")
        print(f"- 尺寸: {info['width']} x {info['height']}")
        print(f"- 波段数: {info['band_count']}")
        print(f"- 数据类型: {info['dtype']}")
        print(f"- CRS: {info['crs']}")
        print(f"- 分辨率: {info['resolution']}")
        print(f"- 空间范围: {info['bounds']}")
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
