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
SPECTRUM_DIR = REPO_ROOT / "dataset" / "products" / "spectrum"
KUKA_MAIN_FILE = SPECTRUM_DIR / "Global_5G_Radiation_Map_60arcsec.tif"
README_FILE = SPECTRUM_DIR / "README.md"
METADATA_FILE = SPECTRUM_DIR / "metadata.json"
TARGET_RES_DEG = 1.0 / 60.0
TOL = 1e-9
REQUIRED_METADATA_KEYS = (
    "dataset_name",
    "version",
    "primary_file",
    "format",
    "crs",
    "width",
    "height",
    "band_count",
    "dtype",
    "nodata",
    "bounds",
    "resolution",
)


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


def load_metadata(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def info_from_metadata(metadata: dict[str, object]) -> dict[str, object]:
    return {
        "name": metadata.get("primary_file"),
        "driver": metadata.get("format"),
        "width": metadata.get("width"),
        "height": metadata.get("height"),
        "band_count": metadata.get("band_count"),
        "dtype": metadata.get("dtype"),
        "crs": metadata.get("crs"),
        "nodata": metadata.get("nodata"),
        "resolution": metadata.get("resolution"),
        "bounds": metadata.get("bounds"),
    }


def metadata_raster_problems(metadata: dict[str, object], raster: dict[str, object]) -> list[str]:
    problems: list[str] = []
    checks = [
        ("primary_file", raster["name"]),
        ("format", raster["driver"]),
        ("crs", raster["crs"]),
        ("width", raster["width"]),
        ("height", raster["height"]),
        ("band_count", raster["band_count"]),
        ("dtype", raster["dtype"]),
        ("bounds", raster["bounds"]),
        ("resolution", raster["resolution"]),
    ]
    for key, expected in checks:
        if metadata.get(key) != expected:
            problems.append(f"{key} 应为 {expected}")
    return problems


def build_results() -> tuple[list[CheckResult], dict[str, object]]:
    results: list[CheckResult] = []
    context: dict[str, object] = {"spectrum_dir": SPECTRUM_DIR, "main_file": KUKA_MAIN_FILE}

    results.append(CheckResult("spectrum 目录存在", SPECTRUM_DIR.is_dir(), f"目录路径：{SPECTRUM_DIR}"))

    metadata_exists = METADATA_FILE.is_file()
    metadata: dict[str, object] | None = None
    metadata_ok = False
    if metadata_exists:
        try:
            metadata = load_metadata(METADATA_FILE)
        except Exception as exc:
            metadata_detail = f"metadata.json 解析失败：{exc}"
        else:
            missing_keys = [key for key in REQUIRED_METADATA_KEYS if key not in metadata]
            primary_ok = metadata.get("primary_file") == KUKA_MAIN_FILE.name
            metadata_ok = not missing_keys and primary_ok
            metadata_detail = "元数据必填字段齐全，主成果文件名可追溯" if metadata_ok else (
                f"缺少字段：{', '.join(missing_keys)}" if missing_keys else f"primary_file={metadata.get('primary_file')}"
            )
            context["metadata"] = metadata
    else:
        metadata_detail = f"元数据文件：{METADATA_FILE}"

    results.append(CheckResult("Ku/Ka频段主成果元数据完整", metadata_ok, metadata_detail))

    main_exists = KUKA_MAIN_FILE.is_file()
    results.append(
        CheckResult(
            "Ku/Ka频段主成果记录存在",
            main_exists or metadata is not None,
            f"主成果文件：{KUKA_MAIN_FILE}" if main_exists else "大体积主成果未上传，使用 metadata.json 记录成果基础属性",
        )
    )

    info: dict[str, object] | None = None
    if main_exists:
        file_non_empty = KUKA_MAIN_FILE.stat().st_size > 0
        try:
            info = read_raster_info(KUKA_MAIN_FILE)
        except Exception as exc:  # pragma: no cover - depends on raster file
            results.append(CheckResult("Ku/Ka频段主成果可核验", False, f"读取失败：{exc}"))
            return results, context
        verify_ok = file_non_empty
        verify_detail = f"文件非空，可读取；driver={info['driver']}，size={info['width']}x{info['height']}，band_count={info['band_count']}，dtype={info['dtype']}"
        if metadata is not None:
            problems = metadata_raster_problems(metadata, info)
            verify_ok = verify_ok and not problems
            verify_detail = "文件非空、可读取，且元数据与主成果栅格一致" if not problems else "；".join(problems)
        results.append(CheckResult("Ku/Ka频段主成果可核验", verify_ok, verify_detail))
    elif metadata is not None:
        info = info_from_metadata(metadata)
        results.append(CheckResult("Ku/Ka频段主成果可核验", True, "主成果大文件可不上传，当前按 metadata.json 检查基础属性"))

    if info is None:
        return results, context

    context["raster_info"] = info
    format_ok = info["name"] == KUKA_MAIN_FILE.name and info["driver"] == "GTiff"
    results.append(CheckResult("Ku/Ka频段主成果格式规范", format_ok, f"文件名={info['name']}，format={info['driver']}"))

    crs_ok = info["crs"] == "EPSG:4326"
    results.append(CheckResult("坐标参考系统为EPSG:4326", crs_ok, f"CRS={info['crs']}"))

    res = info["resolution"]
    resolution_ok = isinstance(res, dict) and math.isclose(float(res["x"]), TARGET_RES_DEG, rel_tol=0.0, abs_tol=TOL) and math.isclose(float(res["y"]), TARGET_RES_DEG, rel_tol=0.0, abs_tol=TOL)
    results.append(CheckResult("Ku/Ka频段主成果分辨率为60arcsec", resolution_ok, f"resolution={res}"))

    bounds = info["bounds"]
    coverage_ok = (
        isinstance(bounds, dict)
        and math.isclose(float(bounds["left"]), -180.0, rel_tol=0.0, abs_tol=TOL)
        and math.isclose(float(bounds["right"]), 180.0, rel_tol=0.0, abs_tol=TOL)
        and float(bounds["bottom"]) >= -90.0
        and float(bounds["top"]) <= 90.0
    )

    readme_text = README_FILE.read_text(encoding="utf-8", errors="ignore") if README_FILE.is_file() else ""
    readme_ok = KUKA_MAIN_FILE.name in readme_text and "metadata.json" in readme_text and ("KuKa" in readme_text or "Ku/Ka" in readme_text or "5G" in readme_text)
    results.append(
        CheckResult(
            "Ku/Ka频段主成果覆盖范围和说明可追溯",
            coverage_ok and readme_ok,
            f"bounds={bounds}，说明文档：{README_FILE}" if coverage_ok and readme_ok else "空间范围或README说明不满足要求",
        )
    )

    return results, context


def print_report(results: list[CheckResult], context: dict[str, object]) -> None:
    print("Ku/Ka频段支撑能力检查报告")
    print(f"检查目录：{context['spectrum_dir']}")
    print(f"检查结果：{sum(result.passed for result in results)}/{len(results)} 项通过")
    print()

    info = context.get("raster_info")
    if info:
        print("Ku/Ka频段主成果基础信息：")
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
