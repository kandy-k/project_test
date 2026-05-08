from __future__ import annotations

import json
import re
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
README_FILE = DEMAND_DIR / "README.md"
METADATA_FILE = DEMAND_DIR / "metadata.json"
VERSION_PATTERN = re.compile(r"^v\d+(?:\.\d+){0,2}$", re.IGNORECASE)
NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*\.tif$", re.IGNORECASE)
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


def add_result(results: list[CheckResult], name: str, passed: bool, details: str) -> None:
    results.append(CheckResult(name=name, passed=passed, details=details))


def read_raster_summary(path: Path) -> dict[str, object]:
    with rasterio.open(path) as src:
        return {
            "driver": src.driver,
            "width": src.width,
            "height": src.height,
            "band_count": src.count,
            "dtype": src.dtypes[0],
            "crs": str(src.crs) if src.crs else None,
            "nodata": src.nodata,
            "bounds": {
                "left": src.bounds.left,
                "bottom": src.bounds.bottom,
                "right": src.bounds.right,
                "top": src.bounds.top,
            },
            "resolution": {
                "x": src.res[0],
                "y": src.res[1],
            },
        }


def compare_metadata(metadata: dict[str, object], raster: dict[str, object]) -> list[str]:
    problems: list[str] = []
    if metadata.get("primary_file") != CORE_FILE.name:
        problems.append(f"primary_file 应为 {CORE_FILE.name}")
    if metadata.get("format") != raster["driver"]:
        problems.append(f"format 应为 {raster['driver']}")
    if metadata.get("crs") != raster["crs"]:
        problems.append(f"crs 应为 {raster['crs']}")
    if metadata.get("width") != raster["width"]:
        problems.append(f"width 应为 {raster['width']}")
    if metadata.get("height") != raster["height"]:
        problems.append(f"height 应为 {raster['height']}")
    if metadata.get("band_count") != raster["band_count"]:
        problems.append(f"band_count 应为 {raster['band_count']}")
    if metadata.get("dtype") != raster["dtype"]:
        problems.append(f"dtype 应为 {raster['dtype']}")
    if metadata.get("nodata") != raster["nodata"]:
        problems.append(f"nodata 应为 {raster['nodata']}")
    if metadata.get("bounds") != raster["bounds"]:
        problems.append("bounds 与栅格实际范围不一致")
    if metadata.get("resolution") != raster["resolution"]:
        problems.append("resolution 与栅格实际分辨率不一致")
    return problems


def build_results() -> tuple[list[CheckResult], dict[str, object]]:
    results: list[CheckResult] = []
    context: dict[str, object] = {"demand_dir": DEMAND_DIR}

    add_result(results, "demand 目录存在", DEMAND_DIR.exists() and DEMAND_DIR.is_dir(), f"目录路径：{DEMAND_DIR}")
    add_result(results, "核心数据文件齐全", CORE_FILE.exists() and CORE_FILE.is_file(), f"核心文件：{CORE_FILE}")

    if not CORE_FILE.exists():
        return results, context

    add_result(results, "核心数据层完整", CORE_FILE.stat().st_size > 0, f"文件大小：{CORE_FILE.stat().st_size} 字节")

    try:
        raster_summary = read_raster_summary(CORE_FILE)
    except Exception as exc:  # pragma: no cover - depends on raster file
        add_result(results, "核心数据层可读取", False, f"读取失败：{exc}")
        return results, context

    context["raster_summary"] = raster_summary
    add_result(
        results,
        "核心数据层可读取",
        True,
        (
            f"driver={raster_summary['driver']}，"
            f"size={raster_summary['width']}x{raster_summary['height']}，"
            f"band_count={raster_summary['band_count']}，"
            f"dtype={raster_summary['dtype']}"
        ),
    )

    readme_ok = README_FILE.exists() and README_FILE.is_file() and README_FILE.stat().st_size > 0
    add_result(results, "说明文档完整", readme_ok, f"说明文档：{README_FILE}")

    metadata_ok = METADATA_FILE.exists() and METADATA_FILE.is_file() and METADATA_FILE.stat().st_size > 0
    add_result(results, "元数据文件齐全", metadata_ok, f"元数据文件：{METADATA_FILE}")

    metadata: dict[str, object] | None = None
    if metadata_ok:
        try:
            metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
        except Exception as exc:
            add_result(results, "元数据格式规范", False, f"metadata.json 解析失败：{exc}")
        else:
            missing_keys = [key for key in REQUIRED_METADATA_KEYS if key not in metadata]
            add_result(
                results,
                "元数据完整",
                not missing_keys,
                "必填字段齐全" if not missing_keys else f"缺少字段：{', '.join(missing_keys)}",
            )

            version = str(metadata.get("version", "")).strip()
            add_result(
                results,
                "版本号规范",
                bool(VERSION_PATTERN.fullmatch(version)),
                f"version={version or '空'}",
            )

            if not missing_keys:
                problems = compare_metadata(metadata, raster_summary)
                add_result(
                    results,
                    "元数据内容一致",
                    not problems,
                    "元数据与栅格一致" if not problems else "；".join(problems),
                )
    else:
        add_result(results, "元数据完整", False, "缺少 metadata.json，无法检查必填字段")
        add_result(results, "版本号规范", False, "缺少 metadata.json，无法检查版本号")

    naming_ok = (
        bool(NAME_PATTERN.fullmatch(CORE_FILE.name))
        and README_FILE.name == "README.md"
        and METADATA_FILE.name == "metadata.json"
    )
    add_result(
        results,
        "成果命名规范",
        naming_ok,
        f"核心文件={CORE_FILE.name}，说明文档={README_FILE.name}，元数据={METADATA_FILE.name}",
    )

    format_ok = CORE_FILE.suffix.lower() == ".tif" and raster_summary["driver"] == "GTiff"
    add_result(
        results,
        "数据格式规范",
        format_ok,
        f"扩展名={CORE_FILE.suffix}，driver={raster_summary['driver']}",
    )

    expected_root_files = {CORE_FILE.name, README_FILE.name, METADATA_FILE.name}
    actual_root_files = {path.name for path in DEMAND_DIR.iterdir() if path.is_file()}
    extra_root_files = sorted(actual_root_files - expected_root_files)
    missing_root_files = sorted(expected_root_files - actual_root_files)
    dir_ok = not extra_root_files and not missing_root_files
    if dir_ok:
        dir_details = "demand 目录根层仅包含核心数据、说明文档和元数据文件"
    else:
        problems = []
        if missing_root_files:
            problems.append(f"缺少文件：{', '.join(missing_root_files)}")
        if extra_root_files:
            problems.append(f"额外文件：{', '.join(extra_root_files)}")
        dir_details = "；".join(problems)
    add_result(results, "目录组织规范", dir_ok, dir_details)

    return results, context


def print_report(results: list[CheckResult], context: dict[str, object]) -> None:
    demand_dir = context["demand_dir"]
    raster_summary = context.get("raster_summary")
    passed_count = sum(result.passed for result in results)

    print("demand 数据库成果完整性检查报告")
    print(f"检查目录：{demand_dir}")
    print(f"检查结果：{passed_count}/{len(results)} 项通过")
    print()

    if raster_summary:
        print("栅格基础信息：")
        print(f"- driver: {raster_summary['driver']}")
        print(f"- 尺寸: {raster_summary['width']} x {raster_summary['height']}")
        print(f"- 波段数: {raster_summary['band_count']}")
        print(f"- 数据类型: {raster_summary['dtype']}")
        print(f"- CRS: {raster_summary['crs']}")
        print(f"- NoData: {raster_summary['nodata']}")
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
