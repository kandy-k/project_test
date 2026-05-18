# validation 目录说明

`validation/` 目录按“共享边界”和“分测试真值”拆分，避免同一套边界文件重复存放。

## 目录结构

- `dataset/validation/boundaries/`
  共享边界数据，目前包含 Natural Earth 国家级与一级行政区边界。
- `dataset/validation/truth/china_province/`
  中国省级一致性测试使用的统计真值。
- `dataset/validation/truth/country_level/`
  国家级一致性测试和国家内部跨尺度测试使用的统计真值。

## 使用原则

- 边界数据尽量只保留一份，供多个验证脚本复用。
- 真值数据按验证任务拆分，便于维护不同统计口径。
