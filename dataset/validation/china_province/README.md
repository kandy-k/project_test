# 中国省级一致性测试数据说明

将中国省级一致性测试所需输入数据放入以下目录：

- `dataset/validation/china_province/boundaries/`
- `dataset/validation/china_province/truth/`

## 1. 省级边界文件

放入 1 份中国省级行政区边界数据，支持：

- `.gpkg`
- `.shp`
- `.geojson`

边界文件需包含省级几何，并且至少有 1 个可识别省份名称的字段。

## 2. 工信部 2024 年省级统计表

放入 1 份省级统计表，支持：

- `.csv`
- `.xlsx`
- `.xls`

统计表至少需要以下列：

- `province_name`
- `traffic_eb`

或者：

- `province_name`
- `traffic_value`
- `traffic_unit`

其中 `traffic_unit` 支持 `B`、`KB`、`MB`、`GB`、`TB`、`PB`、`EB`、`万GB`、`亿GB`、`万TB`、`亿TB`。
