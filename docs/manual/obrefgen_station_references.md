# OBRefGen 站点参考数据（2026-10-31）

新增数据按现有类别放入 `/Volumes/Data/Reference/Station` 下的 `Water`、`Heat`、`Bio`、`Lake`、`Anth`、`Composite`，地面气象放入 `Meteo`。原有 Station 子目录保留，不将旧同名文件视为本次已验收成果。径流和水库入出流数据、相关注册条目保持原样。

OpenBench 以 `OBRefGen_来源_频率` 命名新参考源，例如 `OBRefGen_BEFED_daily`、`OBRefGen_Filled_LE_daily`、`OBRefGen_Urban_PLUMBER_hourly`。在评价项目中选择相应变量，再选择这些参考源；软件保留原 NetCDF 科学变量名，并通过映射关联 OpenBench 评价变量。

## 数据范围

| 来源 | 本次接入范围 | 主要边界 |
|---|---|---|
| FLUX 站点通量 | 已验收小时、半小时合成产品 | 原始 QC、缺测和逐值来源保留；不把旧日/月产品视作本次重新验收 |
| Urban-PLUMBER | 半小时、小时、日，12 项气象及能量变量 | 保留源时间界限、QC 与聚合性质 |
| CEMADEN、ISMN、SONTE-China | 分来源日土壤水分；ISMN另有根区产品 | 浅层深度依来源；根区按 0–1 m 覆盖规则；这些锁定处理产品没有逐值 QC 附属变量 |
| SWE | 日雪水当量 | 区分直接观测与雪深推算，不将全部值称为直接观测 |
| USGS | 日/月地下水埋深 | 保留井与测量参考信息 |
| ReaLSAT | 月湖面积 | 卫星重建与上游填补信息保留 |
| G-REALM | 日槽中的不规则湖水位观测 | 不进行时间补缺或统一不同湖泊的基准 |
| WSE | 日 EGM2008 水面高程 | 无可靠基准转换的来源主值仍缺测；与其他高程基准比较前需配置软件已有的大地水准面转换 |
| SAPFLUXNET | 小时/日冠层蒸腾 | 树液流推算；日值可由至少 18 个合格小时部分累加 |
| BEFED | 独立日蒸散发 | 发布处理值含发布方补缺；保留源日期标签，物理日界未证实 |
| Filled-LE | 日潜热、净辐射、土壤热 | 源 QC：0 实测、1 再分析或计算、2 补缺；日界未证实 |
| Yuan2011 | 原生 8 日/月 LAI 站点副本 | 卫星提取副本，上游版本及提取链未证实 |
| GEBA | 五项月辐射参考 | 未纳入方向不明的 H/LE/G 标准基准 |
| FLUXNET-CH4-ANN | 月湿地甲烷派生产品 | 不是纯实测；缺原始站名及实测/ANN占比，保留 noleap 时间轴 |
| C3S | 仅已通过独立审计的 2011 年五项小时产品 | 最近小时代表观测，不是小时均值；其他年份仍在后台处理，本次不宣称已完成 |

GLEAM 历史 ET/T 副本、GLAST 模型源记录和 Rodal2025 NPP 测量记录一并存放于资料区，但不注册为普通时序评价参考：GLEAM 缺单位且模型身份未证实；GLAST 单位和物理日界未证实；NPP 没有可直接解释为自然年的标准时间轴。CMA 修复版全量和 ISD 独立流程尚未完成，本次不补造产品。

## 文件与配置

每个来源和频率在对应类别中存放文件，例如 `Water/Evapotranspiration/BEFED/daily`、`Heat/GEBA/monthly`、`Anth/Urban/UrbanFlux/hourly`。NetCDF 从已验收成果逐字节复制，不重采样、不填补、不改单位；诊断与来源附属变量随文件保留，但不会自动列作评价项目。`metadata/station-integration-20261031/manifest.json` 列出来源路径、SHA256、变量、单位、时间和站点数；其中 `destination` 相对于 Station 根目录。诊断变量未安装站点列表时，`stationListInstalled` 为 false。

`station_lists/openbench_station_new` 下每个科学变量有独立 CSV，列为 `ID,SYEAR,EYEAR,LON,LAT,DIR`。`DIR` 使用相对 NetCDF 文件名，由该来源根目录解析。缺少有效经纬度的站点保留在原 NetCDF，仅不进入用于空间评价的站点列表；列表时间范围是文件时间轴覆盖，不表示各站全时段都有有效值。

软件读取 `station_id`，也兼容仅有 `source_station_id` 的来源；保留前导零。每变量列表避免不同科学文件被拼成重复站点的共同列表。重新扫描时保留这些显式列表。

本机安装配置为 `~/.openbench/references/obrefgen_stations.yaml`，将新增来源分别指向 Station 下的对应类别目录。不改全局参考根目录，以免影响现有径流或格点配置。包内目录和扫描 profile 使用 `${OPENBENCH_REF_ROOT}/Station/类别/来源/频率`（部分来源沿用既有目录层次），迁移时应同时迁移 CSV 和 NetCDF，并更新本机覆盖路径。

时间偏移保持为零以保留已有标签；这不将日界未明的来源认定为 UTC 日产品。数据流程核验通过也不代表自动科学批准，选用前仍应阅读来源限制。

## 备份与回退

原 Station 数据另存为 ZIP64 备份，备份位置和核验范围见 `metadata/station-integration-20261031/integration-receipt.json`。ZIP 使用存储模式，包含原文件和 SHA256 清单；原 NetCDF 通常已压缩，不再二次压缩。

因为原数据保留不变，回退只需停用新增的本机覆盖配置，并按迁移清单处理本次新增文件，不需要用新产品反向覆盖原数据。软件代码变更由 Git 追踪；不要以整目录恢复覆盖其他人的后续修改。
