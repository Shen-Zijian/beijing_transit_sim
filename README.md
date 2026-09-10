# beijing_transit_sim — 纯净的北京多模式公交/地铁仿真核心 + 2019-05-13 刷卡数据标定

从 `simulator_beijing_simplified/simulator_switch.py` 中剥离出的纯仿真内核：**没有** RL agent、用户分组、老年建模、
static-SO、六边形热力图、线程池与 XGBoost 路径选择模型；保留并重写了网络加载、车辆调度/运行、乘客生命周期、
k 短路路由与指标记录，并用 2019-05-13 全日一卡通数据（10.7M 条）标定了车速、逐站时间、发车间隔、
进出站与换乘时间，生成与数据同期的 2019 版路网。

## 目录

```
config/default.yaml                 未标定的基线参数（全部参数及注释）
config/calibrated_20190513.yaml     标定流水线生成的配置（默认使用）
transit_sim/                        仿真核心（纯 Python，依赖 pandas/numpy/scipy/pyyaml）
  config.py     YAML -> 嵌套 dataclass，支持 --set a.b=c 覆盖
  network.py    站点/线路/步行换乘边；线路级速度表、班距表、同站换乘时间表
  routing.py    线路感知 A*/Dijkstra + Yen k 短路（去重叠），OD 级缓存
  choice.py     路径选择：shortest | logit（广义成本）
  demand.py     需求 CSV -> 按步长分桶；确定性抽样 fraction
  vehicles.py   车辆状态机（首站停留 -> 运行 -> 停站 -> ... -> 终点），事件驱动
  passengers.py 乘客对象与状态（walking/waiting/onboard/arrived/timeout_exit）
  simulator.py  主循环、上下车、候车队列、超时/改路、钩子（SimHooks）
  metrics.py    逐行程记录（总时间/候车/首次候车/步行/车内/进出站/换乘）、小时快照、summary
calibration/                        标定流水线（离线，一次性）
scripts/run_sim.py                  命令行运行仿真
scripts/precompute_paths.py         多进程预计算路径缓存
data/network/                       nodes_full / edges_full（原路网）、nodes_2019 / edges_2019、line_speeds_2019、
                                    line_headways_2019、transfer_times_2019
data/demand/demand_20190513.csv     2.85M 条仿真格式需求（含 observed_tt_s 供验证）+ hourly_profile
data/processed/                     中间文件（匹配后刷卡记录、各步报告 json、路径缓存 pkl）
reports/                            calibration_summary_20190513.md、validation_20190513.md 与图
tests/                              pytest（网络一致性、路由性质、smoke run、可复现性）
```

## 公开仓库的数据范围

仓库包含仿真与标定代码、路网、标定参数、汇总报告以及合成需求示例。
真实刷卡记录、逐人出行需求 `data/demand/demand_20190513.csv`、路径缓存和仿真输出仅保留在本地，不随仓库发布。
下文的全日标定及验证结果基于本地真实数据；合成示例仅用于检查仿真能否运行，不用于复现这些数值。

## 快速开始

需要 Python 3.12 或更新版本。

```bash
git clone https://github.com/Shen-Zijian/beijing_transit_sim.git
cd beijing_transit_sim
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 合成需求演示，无需真实刷卡数据或预计算缓存
.venv/bin/python scripts/run_sim.py --config config/calibrated_20190513.yaml --run-id demo_synthetic \
  --set demand.file=data/demand/demand_example.csv --set demand.fraction=1.0 \
  --set time.sim_start=25200 --set time.sim_end=28800 --set routing.cache_file=null

.venv/bin/python -m pytest -q tests
```

未提供真实需求文件时，依赖该文件的测试会跳过；网络、路由和配置测试仍可执行。

### 已具备本地研究数据时

```bash
cd beijing_transit_sim
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 全天仿真（5% 需求，约 30 s；首次运行会加载 277 MB 路径缓存）
.venv/bin/python scripts/run_sim.py --config config/calibrated_20190513.yaml --run-id demo

# 覆盖任意参数
.venv/bin/python scripts/run_sim.py --set demand.fraction=0.1 --set time.sim_end=43200 --set choice.model=shortest

# 测试
.venv/bin/python -m pytest -q tests
```

Python API：

```python
from transit_sim import SimConfig, TransitSimulator, SimHooks

class MyPolicy(SimHooks):
    def on_passenger_created(self, sim, passenger, routes):
        return routes[0]          # 返回一个 route 即覆盖 logit 选择；返回 None 则用选择模型
    def on_step(self, sim, now):
        pass                      # 每步回调：可读取 sim.passengers / sim.fleet.vehicles / sim.queues

cfg = SimConfig.load("config/calibrated_20190513.yaml", overrides={"demand.fraction": 0.05})
result = TransitSimulator(cfg, hooks=MyPolicy()).run()
result.trips        # DataFrame：每个乘客一行
result.hourly       # 每小时系统快照
result.summary      # dict
```

输出目录 `output/<run_id>/`：`trips.csv`、`hourly_stats.csv`、`summary.json`、`config.yaml`。

## 仿真模型

* 时间：自午夜起的秒，固定步长 `time.time_step`（默认 10 s）；车辆与乘客计时事件驱动，单线程、`seed` 可复现。
* 车辆：每条线路(方向) 按 (线路, 小时) 班距发车（`line_headways_2019.csv` > 配置分小时字典 > 常数），
  首站停留 `terminal_hold` 后出发；每段运行时间 = 段距离 / 线路速度 + `stop_delay`，每站开门 `dwell`；
  到终点回收。仿真开始时按班距把整条线路"热启动"铺满车辆。容量 = `capacity_real × demand.fraction`。
* 乘客：出发时刻来自需求文件；地铁起点先经历 `access_egress`（进站），按 k 条候选路径用 logit 选路，
  在 (站, 线路) FIFO 队列候车，车到即上（受容量限制，到站与发车前各上一次），到站后按路径换乘
  （同站换乘 = 站点 `transfer_time` 步行后重新候车；跨站换乘走步行边），到达目的地后经历出站时间。
  候车超过 `max_wait_time` 触发 `timeout_action`（exit | reroute）。
* 路由：状态 (节点, 当前线路) 上的 A*，代价 = 行驶 + 步行 + 换乘惩罚 + 同站换乘时间；Yen k 短路只在
  上下车/换乘点分支并过滤重叠 > 80% 的近似重复路径。84,086 对 OD 已预计算在 `data/processed/route_cache_2019.pkl`。
* 与旧版的主要差别：同一站对可被多条线路服务（旧版 DiGraph 会覆盖）；环线完整闭合；同站换乘有真实
  步行时间（旧版为 0）；候车/车内/步行分项统计正确累计（旧版 `avg_waiting_time` 从不更新）。

## 标定流水线

```bash
.venv/bin/python -m calibration.run_all --raw /path/to/20190513.csv --precompute --validate
```

| 步骤 | 脚本 | 产出 |
|---|---|---|
| 1 | `preprocess_smartcard --network full` | 清洗（bbox、时间、同站）并按完整路网匹配站名/线路 |
| 2 | `prune_network_2019` | 剔除 2019-05 未开通线路/车站（数据驱动 + 已知事实），生成 `nodes_2019/edges_2019` |
| 3 | `preprocess_smartcard --network 2019` | 按 2019 路网重新匹配 → `smartcard_20190513_matched.csv`（3.27M 行） |
| 4 | `convert_demand` | `demand_20190513.csv`（2.85M 行，84k OD）+ 小时剖面 |
| 5 | `calibrate_speed` | 车速 / 逐站时间 / 首站停留：模式 × 时段，线路 × 时段 |
| 6 | `calibrate_headway` | 公交班距（上车刷卡聚类）、地铁班距 + 进出站（截距分解） |
| 7 | `calibrate_transfer` | 地铁同站换乘时间（换乘行程残差） |
| 8 | `build_config` | `config/calibrated_20190513.yaml` + `reports/calibration_summary_20190513.md` |
| 9 | `scripts/precompute_paths.py` | 路径缓存 |
| 10 | `calibration.validate` | `reports/validation_20190513.md` + 图 |

### 标定结果摘要（详见 reports/）

| 参数 | 标定值 | 旧版实验值 |
|---|---:|---:|
| 地铁站间运行速度 | 42.4 km/h（分时段 41.6–43.2；线路级 24 条方向） | 75 |
| 地铁每站停站 | 35 s（先验，与站距共线不可辨识） | 30 |
| 公交边际速度 / 逐站固定时间 | 54.0 km/h / 102 s（高峰 118 s）→ dwell 25 s + stop_delay 77 s | 20 km/h / 20 s |
| 公交首站停留 | 80 s | – |
| 公交班距 | 分小时 420–720 s（高峰 7 min，平峰 8.5 min，晚间 12 min） | 60 s |
| 地铁班距 | 先验 高峰 180 / 平峰 360 / 夜间 480 s（隐含估计 290–500 s） | 800 s |
| 地铁进出站 | 142 s（有效值；下界 113 s） | – |
| 地铁同站换乘 | 60 s（残差中位数 40 s，下限截断） | 60 s |
| 步行速度 | 1.2 m/s | 12 m/s |
| 单车定员 | 地铁 1460 / 公交 90 × demand.fraction | 65 / 10（固定） |

验证（5% 需求，全天）：仿真 vs 观测行程时间 总体均值 19.7 vs 20.3 min（旧参数 16.2），KS 0.027（旧参数 0.102）；
地铁 28.8 vs 27.6，公交（车内口径）13.2 vs 15.0。

### 需要注意的假设

* 刷卡数据只能识别 进出站 + 候车 + 换乘步行 之和，三者拆分不可识别；默认用典型地铁班距先验并把余量归入
  进出站时间（`--subway-headway-mode implied` 为反推班距的替代方案）。可在 `line_headways_2019.csv` 中按线路覆盖。
* 公交"速度"是扣除逐站固定时间后的边际速度，且基于站间直线距离；逐站固定时间（~100 s）包含加减速、
  进出站与信号延误。二者组合复现观测的站间行程时间（典型 570 m 一站 ≈ 140 s）。
* 地铁进站刷卡为分钟精度（视为向下取整，行程时间 −30 s 补偿）。
* 1 号线与八通线在路网中已合并，2019 版保留合并形态；14 号线中段、8 号线中段、16 号线等按 2019-05 开通情况裁剪
  （见 `calibration/prune_network_2019.py` 顶部的规则与 `data/processed/prune_network_2019_report.json`）。
* 公交刷卡覆盖的 835 条线路中约 19% 不在路网数据中（运通/特字头、郊区线），未参与标定与需求。
* 需求文件中的 `pid` 为脱敏卡号前 12 位；`observed_tt_s` 为对应刷卡行程时间（地铁已补偿）。
