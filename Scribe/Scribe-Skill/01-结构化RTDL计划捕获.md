# 文档 01｜结构化 RTDL 计划捕获

> 阶段 S1：现有系统已经能够保存和重放完整树形 RTDL；本阶段剩余目标是补齐提炼技能所需的执行上下文、权威结果与稳定接口。

## 阶段目标

在现有 query 根记录 + planning round 子计划的基础上，为每次成功 RTDL 执行持久化起点/终点位姿、Scene 对象与区域上下文、前置/后置条件、时间范围和验证结果，使其能够作为 S2 的可靠成功样本，并为 S3 的「移动到初始位置」提供当前可校验的依据。

## 依赖（前置）

- 无硬前置；当前 exact-query replay 与树形 PTDL 持久化是本阶段基线。

## 当前已实现

### 完整 RTDL 树保存

- Pilot 已序列化 Executor `Plan`，保存 `sequence` / `parallel` / `do` 结构、capability 参数、步骤摘要和 `raw_rtdl`。
- 每个 planning round 以自己的 `plan_id` 保存为 query 根记录下的子计划；聚合状态包含成功、失败和取消计数。
- `PtdlStore` 已支持旧扁平记录迁移、树形记录更新、关键字检索和重启恢复。

### exact-query replay

- Pilot 在普通 VLM 规划前检查完全相同的 query。
- 当历史记录包含合法且成功的 RTDL 子计划时，Pilot 重建新的 Executor `Plan` 并直接 dispatch。
- 该路径属于历史计划 replay，不等于 `SkillTemplate` 提炼或相似任务技能匹配。

### 当前接口状态

- Memory service 动态声明 `ptdl_remember` / `ptdl_retrieve` capability。
- 两个 capability 目前没有对应 TOML contract，也未列入 `services/memory/package_manifest.yaml` 的静态导出，需要在本阶段确定正式接口归属。

## 剩余子任务

### 1.1 补齐执行上下文字段

在现有 PTDL 子计划记录中增加：

- 起始和结束机器人位姿（含 frame）。
- 起始和结束 Scene 对象/区域摘要。
- 前置条件和后置条件。
- 任务时间范围。
- Executor 最终状态与 verifier 结果。
- 记录格式版本与旧格式标记。

沿用已有 `CameraPose`、`SpatialContext`、`ObjectCoord`、`TimeRange` 等类型；不再重复增加另一份 RTDL AST 字段。

### 1.2 保持 PtdlStore 的树形结构

- 保留当前 query 根记录 + planning round 子计划结构。
- 为子计划扩展执行上下文和格式版本。
- 旧扁平 `steps` 记录迁移后显式标识旧格式，旧记录可用于展示/检索，但默认不参与技能提炼。
- 保证新字段在服务重启后完整 round-trip。

### 1.3 在 Pilot 的权威生命周期点采集上下文

- 首次派发前通过当前 Scene/robot context 记录 start context。
- Executor 完成后记录 end context、最终状态和验证结果。
- 只有权威完成且验证通过的计划计入 S2 成功样本。
- 失败、取消、部分完成和仍在运行的计划可以保留审计记录，但不得标记为成功样本。

### 1.4 提取前置/后置条件

- 起始位置：记录当前机器人位姿和可解析的 Scene region/object 标识。
- 前置条件：主体流程所需的 capability、对象、区域和状态。
- 后置条件：任务 success criterion 与 verifier 结果。
- S3 使用这些条件判断是否需要先移动到初始位置；命名区域或对象必须重新通过当前 Scene 的 `goal_room` / `goal_near` 解析，不能直接执行历史坐标。

### 1.5 正式化接口

在以下两种方式中选择一条并删除另一条重复路径：

1. 为现有 `ptdl_remember` / `ptdl_retrieve` 增加正式 TOML contract、IDL、package export 和接口测试；或
2. 将其功能合并到既有 memory contract，同时保持树形 PTDL 的字段和行为。

本阶段不新增额外调度接口；执行仍走 Executor `execute` contract。

### 1.6 修正测试

- 更新仍假设根级 `steps` 的单元测试，改为检查 child plan 的 `steps` / `rtdl_plan` / `raw_rtdl`。
- 更新把计划当作普通图节点写入的旧端到端测试；计划当前由 PtdlStore 独立持久化。
- 增加执行上下文 round-trip、旧格式迁移、成功/失败/取消分类、重启恢复测试。
- 增加 Pilot → memory → Executor 的非 Webots replay 集成测试。

## 所需资源/工具

- `system/pilot/src/planner.rs`、`system/pilot/src/memory.rs`、`system/pilot/src/state_context.rs`
- `system/executor/src/service.rs`、`system/executor/src/rtdl_wire.rs`
- `services/memory/memory_service/storage/ptdl_store.py`
- `services/memory/memory_service/core/types.py`
- `services/memory/memory_service/service.py`
- `system/pilot/rtdl_protocol.md`

## 预期交付物

- 含执行上下文、结果和版本的树形 PTDL 计划记录。
- 正式且唯一的 PTDL memory capability 路径。
- 更新后的单元测试与非 Webots 集成测试。

## 验收标准

- 完整 RTDL、起点/终点上下文和验证结果在重启后可无损读回。
- 成功、失败、取消和部分完成不会混淆。
- 旧格式记录能够迁移，但不会误进入 refine 成功样本。
- exact-query replay 现有能力不回退。

## 时间节点（可选）

- 相对估算：约 2–3 人日；验收后再进入 S2。
