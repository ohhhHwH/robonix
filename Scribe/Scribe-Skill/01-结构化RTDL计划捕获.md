# 文档 01｜结构化 RTDL 计划捕获

> 阶段 S1：让「成功执行完的 RTDL 计划」以**结构化、可重放**的形式落库，取代现在的文本摘要。这是后续提炼技能（S2）与实例化执行（S3）的数据基础。

## 阶段目标

任务成功完成后，把当次的**整棵 RTDL 树**（`sequence` / `parallel` / `do` 的 `op`/`cap`/`args`）+ 执行上下文（成功标记、起始/结束位姿、涉及物体、时间窗、相机参数）持久化，使它能被重新实例化执行，而不是只能当文本读。

## 依赖（前置）

- 无硬前置；与 S4 的检索部分可并行。

## 具体子任务

### 1.1 定义计划结构化 schema
- 在 `types.py` 现有类型上扩展，新增「计划」维度的字段（沿用已有 `CameraPose`/`CameraParams`/`SpatialContext`/`ObjectCoord`/`TimeRange`）：
  - `rtdl_ast`：序列化后的 RTDL JSON（`sequence`/`parallel`/`do` 树）。
  - `pre_condition`：起始位姿（坐标或 `goal_room` 的 room_id）、起始 scene_type/objects——**S3 的「移动到初始位置」从这里取**。
  - `post_condition`：结束态（终点位姿、任务完成判据）。
- 结构只定义「是什么」，算法/存储选型遵循 `Scribe-Mem-struct.md` 的原则。

### 1.2 扩展 PtdlStore 存 AST
- 改造 `storage/ptdl_store.py`：`add()` 从存 `steps: List[str]` 文本 → 同时存 `rtdl_ast` + 元数据；`search()` 保留关键字召回，但返回结构化条目。
- 兼容旧数据：读取时对只含文本 `steps` 的历史条目标记 `legacy=true`，不参与技能提炼。

### 1.3 在 Pilot 完成处触发落库
- 挂接 `system/pilot/src/planner.rs`：规划任务 `done` 时，把当次派发的 RTDL 树 + 成功标记写入记忆（对应文档「remember 调用者 = Pilot（规划后）」）。
- 复用现有 `robonix/service/memory/remember`：以 `kv.task_type="plan"` 标记，节点类型用 `NodeType.SKILL` 的源经验（或现有 `LESSON`/`SHORT_TERM` 计划节点），避免两套存储。

### 1.4 提取前置/后置条件
- 起始位姿：从任务开始时的导航目标 / Scene `list_regions`+`goal_room` 结果写入 `pre_condition`。
- 后置条件：任务成功判据（如「已到达终点位姿」「讲解序列执行完毕」）。
- 涉及物体：从 Scene Hook 观察（`spatial_data`）合并进计划节点。

### 1.5 测试
- 用真实 RTDL 树（非文本）替换 `tests/test_e2e_plan_save.py` 的模拟文本，断言结构化字段（`rtdl_ast`、`pre_condition`）落库后可读回。

## 所需资源/工具

- `system/pilot/src/planner.rs`、`system/pilot/src/state_context.rs`
- `system/executor/src/rtdl_wire.rs`（RTDL AST 序列化参照）
- `services/memory/memory_service/storage/ptdl_store.py`、`core/types.py`
- `system/pilot/rtdl_protocol.md`（RTDL 节点语义）

## 预期交付物

- 结构化 PlanNode：`rtdl_ast` + `pre_condition`/`post_condition` + 元数据。
- 改造后的 `ptdl_store.py` + 单测/端到端测试。

## 时间节点（可选）

- 相对估算：约 2–3 人日；与 S4 检索部分并行，建议最先启动。
