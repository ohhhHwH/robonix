# 文档 03｜技能 → RTDL 实例化与前置步骤

> 阶段 S3：命中技能后，把 `SkillTemplate` 还原成一棵可直接执行的 RTDL 树，并在最前面插入「移动到初始位置」节点。

## 阶段目标

实现 skill→RTDL 实例化器：`goal_params` 填参、`causal_template` 还原为 `sequence`/`parallel`/`do`；用 S1 记录的起始位姿生成 `navigate_to_goal(初始位姿)` 作为 root `sequence` 的第一个 child；对能力名做当前部署下的重绑定与参数校验。

## 依赖（前置）

- S1（取 `pre_condition` 起始位姿）、S2（取 `SkillTemplate.causal_template`/`goal_params`）。

## 具体子任务

### 3.1 实现 skill→RTDL 实例化器
- 新增模块（Pilot 侧或 memory 侧内部函数，不新增对外 RPC）：
  - `goal_params` 填充 `goal_template` 的占位参数。
  - `causal_template`（`StepTemplate[]`，含 `order`/`capability_id`/`parallel_group`/`depends_on_step`）还原为 RTDL 树：`parallel_group` → `parallel`，`depends_on_step` → `sequence` 顺序，叶子为 `do{cap,args}`。
- 遵循 `system/pilot/rtdl_protocol.md`：每节点 `op_id=0`、`description` 非空；`do.cap` 必须逐字复制能力名（含 provider 前缀）。

### 3.2 前置步骤注入（移动至初始位置）
- 用 S1 的 `pre_condition.start_pose` 生成导航节点，作为 root `sequence` 第一个 child：
  - 若记录了 room_id → 走 Scene `goal_room` 取可达位姿再导航（遵循 rtdl_protocol 第 9 条：场景数据权威）。
  - 若记录的是坐标 → 直接 `navigate_to_goal(位姿)`。
- 保证该前置步骤在讲解/主体动作之前执行。

### 3.3 能力重绑定与降级
- 技能中的 `capability_id` 与当前「Available capabilities」逐一匹配（provider 前缀 + dot 名）。
- 缺失/不可用 → 降级回 Pilot 从零重新规划（不硬执行过期能力）。

### 3.4 参数校验
- 实例化出的 `do.args` 与当前 `args_schema` 对齐；不匹配则降级或抛错。

### 3.5 测试
- 单测：模板 → RTDL 树结构正确；断言 root 首节点为「移动至初始位置」；能力缺失触发降级。

## 所需资源/工具

- `system/pilot/rtdl_protocol.md`、`system/pilot/src/planner.rs`
- `system/scene/scene_service/mcp_tools.py`（`goal_room`/`goal_near`）
- `capabilities/system/executor/execute.v1.toml`（调度入口）

## 预期交付物

- skill→RTDL 实例化模块 + 单测（含「移动至初始位置」前置步骤断言）。

## 时间节点（可选）

- 相对估算：约 3 人日；阻塞 S4。
