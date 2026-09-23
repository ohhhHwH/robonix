# 文档 02｜技能提炼 refine

> 阶段 S2：把 N≥3 个同类成功计划抽象成可执行的 `SkillTemplate`，并暴露 `refine` / `list_skills` 接口注册为 MCP tool。

## 阶段目标

实现文档「核心链路——提炼与遗忘」中的提炼管线：按 `task_type` 聚类 → 抽象公共模式（前置条件取交集、动作序列取最长公共子序列 LCS、后置条件取交集）→ 模拟回放验证（成功率 ≥ 阈值才入库）→ 写入 SkillStore 并注册为 MCP tool。

## 依赖（前置）

- S1：需要结构化 PlanNode 作为提炼输入（`rtdl_ast` + 前后置条件）。

## 具体子任务

### 2.1 落地 Skill 数据类型
- `types.py` 已有 `SkillTemplate`/`StepTemplate`/`Condition`/`Constraint`/`SkillSummary`（当前是 Phase2 占位），补齐字段语义并确认序列化；对齐 `Scribe-Mem-struct.md §2.5`。

### 2.2 实现 refine 管线
- 新增 `core/refine.py`：
  - 聚类：按 `tags.task_type`（或 query 语义）分组。
  - 结构抽象：动作序列取 LCS（对齐 `Scribe/feat/ReMem/SKG.py` 的结构匹配思路：LCS 比率阈值）。
  - 条件抽象：`pre_condition` 取交集、`post_condition` 取交集。
  - 验证：模拟回放（用 S1 存的结构化计划做重放）计算 `success_rate`，≥ 0.8 才写入。
- 触发条件：同类成功经验 N≥3。

### 2.3 实现 SkillStore
- 新增 `storage/skill_store.py`：Skill 模板持久化（JSON，独立于 `graph_store.json`），提供 `put`/`list`/`get`/`update_version`。

### 2.4 新增契约与注册
- 新增 `capabilities/service/memory/refine.v1.toml`、`list_skills.v1.toml`（ID 沿用毕设文档：`robonix/service/memory/refine`、`robonix/service/memory/list_skills`）。
- `list_skills` 返回 `SkillTemplate[]`，经 Atlas 注册为 LLM 的 MCP function tools（对齐文档「关键设计」）。

### 2.5 测试
- 单测：N≥3 触发、LCS 抽象正确、success_rate 计算、阈值拒绝。
- 集成：S1 落库 → 触发 refine → SkillStore 出现模板 → `list_skills` 可列出。

## 所需资源/工具

- `services/memory/memory_service/core/types.py`（Skill 类型占位）
- `Scribe/feat/ReMem/SKG.py`、`SkillGenerate.md`（提炼/对齐算法参照）
- `capabilities/service/memory/`（契约风格）、`rbnx codegen`

## 预期交付物

- `refine` + `list_skills` contract、`core/refine.py`、`storage/skill_store.py` + 单测/集成测试。

## 时间节点（可选）

- 相对估算：约 3–4 人日；阻塞 S3。
