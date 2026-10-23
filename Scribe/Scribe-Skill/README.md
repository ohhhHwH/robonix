# Scribe-Skill — 场景技能化（RTDL → Skill → 复用）阶段拆解

> 目标：把成功执行的讲解类 RTDL 计划提炼成可复用技能；后续用户提出相同或相似任务时，校验当前 capability 与场景条件，必要时先移动到记录的初始位置，再直接执行实例化后的 RTDL。

## 阶段总览与依赖关系

```text
S1 结构化计划捕获 ──► S2 技能提炼 ──► S3 技能→RTDL 实例化 ──► S4 检索匹配与直接调度 ──► S5 讲解端到端验证 ──► S6 毕设评测对齐
```

- **强串行**：S1 → S2 → S3 → S4 → S5。
- **可并行**：S6 的指标定义可提前开始；实现与实验必须等待 S1–S5 的接口稳定。

## 文档索引

| 序号 | 文档 | 阶段 | 核心产出 |
|---|---|---|---|
| 01 | [结构化 RTDL 计划捕获](01-结构化RTDL计划捕获.md) | S1 | 完整 RTDL 树 + 执行上下文 |
| 02 | [技能提炼 refine](02-技能提炼refine.md) | S2 | `refine` / `list_skills` contract + SkillStore |
| 03 | [技能→RTDL 实例化与前置步骤](03-技能到RTDL实例化.md) | S3 | skill→RTDL 实例化器（含移动至初始位置） |
| 04 | [检索匹配与直接调度](04-检索匹配与直接调度.md) | S4 | 「问 → 技能命中 → 直接调度」路径 |
| 05 | [讲解场景端到端验证](05-讲解场景端到端验证.md) | S5 | 可复现 demo + 运行记录 |
| 06 | [毕设评测对齐](06-毕设评测对齐.md) | S6 | 实验数据 + 论文小节 |

## 当前实现基线（2026-10-09）

### 已落地

- Pilot 已在计划结束时保存完整 Executor `Plan`，包括 `sequence` / `parallel` / `do` 结构、capability 参数、`raw_rtdl`、步骤摘要和每轮 `plan_id`。
- `PtdlStore` 已按 query 根记录 + planning round 子计划持久化到 `ptdl_store.json`，支持旧扁平记录迁移、分轮汇总和检索。
- Pilot 已支持**完全相同 query** 的历史 RTDL 直接 replay：命中合法的成功历史计划时绕过 VLM 规划并直接交给 Executor。
- 模糊检索结果已经可以注入 Pilot 上下文，但目前不会直接执行。
- Memory Phase 1 的 `remember`、`hybrid_search`、`promote`，以及 Scene Hook、VLM 观察、图片和图存储均已存在。

### 已落地但尚未正式化

- Memory service 动态声明了 `ptdl_remember` / `ptdl_retrieve` capability；它们尚无对应 TOML contract，也未列入 `services/memory/package_manifest.yaml` 的静态导出。
- exact-query replay 是“历史计划重放”，不是 `SkillTemplate` 提炼、参数化和相似任务匹配。

### 尚未完成

| 阶段 | 状态 | 主要缺口 |
|---|---|---|
| S1 | 部分完成 | 缺起点/终点位姿、Scene 对象/区域上下文、前置/后置条件和权威验证结果；部分测试仍假设旧的根级 `steps`/图节点保存 |
| S2 | 未实现 | `SkillTemplate` 仅为占位类型；无 refine、SkillStore、list_skills、N≥3 聚类/LCS/验证及测试 |
| S3 | 部分基础 | 历史 Plan 可重建，但未按当前 Atlas 重新绑定；无参数替换、初始位置前置步骤、args schema 校验和降级 |
| S4 | 部分完成 | 只有 exact-query replay；无相似 query 技能匹配和明确阈值；历史 replay 执行失败时不能可靠回退普通规划 |
| S5 | 未实现 | 无导航+相机+讲解的首轮学习/后续复用闭环；Webots manifest 中 speech 仍未启用 |
| S6 | 未实现 | 无基线、消融、重复实验、统计分析和论文结果 |

## exact-query replay 与技能复用的边界

| 能力 | 当前 exact-query replay | 目标 SkillTemplate 路径 |
|---|---|---|
| 输入匹配 | trim 后的 query 完全相等 | 相同或相似任务，带明确匹配阈值 |
| 执行内容 | 原样重建历史 RTDL Plan | 按模板参数实例化新的 RTDL |
| capability 处理 | 复用历史 provider / contract ID | 按当前 Atlas capability 重新绑定并校验 schema |
| 场景条件 | 不补充起点/前后置条件 | 校验当前 Scene，必要时先移动到初始位置 |
| 失败处理 | 历史步骤失败可能直接结束任务 | 实例化/校验/执行失败时回退普通 Pilot 规划 |
| 技能形成 | 单次成功历史即可 replay | N≥3 成功经验经 refine 后生成 SkillTemplate |

## 剩余 TODO（建议实施顺序）

### P1｜补齐 S1 执行上下文

- [ ] 在每个成功 RTDL 计划记录中持久化起始/结束机器人位姿、Scene 对象/区域上下文、时间范围和任务成功判据。
- [ ] 仅在 Executor 权威完成且验证通过后标记成功；失败、取消、部分完成不得成为 refine 成功样本。
- [ ] 为现有树形 PTDL 记录增加版本和明确的旧格式识别，不再以“是否存在根级 `steps`”隐式判断。
- [ ] 修正 `test_e2e_plan_save.py`、`test_remember.py` 等仍依赖旧保存路径或旧字段位置的测试。
- [ ] 将既有 `ptdl_remember` / `ptdl_retrieve` capability 正式化为 contract，或明确合并到既有 memory contract；完成后只保留一条接口路径。

**验收**：重启后完整 RTDL 与执行上下文可 round-trip；成功/失败/取消分类准确。

### P2｜实现 S2 技能提炼

- [ ] 完成 `SkillTemplate` / `StepTemplate` / 条件与约束的序列化、版本和来源记录。
- [ ] 实现 `SkillStore`。
- [ ] 实现确定性 refine：同类任务归组、N≥3 门槛、动作序列 LCS、参数识别、前后置条件交集、成功率验证。
- [ ] 增加既有设计中的 `robonix/service/memory/refine` 与 `robonix/service/memory/list_skills` contract、handler、package export。
- [ ] 覆盖样本不足、结构不一致、成功率不足、持久化恢复和版本升级测试。

**验收**：3 条受控成功记录稳定生成一个可持久化、可列出的 `SkillTemplate`。

### P3｜实现 S3 技能→RTDL 实例化

- [ ] 参数化 `goal_template` 并将 `causal_template` 还原为 `sequence` / `parallel` / `do`。
- [ ] 每一步按当前 Atlas capability 重新绑定 provider / contract，不复用过期 ID。
- [ ] 按当前 `args_schema` 严格校验实例化参数。
- [ ] 使用 Scene 的 `goal_room` / `goal_near` 解析位置；需要时在主体 RTDL 前加入当前 `robonix/service/navigation/navigate` 起点导航。
- [ ] capability 缺失、schema 漂移或前置条件不满足时降级到普通 Pilot 规划。

**验收**：实例化的 RTDL 在当前部署下通过能力与参数校验，且必要时首节点为起点导航。

### P4｜完成 S4 相似匹配、直接调度与回退

- [ ] 保留 exact-query replay 作为最快路径。
- [ ] 增加标签/向量候选和明确阈值的相似技能匹配；阈值以下视为未命中。
- [ ] 技能命中后执行“实例化 → 校验 → dispatch”，不再让 VLM 从零生成同一流程。
- [ ] 实例化失败、执行失败、前置条件失败时自动回退普通规划。
- [ ] 增加 hit、miss、误匹配、stale capability、schema drift 和 fallback 测试。

**验收**：相似问法可命中同一技能，所有失败路径可安全回退。

### P5｜增加非 Webots 闭环集成测试

- [ ] 覆盖“成功计划捕获 ×3 → refine → list_skills → 相似 query 命中 → 实例化 → 选择直接 dispatch”。
- [ ] 覆盖失败/取消计划不入样本、能力失效、参数变化、起点不满足和普通规划回退。

**验收**：不依赖 Webots 或外部 LLM 即可重复验证完整逻辑闭环。

### P6｜完成 Webots 讲解 Demo

- [ ] 启用 speech 服务，固定一条导航+相机+讲解的可复现短路线。
- [ ] 首轮累计成功经验并提炼技能；后续以相似问法触发复用。
- [ ] 验证“移动到初始位置”与后续讲解 RTDL 的执行顺序。
- [ ] 记录规划轮次、模型调用/token、端到端耗时、成功率和 fallback。

**验收**：首轮学习、后续复用和异常回退都有可复现记录。

### P7｜完成 S6 毕设评测

- [ ] 基线：No-Memory、普通检索、exact replay、Scribe-Skill Full。
- [ ] 消融：无 refine、无起点步骤、无 schema 校验、无 fallback。
- [ ] 指标：成功率、规划轮次、模型调用/token、端到端耗时、冷启动成本。
- [ ] 自动化重复试验，输出原始数据、Bootstrap、McNemar、Cohen's d、表格和论文小节。

**验收**：实验可重复，统计脚本可从原始数据生成论文结果。

## 术语与约束

- 沿用 dev guide 中的 capability / contract / primitive / service / skill 和 RTDL，不新增平行概念。
- `refine`、`list_skills`、`remember`、`search` 沿用已有设计中的 contract 名；不为调度新增接口。
- 调度继续使用 Executor 的 `robonix/system/executor/execute` contract。
- 位置必须由当前 Scene 数据解析；历史坐标不能替代当前 `goal_room` / `goal_near` 结果。
- S1/P1 验收通过后再进入 S2；S2–S4 稳定后再启动 Webots 和评测工作。
