# Scribe-Skill — 场景技能化（RTDL → Skill → 复用）阶段拆解

> 目标：把一次成功的讲解类 RTDL 计划保存为可复用技能；后续用户问相同/相似问题时，直接调用技能实例化执行，并在最前面追加「移动到初始位置」步骤。对应毕设核心接口 `refine` + `list_skills` + 重复任务加速（文档 Demo 3/4/6/8）。

## 阶段总览与依赖关系

```
S1 结构化计划捕获 ──► S2 技能提炼 ──► S3 技能→RTDL 实例化 ──► S4 检索匹配与直接调度 ──► S5 讲解端到端验证 ──► S6 毕设评测对齐
```

- **强串行**：S1 → S2 → S3 → S4 → S5。
- **可并行**：S4 的检索部分可与 S1 并行开发；S6 的指标定义可与 S1 并行启动。

## 文档索引

| 序号 | 文档 | 阶段 | 核心产出 |
|---|---|---|---|
| 01 | [结构化 RTDL 计划捕获](01-结构化RTDL计划捕获.md) | S1 | 结构化 PlanNode（RTDL AST + 前后置条件） |
| 02 | [技能提炼 refine](02-技能提炼refine.md) | S2 | `refine` / `list_skills` contract + SkillStore |
| 03 | [技能→RTDL 实例化与前置步骤](03-技能到RTDL实例化.md) | S3 | skill→RTDL 实例化器（含移动至初始位置前插） |
| 04 | [检索匹配与直接调度](04-检索匹配与直接调度.md) | S4 | 「问 → 技能命中 → 直接调度」路径 |
| 05 | [讲解场景端到端验证](05-讲解场景端到端验证.md) | S5 | 可复现 demo + 记录 |
| 06 | [毕设评测对齐](06-毕设评测对齐.md) | S6 | 实验数据 + 论文小节 |

## 现状基线（为什么需要这些阶段）

- **已落地（Phase 1）**：`remember` + `hybrid_search`/`search` + `compact`/`promote` + Scene Hook + VLM 观察。
- **计划目前只以文本形式保存**：`services/memory/tests/test_e2e_plan_save.py` 把 `plan_steps` 文本写进 `kv`；`services/memory/memory_service/storage/ptdl_store.py` 存的是 `steps` 文本列表，**不是**可执行的 RTDL JSON AST。
- **缺**：结构化保存、refine/list_skills 实现、skill→RTDL 实例化、直接调度。`capabilities/service/memory/` 下无 `refine`/`list_skills`/`forget` contract；`types.py` 里 `SkillTemplate`/`ForgetRisk` 仍为 Phase2 占位。
- **可参考旧原型**：`Scribe/feat/ReMem/`（SKG 技能图：语义+结构对齐、promote/decay、三级分层）作为 S2 提炼算法参照。

## 术语与约束

- 契约名沿用毕设文档既有接口，不新增 RPC 名：`robonix/service/memory/refine`、`robonix/service/memory/list_skills`、`robonix/service/memory/remember`、`robonix/service/memory/search`（代码侧对应 `hybrid_search`）。
- 调度走既有 Executor 的 `execute` 契约（`capabilities/system/executor/execute.v1.toml`），不另建调度接口。
- 概念沿用 dev guide：capability / contract / primitive / service / skill；RTDL（`system/pilot/rtdl_protocol.md`）。
