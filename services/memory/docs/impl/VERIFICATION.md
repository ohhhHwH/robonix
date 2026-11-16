# 向量检索 Phase 0 — 真实后端实机验证记录（2026-10-26）

按用户要求，对**真实运行中的服务**做一次端到端验证：不走 `observe` / `list_objects`
自动保存，而是直连 MCP 工具 `robonix/service/memory/remember` 与 `ptdl_remember`
（即直接构造 `MemoryNode` / plan 写入），再核对 MongoDB / Milvus 实际落库情况。

## 环境

- mongod `127.0.0.1:27017`（库 `robonix_memory`）
- Milvus-lite `./memory/robonix_memory.db`（首次创建）
- BGE-M3 ONNX int8 已加载：`bge.semantic=True, dim=1024`（真语义向量，非 hash 回退）
- `HF_ENDPOINT=https://hf-mirror.com`
- 依赖：pymongo 4.19.0 / pymilvus 3.0.2 / milvus-lite 3.2.2 / onnxruntime 1.31.0 /
  memsearch 0.4.22 / numpy 2.5.4

验证脚本为一次性脚本（`/tmp/mem_verify.py`），隔离了 JSON 回退目录
（`AGENT_MEMORY_DIR=<tmp>`、`MEMGRAPH_KEEP_DATA=1`），Mongo / Milvus 走真实默认值，
确保验证的是生产 wiring（`service.py` 构造的 `_mongo/_milvus/_bge/_graph/_ptdl/_images`
→ `RememberPipeline`）。

## 三项验证结果

### 1. 直接 `remember` 直连生成能否存储 — ✅

`remember`（带 `image_base64`，1×1 PNG，67B）→ `node_id=0`。

- Mongo `scene_nodes`：新增 1 条节点文档（`_id=0`，`node_type=short_term`）
- 图片 Binary：`image_store: wrote mongo:0:0 (0.1 KB)` → `scene_nodes.images` 存 BSON Binary
- Milvus `memory_vectors`：新增 1 行 `doc_id="0", kind="scene"`

### 2. 探索任务各记忆落库 — ✅

模拟一次「去厨房探索并拿红杯子」任务，三种记忆各归其位：

| 记忆 | MCP 工具 | Mongo | Milvus `memory_vectors` |
|---|---|---|---|
| 规划记忆 | `ptdl_remember` | `plan_tasks` +1（`_id=sha1(query)`） | `kind=plan` +1 |
| 场景记忆 | `remember`（`parent_node_id` 生成因果边） | `scene_nodes` +1（`_id=1`） | `kind=scene` +1 |
| 因果边 | `remember` 的 parent 链路 | `causal_edges` +1（`0->1`） | — |
| 图片 | `remember` 的 `image_base64` | `scene_nodes.images` Binary + `image_count` 标量 + `mongo:{nid}:{idx}` 懒加载 marker | — |

### 3. 数据库 dump — ✅

| 存储 | BEFORE | AFTER |
|---|---|---|
| Mongo `scene_nodes` | 0 | **2**（node 0 / 1，均为 short_term） |
| Mongo `causal_edges` | 0 | **1**（`0->1`, relation=enables） |
| Mongo `plan_tasks` | 17（历史遗留，见下） | **18**（+本次探索计划） |
| Milvus `memory_vectors` | 0 | **3**（2 scene + 1 plan，写入路径实时 upsert） |

样例（AFTER，截断）：

```text
scene_nodes  {_id:0, node_type:"short_term", causal_chain:[], images:[<Binary 67B>], ...}
scene_nodes  {_id:1, node_type:"short_term", causal_chain:[0], ...}
causal_edges {_id:"0->1", from:0, to:1, relation:"enables"}
plan_tasks   {_id:"<sha1>", query:"VERIFY-... 去厨房探索并拿红杯子", ...}
```

`vector_search(query="red cup in kitchen", kind=all, top_k=10)` 命中 3 条：

```text
doc_id=0       kind=scene  node_type=short_term  score=0.8524
doc_id=<sha1>  kind=plan                          score=0.6697
doc_id=1       kind=scene  node_type=short_term  score=0.4246
```

## 附带发现：`on_init` 全量重建把历史 plan 补进 Milvus — ✅

Mongo `plan_tasks` 在验证前已有 17 条历史 plan（来自更早的 `ptdl_store.json` 一次性
backfill），但 Milvus 是本次才新建的，那 17 条 plan **尚无向量**（Milvus 0 行）。

调用 `_on_init` 后：

```text
milvus BEFORE on_init: 3 rows   （本次写入路径实时同步的 2 scene + 1 plan）
on_init -> Ok()                  （rebuild: 2 scene + 18 plan）
milvus AFTER on_init : 20 rows   （2 scene + 18 plan，17 条历史 plan 被补写）
```

这验证了设计 §5/§7：Milvus 是派生索引，丢失/新建后可由 `on_init` 从 Mongo 全量重建
（`_graph.rebuild_milvus()` + `_ptdl.rebuild_milvus()`），历史 plan 向量被正确补齐。

## 测试数据标记

本次写入的测试条目统一带 `VERIFY-1792980606` 前缀（scene node 0/1 的 `raw_log.msg`、
探索计划 `query`）。其中 scene/edge 属 ephemeral（下次 clean_start 自动清），plan 属
长期记忆会保留 —— 如需清理，按 `query` 前缀 `VERIFY-1792980606` 删除即可。
