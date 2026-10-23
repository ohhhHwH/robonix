# 向量存储与检索设计方案 v3（MongoDB 元数据+RGB图片 + Milvus 向量）

> 目标：为 `services/memory`（Scribe Mem / memgraph）建立存储——**MongoDB 存节点权威
> 元数据 + RGB 图片（直接 Binary 存），Milvus 存向量索引（引用回 MongoDB）**，覆盖
> 场景记忆与规划记忆。本期 embedding 仍只对 summary/描述文本做；深度图 / 视频 clip
> 暂无存储需求，预留 GridFS 备注（见 §3 备注）。

## 1. 总体架构：分工

```text
写入： 节点/任务 ──► MongoDB（权威文档 + RGB 图 Binary）──► embed summary ──► Milvus(vector + _id + 标量)
读取：  query ──► embed ──► Milvus ANN(+filter) ──► 返回 (doc_id, score)
                               ──► MongoDB $in 批量取回完整文档（含 RGB 图）──► 排序返回
```

| 层 | 职责 | 存什么 | 不存什么 |
|---|---|---|---|
| **MongoDB** | 权威元数据源 + RGB 图片 | 节点/任务全字段、因果边、RGB 图（Binary 字段） | 向量 |
| **Milvus Lite** | 纯 ANN 向量索引 | 向量 + doc_id 引用 + 少量 filter 标量 | 原文/大字段 |

核心原则：**MongoDB 是唯一事实源，Milvus 是派生索引**。Milvus 不存原文，只存向量 +
指向 MongoDB 的 id；原文与图片永远从 MongoDB 取。Milvus 丢了可从 MongoDB 全量重建
（沿用现有 `on_init` rebuild 思路）。

## 2. 为什么引入 MongoDB

- 文档型、schema 灵活，天然适配 `MemoryNode` 这种带可选字段（tags/spatial/camera）的异构结构。
- 直接把 RGB 图片作为 Binary 字段存进节点文档，随文档一起读写、查询、备份，无需另配文件系统。
- 支持按字段查询与聚合（node_type / timestamp / region / task 等），因果边可作数组/集合存。
- 与 memsearch 服务统一技术栈方向：Mongo 管元数据、Milvus 管向量。

## 3. MongoDB 集合设计

```text
scene_nodes（场景记忆，替代 graph_store.json）
  _id          INT64          # = node_id
  summary      String
  summary_zh / summary_en     String
  tags         Object         # TagSet 四维
  spatial_data Object
  timestamp    Int64          # ns
  node_type    String
  causal_chain Array[Int64]   # 父节点 id
  weight       Double
  images       Array[Binary]  # RGB 相机帧（BSON Binary，顺序对应 frame_ts）
  frame_ts     Array[Int64]   # 每帧捕获时间戳（ns）
  depth_refs   Array          # 占位，当前不存储
  video_clip_refs Array       # 占位，当前不存储
  version      Int64          # 乐观锁

causal_edges（可选，加速 child/parent 查询）
  { _id, from: Int64, to: Int64, relation: String }

plan_tasks（规划记忆，替代 ptdl_store.json）
  _id          ObjectId（或 sha1(query)）
  query        String
  description  String
  plan_count / canceled_count  Int64
  timestamp_ns Int64
  plans        Array[Object]   # 子计划全量（含 rtdl_plan / raw_rtdl / steps）
```

> **备注（预留，当前不用）**：GridFS（MongoDB 内置 `fs.files` + `fs.chunks` 大文件规范）
> 留作未来存 **深度图 / 视频 clip / 可能超 16MB 的文件**。当前设计**只让 MongoDB 以
> Binary 字段存 RGB 图片**，不引入 GridFS。

## 4. Milvus 集合设计（只存向量 + 引用）

```text
collection memory_vectors（统一；或按 scene/plan 拆两个 collection）
  doc_id    VARCHAR(32)      # 指向 MongoDB 的 _id（node_id 或 plan ObjectId）
  kind      VARCHAR          # "scene" | "plan"
  node_type VARCHAR          # 可选标量，供 ANN 前 filter
  timestamp INT64            # 可选标量，供时间窗口 filter
  vector    FLOAT_VECTOR(1024)
```

- 要 embed 的文本（本期）：场景 = `summary_zh + " / " + summary_en`（缺省回退 summary）；
  规划 = `query + " " + description`。
- Milvus 里**不存** summary/query/description 原文，也不存图片；命中后按 `doc_id` 从
  MongoDB 取回。

## 5. 写入路径（改造现有持久化后端，管道其余不动）

- `GraphStore`：保留内存 `_nodes` dict 与 API，仅把 `_persist`/`_load` 从
  `graph_store.json` 换成 MongoDB `scene_nodes`（`add_node`/`update_node` 后 upsert）。
  新增节点成功后 → `embed(summary)` → Milvus `insert(doc_id, vector, kind/node_type/timestamp)`。
- `PtdlStore`：`_entries` 仍在内存，`_save`/`_load` 换成 MongoDB `plan_tasks`；
  `add()` 成功后 → `embed(query + description)` → Milvus upsert（以 query 为键）。
- `ImageStore`：RGB 相机帧不再落盘文件，改为写入节点文档的 `images` Binary 字段
  （随 `scene_nodes` upsert 一并保存）；读取时随节点文档取回。深度图/视频当前不存。

## 6. 检索路径

- 新增 MCP 工具 `robonix/service/memory/vector_search`：
  `query` + `kind`（scene/plan/all）+ `top_k` + 可选 `time_range`/`node_type`
  → embed → Milvus ANN(+filter) → `doc_id` 列表 → MongoDB `$in` 取回完整文档
  （含 `images`）→ 返回 summary/描述 + score + 元数据。
- `retrieve.py` Path B/C：embedding 启用时用 Milvus 混合检索（dense + BM25，
  `WeightedRanker`）替代 numpy 余弦；保留 `TagIndex` 预过滤（转成 Milvus `filter=`）。
- `ptdl_retrieve`（规划记忆重放，三级判定）：
  1. **精确命中**：query 与 `plan_tasks.query` 完全一致 → 直接返回保存的 `plans[]`
     （Pilot 跳过重新规划）。
  2. **高相似重放**：向量相似度 ≥ `PTDL_REPLAY_SIM`（暂定 0.90）→ 视为等价任务，
     直接执行该历史规划。
  3. **相似参考**：相似度 < 0.90 但存在 > `PTDL_REFERENCE_SIM`（暂定 0.80）的历史
     规划 → 取相似度最高的那条**附带到上下文**，供 Pilot 参考历史规划做下一轮规划
     （不直接执行）；无 > 0.80 命中则全新规划。两阈值均走环境变量可配。

## 7. 迁移与现有 JSON 文件

- MongoDB 成为权威后，`graph_store.json` / `ptdl_store.json` / `text_index.txt` 退役。
  - 平滑迁移：启动时若 MongoDB 集合为空但旧 JSON 存在 → 一次性 backfill 导入；
    之后 Mongo 为准（图片从 `data/images/` 一并读入 Binary）。
  - `text_index.txt`（LLM Tier1 扫描用）可由 MongoDB 聚合即时生成。
- 注意 `clean_start` 语义：现在是「清 graph/images，保留 plan」——迁移后改为按
  collection 清空（scene_nodes/images 可清，plan_tasks 保留）。
- 注意 `ptdl_retrieve` 的 exact-match 记忆重放（Pilot 跳过重规划）依赖 ptdl 树，迁移
  到 MongoDB 后该路径必须保持一致（query 精确匹配仍返回同一 `plans`）。

## 8. 分阶段落地

- **阶段 0（本期）**：MongoDB（元数据 + RGB 图 Binary）+ Milvus Lite + BGE-M3 ONNX；
  `GraphStore`/`PtdlStore`/`ImageStore` 换成 Mongo 持久化；新增 `vector_search` 工具；
  `on_init` 从 Mongo 全量重建 Milvus + 旧 JSON backfill。
- **阶段 1**：接入 `RetrievePipeline` Path B + ptdl 语义兜底。
- **阶段 2（后续）**：图片 CLIP、raw_log、BGE-M3 稀疏/colbert、深度图/视频（走 GridFS）。

## 9. 部署与依赖

- MongoDB：单节点 `mongod`（Docker，`--wiredTigerCacheSizeGB` 限内存），
  URI `mongodb://127.0.0.1:27017/robonix_memory`。
- 依赖：`pymongo`（async 可加 `motor`）、`pymilvus`（milvus-lite）、
  `onnxruntime`（分架构 pin，参考 memsearch）。BGE-M3 走 ONNX 后可去 sentence-transformers。

## 10. 风险与对策

1. **双写一致性**：Mongo 先写、Milvus 后写；Milvus 失败可下次 rebuild 补齐（最终一致），
   或写后校验缺失向量。
2. **RGB 图体积**：单帧 JPEG 压缩存储；控制单文档总大小 <16MB（避免逼近 BSON 上限，
   必要时才走 GridFS）。
3. **MongoDB 无嵌入式模式**：边缘端需跑 mongod 进程，注意资源占用（限 cache + 存储卷）。
4. **跨库无事务**：Mongo 文档与 Milvus 向量无法原子提交，靠「先 Mongo 后 Milvus +
   可重建」保证不丢数据。
5. **EmbeddingModelConfig.dim**：384→1024，`_hash_embedding` fallback 维度同步改。
6. **BM25 中文分词**：Milvus 原生 BM25 需 jieba 分词，或依赖 BGE-M3 dense+sparse。

## 附录：流程图

### 存储流程（场景 / 规划 → 汇聚到 Milvus）

```text
 场景信息                          规划信息
 remember(image,log,tags)          ptdl_remember(query,desc,steps,rtdl)
        │                                  │
        ▼                                  ▼
 生成 MemoryNode                     生成任务树记录
 (summary/zh/en,tags,spatial)        (query+description+plans[])
        │                                  │
 ① Mongo 写 scene_nodes             ① Mongo 写 plan_tasks
    (upsert) → _id                      (upsert by query) → _id
        │                                  │
 ② RGB 图存 images Binary 字段             │
    （随节点文档一起存）                     │
        │                                  │
        ▼                                  ▼
 ③ embed(summary_zh+"/"+en)         ③ embed(query+" "+description)
        │                                  │
        └───────────────┬──────────────────┘
                        ▼  vector(1024)
        ┌────────────────────────────────────────────┐
        │ ④ Milvus insert(doc_id, kind, vector,      │
        │              node_type, timestamp)          │
        │  doc_id = MongoDB _id（Milvus 只存向量+引用） │
        └────────────────────────────────────────────┘
```

### 检索流程（统一入口，命中后回 MongoDB 取原文）

```text
 vector_search(query, kind, top_k, time_range?)
        │
        ▼
 ① embed(query) → vector(1024)
        │
        ▼
 ② Milvus ANN 检索
    filter: kind=scene/plan (+ node_type / timestamp)
        │
        ▼
    [(doc_id, score) × top_k]
        │
        ▼
 ③ MongoDB $in 批量取回完整文档
    scene_nodes 或 plan_tasks（含 images）
        │
        ▼
 ④ 排序/截断 → 返回 summary/描述 + score + 元数据
        │
        └─►（可选）带 images 交 VLM 看图作答
```

### ptdl_retrieve 特殊路径（记忆重放，三级判定）

```text
 ptdl_retrieve(query)
        │
        ▼
 query 与 plan_tasks.query 精确匹配？
        │ 是                                  │ 否
        ▼                                     ▼
 直接返回保存的 plans[]               ① embed(query) → 向量
（Pilot 跳过重新规划）                        │
                                              ▼
                             ② 取相似度最高的历史规划
                                              │
                        ┌─────────────────────┼─────────────────────┐
                   sim ≥ 90%            80% < sim < 90%          sim ≤ 80%
                        │                     │                     │
                        ▼                     ▼                     ▼
                  执行该历史规划        取最相关的历史规划          无相关历史
                  （直接重放）          附带到上下文，             （全新规划）
                                       供下一轮规划参考
```
