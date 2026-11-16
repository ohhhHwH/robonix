# 向量存储与检索 — Phase 0 实现日志

对应设计文档 `VECTOR_RETRIEVAL_DESIGN.md`（v3）。本期只做 **阶段 0**（§8）：
MongoDB 元数据 + RGB 图 Binary、Milvus 向量索引、BGE-M3 ONNX、`vector_search`
MCP 工具、`on_init` 全量重建 + 旧 JSON backfill、§5.1 生命周期同步与 §6 图片懒加载。

阶段 1（RetrievePipeline Path B + ptdl 三级语义）与阶段 2 不在本期范围。

## 落盘 / 新增文件

| 文件 | 内容 |
|---|---|
| `memory_service/storage/mongo_backend.py` | `MongoBackend`：`scene_nodes` / `causal_edges` / `plan_tasks` 三集合，RGB 图以 BSON `Binary` 存节点文档 |
| `memory_service/storage/milvus_backend.py` | `MilvusBackend`：`memory_vectors` 集合（doc_id VARCHAR + kind + node_type + timestamp + FLOAT_VECTOR 1024） |
| `memory_service/storage/bge_embedder.py` | `BgeEmbedder`：封装 memsearch `OnnxEmbedding`，1024 维 BGE-M3 |
| `memory_service/storage/onnx_compat.py` | 从 memsearch 复刻的 aarch64 ONNX 线程数 workaround |

## 六个范围项落地

### 1. MongoDB 持久化（§3）

- `GraphStore` / `PtdlStore` / `ImageStore` 的 `_persist` / `_load` / `_save` /
  `_load` / `save` 全部改为「先写 Mongo、JSON 仅作 graceful-degradation 兜底」。
- `scene_nodes._id = int(node_id)`；`plan_tasks._id = sha1(query)`（§3 的两种键里选 sha1，
  与 `PtdlStore._doc_id` 语义一致）；`causal_edges._id = "from->to"`。
- RGB 相机帧以 `bson.Binary(subtype=0)` 存进 `scene_nodes.images` 数组，不再落盘文件。
- 节点文档额外维护一个 `image_count` 标量（`$inc`/`$dec`/`$set 0`），见 §6。

### 2. Milvus 向量索引（§4）

- `memory_vectors`：`doc_id` VARCHAR(64) 主键（放宽 §4 的 32 以容纳 node_id 十进制与
  sha1 十六进制）、`kind` VARCHAR(16)、`node_type` VARCHAR(64)、`timestamp` INT64、
  `vector` FLOAT_VECTOR(1024)。索引 `FLAT` + 距离 `COSINE`。
- `search()` 支持 `kind` / `node_type` / `time_range` 过滤（拼 `filter` 表达式）。

### 3. BGE-M3 embedding（§4.1）

- `EmbeddingModelConfig.model_name = "gpahal/bge-m3-onnx-int8"`，`dim = 1024`，
  `max_seq_length = 8192`；`_hash_embedding` 回退向量同步改为 1024。
- `TextEmbedder` / `VectorStore` 改委托 `BgeEmbedder`，去掉 sentence-transformers。
- 复用 memsearch 的 `OnnxEmbedding._encode`（同步方法），避免在运行中的 event loop 里
  `asyncio.run`。memsearch 未装或 onnxruntime 不可用时回退 hash embedding。

### 4. `vector_search` MCP 工具（§6）

- 新增 `robonix/service/memory/vector_search`：`query` + `kind`(scene/plan/all) +
  `top_k` + 可选 `time_range` / `node_type`。
- Milvus 命中 `doc_id` 后回 Mongo 取回文档（默认 projection 排除 `images`/`frame_ts`），
  按 score 排序返回 summary/描述 + 元数据。

### 5. `on_init` 重建 + backfill（§7）

- `on_init` 调用 `GraphStore.rebuild_milvus()` + `PtdlStore.rebuild_milvus()`：
  Mongo 为权威，逐节点/逐 plan 重新 embed 并 upsert 到 Milvus。
- 旧 JSON 一次性 backfill：Mongo 集合为空但本地 JSON 存在时，导入 JSON → Mongo。

### 6. §5.1 生命周期同步 + §6 图片懒加载

- 所有写路径「先 Mongo 后 Milvus」，Milvus 以 `doc_id` 为主键 `upsert`：
  add / update / promote / remove（delete）/ clear（`clear_kind`）。
- `promote_to_long_term` 采用 §5.1「备选（推荐）」的**稳定 `_id`**：不重编号，
  `node_type` 是唯一长短时判别（`SHORT_TERM → LONG_TERM`），id 不变，Milvus 仅
  `upsert` 同步 `node_type` 标量。
- 图片懒加载：默认 projection 排除 `images`/`frame_ts`；仅 `vlm_qa` 与
  `VLMFeatureExtractor` 按 `node_id` 通过 `ImageStore.read_bytes_by_ref` 取回。

## 关键设计决策

- **稳定 `_id` 取代 0–999 / 1000+ 区间语义**：单一单调 `_next_id` 计数器，
  `node_type` 单独承担长短时判别。消除 promote 时的 doc_id 迁移与 Milvus delete+insert。
  （omen-1 明确认可：「稳定 _id 决策没问题…双计数器并成单一单调 _next_id」。）
- **共享 `BgeEmbedder` 注入**：`GraphStore` / `PtdlStore` / `VectorStore` 共享同一个
  embedder 实例，避免重复加载模型。
- **`image_count` 标量**：默认 projection 排除 `images` 数组后，仍能据 `image_count`
  生成正确数量的不透明 marker（`mongo:{node_id}:{index}`），供懒加载按序取帧。
- **graceful degradation**：Mongo / Milvus / BGE-M3 全部可选；任一不可用时回退
  JSON 文件 / 内存 / hash 向量，服务在最小环境也能启动。
