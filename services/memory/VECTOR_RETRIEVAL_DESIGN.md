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
  images       Array[Binary]  # RGB 相机帧（BSON Binary，顺序对应 frame_ts；检索默认 projection 排除、懒加载）
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

### 4.1 嵌入模型选型：BGE-M3（中英双语）

统一采用 **BGE-M3**（`BAAI/bge-m3`）作为文本嵌入模型，替代英文单语的
`all-MiniLM-L6-v2`：

- **模型**：`BAAI/bge-m3`，ONNX int8 量化（`gpahal/bge-m3-onnx-int8`），
  **维度 1024**，中英双语、多粒度，同时输出 dense + sparse（colbert 预留）。
- **复用 memsearch**：与 `services/memsearch` 同一套 ONNX 配置
  （`onnx_compat.py` / `embedding_provider="onnx"`），抽共享嵌入模块避免两套实现漂移；
  去掉 sentence-transformers 依赖。
- **dense 向量** 用于余弦 / ANN 语义相似度；**sparse 向量** 用于关键词召回（BM25 等价），
  配合 Milvus `WeightedRanker` 做混合检索，因此**不依赖 Milvus 原生 BM25 分词器 / jieba**，
  中文场景无需额外分词。
- **回退维度同步**：`EmbeddingModelConfig.dim` 384→1024，`_hash_embedding` 回退向量维度
  同步改为 1024。

### 4.2 Milvus Lite 单进程独占锁（运行与只读说明）

milvus-lite 是**嵌入式、单进程**存储：`MilvusLite.__init__` 只接受 `data_dir` 一个
参数，构造时对 `<data_dir>/LOCK` 无条件 `fcntl.flock(LOCK_EX | LOCK_NB)` 加独占锁，
只有 `close()` 或进程退出才释放。**没有只读模式 / `lock=False` 之类的开关**。因此：

- 运行中的 `memory_service` 在启动时打开 Milvus（`service.py` 里 `MilvusBackend()`）
  并**终身持有**该锁；任何第二个进程用 `MilvusClient` 打开同一目录都会抛
  `DataDirLockedError`（外部观察常误判为「Milvus 打不开 / 是空的」）。
- 想在不重启服务的前提下读向量，两条路：
  1. 走服务自己的 `vector_search` MCP（服务是唯一读写者，本就是设计读接口）；
  2. **直读磁盘原始文件**：milvus-lite 数据就是 `collections/memory_vectors/` 下的
     parquet（已合并段）+ arrow WAL（`wal_data_*` 为 upsert、`wal_delta_*` 为 delete），
     用 pyarrow 直接读，按 `_seq` 取最新即可还原当前状态，**不碰锁**。
     `scripts/memory_db_viewer.py` 已按此实现。
- 运维含义：一个数据目录同时只能有一个 milvus-lite 进程；本系统里 `memory_service`
  是唯一 Milvus 使用者，无多写者冲突问题。

## 5. 写入路径（改造现有持久化后端，管道其余不动）

- `GraphStore`：保留内存 `_nodes` dict 与 API，仅把 `_persist`/`_load` 从
  `graph_store.json` 换成 MongoDB `scene_nodes`（`add_node`/`update_node` 后 upsert）。
  新增节点成功后 → `embed(summary)` → Milvus `memory_vectors`
  `insert(doc_id, vector, kind/node_type/timestamp)`。
- `PtdlStore`：`_entries` 仍在内存，`_save`/`_load` 换成 MongoDB `plan_tasks`；
  `add()` 成功后 → `embed(query + description)` → Milvus `memory_vectors`
  upsert（以 `sha1(query)` 为键）。
- `ImageStore`：RGB 相机帧不再落盘文件，改为写入节点文档的 `images` Binary 字段
  （随 `scene_nodes` upsert 一并保存）；读取改为**按需懒加载**——批量检索不取图，
  仅 `vlm_qa` / VLM 富化路径按 `node_id` 单独取回（见 §6）。深度图/视频当前不存。

### 5.1 更新 / 删除 / promote / 清空的同步（Milvus 生命周期）

Milvus 是派生索引，必须随 MongoDB 的每次变更同步，否则检索命中后回源取不到或取到
过期数据。所有写操作按「先 Mongo、后 Milvus」执行，Milvus 以 `doc_id` 为主键
`upsert`（存在则替换、不存在则插入）：

- **新增** `add_node`：Mongo `insert` → `embed(summary)` → Milvus `memory_vectors` `insert`。
- **更新** `update_node`（含 VLM 富化改 summary/tags/weight）：Mongo `update`（`version`
  乐观锁自增）→ 若摘要/嵌入文本变化则重新 `embed`，再 Milvus `upsert(doc_id, vector, 标量)`；
  仅 `node_type`/`weight` 等标量变化时也 `upsert` 同步标量字段。
- **删除** `remove_node`：Mongo `delete` → Milvus `delete(doc_id)`；`causal_edges` 同删。
- **promote** `promote_to_long_term`：采用**稳定 `_id`**（不重编号），仅将 `node_type`
  由 `SHORT_TERM` 改为 `LONG_TERM`，`_id` 保持不变。Mongo `update` 同步 `node_type`
  字段，Milvus 仅 `upsert` 同步 `node_type` 标量（无需 delete 旧 + insert 新）。
  旧的 0–999 / 1000+ 区间重编号方案已废弃（阶段 0 已评估并采纳稳定 `_id`）。
- **清空** `clear_all` / `clean_start`：Mongo 按 collection 清空（`scene_nodes` /
  `causal_edges` 可清，`plan_tasks` 保留）→ Milvus 清空或 `drop` 对应分区（保留 plan 向量）。
- **失败恢复**：任一 Milvus 写入失败不阻塞主流程（Mongo 已落为准），由 `on_init` / 周期对账
  「Mongo vs Milvus 差集」补写缺失向量（见 §10 风险 1）。

> 读路径为 Milvus→Mongo：只要 Milvus 有向量，Mongo 必有对应文档；向量陈旧只影响相似度
> 分数，不影响文档内容正确性，因此「先 Mongo 后 Milvus」足以保证一致。

## 6. 检索路径

- 新增 MCP 工具 `robonix/service/memory/vector_search`：
  `query` + `kind`（scene/plan/all）+ `top_k` + 可选 `time_range`/`node_type`
  → embed → Milvus ANN(+filter) → `doc_id` 列表 → MongoDB `$in` 取回文档
  （**默认 projection 排除 `images`/`frame_ts`，仅取 summary/描述 + score + 元数据**）
  → 排序返回。
- **kind 分路检索**：`kind="all"` 时对**同一条 query** 走**两条独立的 ANN**——
  `kind="scene"`（场景物体）与 `kind="plan"`（历史规划）各自 `top_k`，**不合并排名**
  （两条路径分开返回，避免跨类型分数混排），分别返回 `scene` / `plan` 两个列表。
  `kind="scene"`/`"plan"` 仍返回单一路径的平铺列表。
- **图片懒加载（读取粒度）**：`images` 默认不出现在任何检索结果里，只在两条路径按需读回——
  ① `vlm_qa=True` 时 `vlm_answer_question` 按 `node_id` 取回命中节点图片交 VLM 看图作答；
  ② Tier-3 `VLMFeatureExtractor` 对节点视觉富化时按 `node_id` 取回该节点图片（每次 ≤3 帧）。
  两条路径均为「按 `node_id` 单文档 `$slice` / projection 取回 `images`」的懒加载，而非随
  `top_k` 批量检索返回，避免把每张图都带出来。
- `retrieve.py` Path B/C：embedding 启用时用 Milvus 混合检索（dense + sparse
  `WeightedRanker`）替代 numpy 余弦；关键词侧叠加既有 `BM25Index`（内存倒排），保留
  `TagIndex` 预过滤（转成 Milvus `filter=`）。
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

- **阶段 0（本期）**：MongoDB（元数据 + RGB 图 Binary）+ Milvus Lite + BGE-M3（1024）；
  `GraphStore`/`PtdlStore`/`ImageStore` 换成 Mongo 持久化；新增 `vector_search` 工具
  （按 kind 分路）；`on_init` 从 Mongo 全量重建 Milvus + 旧 JSON backfill。
- **阶段 1**：接入 `RetrievePipeline` Path B + ptdl 语义兜底。
- **阶段 2（后续）**：图片 CLIP、raw_log、深度图/视频（走 GridFS）。

## 9. 部署与依赖

- MongoDB：单节点 `mongod`（Docker，`--wiredTigerCacheSizeGB` 限内存），
  URI `mongodb://127.0.0.1:27017/robonix_memory`。
- 依赖：`pymongo`（async 可加 `motor`）、`pymilvus`（milvus-lite）、
  `onnxruntime`（分架构 pin，参考 memsearch）。BGE-M3 走 ONNX int8 后可去
  sentence-transformers。

## 10. 风险与对策

1. **双写一致性**：Mongo 先写、Milvus 后写；Milvus 失败可下次 rebuild 补齐（最终一致），
   或写后校验缺失向量。
2. **RGB 图体积**：单帧 JPEG 压缩存储；控制单文档总大小 <16MB（避免逼近 BSON 上限，
   必要时才走 GridFS）。
3. **MongoDB 无嵌入式模式**：边缘端需跑 mongod 进程，注意资源占用（限 cache + 存储卷）。
4. **跨库无事务**：Mongo 文档与 Milvus 向量无法原子提交，靠「先 Mongo 后 Milvus +
   可重建」保证不丢数据。
5. **EmbeddingModelConfig 维度回退**：BGE-M3 `dim` 384→1024，`_hash_embedding` 回退
   向量维度同步改 1024，避免模型加载失败时维度不一致。
6. **BM25 中文分词**：不依赖 Milvus 原生 BM25 分词器 / jieba——关键词召回用 BGE-M3
   自带 sparse（阶段 2）或内存 `BM25Index` 倒排（阶段 0/1）兜底。
7. **Milvus Lite 独占锁**：见 §4.2。单进程独占 `flock`，无法并发只读；外部查看须走
   服务 API 或直读原始文件，否则会得 `DataDirLockedError` 并误判为「空」。

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
 ③ embed(summary_zh+" / "+summary_en)  ③ embed(query+" "+description)
        │                                  │
        ├──────────────┬───────────────────┘
        ▼              ▼
   vector(1024)    vector(1024)
        │              │
        ▼              ▼
 ④ Milvus insert (kind=scene)    ④ Milvus upsert (kind=plan)
   doc_id=str(node_id)             doc_id=sha1(query)
   （Milvus 只存向量+引用）        （Milvus 只存向量+引用）
```

### 检索流程（统一入口，命中后回 MongoDB 取原文）

```text
 vector_search(query, kind, top_k, time_range?)
        │
        ▼
 ① embed(query) → vector(1024)
        │
        ▼
 ② Milvus ANN 检索（kind filter: scene / plan / all）
    filter: node_type / timestamp
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

---

## 补充：双路径 embedding（BGE-M3 单模型）与存储精简

> 本节为**追加补充**，不改动上文既有内容。核心结论：**仍然只用 BGE-M3 一个模型**
> （1024 维、中英双语、一个 `memory_vectors` collection），把「场景 / 规划」拆成
> **两条 embedding 输入路径**，而非拆成两个模型。

### 1) 双路径 embedding（场景英文、规划中文）

BGE-M3 本身中英双语、共享一个 1024 维语义空间，因此场景与规划**继续共用同一模型、
同一 Milvus 集合**，只是喂给模型的**输入文本语言不同**：

```text
                      ┌──────────────────────────────────┐
                      │        BGE-M3（单模型，1024）     │
                      └──────────────────────────────────┘
                          ▲                        ▲
         路径 ① scene（英文）               路径 ② plan（中文）
          embed(summary_en)                  embed(query + " " + description)
                          │                        │
                          ▼                        ▼
              Milvus memory_vectors           Milvus memory_vectors
              （kind="scene"）                 （kind="plan"）
```

| 路径 | 要 embed 的文本 | 语言 | 落库 kind |
|---|---|---|---|
| ① 场景 | `summary_en`（缺省回退 `summary`） | 英文 | `scene` |
| ② 规划 | `query + " " + description` | 中文 | `plan` |

- **为什么场景走英文、规划走中文**：场景摘要由 VLM 富化产出，英文摘要语义最稳定
  （当前 63 条 scene 向量全为英文 summary）；规划是 LLM 对中文指令的任务描述，
  `query`/`description` 天然中文。各用母语路径，避免跨语言转译折损。
- **检索侧不变**：`vector_search(query, kind)` 把 query 用 BGE-M3 embed 一次，再按
  `kind` filter 到对应路径的向量上做 ANN；`kind="all"` 时对 scene / plan 各自 top_k、
  **不合并排名**（见 §6「kind 分路检索」）。
- **与 §4.1 原文的衔接**：§4.1 写「场景 = `summary_zh + " / " + summary_en`」为早期
  折衷；本补充将其收敛为**场景只取 `summary_en`**（规划仍 `query + description`）。
  代码落地时以本节为准：`graph_store._scene_embed_text` 优先 `summary_en`，
  `ptdl_store._sync_milvus` 保持 `query + description`。
- **好处**：模型、权重、ONNX、schema、`_hash_embedding` 兜底全部只需一份，不引入
  第二套 768 维模型与第二个 collection，运维与重建成本不变。

### 2) 存储信息精简（场景物体 / 规划，不存不必要信息）

原则：**Mongo 只存「检索 / 回放必需 + 重建所需」**；派生向量进 Milvus，图片懒加载，
VLM/LLM 的原始中间输出一律不落库。场景与规划各保留唯一 embedding 文本源
（scene=`summary_en`、plan=`query + description`），其余字段按需精简。

**场景节点 `scene_nodes` 精简模板：**

```json
{
  "_id": 42,
  "node_type": "short_term",
  "timestamp": 1785600000000000000,
  "summary_en": "observe cup on table in kitchen",
  "summary_zh": "在厨房观察到桌上的杯子",
  "tags": {
    "scene_type": "kitchen",
    "objects_present": ["cup", "table"],
    "region": "near_window",
    "action_type": "observe",
    "success": true,
    "tool_used": ["tiago_camera.camera_snapshot"],
    "task_type": "explore",
    "difficulty": "easy",
    "intent": "记录当前场景物体"
  },
  "causal_chain": [40],
  "weight": 0.8,
  "images": ["<bson_binary_jpeg>"],
  "frame_ts": [1785600000000000000],
  "version": 3
}
```

- **保留**：`_id`(node_id)、`node_type`、`timestamp`、`summary_en`（英文 embedding 源）+
  `summary_zh`（展示用，可选）、`tags`（四维标签，含 `objects_present` 紧凑物体标签，
  供 TagIndex 过滤）、`causal_chain`、`weight`、`images`（懒加载、JPEG 压缩、限帧数）、
  `frame_ts`、`version`。
- **不存**：`embedding`（派生向量已在 Milvus）、`raw_log`（原始日志正文）、
  `spatial_data` 的 3D 物体坐标（无空间查询时只用 `tags.objects_present` 标签）、
  `depth_refs` / `video_clip_refs`（占位）、`time_range` / `camera_params` /
  `confidence_flags` / `access_count` / `last_access`（检索与回放用不到）。

**规划节点 `plan_tasks` 精简模板：**

```json
{
  "_id": "<sha1(query)>",
  "query": "去厨房拿一个水杯",
  "description": "complete task (2 step(s) across 2 planning round(s))",
  "timestamp_ns": 1785600000000000000,
  "plan_count": 2,
  "canceled_count": 0,
  "plans": [
    {
      "plan_id": "1",
      "status": "success",
      "description": "记录起点并开始旋转扫描",
      "steps": [
        "1. [mapping.map_get_pose] 读取当前位姿",
        "2. [tiago_camera.camera_snapshot] 拍摄初始图像"
      ],
      "rtdl_plan": "{\"plan_id\":\"1\",\"nodes\":[…]}"
    }
  ]
}
```

- **保留**：`_id`(sha1(query))、`query`（精确匹配 + embedding）、`description`
  （embedding）、`plans[]` 的 `steps` + `rtdl_plan`（重放必需的可执行 Plan AST）、
  派生计数 `plan_count` / `canceled_count`。
- **不存**：`raw_rtdl`（原始 LLM 输出，重放只依赖 `rtdl_plan`，仅调试保留）、
  每轮 LLM 对话 / token 日志、重复步骤文本（`_merge_steps` 已去重）。

> 精简一致性：Milvus 侧 scene/plan 各只有唯一 embedding 文本源，Mongo 去掉冗余字段后
> 该文本源仍是唯一事实，检索命中回源取回时不缺字段。

### 3) 精简数据结构（字段级对照）+ 测试双写 + 非向量库不含向量

#### 3.1 场景节点 `MemoryNode` 字段级精简（对齐 `types.py`）

`MemoryNode` 全量字段落库（Mongo `scene_nodes` 与 JSON 快照）时按下面口径精简：

| 字段 | 精简后处理 | 理由 |
|---|---|---|
| `node_id` / `node_type` / `version` | **保留** | 主键、类型、乐观锁 |
| `summary` / `summary_zh` / `summary_en` | **保留** | `summary_en` 是 scene embedding 源 |
| `timestamp` / `created_at` | **保留** | 时间窗口 filter / 审计 |
| `tags`（四维，含 `objects_present`） | **保留** | TagIndex 过滤；紧凑物体标签替代 3D 坐标 |
| `causal_chain` / `weight` | **保留** | 因果 / 检索权重 |
| `images`（Binary）/ `frame_ts` | **保留（懒加载）** | RGB 帧，检索默认 projection 排除 |
| `embedding` | **不落库**（仅 Milvus + 内存缓存） | 派生向量，非向量库一律不含 |
| `raw_log` | **不落库** | 原始日志正文，检索/回放用不到 |
| `spatial_data`（3D 物体坐标） | **不落库** | 无空间查询时只用 `tags.objects_present` |
| `camera_params` / `time_range` | **不落库** | 检索与回放用不到 |
| `depth_refs` / `video_clip_refs` | **不落库** | 占位，深度图/视频走 GridFS（阶段 2） |
| `confidence_flags` | **不落库** | 逐字段置信度，检索/回放用不到 |
| `last_access` / `access_count` | **不落库** | 遗忘打分内存算，不需持久化 |

#### 3.2 规划节点 `plan_tasks` 字段级精简

plan 条目为 dict（非 dataclass），落库保留 `_id`(sha1(query))、`query`、`description`、
`timestamp_ns`、`plan_count`/`canceled_count`、`plans[]` 的 `plan_id`/`status`/
`description`/`steps`/`rtdl_plan`；**不存** `raw_rtdl`（原始 LLM 输出）、每轮对话/token
日志、重复步骤文本。plan 的 embedding 只在 `_sync_milvus` 里现算、写 Milvus，从不进
`plan_tasks` 文档，故 plan 侧天然无向量泄漏。

#### 3.3 非向量库不含向量（原则）

**MongoDB 与 JSON 快照是「非向量库」，一律不存 `embedding` 向量**——向量只存在于
Milvus（`memory_vectors`）与内存 `node.embedding` 缓存：

- Mongo `MongoBackend.upsert_node` 已 `pop("embedding")` / `pop("image_refs")`，只写
  元数据 + `images` Binary。
- JSON 快照 `graph_store.py::_persist` 序列化时同样 `pop("embedding")`，与 Mongo 口径
  一致（`ptdl_store.json` 本就无 embedding）。
- 回源重建：Mongo/JSON 读回的节点无 `embedding`，`_milvus_vector` 按 `summary_en`
  现算（BGE-M3）再写 Milvus——向量是**派生索引**，丢了可全量重建，不属事实源。

#### 3.4 测试双写（DB + JSON）

测试/验证场景下，每次写操作**除写 Mongo（权威）+ Milvus（向量）外，同时落盘 JSON
快照**（`graph_store.json` / `ptdl_store.json`），便于人工核对与离线检查：

- `graph_store.py::_persist` / `ptdl_store.py::_save` 已实现「JSON + Mongo 双写」：
  JSON 是 fallback 与调试快照，Mongo 是权威源（`_load` 时 Mongo 覆盖 JSON）。
- JSON 快照同样是**精简后**文档（不含 `embedding`），与「非向量库不含向量」一致。
- 独立验证脚本 `scripts/vector_retrieval_test.py` 可直读 Milvus 原始文件（parquet+WAL）
  与 Mongo 文本，配合 JSON 快照做三层对账（JSON / Mongo / Milvus）。
