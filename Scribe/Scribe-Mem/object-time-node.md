# 场景记忆节点的时间信息现状 与「时间节点」方案

> 问题：场景记忆节点是否包含「一个物体在两个时间点」的信息，从而支持
> 「昨天 vs 今天」的前后对比（如 Demo 1 展品巡检：地球仪昨天在展台中央，
> 今天偏移了 5cm）？若不足，给出添加「时间节点」使其可前后对比的方案。

## 1. 现状：场景节点已经带时间，但缺「对象时间线」

`MemoryNode`（`memory_service/core/types.py`）**已经**携带了时间维度的原始材料：

| 字段 | 类型 | 含义 |
|---|---|---|
| `timestamp` | int（chronos ns） | 单一事件时间点（节点发生的时刻） |
| `time_range` | `TimeRange{start_ts,end_ts}` | 观测窗口（如"巡检这一段"） |
| `created_at` / `last_access` / `access_count` | int | 落库 / 检索审计 |
| `causal_chain` | `List[int]` | 父节点 ID 列表（因果 / 前驱） |
| `spatial_data.objects[]` | `List[ObjectCoord]` | 坐标快照，`ObjectCoord{obj_id,label,x,y,z,label_zh,label_en}` |
| `version` | int | 乐观锁（**不是**快照版本号） |

因此：

- **「一个物体在两个时间点的信息」当前 = 两个独立节点**。每次 `remember`
  生成一个新 `node_id`，各自带一个 `timestamp`。两天观测地球仪 = 两个节点，
  各自有 `spatial_data` 坐标。
- **缺的不是"能不能存两个时间点"，而是"能不能一键前后对比"**：
  1. **无稳定对象主键** —— `tags.objects_present` 是自由标签（`"地球仪"`），
     跨天改名 / 中英不一致（`label` vs `label_zh`）会匹配不上；
     `ObjectCoord.obj_id` 字段虽已存在，但未强制使用。
  2. **无对象历史链** —— `causal_chain` 是通用因果边，不是"同一物体的时间线"，
     没有 `get_history(obj_id)` 或 `diff(obj_id)` 这类原语。
  3. **无坐标对比原语** —— 要对比两天坐标，调用方得自己搜出两天节点、自己算 Δ。

**结论**：能存"两个时间点"，但"前后对比"目前要靠调用方手动编排，不是
记忆服务的一等能力。

## 2. 方案：新增「对象时间线 ObjectTimeline」（推荐）

最小改动、复用现有字段，新增**一个**与 `TagIndex` / `VectorStore` 并列的索引，
把"同一物体跨时间"升级为一等公民。

### 2.1 复用现有字段：稳定对象主键

把 `ObjectCoord.obj_id`（已存在，默认值 `""`）**升级为跨时间同一物体的身份**。
Demo 的"地球仪"固定 `obj_id="exhibit.globe_001"`，两天共用同一个 id，与自由
标签 `label="地球仪"` 解耦。

### 2.2 保持「一条记忆 = 一个事件」

每次巡检仍写一个**新节点**：`timestamp` = 当天开馆时刻，`time_range` = 巡检
窗口。两时间点 = 两节点，写入语义不变。

### 2.3 用 `causal_chain` 串时间链（零新增字段）

今天写入地球仪观测时，把昨天地球仪节点的 id 填进 `RememberRequest.parent_node_id`，
使今天节点的 `causal_chain` 指向昨天节点 —— 形成"昨天基准 → 今天观测"的时间链。
这是现成字段，语义上"昨天是今天的父"。

### 2.4 新增 `ObjectTimeline` 索引（唯一新抽象）

```text
{ obj_id -> [ (timestamp, node_id, ObjectCoord), ... ] }   # 按 timestamp 升序
```

- **写**：`remember` 落库时 `record(obj_id, node_id, coord)`（在 `graph_store`
  的节点写入路径上，与 `TagIndex` 的 `_index_tags` 同一处）。
- **读**：
  - `history(obj_id, limit=N)` → 最近 N 次快照（对象时间线）；
  - `diff(obj_id)` → 最近两次快照的坐标差 `Δx/Δy/Δz` + 欧氏距离。
- **持久化**：像 `TagIndex.rebuild(nodes)` 一样从全量节点重算 —— 每个节点的
  `spatial_data.objects[]` 已含坐标，**无需单独落库**，丢了可全量重建。

### 2.5 检索 / API 面

- 新增 MCP tool `robonix/service/memory/object_history`（暴露 `history` / `diff`），
  供 Pilot 直接调用做"前后对比"。
- `hybrid_search` 的 `objects` 过滤键从自由标签升级为**支持 `obj_id`**，保证跨天
  命中同一物体（而不是靠标签字符串相等）。

### 2.6 零改动兜底方案（若暂不新增索引）

只用现有 `causal_chain` + `timestamp`：`hybrid_search` 按
`objects_present ⊇ {物体} ∩ action_type=巡检` 过滤，按 `timestamp` 排序取最近两次，
调用方自己算坐标差。够 Demo 用，但每次都要"检索 + 手动 diff"，且对象主键不稳。

## 3. 落地到 Demo 1（地球仪：昨天 vs 今天）

```text
Day1 开馆前巡检
  remember(obj_id=exhibit.globe_001, action=巡检, success=true,
           spatial={globe_001: (3.2,1.5,0.8)}, summary="…展台中央…")
        → node #42  (timestamp=T1, spatial=3.2/1.5/0.8)
        → ObjectTimeline.record("exhibit.globe_001", 42, coord)

Day2 开馆前巡检
  remember(obj_id=exhibit.globe_001, action=巡检, success=true,
           parent_node_id=42, spatial={globe_001: (3.25,1.5,0.8)},
           summary="…位置偏移…")
        → node #57  (timestamp=T2, causal_chain=[42], spatial=3.25/1.5/0.8)
        → ObjectTimeline.record("exhibit.globe_001", 57, coord)

Pilot 问"地球仪昨天被碰过吗？"
  object_history.diff("exhibit.globe_001")
        → 最近两次: T2(3.25,1.5,0.8) vs T1(3.2,1.5,0.8)
        → Δx=+0.05m, 欧氏距离=0.05m → "偏移 5cm，可能被碰过，我来复位"
```

## 4. 变更清单

| 类别 | 内容 |
|---|---|
| 新增 | `memory_service/storage/object_timeline.py`（`ObjectTimeline`） |
| 新增 | MCP tool `object_history`（`history` / `diff`）+ contract `.srv` / `.v1.toml` |
| 改动 | `ObjectCoord.obj_id` 语义明确为"跨时间对象主键"，`remember` 写入时 `record` 时间线 |
| 改动 | `hybrid_search.objects` 支持 `obj_id` 匹配 |
| 复用 | `timestamp` / `time_range` / `causal_chain` / `parent_node_id` 均不改结构 |

## 5. 拍摄记忆：空间去重 + 10min 时间门控（场景物体节点结构·定稿）

> 本节与 §2 互补：§2 的 ObjectTimeline 用「每个观测一个独立节点 + 时间线索引」
> 支撑「昨天 vs 今天」的前后对比；本节针对**拍摄记忆**——同一物体反复拍摄时，
> 以**空间去重为根**，在**一个节点内**累积多张图，用 10min 时间门控决定
> 「更新节点还是跳过」。

### 5.1 规则

空间去重命中同一物体（同一 1m 网格 / 2m 半径）时：

- `now − timestamp ≤ 10min`：视为同一次观测，**跳过**（不追加、不更新）；
- `now − timestamp > 10min`：**复用**该节点，`timestamp = now`，`image_refs`
  追加一张，`frame_ts` 追加 `now`，`version += 1`；
- 位置变化（超出网格 / 半径）→ 仍**另起新节点**。

其中 `timestamp` 兼任「最新一次有效拍摄时间」与「10min 门控比较基准」，故
**不需要** `last_shot_ts`；首末时间可由 `frame_ts` 首尾（或 `created_at` +
`timestamp`）推得，故**不需要** `time_range`。

### 5.2 节点数据结构（object_observation）

```jsonc
{
  "node_id": 42,                          // 空间去重键对应节点，同物体+位置只此一个
  "node_type": "object_observation",
  "summary": "VLM observed globe (0.95) at center-mid",
  "summary_zh": "VLM 观察到地球仪（0.95）在中央",
  "summary_en": "VLM observed globe (0.95) at center-mid",

  "timestamp": 1760000000000000000,       // 最新一次有效拍摄时间（>10min 追加时更新，兼作门控基准）

  "image_refs": [                         // 每次 >10min 有效拍摄追加一张
    "data/images/42/frame_0001.png",
    "data/images/42/frame_0002.png"
  ],
  "frame_ts": [                           // 与 image_refs 一一对应的拍摄时间 ns（完整时间线）
    1759000000000000000,
    1760000000000000000
  ],

  "spatial_data": {                       // 去重根，不变
    "origin": "world",
    "objects": [{
      "obj_id": "exhibit.globe_001",
      "label": "globe", "label_zh": "地球仪",
      "x": 3.2, "y": 1.5, "z": 0.8
    }]
  },

  "tags": {                               // 检索键，不变
    "scene_type": "exhibition_hall_a",
    "objects_present": ["globe"],
    "action_type": "observe",
    "success": true
  },
  "causal_chain": [5],
  "weight": 0.5,
  "version": 2,                           // 每次追加图片 +1
  "created_at": 1759000000000000000,      // 首拍时间，不变
  "last_access": 0, "access_count": 0
}
```

### 5.3 字段定稿对照

| 字段 | 状态 | 说明 |
|---|---|---|
| `timestamp` | **保留** | 最新一次有效拍摄时间；兼作 10min 门控基准 |
| `image_refs[]` | 复用 | 每次 >10min 有效拍摄追加一张 |
| `frame_ts[]` | 复用 | 与 `image_refs` 平行的每图拍摄时间（时间线） |
| `last_shot_ts` | **去除** | 用 `timestamp` 兼任 |
| `time_range` | **去除** | 首末时间由 `frame_ts` / `created_at + timestamp` 推得 |
| `spatial_data` / `tags` / `summary` | 不变 | 去重根 + 检索键 |
| 阈值 | 新增常量 | `SHOT_MIN_INTERVAL_S = 600`（可 env 覆盖，如 `VLM_OBSERVER_SHOT_GAP_S`） |

### 5.4 落地改动点（待实现 ⬜）

- `_find_existing_node` / `vlm_observer._find_cross_segment_match` 命中后加
  `now − node.timestamp > 10min` 判断；
- `_append_image_ref` 同步更新 `timestamp` / `frame_ts` / `version`；
- `to_dict` / `from_dict` 与 Mongo 精简模板**无需改字段**（只用现有
  `timestamp` / `image_refs` / `frame_ts`）。

## 6. 统一 MCP 工具：mem_remember + mem_search（多路检索 + 重排）（待实现 ⬜）

> 目标：把现有 7 个 MCP tool 收敛为 **2 个** —— `mem_remember`（统一写）+
> `mem_search`（统一读，多路召回 + RRF 重排）；`promote` 转内部自动触发。

### 6.1 最终接口

| tool | 职责 |
|---|---|
| `mem_remember` | 统一写入：`type=memory`（CKG 节点）/ `type=plan`（计划） |
| `mem_search` | 统一读取：多路召回（向量/关键字/标签/因果/LLM）+ RRF 重排 top-k |

### 6.2 mem_remember（写）

用 `type` 判别，一路 handler、两套字段：

```jsonc
{ "type": "memory", "session_id": "…", "log_record": { … } }
{ "type": "plan",   "query": "去厨房拿可乐", "steps": [ … ] }
```

- `type=memory` → 现 `remember`（session_id / plan_id / log_record / spatial /
  parent_node_id / image_base64 / depth_base64 / camera_params / time_range / kv）。
- `type=plan` → 现 `ptdl_remember`（query / description / steps / plan_count /
  canceled_count / rtdl_plan / raw_rtdl / plan_id）。
- 内部路由到 `RememberPipeline` 或 `ptdl_store.add`；返回 `node_id` 或 `{ok}`。

### 6.3 mem_search（读）：多路 + 重排

#### 6.3.1 多路召回（5 路）

| 路 | 数据 / 机制 | 分数 | 复用 |
|---|---|---|---|
| vector | BGE-M3 → Milvus ANN | cosine | `MilvusBackend.search` |
| keyword | BM25 词法匹配 summary/raw_log/tags | BM25 | `VectorStore.search`（BM25 侧） |
| tag | 标签倒排 AND + 重叠度 | Jaccard | `TagIndex.query` |
| causal | `causal_chain` 因果/上下文扩展 | 增益 | `retrieve.py` Stage3 |
| llm | LLM 相关性排序（可选、贵） | rank | `llm_search.llm_rank` |

#### 6.3.2 重排：RRF（默认）

各路由分数量纲不同（cosine vs BM25 vs LLM rank）不能直接相加，用 Reciprocal
Rank Fusion：

```text
RRF(d) = Σ_r  1 / (60 + rank_r(d))
```

`rank_r(d)` = 文档 d 在路由 r 中的名次（未进该路记 0）；可选加权 RRF：
`Σ_r w_r / (60 + rank_r(d))`。排序取 top-k。备选 weighted 线性融合需先归一化、
对权重敏感，不推荐做默认。

#### 6.3.3 预过滤 + 后处理

- **预过滤**：`domain`（all/memory/scene/plan）+ `tags`（AND）+ `time`
  （relative_time / time_range），先缩小候选再跑各路。
- **后处理**：gap 聚类 + 多图 VLM（承接 §5 / 上轮设计：top-k 相邻 score 落差
  截断 → 按时间 anchor 取图 → 喂 VLM），`vlm_qa:true` 时启用。

#### 6.3.4 请求 / 响应

```jsonc
// 请求
{ "query": "…", "top_k": 5, "domain": "all",            // all|memory|scene|plan
  "routes": ["vector","keyword","tag","causal"],         // 默认不含 llm（贵）
  "fusion": "rrf", "route_weights": { "vector": 1.0 },
  "tags": { "objects": ["red cup"], "action_type": "grasp" },
  "time": { "relative_time": "yesterday" } | { "start_ts": 0, "end_ts": 0 },
  "vlm_qa": false, "gap_threshold": 0.05, "min_sim": 0.5 }

// 响应
{ "results": [ { "doc_id", "kind", "score",
                 "route_scores": { "vector": …, "keyword": … },
                 "node" | "plan", "timestamp" } ],
  "vlm_answer": "…" }
```

### 6.4 融合 pipeline（伪码）

```text
cand      = prefilter(domain, tags, time)        # tag 倒排 + 时间窗 + domain
rankings  = { r: route_r(query, cand) for r in routes }   # 可并行
fused     = RRF(rankings, weights)               # 逐文档累加 1/(60+rank)
topk      = fused.sort(desc)[:top_k]
if vlm_qa: topk = gap_cluster(topk) → multi_image_vlm(topk)
```

### 6.5 代码映射（复用 vs 新增）

- **复用**：`TagIndex.query`、`VectorStore.search`、`MilvusBackend.search`、
  `llm_search.llm_rank`、`retrieve.py` 因果扩展、`vlm_answer_question` +
  `llm_decide_vlm`、`_relative_window_ns`。
- **新增**：`core/fusion.py`（RRF + weighted + router）；`mem_remember` /
  `mem_search` 两个 handler + 统一 schema；`type` / `domain` 判别。
- **删除 / 别名**：`hybrid_search`、`vector_search`、`ptdl_retrieve`、
  `search_history` → 并入 `mem_search`；`ptdl_remember` → `mem_remember`；
  `promote` → 内部自动。

### 6.6 迁移与兼容

1. 新增 `mem_search` / `mem_remember`，旧 tool 先并行保留。
2. 旧 tool 转别名 → 转发新入口，打 deprecation 日志。
3. 确认 Pilot / 调用方切走后删旧 tool。
4. `promote` 改写入超限 / 后台触发。

### 6.7 取舍 / 风险

- 混合 domain 用 RRF（名次融合）比 score 融合稳，但 scene/plan 语义空间差异
  仍在，`domain=all` 可能偏科 → 建议默认由调用意图给 domain，或 all 时按域召回
  再 RRF。
- `llm` 路贵，默认关闭。
- 改名 / 删 RPC 属对外接口变更，需确认后再动代码。
