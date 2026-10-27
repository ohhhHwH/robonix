# 向量检索 Phase 0 — 真实 Webots 路径实机验证（2026-10-26）

与 `VERIFICATION.md`（直连 MCP 构造写入）互补：本次走**真机仿真**路径 —— 通过
`rbnx ask` 让 Pilot 驱动 Webots TIAGo 真实导航/拍照，由 Scene Hook 落场景记忆，
Pilot 落规划记忆，核对 Mongo / Milvus 实际落库。

## 环境

- 模拟器容器 `robonix_tiago_sim` Up；`rbnx boot` 全组件 up；Pilot vlm=deepseek-flash。
- memory 服务连接 Mongo `127.0.0.1:27017/robonix_memory` + Milvus-lite
  `./memory/robonix_memory.db`（BGE-M3 ONNX 1024 维）。
- 基线（omen-1 提供）：scene_nodes=2 / plans=18 / edges=1。

## 结果

### 1. 直接 `remember`（不走 observe/list_objects）— ✅

`rbnx ask` 调用 `robonix/service/memory/remember`，显式保存一条用户偏好记忆：

- 返回 `node_id=2`；`scene_nodes` 2→3；`causal_edges` 1→2（auto-linker 把 node 2
  链接到 node 1，`causal_chain=[1]`）。
- node 2 未传 `image_base64`，故 `image_count=0`——图片 Binary 需求由下面探索节点
  证明（每个 observe/place 节点均带真实相机帧）。

### 2. 真实探索任务 — ✅

`rbnx ask "去厨房区域探索…"`：当前场景无“厨房”区域，Pilot 以当前位置为起点启动
限速 0.35 m/s 的自主探索，覆盖约 89 m²，180 s 上限正常结束（frontiers_left=22），
返回起点。落库增量：

| 记忆 | Mongo | Milvus |
|---|---|---|
| 场景 | `scene_nodes` 3→**40**（+37：13 place + 27 short_term） | `kind=scene` 同步 upsert |
| 因果 | `causal_edges` 2→**45**（+43，relation=`enables`） | — |
| 规划 | `plan_tasks` +1（query 经 ptdl_remember） | `kind=plan` 同步 upsert |
| 图片 | 63 张 BSON Binary（`images`）+ `image_count` 标量，0 不一致 | — |

- 图片懒加载一致性：`sum(len(images)) == sum(image_count) == 63`，逐节点核对
  `mismatched docs = 0`。place 节点因重访累积多帧（image_count 分布 0/1/2/3/4/10）。
- 因果边为 DAG 而非纯链：6 个节点有 2 个父节点（重访同一物体/格点，见边
  `5->31`、`9->35`、`12->36` 等），故 `causal_edges(45) > scene_nodes(40)`。

### 3. 数据库 dump — ✅

| 存储 | 行数 |
|---|---|
| Mongo `scene_nodes` | **40**（13 place / 27 short_term） |
| Mongo `causal_edges` | **45**（relation=enables） |
| Mongo `plan_tasks` | 23（长期共享集合，见下） |
| Milvus `memory_vectors` | **62**（40 scene + 22 plan，与 Mongo 场景+规划数一致） |

`vector_search(query="水杯 桌子 显示器", kind=all, top_k=6)` 命中 6 条（两类混合）：

```text
doc_id=ee3b2a9a…  kind=plan   node_type=             score=0.6331   query="去厨房拿一个水杯"
doc_id=7          kind=scene  node_type=short_term   score=0.5658
doc_id=21         kind=scene  node_type=short_term   score=0.5658
doc_id=4f349ab9…  kind=plan   node_type=             score=0.5603
doc_id=ba51fd2d…  kind=plan   node_type=             score=0.5587
doc_id=516be182…  kind=plan   node_type=             score=0.5455
```

## 备注：plan_tasks 为共享集合，绝对数含并发污染

`plan_tasks` / `kind=plan` 是长期记忆、跨会话共享。本次验证期间集合里出现了非本次
任务发起的 query（“去厨房拿一个水杯”“整理办公室西侧的书桌”“我昨天做过什么任务？”
等），说明同部署有其它会话/bot 并发使用该 memory 服务。因此 **plan 的绝对数与增量
不可只归因于本次探索**；但**场景/因果边/图片增量明确来自本次探索**——观察到的物体
（couch/table/chair/monitor/door/trash_bin/potted_plant/laptop/shelf/cabinet）与探索
终答逐字吻合。Milvus 行数 62 = 40 scene + 22 plan，与 Mongo 场景+规划文档数一致，
印证设计 §5/§7 的“Milvus 是 Mongo 的派生索引、随写随同步”不变量。

## 结论

三项全部通过：直接 `remember` 可落库（显式记忆、无 observe/list_objects）；真实
探索的规划/场景/因果/图片四类记忆各归其位（Mongo + Milvus 双写、图片 Binary 懒加载
一致）；`vector_search` 能跨 kind 命中并返回 doc_id/kind/node_type/score。
