# 向量存储与检索 — Phase 0 测试记录

依赖装齐后验证分三层：

1. **单元测试**：`tests/test_graph_store.py` 自带 `__main__` runner，`python3` 直接跑。
2. **graceful-degradation 冒烟**：在缺依赖的最小环境下确认 JSON / 磁盘 / hash 回退路径可用。
3. **真实后端 e2e**：`tests/test_vector_backends.py`（本次新增）在 Mongo/Milvus/BGE 全齐时验证真实读写。

## 运行命令

```sh
cd services/memory
python3 tests/test_graph_store.py
python3 tests/test_vector_store.py
python3 tests/test_image_store.py
```

## 结果

```
test_graph_store.py : 20 passed, 0 failed
test_vector_store.py : 26 passed, 0 failed
test_image_store.py  : 7 passed, 0 failed
```

覆盖：节点 CRUD、批量取回、乐观锁 version 冲突、remove、按类型/时间列举、边与
causal_chain 更新、JSON 持久化 reload、promote 四例；embedding 配置 / 哈希回退 /
BM25 / 混合检索 / 多模态接口；图片磁盘存取与懒加载 marker。

> `test_vector_store.py` 原断言沿用 MiniLM 384 维，BGE-M3 迁移后陈旧（21 过 5 败）。
> 2026-10-25 装依赖回归时已把 `model_name`/`dim`/`len(v)`/`get_dim` 断言 384→1024、
> 模型名改为 `gpahal/bge-m3-onnx-int8`，现 26/26 全绿。

## promote 语义变更（本次重点）

设计从「renumber 到 1000+」改为**稳定 `_id`**，原 3 个 promote 测试按新语义改写：

| 测试 | 旧断言 | 新断言 |
|---|---|---|
| `test_promote_short_to_long` | `new_id >= 1000`、`get_node(nid) is None` | `new_id == nid`、`get_node(nid).node_type == LONG_TERM` |
| `test_promote_rewrites_child_causal_chain` | child causal_chain 被重写到新 id | child causal_chain 仍引用同一 parent id，无需重写 |
| `test_promote_with_multiple_children` | 子节点 causal_chain 换成新 id | 子节点 causal_chain 保持原 parent id，边完好 |
| `test_promote_long_term_does_nothing` | 已长时 → 返回 None | 不变（已长时 → 返回 None） |

## 冒烟验证（缺依赖环境）

独立脚本确认以下路径无 pymongo/pymilvus/onnxruntime/memsearch 时仍可工作：

- `GraphStore`：add / promote（稳定 id）/ reload（JSON 兜底）/ `_next_id = max(ids)+1`。
- `PtdlStore`：`add` / `get_by_query_key` / `remove`（JSON 兜底）。
- `ImageStore`：磁盘保存 / `list` / `read_bytes_by_ref`。
- `BgeEmbedder`：模型缺失时 `_hash_embedding` 回退，输出 1024 维。
- `MilvusBackend` / `MongoBackend`：`available=False` 时所有方法 no-op，不抛异常。

## 真实后端 e2e（2026-10-25 补）

依赖装齐（pymongo 4.19.0 / pymilvus 3.0.2 / milvus-lite 3.2.2 / onnxruntime 1.31.0 /
memsearch 0.4.22 / numpy 2.5.4）后，新增 `tests/test_vector_backends.py` 验证真实
Mongo / Milvus / BGE-M3 读写路径。

环境准备：

- 起 mongod：`docker run -d --name mongo-e2e -p 127.0.0.1:27017:27017 mongo:7`
- BGE-M3 走镜像下载：`HF_ENDPOINT=https://hf-mirror.com`（本机 huggingface.co 不可达）
- numpy 必须 2.x（milvus-lite 3.2.2 / pyarrow 26 依赖；1.26.4 会 ImportError）

运行：

```sh
cd services/memory
HF_ENDPOINT=https://hf-mirror.com \
  ../rbnx-build/venv/bin/python tests/test_vector_backends.py
```

结果：

```
backends: mongo=True milvus=True bge_semantic=True (dim=1024)
8 passed, 0 failed
```

覆盖：Mongo 连接 / Milvus 集合创建 / BGE-M3 1024 维单位向量 / `add_node` 双写
（Mongo + Milvus）/ ANN 检索命中 / promote 稳定 `_id` 的 `node_type` 标量同步 /
`rebuild_milvus` 从 Mongo 重建 / 图片 Binary 往返 + 懒加载 marker / plan 的
Mongo + Milvus 双写。

真实后端验证已通过；`image_count` / marker 懒加载往返已包含在 `test_image_binary_roundtrip`
与 `test_add_node_persists_mongo_and_milvus` 中。
