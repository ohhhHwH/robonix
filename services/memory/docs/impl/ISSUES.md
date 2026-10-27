# 向量存储与检索 — Phase 0 遗留问题与风险

## ✅ 已解决：真实后端路径已验证（2026-10-25）

依赖装齐并跑通真实后端 e2e（见 TESTS.md）：pymongo 4.19.0 / pymilvus 3.0.2 /
milvus-lite 3.2.2 / onnxruntime 1.31.0 / memsearch 0.4.22 / numpy 2.5.4；
`add_node → vector_search`、图片 Binary 存取、`rebuild_milvus` 重建均 8/8 通过。

安装/验证过程中新发现两个问题，记录如下。

## ✅ 已解决：`test_vector_store.py` 陈旧断言（2026-10-25）

BGE-M3 迁移（`dim` 384→1024、`model_name` → `gpahal/bge-m3-onnx-int8`）漏改了
`test_vector_store.py`，导致 5 个测试断言旧值（21 过 5 败）。已把 `test_defaults` /
`test_hash_fallback_dim` / `test_encode_batch` / `test_text_modality_works` /
`test_get_dim_text` 的 384→1024 与模型名断言一并修正，现 26/26 全绿。

## 备注：`test_ptdl_store.py` 需 pytest（当前 venv 未装）

该测试用 `tmp_path` fixture、无 `sys.path.insert`，属 pytest 风格，`python3` 直跑会
`ModuleNotFoundError: memory_service`；venv 未装 pytest。属既有 harness 差异，非本次
回归，暂不处理（Phase 0 测试记录只覆盖 `test_graph_store.py` / `test_vector_backends.py`）。

## ⚠️ 新问题 1：huggingface.co 不可达，BGE-M3 需走镜像

本机 `huggingface.co` 网络不可达（`Network is unreachable`），`OnnxEmbedding`
默认下载会重试后失败并回退 hash。**必须设置 `HF_ENDPOINT=https://hf-mirror.com`**
让 `huggingface_hub` 走镜像下载。首次下载 `model_quantized.onnx`（约 570MB）后
缓存于 `~/.cache/huggingface`，之后离线可用。

部署侧需把 `HF_ENDPOINT=https://hf-mirror.com`（或预置 `EMBEDDING_MODEL_PATH`
指向本地已下载目录）写进服务 env，否则 `vector_search` 会静默退化为 hash 向量。

## ⚠️ 新问题 2：numpy 1.x ↔ 2.x 冲突，memory 服务必须升 numpy 2.x

`milvus-lite 3.2.2` → `pyarrow 26` 要求 numpy ≥ 2.0；而 memory 服务 venv 原为
numpy 1.26.4，导致 `import milvus_lite` 抛 `ImportError: pyarrow requires NumPy
2.0 or newer`。解法：memory venv 升 numpy 2.5.4（memory 服务的 numpy 用法仅
`np.array/dot/linalg.norm`，2.x 兼容，20/20 单元测试回归通过）。

注意 memory 的 `pyproject.toml` 目前只写 `numpy>=1.26`（无上界），一旦把
pymongo/pymilvus/memsearch 加入依赖，需把 numpy 约束与 memsearch 服务
（`numpy<2`）区分开：**memory 服务用 numpy 2.x，memsearch 服务仍锁 numpy 1.x**。

## ✅ 已解决：BGE-M3 ONNX int8 与 memsearch 0.4.22 兼容

实测 `memsearch==0.4.22` 的 `OnnxEmbedding(model=…, batch_size=…)` 签名、
`._encode(texts)`、`.dimension` 属性与 `BgeEmbedder` 的调用完全一致；模型加载后
`encode` 返回 1024 维、L2 单位向量（norm=1.0），无需改动 `BgeEmbedder`。
（ISSUES 原注的 `dense_vecs` 命名问题未出现——0.4.22 的 `_encode` 直接返回向量列表。）

## `doc_id` VARCHAR(64) vs §4 的 VARCHAR(32)

实现放宽到 64 以容纳 `sha1` 十六进制（40 字符）与 node_id 十进制；§4 原文写 32。
对 Milvus Lite 无实质影响，但若外部有 schema 校验需对齐。已偏离设计原文，属有意放宽。

## promote 稳定 `_id` 与 §5.1 原文的差异

§5.1 正文仍按「重编号」描述（delete 旧 + insert 新），仅在「备选（推荐）」里提稳定 `_id`。
本实现采用稳定 `_id`（omen-1 明确认可），因此 **Milvus 不执行 delete+insert，只 upsert
同步 `node_type` 标量**。设计文档正文尚未回填此决策，需一并更新 §5.1 措辞，避免后续读者
按重编号实现。

## 图片懒加载：`image_count` 标量与 `images` 数组的一致性

默认 projection 排除 `images` 后，用 `image_count` 生成 marker 数量；`push_image` /
`pop_oldest_image` / `clear_images` 需严格配对 `$inc`/`$dec`/`$set 0`。若历史节点文档
（backfill 前）没有 `image_count` 字段，`doc_to_node` 以 `doc.get("image_count") or 0`
兜底为 0，此时旧图不会被懒加载 —— backfill 从 `data/images/` 导入时需同步写 `image_count`
与 `images`，否则旧图丢失。（设计 §7 要求图片一并导入，此路径尚未实现 `data/images/` 导入。）

## 未实现项（明确不在本期）

- **阶段 1**：`RetrievePipeline` Path B 接 Milvus 混合检索（dense + sparse `WeightedRanker`）、
  `ptdl_retrieve` 三级判定（精确命中 / ≥0.90 重放 / ≥0.80 参考）。
- **§7 `data/images/` backfill 导入**：旧 JSON backfill 已做，但旧图片目录 → Mongo Binary
  的导入未做（本期只覆盖新写入路径）。
- **阶段 2**：CLIP 图片向量、raw_log、BGE-M3 sparse/colbert、深度图/视频（GridFS）。
