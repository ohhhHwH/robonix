"""EN→ZH translator via quantized opus-mt ONNX (CPU, no torch).

Scene summaries are written in English by the VLM (``summary_en``) while
retrieval queries arrive in Chinese, so a same-language scene embedding
matches better than a cross-lingual one. This translates the English summary
to Chinese before ``BgeEmbedder`` encodes it.

Reuses the two runtimes already in the service — ``onnxruntime`` (the same one
``BgeEmbedder`` drives) plus ``sentencepiece`` for the Marian tokenizer — and
the ``Xenova/opus-mt-en-zh`` quantized ONNX encoder/decoder. No torch or
transformers dependency. When any of it is missing the translator degrades to
returning the input unchanged, so the service still boots with English scene
embeddings (the pre-translator behaviour).
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional

import numpy as np

log = logging.getLogger("scribe_mem")

# Marian decodes from the pad token and stops at eos. The 65k vocab is the union
# of the English source and Chinese target sentencepiece vocabularies, so pieces
# must be mapped through vocab.json instead of used as raw sentencepiece ids.
_EOS = 0
_UNK = 1
_DEC_START = 65000

_SRC_REPO = "Helsinki-NLP/opus-mt-en-zh"   # tokenizer: source/target .spm + vocab.json
_ONNX_REPO = "Xenova/opus-mt-en-zh"         # quantized encoder/decoder ONNX


def _download(repo: str, filename: str) -> str:
    """Fetch one model file, offline cache first (mirrors BgeEmbedder)."""
    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(repo, filename, local_files_only=True)
    except Exception:
        return hf_hub_download(repo, filename)


class Translator:
    """English→Chinese text translator; a no-op pass-through when unloaded."""

    def __init__(self, *, max_length: int = 128, repetition_penalty: float = 1.3):
        self._max_length = max_length
        self._repetition_penalty = repetition_penalty
        self._loaded = False
        self._enc: Optional[object] = None
        self._dec: Optional[object] = None
        self._dec_wp: Optional[object] = None
        self._src_spm = None
        self._tgt_spm = None
        self._vocab = None
        self._inv = None
        self._try_load()

    def _try_load(self) -> None:
        try:
            from .onnx_compat import configure_onnxruntime
            import sentencepiece as spm
            import onnxruntime as ort

            configure_onnxruntime()
            self._src_spm = spm.SentencePieceProcessor(
                model_file=_download(_SRC_REPO, "source.spm"))
            self._tgt_spm = spm.SentencePieceProcessor(
                model_file=_download(_SRC_REPO, "target.spm"))
            with open(_download(_SRC_REPO, "vocab.json"), encoding="utf-8") as f:
                self._vocab = json.load(f)
            self._inv = {v: k for k, v in self._vocab.items()}
            self._enc = ort.InferenceSession(
                _download(_ONNX_REPO, "onnx/encoder_model_quantized.onnx"))
            self._dec = ort.InferenceSession(
                _download(_ONNX_REPO, "onnx/decoder_model_quantized.onnx"))
            self._dec_wp = ort.InferenceSession(
                _download(_ONNX_REPO, "onnx/decoder_with_past_model_quantized.onnx"))
            self._loaded = True
            log.info("Translator: loaded opus-mt-en-zh ONNX (en→zh)")
        except Exception as e:  # noqa: BLE001 — any load failure → no-op
            log.warning(
                "Translator: unavailable (%s); scene embedding stays English", e)
            self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def translate(self, text: str) -> str:
        if not self._loaded or not text or not text.strip():
            return text
        try:
            return self._translate(text)
        except Exception as e:  # noqa: BLE001 — runtime failure → input unchanged
            log.warning("Translator: translate failed (%s); returning input", e)
            return text

    def _tokenize(self, text: str) -> List[int]:
        return [self._vocab.get(p, _UNK) for p in self._src_spm.encode_as_pieces(text)]

    def _detokenize(self, ids: List[int]) -> str:
        return self._tgt_spm.decode_pieces([self._inv.get(i, "") for i in ids])

    def _translate(self, text: str) -> str:
        ids = self._tokenize(text)
        mask = np.ones((1, len(ids)), dtype=np.int64)
        hidden = self._enc.run(None, {"input_ids": np.array([ids], dtype=np.int64),
                                      "attention_mask": mask})[0]

        # First step runs the full decoder (no past), then switches to the
        # incremental with-past decoder for each following token.
        out = self._dec.run(None, {
            "input_ids": np.array([[_DEC_START]], dtype=np.int64),
            "encoder_hidden_states": hidden,
            "encoder_attention_mask": mask,
        })
        logits = out[0][0, -1, :].astype(np.float32)
        present = {o.name.replace("present.", "past_key_values."): v
                   for o, v in zip(self._dec.get_outputs()[1:], out[1:])}

        out_ids: List[int] = []
        for _ in range(self._max_length):
            if self._repetition_penalty != 1.0:
                for pid in set(out_ids):
                    s = logits[pid]
                    logits[pid] = s / self._repetition_penalty if s >= 0 else s * self._repetition_penalty
            token = int(np.argmax(logits))
            if token == _EOS:
                break
            out_ids.append(token)
            o2 = self._dec_wp.run(None, {
                "input_ids": np.array([[token]], dtype=np.int64),
                "encoder_attention_mask": mask,
                **present,
            })
            logits = o2[0][0, -1, :].astype(np.float32)
            new_present = {o.name.replace("present.", "past_key_values."): v
                           for o, v in zip(self._dec_wp.get_outputs()[1:], o2[1:])}
            present.update(new_present)
        return self._detokenize(out_ids)
