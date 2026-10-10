"""Sentence encoder: the pretrained MiniLM transformer (all-MiniLM-L6-v2) run with ONNX Runtime.

ONNX Runtime gives the same embeddings as sentence-transformers/PyTorch but uses a fraction of the
memory, so the app fits on free hosting. Model files are fetched by setup_models.py.
"""
import os

import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "minilm")
EMBED_DIM = 384

_session = _tokenizer = None


def _load():
    global _session, _tokenizer
    if _session is None:
        import onnxruntime as ort
        from tokenizers import Tokenizer
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        _session = ort.InferenceSession(os.path.join(MODEL_DIR, "model.onnx"), opts,
                                        providers=["CPUExecutionProvider"])
        _tokenizer = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
        _tokenizer.enable_padding()
        _tokenizer.enable_truncation(256)


def embed(sentences, batch_size=64):
    """Encode sentences into L2-normalised 384-dim vectors (mean pooling, like sentence-transformers)."""
    _load()
    out = []
    for i in range(0, len(sentences), batch_size):
        enc = _tokenizer.encode_batch(list(sentences[i:i + batch_size]))
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        hidden = _session.run(None, {"input_ids": ids, "attention_mask": mask,
                                     "token_type_ids": np.zeros_like(ids)})[0]
        vecs = (hidden * mask[..., None]).sum(1) / mask.sum(1, keepdims=True)
        out.append(vecs / np.linalg.norm(vecs, axis=1, keepdims=True))
    return np.vstack(out).astype(np.float32)
