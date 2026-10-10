"""Download the MiniLM sentence encoder (ONNX) into models/minilm. Runs once at build time."""
import os
import shutil

from huggingface_hub import hf_hub_download

from nlp_utils import MODEL_DIR

REPO = "sentence-transformers/all-MiniLM-L6-v2"

os.makedirs(MODEL_DIR, exist_ok=True)
for remote, local in [("onnx/model.onnx", "model.onnx"), ("tokenizer.json", "tokenizer.json")]:
    target = os.path.join(MODEL_DIR, local)
    if not os.path.exists(target):
        shutil.copy(hf_hub_download(REPO, remote), target)
    print("ready:", target)
