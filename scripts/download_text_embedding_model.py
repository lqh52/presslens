#!/usr/bin/env python3
"""Download the local sentence-embedding model used by the demo."""

from huggingface_hub import snapshot_download


snapshot_download(
    "sentence-transformers/all-MiniLM-L6-v2",
    local_dir="models/text-embedding/all-MiniLM-L6-v2",
    ignore_patterns=["*.onnx", "*.h5", "*.ot", "openvino/*", "pytorch_model.bin"],
)
print("Model ready at models/text-embedding/all-MiniLM-L6-v2")
