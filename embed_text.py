"""Step 3 of building external.parquet: embed each business description with approved local models.

Each model is loaded at the pinned revision from the handout and run locally; it only ever sees the
filing text we extracted (never a question about a named company). A description is split into
~200-word chunks (first 12 chunks), each chunk is embedded, and the unit-normalised mean is kept.
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

MODELS = {
    "e5": ("intfloat/e5-large-v2", "b322e09026e4ea05f42beadf4d661fb4e101d311", "passage: "),
    "bge": ("BAAI/bge-base-en-v1.5", "70e8c52057794ed37381f7da00a03097927500a7", ""),
    "mpnet": ("sentence-transformers/all-mpnet-base-v2", "5681fe04da6e48e851d5dd1af673670cdb299753", ""),
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", "7dbbc90392e2f80f3d3c277d6e90027e55de9125", ""),
    "gte": ("thenlper/gte-base", "7cb59c7765dcf58720e0e63bdbbf8c10cbba8668", ""),
}
CHUNK, MAX_CHUNKS = 200, 12


def chunks(text):
    w = text.split()
    return [" ".join(w[i:i + CHUNK]) for i in range(0, len(w), CHUNK)][:MAX_CHUNKS] or [""]


def main(names):
    torch.manual_seed(0)
    texts = pd.read_parquet("cache/texts.parquet")
    for name in names:
        repo, rev, prefix = MODELS[name]
        model = SentenceTransformer(repo, revision=rev, device="cpu")
        model.max_seq_length = 512
        embs = []
        for t in texts.text:
            ch = [prefix + x for x in chunks(t)]
            e = model.encode(ch, batch_size=8, normalize_embeddings=True, show_progress_bar=False)
            v = e.mean(0)
            embs.append(v / np.linalg.norm(v))
        np.save(f"cache/emb_{name}.npy", np.vstack(embs).astype(np.float32))
        print(name, np.vstack(embs).shape, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(MODELS))
