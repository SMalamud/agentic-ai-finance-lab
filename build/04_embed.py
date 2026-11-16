"""Embed every passage with a small open model and compute a 2D map.

Writes data/embeddings.npy (float16, unit-length rows, same order as passages.csv)
and adds map coordinates x, y to data/passages.csv.
"""
from pathlib import Path
import numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE

DATA = Path(__file__).resolve().parent.parent / "data"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"

if __name__ == "__main__":
    df = pd.read_csv(DATA / "passages.csv", keep_default_na=False)
    model = SentenceTransformer(MODEL, device="cpu")
    E = model.encode(df.text.tolist(), batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    np.save(DATA / "embeddings.npy", E.astype(np.float16))
    xy = TSNE(n_components=2, metric="cosine", init="pca", perplexity=30, random_state=0).fit_transform(E)
    df["x"], df["y"] = xy[:, 0].round(2), xy[:, 1].round(2)
    df.to_csv(DATA / "passages.csv", index=False)
    print(E.shape, "saved")
