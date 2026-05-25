"""
embeddings.py
Sentence-Transformer Embedding Pipeline — all-MiniLM-L6-v2
"""

import hashlib
import logging
import os
import pickle
from pathlib import Path
from sys import prefix
from sys import prefix
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ─── Config ──────────────────────────────────────────────────────────────────

MODEL_NAME     = "all-MiniLM-L6-v2"
EMBEDDING_DIM  = 384
DEFAULT_BATCH  = 256
CACHE_DIR      = Path("data/cache/embeddings")
CACHE_VERSION  = "v1"


# ─── Device Resolution ───────────────────────────────────────────────────────

def resolve_device(preferred: Optional[str] = None) -> str:
    if preferred:
        return preferred
    if torch.cuda.is_available():
        dev = "cuda"
    elif torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"
    logger.info("Embedding device: %s", dev)
    return dev


# ─── Model Loader ────────────────────────────────────────────────────────────

_MODEL_CACHE: dict[str, SentenceTransformer] = {}


def load_model(model_name: str = MODEL_NAME, device: Optional[str] = None) -> SentenceTransformer:
    device = resolve_device(device)
    key = f"{model_name}:{device}"
    if key not in _MODEL_CACHE:
        logger.info("Loading model %s on %s", model_name, device)
        model = SentenceTransformer(model_name, device=device)
        model.eval()
        _MODEL_CACHE[key] = model
        logger.info("Model loaded — embedding dim: %d", model.get_sentence_embedding_dimension())
    return _MODEL_CACHE[key]


# ─── Cache Utilities ─────────────────────────────────────────────────────────

def _cache_key(texts: list[str], model_name: str) -> str:
    content = model_name + CACHE_VERSION + "".join(texts)
    return hashlib.sha256(content.encode()).hexdigest()


def _cache_path(cache_key: str, cache_dir: Path) -> Path:
    return cache_dir / f"{cache_key}.pkl"


def _load_cache(cache_key: str, cache_dir: Path) -> Optional[np.ndarray]:
    path = _cache_path(cache_key, cache_dir)
    if path.exists():
        with open(path, "rb") as f:
            data = pickle.load(f)
        logger.info("Cache hit — loaded embeddings from %s", path)
        return data
    return None


def _save_cache(embeddings: np.ndarray, cache_key: str, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_key, cache_dir)
    with open(path, "wb") as f:
        pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Cached embeddings → %s", path)


# ─── Partial Cache (per-text) ─────────────────────────────────────────────────

class EmbeddingCache:
    """Persistent per-text cache backed by a single numpy memmap index."""

    def __init__(self, cache_dir: Path = CACHE_DIR, dim: int = EMBEDDING_DIM):
        self.cache_dir = Path(cache_dir)
        self.dim = dim
        self.index_path = self.cache_dir / "index.pkl"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, np.ndarray] = self._load_index()

    def _load_index(self) -> dict[str, np.ndarray]:
        if self.index_path.exists():
            with open(self.index_path, "rb") as f:
                idx = pickle.load(f)
            logger.info("Loaded embedding index — %d entries", len(idx))
            return idx
        return {}

    def _flush_index(self) -> None:
        with open(self.index_path, "wb") as f:
            pickle.dump(self._index, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _text_key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def get(self, texts: list[str]) -> tuple[dict[int, np.ndarray], list[int], list[str]]:
        """
        Returns:
            cached   — {original_index: embedding}
            miss_idx — list of original indices not in cache
            miss_txt — corresponding texts to embed
        """
        cached: dict[int, np.ndarray] = {}
        miss_idx, miss_txt = [], []
        for i, text in enumerate(texts):
            key = self._text_key(text)
            if key in self._index:
                cached[i] = self._index[key]
            else:
                miss_idx.append(i)
                miss_txt.append(text)
        return cached, miss_idx, miss_txt

    def put(self, texts: list[str], embeddings: np.ndarray) -> None:
        for text, vec in zip(texts, embeddings):
            self._index[self._text_key(text)] = vec
        self._flush_index()

    def clear(self) -> None:
        self._index = {}
        if self.index_path.exists():
            self.index_path.unlink()
        logger.info("Embedding cache cleared")


# ─── Core Embedding Engine ───────────────────────────────────────────────────

def embed_texts(
    texts: list[str],
    model: Optional[SentenceTransformer] = None,
    batch_size: int = DEFAULT_BATCH,
    device: Optional[str] = None,
    normalize: bool = True,
    cache: Optional[EmbeddingCache] = None,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Embed a list of texts with optional per-text caching.

    Returns:
        np.ndarray of shape (len(texts), EMBEDDING_DIM), float32
    """
    texts = [t.strip() if isinstance(t, str) else "" for t in texts]

    if model is None:
        model = load_model(device=device)

    embeddings = np.zeros((len(texts), model.get_sentence_embedding_dimension()), dtype=np.float32)

    # Resolve cache misses
    if cache is not None:
        cached, miss_idx, miss_txt = cache.get(texts)
        for i, vec in cached.items():
            embeddings[i] = vec
        logger.info("Cache: %d hits / %d misses", len(cached), len(miss_idx))
    else:
        miss_idx = list(range(len(texts)))
        miss_txt = texts

    if miss_txt:
        with torch.inference_mode():
            vecs = model.encode(
                miss_txt,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=normalize,
                convert_to_numpy=True,
                device=device,
            )
        for orig_i, vec in zip(miss_idx, vecs):
            embeddings[orig_i] = vec

        if cache is not None:
            cache.put(miss_txt, vecs)

    return embeddings.astype(np.float32)


# ─── DataFrame Integration ───────────────────────────────────────────────────

def embed_dataframe(
    df: pd.DataFrame,
    text_col: str = "title",
    prefix: str = "emb",
    model: Optional[SentenceTransformer] = None,
    batch_size: int = DEFAULT_BATCH,
    device: Optional[str] = None,
    normalize: bool = True,
    use_cache: bool = True,
    cache_dir: Path = CACHE_DIR,
    as_column: bool = False,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Embed a DataFrame text column and return the original df enriched with
    either:
        - `as_column=False` → expanded embedding dimensions as `{prefix}_0` … `{prefix}_383`
        - `as_column=True`  → single object column `{prefix}_vec` storing np.ndarray per row

    Args:
        df          : source DataFrame
        text_col    : column with raw text
        prefix      : column prefix for output embeddings
        model       : pre-loaded SentenceTransformer (loaded lazily if None)
        batch_size  : inference batch size
        device      : 'cuda' | 'mps' | 'cpu' | None (auto-detect)
        normalize   : L2-normalize embeddings
        use_cache   : enable per-text persistent cache
        cache_dir   : cache directory path
        as_column   : store vectors as a single object column instead of expanding
        show_progress: tqdm progress bar

    Returns:
        pd.DataFrame with embedding columns appended
    """
    texts_df = (
    df[["news_id", text_col]]
    .drop_duplicates("news_id")
    .reset_index(drop=True)
    )

    logger.info(
        "Embedding unique articles only — %d unique news items",
        len(texts_df),
    )

    texts = texts_df[text_col].fillna("").tolist()

    cache = EmbeddingCache(cache_dir) if use_cache else None

    embeddings = embed_texts(
        texts,
        model=model,
        batch_size=batch_size,
        device=device,
        normalize=normalize,
        cache=cache,
        show_progress=show_progress,
)

    emb_cols = [
        f"{prefix}_{i}"
        for i in range(embeddings.shape[1])
    ]

    emb_df = pd.DataFrame(
        embeddings,
        columns=emb_cols,
    )

    texts_df = pd.concat(
        [texts_df[["news_id"]], emb_df],
        axis=1,
    )

    logger.info("Merging embeddings back to full dataframe")

    out = df.merge(
        texts_df,
        on="news_id",
        how="left",
    )

    return out

# ─── Multi-field Fusion Embedding ────────────────────────────────────────────

def embed_fused(
    df: pd.DataFrame,
    fields: list[tuple[str, float]],          # [(col, weight), ...]
    prefix: str = "fused_emb",
    model: Optional[SentenceTransformer] = None,
    batch_size: int = DEFAULT_BATCH,
    device: Optional[str] = None,
    use_cache: bool = True,
    cache_dir: Path = CACHE_DIR,
) -> pd.DataFrame:
    """
    Produce a single weighted-average embedding from multiple text fields.

    Example:
        embed_fused(df, [("title", 0.7), ("abstract", 0.3)])
    """
    if model is None:
        model = load_model(device=device)

    cache = EmbeddingCache(cache_dir) if use_cache else None
    weighted = np.zeros((len(df), model.get_sentence_embedding_dimension()), dtype=np.float32)
    total_weight = sum(w for _, w in fields)

    for col, weight in fields:
        texts = df[col].fillna("").tolist()
        vecs = embed_texts(texts, model=model, batch_size=batch_size,
                           device=device, normalize=False, cache=cache, show_progress=False)
        weighted += (weight / total_weight) * vecs

    # L2-normalize fused vector
    norms = np.linalg.norm(weighted, axis=1, keepdims=True).clip(min=1e-9)
    weighted /= norms

    emb_df = pd.DataFrame(
        weighted,
        columns=[f"{prefix}_{i}" for i in range(weighted.shape[1])],
        index=df.index,
    )
    return pd.concat([df, emb_df], axis=1)


# ─── Similarity Utilities ────────────────────────────────────────────────────

def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between two sets of L2-normalized vectors."""
    a = a / np.linalg.norm(a, axis=1, keepdims=True).clip(min=1e-9)
    b = b / np.linalg.norm(b, axis=1, keepdims=True).clip(min=1e-9)
    return (a @ b.T).astype(np.float32)


def top_k_similar(
    query_vec: np.ndarray,
    corpus_vecs: np.ndarray,
    k: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices, scores) of top-k most similar corpus vectors."""
    scores = cosine_similarity_matrix(query_vec.reshape(1, -1), corpus_vecs)[0]
    top_k_idx = np.argpartition(scores, -k)[-k:]
    top_k_idx = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]
    return top_k_idx, scores[top_k_idx]


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate sentence-transformer embeddings for MIND dataset")
    parser.add_argument("--input",      required=True,  help="Input parquet path")
    parser.add_argument("--output",     required=True,  help="Output parquet path")
    parser.add_argument("--text-col",   default="title")
    parser.add_argument("--prefix",     default="emb")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--device",     default=None,   help="cuda | mps | cpu")
    parser.add_argument("--fuse-abstract", action="store_true",
                        help="Fuse title (0.7) + abstract (0.3)")
    parser.add_argument("--as-column",  action="store_true",
                        help="Store vectors as object column instead of expanding dims")
    parser.add_argument("--no-cache",   action="store_true")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    model = load_model(device=args.device)

    if args.fuse_abstract and "abstract" in df.columns:
        out = embed_fused(
            df,
            fields=[("title", 0.7), ("abstract", 0.3)],
            prefix=args.prefix,
            model=model,
            batch_size=args.batch_size,
            device=args.device,
            use_cache=not args.no_cache,
        )
    else:
        out = embed_dataframe(
            df,
            text_col=args.text_col,
            prefix=args.prefix,
            model=model,
            batch_size=args.batch_size,
            device=args.device,
            use_cache=not args.no_cache,
            as_column=args.as_column,
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    logger.info("Saved → %s  shape=%s", args.output, out.shape)
    print(out.filter(regex=f"^{args.prefix}_").iloc[:2])

    