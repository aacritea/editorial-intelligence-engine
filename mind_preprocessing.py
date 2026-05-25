"""
mind_preprocessing.py
Microsoft MIND Dataset Preprocessing Pipeline
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─── Loaders ────────────────────────────────────────────────────────────────

NEWS_COLS = ["news_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]
BEHAVIORS_COLS = ["impression_id", "user_id", "time", "history", "impressions"]


def load_news(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=NEWS_COLS, usecols=range(len(NEWS_COLS)))
    df["title"] = df["title"].str.strip()
    df["category"] = df["category"].str.lower().str.strip()
    df["subcategory"] = df["subcategory"].str.lower().str.strip()
    df["abstract"] = df["abstract"].fillna("").str.strip()
    logger.info("Loaded news.tsv — %d articles", len(df))
    return df


def load_behaviors(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=BEHAVIORS_COLS, usecols=range(len(BEHAVIORS_COLS)))
    df["time"] = pd.to_datetime(df["time"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    df["history"] = df["history"].fillna("").str.strip()
    logger.info("Loaded behaviors.tsv — %d sessions", len(df))
    return df


# ─── Impression Parser ───────────────────────────────────────────────────────

def parse_impressions(behaviors: pd.DataFrame) -> pd.DataFrame:
    """Explode impressions into (impression_id, user_id, time, news_id, clicked) rows."""
    records = []
    for idx, row in enumerate(behaviors.itertuples(index=False)):
        if not isinstance(row.impressions, str) or not row.impressions.strip():
            continue
        for item in row.impressions.strip().split():
            parts = item.split("-")
            if len(parts) != 2:
                continue
            news_id, label = parts
            records.append((
                row.impression_id,
                row.user_id,
                row.time,
                row.history,
                news_id,
                int(label),
            ))
        if idx % 10000 == 0:
            logger.info("Processed %d behavior rows", idx)
    df = pd.DataFrame(records, columns=[
        "impression_id",
        "user_id",
        "time",
        "history",
        "news_id",
        "clicked",
    ])
    logger.info("Parsed impressions — %d (impression, article) pairs | CTR=%.4f",
                len(df), df["clicked"].mean())
    return df

# ─── History Features ────────────────────────────────────────────────────────

def build_history_features(impressions: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
    """Enrich each impression with user history click count and category diversity."""
    news_cat = news.set_index("news_id")["category"]

    def _history_stats(hist_str: str):
        ids = hist_str.split() if hist_str else []
        cats = [news_cat.get(nid) for nid in ids if news_cat.get(nid) is not None]
        return len(ids), len(set(cats))

    stats = impressions["history"].map(_history_stats)
    impressions = impressions.copy()
    impressions[["history_len", "history_cat_diversity"]] = pd.DataFrame(stats.tolist(), index=impressions.index)
    return impressions


# ─── Article Features ────────────────────────────────────────────────────────

def build_article_features(impressions: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
    """Attach article-level features and compute global popularity signals."""
    # Global click stats per article
    pop = (
        impressions.groupby("news_id")["clicked"]
        .agg(global_impressions="count", global_clicks="sum")
        .assign(global_ctr=lambda d: d["global_clicks"] / d["global_impressions"].clip(lower=1))
        .reset_index()
    )

    df = (
        impressions
        .merge(news[["news_id", "category", "subcategory", "title", "abstract"]], on="news_id", how="left")
        .merge(pop, on="news_id", how="left")
    )
    df["title_len"] = df["title"].str.split().str.len().fillna(0).astype(int)
    df["has_abstract"] = df["abstract"].str.len().gt(0).astype(int)
    return df


# ─── Category Encoding ───────────────────────────────────────────────────────

def encode_categories(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ("category", "subcategory"):
        df[f"{col}_id"] = df[col].astype("category").cat.codes
    return df


# ─── Session-level Features ──────────────────────────────────────────────────

def build_session_features(df: pd.DataFrame) -> pd.DataFrame:
    session_stats = (
        df.groupby("impression_id")
        .agg(session_size=("news_id", "count"))
        .reset_index()
    )
    return df.merge(session_stats, on="impression_id", how="left")


# ─── Pipeline ────────────────────────────────────────────────────────────────

def preprocess(
    news_path: str | Path,
    behaviors_path: str | Path,
    output_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    news = load_news(news_path)
    behaviors = load_behaviors(behaviors_path)

    # Development subset for faster iteration
    #behaviors = behaviors.sample(50000, random_state=42)
    MAX_SESSIONS = 300000

    if len(behaviors) > MAX_SESSIONS:
        behaviors = behaviors.sample(MAX_SESSIONS, random_state=42)

    logger.info(
        "Using sampled subset — %d sessions",
        len(behaviors)
    )

    df = parse_impressions(behaviors)

    # Save intermediate parsed impressions
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df.to_parquet("data/processed/impressions_only.parquet", index=False)
    logger.info("Saved intermediate impressions parquet")

    df = build_history_features(df, news)
    df = build_article_features(df, news)
    df = build_session_features(df)
    df = encode_categories(df)

    # Final schema cleanup
    df = df.sort_values(["impression_id", "news_id"]).reset_index(drop=True)

    feature_cols = [
        "impression_id", "user_id", "time", "news_id",
        "clicked",
        "category", "subcategory", "category_id", "subcategory_id",
        "title", "title_len", "has_abstract",
        "global_impressions", "global_clicks", "global_ctr",
        "history_len", "history_cat_diversity",
        "session_size",
    ]
    df = df[feature_cols]

    logger.info("Final dataframe shape: %s", df.shape)
    logger.info("Feature columns: %s", feature_cols)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        logger.info("Saved to %s", output_path)

    return df


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MIND dataset preprocessing pipeline")
    parser.add_argument("--news", required=True, help="Path to news.tsv")
    parser.add_argument("--behaviors", required=True, help="Path to behaviors.tsv")
    parser.add_argument("--output", default="data/processed/mind_clean.parquet", help="Output parquet path")
    args = parser.parse_args()

    df = preprocess(args.news, args.behaviors, args.output)
    print(df.head())
    print(df.dtypes)