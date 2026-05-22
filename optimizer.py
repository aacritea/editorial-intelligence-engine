"""
optimizer.py
Front-Page Ranking Optimizer — Editorial Intelligence System

Slot assignment pipeline:
  1. Score articles by engagement × position visibility
  2. Enforce category diversity quotas
  3. Deduplicate overlapping topic clusters
  4. Produce a ranked slot manifest for front-page rendering
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


# ─── Layout Definitions ──────────────────────────────────────────────────────

class SlotZone(str, Enum):
    HERO        = "hero"
    TOP_RAIL    = "top_rail"
    MID_RAIL    = "mid_rail"
    SIDEBAR     = "sidebar"
    BOTTOM_RAIL = "bottom_rail"
    BELOW_FOLD  = "below_fold"


@dataclass(frozen=True)
class LayoutSlot:
    slot_id:    int
    zone:       SlotZone
    position:   int          # 1-indexed within zone
    visibility: float        # ∈ (0, 1] — editorial eye-tracking weight


# Standard front-page layout — 20 slots
DEFAULT_LAYOUT: list[LayoutSlot] = [
    # Hero banner
    LayoutSlot(0,  SlotZone.HERO,        1, 1.000),
    # Top rail (positions 1-4)
    LayoutSlot(1,  SlotZone.TOP_RAIL,    1, 0.820),
    LayoutSlot(2,  SlotZone.TOP_RAIL,    2, 0.760),
    LayoutSlot(3,  SlotZone.TOP_RAIL,    3, 0.700),
    LayoutSlot(4,  SlotZone.TOP_RAIL,    4, 0.640),
    # Sidebar (positions 1-4)
    LayoutSlot(5,  SlotZone.SIDEBAR,     1, 0.580),
    LayoutSlot(6,  SlotZone.SIDEBAR,     2, 0.520),
    LayoutSlot(7,  SlotZone.SIDEBAR,     3, 0.460),
    LayoutSlot(8,  SlotZone.SIDEBAR,     4, 0.400),
    # Mid rail (positions 1-4)
    LayoutSlot(9,  SlotZone.MID_RAIL,    1, 0.380),
    LayoutSlot(10, SlotZone.MID_RAIL,    2, 0.340),
    LayoutSlot(11, SlotZone.MID_RAIL,    3, 0.300),
    LayoutSlot(12, SlotZone.MID_RAIL,    4, 0.260),
    # Bottom rail (positions 1-4)
    LayoutSlot(13, SlotZone.BOTTOM_RAIL, 1, 0.200),
    LayoutSlot(14, SlotZone.BOTTOM_RAIL, 2, 0.175),
    LayoutSlot(15, SlotZone.BOTTOM_RAIL, 3, 0.150),
    LayoutSlot(16, SlotZone.BOTTOM_RAIL, 4, 0.125),
    # Below fold
    LayoutSlot(17, SlotZone.BELOW_FOLD,  1, 0.080),
    LayoutSlot(18, SlotZone.BELOW_FOLD,  2, 0.060),
    LayoutSlot(19, SlotZone.BELOW_FOLD,  3, 0.040),
]


# ─── Optimizer Config ────────────────────────────────────────────────────────

@dataclass
class OptimizerConfig:
    # Scoring weights
    engagement_weight:   float = 0.65   # ranker score weight
    recency_weight:      float = 0.20   # time-decay contribution
    sentiment_weight:    float = 0.10   # positive sentiment bonus
    clickbait_penalty:   float = 0.05   # cb_composite_score penalty

    # Recency decay
    recency_half_life_hours: float = 6.0

    # Category diversity
    max_per_category:    int   = 3       # hard cap per category on front page
    min_categories:      int   = 4       # minimum distinct categories required

    # Topic deduplication
    dedup_similarity_threshold: float = 0.82   # cosine sim above which second article is suppressed
    dedup_embedding_prefix:     str   = "emb"  # column prefix for embedding dims

    # Slot constraints
    hero_categories_allowed: Optional[list[str]] = None   # None = any
    sidebar_max_entertainment: int = 1

    # Column names
    score_col:       str = "rank_score"
    category_col:    str = "category"
    news_id_col:     str = "news_id"
    time_col:        str = "time"
    sentiment_col:   str = "sent_polarity"
    clickbait_col:   str = "cb_composite_score"
    topic_id_col:    str = "topic_primary"


# ─── Scoring ─────────────────────────────────────────────────────────────────

def _recency_decay(df: pd.DataFrame, time_col: str, half_life_hours: float) -> pd.Series:
    """Exponential decay: score=1 at publish time, halves every `half_life_hours`."""
    if time_col not in df.columns:
        return pd.Series(1.0, index=df.index)
    now   = df[time_col].max()
    delta = (now - df[time_col]).dt.total_seconds() / 3600
    return np.exp(-np.log(2) * delta / half_life_hours).fillna(0.5)


def compute_composite_score(df: pd.DataFrame, cfg: OptimizerConfig) -> pd.Series:
    """
    Composite engagement signal:
        score = w_eng * rank_score
              + w_rec * recency_decay
              + w_sen * norm(sent_polarity)
              - w_cb  * cb_composite_score
    All components normalised to [0, 1] before weighting.
    """
    def _norm(s: pd.Series) -> pd.Series:
        lo, hi = s.min(), s.max()
        return (s - lo) / (hi - lo + 1e-9)

    engagement = _norm(df[cfg.score_col]) if cfg.score_col in df.columns else pd.Series(0.5, index=df.index)
    recency    = _recency_decay(df, cfg.time_col, cfg.recency_half_life_hours)
    sentiment  = _norm(df[cfg.sentiment_col]) if cfg.sentiment_col in df.columns else pd.Series(0.5, index=df.index)
    clickbait  = df[cfg.clickbait_col].clip(0, 1) if cfg.clickbait_col in df.columns else pd.Series(0.0, index=df.index)

    composite = (
        cfg.engagement_weight * engagement
        + cfg.recency_weight  * recency
        + cfg.sentiment_weight * sentiment
        - cfg.clickbait_penalty * clickbait
    ).clip(0, 1)

    return composite


# ─── Topic Deduplication ─────────────────────────────────────────────────────

def _extract_embeddings(df: pd.DataFrame, prefix: str) -> Optional[np.ndarray]:
    emb_cols = sorted([c for c in df.columns if c.startswith(f"{prefix}_") and c[len(prefix)+1:].isdigit()])
    if not emb_cols:
        return None
    return df[emb_cols].values.astype(np.float32)


def deduplicate_topics(
    df: pd.DataFrame,
    cfg: OptimizerConfig,
) -> pd.DataFrame:
    """
    Greedy cosine-similarity deduplication.
    Articles exceeding the similarity threshold to any already-selected article
    are suppressed (flagged `dedup_suppressed=True`).
    Falls back to exact `topic_id_col` matching when no embeddings are present.
    """
    df = df.copy().sort_values("composite_score", ascending=False).reset_index(drop=True)
    df["dedup_suppressed"] = False

    embeddings = _extract_embeddings(df, cfg.dedup_embedding_prefix)

    if embeddings is not None:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-9)
        normed = embeddings / norms
        selected_vecs: list[np.ndarray] = []

        for i in range(len(df)):
            if not selected_vecs:
                selected_vecs.append(normed[i])
                continue
            pool  = np.vstack(selected_vecs)
            sims  = (pool @ normed[i]).max()
            if sims >= cfg.dedup_similarity_threshold:
                df.at[i, "dedup_suppressed"] = True
            else:
                selected_vecs.append(normed[i])
    else:
        # Fallback: exact topic cluster dedup
        if cfg.topic_id_col in df.columns:
            seen_topics: set = set()
            for i, row in df.iterrows():
                tid = row[cfg.topic_id_col]
                if tid in seen_topics:
                    df.at[i, "dedup_suppressed"] = True
                else:
                    seen_topics.add(tid)

    n_suppressed = df["dedup_suppressed"].sum()
    logger.info("Topic dedup: suppressed %d / %d articles", n_suppressed, len(df))
    return df


# ─── Category Diversity Enforcer ─────────────────────────────────────────────

def enforce_category_diversity(
    df: pd.DataFrame,
    cfg: OptimizerConfig,
    n_slots: int,
) -> pd.DataFrame:
    """
    Greedy selection enforcing:
      - max `cfg.max_per_category` articles per category
      - at least `cfg.min_categories` distinct categories in final slate

    Articles not satisfying constraints are pushed to overflow (rank > n_slots).
    """
    df = df.copy().sort_values("composite_score", ascending=False).reset_index(drop=True)
    df["diversity_rank"] = -1

    category_counts: dict[str, int] = {}
    selected_indices: list[int]     = []
    overflow_indices: list[int]     = []

    for i, row in df.iterrows():
        cat   = row.get(cfg.category_col, "unknown")
        count = category_counts.get(cat, 0)

        if count < cfg.max_per_category and len(selected_indices) < n_slots:
            selected_indices.append(i)
            category_counts[cat] = count + 1
        else:
            overflow_indices.append(i)

    # Check minimum category constraint — backfill from overflow if needed
    unique_cats = len({df.at[i, cfg.category_col] for i in selected_indices if cfg.category_col in df.columns})
    if unique_cats < cfg.min_categories:
        needed_cats = set(df[cfg.category_col].unique()) - {
            df.at[i, cfg.category_col] for i in selected_indices
        }
        for i in overflow_indices[:]:
            if not needed_cats:
                break
            cat = df.at[i, cfg.category_col] if cfg.category_col in df.columns else "unknown"
            if cat in needed_cats and len(selected_indices) < n_slots:
                selected_indices.append(i)
                overflow_indices.remove(i)
                needed_cats.discard(cat)

    for rank, i in enumerate(selected_indices):
        df.at[i, "diversity_rank"] = rank

    logger.info(
        "Diversity enforcement: %d selected across %d categories",
        len(selected_indices),
        len({df.at[i, cfg.category_col] for i in selected_indices if cfg.category_col in df.columns}),
    )
    return df


# ─── Slot Constraint Filters ─────────────────────────────────────────────────

def apply_slot_constraints(
    candidates: pd.DataFrame,
    slot: LayoutSlot,
    cfg: OptimizerConfig,
    assigned_categories: dict[str, int],
) -> pd.DataFrame:
    """Filter candidate pool to articles eligible for a given slot."""
    pool = candidates.copy()

    if slot.zone == SlotZone.HERO and cfg.hero_categories_allowed:
        pool = pool[pool[cfg.category_col].isin(cfg.hero_categories_allowed)]

    if slot.zone == SlotZone.SIDEBAR:
        ent_count = assigned_categories.get("entertainment", 0)
        if ent_count >= cfg.sidebar_max_entertainment:
            pool = pool[pool[cfg.category_col] != "entertainment"]

    return pool


# ─── Visibility-Weighted Scorer ──────────────────────────────────────────────

def visibility_weighted_score(composite: float, visibility: float) -> float:
    """Final slot utility = composite engagement × position visibility."""
    return float(composite * visibility)


# ─── Core Optimizer ──────────────────────────────────────────────────────────

@dataclass
class SlotAssignment:
    slot_id:          int
    zone:             str
    position:         int
    visibility:       float
    news_id:          str
    category:         str
    composite_score:  float
    visibility_score: float
    title:            str = ""
    rank_score:       float = 0.0


def optimize_front_page(
    df: pd.DataFrame,
    cfg:    Optional[OptimizerConfig] = None,
    layout: Optional[list[LayoutSlot]] = None,
) -> tuple[list[SlotAssignment], pd.DataFrame]:
    """
    Full front-page optimization pipeline.

    Steps:
      1. Composite scoring (engagement × recency × sentiment − clickbait)
      2. Topic deduplication via embedding cosine similarity
      3. Category diversity enforcement
      4. Greedy slot assignment with per-slot constraints
      5. Visibility-weighted final utility scoring

    Args:
        df     : enriched article DataFrame (output of ranker + feature pipeline)
        cfg    : OptimizerConfig (defaults used if None)
        layout : list of LayoutSlots (DEFAULT_LAYOUT used if None)

    Returns:
        assignments : ordered list of SlotAssignment
        manifest_df : DataFrame with full slot manifest + scores
    """
    cfg    = cfg    or OptimizerConfig()
    layout = layout or DEFAULT_LAYOUT
    n_slots = len(layout)

    df = df.copy().reset_index(drop=True)

    # ── 1. Composite score
    df["composite_score"] = compute_composite_score(df, cfg)
    logger.info("Composite scores computed — mean=%.4f std=%.4f",
                df["composite_score"].mean(), df["composite_score"].std())

    # ── 2. Topic deduplication
    df = deduplicate_topics(df, cfg)
    candidates = df[~df["dedup_suppressed"]].copy()
    logger.info("Candidates after dedup: %d", len(candidates))

    # ── 3. Diversity enforcement
    candidates = enforce_category_diversity(candidates, cfg, n_slots)
    viable = candidates[candidates["diversity_rank"] >= 0].sort_values("diversity_rank")
    logger.info("Viable articles after diversity pass: %d", len(viable))

    # ── 4. Greedy slot assignment
    assignments:         list[SlotAssignment] = []
    assigned_news_ids:   set[str]             = set()
    assigned_categories: dict[str, int]       = {}
    unassigned_pool      = viable.copy()

    for slot in sorted(layout, key=lambda s: s.slot_id):
        pool = apply_slot_constraints(
            unassigned_pool[~unassigned_pool[cfg.news_id_col].isin(assigned_news_ids)],
            slot, cfg, assigned_categories,
        )
        if pool.empty:
            logger.warning("Slot %d (%s/%d) — no eligible candidates", slot.slot_id, slot.zone, slot.position)
            continue

        # Best candidate = highest composite × visibility
        pool = pool.copy()
        pool["_slot_score"] = pool["composite_score"].apply(
            lambda s: visibility_weighted_score(s, slot.visibility)
        )
        best = pool.nlargest(1, "_slot_score").iloc[0]

        cat = best.get(cfg.category_col, "unknown")
        assigned_categories[cat] = assigned_categories.get(cat, 0) + 1
        assigned_news_ids.add(best[cfg.news_id_col])

        assignments.append(SlotAssignment(
            slot_id          = slot.slot_id,
            zone             = slot.zone.value,
            position         = slot.position,
            visibility       = slot.visibility,
            news_id          = best[cfg.news_id_col],
            category         = cat,
            composite_score  = float(best["composite_score"]),
            visibility_score = float(best["_slot_score"]),
            title            = best.get("title", ""),
            rank_score       = float(best.get(cfg.score_col, 0.0)),
        ))

    # ── 5. Build manifest DataFrame
    manifest_df = pd.DataFrame([
        {
            "slot_id":          a.slot_id,
            "zone":             a.zone,
            "position":         a.position,
            "visibility":       a.visibility,
            "news_id":          a.news_id,
            "category":         a.category,
            "title":            a.title,
            "rank_score":       a.rank_score,
            "composite_score":  a.composite_score,
            "visibility_score": a.visibility_score,
        }
        for a in assignments
    ])

    _log_manifest_summary(manifest_df, cfg)
    return assignments, manifest_df


# ─── Diagnostics ─────────────────────────────────────────────────────────────

def _log_manifest_summary(manifest: pd.DataFrame, cfg: OptimizerConfig) -> None:
    cat_dist = manifest["category"].value_counts().to_dict()
    zone_dist = manifest["zone"].value_counts().to_dict()
    logger.info(
        "Front-page manifest: %d slots filled | categories=%s | zones=%s",
        len(manifest), cat_dist, zone_dist,
    )


def manifest_report(manifest_df: pd.DataFrame) -> pd.DataFrame:
    """Human-readable audit table sorted by slot position."""
    cols = ["slot_id", "zone", "position", "visibility",
            "category", "title", "rank_score", "composite_score", "visibility_score"]
    return manifest_df[[c for c in cols if c in manifest_df.columns]].sort_values("slot_id")


# ─── Re-ranking Utilities ────────────────────────────────────────────────────

def boost_breaking_news(
    df: pd.DataFrame,
    news_ids: list[str],
    boost_factor: float = 1.5,
    score_col: str = "rank_score",
) -> pd.DataFrame:
    """Apply editorial boost to breaking-news articles before optimization."""
    df = df.copy()
    mask = df["news_id"].isin(news_ids)
    df.loc[mask, score_col] = (df.loc[mask, score_col] * boost_factor).clip(upper=1.0)
    logger.info("Breaking news boost applied to %d articles", mask.sum())
    return df


def pin_article(
    assignments: list[SlotAssignment],
    news_id: str,
    target_slot_id: int,
) -> list[SlotAssignment]:
    """
    Force a specific article to a target slot (editorial pin).
    Swaps the pinned article with whatever occupies the target slot.
    """
    assignments = list(assignments)
    target_idx   = next((i for i, a in enumerate(assignments) if a.slot_id == target_slot_id), None)
    article_idx  = next((i for i, a in enumerate(assignments) if a.news_id == news_id), None)

    if target_idx is None or article_idx is None:
        logger.warning("Pin failed — slot_id=%d or news_id=%s not found", target_slot_id, news_id)
        return assignments

    # Swap slot metadata
    a, b = assignments[article_idx], assignments[target_idx]
    assignments[article_idx] = SlotAssignment(**{**vars(a), "slot_id": b.slot_id,
                                                  "zone": b.zone, "position": b.position,
                                                  "visibility": b.visibility})
    assignments[target_idx]  = SlotAssignment(**{**vars(b), "slot_id": a.slot_id,
                                                  "zone": a.zone, "position": a.position,
                                                  "visibility": a.visibility})
    logger.info("Pinned news_id=%s to slot_id=%d", news_id, target_slot_id)
    return assignments


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Front-Page Ranking Optimizer")
    parser.add_argument("--input",            required=True,  help="Ranked parquet (output of ranker.py)")
    parser.add_argument("--output",           required=True,  help="Manifest parquet output path")
    parser.add_argument("--engagement-w",     type=float, default=0.65)
    parser.add_argument("--recency-w",        type=float, default=0.20)
    parser.add_argument("--sentiment-w",      type=float, default=0.10)
    parser.add_argument("--clickbait-pen",    type=float, default=0.05)
    parser.add_argument("--max-per-category", type=int,   default=3)
    parser.add_argument("--min-categories",   type=int,   default=4)
    parser.add_argument("--dedup-threshold",  type=float, default=0.82)
    parser.add_argument("--half-life-hours",  type=float, default=6.0)
    parser.add_argument("--boost-ids",        nargs="*",  default=[],
                        help="news_ids to apply breaking-news boost")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)

    cfg = OptimizerConfig(
        engagement_weight          = args.engagement_w,
        recency_weight             = args.recency_w,
        sentiment_weight           = args.sentiment_w,
        clickbait_penalty          = args.clickbait_pen,
        max_per_category           = args.max_per_category,
        min_categories             = args.min_categories,
        dedup_similarity_threshold = args.dedup_threshold,
        recency_half_life_hours    = args.half_life_hours,
    )

    if args.boost_ids:
        df = boost_breaking_news(df, args.boost_ids)

    assignments, manifest = optimize_front_page(df, cfg)

    from pathlib import Path
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(args.output, index=False)
    logger.info("Manifest saved → %s", args.output)
    print(manifest_report(manifest).to_string(index=False))