"""
ranker.py
LightGBM Learning-to-Rank Pipeline — Editorial Intelligence System
"""

import json
import logging
import pickle
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(__name__)

# ─── Config ──────────────────────────────────────────────────────────────────

@dataclass
class RankerConfig:
    # LightGBM core
    objective: str           = "lambdarank"
    metric: str              = "ndcg"
    ndcg_eval_at: list[int]  = field(default_factory=lambda: [5, 10, 20])
    boosting_type: str       = "gbdt"
    num_leaves: int          = 63
    max_depth: int           = -1
    learning_rate: float     = 0.05
    n_estimators: int        = 1000
    min_child_samples: int   = 20
    feature_fraction: float  = 0.8
    bagging_fraction: float  = 0.8
    bagging_freq: int        = 5
    reg_alpha: float         = 0.1
    reg_lambda: float        = 0.1
    label_gain: list[float]  = field(default_factory=lambda: [0, 1, 3, 7, 15])

    # Training
    early_stopping_rounds: int = 50
    verbose_eval: int          = 100
    num_threads: int           = -1
    device: str                = "cpu"          # "gpu" if CUDA available

    # Data
    group_col: str             = "impression_id"
    label_col: str             = "clicked"
    val_size: float            = 0.2
    random_state: int          = 42

    # Artifacts
    model_dir: str             = "models/ranker"

    def to_lgb_params(self) -> dict:
        return {
            "objective":       self.objective,
            "metric":          self.metric,
            "ndcg_eval_at":    self.ndcg_eval_at,
            "boosting_type":   self.boosting_type,
            "num_leaves":      self.num_leaves,
            "max_depth":       self.max_depth,
            "learning_rate":   self.learning_rate,
            "min_child_samples": self.min_child_samples,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq":    self.bagging_freq,
            "reg_alpha":       self.reg_alpha,
            "reg_lambda":      self.reg_lambda,
            "label_gain":      self.label_gain,
            "num_threads":     self.num_threads,
            "device":          self.device,
            "verbosity":       -1,
        }


# ─── Feature Schema ──────────────────────────────────────────────────────────

EXCLUDE_COLS = {
    "impression_id", "user_id", "time", "news_id", "clicked",
    "history", "title", "abstract", "category", "subcategory",
    "sent_label", "cb_model_label", "topic_primary",
    "emb_vec", "fused_emb_vec",
}


def get_feature_cols(df: pd.DataFrame, exclude: Optional[set] = None) -> list[str]:
    excl = EXCLUDE_COLS | (exclude or set())
    numeric_types = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
    return [
        c for c in df.columns
        if c not in excl and df[c].dtype.name in numeric_types
    ]


# ─── Train / Val Split ───────────────────────────────────────────────────────

def group_train_val_split(
    df: pd.DataFrame,
    group_col: str,
    val_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Group-aware split: impression sessions stay intact in one partition.
    Prevents data leakage across user sessions.
    """
    groups = df[group_col].values
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    train_idx, val_idx = next(splitter.split(df, groups=groups))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df   = df.iloc[val_idx].reset_index(drop=True)

    logger.info(
        "Split → train: %d rows / %d groups | val: %d rows / %d groups",
        len(train_df), train_df[group_col].nunique(),
        len(val_df),   val_df[group_col].nunique(),
    )
    return train_df, val_df


# ─── Group Arrays ────────────────────────────────────────────────────────────

def build_group_array(df: pd.DataFrame, group_col: str) -> np.ndarray:
    """LightGBM `group` parameter: sorted consecutive group sizes."""
    counts = df.groupby(group_col, sort=False)[group_col].count()
    return counts.values.astype(np.int32)


# ─── Dataset Builder ─────────────────────────────────────────────────────────

def build_lgb_dataset(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    group_col: str,
    reference: Optional[lgb.Dataset] = None,
) -> lgb.Dataset:
    X = df[feature_cols].values.astype(np.float32)
    y = df[label_col].values.astype(np.int32)
    g = build_group_array(df, group_col)

    dataset = lgb.Dataset(
        data=X,
        label=y,
        group=g,
        feature_name=feature_cols,
        reference=reference,
        free_raw_data=False,
    )
    return dataset


# ─── NDCG Evaluator ──────────────────────────────────────────────────────────

def dcg_at_k(relevances: np.ndarray, k: int) -> float:
    r = relevances[:k].astype(float)
    if r.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, r.size + 2))
    return float(np.sum((2 ** r - 1) / discounts))


def ndcg_at_k(relevances: np.ndarray, k: int) -> float:
    ideal = np.sort(relevances)[::-1]
    idcg  = dcg_at_k(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(relevances, k) / idcg


def evaluate_ndcg(
    df: pd.DataFrame,
    score_col: str,
    label_col: str,
    group_col: str,
    k_values: list[int] = [5, 10, 20],
) -> dict[str, float]:
    results: dict[str, list[float]] = {f"ndcg@{k}": [] for k in k_values}

    for _, grp in df.groupby(group_col):
        grp_sorted = grp.sort_values(score_col, ascending=False)
        rel = grp_sorted[label_col].values
        for k in k_values:
            results[f"ndcg@{k}"].append(ndcg_at_k(rel, k))

    return {metric: float(np.mean(vals)) for metric, vals in results.items()}


# ─── Feature Importance ──────────────────────────────────────────────────────

def extract_feature_importance(
    booster: lgb.Booster,
    feature_cols: list[str],
    top_n: int = 30,
) -> pd.DataFrame:
    fi = pd.DataFrame({
        "feature":    feature_cols,
        "gain":       booster.feature_importance(importance_type="gain"),
        "split":      booster.feature_importance(importance_type="split"),
    })
    fi["gain_pct"]  = fi["gain"]  / fi["gain"].sum().clip(min=1)
    fi["split_pct"] = fi["split"] / fi["split"].sum().clip(min=1)
    fi = fi.sort_values("gain", ascending=False).reset_index(drop=True)

    logger.info("Top %d features by gain:\n%s", top_n, fi.head(top_n).to_string(index=False))
    return fi


# ─── Model Persistence ───────────────────────────────────────────────────────

def save_model(
    booster: lgb.Booster,
    feature_cols: list[str],
    config: RankerConfig,
    metrics: dict,
    model_dir: str | Path,
) -> Path:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    booster.save_model(str(model_dir / "ranker.lgb"))

    with open(model_dir / "feature_cols.pkl", "wb") as f:
        pickle.dump(feature_cols, f)

    with open(model_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    with open(model_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Model artifacts saved → %s", model_dir)
    return model_dir


def load_model(model_dir: str | Path) -> tuple[lgb.Booster, list[str], RankerConfig]:
    model_dir = Path(model_dir)

    booster = lgb.Booster(model_file=str(model_dir / "ranker.lgb"))

    with open(model_dir / "feature_cols.pkl", "rb") as f:
        feature_cols = pickle.load(f)

    with open(model_dir / "config.json") as f:
        cfg_dict = json.load(f)
    config = RankerConfig(**{k: v for k, v in cfg_dict.items() if k in RankerConfig.__dataclass_fields__})

    logger.info("Model loaded from %s — %d features", model_dir, len(feature_cols))
    return booster, feature_cols, config


# ─── Training Pipeline ───────────────────────────────────────────────────────

def train(
    df: pd.DataFrame,
    config: Optional[RankerConfig] = None,
    feature_cols: Optional[list[str]] = None,
) -> tuple[lgb.Booster, pd.DataFrame, dict]:
    config = config or RankerConfig()

    # Sort by group to satisfy LightGBM constraint
    df = df.sort_values(config.group_col).reset_index(drop=True)

    feature_cols = feature_cols or get_feature_cols(df)
    logger.info("Training ranker with %d features on %d rows", len(feature_cols), len(df))

    train_df, val_df = group_train_val_split(
        df, config.group_col, config.val_size, config.random_state
    )

    train_ds = build_lgb_dataset(train_df, feature_cols, config.label_col, config.group_col)
    val_ds   = build_lgb_dataset(val_df,   feature_cols, config.label_col, config.group_col,
                                  reference=train_ds)

    callbacks = [
        lgb.early_stopping(stopping_rounds=config.early_stopping_rounds, verbose=True),
        lgb.log_evaluation(period=config.verbose_eval),
    ]

    booster = lgb.train(
        params          = config.to_lgb_params(),
        train_set       = train_ds,
        num_boost_round = config.n_estimators,
        valid_sets      = [train_ds, val_ds],
        valid_names     = ["train", "val"],
        callbacks       = callbacks,
    )

    # Evaluation
    val_scores           = booster.predict(val_df[feature_cols].values.astype(np.float32))
    val_df               = val_df.copy()
    val_df["rank_score"] = val_scores
    ndcg_metrics         = evaluate_ndcg(val_df, "rank_score", config.label_col, config.group_col)

    logger.info("Validation NDCG metrics: %s", ndcg_metrics)

    # Feature importance
    fi_df = extract_feature_importance(booster, feature_cols)

    # Persist
    metrics = {
        "best_iteration": booster.best_iteration,
        "best_score":     booster.best_score,
        **ndcg_metrics,
    }
    save_model(booster, feature_cols, config, metrics, config.model_dir)

    return booster, fi_df, metrics


# ─── Inference ───────────────────────────────────────────────────────────────

def rank(
    df: pd.DataFrame,
    booster: lgb.Booster,
    feature_cols: list[str],
    group_col: str = "impression_id",
    score_col: str = "rank_score",
) -> pd.DataFrame:
    """
    Score and rank articles within each impression group.

    Returns df with `rank_score` and `rank_position` columns,
    sorted by (group, rank_position).
    """
    df = df.copy()
    df[score_col] = booster.predict(df[feature_cols].values.astype(np.float32))
    df["rank_position"] = (
        df.groupby(group_col)[score_col]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return df.sort_values([group_col, "rank_position"]).reset_index(drop=True)


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LightGBM Ranker — MIND dataset")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Train
    tr = sub.add_parser("train")
    tr.add_argument("--input",        required=True)
    tr.add_argument("--model-dir",    default="models/ranker")
    tr.add_argument("--num-leaves",   type=int,   default=63)
    tr.add_argument("--lr",           type=float, default=0.05)
    tr.add_argument("--n-estimators", type=int,   default=1000)
    tr.add_argument("--val-size",     type=float, default=0.2)
    tr.add_argument("--device",       default="cpu")

    # Predict
    pr = sub.add_parser("predict")
    pr.add_argument("--input",     required=True)
    pr.add_argument("--model-dir", default="models/ranker")
    pr.add_argument("--output",    required=True)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.cmd == "train":
        df  = pd.read_parquet(args.input)
        cfg = RankerConfig(
            num_leaves   = args.num_leaves,
            learning_rate= args.lr,
            n_estimators = args.n_estimators,
            val_size     = args.val_size,
            device       = args.device,
            model_dir    = args.model_dir,
        )
        booster, fi, metrics = train(df, cfg)
        print("\nFinal metrics:")
        print(json.dumps(metrics, indent=2, default=str))
        print("\nTop 10 features:")
        print(fi.head(10).to_string(index=False))

    elif args.cmd == "predict":
        df = pd.read_parquet(args.input)
        booster, feature_cols, config = load_model(args.model_dir)
        ranked = rank(df, booster, feature_cols, config.group_col)
        ranked.to_parquet(args.output, index=False)
        logger.info("Ranked output → %s  shape=%s", args.output, ranked.shape)
        print(ranked[["impression_id", "news_id", "rank_score", "rank_position"]].head(20))