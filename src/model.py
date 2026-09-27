"""
Phase 5: Matching Model
========================
LightGBM-based binary classifier for candidate pair scoring.

Features:
  - Hard negative sampling (5-10× negatives per positive)
  - Class weight balancing
  - Threshold sweep to maximize F_0.5 on validation set
  - Singleton handling via confidence threshold
"""

import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

logger = logging.getLogger(__name__)

# Try importing LightGBM; fall back to RandomForest
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    logger.warning("lightgbm not found; falling back to RandomForestClassifier.")
    from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score


# ---------------------------------------------------------------------------
# F_0.5 score
# ---------------------------------------------------------------------------

def f_beta_score(precision: float, recall: float, beta: float = 0.5) -> float:
    beta2 = beta ** 2
    denom = beta2 * precision + recall
    if denom == 0:
        return 0.0
    return (1 + beta2) * precision * recall / denom


def compute_f05_per_entity(
    df_preds: pd.DataFrame, threshold: float, label_col: str = "label"
) -> float:
    """
    Compute macro-averaged F_0.5 per Source 1 entity.

    df_preds must have: source1_entity_id, candidate_entity_id, score, label.
    """
    scores = []
    for s1_id, group in df_preds.groupby("source1_entity_id"):
        predicted = set(group.loc[group["score"] >= threshold, "candidate_entity_id"])
        true_pos = set(group.loc[group[label_col] == 1, "candidate_entity_id"])

        if not true_pos and not predicted:
            scores.append(1.0)  # Correctly predicted singleton
        elif not predicted:
            scores.append(0.0)  # Missed all true matches
        elif not true_pos:
            # False merges on a singleton
            p, r = 0.0, 1.0  # precision is 0 (all predictions wrong)
            scores.append(0.0)
        else:
            tp = len(predicted & true_pos)
            p = tp / len(predicted)
            r = tp / len(true_pos)
            scores.append(f_beta_score(p, r, beta=0.5))

    return float(np.mean(scores)) if scores else 0.0


def sweep_threshold(df_preds: pd.DataFrame, label_col: str = "label") -> Tuple[float, float]:
    """
    Sweep decision threshold from 0.2 to 0.95, return (best_threshold, best_f05).
    """
    best_t, best_f = 0.5, 0.0
    for t in np.arange(0.20, 0.96, 0.02):
        f = compute_f05_per_entity(df_preds, threshold=t, label_col=label_col)
        if f > best_f:
            best_f = f
            best_t = t
    logger.info(f"Best threshold: {best_t:.2f} -> F_0.5 = {best_f:.4f}")
    return round(float(best_t), 2), round(best_f, 4)


# ---------------------------------------------------------------------------
# Negative sampling
# ---------------------------------------------------------------------------

def sample_negatives(
    df_features: pd.DataFrame,
    label_col: str = "label",
    neg_ratio: int = 7,
    hard_neg_frac: float = 0.5,
    hard_neg_score_col: Optional[str] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    For each positive pair, sample `neg_ratio` negatives.
    If hard_neg_score_col is provided, half the negatives are 'hard'
    (highest-scoring non-matches from TF-IDF or feature similarity).
    """
    positives = df_features[df_features[label_col] == 1]
    negatives = df_features[df_features[label_col] == 0]

    n_pos = len(positives)
    n_neg_target = n_pos * neg_ratio

    rng = np.random.RandomState(random_state)

    if hard_neg_score_col and hard_neg_score_col in negatives.columns:
        n_hard = min(int(n_neg_target * hard_neg_frac), len(negatives))
        n_easy = min(n_neg_target - n_hard, len(negatives))

        # Hard negatives: top-scoring non-matches
        hard_neg_idx = (
            negatives.nlargest(n_hard * 2, hard_neg_score_col)
            .sample(n=min(n_hard, len(negatives)), random_state=random_state)
            .index
        )
        remaining = negatives.drop(hard_neg_idx)
        easy_neg_idx = remaining.sample(
            n=min(n_easy, len(remaining)), random_state=random_state
        ).index
        sampled_negatives = negatives.loc[list(hard_neg_idx) + list(easy_neg_idx)]
    else:
        sampled_negatives = negatives.sample(
            n=min(n_neg_target, len(negatives)), random_state=random_state
        )

    df_sampled = pd.concat([positives, sampled_negatives]).sample(
        frac=1, random_state=random_state
    ).reset_index(drop=True)

    logger.info(
        f"Training set after sampling: {len(positives)} positives, "
        f"{len(sampled_negatives)} negatives (ratio 1:{len(sampled_negatives)//max(n_pos,1)})"
    )
    return df_sampled


# ---------------------------------------------------------------------------
# Model Trainer
# ---------------------------------------------------------------------------

class MatchingModel:
    """
    Wraps LightGBM (or RandomForest fallback) for binary pair classification.
    """

    def __init__(
        self,
        feature_cols: List[str],
        neg_ratio: int = 7,
        random_state: int = 42,
        lgbm_params: Optional[Dict[str, Any]] = None,
    ):
        self.feature_cols = feature_cols
        self.neg_ratio = neg_ratio
        self.random_state = random_state
        self.threshold = 0.5  # updated by tune_threshold()
        self.model = None
        self.scaler = None

        if lgbm_params is None:
            lgbm_params = {}

        self._lgbm_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": 127,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "min_child_samples": 10,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "random_state": random_state,
            "n_jobs": -1,
            "verbose": -1,
            **lgbm_params,
        }

    def fit(
        self,
        df_train: pd.DataFrame,
        label_col: str = "label",
        df_val: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Train model on df_train (after negative sampling).
        Optionally tune threshold on df_val.
        """
        # Sample negatives
        df_sampled = sample_negatives(
            df_train,
            label_col=label_col,
            neg_ratio=self.neg_ratio,
            random_state=self.random_state,
        )

        X_train = df_sampled[self.feature_cols].fillna(0).values
        y_train = df_sampled[label_col].values

        logger.info(f"Training on {len(X_train)} pairs ({y_train.sum()} positives)...")

        if LGBM_AVAILABLE:
            # Compute scale_pos_weight for imbalance
            n_pos = y_train.sum()
            n_neg = len(y_train) - n_pos
            scale_pos_weight = n_neg / max(n_pos, 1)
            self._lgbm_params["scale_pos_weight"] = scale_pos_weight

            callbacks = [lgb.log_evaluation(period=50), lgb.early_stopping(50, verbose=False)]

            if df_val is not None and label_col in df_val.columns:
                X_val = df_val[self.feature_cols].fillna(0).values
                y_val = df_val[label_col].values
                self.model = lgb.LGBMClassifier(**self._lgbm_params)
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=callbacks,
                )
            else:
                self.model = lgb.LGBMClassifier(**self._lgbm_params)
                self.model.fit(X_train, y_train)
        else:
            # RandomForest fallback
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1,
            )
            self.model.fit(X_train, y_train)

        logger.info("Model training complete.")

        # Tune threshold on validation set if provided
        if df_val is not None and label_col in df_val.columns:
            self.tune_threshold(df_val, label_col=label_col)

    def predict_scores(self, df: pd.DataFrame) -> np.ndarray:
        """Return predicted match probabilities for all pairs in df."""
        X = df[self.feature_cols].fillna(0).values
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X).astype(float)

    def tune_threshold(self, df_val: pd.DataFrame, label_col: str = "label") -> float:
        """
        Sweep thresholds on validation set to maximize macro-averaged F_0.5.
        Updates self.threshold in place.
        """
        scores = self.predict_scores(df_val)
        df_val = df_val.copy()
        df_val["score"] = scores

        best_t, best_f = sweep_threshold(df_val, label_col=label_col)
        self.threshold = best_t
        logger.info(f"Threshold set to {self.threshold} (val F_0.5 = {best_f:.4f})")
        return best_f

    def predict_matches(
        self,
        df_features: pd.DataFrame,
        threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        For each S1 entity, return predicted matched IDs above threshold.

        Returns DataFrame: source1_entity_id, matched_entity_ids (comma-separated or empty).
        """
        t = threshold if threshold is not None else self.threshold
        scores = self.predict_scores(df_features)
        df_out = df_features[["source1_entity_id", "candidate_entity_id"]].copy()
        df_out["score"] = scores

        results = []
        for s1_id, group in df_out.groupby("source1_entity_id"):
            matched = group.loc[group["score"] >= t, "candidate_entity_id"].tolist()
            results.append({
                "source1_entity_id": s1_id,
                "matched_entity_ids": ",".join(sorted(set(matched))),
            })

        return pd.DataFrame(results)

    def get_feature_importance(self) -> Optional[pd.Series]:
        """Return feature importances from the trained model."""
        if self.model is None:
            return None
        if LGBM_AVAILABLE and isinstance(self.model, lgb.LGBMClassifier):
            return pd.Series(
                self.model.feature_importances_,
                index=self.feature_cols,
            ).sort_values(ascending=False)
        if hasattr(self.model, "feature_importances_"):
            return pd.Series(
                self.model.feature_importances_,
                index=self.feature_cols,
            ).sort_values(ascending=False)
        return None

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "MatchingModel":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"Model loaded from {path}")
        return obj


# ---------------------------------------------------------------------------
# Phase 6: Singleton handling
# ---------------------------------------------------------------------------

def apply_singleton_handling(
    df_matches: pd.DataFrame,
    df_features: pd.DataFrame,
    singleton_max_score: float = 0.35,
) -> pd.DataFrame:
    """
    For each S1 entity predicted to have matches, check if the top
    candidate score is above singleton_max_score. If below, call it a singleton.

    This adds an extra layer of precision protection.
    """
    # Compute max score per S1 entity
    if "score" not in df_features.columns:
        return df_matches  # Can't apply without scores

    max_scores = (
        df_features.groupby("source1_entity_id")["score"].max().reset_index()
        .rename(columns={"score": "max_score"})
    )

    df_out = df_matches.merge(max_scores, on="source1_entity_id", how="left")

    # If max_score is below singleton threshold → clear the prediction
    mask_low_conf = df_out["max_score"] < singleton_max_score
    df_out.loc[mask_low_conf, "matched_entity_ids"] = ""
    n_cleared = mask_low_conf.sum()
    logger.info(
        f"Singleton handling: cleared {n_cleared} low-confidence predictions "
        f"(max_score < {singleton_max_score})"
    )

    return df_out[["source1_entity_id", "matched_entity_ids"]]
