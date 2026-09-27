"""
Main Pipeline: End-to-End Business Entity Resolution
======================================================

Usage:
  python pipeline.py --mode train   # Train model on train data
  python pipeline.py --mode predict # Generate predictions on test data
  python pipeline.py --mode full    # Train + predict in one shot

Outputs:
  output/matching_results.tsv
  output/candidate_pairs.tsv
"""

import argparse
import logging
import os
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data import (
    load_all_train_data,
    load_all_test_data,
    profile_dataset,
    print_profile,
    inspect_matched_pairs,
    entity_level_train_val_split,
    attach_labels,
    evaluate_per_country,
)
from src.normalize import normalize_record
from src.blocking import CandidateBlocker
from src.features import compute_features_for_candidates, FEATURE_COLUMNS
from src.model import MatchingModel, apply_singleton_handling, compute_f05_per_entity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Helper: normalize a full DataFrame of records
# ---------------------------------------------------------------------------

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply normalize_record to every row in df. Returns df with norm columns appended."""
    norm_rows = []
    for _, row in df.iterrows():
        norm = normalize_record(
            business_name=row.get("business_name", ""),
            business_address=row.get("business_address", ""),
            country=row.get("country", ""),
        )
        norm["entity_id"] = row["entity_id"]
        norm_rows.append(norm)

    df_norm = pd.DataFrame(norm_rows)
    return df_norm


# ---------------------------------------------------------------------------
# Helper: save candidate pairs TSV
# ---------------------------------------------------------------------------

def save_candidate_pairs(df_cands: pd.DataFrame, path: str) -> None:
    """
    Save candidate_pairs.tsv.
    Format: source1_entity_id \t candidate_entity_ids (comma-separated)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = []
    for _, row in df_cands.iterrows():
        cand_ids = row.get("candidate_entity_ids") or set()
        if isinstance(cand_ids, (set, list)):
            cand_ids_str = ",".join(sorted(cand_ids))
        else:
            cand_ids_str = str(cand_ids)
        rows.append({
            "source1_entity_id": row["source1_entity_id"],
            "candidate_entity_ids": cand_ids_str,
        })
    df_out = pd.DataFrame(rows)
    df_out.to_csv(path, sep="\t", index=False)
    logger.info(f"Candidate pairs saved to {path}")


# ---------------------------------------------------------------------------
# Helper: save matching results TSV
# ---------------------------------------------------------------------------

def save_matching_results(df_matches: pd.DataFrame, path: str) -> None:
    """
    Save matching_results.tsv.
    Format: source1_entity_id \t matched_entity_ids (comma-separated or empty)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_out = df_matches[["source1_entity_id", "matched_entity_ids"]].copy()
    df_out.to_csv(path, sep="\t", index=False)
    logger.info(f"Matching results saved to {path}")


# ---------------------------------------------------------------------------
# Ensure all S1 entities are in output (required by contest rules)
# ---------------------------------------------------------------------------

def ensure_all_s1_entities(
    df_matches: pd.DataFrame,
    all_s1_ids: pd.Series,
) -> pd.DataFrame:
    """
    Contest requires every S1 entity to appear in output.
    Fill in empty rows for any missing S1 IDs.
    """
    existing_ids = set(df_matches["source1_entity_id"])
    missing = [eid for eid in all_s1_ids if eid not in existing_ids]
    if missing:
        logger.warning(f"Adding {len(missing)} missing S1 entities with empty predictions.")
        new_rows = pd.DataFrame({
            "source1_entity_id": missing,
            "matched_entity_ids": [""] * len(missing),
        })
        df_matches = pd.concat([df_matches, new_rows], ignore_index=True)
    return df_matches


# ---------------------------------------------------------------------------
# Core training pipeline
# ---------------------------------------------------------------------------

def run_train(
    train_dir: str = "dataset/train",
    model_save_path: str = "output/model.pkl",
    val_frac: float = 0.20,
    run_eda: bool = True,
) -> MatchingModel:
    """
    Full training pipeline:
    1. Load data
    2. EDA
    3. Normalize
    4. Block
    5. Feature engineering
    6. Train model + tune threshold
    7. Save model
    """
    t0 = time.time()

    # -----------------------------------------------------------------------
    # Phase 1: Load data
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Phase 1: Loading training data...")
    logger.info("=" * 60)
    df_s1, df_s2, df_s3, df_gt = load_all_train_data(train_dir)

    # Combine S2 + S3 for blocking/feature purposes
    df_s23 = pd.concat([df_s2, df_s3], ignore_index=True)

    # EDA
    if run_eda:
        profile = profile_dataset(df_s1, df_s2, df_s3, df_gt)
        print_profile(profile)

        # Show sample matched pairs for manual inspection
        logger.info("Sample matched pairs for inspection:")
        sample_pairs = inspect_matched_pairs(df_s1, df_s23, df_gt, n=5)
        for _, row in sample_pairs.iterrows():
            logger.info(f"  S1: {row['s1_name']} | Match: {row['match_name']}")
            logger.info(f"      {row['s1_addr']} | {row['match_addr']}")

    # -----------------------------------------------------------------------
    # Train/Val entity split
    # -----------------------------------------------------------------------
    logger.info("Splitting train/val at entity level...")
    train_ids, val_ids = entity_level_train_val_split(df_s1, val_frac=val_frac)
    df_s1_train = df_s1[df_s1["entity_id"].isin(set(train_ids))].reset_index(drop=True)
    df_s1_val = df_s1[df_s1["entity_id"].isin(set(val_ids))].reset_index(drop=True)
    df_gt_train = df_gt[df_gt["source1_entity_id"].isin(set(train_ids))].reset_index(drop=True)
    df_gt_val = df_gt[df_gt["source1_entity_id"].isin(set(val_ids))].reset_index(drop=True)

    # -----------------------------------------------------------------------
    # Phase 2: Normalize
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Phase 2: Normalizing records...")
    logger.info("=" * 60)
    df_s1_norm = normalize_df(df_s1)
    df_s23_norm = normalize_df(df_s23)

    df_s1_train_norm = df_s1_norm[df_s1_norm["entity_id"].isin(set(train_ids))]
    df_s1_val_norm = df_s1_norm[df_s1_norm["entity_id"].isin(set(val_ids))]

    # -----------------------------------------------------------------------
    # Phase 3: Blocking
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Phase 3: Building blocking indices...")
    logger.info("=" * 60)
    blocker = CandidateBlocker(tfidf_top_k=15)
    blocker.fit(df_s23_norm)

    logger.info("Generating candidates for TRAIN S1 entities...")
    df_cands_train = blocker.generate_candidates(df_s1_train_norm)

    logger.info("Generating candidates for VAL S1 entities...")
    df_cands_val = blocker.generate_candidates(df_s1_val_norm)

    # Evaluate blocking recall
    def _gt_to_blocking_format(df_gt_sub):
        return pd.DataFrame({
            "source1_entity_id": df_gt_sub["source1_entity_id"],
            "matched_entity_ids": df_gt_sub["matched_entity_ids"],
        })

    train_recall = blocker.evaluate_blocking_recall(
        df_cands_train, _gt_to_blocking_format(df_gt_train)
    )
    val_recall = blocker.evaluate_blocking_recall(
        df_cands_val, _gt_to_blocking_format(df_gt_val)
    )
    logger.info(f"Blocking recall — Train: {train_recall:.4f}, Val: {val_recall:.4f}")
    if val_recall < 0.90:
        logger.warning(
            f"⚠ Blocking recall {val_recall:.4f} < 0.90! "
            "Consider expanding blocking strategies before proceeding."
        )

    # -----------------------------------------------------------------------
    # Phase 4: Feature Engineering
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Phase 4: Computing pair features for TRAIN...")
    logger.info("=" * 60)
    df_feat_train = compute_features_for_candidates(
        df_s1_train_norm, df_s23_norm, df_cands_train
    )
    df_feat_train = attach_labels(df_feat_train, df_gt_train)

    logger.info("Computing pair features for VAL...")
    df_feat_val = compute_features_for_candidates(
        df_s1_val_norm, df_s23_norm, df_cands_val
    )
    df_feat_val = attach_labels(df_feat_val, df_gt_val)

    logger.info(f"Train features: {df_feat_train.shape}, Val features: {df_feat_val.shape}")
    logger.info(
        f"Train positives: {df_feat_train['label'].sum()}, "
        f"Val positives: {df_feat_val['label'].sum()}"
    )

    # -----------------------------------------------------------------------
    # Phase 5: Train matching model
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Phase 5: Training matching model...")
    logger.info("=" * 60)
    model = MatchingModel(
        feature_cols=FEATURE_COLUMNS,
        neg_ratio=7,
        random_state=42,
    )
    model.fit(df_feat_train, label_col="label", df_val=df_feat_val)

    # Feature importance
    importance = model.get_feature_importance()
    if importance is not None:
        logger.info("Top 10 features by importance:")
        for feat, imp in importance.head(10).items():
            logger.info(f"  {feat}: {imp:.4f}")

    # -----------------------------------------------------------------------
    # Validation evaluation
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Phase 8: Validation evaluation...")
    logger.info("=" * 60)
    val_scores = model.predict_scores(df_feat_val)
    df_feat_val = df_feat_val.copy()
    df_feat_val["score"] = val_scores

    val_f05 = compute_f05_per_entity(df_feat_val, threshold=model.threshold)
    logger.info(f"Validation F_0.5 @ threshold={model.threshold}: {val_f05:.4f}")

    # Per-country breakdown
    logger.info("Per-country F_0.5:")
    evaluate_per_country(df_feat_val, df_s1_val, threshold=model.threshold)

    # -----------------------------------------------------------------------
    # Save model
    # -----------------------------------------------------------------------
    model.save(model_save_path)
    logger.info(f"Training complete in {time.time() - t0:.1f}s")
    return model


# ---------------------------------------------------------------------------
# Core prediction pipeline
# ---------------------------------------------------------------------------

def run_predict(
    test_dir: str = "dataset/test",
    model_path: str = "output/model.pkl",
    output_dir: str = "output",
    singleton_threshold: float = 0.35,
) -> None:
    """
    Load test data, generate candidates, compute features, predict matches.
    Saves output/matching_results.tsv and output/candidate_pairs.tsv.
    """
    t0 = time.time()

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    logger.info("Loading trained model...")
    model = MatchingModel.load(model_path)

    # -----------------------------------------------------------------------
    # Load test data
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Loading test data...")
    logger.info("=" * 60)
    df_s1, df_s2, df_s3 = load_all_test_data(test_dir)
    df_s23 = pd.concat([df_s2, df_s3], ignore_index=True)

    # -----------------------------------------------------------------------
    # Normalize
    # -----------------------------------------------------------------------
    logger.info("Normalizing test records...")
    df_s1_norm = normalize_df(df_s1)
    df_s23_norm = normalize_df(df_s23)

    # -----------------------------------------------------------------------
    # Block
    # -----------------------------------------------------------------------
    logger.info("Building blocking indices for test...")
    blocker = CandidateBlocker(tfidf_top_k=15)
    blocker.fit(df_s23_norm)

    logger.info("Generating test candidates...")
    df_cands = blocker.generate_candidates(df_s1_norm)

    # -----------------------------------------------------------------------
    # Save candidate pairs (required output)
    # -----------------------------------------------------------------------
    cands_path = os.path.join(output_dir, "candidate_pairs.tsv")
    save_candidate_pairs(df_cands, cands_path)

    # -----------------------------------------------------------------------
    # Feature Engineering
    # -----------------------------------------------------------------------
    logger.info("Computing pair features for test...")
    df_feat = compute_features_for_candidates(df_s1_norm, df_s23_norm, df_cands)

    # -----------------------------------------------------------------------
    # Predict
    # -----------------------------------------------------------------------
    logger.info(f"Predicting matches at threshold={model.threshold}...")
    if len(df_feat) > 0:
        df_feat["score"] = model.predict_scores(df_feat)
        df_matches = model.predict_matches(df_feat)

        # Phase 6: Singleton handling
        df_matches = apply_singleton_handling(
            df_matches, df_feat, singleton_max_score=singleton_threshold
        )
    else:
        logger.warning("No candidate pairs generated! All predictions will be empty.")
        df_matches = pd.DataFrame({
            "source1_entity_id": df_s1["entity_id"],
            "matched_entity_ids": [""] * len(df_s1),
        })

    # Ensure all S1 entities are present (contest requirement)
    df_matches = ensure_all_s1_entities(df_matches, df_s1["entity_id"])

    # -----------------------------------------------------------------------
    # Save matching results
    # -----------------------------------------------------------------------
    results_path = os.path.join(output_dir, "matching_results.tsv")
    save_matching_results(df_matches, results_path)

    # Summary stats
    n_matched = (df_matches["matched_entity_ids"] != "").sum()
    n_singleton = (df_matches["matched_entity_ids"] == "").sum()
    logger.info(f"Predictions: {n_matched} entities with matches, {n_singleton} singletons")
    logger.info(f"Prediction complete in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Business Entity Resolution Pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "predict", "full"],
        default="full",
        help="Pipeline mode: train, predict, or full (train+predict).",
    )
    parser.add_argument(
        "--train-dir",
        default="dataset/train",
        help="Directory containing train_source*.tsv and train_labels.tsv",
    )
    parser.add_argument(
        "--test-dir",
        default="dataset/test",
        help="Directory containing test_source*.tsv",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory to write output TSV files",
    )
    parser.add_argument(
        "--model-path",
        default="output/model.pkl",
        help="Where to save/load the trained model",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.20,
        help="Fraction of S1 entities held out for validation",
    )
    parser.add_argument(
        "--singleton-threshold",
        type=float,
        default=0.35,
        help="Max score below which a predicted match is cleared (singleton handling)",
    )
    parser.add_argument(
        "--no-eda",
        action="store_true",
        help="Skip EDA / data profiling",
    )

    args = parser.parse_args()

    if args.mode in ("train", "full"):
        run_train(
            train_dir=args.train_dir,
            model_save_path=args.model_path,
            val_frac=args.val_frac,
            run_eda=not args.no_eda,
        )

    if args.mode in ("predict", "full"):
        run_predict(
            test_dir=args.test_dir,
            model_path=args.model_path,
            output_dir=args.output_dir,
            singleton_threshold=args.singleton_threshold,
        )


if __name__ == "__main__":
    main()
