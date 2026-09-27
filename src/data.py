"""
Phase 1: Data Loading, EDA & Validation Framework
===================================================
Handles:
  - Loading train/test TSV files
  - Data profiling (distributions, missing values, match stats)
  - Ground truth parsing for labeled data
  - Train/val split at entity level (no pair-level leakage)
"""

import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_source_file(path: str, source_id: Optional[str] = None) -> pd.DataFrame:
    """
    Load a source TSV file.
    Handles missing sep=\t silently with validation.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Source file not found: {path}")

    df = pd.read_csv(path, sep="\t", dtype=str)

    # Validate expected columns
    expected_cols = {"entity_id", "business_name", "business_address", "country"}
    if not expected_cols.issubset(set(df.columns)):
        # Check if it loaded as single column (missing sep)
        if len(df.columns) == 1:
            raise ValueError(
                f"File '{path}' loaded as single column — "
                "make sure it is tab-separated (sep='\\t')."
            )
        missing = expected_cols - set(df.columns)
        logger.warning(f"Missing columns in {path}: {missing}")

    df = df.fillna("")
    if source_id:
        df["source"] = source_id

    logger.info(f"Loaded {len(df)} records from {path}")
    return df


def load_ground_truth(path: str) -> pd.DataFrame:
    """
    Load training ground truth / labels file.
    Expected format: source1_entity_id \t matched_entity_ids
    where matched_entity_ids is comma-separated (or empty for singletons).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Ground truth file not found: {path}")

    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")

    # Normalize column names
    if "matched_entity_ids" in df.columns:
        pass
    elif "match_ids" in df.columns:
        df = df.rename(columns={"match_ids": "matched_entity_ids"})

    # Parse matched_entity_ids into list
    df["matched_entity_ids"] = df["matched_entity_ids"].apply(
        lambda x: [m.strip() for m in x.split(",") if m.strip()] if x.strip() else []
    )

    logger.info(f"Loaded ground truth: {len(df)} S1 entities, "
                f"{df['matched_entity_ids'].apply(len).sum()} total match pairs.")
    return df


def load_all_train_data(
    train_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all train source files and ground truth.
    Returns: (df_s1, df_s2, df_s3, df_gt)
    """
    df_s1 = load_source_file(os.path.join(train_dir, "train_source1.tsv"), "S1")
    df_s2 = load_source_file(os.path.join(train_dir, "train_source2.tsv"), "S2")
    df_s3 = load_source_file(os.path.join(train_dir, "train_source3.tsv"), "S3")
    # Support both file names: the contest uses train_ground_truth.tsv
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")
    if not os.path.exists(gt_path):
        gt_path = os.path.join(train_dir, "train_labels.tsv")
    df_gt = load_ground_truth(gt_path)
    return df_s1, df_s2, df_s3, df_gt


def load_all_test_data(
    test_dir: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all test source files.
    Returns: (df_s1, df_s2, df_s3)
    """
    df_s1 = load_source_file(os.path.join(test_dir, "test_source1.tsv"), "S1")
    df_s2 = load_source_file(os.path.join(test_dir, "test_source2.tsv"), "S2")
    df_s3 = load_source_file(os.path.join(test_dir, "test_source3.tsv"), "S3")
    return df_s1, df_s2, df_s3


# ---------------------------------------------------------------------------
# EDA / Profiling
# ---------------------------------------------------------------------------

def profile_dataset(
    df_s1: pd.DataFrame,
    df_s2: pd.DataFrame,
    df_s3: pd.DataFrame,
    df_gt: Optional[pd.DataFrame] = None,
) -> Dict:
    """
    Comprehensive EDA profiling.
    Returns a dict of statistics useful for understanding the data.
    """
    profile = {}

    # Record counts
    profile["n_s1"] = len(df_s1)
    profile["n_s2"] = len(df_s2)
    profile["n_s3"] = len(df_s3)
    profile["n_total"] = len(df_s1) + len(df_s2) + len(df_s3)

    # Country distributions
    for name, df in [("s1", df_s1), ("s2", df_s2), ("s3", df_s3)]:
        if "country" in df.columns:
            profile[f"country_dist_{name}"] = df["country"].value_counts().to_dict()

    # Missing value rates
    for name, df in [("s1", df_s1), ("s2", df_s2), ("s3", df_s3)]:
        for col in ["business_name", "business_address", "country"]:
            if col in df.columns:
                missing_rate = (df[col] == "").mean()
                profile[f"missing_{col}_{name}"] = round(missing_rate, 4)

    # Name/address lengths
    for name, df in [("s1", df_s1), ("s2", df_s2), ("s3", df_s3)]:
        if "business_name" in df.columns:
            lens = df["business_name"].str.len()
            profile[f"name_len_mean_{name}"] = round(lens.mean(), 1)
            profile[f"name_len_median_{name}"] = round(lens.median(), 1)
        if "business_address" in df.columns:
            lens = df["business_address"].str.len()
            profile[f"addr_len_mean_{name}"] = round(lens.mean(), 1)

    # Ground truth analysis
    if df_gt is not None:
        match_counts = df_gt["matched_entity_ids"].apply(len)
        profile["gt_singletons"] = int((match_counts == 0).sum())
        profile["gt_singleton_frac"] = round((match_counts == 0).mean(), 4)
        profile["gt_total_matches"] = int(match_counts.sum())
        profile["gt_matches_per_s1_mean"] = round(match_counts.mean(), 3)
        profile["gt_match_dist"] = match_counts.value_counts().sort_index().to_dict()

    return profile


def print_profile(profile: Dict) -> None:
    """Pretty-print EDA profile."""
    print("\n" + "="*60)
    print("DATASET PROFILE")
    print("="*60)
    print(f"Records: S1={profile['n_s1']}, S2={profile['n_s2']}, S3={profile['n_s3']}")

    for source in ["s1", "s2", "s3"]:
        key = f"country_dist_{source}"
        if key in profile:
            print(f"Country dist {source.upper()}: {profile[key]}")

    print("\nMissing value rates:")
    for key, val in profile.items():
        if key.startswith("missing_"):
            print(f"  {key}: {val:.2%}")

    print("\nName lengths:")
    for source in ["s1", "s2", "s3"]:
        k = f"name_len_mean_{source}"
        if k in profile:
            print(f"  {source.upper()} mean={profile[k]:.1f}, median={profile.get(f'name_len_median_{source}', 'N/A')}")

    if "gt_singletons" in profile:
        print(f"\nGround truth:")
        print(f"  Singletons: {profile['gt_singletons']} ({profile['gt_singleton_frac']:.2%})")
        print(f"  Total matches: {profile['gt_total_matches']}")
        print(f"  Avg matches/S1: {profile['gt_matches_per_s1_mean']:.3f}")
        print(f"  Match distribution: {profile['gt_match_dist']}")
    print("="*60 + "\n")


def inspect_matched_pairs(
    df_s1: pd.DataFrame,
    df_s23: pd.DataFrame,
    df_gt: pd.DataFrame,
    n: int = 20,
) -> pd.DataFrame:
    """
    Show n random matched pairs side-by-side for manual inspection.
    Helps understand what kinds of transformations link matched entities.
    """
    s1_map = df_s1.set_index("entity_id")[["business_name", "business_address", "country"]].to_dict("index")
    s23_map = df_s23.set_index("entity_id")[["business_name", "business_address", "country"]].to_dict("index")

    pairs = []
    for _, row in df_gt.iterrows():
        s1_id = row["source1_entity_id"]
        for match_id in row["matched_entity_ids"]:
            pairs.append((s1_id, match_id))

    import random
    random.shuffle(pairs)
    pairs = pairs[:n]

    rows = []
    for s1_id, m_id in pairs:
        s1 = s1_map.get(s1_id, {})
        s23 = s23_map.get(m_id, {})
        rows.append({
            "s1_id": s1_id,
            "match_id": m_id,
            "s1_name": s1.get("business_name", ""),
            "match_name": s23.get("business_name", ""),
            "s1_addr": s1.get("business_address", ""),
            "match_addr": s23.get("business_address", ""),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Phase 8: Validation framework
# ---------------------------------------------------------------------------

def entity_level_train_val_split(
    df_s1: pd.DataFrame,
    val_frac: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.Series, pd.Series]:
    """
    Split S1 entity IDs into train/val at the entity level.
    Returns (train_ids, val_ids).
    """
    entity_ids = df_s1["entity_id"].unique()
    n_val = max(1, int(len(entity_ids) * val_frac))
    rng = np.random.RandomState(random_state)
    val_ids = rng.choice(entity_ids, size=n_val, replace=False)
    train_ids = np.array([eid for eid in entity_ids if eid not in set(val_ids)])
    logger.info(
        f"Train/val split: {len(train_ids)} train entities, {len(val_ids)} val entities"
    )
    return pd.Series(train_ids), pd.Series(val_ids)


def attach_labels(
    df_features: pd.DataFrame,
    df_gt: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add a 'label' column (1 = true match, 0 = non-match) to df_features.
    """
    # Build a set of (s1_id, match_id) positive pairs
    positive_pairs = set()
    for _, row in df_gt.iterrows():
        s1_id = row["source1_entity_id"]
        for mid in row["matched_entity_ids"]:
            positive_pairs.add((s1_id, mid))

    df_features = df_features.copy()
    df_features["label"] = df_features.apply(
        lambda r: 1 if (r["source1_entity_id"], r["candidate_entity_id"]) in positive_pairs else 0,
        axis=1,
    )
    pos_count = df_features["label"].sum()
    logger.info(
        f"Labels attached: {pos_count} positives, "
        f"{len(df_features) - pos_count} negatives "
        f"(ratio 1:{(len(df_features)-pos_count)//max(pos_count,1)})"
    )
    return df_features


def evaluate_per_country(
    df_preds: pd.DataFrame,
    df_s1: pd.DataFrame,
    threshold: float,
    label_col: str = "label",
) -> Dict[str, float]:
    """
    Compute F_0.5 per country breakdown.
    df_preds must have source1_entity_id, candidate_entity_id, score, label.
    df_s1 must have entity_id, country.
    """
    from src.model import compute_f05_per_entity

    country_map = df_s1.set_index("entity_id")["country"].to_dict()
    df_preds = df_preds.copy()
    df_preds["country"] = df_preds["source1_entity_id"].map(country_map)

    results = {}
    for country, group in df_preds.groupby("country"):
        f05 = compute_f05_per_entity(group, threshold=threshold, label_col=label_col)
        results[country] = round(f05, 4)
        logger.info(f"  {country}: F_0.5 = {f05:.4f}")

    return results
