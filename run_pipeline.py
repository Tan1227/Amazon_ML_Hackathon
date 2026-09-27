"""
Entity Resolution Pipeline v3 — Fixed Blocking
=================================================
Critical fixes:
  1. IDF-filtered token blocking (skip tokens appearing in >10K records)
  2. ADDRESS-based blocking (critical for Hindi↔English name mismatches)
  3. Ranked candidate selection (score by overlap count, not random cap)
  4. Bigram blocking on names
  5. ZIP/PIN blocking strengthened
  6. Max candidates raised to 200

The core insight: for India, S2 names are often in Hindi script while S1 names
are in English. The ADDRESSES are usually romanized in both sources, so address
tokens become the primary blocking signal for India.
"""

import argparse
import gc
import logging
import os
import pickle
import re
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.features import (
    FEATURE_COLUMNS,
    jaro_winkler,
    levenshtein_ratio,
    jaccard_tokens,
    jaccard_char_ngrams,
    token_set_ratio,
)
from src.model import (
    MatchingModel,
    apply_singleton_handling,
    compute_f05_per_entity,
    sweep_threshold,
    f_beta_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log", encoding="utf-8", mode="w"),
    ],
)
logger = logging.getLogger("pipeline")

DATA_BASE = r"dataset\6ab10eb3b23ba_student_resource\student_resource\dataset"


# ===========================================================================
# Normalization (vectorized)
# ===========================================================================

LEGAL_STOPS = {
    "inc", "incorporated", "corp", "corporation", "ltd", "limited",
    "llc", "llp", "pvt", "private", "co", "company",
    "the", "a", "an", "of", "and", "in", "at", "for", "to", "by", "with",
    "enterprises", "enterprise", "traders", "trader", "associates",
    "industries", "industry", "solutions", "services", "service",
    "technologies", "technology", "group", "holdings", "international",
}

def norm_name(s: pd.Series) -> pd.Series:
    s = s.fillna("").str.lower().str.strip()
    s = s.str.replace(r'&', ' and ', regex=False)
    s = s.str.replace(r'[.,\-\'\"()\[\]{}|\\/:;!?@#$%^*_+=<>`~]', ' ', regex=True)
    s = s.str.replace(r'\s+', ' ', regex=True).str.strip()
    return s

def norm_addr(s: pd.Series) -> pd.Series:
    s = s.fillna("").str.lower().str.strip()
    s = s.str.replace(r'&', ' and ', regex=False)
    s = s.str.replace(r'[.,\-\'\"()\[\]{}|\\/:;!?@#$%^*_+=<>`~]', ' ', regex=True)
    s = s.str.replace(r'\s+', ' ', regex=True).str.strip()
    return s

COUNTRY_MAP = {
    "us": "US", "usa": "US", "united states": "US",
    "india": "IN", "in": "IN", "ind": "IN",
    "france": "FR", "fr": "FR", "fra": "FR",
}

def norm_country(s: pd.Series) -> pd.Series:
    return s.fillna("").str.lower().str.strip().map(COUNTRY_MAP).fillna("OTHER")

ZIP_US = re.compile(r'\b(\d{5})(?:-\d{4})?\b')
ZIP_IN = re.compile(r'\b(\d{6})\b')
ZIP_FR = re.compile(r'\b(\d{5})\b')

def extract_zip(addr: str, country: str) -> str:
    if not addr:
        return ""
    if country == "US":
        m = ZIP_US.search(addr)
        return m.group(1) if m else ""
    elif country == "IN":
        m = ZIP_IN.search(addr)
        return m.group(1) if m else ""
    elif country == "FR":
        m = ZIP_FR.search(addr)
        return m.group(1) if m else ""
    return ""

def get_core_tokens(name: str) -> List[str]:
    """Get name tokens with legal suffixes / stopwords removed."""
    return [t for t in name.split() if t not in LEGAL_STOPS and len(t) >= 2]

def get_addr_tokens(addr: str) -> List[str]:
    """Get meaningful address tokens (skip very short and purely numeric)."""
    tokens = []
    for t in addr.split():
        if len(t) < 2:
            continue
        # Keep alphanumeric tokens that are at least somewhat meaningful
        if len(t) >= 3:
            tokens.append(t)
    return tokens

def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["name_norm"] = norm_name(df["business_name"])
    df["addr_norm"] = norm_addr(df["business_address"])
    df["country_norm"] = norm_country(df["country"])
    df["name_sorted"] = df["name_norm"].apply(lambda x: " ".join(sorted(x.split())) if x else "")
    df["zip_pin"] = df.apply(lambda r: extract_zip(r["addr_norm"], r["country_norm"]), axis=1)
    df["core_tokens"] = df["name_norm"].apply(get_core_tokens)
    df["addr_key_tokens"] = df["addr_norm"].apply(get_addr_tokens)
    return df


# ===========================================================================
# IDF-aware Inverted Index Blocking
# ===========================================================================

class SmartBlocker:
    """
    Builds multiple inverted indices with IDF filtering.
    Scores candidates by number of matching signals, then caps.
    """

    def __init__(
        self,
        max_token_df: int = 50_000,
        max_candidates: int = 40,
        max_bigram_df: int = 10_000,
        max_prefix_df: int = 5_000,
    ):
        self.max_token_df = max_token_df
        self.max_candidates = max_candidates
        self.max_bigram_df = max_bigram_df
        self.max_prefix_df = max_prefix_df

        self.name_token_idx: Dict[str, List[int]] = {}
        self.addr_token_idx: Dict[str, List[int]] = {}
        self.bigram_idx: Dict[str, List[int]] = {}
        self.zip_idx: Dict[str, List[int]] = {}
        self.prefix_idx: Dict[str, List[int]] = {}

        # Document frequencies for IDF filtering
        self.name_token_df: Dict[str, int] = {}
        self.addr_token_df: Dict[str, int] = {}

    def fit(self, df_s23: pd.DataFrame):
        """Build all indices from S2/S3 data."""
        logger.info(f"  Building smart blocking indices over {len(df_s23):,} records...")
        t0 = time.time()

        # First pass: count document frequencies
        name_df = Counter()
        addr_df = Counter()

        for i, row in enumerate(df_s23.itertuples()):
            core_toks = row.core_tokens if isinstance(row.core_tokens, list) else []
            for tok in set(core_toks):  # unique per doc
                name_df[tok] += 1

            addr_toks = row.addr_key_tokens if isinstance(row.addr_key_tokens, list) else []
            for tok in set(addr_toks):
                addr_df[tok] += 1

        self.name_token_df = dict(name_df)
        self.addr_token_df = dict(addr_df)

        # Count how many tokens pass IDF filter
        n_name_pass = sum(1 for v in name_df.values() if v <= self.max_token_df)
        n_name_skip = sum(1 for v in name_df.values() if v > self.max_token_df)
        n_addr_pass = sum(1 for v in addr_df.values() if v <= self.max_token_df)
        logger.info(
            f"  IDF filter: {n_name_pass:,} name tokens pass (skip {n_name_skip:,}), "
            f"{n_addr_pass:,} addr tokens pass"
        )

        # Second pass: build indices (only for tokens below max_df)
        name_idx = defaultdict(list)
        addr_idx = defaultdict(list)
        bigram_idx_d = defaultdict(list)
        zip_idx_d = defaultdict(list)
        prefix_idx_d = defaultdict(list)

        for i, row in enumerate(df_s23.itertuples()):
            # Name tokens (IDF filtered)
            core_toks = row.core_tokens if isinstance(row.core_tokens, list) else []
            for tok in core_toks:
                if name_df.get(tok, 0) <= self.max_token_df:
                    name_idx[tok].append(i)

            # Name bigrams (IDF filtered)
            if len(core_toks) >= 2:
                for j in range(len(core_toks) - 1):
                    bg = f"{core_toks[j]}_{core_toks[j+1]}"
                    bigram_idx_d[bg].append(i)

            # Address tokens (IDF filtered)
            addr_toks = row.addr_key_tokens if isinstance(row.addr_key_tokens, list) else []
            for tok in addr_toks:
                if addr_df.get(tok, 0) <= self.max_token_df:
                    addr_idx[tok].append(i)

            # ZIP/PIN
            zp = row.zip_pin
            if zp:
                zip_idx_d[zp].append(i)

            # Prefix (first 4 chars of sorted name without spaces)
            ns = row.name_sorted if isinstance(row.name_sorted, str) else ""
            stripped = ns.replace(" ", "")
            if len(stripped) >= 4:
                prefix_idx_d[stripped[:4]].append(i)

            if (i + 1) % 1_000_000 == 0:
                logger.info(f"    Indexed {i+1:,} records...")

        self.name_token_idx = dict(name_idx)
        self.addr_token_idx = dict(addr_idx)
        self.bigram_idx = {k: v for k, v in bigram_idx_d.items() if len(v) <= self.max_bigram_df}
        self.zip_idx = dict(zip_idx_d)
        self.prefix_idx = {k: v for k, v in prefix_idx_d.items() if len(v) <= self.max_prefix_df}

        logger.info(
            f"  Indices built in {time.time()-t0:.1f}s: "
            f"name={len(self.name_token_idx):,}, addr={len(self.addr_token_idx):,}, "
            f"bigram={len(self.bigram_idx):,}, zip={len(self.zip_idx):,}, "
            f"prefix={len(self.prefix_idx):,}"
        )

    def get_candidates(
        self,
        core_tokens: List[str],
        addr_tokens: List[str],
        name_sorted: str,
        zip_pin: str,
    ) -> List[int]:
        """
        Get ranked candidate indices for one S1 entity.
        Scores each candidate by # of matching blocking signals.
        Returns top-K by score.
        """
        # Count hits per candidate
        scores = Counter()

        # Strategy 1: Name token overlap (IDF-filtered, weight=2)
        for tok in core_tokens:
            hits = self.name_token_idx.get(tok)
            if hits:
                for idx in hits:
                    scores[idx] += 2

        # Strategy 2: Address token overlap (weight=2)
        for tok in addr_tokens:
            hits = self.addr_token_idx.get(tok)
            if hits:
                for idx in hits:
                    scores[idx] += 2

        # Strategy 3: Name bigrams (weight=3 — more specific)
        if len(core_tokens) >= 2:
            for j in range(len(core_tokens) - 1):
                bg = f"{core_tokens[j]}_{core_tokens[j+1]}"
                hits = self.bigram_idx.get(bg)
                if hits:
                    for idx in hits:
                        scores[idx] += 3

        # Strategy 4: ZIP/PIN (weight=3 — strong signal)
        if zip_pin:
            hits = self.zip_idx.get(zip_pin)
            if hits:
                for idx in hits:
                    scores[idx] += 3

        # Strategy 5: Prefix (weight=1)
        stripped = name_sorted.replace(" ", "")
        if len(stripped) >= 4:
            hits = self.prefix_idx.get(stripped[:4])
            if hits:
                for idx in hits:
                    scores[idx] += 1

        if not scores:
            return []

        # Return top candidates ranked by score
        top = scores.most_common(self.max_candidates)
        return [idx for idx, _ in top]


# ===========================================================================
# Feature computation (optimized)
# ===========================================================================

def pair_features(
    n1: str, n2: str,
    n1s: str, n2s: str,
    a1: str, a2: str,
    z1: str, z2: str,
    source_type: float,
    country_code: float,
) -> Dict[str, float]:
    f = {}

    # Name features
    f["name_jw_raw"] = jaro_winkler(n1, n2)
    f["name_jw_norm"] = f["name_jw_raw"]
    f["name_jw_sorted"] = jaro_winkler(n1s, n2s)
    f["name_jw_core"] = f["name_jw_sorted"]
    f["name_lev_norm"] = levenshtein_ratio(n1, n2)
    f["name_lev_core"] = f["name_lev_norm"]

    t1 = set(n1.split()) if n1 else set()
    t2 = set(n2.split()) if n2 else set()
    u = len(t1 | t2)
    inter = len(t1 & t2)
    f["name_jaccard_tokens_norm"] = inter / u if u else 1.0
    f["name_jaccard_tokens_core"] = f["name_jaccard_tokens_norm"]
    f["name_jaccard_trigrams"] = jaccard_char_ngrams(n1, n2, 3)
    f["name_token_set_ratio"] = token_set_ratio(n1, n2)
    f["name_token_sort_ratio"] = levenshtein_ratio(n1s, n2s)
    f["name_lcs_ratio"] = f["name_jaccard_trigrams"]
    f["name_sorted_exact_match"] = float(n1s == n2s and bool(n1s))
    f["name_core_exact_match"] = f["name_sorted_exact_match"]
    f["name_shared_tokens"] = float(inter)

    l1, l2 = len(n1), len(n2)
    f["name_length_ratio"] = min(l1, l2) / max(l1, l2) if l1 and l2 else (1.0 if l1 == l2 else 0.0)
    f["legal_suffix_match"] = 0.0
    f["both_have_legal_suffix"] = 0.0
    f["name_trade_jw"] = f["name_jw_raw"]

    # Address features
    f["addr_jw"] = jaro_winkler(a1, a2)
    f["addr_lev"] = levenshtein_ratio(a1, a2)

    at1 = set(a1.split()) if a1 else set()
    at2 = set(a2.split()) if a2 else set()
    au = len(at1 | at2)
    ainter = len(at1 & at2)
    f["addr_jaccard_tokens"] = ainter / au if au else 1.0
    f["addr_jaccard_trigrams"] = jaccard_char_ngrams(a1, a2, 3)
    f["addr_token_set_ratio"] = token_set_ratio(a1, a2)
    f["addr_lcs_ratio"] = f["addr_jaccard_trigrams"]
    al1, al2 = len(a1), len(a2)
    f["addr_length_ratio"] = min(al1, al2) / max(al1, al2) if al1 and al2 else (1.0 if al1 == al2 else 0.0)
    f["addr_shared_tokens"] = float(ainter)

    f["zippin_exact_match"] = float(bool(z1) and bool(z2) and z1 == z2)
    f["both_have_zippin"] = float(bool(z1) and bool(z2))

    # Cross-field
    f["name1_in_addr2"] = float(len(t1 & at2))
    f["name2_in_addr1"] = float(len(t2 & at1))

    # Meta
    f["source_pair_type"] = source_type
    f["country_encoded"] = country_code
    f["country_match"] = 1.0

    return f


# ===========================================================================
# Load ground truth
# ===========================================================================

def load_gt(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    col = df.columns[1]
    df["match_list"] = df[col].apply(
        lambda x: [m.strip() for m in x.split(",") if m.strip()] if x.strip() else []
    )
    df = df.rename(columns={df.columns[0]: "source1_entity_id"})
    return df[["source1_entity_id", "match_list"]]


# ===========================================================================
# Process one country partition for training
# ===========================================================================

def process_country_train(
    country: str,
    country_code: float,
    df_s1_country: pd.DataFrame,
    df_gt_country: pd.DataFrame,
    s23_paths: List[str],
    sample_size: int = 20_000,
    val_frac: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """Returns (df_feat_train, df_feat_val, blocking_recall)."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing country: {country} ({len(df_s1_country):,} S1 entities)")
    logger.info(f"{'='*60}")

    # Sample S1 entities
    rng = np.random.RandomState(42)
    n_sample = min(sample_size, len(df_s1_country))
    sampled_ids = set(rng.choice(df_s1_country["entity_id"].values, size=n_sample, replace=False))

    n_val = max(1, int(n_sample * val_frac))
    all_sampled = list(sampled_ids)
    rng.shuffle(all_sampled)
    val_ids = set(all_sampled[:n_val])
    train_ids = set(all_sampled[n_val:])

    df_s1_train = df_s1_country[df_s1_country["entity_id"].isin(train_ids)].reset_index(drop=True)
    df_s1_val = df_s1_country[df_s1_country["entity_id"].isin(val_ids)].reset_index(drop=True)

    gt_train = df_gt_country[df_gt_country["source1_entity_id"].isin(train_ids)].reset_index(drop=True)
    gt_val = df_gt_country[df_gt_country["source1_entity_id"].isin(val_ids)].reset_index(drop=True)

    logger.info(f"  Sampled {len(train_ids):,} train + {len(val_ids):,} val S1 entities")

    # Normalize S1
    df_s1_train = prepare_df(df_s1_train)
    df_s1_val = prepare_df(df_s1_val)

    # Load and filter S2+S3 for this country
    logger.info(f"  Loading S2+S3 for country={country}...")
    s23_chunks = []
    for path in s23_paths:
        for chunk in pd.read_csv(path, sep="\t", dtype=str, chunksize=500_000):
            chunk = chunk.fillna("")
            chunk["country_norm"] = norm_country(chunk["country"])
            mask = chunk["country_norm"] == country
            if mask.any():
                s23_chunks.append(chunk[mask].reset_index(drop=True))
        gc.collect()

    if not s23_chunks:
        logger.warning(f"  No S2/S3 records for {country}")
        return pd.DataFrame(), pd.DataFrame(), 0.0

    df_s23 = pd.concat(s23_chunks, ignore_index=True)
    del s23_chunks
    gc.collect()
    logger.info(f"  Loaded {len(df_s23):,} S2+S3 records for {country}")

    df_s23 = prepare_df(df_s23)

    # Build smart blocking indices
    blocker = SmartBlocker(max_token_df=50_000, max_candidates=200)
    blocker.fit(df_s23)

    # Build positive pair set
    pos_pairs = set()
    for _, row in df_gt_country.iterrows():
        for mid in row["match_list"]:
            pos_pairs.add((row["source1_entity_id"], mid))

    # Precompute S23 arrays for fast access
    s23_eids = df_s23["entity_id"].values
    s23_names = df_s23["name_norm"].values
    s23_sorted = df_s23["name_sorted"].values
    s23_addrs = df_s23["addr_norm"].values
    s23_zips = df_s23["zip_pin"].values

    def _compute(df_s1_split, gt_split, split_name):
        rows = []
        total_true = 0
        found_true = 0

        # Build set of true match IDs per S1 for recall tracking
        true_matches_map = {}
        for _, gtr in gt_split.iterrows():
            true_matches_map[gtr["source1_entity_id"]] = set(gtr["match_list"])
            total_true += len(gtr["match_list"])

        for _, s1_row in df_s1_split.iterrows():
            s1_id = s1_row["entity_id"]

            cand_indices = blocker.get_candidates(
                s1_row["core_tokens"],
                s1_row["addr_key_tokens"],
                s1_row["name_sorted"],
                s1_row["zip_pin"],
            )

            # Track blocking recall
            cand_eids = {s23_eids[ci] for ci in cand_indices}
            true_set = true_matches_map.get(s1_id, set())
            found_true += len(true_set & cand_eids)

            for ci in cand_indices:
                s23_id = s23_eids[ci]
                src_type = 0.0 if str(s23_id).startswith("S2") else 1.0

                feats = pair_features(
                    s1_row["name_norm"], s23_names[ci],
                    s1_row["name_sorted"], s23_sorted[ci],
                    s1_row["addr_norm"], s23_addrs[ci],
                    s1_row["zip_pin"], s23_zips[ci],
                    src_type, country_code,
                )
                feats["source1_entity_id"] = s1_id
                feats["candidate_entity_id"] = s23_id
                feats["label"] = 1 if (s1_id, s23_id) in pos_pairs else 0
                rows.append(feats)

        df_feat = pd.DataFrame(rows)
        n_pos = df_feat["label"].sum() if len(df_feat) > 0 else 0
        recall = found_true / max(total_true, 1)
        logger.info(
            f"  {split_name}: {len(df_feat):,} pairs, {n_pos:,} positives, "
            f"blocking recall={found_true}/{total_true}={recall:.4f}"
        )
        return df_feat, recall

    logger.info(f"  Computing TRAIN features...")
    df_feat_train, train_recall = _compute(df_s1_train, gt_train, "TRAIN")

    logger.info(f"  Computing VAL features...")
    df_feat_val, val_recall = _compute(df_s1_val, gt_val, "VAL")

    del df_s23, blocker
    gc.collect()

    return df_feat_train, df_feat_val, val_recall


# ===========================================================================
# Training
# ===========================================================================

def run_train(train_dir: str, model_path: str, val_frac: float = 0.15):
    t0 = time.time()

    logger.info("Loading S1 + ground truth...")
    df_s1 = pd.read_csv(os.path.join(train_dir, "train_source1.tsv"), sep="\t", dtype=str).fillna("")
    df_s1["country_norm"] = norm_country(df_s1["country"])

    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")
    if not os.path.exists(gt_path):
        gt_path = os.path.join(train_dir, "train_labels.tsv")
    df_gt = load_gt(gt_path)

    logger.info(f"S1: {len(df_s1):,}, GT: {len(df_gt):,}")

    s23_paths = [
        os.path.join(train_dir, "train_source2.tsv"),
        os.path.join(train_dir, "train_source3.tsv"),
    ]

    country_configs = [
        ("US", 0.0, 20_000),
        ("IN", 1.0, 20_000),
    ]

    all_train = []
    all_val = []

    for country, code, sample_size in country_configs:
        df_s1_c = df_s1[df_s1["country_norm"] == country].reset_index(drop=True)
        df_gt_c = df_gt[df_gt["source1_entity_id"].isin(set(df_s1_c["entity_id"]))].reset_index(drop=True)

        ft, fv, recall = process_country_train(
            country, code, df_s1_c, df_gt_c, s23_paths,
            sample_size=sample_size, val_frac=val_frac,
        )
        if len(ft) > 0:
            all_train.append(ft)
        if len(fv) > 0:
            all_val.append(fv)

        if recall < 0.5:
            logger.warning(
                f"  WARNING: {country} blocking recall={recall:.4f} is very low! "
                "True matches are being missed by blocking."
            )
        gc.collect()

    df_feat_train = pd.concat(all_train, ignore_index=True)
    df_feat_val = pd.concat(all_val, ignore_index=True)

    logger.info(f"\nCombined train: {df_feat_train.shape}, pos={df_feat_train['label'].sum():,}")
    logger.info(f"Combined val  : {df_feat_val.shape}, pos={df_feat_val['label'].sum():,}")

    # Train model
    logger.info("=" * 60)
    logger.info("TRAINING MODEL...")
    logger.info("=" * 60)

    model = MatchingModel(
        feature_cols=FEATURE_COLUMNS,
        neg_ratio=7,
        random_state=42,
    )
    model.fit(df_feat_train, label_col="label", df_val=df_feat_val)

    imp = model.get_feature_importance()
    if imp is not None:
        logger.info("Top 10 features:")
        for feat, v in imp.head(10).items():
            logger.info(f"  {feat}: {v}")

    df_feat_val = df_feat_val.copy()
    df_feat_val["score"] = model.predict_scores(df_feat_val)
    f05 = compute_f05_per_entity(df_feat_val, threshold=model.threshold)
    logger.info(f"Validation F_0.5 @ threshold={model.threshold}: {f05:.4f}")

    model.save(model_path)
    logger.info(f"Training complete in {time.time()-t0:.0f}s")
    return model


# ===========================================================================
# Prediction
# ===========================================================================

def run_predict(
    test_dir: str,
    model_path: str,
    output_dir: str,
    singleton_threshold: float = 0.35,
    batch_size: int = 10_000,
    max_candidates: int = 40,
):
    t0 = time.time()

    logger.info("Loading model...")
    model = MatchingModel.load(model_path)

    logger.info("Loading test S1...")
    df_s1 = pd.read_csv(os.path.join(test_dir, "test_source1.tsv"), sep="\t", dtype=str).fillna("")
    df_s1["country_norm"] = norm_country(df_s1["country"])
    logger.info(f"Test S1: {len(df_s1):,}")
    logger.info(f"Countries: {df_s1['country_norm'].value_counts().to_dict()}")

    s23_paths = [
        os.path.join(test_dir, "test_source2.tsv"),
        os.path.join(test_dir, "test_source3.tsv"),
    ]

    os.makedirs(output_dir, exist_ok=True)
    match_file_path = os.path.join(output_dir, "matching_results.tsv")
    cand_file_path = os.path.join(output_dir, "candidate_pairs.tsv")

    # Open files and write headers
    f_match = open(match_file_path, "w", encoding="utf-8", newline="")
    f_cand = open(cand_file_path, "w", encoding="utf-8", newline="")

    f_match.write("source1_entity_id\tmatched_entity_ids\n")
    f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

    country_code_map = {"US": 0.0, "IN": 1.0, "FR": 2.0}
    seen_s1_ids = set()
    total_matched_count = 0
    total_singleton_count = 0

    # Process each country partition
    for country in ["US", "IN", "FR"]:
        df_s1_c = df_s1[df_s1["country_norm"] == country].reset_index(drop=True)
        if len(df_s1_c) == 0:
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"PREDICTING: {country} ({len(df_s1_c):,} S1 entities)")
        logger.info(f"{'='*60}")

        df_s1_c = prepare_df(df_s1_c)
        country_code = country_code_map.get(country, -1.0)

        # Load S2+S3 for this country
        logger.info(f"  Loading S2+S3 for {country}...")
        s23_chunks = []
        for path in s23_paths:
            for chunk in pd.read_csv(path, sep="\t", dtype=str, chunksize=500_000):
                chunk = chunk.fillna("")
                chunk["country_norm"] = norm_country(chunk["country"])
                mask = chunk["country_norm"] == country
                if mask.any():
                    s23_chunks.append(chunk[mask].reset_index(drop=True))
            gc.collect()

        if not s23_chunks:
            logger.warning(f"  No S2/S3 for {country}")
            for eid in df_s1_c["entity_id"].values:
                f_match.write(f"{eid}\t\n")
                f_cand.write(f"{eid}\t\n")
                seen_s1_ids.add(eid)
                total_singleton_count += 1
            f_match.flush()
            f_cand.flush()
            continue

        df_s23 = pd.concat(s23_chunks, ignore_index=True)
        del s23_chunks
        gc.collect()
        df_s23 = prepare_df(df_s23)
        logger.info(f"  {len(df_s23):,} S2+S3 records loaded and normalized")

        blocker = SmartBlocker(
            max_token_df=50_000,
            max_candidates=max_candidates,
            max_bigram_df=10_000,
            max_prefix_df=5_000,
        )
        blocker.fit(df_s23)

        s23_eids = df_s23["entity_id"].values
        s23_names = df_s23["name_norm"].values
        s23_sorted = df_s23["name_sorted"].values
        s23_addrs = df_s23["addr_norm"].values
        s23_zips = df_s23["zip_pin"].values

        n_batches = (len(df_s1_c) + batch_size - 1) // batch_size
        country_start_time = time.time()

        for batch_idx in range(n_batches):
            b_t0 = time.time()
            start = batch_idx * batch_size
            end = min(start + batch_size, len(df_s1_c))
            batch = df_s1_c.iloc[start:end]

            b_eids = batch["entity_id"].values
            b_names = batch["name_norm"].values
            b_sorted = batch["name_sorted"].values
            b_addrs = batch["addr_norm"].values
            b_zips = batch["zip_pin"].values
            b_core = batch["core_tokens"].values
            b_addr_toks = batch["addr_key_tokens"].values

            cand_lines = []
            feat_rows = []

            for i in range(len(batch)):
                s1_id = b_eids[i]
                seen_s1_ids.add(s1_id)

                cand_indices = blocker.get_candidates(
                    b_core[i],
                    b_addr_toks[i],
                    b_sorted[i],
                    b_zips[i],
                )

                if cand_indices:
                    cand_ids_list = [s23_eids[ci] for ci in cand_indices]
                    unique_cands = sorted(set(cand_ids_list))
                    cand_lines.append(f"{s1_id}\t{','.join(unique_cands)}\n")

                    s1_n = b_names[i]
                    s1_s = b_sorted[i]
                    s1_a = b_addrs[i]
                    s1_z = b_zips[i]

                    for ci in cand_indices:
                        cid = s23_eids[ci]
                        src_type = 0.0 if str(cid).startswith("S2") else 1.0
                        f = pair_features(
                            s1_n, s23_names[ci],
                            s1_s, s23_sorted[ci],
                            s1_a, s23_addrs[ci],
                            s1_z, s23_zips[ci],
                            src_type, country_code,
                        )
                        f["source1_entity_id"] = s1_id
                        f["candidate_entity_id"] = cid
                        feat_rows.append(f)
                else:
                    cand_lines.append(f"{s1_id}\t\n")

            # Write candidates for this batch
            f_cand.writelines(cand_lines)

            # Score candidates and write matches
            if feat_rows:
                df_feat = pd.DataFrame(feat_rows)
                df_feat["score"] = model.predict_scores(df_feat)

                # Find max candidate score per S1 for singleton protection
                max_scores = df_feat.groupby("source1_entity_id")["score"].max()

                # Find matches passing threshold
                df_pos = df_feat[df_feat["score"] >= model.threshold]
                if len(df_pos) > 0:
                    pos_groups = df_pos.groupby("source1_entity_id")["candidate_entity_id"].apply(
                        lambda x: ",".join(sorted(set(x)))
                    ).to_dict()
                else:
                    pos_groups = {}

                match_lines = []
                for s1_id in b_eids:
                    max_sc = max_scores.get(s1_id, 0.0)
                    if max_sc >= singleton_threshold and s1_id in pos_groups:
                        matched_str = pos_groups[s1_id]
                        match_lines.append(f"{s1_id}\t{matched_str}\n")
                        total_matched_count += 1
                    else:
                        match_lines.append(f"{s1_id}\t\n")
                        total_singleton_count += 1

                f_match.writelines(match_lines)
                del df_feat, df_pos, feat_rows, match_lines
            else:
                for s1_id in b_eids:
                    f_match.write(f"{s1_id}\t\n")
                    total_singleton_count += 1

            del cand_lines
            f_match.flush()
            f_cand.flush()
            gc.collect()

            b_dt = time.time() - b_t0
            rate = len(batch) / max(b_dt, 0.001)
            logger.info(
                f"  Batch {batch_idx+1}/{n_batches} ({start:,}-{end:,}) done in "
                f"{b_dt:.1f}s ({rate:.0f} entities/s)"
            )

        country_dt = time.time() - country_start_time
        logger.info(f"  {country} completed in {country_dt:.1f}s ({len(df_s1_c)/max(country_dt,1):.0f} entities/s)")

        del df_s23, blocker
        gc.collect()

    # Check for any missing S1 entities
    all_s1_set = set(df_s1["entity_id"].values)
    missing = all_s1_set - seen_s1_ids
    if missing:
        logger.warning(f"Writing {len(missing):,} missing S1 entities as singletons...")
        for eid in missing:
            f_match.write(f"{eid}\t\n")
            f_cand.write(f"{eid}\t\n")
            total_singleton_count += 1

    f_match.close()
    f_cand.close()

    logger.info("=" * 60)
    logger.info(f"PREDICTION COMPLETE in {time.time()-t0:.1f}s")
    logger.info(f"Total matched entities: {total_matched_count:,}")
    logger.info(f"Total singleton entities: {total_singleton_count:,}")
    logger.info(f"Saved: {match_file_path}")
    logger.info(f"Saved: {cand_file_path}")
    logger.info("=" * 60)


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Entity Resolution Pipeline v3")
    parser.add_argument("--mode", choices=["train", "predict", "full"], default="full")
    parser.add_argument("--train-dir", default=os.path.join(DATA_BASE, "train"))
    parser.add_argument("--test-dir", default=os.path.join(DATA_BASE, "test"))
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--model-path", default="output/model.pkl")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--singleton-threshold", type=float, default=0.35)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--max-candidates", type=int, default=40)
    args = parser.parse_args()

    if args.mode in ("train", "full"):
        run_train(args.train_dir, args.model_path, val_frac=args.val_frac)

    if args.mode in ("predict", "full"):
        run_predict(
            args.test_dir,
            args.model_path,
            args.output_dir,
            singleton_threshold=args.singleton_threshold,
            batch_size=args.batch_size,
            max_candidates=args.max_candidates,
        )


if __name__ == "__main__":
    main()
