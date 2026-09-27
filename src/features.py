"""
Phase 4: Feature Engineering
==============================
Compute a rich feature vector for each (S1, S2/S3) candidate pair.

Feature groups:
  - Name similarity (10+ features on raw + normalized + sorted strings)
  - Address similarity (8+ features)
  - Cross-field features
  - Meta features (source pair type, country, blocking info)
"""

import re
import math
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional fast imports — fall back gracefully
# ---------------------------------------------------------------------------

try:
    from rapidfuzz import fuzz as rfuzz
    from rapidfuzz.distance import JaroWinkler as _JaroWinkler
    RAPIDFUZZ = True
except ImportError:
    RAPIDFUZZ = False
    rfuzz = None  # type: ignore
    _JaroWinkler = None  # type: ignore
    logger.warning("rapidfuzz not found; using slower pure-Python fallbacks.")

try:
    import jellyfish
    JELLYFISH = True
except ImportError:
    JELLYFISH = False
    logger.warning("jellyfish not found; Jaro-Winkler will use rapidfuzz or fallback.")


# ---------------------------------------------------------------------------
# Basic string similarity primitives
# ---------------------------------------------------------------------------

def _safe_str(x) -> str:
    return str(x) if x is not None else ""


def jaro_winkler(s1: str, s2: str) -> float:
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if RAPIDFUZZ and _JaroWinkler is not None:
        # rapidfuzz 3.x API: JaroWinkler.similarity returns 0-1
        return _JaroWinkler.similarity(s1, s2)
    if JELLYFISH:
        return jellyfish.jaro_winkler_similarity(s1, s2)
    # Fallback: simple Jaro
    return _jaro(s1, s2)


def _jaro(s1: str, s2: str) -> float:
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0
    match_dist = max(len1, len2) // 2 - 1
    if match_dist < 0:
        match_dist = 0
    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = transpositions = 0
    for i in range(len1):
        start = max(0, i - match_dist)
        end = min(i + match_dist + 1, len2)
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = s2_matches[j] = True
            matches += 1
            break
    if matches == 0:
        return 0.0
    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1
    return (matches / len1 + matches / len2 + (matches - transpositions / 2) / matches) / 3


def levenshtein_ratio(s1: str, s2: str) -> float:
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if RAPIDFUZZ and rfuzz is not None:
        # rapidfuzz.fuzz.ratio returns 0-100
        return rfuzz.ratio(s1, s2) / 100.0
    # Pure Python DP
    d = _levenshtein_dist(s1, s2)
    max_len = max(len(s1), len(s2))
    return 1.0 - d / max_len


def _levenshtein_dist(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            tmp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = tmp
    return dp[n]


def jaccard_tokens(s1: str, s2: str) -> float:
    t1 = set(s1.split()) if s1 else set()
    t2 = set(s2.split()) if s2 else set()
    if not t1 and not t2:
        return 1.0
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


def jaccard_char_ngrams(s1: str, s2: str, n: int = 3) -> float:
    def _ngrams(s):
        return set(s[i:i+n] for i in range(len(s) - n + 1))
    g1 = _ngrams(s1.replace(" ", "")) if s1 else set()
    g2 = _ngrams(s2.replace(" ", "")) if s2 else set()
    if not g1 and not g2:
        return 1.0
    if not g1 or not g2:
        return 0.0
    return len(g1 & g2) / len(g1 | g2)


def token_set_ratio(s1: str, s2: str) -> float:
    """FuzzyWuzzy-style token set ratio."""
    if RAPIDFUZZ and rfuzz is not None:
        return rfuzz.token_set_ratio(s1, s2) / 100.0
    # Fallback: token sort ratio
    t1 = " ".join(sorted(s1.split())) if s1 else ""
    t2 = " ".join(sorted(s2.split())) if s2 else ""
    return levenshtein_ratio(t1, t2)


def token_sort_ratio(s1: str, s2: str) -> float:
    t1 = " ".join(sorted(s1.split())) if s1 else ""
    t2 = " ".join(sorted(s2.split())) if s2 else ""
    if RAPIDFUZZ and rfuzz is not None:
        return rfuzz.ratio(t1, t2) / 100.0
    return levenshtein_ratio(t1, t2)


def lcs_ratio(s1: str, s2: str) -> float:
    """Longest Common Subsequence ratio."""
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    m, n = len(s1), len(s2)
    # DP for LCS
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(prev[j], curr[j-1])
        prev = curr
    lcs_len = prev[n]
    return 2 * lcs_len / (m + n)


def shared_token_count(s1: str, s2: str) -> int:
    t1 = set(s1.split()) if s1 else set()
    t2 = set(s2.split()) if s2 else set()
    return len(t1 & t2)


def length_ratio(s1: str, s2: str) -> float:
    l1, l2 = len(s1), len(s2)
    if l1 == 0 and l2 == 0:
        return 1.0
    if l1 == 0 or l2 == 0:
        return 0.0
    return min(l1, l2) / max(l1, l2)


def exact_match(s1: str, s2: str) -> int:
    return int(s1.strip() == s2.strip()) if s1 and s2 else 0


def _legal_suffix(name: str) -> str:
    """Extract the last meaningful legal suffix token."""
    from src.normalize import LEGAL_SUFFIX_MAP
    tokens = name.split()
    for tok in reversed(tokens):
        if tok in LEGAL_SUFFIX_MAP:
            return LEGAL_SUFFIX_MAP[tok]
    return ""


# ---------------------------------------------------------------------------
# TF-IDF cosine (precomputed from feature pipeline)
# ---------------------------------------------------------------------------

def tfidf_cosine_precomputed(v1, v2) -> float:
    """Cosine similarity between two sparse TF-IDF vectors."""
    if v1 is None or v2 is None:
        return 0.0
    try:
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim
        return float(cos_sim(v1, v2)[0, 0])
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Feature extraction for a single pair
# ---------------------------------------------------------------------------

def compute_pair_features(
    s1_row: Dict[str, Any],
    s23_row: Dict[str, Any],
    source_pair_type: Optional[str] = None,
) -> Dict[str, float]:
    """
    Compute all features for one (S1, S2/S3) candidate pair.

    Both rows are dicts with keys from normalize_record():
      name_raw_lower, name_normalized, name_sorted, name_core, name_trade,
      name_tokens, name_core_tokens, addr_normalized, addr_tokens, zip_pin,
      country_norm.

    Returns a flat dict of feature name → float.
    """
    feats: Dict[str, float] = {}

    # ---------------------------------------------------------------
    # Name similarity features
    # ---------------------------------------------------------------

    n1_raw = _safe_str(s1_row.get("name_raw_lower"))
    n2_raw = _safe_str(s23_row.get("name_raw_lower"))
    n1_norm = _safe_str(s1_row.get("name_normalized"))
    n2_norm = _safe_str(s23_row.get("name_normalized"))
    n1_sorted = _safe_str(s1_row.get("name_sorted"))
    n2_sorted = _safe_str(s23_row.get("name_sorted"))
    n1_core = _safe_str(s1_row.get("name_core"))
    n2_core = _safe_str(s23_row.get("name_core"))

    # Jaro-Winkler on raw and normalized
    feats["name_jw_raw"] = jaro_winkler(n1_raw, n2_raw)
    feats["name_jw_norm"] = jaro_winkler(n1_norm, n2_norm)
    feats["name_jw_sorted"] = jaro_winkler(n1_sorted, n2_sorted)
    feats["name_jw_core"] = jaro_winkler(n1_core, n2_core)

    # Levenshtein ratio
    feats["name_lev_norm"] = levenshtein_ratio(n1_norm, n2_norm)
    feats["name_lev_core"] = levenshtein_ratio(n1_core, n2_core)

    # Jaccard on tokens
    feats["name_jaccard_tokens_norm"] = jaccard_tokens(n1_norm, n2_norm)
    feats["name_jaccard_tokens_core"] = jaccard_tokens(n1_core, n2_core)

    # Jaccard on char trigrams
    feats["name_jaccard_trigrams"] = jaccard_char_ngrams(n1_norm, n2_norm, n=3)

    # Token set ratio (handles subset matching)
    feats["name_token_set_ratio"] = token_set_ratio(n1_norm, n2_norm)

    # Token sort ratio (handles word-order transpositions)
    feats["name_token_sort_ratio"] = token_sort_ratio(n1_norm, n2_norm)

    # LCS ratio
    feats["name_lcs_ratio"] = lcs_ratio(n1_norm, n2_norm)

    # Sorted-token exact match (binary)
    feats["name_sorted_exact_match"] = float(n1_sorted == n2_sorted and bool(n1_sorted))

    # Core exact match (binary)
    feats["name_core_exact_match"] = float(n1_core == n2_core and bool(n1_core))

    # Shared token count (raw count)
    feats["name_shared_tokens"] = float(shared_token_count(n1_norm, n2_norm))

    # Length ratio
    feats["name_length_ratio"] = length_ratio(n1_norm, n2_norm)

    # Legal suffix match
    suf1 = _legal_suffix(n1_norm)
    suf2 = _legal_suffix(n2_norm)
    feats["legal_suffix_match"] = float(suf1 == suf2 and bool(suf1))
    feats["both_have_legal_suffix"] = float(bool(suf1) and bool(suf2))

    # Trade name similarity (when available)
    n1_trade = _safe_str(s1_row.get("name_trade"))
    n2_trade = _safe_str(s23_row.get("name_trade"))
    if n1_trade or n2_trade:
        feats["name_trade_jw"] = jaro_winkler(n1_trade or n1_norm, n2_trade or n2_norm)
    else:
        feats["name_trade_jw"] = feats["name_jw_norm"]

    # ---------------------------------------------------------------
    # Address similarity features
    # ---------------------------------------------------------------

    a1 = _safe_str(s1_row.get("addr_normalized"))
    a2 = _safe_str(s23_row.get("addr_normalized"))

    feats["addr_jw"] = jaro_winkler(a1, a2)
    feats["addr_lev"] = levenshtein_ratio(a1, a2)
    feats["addr_jaccard_tokens"] = jaccard_tokens(a1, a2)
    feats["addr_jaccard_trigrams"] = jaccard_char_ngrams(a1, a2, n=3)
    feats["addr_token_set_ratio"] = token_set_ratio(a1, a2)
    feats["addr_lcs_ratio"] = lcs_ratio(a1, a2)
    feats["addr_length_ratio"] = length_ratio(a1, a2)
    feats["addr_shared_tokens"] = float(shared_token_count(a1, a2))

    # Exact PIN/ZIP match
    zip1 = _safe_str(s1_row.get("zip_pin"))
    zip2 = _safe_str(s23_row.get("zip_pin"))
    feats["zippin_exact_match"] = float(
        bool(zip1) and bool(zip2) and zip1 == zip2
    )
    feats["both_have_zippin"] = float(bool(zip1) and bool(zip2))

    # ---------------------------------------------------------------
    # Cross-field features
    # ---------------------------------------------------------------

    # S1 name tokens appearing in S2/S3 address
    n1_toks = set(s1_row.get("name_core_tokens") or [])
    a2_toks = set((s23_row.get("addr_tokens") or []))
    cross_1in2 = len(n1_toks & a2_toks)
    feats["name1_in_addr2"] = float(cross_1in2)

    # S2/S3 name tokens appearing in S1 address
    n2_toks = set(s23_row.get("name_core_tokens") or [])
    a1_toks = set((s1_row.get("addr_tokens") or []))
    cross_2in1 = len(n2_toks & a1_toks)
    feats["name2_in_addr1"] = float(cross_2in1)

    # ---------------------------------------------------------------
    # Meta features
    # ---------------------------------------------------------------

    # Source pair type: S1-S2 = 0, S1-S3 = 1
    s23_id = _safe_str(s23_row.get("entity_id", ""))
    if source_pair_type is not None:
        feats["source_pair_type"] = float(source_pair_type)
    elif s23_id.startswith("S2"):
        feats["source_pair_type"] = 0.0
    elif s23_id.startswith("S3"):
        feats["source_pair_type"] = 1.0
    else:
        feats["source_pair_type"] = -1.0

    # Country encoding
    c1 = _safe_str(s1_row.get("country_norm"))
    country_map = {"US": 0, "IN": 1, "FR": 2}
    feats["country_encoded"] = float(country_map.get(c1, -1))
    feats["country_match"] = float(c1 == _safe_str(s23_row.get("country_norm")))

    return feats


# ---------------------------------------------------------------------------
# Batch feature computation
# ---------------------------------------------------------------------------

def compute_features_for_candidates(
    df_s1_norm: pd.DataFrame,
    df_s23_norm: pd.DataFrame,
    candidate_pairs: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compute features for all candidate pairs.

    Parameters
    ----------
    df_s1_norm : pd.DataFrame
        Normalized S1 records. Must have 'entity_id' column + norm fields.
    df_s23_norm : pd.DataFrame
        Normalized S2+S3 records. Must have 'entity_id' column + norm fields.
    candidate_pairs : pd.DataFrame
        Output of CandidateBlocker.generate_candidates().
        Must have 'source1_entity_id' and 'candidate_entity_ids' (set/list).

    Returns
    -------
    pd.DataFrame with one row per candidate pair, columns:
        source1_entity_id, candidate_entity_id, label (if available),
        + all feature columns.
    """
    # Build lookup dicts
    s1_map = {row["entity_id"]: row.to_dict() for _, row in df_s1_norm.iterrows()}
    s23_map = {row["entity_id"]: row.to_dict() for _, row in df_s23_norm.iterrows()}

    rows = []
    total = len(candidate_pairs)
    for i, (_, cp_row) in enumerate(candidate_pairs.iterrows()):
        s1_id = cp_row["source1_entity_id"]
        cand_ids = cp_row.get("candidate_entity_ids") or []
        if isinstance(cand_ids, str):
            cand_ids = [c.strip() for c in cand_ids.split(",") if c.strip()]

        s1_data = s1_map.get(s1_id)
        if s1_data is None:
            continue

        for cand_id in cand_ids:
            s23_data = s23_map.get(cand_id)
            if s23_data is None:
                continue

            feats = compute_pair_features(s1_data, s23_data)
            feats["source1_entity_id"] = s1_id
            feats["candidate_entity_id"] = cand_id
            rows.append(feats)

        if verbose and (i + 1) % 500 == 0:
            logger.info(f"  Features computed for {i+1}/{total} S1 entities...")

    df_features = pd.DataFrame(rows)
    logger.info(f"Feature matrix shape: {df_features.shape}")
    return df_features


# ---------------------------------------------------------------------------
# Feature columns (for model training)
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    # Name features
    "name_jw_raw", "name_jw_norm", "name_jw_sorted", "name_jw_core",
    "name_lev_norm", "name_lev_core",
    "name_jaccard_tokens_norm", "name_jaccard_tokens_core",
    "name_jaccard_trigrams",
    "name_token_set_ratio", "name_token_sort_ratio",
    "name_lcs_ratio",
    "name_sorted_exact_match", "name_core_exact_match",
    "name_shared_tokens", "name_length_ratio",
    "legal_suffix_match", "both_have_legal_suffix",
    "name_trade_jw",
    # Address features
    "addr_jw", "addr_lev",
    "addr_jaccard_tokens", "addr_jaccard_trigrams",
    "addr_token_set_ratio", "addr_lcs_ratio",
    "addr_length_ratio", "addr_shared_tokens",
    "zippin_exact_match", "both_have_zippin",
    # v4: Street number discrepancy features
    "street_num_exact_match", "street_num_conflict",
    "street_num_either_missing",
    # v4: ZIP/PIN conflict feature
    "zippin_conflict",
    # Cross-field features
    "name1_in_addr2", "name2_in_addr1",
    # Meta features
    "source_pair_type", "country_encoded", "country_match",
]
