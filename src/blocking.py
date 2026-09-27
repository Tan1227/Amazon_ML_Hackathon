"""
Phase 3: Blocking / Candidate Generation
=========================================
6 independent blocking strategies whose results are unioned
to produce candidate_pairs.tsv.

Strategies:
  1. Hard country filter (applied to ALL other strategies)
  2. Token overlap inverted index (unigram + bigram)
  3. TF-IDF character n-gram ANN (name + address)
  4. Prefix/suffix blocking on sorted normalized name
  5. Exact PIN/ZIP code blocking
  6. Phonetic blocking (Soundex + Double Metaphone)
"""

import re
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: country normalizer for matching
# ---------------------------------------------------------------------------

def _country_key(country: str) -> str:
    """Return a canonical country key used for hard-filter matching."""
    c = str(country).strip().upper()
    if c in ("US", "USA", "UNITED STATES", "UNITED STATES OF AMERICA"):
        return "US"
    if c in ("IN", "IND", "INDIA"):
        return "IN"
    if c in ("FR", "FRA", "FRANCE"):
        return "FR"
    return c  # pass-through for any other country


# ---------------------------------------------------------------------------
# Phonetic helpers (stdlib only fallback)
# ---------------------------------------------------------------------------

def _soundex(name: str) -> str:
    """Simple Soundex implementation (stdlib-only fallback)."""
    name = re.sub(r'[^a-zA-Z]', '', name).upper()
    if not name:
        return "0000"
    codes = {
        'BFPV': '1', 'CGJKQSXYZ': '2', 'DT': '3',
        'L': '4', 'MN': '5', 'R': '6',
    }
    result = [name[0]]
    prev_code = ''
    for c in name[1:]:
        code = '0'
        for key, val in codes.items():
            if c in key:
                code = val
                break
        if code != '0' and code != prev_code:
            result.append(code)
        prev_code = code
        if len(result) == 4:
            break
    return (''.join(result) + '000')[:4]


def _token_soundex(text: str) -> List[str]:
    """Soundex for each token in the text."""
    tokens = re.findall(r'[a-zA-Z]+', text)
    return [_soundex(t) for t in tokens if len(t) > 1]


# Try to import jellyfish for better phonetic support
try:
    import jellyfish
    def _phonetic_keys(text: str) -> List[str]:
        tokens = re.findall(r'[a-zA-Z]+', text)
        keys = []
        for t in tokens:
            if len(t) < 2:
                continue
            try:
                keys.append(jellyfish.soundex(t))
            except Exception:
                keys.append(_soundex(t))
            try:
                mp = jellyfish.metaphone(t)
                if mp:
                    keys.append(mp)
            except Exception:
                pass
        return list(set(keys))
    JELLYFISH_AVAILABLE = True
except ImportError:
    def _phonetic_keys(text: str) -> List[str]:
        return list(set(_token_soundex(text)))
    JELLYFISH_AVAILABLE = False
    logger.warning("jellyfish not found; using basic Soundex only for phonetic blocking.")


# ---------------------------------------------------------------------------
# Bigram generator
# ---------------------------------------------------------------------------

def _bigrams(tokens: List[str]) -> List[str]:
    if len(tokens) < 2:
        return []
    return [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]


def _char_trigrams(text: str) -> Set[str]:
    """Character trigrams from a string."""
    return {text[i:i+3] for i in range(len(text) - 2)} if len(text) >= 3 else set()


# ---------------------------------------------------------------------------
# Main Blocker Class
# ---------------------------------------------------------------------------

class CandidateBlocker:
    """
    Builds blocking indices from Source 2 & 3 records and retrieves
    candidate pairs for each Source 1 entity.
    """

    def __init__(
        self,
        tfidf_top_k: int = 15,
        tfidf_name_weight: float = 0.7,
        tfidf_addr_weight: float = 0.3,
        min_token_len: int = 2,
    ):
        self.tfidf_top_k = tfidf_top_k
        self.tfidf_name_weight = tfidf_name_weight
        self.tfidf_addr_weight = tfidf_addr_weight
        self.min_token_len = min_token_len

        # Inverted indices (built at fit time)
        self._unigram_idx: Dict[str, Set[str]] = defaultdict(set)
        self._bigram_idx: Dict[str, Set[str]] = defaultdict(set)
        self._trigram_idx: Dict[str, Set[str]] = defaultdict(set)
        self._prefix_idx: Dict[str, Set[str]] = defaultdict(set)
        self._suffix_idx: Dict[str, Set[str]] = defaultdict(set)
        self._zippin_idx: Dict[str, Set[str]] = defaultdict(set)
        self._phonetic_idx: Dict[str, Set[str]] = defaultdict(set)
        self._country_idx: Dict[str, Set[str]] = defaultdict(set)

        # TF-IDF models & matrices
        self._tfidf_name_vec: Optional[TfidfVectorizer] = None
        self._tfidf_addr_vec: Optional[TfidfVectorizer] = None
        self._tfidf_name_matrix = None
        self._tfidf_addr_matrix = None
        self._s23_ids: List[str] = []  # ordered list of S2/S3 entity IDs

        self._fitted = False

    # ------------------------------------------------------------------
    # Fit: build indices from Source 2 + 3 normalized records
    # ------------------------------------------------------------------

    def fit(self, df_s23: pd.DataFrame) -> None:
        """
        df_s23 must have columns:
          entity_id, name_normalized, name_sorted, name_tokens (list),
          addr_normalized, addr_tokens (list), zip_pin, country_norm
        """
        logger.info(f"Building blocking indices over {len(df_s23)} S2/S3 records...")

        self._s23_ids = list(df_s23["entity_id"])
        id_set = set(self._s23_ids)

        for _, row in df_s23.iterrows():
            eid = row["entity_id"]
            country = _country_key(row.get("country_norm", ""))
            name_tokens = row.get("name_tokens") or []
            addr_tokens = row.get("addr_tokens") or []
            name_sorted = str(row.get("name_sorted", "") or "")
            zip_pin = row.get("zip_pin")

            # Country index
            self._country_idx[country].add(eid)

            # Filter trivial tokens
            valid_name_tokens = [
                t for t in name_tokens if len(t) >= self.min_token_len
            ]

            # Strategy 2a: Unigram inverted index
            for tok in valid_name_tokens:
                self._unigram_idx[tok].add(eid)

            # Strategy 2b: Bigram inverted index
            for bg in _bigrams(valid_name_tokens):
                self._bigram_idx[bg].add(eid)

            # Strategy 2c: Char trigram index (on normalized name)
            for tg in _char_trigrams(name_sorted.replace(" ", "")):
                self._trigram_idx[tg].add(eid)

            # Strategy 4: Prefix/suffix blocking
            ns_stripped = name_sorted.replace(" ", "")
            if len(ns_stripped) >= 3:
                self._prefix_idx[ns_stripped[:3]].add(eid)
                self._suffix_idx[ns_stripped[-3:]].add(eid)

            # Strategy 5: PIN/ZIP blocking
            if zip_pin:
                self._zippin_idx[zip_pin].add(eid)

            # Strategy 6: Phonetic blocking
            name_text = " ".join(valid_name_tokens)
            for pkey in _phonetic_keys(name_text):
                self._phonetic_idx[pkey].add(eid)

        # Strategy 3: TF-IDF ANN
        name_corpus = [str(row.get("name_normalized", "") or "") for _, row in df_s23.iterrows()]
        addr_corpus = [str(row.get("addr_normalized", "") or "") for _, row in df_s23.iterrows()]

        self._tfidf_name_vec = TfidfVectorizer(
            analyzer='char_wb', ngram_range=(2, 4), min_df=1, sublinear_tf=True
        )
        self._tfidf_addr_vec = TfidfVectorizer(
            analyzer='char_wb', ngram_range=(2, 4), min_df=1, sublinear_tf=True
        )

        self._tfidf_name_matrix = self._tfidf_name_vec.fit_transform(name_corpus)
        self._tfidf_addr_matrix = self._tfidf_addr_vec.fit_transform(addr_corpus)

        self._fitted = True
        logger.info("Blocking indices built successfully.")

    # ------------------------------------------------------------------
    # Retrieve candidates for a single S1 record
    # ------------------------------------------------------------------

    def get_candidates(self, row: pd.Series, strategies: Optional[List[str]] = None) -> Set[str]:
        """
        Return a set of candidate S2/S3 entity IDs for the given S1 row.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before get_candidates().")

        all_strategies = strategies or [
            "token_unigram", "token_bigram", "char_trigram",
            "tfidf_ann", "prefix_suffix", "zippin", "phonetic"
        ]

        country = _country_key(row.get("country_norm", ""))
        allowed = self._country_idx.get(country, set())

        if not allowed:
            return set()

        candidates: Set[str] = set()

        name_tokens = row.get("name_tokens") or []
        addr_tokens = row.get("addr_tokens") or []
        name_sorted = str(row.get("name_sorted", "") or "")
        zip_pin = row.get("zip_pin")
        name_normalized = str(row.get("name_normalized", "") or "")
        addr_normalized = str(row.get("addr_normalized", "") or "")

        valid_name_tokens = [t for t in name_tokens if len(t) >= self.min_token_len]

        # Strategy 2a: Unigram
        if "token_unigram" in all_strategies:
            for tok in valid_name_tokens:
                candidates.update(self._unigram_idx.get(tok, set()))

        # Strategy 2b: Bigram
        if "token_bigram" in all_strategies:
            for bg in _bigrams(valid_name_tokens):
                candidates.update(self._bigram_idx.get(bg, set()))

        # Strategy 2c: Char trigram
        if "char_trigram" in all_strategies:
            ns_stripped = name_sorted.replace(" ", "")
            for tg in _char_trigrams(ns_stripped):
                candidates.update(self._trigram_idx.get(tg, set()))

        # Strategy 3: TF-IDF ANN
        if "tfidf_ann" in all_strategies:
            tfidf_cands = self._tfidf_retrieve(name_normalized, addr_normalized)
            candidates.update(tfidf_cands)

        # Strategy 4: Prefix/suffix
        if "prefix_suffix" in all_strategies:
            ns_stripped = name_sorted.replace(" ", "")
            if len(ns_stripped) >= 3:
                candidates.update(self._prefix_idx.get(ns_stripped[:3], set()))
                candidates.update(self._suffix_idx.get(ns_stripped[-3:], set()))

        # Strategy 5: PIN/ZIP
        if "zippin" in all_strategies and zip_pin:
            candidates.update(self._zippin_idx.get(zip_pin, set()))

        # Strategy 6: Phonetic
        if "phonetic" in all_strategies:
            for pkey in _phonetic_keys(" ".join(valid_name_tokens)):
                candidates.update(self._phonetic_idx.get(pkey, set()))

        # Apply hard country filter
        candidates &= allowed

        return candidates

    def _tfidf_retrieve(self, name: str, addr: str) -> Set[str]:
        """TF-IDF cosine similarity retrieval for one query record."""
        if self._tfidf_name_vec is None:
            return set()

        result: Set[str] = set()
        k = self.tfidf_top_k

        try:
            name_vec = self._tfidf_name_vec.transform([name or " "])
            name_sims = cosine_similarity(name_vec, self._tfidf_name_matrix).flatten()
        except Exception:
            name_sims = np.zeros(len(self._s23_ids))

        try:
            addr_vec = self._tfidf_addr_vec.transform([addr or " "])
            addr_sims = cosine_similarity(addr_vec, self._tfidf_addr_matrix).flatten()
        except Exception:
            addr_sims = np.zeros(len(self._s23_ids))

        combined = (
            self.tfidf_name_weight * name_sims
            + self.tfidf_addr_weight * addr_sims
        )

        top_indices = np.argpartition(combined, -min(k, len(combined)))[-min(k, len(combined)):]
        for idx in top_indices:
            if combined[idx] > 0.0:
                result.add(self._s23_ids[idx])

        return result

    # ------------------------------------------------------------------
    # Batch retrieval for all S1 entities
    # ------------------------------------------------------------------

    def generate_candidates(
        self, df_s1: pd.DataFrame, strategies: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        For every S1 entity, retrieve candidates from S2/S3.
        Returns a DataFrame with columns: source1_entity_id, candidate_entity_ids (set).
        """
        logger.info(f"Generating candidates for {len(df_s1)} S1 entities...")
        rows = []
        for i, (_, s1_row) in enumerate(df_s1.iterrows()):
            cands = self.get_candidates(s1_row, strategies=strategies)
            rows.append({
                "source1_entity_id": s1_row["entity_id"],
                "candidate_entity_ids": cands,
                "num_candidates": len(cands),
            })
            if (i + 1) % 500 == 0:
                logger.info(f"  Processed {i+1}/{len(df_s1)} S1 entities...")

        df_cands = pd.DataFrame(rows)
        total_cands = df_cands["num_candidates"].sum()
        logger.info(
            f"Generated {total_cands} total candidate pairs "
            f"(avg {total_cands/max(len(df_s1),1):.1f} per S1 entity)."
        )
        return df_cands

    # ------------------------------------------------------------------
    # Blocking recall evaluation (on labeled data)
    # ------------------------------------------------------------------

    def evaluate_blocking_recall(
        self, df_cands: pd.DataFrame, ground_truth: pd.DataFrame
    ) -> float:
        """
        Compute blocking recall:
          # true matches that appear in candidate set / # total true matches.

        ground_truth must have columns: source1_entity_id, matched_entity_ids (set/list).
        df_cands must have columns: source1_entity_id, candidate_entity_ids (set/list).
        """
        cand_map = {
            row["source1_entity_id"]: set(row["candidate_entity_ids"])
            for _, row in df_cands.iterrows()
        }

        total_true = 0
        found = 0
        for _, row in ground_truth.iterrows():
            s1_id = row["source1_entity_id"]
            true_matches = set(row["matched_entity_ids"]) if row["matched_entity_ids"] else set()
            cands = cand_map.get(s1_id, set())
            total_true += len(true_matches)
            found += len(true_matches & cands)

        recall = found / max(total_true, 1)
        logger.info(
            f"Blocking Recall: {found}/{total_true} = {recall:.4f} "
            f"({recall*100:.2f}%)"
        )
        return recall
