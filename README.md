# Business Entity Resolution — Pipeline Guide

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place your data files
```
dataset/
  train/
    train_source1.tsv
    train_source2.tsv
    train_source3.tsv
    train_labels.tsv
  test/
    test_source1.tsv
    test_source2.tsv
    test_source3.tsv
```

### 3. Run full pipeline (train + predict)
```bash
python pipeline.py --mode full
```

### 4. Validate submission
```bash
python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

---

## Pipeline Phases

| Phase | Module | Description |
|-------|--------|-------------|
| 1 | `src/data.py` | Load TSVs, EDA profiling, train/val split |
| 2 | `src/normalize.py` | Country-aware text normalization |
| 3 | `src/blocking.py` | 6-strategy candidate generation |
| 4 | `src/features.py` | 35+ similarity features per pair |
| 5 | `src/model.py` | LightGBM classifier + threshold tuning |
| 6 | `src/model.py` | Singleton handling |
| 7 | `pipeline.py` | Country-specific considerations (built in) |
| 8 | `src/data.py` | Validation framework, per-country F_0.5 |

---

## Blocking Strategies

1. **Hard country filter** — S1 entities only matched within same country
2. **Token unigram/bigram inverted index** — shared word tokens in business name
3. **Character trigram index** — handles typos at character level
4. **TF-IDF ANN** — char n-gram vectorization, top-K cosine retrieval (name + addr)
5. **Prefix/suffix blocking** — first/last 3 chars of sorted normalized name
6. **Exact PIN/ZIP blocking** — 5-digit US ZIP or 6-digit India PIN
7. **Phonetic blocking** — Soundex + Double Metaphone on name tokens

Target blocking recall: **>95%** on validation set.

---

## Feature Groups (35+ features)

- **Name similarity**: Jaro-Winkler, Levenshtein, Jaccard (token + trigram), TF-IDF cosine, token set ratio, LCS, sorted-token match, legal suffix match, length ratio, trade name similarity
- **Address similarity**: Jaro-Winkler, Levenshtein, Jaccard, trigram, token set, LCS, PIN/ZIP match, length ratio
- **Cross-field**: S1 name tokens in S2/S3 address, and vice versa
- **Meta**: source pair type (S2 vs S3), country encoding, country match flag

---

## CLI Options

```
python pipeline.py [OPTIONS]

Options:
  --mode {train,predict,full}   Pipeline mode (default: full)
  --train-dir PATH              Training data directory (default: dataset/train)
  --test-dir PATH               Test data directory (default: dataset/test)
  --output-dir PATH             Output directory (default: output)
  --model-path PATH             Model save/load path (default: output/model.pkl)
  --val-frac FLOAT              Val fraction of S1 entities (default: 0.20)
  --singleton-threshold FLOAT   Max score to call singleton (default: 0.35)
  --no-eda                      Skip EDA/profiling step
```

---

## Output Files

| File | Description |
|------|-------------|
| `output/matching_results.tsv` | Final predictions — one row per S1 entity |
| `output/candidate_pairs.tsv` | Blocking output — candidates fed to the model |
| `output/model.pkl` | Trained LightGBM model |
| `pipeline.log` | Full run log |

---

## Key Design Decisions

- **F_0.5 is precision-heavy**: threshold is swept on val set from 0.2–0.95 and chosen to maximize F_0.5 (not accuracy or F1). Expect threshold > 0.5.
- **Singleton handling**: after prediction, any S1 entity whose best candidate score is below `--singleton-threshold` is forced to empty (no matches).
- **Entity-level split**: validation split is done at the S1 entity level, never pair-level, to prevent label leakage.
- **Negative sampling**: 7 negatives per positive during training; hard negatives (high TF-IDF similarity non-matches) are over-sampled.
- **No external APIs**: fully self-contained — no geocoding, no company lookup databases.
