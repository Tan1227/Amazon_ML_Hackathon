"""Smoke test for the full pipeline (run from project root)."""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, r'c:\Users\tanvi\OneDrive\Sem 5\-_-\Amazon_ML_Hackathon')

import rapidfuzz, lightgbm
print(f'rapidfuzz {rapidfuzz.__version__}  OK')
print(f'lightgbm  {lightgbm.__version__}  OK')

from src.normalize import normalize_record
from src.blocking import CandidateBlocker, JELLYFISH_AVAILABLE
from src.features import FEATURE_COLUMNS, compute_features_for_candidates, jaro_winkler, levenshtein_ratio
from src.model import MatchingModel, f_beta_score
import pandas as pd

print(f'jellyfish available: {JELLYFISH_AVAILABLE}')

# --- Normalization ---
r = normalize_record('Sharma Electronics Pvt Ltd', '123 MG Road, Near SBI ATM, Bangalore 560001', 'India')
assert r['zip_pin'] == '560001', f"PIN extraction failed: {r['zip_pin']}"
assert r['name_core'] == 'sharma electronics', f"Core wrong: {r['name_core']}"
print(f"Normalize OK  zip_pin={r['zip_pin']}  name_core={r['name_core']}")

# DBA split test
r2 = normalize_record('Reliance Ltd DBA Jio Stores', 'MG Road Delhi', 'India')
assert r2['name_trade'] is not None, "DBA split failed"
print(f"DBA split OK  trade={r2['name_trade']}")

# US ZIP test
r3 = normalize_record('ABC Corp', '500 Main Street New York 10001', 'US')
assert r3['zip_pin'] == '10001', f"US ZIP failed: {r3['zip_pin']}"
print(f"US ZIP OK  zip={r3['zip_pin']}")

# --- Blocking ---
def norm_df(df):
    rows = []
    for _, row in df.iterrows():
        n = normalize_record(row['business_name'], row['business_address'], row['country'])
        n['entity_id'] = row['entity_id']
        rows.append(n)
    return pd.DataFrame(rows)

s1 = pd.DataFrame([
    {'entity_id': 'S1-001', 'business_name': 'Sharma Electronics Pvt Ltd',
     'business_address': '123 MG Road Bangalore 560001', 'country': 'India'},
    {'entity_id': 'S1-002', 'business_name': 'ABC Corp Inc',
     'business_address': '500 Main St New York 10001', 'country': 'US'},
])
s23 = pd.DataFrame([
    {'entity_id': 'S2-001', 'business_name': 'Sharma Electronics Private Limited',
     'business_address': 'MG Road Bangalore 560001', 'country': 'India'},
    {'entity_id': 'S2-002', 'business_name': 'XYZ Traders',
     'business_address': 'Park Street New York', 'country': 'US'},
    {'entity_id': 'S3-001', 'business_name': 'Sharma Electrnics Pvt Ltd',
     'business_address': '123 MG Road Bangalore', 'country': 'India'},
])

s1_norm  = norm_df(s1)
s23_norm = norm_df(s23)

blocker = CandidateBlocker(tfidf_top_k=5)
blocker.fit(s23_norm)
cands = blocker.generate_candidates(s1_norm)

print('\nCandidates generated:')
for _, row in cands.iterrows():
    print(f"  {row['source1_entity_id']} -> {row['candidate_entity_ids']}")

# Verify country filter: S1-002 (US) should NOT get India candidates
s1_002_cands = cands[cands['source1_entity_id'] == 'S1-002']['candidate_entity_ids'].iloc[0]
for cid in s1_002_cands:
    assert not cid.startswith('S2-001'), f"Country filter broken: US entity got India candidate {cid}"
print('Country filter OK')

# --- Features ---
feats = compute_features_for_candidates(s1_norm, s23_norm, cands, verbose=False)
n_feat_cols = sum(1 for c in feats.columns if c in FEATURE_COLUMNS)
print(f'\nFeature matrix: {feats.shape}  ({n_feat_cols} feature cols present)')

# Check Jaro-Winkler via rapidfuzz
jw = jaro_winkler('sharma electronics pvt ltd', 'sharma electronics private limited')
lev = levenshtein_ratio('sharma electronics pvt ltd', 'sharma electronics private limited')
print(f'Jaro-Winkler : {jw:.4f}  (expected > 0.8)')
print(f'Levenshtein  : {lev:.4f}  (expected > 0.7)')
assert jw > 0.8, f'JW too low: {jw}'
assert lev > 0.7, f'Lev too low: {lev}'

# F0.5 metric
f = f_beta_score(0.667, 1.0, 0.5)
assert abs(f - 0.7146) < 0.001, f'F0.5 wrong: {f}'
print(f'F0.5 check   : {f:.4f}  OK')

print('\n=== ALL SYSTEMS GO ===')
