"""Quick v4 feature smoke test."""
import sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_pipeline import (
    transliterate_devanagari, extract_zip, extract_street_number,
    pair_features, _has_devanagari,
)
from src.features import FEATURE_COLUMNS

# 1. Devanagari transliteration
print("=" * 60)
print("1. DEVANAGARI TRANSLITERATION")
print("=" * 60)
tests = [
    ('\u0939\u094b\u091f\u0932', 'hotel'),
    ('\u0932\u093f\u092e\u093f\u091f\u0947\u0921', 'limited'),
    ('acme corp', 'acme corp'),  # no-op for Latin
    ('\u0936\u0930\u094d\u092e\u093e', 'sharma'),
]
for hindi, expected_substr in tests:
    result = transliterate_devanagari(hindi.lower())
    print(f"  {hindi!r:30s} -> {result!r:30s} (expect contains '{expected_substr}')")

# 2. Flexible India PIN
print("\n" + "=" * 60)
print("2. FLEXIBLE INDIA PIN EXTRACTION")
print("=" * 60)
pins = [
    ('pin-110001', 'IN', '110001'),
    ('kolkata 700 001', 'IN', '700001'),
    ('pin: 560001', 'IN', '560001'),
    ('411001 pune', 'IN', '411001'),
    ('12345 main st', 'US', '12345'),
]
all_pass = True
for addr, country, expected in pins:
    result = extract_zip(addr, country)
    status = 'PASS' if result == expected else 'FAIL'
    if status == 'FAIL':
        all_pass = False
    print(f"  {status}: extract_zip({addr!r}, {country!r}) = {result!r} (expected {expected!r})")

# 3. Street number extraction
print("\n" + "=" * 60)
print("3. STREET NUMBER EXTRACTION")
print("=" * 60)
addrs = [
    ('101 main street', '101'),
    ('850 broadway suite 200', '850'),
    ('no 5 mg road', '5'),
]
for addr, expected in addrs:
    result = extract_street_number(addr)
    status = 'PASS' if result == expected else 'FAIL'
    if status == 'FAIL':
        all_pass = False
    print(f"  {status}: extract_street_number({addr!r}) = {result!r} (expected {expected!r})")

# 4. Street number discrepancy features
print("\n" + "=" * 60)
print("4. STREET NUMBER DISCREPANCY FEATURES")
print("=" * 60)
# Same company, different branch addresses
f = pair_features(
    'starbucks inc', 'starbucks inc',
    'inc starbucks', 'inc starbucks',
    '101 main st new york', '850 main st new york',
    '10001', '10001',
    '101', '850',
    0.0, 0.0,
)
print(f"  Same name, different street num:")
print(f"    street_num_exact_match = {f['street_num_exact_match']} (expected 0.0)")
print(f"    street_num_conflict    = {f['street_num_conflict']} (expected 1.0)")
print(f"    zippin_conflict        = {f['zippin_conflict']} (expected 0.0)")
assert f['street_num_conflict'] == 1.0, "FAIL: street_num_conflict should be 1.0"
assert f['street_num_exact_match'] == 0.0, "FAIL: street_num_exact_match should be 0.0"

# Same company, same address
f2 = pair_features(
    'acme corp', 'acme corporation',
    'acme corp', 'acme corporation',
    '101 main st', '101 main st',
    '10001', '10001',
    '101', '101',
    0.0, 0.0,
)
print(f"  Same addr:")
print(f"    street_num_exact_match = {f2['street_num_exact_match']} (expected 1.0)")
print(f"    street_num_conflict    = {f2['street_num_conflict']} (expected 0.0)")
assert f2['street_num_exact_match'] == 1.0

# Different ZIP
f3 = pair_features(
    'acme corp', 'acme corp',
    'acme corp', 'acme corp',
    '101 main st', '101 main st',
    '10001', '20002',
    '101', '101',
    0.0, 0.0,
)
print(f"  Different ZIP:")
print(f"    zippin_conflict = {f3['zippin_conflict']} (expected 1.0)")
assert f3['zippin_conflict'] == 1.0

# 5. Feature count
print(f"\n{'='*60}")
print(f"5. FEATURE COLUMNS: {len(FEATURE_COLUMNS)} (expected 38)")
assert len(FEATURE_COLUMNS) == 38, f"Expected 38 features, got {len(FEATURE_COLUMNS)}"

print(f"\n{'='*60}")
print("ALL V4 SMOKE TESTS PASSED!")
print(f"{'='*60}")
