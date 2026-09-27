"""Quick EDA on the dataset."""
import sys, os
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)
import pandas as pd

base = r'c:\Users\tanvi\OneDrive\Sem 5\-_-\Amazon_ML_Hackathon\dataset\6ab10eb3b23ba_student_resource\student_resource\dataset'

# === Quick peek ===
for name in ['train/train_source1.tsv', 'train/train_source2.tsv', 'train/train_source3.tsv', 'train/train_ground_truth.tsv']:
    path = os.path.join(base, name)
    df = pd.read_csv(path, sep='\t', nrows=3, dtype=str)
    print(f'=== {name} ===')
    print(f'  Columns: {list(df.columns)}')
    print(f'  First row: {dict(df.iloc[0])}')
    print()

# === Row counts ===
print('=== ROW COUNTS ===')
for name in ['train/train_source1.tsv', 'train/train_source2.tsv', 'train/train_source3.tsv', 
             'train/train_ground_truth.tsv',
             'test/test_source1.tsv', 'test/test_source2.tsv', 'test/test_source3.tsv']:
    path = os.path.join(base, name)
    df = pd.read_csv(path, sep='\t', usecols=[0], dtype=str)
    print(f'  {name}: {len(df):,} rows')

# === Ground truth analysis ===
print('\n=== GROUND TRUTH ANALYSIS ===')
gt = pd.read_csv(os.path.join(base, 'train/train_ground_truth.tsv'), sep='\t', dtype=str).fillna('')
print(f'Columns: {list(gt.columns)}')
print(f'First 5 rows:')
print(gt.head().to_string())

# Match distribution
gt['match_list'] = gt.iloc[:, 1].apply(lambda x: [m.strip() for m in x.split(',') if m.strip()] if x.strip() else [])
gt['match_count'] = gt['match_list'].apply(len)
print(f'\nMatch count distribution:')
print(gt['match_count'].value_counts().sort_index().head(10))
print(f'\nSingletons (0 matches): {(gt["match_count"] == 0).sum():,}')
print(f'Singleton fraction: {(gt["match_count"] == 0).mean():.4f}')
print(f'Total match pairs: {gt["match_count"].sum():,}')
print(f'Mean matches per S1: {gt["match_count"].mean():.3f}')

# Country distribution from S1 train
print('\n=== COUNTRY DISTRIBUTION ===')
s1 = pd.read_csv(os.path.join(base, 'train/train_source1.tsv'), sep='\t', dtype=str)
print('Train S1:', s1['country'].value_counts().to_dict())

s1_test = pd.read_csv(os.path.join(base, 'test/test_source1.tsv'), sep='\t', dtype=str)
print('Test  S1:', s1_test['country'].value_counts().to_dict())

# Sample matched pairs
print('\n=== SAMPLE MATCHED PAIRS ===')
s2 = pd.read_csv(os.path.join(base, 'train/train_source2.tsv'), sep='\t', dtype=str, nrows=500000)
s3 = pd.read_csv(os.path.join(base, 'train/train_source3.tsv'), sep='\t', dtype=str, nrows=500000)
s23 = pd.concat([s2, s3]).set_index('entity_id')

matched_gt = gt[gt['match_count'] > 0].head(30)
s1_idx = s1.set_index('entity_id')

count = 0
for _, row in matched_gt.iterrows():
    s1_id = row.iloc[0]
    for mid in row['match_list'][:1]:
        if s1_id in s1_idx.index and mid in s23.index:
            s1_rec = s1_idx.loc[s1_id]
            s23_rec = s23.loc[mid]
            print(f'  S1: {s1_rec["business_name"]:50s} | {mid}: {s23_rec["business_name"]}')
            print(f'      {s1_rec["business_address"][:50]:50s} |       {str(s23_rec["business_address"])[:50]}')
            count += 1
            if count >= 10:
                break
    if count >= 10:
        break

print('\nDone.')
