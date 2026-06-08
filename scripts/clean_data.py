import pandas as pd
import os

all_dfs = []

# ── 1. big_startup_secsees_dataset.csv ──────────────────────────
df1 = pd.read_csv('data/raw/big_startup_secsees_dataset.csv')
df1 = df1.rename(columns={
    'name': 'name',
    'category_list': 'category',
    'funding_total_usd': 'funding',
    'status': 'status',
    'founded_at': 'year'
})
df1['description'] = df1['category'].fillna('')
df1['failure_reason'] = ''
df1['source'] = 'crunchbase'
df1 = df1[['name', 'description', 'category', 'status', 'failure_reason', 'year', 'source']]
all_dfs.append(df1)
print(f"✓ crunchbase: {len(df1)} rows")

# ── 2. Sector-based failure CSVs (all same format) ───────────────
sector_files = [
    'Startup Failure (Finance and Insurance).csv',
    'Startup Failure (Food and services).csv',
    'Startup Failure (Health Care).csv',
    'Startup Failure (Manufactures).csv',
    'Startup Failure (Retail Trade).csv',
    'Startup Failures (Information Sector).csv',
]

for fname in sector_files:
    path = os.path.join('data/raw', fname)
    try:
        df = pd.read_csv(path)
        df = df.rename(columns={
            'Name': 'name',
            'Sector': 'category',
            'What They Did': 'description',
            'Why They Failed': 'failure_reason'
        })
        df['status'] = 'failed'
        df['year'] = ''
        df['source'] = 'startup_failures_dataset'
        df = df[['name', 'description', 'category', 'status', 'failure_reason', 'year', 'source']]
        all_dfs.append(df)
        print(f"✓ {fname}: {len(df)} rows")
    except Exception as e:
        print(f"✗ Error on {fname}: {e}")

# ── 3. startup_failure_prediction.csv ───────────────────────────
df3 = pd.read_csv('data/raw/startup_failure_prediction.csv')
df3 = df3.rename(columns={
    'Startup_Name': 'name',
    'Industry': 'category',
    'Startup_Status': 'status'
})
df3['description'] = df3['category'].fillna('')
df3['failure_reason'] = ''
df3['year'] = ''
df3['source'] = 'failure_prediction_dataset'
df3 = df3[['name', 'description', 'category', 'status', 'failure_reason', 'year', 'source']]
all_dfs.append(df3)
print(f"✓ startup_failure_prediction: {len(df3)} rows")

# ── 4. investments_VC.csv (encoding fix) ────────────────────────
try:
    df4 = pd.read_csv('data/raw/investments_VC.csv', encoding='latin-1')
    df4 = df4.rename(columns={
        'name': 'name',
        'category_list': 'category',
        'status': 'status',
        'founded_at': 'year'
    })
    df4['description'] = df4['category'].fillna('')
    df4['failure_reason'] = ''
    df4['source'] = 'investments_vc'
    df4 = df4[['name', 'description', 'category', 'status', 'failure_reason', 'year', 'source']]
    all_dfs.append(df4)
    print(f"✓ investments_VC: {len(df4)} rows")
except Exception as e:
    print(f"✗ investments_VC skipped: {e}")

# ── 5. Combine everything ────────────────────────────────────────
combined = pd.concat(all_dfs, ignore_index=True)

# Standardize status values
combined['status'] = combined['status'].str.lower().str.strip()
combined['status'] = combined['status'].replace({
    'closed': 'failed',
    'shutdown': 'failed',
    'operating': 'active',
    'acquired': 'acquired',
    'ipo': 'active'
})

# Drop rows with no name or description
combined = combined.dropna(subset=['name'])
combined['description'] = combined['description'].fillna('')
combined['failure_reason'] = combined['failure_reason'].fillna('')

# Clean text
combined['name'] = combined['name'].str.strip()
combined['description'] = combined['description'].str.strip()

# Save
os.makedirs('data/processed', exist_ok=True)
combined.to_csv('data/processed/startups_clean.csv', index=False)

print(f"\n{'='*40}")
print(f"TOTAL ROWS: {len(combined)}")
print(f"Status breakdown:\n{combined['status'].value_counts()}")
print(f"\n✓ Saved to data/processed/startups_clean.csv")