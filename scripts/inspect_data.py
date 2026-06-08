import pandas as pd
import os

raw_path = 'data/raw/'

for file in os.listdir(raw_path):
    if file.endswith('.csv'):
        try:
            df = pd.read_csv(os.path.join(raw_path, file), nrows=3)
            print(f"\n{'='*50}")
            print(f"FILE: {file}")
            print(f"Rows (sample): {df.shape[0]} | Columns: {df.shape[1]}")
            print(f"Columns: {list(df.columns)}")
        except Exception as e:
            print(f"ERROR reading {file}: {e}")