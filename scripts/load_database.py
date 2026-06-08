import pandas as pd
import sqlite3
import os

df = pd.read_csv('data/processed/startups_clean.csv')

os.makedirs('database', exist_ok=True)
conn = sqlite3.connect('database/startups.db')

df.to_sql('startups', conn, if_exists='replace', index=False)

# Verify
count = conn.execute("SELECT COUNT(*) FROM startups").fetchone()[0]
failed = conn.execute("SELECT COUNT(*) FROM startups WHERE status='failed'").fetchone()[0]
active = conn.execute("SELECT COUNT(*) FROM startups WHERE status='active'").fetchone()[0]

print(f"✓ Database created: database/startups.db")
print(f"  Total rows:  {count}")
print(f"  Failed:      {failed}")
print(f"  Active:      {active}")

conn.close()