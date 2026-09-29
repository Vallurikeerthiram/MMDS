import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')

f_path = 'data/letterboxd_final_dataset.csv'
df = pd.read_csv(f_path)

for s in [6458, 6459, 25141, 25142]:
    idx = s - 2
    row = df.iloc[idx]
    print(f"=== CSV Line {s} -> Index {idx} (ID: {row['review_id']}) ===")
    print(f"Movie: {row['movie_title']} | Rating: {row['rating']} | Date: {row['review_date']}")
    print(f"Review Text: {repr(str(row['review_text'])[:120])}")
    for c in df.columns:
        val = row[c]
        is_empty = pd.isna(val) or str(val).strip() == ''
        print(f"  [{'EMPTY' if is_empty else 'OK'}] {c}: {repr(val)}")
