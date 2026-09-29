import pandas as pd
import linecache
import sys
sys.stdout.reconfigure(encoding='utf-8')

f_path = 'data/letterboxd_final_dataset.csv'
df = pd.read_csv(f_path)

pairs = [
    (6458, 6459),
    (25141, 25142),
    (27529, 27530),
    (42209, 42210),
    (50209, 50210),
    (50651, 50652)
]

print("=== 1. CHECKING BY PHYSICAL CSV LINE NUMBER ===")
for s, e in pairs:
    print(f"\n*** CSV Lines {s} to {e} ***")
    for l in range(s, e + 1):
        raw = linecache.getline(f_path, l).strip()
        print(f"Line {l}: {raw[:130]}...")

print("\n\n=== 2. DETAILED CELL-BY-CELL CHECK BY DATAFRAME ILOC ===")
# In CSV, row 2 is iloc[0], so line L is iloc[L-2]. Let's check both iloc[s] and iloc[s-2]
for s, e in pairs:
    # Let's inspect the exact lines where line number in CSV is s
    idx = s - 2
    row = df.iloc[idx]
    print(f"\n=======================================================")
    print(f"CSV Line {s} -> Dataframe Index {idx} (Review ID: {row['review_id']})")
    print(f"Movie: {row['movie_title']} | Rating: {row['rating']} | Date: {row['review_date']}")
    print(f"Review Text: {repr(str(row['review_text'])[:120])}")
    print("ALL 21 COLUMNS AND THEIR VALUES:")
    for col in df.columns:
        val = row[col]
        is_empty = pd.isna(val) or str(val).strip() == ''
        flag = "[EMPTY!]" if is_empty else "[OK]"
        print(f"   {flag} {col}: {repr(val)}")
