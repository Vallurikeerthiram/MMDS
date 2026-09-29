import pandas as pd
import unicodedata
import os
import re

in_path = 'data/letterboxd_final_dataset.csv'
df = pd.read_csv(in_path)
initial_len = len(df)
print(f"Starting with {initial_len} rows")

# 1. Prune rows with Show All
mask_bad = df['genres'].astype(str).str.contains('Show All', case=False, na=False) | \
           df['primary_genre'].astype(str).str.contains('Show All', case=False, na=False)

# 2. Prune rows with non-genre theme tags
theme_tags = ['Dreamlike', 'Gripping', 'Religious faith', 'Brutal', 'Terrifying', 'patriotism']
mask_bad = mask_bad | df['primary_genre'].isin(theme_tags)

# 3. Prune rows with any NaN or empty strings
mask_bad = mask_bad | df.isna().any(axis=1)
mask_bad = mask_bad | (df['review_text'].astype(str).str.strip() == '')

df_clean = df[~mask_bad].copy()
print(f"Rows after dropping missing/artifact rows: {len(df_clean)} (dropped {initial_len - len(df_clean)} rows)")

# 4. Text cleaning function
def clean_text(t):
    if not isinstance(t, str):
        return ''
    # Normalize unicode (NFKD decomposes mathematical script fonts into standard latin characters)
    t = unicodedata.normalize('NFKD', t)
    # Fix broken replacement characters
    t = t.replace('\ufffd', "'")
    # Fix repeated quadruple quotes
    t = re.sub(r'"{2,}', '"', t)
    return t.strip()

df_clean['review_text'] = df_clean['review_text'].apply(clean_text)
df_clean['movie_title'] = df_clean['movie_title'].apply(lambda x: x.replace('\ufffd', '...').strip())

# Recalculate word and char counts accurately
df_clean['review_char_count'] = df_clean['review_text'].str.len()
df_clean['review_word_count'] = df_clean['review_text'].str.split().str.len()

# Re-index review_id sequentially
df_clean['review_id'] = [f"REV_{i+1:06d}" for i in range(len(df_clean))]

out_csv = 'data/letterboxd_final_dataset.csv'
out_parquet = 'data/letterboxd_final_dataset.parquet'
df_clean.to_csv(out_csv, index=False, encoding='utf-8')
df_clean.to_parquet(out_parquet, index=False)

print(f"Successfully saved {out_csv} ({os.path.getsize(out_csv)/(1024*1024):.2f} MB)")
print(f"Successfully saved {out_parquet} ({os.path.getsize(out_parquet)/(1024*1024):.2f} MB)")
print(f"Final Row count: {len(df_clean)}")
print(f"Final Unique Movies count: {df_clean['movie_title'].nunique()}")
print("Primary genres distribution:")
print(df_clean['primary_genre'].value_counts())
