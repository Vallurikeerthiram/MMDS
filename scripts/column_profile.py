import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('data/letterboxd_final_dataset.csv')

print(f"Total Records: {len(df)}")
print("=" * 110)
print(f"{'#':<3} | {'Column Name':<20} | {'Distinct':<10} | {'Data Type':<10} | {'Range / Specific Values'}")
print("=" * 110)

for i, col in enumerate(df.columns, 1):
    n_unique = df[col].nunique()
    dtype = str(df[col].dtype)
    
    if pd.api.types.is_numeric_dtype(df[col]):
        col_min = df[col].min()
        col_max = df[col].max()
        if col in ['release_year']:
            details = f"Range: [{int(col_min)} to {int(col_max)}]"
        elif col in ['runtime', 'genre_count', 'review_word_count', 'review_char_count']:
            details = f"Range: [{int(col_min)} to {int(col_max)}], Mean: {df[col].mean():.1f}"
        else:
            details = f"Range: [{col_min} to {col_max}], Mean: {df[col].mean():.2f}"
    elif col == 'review_date':
        details = f"Range: [{df[col].min()} to {df[col].max()}]"
    elif n_unique <= 6:
        vals = df[col].unique().tolist()
        details = f"Categories: {vals}"
    elif n_unique <= 20:
        vals = df[col].unique().tolist()
        details = f"All {n_unique} values: {vals}"
    else:
        details = f"{n_unique} distinct entries (e.g., '{df[col].iloc[0]}', '{df[col].iloc[1]}')"
        
    print(f"{i:<3} | {col:<20} | {n_unique:<10} | {dtype:<10} | {details}")
