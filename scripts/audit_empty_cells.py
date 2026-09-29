import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')

for fname in ['data/letterboxd_final_dataset.csv', 'data/letterboxd_cleaned_dataset.csv', 'data/letterboxd_unified_dataset.csv']:
    print('==================================================')
    print(f'AUDITING FILE: {fname}')
    df = pd.read_csv(fname)
    print(f'Total rows: {len(df)}, Total columns: {len(df.columns)}')
    
    issues_found = 0
    for col in df.columns:
        s = df[col]
        null_count = int(s.isna().sum())
        s_str = s.astype(str).str.strip()
        empty_str_count = int((s_str == '').sum())
        literal_nan_count = int(s_str.str.lower().isin(['nan', 'none', 'null', 'undefined', 'n/a', 'na', '?']).sum())
        
        if null_count > 0 or empty_str_count > 0 or literal_nan_count > 0:
            print(f'  --> Column "{col}": nulls={null_count}, empty_str={empty_str_count}, literal_nan_words={literal_nan_count}')
            issues_found += (null_count + empty_str_count)
            
    if issues_found == 0:
        print('  ==> RESULT: ABSOLUTELY 0 NULL OR EMPTY CELLS FOUND!')
    else:
        print(f'  ==> RESULT: Found {issues_found} total empty/null cells.')
