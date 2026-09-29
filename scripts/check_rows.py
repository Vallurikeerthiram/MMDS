import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')

for fname in ['data/letterboxd_final_dataset.csv', 'data/letterboxd_cleaned_dataset.csv']:
    print('**************************************************')
    print('FILE:', fname)
    df = pd.read_csv(fname)
    ranges = [
        ("798 (or 6797-6798)", 796, 800),
        ("6797-6798", 6795, 6800),
        ("21545-21546", 21543, 21548),
        ("29131-29133", 29129, 29135),
        ("31520-31521", 31518, 31523),
        ("47860-47861", 47858, 47863),
        ("57182-57183", 57180, 57185),
        ("57624-57625", 57622, 57627)
    ]
    for label, start, end in ranges:
        print(f'=== Range {label} (Idx {start} to {end}) ===')
        for idx in range(start, min(end, len(df))):
            row = df.iloc[idx]
            rev_id = row.get('review_id')
            title = row.get('movie_title')
            rating = row.get('rating', row.get('numeric_rating'))
            txt = str(row.get('review_text'))[:120].replace('\n', ' ')
            print(f'  [Idx {idx} / Line {idx+2}] ID: {rev_id} | Title: {title} | Rating: {rating} | Text: {txt}')
