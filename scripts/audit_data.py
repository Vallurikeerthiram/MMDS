import pandas as pd

df = pd.read_csv('data/letterboxd_final_dataset.csv')
print(f"Total rows currently: {len(df)}")

# Check for rows with Show All
bad_genres = ['Show All', 'Show All…', 'Show All']
mask_show_all = df['genres'].astype(str).str.contains('Show All', case=False, na=False) | \
                df['primary_genre'].astype(str).str.contains('Show All', case=False, na=False)
print(f"Rows with Show All: {mask_show_all.sum()}")

# Check for non-genre themes
theme_tags = ['Dreamlike', 'Gripping', 'Religious faith', 'Brutal', 'Terrifying', 'patriotism']
mask_themes = df['primary_genre'].isin(theme_tags)
print(f"Rows with theme tags in primary_genre: {mask_themes.sum()}")

# Check for missing values across all columns
mask_na = df.isna().any(axis=1)
print(f"Rows with NA in any column: {mask_na.sum()}")

# Check for empty review text or empty values
mask_empty_text = df['review_text'].astype(str).str.strip() == ''
print(f"Rows with empty review text: {mask_empty_text.sum()}")
