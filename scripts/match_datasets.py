import os
import re
import unicodedata
import pandas as pd

def slugify(text):
    if not isinstance(text, str):
        return ''
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '-', text).strip('-')
    return text

def main():
    print("Loading datasets...")
    movies_df = pd.read_csv('data/letterboxd_movies_dataset.csv')
    reviews_df = pd.read_csv('data/letterboxd_250movie_reviews.csv')
    
    print(f"Movies metadata count: {len(movies_df)}")
    print(f"Reviews count: {len(reviews_df)}")
    
    movies_df['clean_slug'] = movies_df['title'].apply(slugify)
    
    movie_slug_to_row = {}
    for idx, row in movies_df.iterrows():
        slug = row['clean_slug']
        if slug not in movie_slug_to_row:
            movie_slug_to_row[slug] = row
            
    review_slugs = reviews_df['Movie'].dropna().unique()
    matched = []
    unmatched = []
    
    for s in review_slugs:
        if s in movie_slug_to_row:
            matched.append(s)
        else:
            unmatched.append(s)
            
    print(f"Direct normalized slug matches: {len(matched)} / {len(review_slugs)}")
    print(f"Unmatched: {len(unmatched)}")
    
    for u in unmatched:
        # fuzzy match or substring
        sub = u.replace('-', ' ')
        candidates = movies_df[movies_df['title'].str.lower().str.contains(sub[:min(len(sub), 8)], na=False)]['title'].head(3).tolist()
        print(f"Unmatched slug: '{u}' | Candidates: {candidates}")

if __name__ == '__main__':
    main()
