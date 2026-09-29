"""
Macro Temporal Drift & Semantic Shift Analysis Across All Movies
================================================================
MMDS Part C Extension:
This script computes pairwise temporal similarity across all 185 movies in the
Letterboxd dataset (61,786 reviews).

For each movie:
1. Reviews are partitioned chronologically:
   - Early Corpus: First 35% of reviews by date.
   - Recent Corpus: Last 35% of reviews by date.
2. Computes Text Similarity & Distance:
   - TF-IDF Cosine Similarity & Cosine Distance (Semantic Drift)
   - Word Bag-of-Words Jaccard Similarity & Distance (Lexical Drift)
   - Character 3-Shingle Jaccard Similarity & Distance (Morphological Drift)
3. Computes Rating Drift:
   - Delta Rating = Mean(Recent Ratings) - Mean(Early Ratings)
   - Categorizes into: Positive Drift (Canonization), Negative Drift (Hype Decay), or Stable.
4. Aggregates findings by:
   - Primary Genre
   - Spoken Language
   - Cinematic Era (Pre-1970 Golden Era, 1970-1999 Classic Era, 2000-2024 Modern)
5. Exports clean CSV deliverables for academic reporting.
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def clean_text(text):
    if not isinstance(text, str):
        return ""
    clean = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    return ' '.join(clean.split())

def get_char_shingles(text, k=3):
    cleaned = clean_text(text)
    if len(cleaned) < k:
        return set([cleaned]) if cleaned else set()
    return set(cleaned[i:i+k] for i in range(len(cleaned) - k + 1))

def compute_jaccard(set1, set2):
    if not set1 and not set2:
        return 1.0, 0.0
    union = set1.union(set2)
    if not union:
        return 0.0, 1.0
    inter = set1.intersection(set2)
    sim = len(inter) / len(union)
    return sim, 1.0 - sim

def get_era(year):
    if year < 1970:
        return "Golden Era (Pre-1970)"
    elif year < 2000:
        return "Classic Era (1970-1999)"
    else:
        return "Modern Era (2000-2024)"

def main():
    print("=" * 80)
    print("RUNNING MACRO TEMPORAL DRIFT & SIMILARITY ANALYSIS ACROSS ALL 185 MOVIES")
    print("=" * 80)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'letterboxd_final_dataset.csv')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce')
    df['clean_lang'] = df['language'].fillna('Unknown').astype(str).str.replace('\xa0', ' ')
    df = df.sort_values(['movie_title', 'review_date'])

    print(f"Loaded {len(df):,} total reviews across {df['movie_title'].nunique()} unique films.")

    movie_records = []

    for movie_title, group in df.groupby('movie_title'):
        total_revs = len(group)
        if total_revs < 10:
            continue

        split_n = max(5, int(total_revs * 0.35))
        early_df = group.iloc[:split_n]
        recent_df = group.iloc[-split_n:]

        # Rating Metrics
        r_early = float(early_df['rating'].mean())
        r_recent = float(recent_df['rating'].mean())
        r_diff = r_recent - r_early

        # Dates
        d_early_start = str(early_df['review_date'].min().date())
        d_early_end = str(early_df['review_date'].max().date())
        d_recent_start = str(recent_df['review_date'].min().date())
        d_recent_end = str(recent_df['review_date'].max().date())

        # Review length shift
        words_early = float(early_df['review_word_count'].mean())
        words_recent = float(recent_df['review_word_count'].mean())
        words_diff = words_recent - words_early

        # Text Aggregation
        early_corpus = ' '.join(early_df['review_text'].dropna().tolist())
        recent_corpus = ' '.join(recent_df['review_text'].dropna().tolist())

        # 1. Cosine Distance on TF-IDF
        vec = TfidfVectorizer(stop_words='english', max_features=4000)
        try:
            tfidf = vec.fit_transform([early_corpus, recent_corpus])
            cos_sim = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
        except Exception:
            cos_sim = 0.5
        cos_dist = 1.0 - cos_sim

        # 2. Word-Level Jaccard
        w_early = set(clean_text(early_corpus).split())
        w_recent = set(clean_text(recent_corpus).split())
        word_jaccard_sim, word_jaccard_dist = compute_jaccard(w_early, w_recent)

        # 3. Char 3-Shingle Jaccard
        s_early = get_char_shingles(early_corpus[:50000], k=3)
        s_recent = get_char_shingles(recent_corpus[:50000], k=3)
        char_jaccard_sim, char_jaccard_dist = compute_jaccard(s_early, s_recent)

        # Categorize Rating Drift
        if r_diff >= 0.10:
            drift_type = "Positive (Canonization)"
        elif r_diff <= -0.10:
            drift_type = "Negative (Hype Decay)"
        else:
            drift_type = "Stable Consensual"

        rel_year = int(group['release_year'].iloc[0])
        genre = str(group['primary_genre'].iloc[0])
        lang = str(group['clean_lang'].iloc[0])

        movie_records.append({
            'movie_title': movie_title,
            'release_year': rel_year,
            'era': get_era(rel_year),
            'primary_genre': genre,
            'language': lang,
            'total_reviews': total_revs,
            'early_period': f"{d_early_start} to {d_early_end}",
            'recent_period': f"{d_recent_start} to {d_recent_end}",
            'mean_early_rating': round(r_early, 3),
            'mean_recent_rating': round(r_recent, 3),
            'rating_drift': round(r_diff, 3),
            'drift_direction': drift_type,
            'mean_early_words': round(words_early, 1),
            'mean_recent_words': round(words_recent, 1),
            'word_count_drift': round(words_diff, 1),
            'cosine_similarity': round(cos_sim, 4),
            'cosine_distance': round(cos_dist, 4),
            'word_jaccard_similarity': round(word_jaccard_sim, 4),
            'word_jaccard_distance': round(word_jaccard_dist, 4),
            'char3_jaccard_similarity': round(char_jaccard_sim, 4),
            'char3_jaccard_distance': round(char_jaccard_dist, 4)
        })

    m_df = pd.DataFrame(movie_records)

    # Export full movie-level drift CSV
    m_csv = os.path.join(script_dir, 'movie_temporal_drift_full.csv')
    m_df.to_csv(m_csv, index=False)
    print(f"\n[OK] Movie-level metrics saved: {m_csv}")

    # Genre aggregation
    genre_summary = m_df.groupby('primary_genre').agg(
        movie_count=('movie_title', 'count'),
        mean_rating_drift=('rating_drift', 'mean'),
        mean_cosine_dist=('cosine_distance', 'mean'),
        mean_word_jaccard_dist=('word_jaccard_distance', 'mean'),
        mean_char3_jaccard_dist=('char3_jaccard_distance', 'mean'),
        mean_word_count_drift=('word_count_drift', 'mean')
    ).round(4).sort_values('mean_rating_drift', ascending=False).reset_index()

    g_csv = os.path.join(script_dir, 'genre_drift_summary.csv')
    genre_summary.to_csv(g_csv, index=False)
    print(f"[OK] Genre-level summary saved: {g_csv}")

    # Language aggregation
    lang_summary = m_df.groupby('language').agg(
        movie_count=('movie_title', 'count'),
        mean_rating_drift=('rating_drift', 'mean'),
        mean_cosine_dist=('cosine_distance', 'mean'),
        mean_word_jaccard_dist=('word_jaccard_distance', 'mean')
    ).round(4).sort_values('movie_count', ascending=False).reset_index()

    l_csv = os.path.join(script_dir, 'language_drift_summary.csv')
    lang_summary.to_csv(l_csv, index=False)
    print(f"[OK] Language-level summary saved: {l_csv}")

    # Era aggregation
    era_summary = m_df.groupby('era').agg(
        movie_count=('movie_title', 'count'),
        mean_rating_drift=('rating_drift', 'mean'),
        mean_cosine_dist=('cosine_distance', 'mean'),
        mean_word_jaccard_dist=('word_jaccard_distance', 'mean'),
        mean_word_count_drift=('word_count_drift', 'mean')
    ).round(4).sort_values('mean_rating_drift', ascending=False).reset_index()

    e_csv = os.path.join(script_dir, 'era_drift_summary.csv')
    era_summary.to_csv(e_csv, index=False)
    print(f"[OK] Era-level summary saved: {e_csv}")

    # Console Highlights
    print("\n" + "=" * 80)
    print("KEY FINDINGS: TOP POSITIVE DRIFT (CANONIZATION / RISING CULT APPRECIATION)")
    print("=" * 80)
    pos_top = m_df.sort_values('rating_drift', ascending=False).head(5)
    print(pos_top[['movie_title', 'release_year', 'primary_genre', 'mean_early_rating', 'mean_recent_rating', 'rating_drift', 'cosine_distance']].to_string(index=False))

    print("\n" + "=" * 80)
    print("KEY FINDINGS: TOP NEGATIVE DRIFT (HYPE DECAY / RETROSPECTIVE SCRUTINY)")
    print("=" * 80)
    neg_top = m_df.sort_values('rating_drift').head(5)
    print(neg_top[['movie_title', 'release_year', 'primary_genre', 'mean_early_rating', 'mean_recent_rating', 'rating_drift', 'cosine_distance']].to_string(index=False))

    print("\n" + "=" * 80)
    print("GENRE-LEVEL DRIFT SUMMARY")
    print("=" * 80)
    print(genre_summary.to_string(index=False))

    print("\n" + "=" * 80)
    print("ERA-LEVEL DRIFT SUMMARY")
    print("=" * 80)
    print(era_summary.to_string(index=False))

if __name__ == '__main__':
    main()
