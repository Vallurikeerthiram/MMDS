"""
Generate Complete Excel Workbook for MMDS Part C
================================================
This script generates a comprehensive single Excel file:
'complete_movie_pairwise_similarity_analysis.xlsx'
(and .csv counterpart) containing:

For every movie (185 movies):
- movie_title, release_year, primary_genre, language, production_scale
- review_id_1 (Earliest review), review_1_date, review_1_rating, review_1_word_count
- review_id_2 (Latest review), review_2_date, review_2_rating, review_2_word_count
- rating_difference (review_2_rating - review_1_rating)
- rating_trend (Positive Canonization / Negative Hype Decay / Stable)
- word_count_difference (review_2_words - review_1_words)
- char3_jaccard_similarity & char3_jaccard_distance
- word2_jaccard_similarity & word2_jaccard_distance
- unigram_jaccard_similarity & unigram_jaccard_distance
- tfidf_cosine_similarity & tfidf_cosine_distance
- minhash_estimated_similarity (n=100)
- lsh_candidate_collision (b=20, r=5, threshold ~= 0.549: YES / NO)
- discourse_drift_status (High Semantic Drift / Stable Discourse)
- review_1_snippet & review_2_snippet

Includes additional sheets in Excel:
1. 'Movie Pairwise Analysis' (The complete 185-movie master table)
2. 'Genre Drift Aggregation' (Drift and distances by genre)
3. 'LSH S-Curve & Theory' (Mathematical probability table and parameters)
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
    clean = clean_text(text)
    if len(clean) < k:
        return set([clean]) if clean else set()
    return set(clean[i:i+k] for i in range(len(clean) - k + 1))

def get_word_shingles(text, k=2):
    words = clean_text(text).split()
    if len(words) < k:
        return set([' '.join(words)]) if words else set()
    return set(' '.join(words[i:i+k]) for i in range(len(words) - k + 1))

def compute_jaccard(set1, set2):
    if not set1 and not set2:
        return 1.0, 0.0
    union = set1.union(set2)
    if not union:
        return 0.0, 1.0
    inter = set1.intersection(set2)
    sim = len(inter) / len(union)
    return sim, 1.0 - sim

class MinHashLSHProcessor:
    def __init__(self, num_perm=100, num_bands=20, prime=4294967311, seed=42):
        self.n = num_perm
        self.b = num_bands
        self.r = num_perm // num_bands
        self.prime = prime
        self.threshold = (1.0 / self.b) ** (1.0 / self.r)
        
        rng = np.random.default_rng(seed)
        self.a_coeffs = rng.integers(1, self.prime - 1, size=self.n, dtype=np.int64)
        self.b_coeffs = rng.integers(0, self.prime - 1, size=self.n, dtype=np.int64)

    def compute_signature(self, shingles_set):
        if not shingles_set:
            return np.zeros(self.n, dtype=np.int64)
        shingle_hashes = np.array([hash(s) & 0xFFFFFFFF for s in shingles_set], dtype=np.int64)
        signature = np.full(self.n, np.iinfo(np.int64).max, dtype=np.int64)
        for i in range(self.n):
            a = self.a_coeffs[i]
            b = self.b_coeffs[i]
            hashed_values = (a * shingle_hashes + b) % self.prime
            signature[i] = np.min(hashed_values)
        return signature

    def check_lsh_collision(self, sig1, sig2):
        for band_idx in range(self.b):
            start = band_idx * self.r
            end = start + self.r
            if np.array_equal(sig1[start:end], sig2[start:end]):
                return "YES (Collided)"
        return "NO (Pruned)"

def main():
    print("=" * 80)
    print("BUILDING COMPLETE EXCEL WORKBOOK ACROSS ALL 185 MOVIES")
    print("=" * 80)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'letterboxd_final_dataset.csv')
    df = pd.read_csv(data_path)
    df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce')
    df['clean_lang'] = df['language'].fillna('Unknown').astype(str).str.replace('\xa0', ' ')
    df = df.sort_values(['movie_title', 'review_date'])

    lsh_processor = MinHashLSHProcessor(num_perm=100, num_bands=20, seed=42)

    rows = []

    for movie_title, group in df.groupby('movie_title'):
        if len(group) < 2:
            continue

        # Review 1: Earliest chronologically
        r1 = group.iloc[0]
        # Review 2: Latest chronologically
        r2 = group.iloc[-1]

        t1 = str(r1['review_text'])
        t2 = str(r2['review_text'])

        # 1. Shingle extractions
        s1_char3 = get_char_shingles(t1, k=3)
        s2_char3 = get_char_shingles(t2, k=3)
        char3_sim, char3_dist = compute_jaccard(s1_char3, s2_char3)

        s1_word2 = get_word_shingles(t1, k=2)
        s2_word2 = get_word_shingles(t2, k=2)
        word2_sim, word2_dist = compute_jaccard(s1_word2, s2_word2)

        s1_words = set(clean_text(t1).split())
        s2_words = set(clean_text(t2).split())
        word_sim, word_dist = compute_jaccard(s1_words, s2_words)

        # 2. TF-IDF Cosine
        try:
            vec = TfidfVectorizer(stop_words='english')
            mat = vec.fit_transform([t1, t2])
            cos_sim = float(cosine_similarity(mat[0:1], mat[1:2])[0][0])
        except Exception:
            cos_sim = 0.0
        cos_dist = 1.0 - cos_sim

        # 3. MinHash & LSH
        sig1 = lsh_processor.compute_signature(s1_char3)
        sig2 = lsh_processor.compute_signature(s2_char3)
        minhash_est_sim = float(np.mean(sig1 == sig2))
        lsh_collision = lsh_processor.check_lsh_collision(sig1, sig2)

        # 4. Rating & Word dynamics
        rating_1 = float(r1['rating'])
        rating_2 = float(r2['rating'])
        r_diff = rating_2 - rating_1

        if r_diff > 0.0:
            trend = "Positive (Canonization)"
        elif r_diff < 0.0:
            trend = "Negative (Hype Decay)"
        else:
            trend = "Identical Consensus"

        words_1 = int(r1['review_word_count'])
        words_2 = int(r2['review_word_count'])
        w_diff = words_2 - words_1

        discourse_shift = "High Semantic Drift" if cos_dist > 0.80 else "Stable Discourse"

        rows.append({
            'movie_name': movie_title,
            'release_year': int(r1['release_year']),
            'primary_genre': str(r1['primary_genre']),
            'language': str(r1['clean_lang']),
            'production_scale': str(r1['production_scale']),
            'review_id_1': str(r1['review_id']),
            'review_1_date': str(r1['review_date'].date()),
            'review_1_rating': rating_1,
            'review_1_word_count': words_1,
            'review_id_2': str(r2['review_id']),
            'review_2_date': str(r2['review_date'].date()),
            'review_2_rating': rating_2,
            'review_2_word_count': words_2,
            'rating_difference': round(r_diff, 1),
            'rating_trend': trend,
            'word_count_difference': w_diff,
            'char3_jaccard_similarity': round(char3_sim, 4),
            'char3_jaccard_distance': round(char3_dist, 4),
            'word2_jaccard_similarity': round(word2_sim, 4),
            'word2_jaccard_distance': round(word2_dist, 4),
            'word_unigram_similarity': round(word_sim, 4),
            'word_unigram_distance': round(word_dist, 4),
            'tfidf_cosine_similarity': round(cos_sim, 4),
            'tfidf_cosine_distance': round(cos_dist, 4),
            'minhash_estimated_similarity': round(minhash_est_sim, 4),
            'lsh_candidate_collision': lsh_collision,
            'discourse_shift': discourse_shift,
            'review_1_snippet': t1[:100] + "...",
            'review_2_snippet': t2[:100] + "..."
        })

    master_df = pd.DataFrame(rows)
    
    # Sort strictly by movie_name, then review_1_date, then review_2_date
    master_df['r1_dt'] = pd.to_datetime(master_df['review_1_date'])
    master_df['r2_dt'] = pd.to_datetime(master_df['review_2_date'])
    master_df = master_df.sort_values(by=['movie_name', 'r1_dt', 'r2_dt'], ascending=[True, True, True]).reset_index(drop=True)
    master_df = master_df.drop(columns=['r1_dt', 'r2_dt'])

    # Build multi-sheet Excel workbook
    excel_path = os.path.join(script_dir, 'complete_movie_pairwise_similarity_analysis.xlsx')
    
    # Genre sheet
    genre_df = master_df.groupby('primary_genre').agg(
        total_movies=('movie_name', 'count'),
        avg_rating_diff=('rating_difference', 'mean'),
        avg_word_count_diff=('word_count_difference', 'mean'),
        avg_char3_distance=('char3_jaccard_distance', 'mean'),
        avg_word2_distance=('word2_jaccard_distance', 'mean'),
        avg_unigram_distance=('word_unigram_distance', 'mean'),
        avg_cosine_distance=('tfidf_cosine_distance', 'mean')
    ).round(4).sort_values('avg_rating_diff', ascending=False).reset_index()

    # LSH S-Curve sheet
    s_vals = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.5493, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
    lsh_data = []
    b = lsh_processor.b
    r = lsh_processor.r
    thresh = lsh_processor.threshold
    for s in s_vals:
        p = 1.0 - (1.0 - s ** r) ** b
        filt = "Filtered Out (>99% rejected)" if p < 0.05 else ("Transition Zone" if p < 0.95 else "Candidate Pair Collision")
        lsh_data.append({
            'Jaccard_Similarity_s': s,
            'Band_Collision_Prob_P': round(p, 4),
            'Filtering_Status': filt,
            'Parameters': f"b={b}, r={r}, n={b*r}, t~={thresh:.4f}"
        })
    lsh_df = pd.DataFrame(lsh_data)

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        master_df.to_excel(writer, sheet_name='Movie Pairwise Analysis', index=False)
        genre_df.to_excel(writer, sheet_name='Genre Drift Summary', index=False)
        lsh_df.to_excel(writer, sheet_name='LSH S-Curve & Theory', index=False)

    print(f"[OK] Master Excel Workbook saved: {excel_path}")
    print("\nMaster Table Overview:")
    print(f"Columns ({len(master_df.columns)}): {list(master_df.columns)[:8]}...")
    print(f"Total Rows: {len(master_df)}")

if __name__ == '__main__':
    main()
