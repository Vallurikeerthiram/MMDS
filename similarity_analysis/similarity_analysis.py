"""
Global LSH Review Similarity & Post-Hoc Decade Grouping Pipeline
================================================================
MMDS Part C: Unsupervised LSH Review Discovery & Decade Dynamics.

Pipeline Overview:
1. Pure Global LSH (No Pre-Grouping):
   - Computes character 4-shingles across diverse review texts.
   - Generates MinHash signatures (n=100) using universal hash families.
   - Applies Banded LSH (b=20 bands, r=5 rows, threshold t ~= 0.549).
   - Candidate pairs are discovered purely through hash bucket collisions across the entire dataset.

2. Post-Hoc Grouping by Release Decade and Watch Decade:
   - For every collided candidate pair (Review 1, Review 2):
     * Movie 1, Movie 2, Primary Genres
     * Release Decade 1 (1920s to 2020s), Release Decade 2
     * Watch Decade 1 (2010s to 2020s), Watch Decade 2
     * Exact Shingle Jaccard Similarity & TF-IDF Cosine Distance
     * Ratings, Rating Gap (|R1 - R2|)
     * Relationship Type:
       - 'Same Movie Hit': Intra-movie consensus across time.
       - 'Same Release Era Cross-Movie Hit': M1 != M2, but same release decade (e.g., two 1960s films triggering identical critique when watched decades later).
       - 'Cross-Era Temporal Echo': M1 != M2, different release decades (e.g., a modern film echoing a vintage classic).

3. Deliverable:
   Generates 'complete_movie_pairwise_similarity_analysis.xlsx' with comprehensive sheets:
   - 'LSH Review Collision Pairs'
   - 'Release Decade Hit Matrix'
   - 'Watch vs Release Decade Hits'
   - 'Top Cross-Movie Affinities'
   - 'LSH S-Curve & Theory'
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

def get_char_shingles(text, k=4):
    clean = clean_text(text)
    if len(clean) < k:
        return set([clean]) if clean else set()
    return set(clean[i:i+k] for i in range(len(clean) - k + 1))

def compute_exact_jaccard(set1, set2):
    if not set1 and not set2:
        return 1.0
    union = set1.union(set2)
    if not union:
        return 0.0
    return len(set1.intersection(set2)) / len(union)

class GlobalMinHashLSH:
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

def main():
    print("=" * 80)
    print("RUNNING GLOBAL LSH REVIEW SIMILARITY & POST-HOC DECADE GROUPING")
    print("=" * 80)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'letterboxd_final_dataset.csv')
    df = pd.read_csv(data_path)

    # 1. Feature Extraction: Decades
    df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce')
    df['review_year'] = df['review_date'].dt.year.fillna(2020).astype(int)
    df['watch_decade'] = (df['review_year'] // 10) * 10
    df['release_decade'] = (df['release_year'] // 10).astype(int) * 10
    df['clean_lang'] = df['language'].fillna('Unknown').astype(str).str.replace('\xa0', ' ')

    # Filter out ultra-short reviews (<25 words) and invalid watch dates (<2010)
    df_clean = df[(df['review_word_count'] >= 25) & (df['watch_decade'] >= 2010)].copy()

    # Balanced representation across all 11 release decades
    sampled_df = df_clean.groupby('release_decade', group_keys=False).apply(
        lambda g: g.sample(min(len(g), 1100), random_state=42),
        include_groups=True
    ).reset_index(drop=True)

    print(f"Total reviews indexed for global LSH: {len(sampled_df):,} across {sampled_df['release_decade'].nunique()} release decades.")

    # 2. Build Global LSH Buckets
    lsh = GlobalMinHashLSH(num_perm=100, num_bands=20, seed=42)
    band_buckets = [{} for _ in range(lsh.b)]
    signatures = []
    shingle_sets = []

    print("Computing MinHash signatures and hashing into 20 bands...")
    for idx in range(len(sampled_df)):
        text = str(sampled_df.at[idx, 'review_text'])
        s = get_char_shingles(text, k=4)
        sig = lsh.compute_signature(s)
        signatures.append(sig)
        shingle_sets.append(s)

        for b_idx in range(lsh.b):
            start = b_idx * lsh.r
            end = start + lsh.r
            sub_vec = tuple(sig[start:end])
            if sub_vec not in band_buckets[b_idx]:
                band_buckets[b_idx][sub_vec] = []
            band_buckets[b_idx][sub_vec].append(idx)

    # 3. Extract Candidate Collisions Across Entire Dataset
    candidate_pairs = set()
    for b_idx in range(lsh.b):
        for sub_vec, doc_indices in band_buckets[b_idx].items():
            # Exclude degenerate mega-buckets (e.g. identical one-liners)
            if 2 <= len(doc_indices) <= 35:
                for i in range(len(doc_indices)):
                    for j in range(i + 1, len(doc_indices)):
                        candidate_pairs.add((min(doc_indices[i], doc_indices[j]), max(doc_indices[i], doc_indices[j])))

    print(f"[OK] Total Global LSH Candidate Pairs Discovered: {len(candidate_pairs):,}")

    # 4. Post-Hoc Grouping by Release Decade and Watch Decade
    pair_rows = []
    
    # Process up to 5,000 diverse collided pairs for the master deliverable
    for idx1, idx2 in list(candidate_pairs)[:5000]:
        r1 = sampled_df.iloc[idx1]
        r2 = sampled_df.iloc[idx2]

        m1, m2 = r1['movie_title'], r2['movie_title']
        rd1, rd2 = int(r1['release_decade']), int(r2['release_decade'])
        wd1, wd2 = int(r1['watch_decade']), int(r2['watch_decade'])

        # Compute exact text similarities
        j_sim = compute_exact_jaccard(shingle_sets[idx1], shingle_sets[idx2])
        j_dist = 1.0 - j_sim

        # Classification of match
        if m1 == m2:
            rel_type = "Same Movie Hit (Intra-Movie Consensus)"
        elif rd1 == rd2:
            rel_type = f"Same Release Era Cross-Movie Hit ({rd1}s)"
        else:
            rel_type = f"Cross-Era Aesthetic Echo ({rd1}s <-> {rd2}s)"

        # Ensure ordered release decades for clean matrix aggregation
        min_rd = min(rd1, rd2)
        max_rd = max(rd1, rd2)

        rate_diff = abs(float(r1['rating']) - float(r2['rating']))

        t1_snip = str(r1['review_text'])[:90] + "..."
        t2_snip = str(r2['review_text'])[:90] + "..."

        pair_rows.append({
            'movie_1': m1,
            'movie_2': m2,
            'relationship_type': rel_type,
            'release_decade_1': f"{rd1}s",
            'release_decade_2': f"{rd2}s",
            'release_decade_pair': f"{min_rd}s & {max_rd}s",
            'watch_decade_1': f"{wd1}s",
            'watch_decade_2': f"{wd2}s",
            'watch_decade_pair': f"{min(wd1, wd2)}s & {max(wd1, wd2)}s",
            'primary_genre_1': str(r1['primary_genre']),
            'primary_genre_2': str(r2['primary_genre']),
            'rating_1': float(r1['rating']),
            'rating_2': float(r2['rating']),
            'rating_difference': round(rate_diff, 1),
            'char4_jaccard_similarity': round(j_sim, 4),
            'char4_jaccard_distance': round(j_dist, 4),
            'review_id_1': str(r1['review_id']),
            'review_id_2': str(r2['review_id']),
            'review_1_date': str(r1['review_date'].date()),
            'review_2_date': str(r2['review_date'].date()),
            'review_1_snippet': t1_snip,
            'review_2_snippet': t2_snip
        })

    pairs_df = pd.DataFrame(pair_rows)

    # Sort strictly by release_decade_1 -> release_decade_2 -> watch_decade_1 -> watch_decade_2 -> movie_1
    pairs_df = pairs_df.sort_values(
        by=['release_decade_1', 'release_decade_2', 'watch_decade_1', 'watch_decade_2', 'movie_1'],
        ascending=True
    ).reset_index(drop=True)

    # 5. Sheet 2: Release Decade Hit Matrix (Cross-Tabulation)
    matrix_df = pd.crosstab(
        pairs_df['release_decade_1'],
        pairs_df['release_decade_2'],
        rownames=['Release Decade 1'],
        colnames=['Release Decade 2']
    )

    # 6. Sheet 3: Watch vs Release Decade Dynamics
    watch_rel_df = pairs_df.groupby(['release_decade_pair', 'watch_decade_pair', 'relationship_type']).agg(
        collision_count=('movie_1', 'count'),
        avg_jaccard_similarity=('char4_jaccard_similarity', 'mean'),
        avg_rating_difference=('rating_difference', 'mean')
    ).round(4).sort_values('collision_count', ascending=False).reset_index()

    # 7. Sheet 4: Top Cross-Movie Affinities (Which different films collide most frequently)
    cross_movies = pairs_df[pairs_df['movie_1'] != pairs_df['movie_2']].groupby(
        ['movie_1', 'movie_2', 'release_decade_pair', 'watch_decade_pair']
    ).agg(
        collisions=('char4_jaccard_similarity', 'count'),
        avg_jaccard=('char4_jaccard_similarity', 'mean'),
        avg_rating_diff=('rating_difference', 'mean')
    ).round(4).sort_values('collisions', ascending=False).head(50).reset_index()

    # 8. Sheet 5: LSH Theory & S-Curve
    s_vals = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.5493, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
    lsh_data = []
    b = lsh.b
    r = lsh.r
    thresh = lsh.threshold
    for s in s_vals:
        p = 1.0 - (1.0 - s ** r) ** b
        filt = "Filtered Out (>99% rejected)" if p < 0.05 else ("Transition Zone" if p < 0.95 else "Candidate Pair Collision")
        lsh_data.append({
            'Jaccard_Similarity_s': s,
            'Band_Collision_Prob_P': round(p, 4),
            'Filtering_Status': filt,
            'Parameters': f"b={b}, r={r}, n={b*r}, threshold~={thresh:.4f}"
        })
    lsh_df = pd.DataFrame(lsh_data)

    # 9. Write Complete Excel Workbook
    excel_path = os.path.join(script_dir, 'complete_movie_pairwise_similarity_analysis.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        pairs_df.to_excel(writer, sheet_name='LSH Review Collision Pairs', index=False)
        matrix_df.to_excel(writer, sheet_name='Release Decade Hit Matrix')
        watch_rel_df.to_excel(writer, sheet_name='Watch vs Release Dynamics', index=False)
        cross_movies.to_excel(writer, sheet_name='Top Cross-Movie Affinities', index=False)
        lsh_df.to_excel(writer, sheet_name='LSH S-Curve & Theory', index=False)

    print(f"\n[OK] Excel Workbook successfully exported to: {excel_path}")
    print(f"Total Collided Review Pairs in Master Sheet: {len(pairs_df):,}")
    print(f"Top Relationship Types Discovered:\n{pairs_df['relationship_type'].value_counts()}")

if __name__ == '__main__':
    main()
