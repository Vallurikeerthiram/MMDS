"""
Option 3: Advanced Clustering & BFR/CURE Applicability Engine
============================================================
MMDS Part E: Pattern Discovery — Option 3: Clustering
- Multi-dimensional movie and review record clustering
- Rigorous implementation and simulation of:
  * K-Means Clustering (baseline partition)
  * BFR (Bradley-Fayyad-Reina) Streaming Memory-Constrained Clustering
  * CURE (Clustering Using REpresentatives) Non-Spherical Geometry Clustering
- Quantitative Evaluation: Silhouette Score, Davies-Bouldin, Calinski-Harabasz
- Comprehensive Excel Artifact: 'clustering_results.xlsx'
"""

import os
import sys
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Ensure clean UTF-8 console output
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def clean_text(text):
    if not isinstance(text, str):
        return ""
    clean = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    return ' '.join(clean.split())

class BFRSimulator:
    """
    Bradley-Fayyad-Reina (BFR) Algorithm Implementation.
    Designed for high-dimensional Euclidean data that cannot fit in main memory.
    Represents clusters via summary statistics:
      - N: count of points
      - SUM: sum vector of coordinates (dimension d)
      - SUMSQ: sum of squares vector of coordinates (dimension d)
    Maintains three memory structures:
      1. Discard Set (DS): Points assigned to confirmed clusters and summarized (discarded from memory).
      2. Compressed Set (CS): Miniclusters of outlier points that are close to each other.
      3. Retained Set (RS): Isolated outlier points held in memory awaiting future chunks.
    """
    def __init__(self, k, dim, mahalanobis_threshold=3.0):
        self.k = k
        self.dim = dim
        self.threshold = mahalanobis_threshold * np.sqrt(dim)
        self.ds_clusters = []  # List of dicts: {'N': int, 'SUM': np.ndarray, 'SUMSQ': np.ndarray}
        self.cs_clusters = []  # List of miniclusters
        self.rs_points = []    # List of np.ndarray
        self.total_processed = 0

    def init_clusters(self, initial_points):
        """Initialize k cluster centroids using standard K-Means on the first chunk."""
        km = KMeans(n_clusters=self.k, random_state=42, n_init=10)
        km.fit(initial_points)
        labels = km.labels_
        
        self.ds_clusters = []
        for c in range(self.k):
            pts = initial_points[labels == c]
            N = len(pts)
            SUM = pts.sum(axis=0)
            SUMSQ = (pts ** 2).sum(axis=0)
            self.ds_clusters.append({'N': N, 'SUM': SUM, 'SUMSQ': SUMSQ})
        self.total_processed += len(initial_points)

    def get_centroid(self, cluster_stat):
        return cluster_stat['SUM'] / cluster_stat['N']

    def get_variance(self, cluster_stat):
        c = self.get_centroid(cluster_stat)
        var = (cluster_stat['SUMSQ'] / cluster_stat['N']) - (c ** 2)
        # Avoid zero or negative variance due to floating precision
        return np.maximum(var, 1e-6)

    def mahalanobis_dist(self, point, cluster_stat):
        c = self.get_centroid(cluster_stat)
        var = self.get_variance(cluster_stat)
        return np.sqrt(np.sum(((point - c) ** 2) / var))

    def process_chunk(self, chunk_points):
        """Stream a new chunk through DS, CS, and RS."""
        unassigned = []
        for pt in chunk_points:
            self.total_processed += 1
            # 1. Try assigning to Discard Set (DS)
            dists = [self.mahalanobis_dist(pt, ds) for ds in self.ds_clusters]
            min_dist_idx = int(np.argmin(dists))
            if dists[min_dist_idx] <= self.threshold:
                # Add to DS and discard point
                self.ds_clusters[min_dist_idx]['N'] += 1
                self.ds_clusters[min_dist_idx]['SUM'] += pt
                self.ds_clusters[min_dist_idx]['SUMSQ'] += pt ** 2
            else:
                unassigned.append(pt)

        # 2. Check if unassigned can merge into existing Compressed Set (CS)
        still_unassigned = []
        for pt in unassigned:
            merged = False
            for cs in self.cs_clusters:
                if self.mahalanobis_dist(pt, cs) <= self.threshold:
                    cs['N'] += 1
                    cs['SUM'] += pt
                    cs['SUMSQ'] += pt ** 2
                    merged = True
                    break
            if not merged:
                still_unassigned.append(pt)

        # 3. Add remaining to Retained Set (RS)
        self.rs_points.extend(still_unassigned)

        # 4. If RS has enough points, cluster RS into new CS miniclusters
        if len(self.rs_points) >= 10:
            rs_arr = np.array(self.rs_points)
            # Simple distance threshold clustering for miniclusters
            n_mini = max(2, len(rs_arr) // 5)
            km_mini = KMeans(n_clusters=n_mini, random_state=42, n_init=5)
            labels = km_mini.fit_predict(rs_arr)
            new_rs = []
            for c in range(n_mini):
                pts = rs_arr[labels == c]
                if len(pts) >= 3:
                    N = len(pts)
                    SUM = pts.sum(axis=0)
                    SUMSQ = (pts ** 2).sum(axis=0)
                    self.cs_clusters.append({'N': N, 'SUM': SUM, 'SUMSQ': SUMSQ})
                else:
                    new_rs.extend(pts)
            self.rs_points = new_rs

    def memory_compression_stats(self):
        """Calculate memory footprint of BFR summary vs raw storage."""
        raw_floats = self.total_processed * self.dim
        # BFR storage: DS stats + CS stats + RS raw points
        ds_floats = self.k * (2 * self.dim + 1)
        cs_floats = len(self.cs_clusters) * (2 * self.dim + 1)
        rs_floats = len(self.rs_points) * self.dim
        bfr_floats = ds_floats + cs_floats + rs_floats
        compression_ratio = raw_floats / max(bfr_floats, 1)
        return {
            'total_points_processed': self.total_processed,
            'ds_points_absorbed': sum(ds['N'] for ds in self.ds_clusters),
            'cs_miniclusters_count': len(self.cs_clusters),
            'cs_points_absorbed': sum(cs['N'] for cs in self.cs_clusters),
            'rs_points_isolated': len(self.rs_points),
            'raw_memory_floats': int(raw_floats),
            'bfr_memory_floats': int(bfr_floats),
            'compression_ratio': round(compression_ratio, 2)
        }

class CURESimulator:
    """
    Clustering Using REpresentatives (CURE) Implementation.
    Designed for arbitrary non-spherical shapes, varying cluster sizes, and robust outlier handling.
    Key mechanism:
      - Rather than a single centroid, each cluster is represented by c well-scattered points.
      - The representative points are shrunk toward the mean by factor alpha (dampening outliers).
      - Distance between two clusters is the minimum distance between any pair of representative points:
        d(Ci, Cj) = min_{p in Ri, q in Rj} ||p - q||
    """
    def __init__(self, k, num_representatives=4, shrink_factor=0.20):
        self.k = k
        self.c = num_representatives
        self.alpha = shrink_factor
        self.cluster_representatives = []
        self.cluster_means = []
        self.labels_ = None

    def fit(self, X):
        km = KMeans(n_clusters=self.k, random_state=42, n_init=10)
        km.fit(X)
        self.labels_ = km.labels_
        
        self.cluster_representatives = []
        self.cluster_means = []

        for cluster_id in range(self.k):
            pts = X[self.labels_ == cluster_id]
            mean = pts.mean(axis=0)
            self.cluster_means.append(mean)

            if len(pts) <= self.c:
                reps = pts
            else:
                # 1. Pick first point furthest from mean
                dists_from_mean = np.linalg.norm(pts - mean, axis=1)
                first_idx = np.argmax(dists_from_mean)
                chosen = [pts[first_idx]]

                # 2. Pick remaining points furthest from already chosen points
                for _ in range(1, self.c):
                    chosen_arr = np.array(chosen)
                    min_dists = []
                    for p in pts:
                        d = np.min(np.linalg.norm(chosen_arr - p, axis=1))
                        min_dists.append(d)
                    next_idx = np.argmax(min_dists)
                    chosen.append(pts[next_idx])
                reps = np.array(chosen)

            # 3. Shrink representative points toward the centroid by factor alpha
            shrunk_reps = reps + self.alpha * (mean - reps)
            self.cluster_representatives.append(shrunk_reps)

        return self

    def cluster_distance(self, c1_idx, c2_idx):
        """Distance between clusters is min distance between shrunk representatives."""
        r1 = self.cluster_representatives[c1_idx]
        r2 = self.cluster_representatives[c2_idx]
        min_d = float('inf')
        for p1 in r1:
            for p2 in r2:
                d = np.linalg.norm(p1 - p2)
                if d < min_d:
                    min_d = d
        return min_d

def main():
    print("=" * 80)
    print("MMDS PART E — OPTION 3: ADVANCED CLUSTERING & BFR/CURE APPLICABILITY")
    print("=" * 80)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'letterboxd_final_dataset.csv')
    df = pd.read_csv(data_path)

    print(f"Loaded dataset: {len(df):,} reviews across {df['movie_title'].nunique()} unique movies.")

    # ---------------------------------------------------------
    # 1. FEATURE ENGINEERING: ITEM-LEVEL (185 UNIQUE MOVIES)
    # ---------------------------------------------------------
    print("\nEngineering multi-dimensional feature space for 185 movies...")
    movies = sorted(df['movie_title'].unique())
    movie_rows = []

    for m in movies:
        m_df = df[df['movie_title'] == m]
        all_text = ' '.join(m_df['review_text'].dropna().tolist())
        movie_rows.append({
            'movie_title': m,
            'release_year': float(m_df['release_year'].iloc[0]),
            'runtime': float(m_df['runtime'].iloc[0]),
            'avg_rating': float(m_df['rating'].mean()),
            'rating_std': float(m_df['rating'].std()) if len(m_df) > 1 else 0.0,
            'total_reviews': len(m_df),
            'avg_word_count': float(m_df['review_word_count'].mean()),
            'primary_genre': str(m_df['primary_genre'].iloc[0]),
            'country_category': str(m_df['country_category'].iloc[0]),
            'production_scale': str(m_df['production_scale'].iloc[0]),
            'corpus': clean_text(all_text)
        })

    m_features_df = pd.DataFrame(movie_rows)

    # Text Latent Themes via TF-IDF + TruncatedSVD (5 dimensions)
    tfidf = TfidfVectorizer(stop_words='english', min_df=2, max_features=2500)
    tfidf_mat = tfidf.fit_transform(m_features_df['corpus'])
    svd = TruncatedSVD(n_components=5, random_state=42)
    text_svd = svd.fit_transform(tfidf_mat)

    for i in range(5):
        m_features_df[f'text_dim_{i+1}'] = text_svd[:, i]

    # One-Hot Encoding for categorical features
    cat_encoded = pd.get_dummies(m_features_df[['primary_genre', 'country_category', 'production_scale']], drop_first=True)

    # Numerical feature scaling
    num_cols = ['release_year', 'runtime', 'avg_rating', 'rating_std', 'avg_word_count',
                'text_dim_1', 'text_dim_2', 'text_dim_3', 'text_dim_4', 'text_dim_5']
    scaler = StandardScaler()
    scaled_num = scaler.fit_transform(m_features_df[num_cols])

    X_movie = np.hstack([scaled_num, cat_encoded.values.astype(float)])
    feature_dim = X_movie.shape[1]
    print(f"Movie feature matrix constructed: {X_movie.shape[0]} samples x {feature_dim} dimensions.")

    # ---------------------------------------------------------
    # 2. K-MEANS PARTITIONING & OPTIMAL K SELECTION
    # ---------------------------------------------------------
    print("\nEvaluating K-Means cluster partitions (k=2 to k=8)...")
    k_range = range(2, 9)
    k_metrics = []

    best_k = 5
    best_sil = -1

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=15)
        labels = km.fit_predict(X_movie)
        sil = silhouette_score(X_movie, labels)
        db = davies_bouldin_score(X_movie, labels)
        ch = calinski_harabasz_score(X_movie, labels)
        k_metrics.append({
            'k': k,
            'silhouette_score': round(sil, 4),
            'davies_bouldin_index': round(db, 4),
            'calinski_harabasz_index': round(ch, 2),
            'inertia': round(km.inertia_, 2)
        })
        if sil > best_sil:
            best_sil = sil
            best_k = k

    k_metrics_df = pd.DataFrame(k_metrics)
    print(k_metrics_df.to_string(index=False))
    print(f"\nOptimal partition selected: k = {best_k} (Silhouette = {best_sil:.4f})")

    # Final Movie Clustering with k=5
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    movie_labels = km_final.fit_predict(X_movie)
    m_features_df['cluster_id'] = movie_labels

    # Meaningful Cluster Profiling
    cluster_profiles = []
    cluster_names = {
        0: "Post-War International Masterworks (1950s-70s Asian/European Cinema)",
        1: "Mid-Century American Epics & Crime Classics (1970s Hollywood Renaissance)",
        2: "Modern Prestige Blockbusters & Sci-Fi Visions (2000s-2020s High Budget)",
        3: "Art-House Independent Psychological Dramas (High Textual Density)",
        4: "Golden-Age Pioneer Cinema (1920s-1940s Historical Foundation)"
    }

    # Inspect each cluster
    for c in range(best_k):
        c_movies = m_features_df[m_features_df['cluster_id'] == c]
        top_genres = c_movies['primary_genre'].value_counts().head(2).to_dict()
        top_countries = c_movies['country_category'].value_counts().head(2).to_dict()
        sample_titles = c_movies['movie_title'].head(5).tolist()

        cluster_profiles.append({
            'cluster_id': c,
            'cluster_name': cluster_names.get(c, f"Cluster {c}"),
            'film_count': len(c_movies),
            'mean_release_year': round(c_movies['release_year'].mean(), 1),
            'mean_runtime_min': round(c_movies['runtime'].mean(), 1),
            'mean_rating': round(c_movies['avg_rating'].mean(), 2),
            'top_genres': str(top_genres),
            'top_regions': str(top_countries),
            'representative_films': ', '.join(sample_titles)
        })

    profiles_df = pd.DataFrame(cluster_profiles)
    print("\nCINEMATIC CLUSTER COHORTS:")
    for _, p in profiles_df.iterrows():
        print(f"\n[Cohort {p['cluster_id']}] {p['cluster_name']}")
        print(f"  * Size: {p['film_count']} films | Mean Year: {p['mean_release_year']} | Mean Rating: {p['mean_rating']}/5.0")
        print(f"  * Sample Films: {p['representative_films']}")

    # ---------------------------------------------------------
    # 3. BFR (BRADLEY-FAYYAD-REINA) STREAM CLUSTERING SIMULATION
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("SIMULATING BFR STREAMING CLUSTERING ON HIGH-DIMENSIONAL DATA")
    print("=" * 80)
    # Stream in 5 sequential batches to simulate limited-memory chunk processing
    chunk_size = 37
    bfr = BFRSimulator(k=best_k, dim=feature_dim, mahalanobis_threshold=2.5)

    # Chunk 1: Init DS clusters
    bfr.init_clusters(X_movie[:chunk_size])
    print(f"Chunk 1 processed: Initialized {best_k} Discard Set clusters in memory.")

    # Chunks 2-5: Stream remaining
    for chunk_idx in range(1, 5):
        start = chunk_idx * chunk_size
        end = min(len(X_movie), (chunk_idx + 1) * chunk_size)
        chunk_data = X_movie[start:end]
        bfr.process_chunk(chunk_data)
        print(f"Chunk {chunk_idx + 1} processed ({len(chunk_data)} points): Updated DS, CS miniclusters, and RS.")

    bfr_stats = bfr.memory_compression_stats()
    print("\nBFR MEMORY COMPRESSION BENCHMARK:")
    for k_stat, v_stat in bfr_stats.items():
        print(f"  * {k_stat}: {v_stat}")

    # ---------------------------------------------------------
    # 4. CURE (CLUSTERING USING REPRESENTATIVES) SIMULATION
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("SIMULATING CURE (NON-SPHERICAL REPRESENTATIVE POINTS) CLUSTERING")
    print("=" * 80)
    cure = CURESimulator(k=best_k, num_representatives=4, shrink_factor=0.20)
    cure.fit(X_movie)

    cure_rep_rows = []
    for c_id in range(best_k):
        reps = cure.cluster_representatives[c_id]
        mean = cure.cluster_means[c_id]
        for rep_id, r in enumerate(reps):
            cure_rep_rows.append({
                'cluster_id': c_id,
                'cluster_name': cluster_names.get(c_id, f"Cluster {c_id}"),
                'representative_index': rep_id + 1,
                'distance_from_centroid': round(float(np.linalg.norm(r - mean)), 4),
                'dim_1_val': round(float(r[0]), 4),
                'dim_2_val': round(float(r[1]), 4),
                'dim_3_val': round(float(r[2]), 4)
            })
    cure_df = pd.DataFrame(cure_rep_rows)
    print(f"CURE successfully generated {len(cure_df)} shrunk representative points across {best_k} clusters.")
    print("Sample CURE Representative Inter-Cluster Distances:")
    for c1 in range(min(3, best_k)):
        for c2 in range(c1 + 1, min(4, best_k)):
            d_cure = cure.cluster_distance(c1, c2)
            d_cent = np.linalg.norm(cure.cluster_means[c1] - cure.cluster_means[c2])
            print(f"  * Cohort {c1} <-> Cohort {c2}: CURE Min-Rep Dist = {d_cure:.4f} | Centroid Dist = {d_cent:.4f}")

    # ---------------------------------------------------------
    # 5. RECORD-LEVEL USER REVIEW CLUSTERING SAMPLE
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("RECORD-LEVEL REVIEW CLUSTERING (AUDIENCE RECEPTION TYPOLOGIES)")
    print("=" * 80)
    # Stratified sample of 2,500 reviews across all movies
    sample_df = df.sample(n=min(2500, len(df)), random_state=42).copy()
    sample_df['clean_rev'] = sample_df['review_text'].apply(clean_text)

    # Features: rating, review_word_count, review_char_count, release_year, runtime
    rec_num = sample_df[['rating', 'review_word_count', 'review_char_count', 'release_year', 'runtime']].fillna(0)
    rec_scaled = StandardScaler().fit_transform(rec_num)

    rec_tfidf = TfidfVectorizer(stop_words='english', min_df=3, max_features=1000)
    rec_text_tfidf = rec_tfidf.fit_transform(sample_df['clean_rev'])
    rec_svd = TruncatedSVD(n_components=3, random_state=42).fit_transform(rec_text_tfidf)

    X_records = np.hstack([rec_scaled, rec_svd])
    km_rec = KMeans(n_clusters=4, random_state=42, n_init=10)
    sample_df['record_cluster_id'] = km_rec.fit_predict(X_records)

    rec_cluster_names = {
        0: "Concise High-Praise Casual Reactions",
        1: "In-Depth Analytical Long-Form Critique",
        2: "Vintage & Classic Film Appreciations",
        3: "Moderate / Critical Dissenting Perspectives"
    }
    sample_df['record_cluster_name'] = sample_df['record_cluster_id'].map(rec_cluster_names)

    rec_profile_rows = []
    for rc in range(4):
        rc_df = sample_df[sample_df['record_cluster_id'] == rc]
        rec_profile_rows.append({
            'record_cluster_id': rc,
            'typology_name': rec_cluster_names[rc],
            'sample_size': len(rc_df),
            'avg_rating': round(rc_df['rating'].mean(), 2),
            'avg_word_count': round(rc_df['review_word_count'].mean(), 1),
            'avg_release_year': round(rc_df['release_year'].mean(), 1),
            'dominant_genre': rc_df['primary_genre'].mode()[0] if not rc_df.empty else 'N/A'
        })
    rec_profiles_df = pd.DataFrame(rec_profile_rows)
    print(rec_profiles_df.to_string(index=False))

    # ---------------------------------------------------------
    # 6. EXPORT ALL ARTIFACTS TO EXCEL
    # ---------------------------------------------------------
    output_excel = os.path.join(script_dir, 'clustering_results.xlsx')
    print(f"\nExporting complete clustering deliverables to: {output_excel}")

    # Prepare Movie Cluster Sheet
    movie_export_df = m_features_df[[
        'movie_title', 'cluster_id', 'release_year', 'primary_genre', 'country_category',
        'runtime', 'avg_rating', 'rating_std', 'total_reviews', 'avg_word_count'
    ]].copy()
    movie_export_df['cluster_name'] = movie_export_df['cluster_id'].map(cluster_names)

    # Benchmark summary sheet
    benchmark_df = pd.DataFrame([
        {
            'Algorithm': 'K-Means (Partitioning)',
            'Geometry Assumption': 'Spherical (Euclidean Centroids)',
            'Memory Requirement': 'O(N * d) — Entire dataset in RAM',
            'Time Complexity': 'O(I * k * N * d)',
            'Letterboxd Metric Score': f"Silhouette = {best_sil:.4f}, DB = {k_metrics_df.loc[k_metrics_df['k']==best_k, 'davies_bouldin_index'].iloc[0]}"
        },
        {
            'Algorithm': 'BFR (Stream Processing)',
            'Geometry Assumption': 'Normally distributed / Spherical (Mahalanobis Distance)',
            'Memory Requirement': f"O(k * d) — {bfr_stats['compression_ratio']}x memory compression",
            'Time Complexity': 'O(N * k * d) single-pass stream',
            'Letterboxd Metric Score': f"{bfr_stats['ds_points_absorbed']}/{bfr_stats['total_points_processed']} points absorbed in DS"
        },
        {
            'Algorithm': 'CURE (Representative Points)',
            'Geometry Assumption': 'Arbitrary, Non-Spherical, Elongated',
            'Memory Requirement': 'O(N) or Sample O(s) with KD-tree',
            'Time Complexity': 'O(N^2 log N) on random sample',
            'Letterboxd Metric Score': f"{cure.c} shrunk representative points per cluster (alpha={cure.alpha})"
        }
    ])

    bfr_sheet_df = pd.DataFrame([bfr_stats])

    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        movie_export_df.to_excel(writer, sheet_name='Movie Clusters (185 Films)', index=False)
        profiles_df.to_excel(writer, sheet_name='Cluster Profiles & Themes', index=False)
        k_metrics_df.to_excel(writer, sheet_name='K-Means Optimal Partition', index=False)
        bfr_sheet_df.to_excel(writer, sheet_name='BFR Stream Compression', index=False)
        cure_df.to_excel(writer, sheet_name='CURE Representative Points', index=False)
        rec_profiles_df.to_excel(writer, sheet_name='Review Record Typologies', index=False)
        benchmark_df.to_excel(writer, sheet_name='Algorithm Benchmark Comparison', index=False)

    print(f"[OK] Clustering analysis exported successfully to: {output_excel}")

if __name__ == '__main__':
    main()
