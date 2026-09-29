"""
Multi-Decade Temporal Trajectory Alignment (TTA) Item-Item Recommendation System
================================================================================
MMDS Part E: Advanced Recommendation Engine with Temporal Dynamics & Decade Trajectory Matching

Theoretical & Mathematical Architecture:
----------------------------------------
Standard Collaborative Filtering (CF) assumes static item-item similarities.
In reality, cultural and aesthetic reception of cinematic art shifts across decades.

This engine implements the Multi-Decade Temporal Trajectory Alignment (TTA) model:
1. Release Era Context (Decade X):
   - Movie A and Movie B originate in theatrical release decades X_A and X_B.
   - Cinematic era proximity factor: S_release(A, B) = exp(-|X_A - X_B| / 30.0)
     Captures shared historical context, golden-age filmmaking techniques, and cultural origins.

2. Multi-Era Watch/Review Reception (Decades Y and Z):
   - Decades Y (e.g. 2010s) and Z (e.g. 2020s) capture distinct audience viewing eras.
   - For each era d in {2010s, 2020s}:
     * Sim_text,d(A, B) = Cosine_Similarity(TFIDF_d(A), TFIDF_d(B))
     * Sim_rating,d(A, B) = 1.0 - |Rating_d(A) - Rating_d(B)| / 5.0
     * Sim_d(A, B) = 0.85 * Sim_text,d(A, B) + 0.15 * Sim_rating,d(A, B)

3. Recency-Weighted Temporal Decay:
   - Newer reviews carry higher weight, older reviews carry decayed weight:
     w_2020s = 1.0 (recent, high confidence)
     w_2010s = 0.5 (older, decayed memory)
   - Base Temporal Similarity:
     Sim_base(A, B) = [w_2020s * Sim_2020s + w_2010s * Sim_2010s] / [w_2020s + w_2010s]

4. Trajectory Alignment Multiplier (gamma):
   - Evaluates whether the reception trajectory matched across decades:
     * Case 1: Full Multi-Decade Alignment ("Trends Fell Same across Y and Z")
       Both Sim_2010s >= tau_10 AND Sim_2020s >= tau_20
       => gamma = 1.25 (Maximum Confidence, 'Permanent Multi-Decade Twin')
     * Case 2: Modern Emerging Trend ("Similar in Decade Z Only")
       Sim_2020s >= tau_20, but Sim_2010s < tau_10
       => gamma = 0.95 (Reduced weight, but still recommended as modern convergence)
     * Case 3: Faded Historical Parallel ("Similar in Decade Y Only")
       Sim_2010s >= tau_10, but Sim_2020s < tau_20
       => gamma = 0.70 (Decayed historical memory)
     * Case 4: Low Cross-Decade Association
       Neither threshold met => gamma = 0.55

5. Final Composite Recommendation Score:
   Score(A, B) = min(1.0, Sim_base(A, B) * gamma(A, B) * [0.85 + 0.15 * S_release(A, B)])

6. User Inference Scenario:
   If a user watches Movie A now and liked it, the engine retrieves candidate films B
   ranked by Score(A, B), providing full mathematical trajectory diagnostics.
"""

import os
import re
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure utf-8 stdout
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

class TemporalTrajectoryRecommender:
    def __init__(self, data_path=None):
        if data_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(script_dir, '..', 'data', 'letterboxd_final_dataset.csv')
        self.data_path = data_path
        self.movies = []
        self.movie_meta = {}
        self.sim_10_mat = None
        self.sim_20_mat = None
        self.temporal_utility_mat = None
        self.tau_10 = 0.20
        self.tau_20 = 0.28
        self.w_recent = 1.0  # Decade Z (2020s)
        self.w_old = 0.5     # Decade Y (2010s)
        self._load_and_train()

    def _load_and_train(self):
        print(f"Loading Letterboxd dataset from: {self.data_path}")
        df = pd.read_csv(self.data_path)

        df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce')
        df['review_year'] = df['review_date'].dt.year.fillna(2020).astype(int)
        df['watch_era'] = np.where(df['review_year'] <= 2019, '2010s', '2020s')
        df['release_decade'] = (df['release_year'] // 10).astype(int) * 10

        self.movies = sorted(df['movie_title'].unique())
        n_movies = len(self.movies)
        print(f"Total catalogued movies: {n_movies}")

        era_texts = {'2010s': {}, '2020s': {}}
        era_ratings = {'2010s': {}, '2020s': {}}

        for m in self.movies:
            m_df = df[df['movie_title'] == m]
            avg_rat = float(m_df['rating'].mean())
            self.movie_meta[m] = {
                'release_year': int(m_df['release_year'].iloc[0]),
                'release_decade': int(m_df['release_decade'].iloc[0]),
                'genre': str(m_df['primary_genre'].iloc[0]),
                'avg_rating': round(avg_rat, 2),
                'total_reviews': len(m_df)
            }

            t_10 = clean_text(' '.join(m_df[m_df['watch_era'] == '2010s']['review_text'].dropna().tolist()))
            t_20 = clean_text(' '.join(m_df[m_df['watch_era'] == '2020s']['review_text'].dropna().tolist()))
            era_texts['2010s'][m] = t_10
            era_texts['2020s'][m] = t_20

            r_10 = m_df[m_df['watch_era'] == '2010s']['rating'].mean()
            r_20 = m_df[m_df['watch_era'] == '2020s']['rating'].mean()
            era_ratings['2010s'][m] = r_10 if not np.isnan(r_10) else avg_rat
            era_ratings['2020s'][m] = r_20 if not np.isnan(r_20) else avg_rat

        self.era_ratings = era_ratings

        print("Fitting Decade Y (2010s) TF-IDF text similarity space...")
        vec_10 = TfidfVectorizer(stop_words='english', min_df=2, max_features=3500)
        tfidf_10 = vec_10.fit_transform([era_texts['2010s'][m] for m in self.movies])
        self.sim_10_mat = cosine_similarity(tfidf_10)

        print("Fitting Decade Z (2020s) TF-IDF text similarity space...")
        vec_20 = TfidfVectorizer(stop_words='english', min_df=2, max_features=3500)
        tfidf_20 = vec_20.fit_transform([era_texts['2020s'][m] for m in self.movies])
        self.sim_20_mat = cosine_similarity(tfidf_20)

        print("Synthesizing Full Multi-Decade Trajectory Item Utility Matrix...")
        self.temporal_utility_mat = np.zeros((n_movies, n_movies))
        self.pairwise_metadata = {}

        for i in range(n_movies):
            movie_a = self.movies[i]
            for j in range(n_movies):
                if i == j:
                    self.temporal_utility_mat[i][j] = 1.0
                    continue

                movie_b = self.movies[j]
                s_text_10 = self.sim_10_mat[i, j]
                s_text_20 = self.sim_20_mat[i, j]

                s_rat_10 = 1.0 - abs(era_ratings['2010s'][movie_a] - era_ratings['2010s'][movie_b]) / 5.0
                s_rat_20 = 1.0 - abs(era_ratings['2020s'][movie_a] - era_ratings['2020s'][movie_b]) / 5.0

                sim_10 = 0.85 * s_text_10 + 0.15 * s_rat_10
                sim_20 = 0.85 * s_text_20 + 0.15 * s_rat_20

                base_sim = (self.w_recent * sim_20 + self.w_old * sim_10) / (self.w_recent + self.w_old)

                # Trajectory alignment logic
                high_10 = s_text_10 >= self.tau_10
                high_20 = s_text_20 >= self.tau_20

                if high_10 and high_20:
                    gamma = 1.25
                    trend_type = "Permanent Multi-Decade Twin (Full Trajectory Match)"
                elif high_20 and not high_10:
                    gamma = 0.95
                    trend_type = "Modern Emerging Match (2020s Only - Reduced Weight)"
                elif high_10 and not high_20:
                    gamma = 0.70
                    trend_type = "Faded Historic Match (Decayed Historical Memory)"
                else:
                    gamma = 0.55
                    trend_type = "Uncorrelated Trajectory"

                # Release decade proximity (Decade X)
                dec_diff = abs(self.movie_meta[movie_a]['release_decade'] - self.movie_meta[movie_b]['release_decade'])
                rel_sim = np.exp(-dec_diff / 30.0)
                rel_factor = 0.85 + 0.15 * rel_sim

                final_score = min(1.0, round(base_sim * gamma * rel_factor, 4))
                self.temporal_utility_mat[i][j] = final_score

                self.pairwise_metadata[(movie_a, movie_b)] = {
                    'final_score': final_score,
                    'trend_type': trend_type,
                    's_text_20': round(s_text_20, 4),
                    's_text_10': round(s_text_10, 4),
                    'rel_decade_diff': dec_diff,
                    'same_release_decade': (dec_diff == 0)
                }

    def recommend_for_liked_movie(self, liked_movie, top_k=5):
        """
        Inference Engine:
        Given that a user watched and liked `liked_movie` now, find candidate movies B
        whose temporal reception trajectory mirrors `liked_movie`.
        """
        # Exact or case-insensitive match
        matched = None
        for m in self.movies:
            if m.lower() == liked_movie.lower():
                matched = m
                break
        if not matched:
            candidates = [m for m in self.movies if liked_movie.lower() in m.lower()]
            if candidates:
                matched = candidates[0]
            else:
                raise ValueError(f"Movie '{liked_movie}' not found in catalog.")

        idx_a = self.movies.index(matched)
        scores = []
        for idx_b, other_movie in enumerate(self.movies):
            if idx_a == idx_b:
                continue
            pair_info = self.pairwise_metadata[(matched, other_movie)]
            other_meta = self.movie_meta[other_movie]
            scores.append({
                'movie': other_movie,
                'score': pair_info['final_score'],
                'trend_type': pair_info['trend_type'],
                'release_year': other_meta['release_year'],
                'release_decade': f"{other_meta['release_decade']}s",
                'same_release_decade': pair_info['same_release_decade'],
                'genre': other_meta['genre'],
                'avg_rating': other_meta['avg_rating'],
                'sim_2020s': pair_info['s_text_20'],
                'sim_2010s': pair_info['s_text_10']
            })

        scores = sorted(scores, key=lambda x: x['score'], reverse=True)
        return matched, scores[:top_k]

    def export_excel_report(self, output_path=None):
        if output_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_path = os.path.join(script_dir, 'temporal_item_recommendations.xlsx')

        print(f"\nExporting comprehensive multi-sheet Excel report to: {output_path}")

        # 1. Sheet: Top 5 Recommendations for all 185 movies
        recs_list = []
        dynamics_list = []

        for m in self.movies:
            meta = self.movie_meta[m]
            _, top5 = self.recommend_for_liked_movie(m, top_k=5)

            row = {
                'target_movie': m,
                'release_year': meta['release_year'],
                'release_decade': f"{meta['release_decade']}s",
                'primary_genre': meta['genre'],
                'avg_rating': meta['avg_rating'],
            }

            for rank, r in enumerate(top5, 1):
                row[f'rec_{rank}_movie'] = r['movie']
                row[f'rec_{rank}_year'] = r['release_year']
                row[f'rec_{rank}_genre'] = r['genre']
                row[f'rec_{rank}_score'] = r['score']
                row[f'rec_{rank}_trajectory'] = r['trend_type']
                row[f'rec_{rank}_same_decade'] = 'YES' if r['same_release_decade'] else 'NO'

            recs_list.append(row)

            dynamics_list.append({
                'target_movie': m,
                'release_decade': f"{meta['release_decade']}s",
                'top_recommendation': top5[0]['movie'],
                'rec_release_decade': top5[0]['release_decade'],
                'trajectory_dynamic': top5[0]['trend_type'],
                'composite_score': top5[0]['score'],
                'similarity_2020s (Newer)': top5[0]['sim_2020s'],
                'similarity_2010s (Decayed)': top5[0]['sim_2010s']
            })

        recs_df = pd.DataFrame(recs_list)
        dyn_df = pd.DataFrame(dynamics_list)
        utility_df = pd.DataFrame(self.temporal_utility_mat, index=self.movies, columns=self.movies)

        # 4. Sheet: User Liked Simulation Scenarios
        simulation_targets = [
            'Harakiri', 'Seven Samurai', 'The Godfather', 'Interstellar',
            'Everything Everywhere All at Once', 'Whiplash', 'Spirited Away', 'GoodFellas'
        ]
        sim_list = []
        for st in simulation_targets:
            if st in self.movies:
                target_m, top4 = self.recommend_for_liked_movie(st, top_k=4)
                for rank, r in enumerate(top4, 1):
                    sim_list.append({
                        'User Liked Movie (Watched Now)': target_m,
                        'Target Release Year': self.movie_meta[target_m]['release_year'],
                        'Target Decade X': f"{self.movie_meta[target_m]['release_decade']}s",
                        'Rank': rank,
                        'Recommended Movie B': r['movie'],
                        'Candidate Release Year': r['release_year'],
                        'Candidate Decade X': r['release_decade'],
                        'Same Release Era?': 'YES' if r['same_release_decade'] else 'NO',
                        'Composite Score': r['score'],
                        'Trajectory Alignment Category': r['trend_type'],
                        'Decade Z (2020s) Sim': r['sim_2020s'],
                        'Decade Y (2010s) Sim': r['sim_2010s']
                    })
        sim_df = pd.DataFrame(sim_list)

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            recs_df.to_excel(writer, sheet_name='Top 5 Movie Recommendations', index=False)
            dyn_df.to_excel(writer, sheet_name='Temporal Trajectory Analysis', index=False)
            sim_df.to_excel(writer, sheet_name='User Liked Simulation Demo', index=False)
            utility_df.to_excel(writer, sheet_name='Temporal Item Utility Matrix')

        print(f"[SUCCESS] Multi-decade recommendation dataset written to: {output_path}")

def print_user_recommendation(recommender, target_movie, top_k=5):
    try:
        matched, recs = recommender.recommend_for_liked_movie(target_movie, top_k=top_k)
    except Exception as e:
        print(f"Error: {e}")
        return

    meta = recommender.movie_meta[matched]
    print("\n" + "=" * 80)
    print(f"USER LIKED: {matched} ({meta['release_year']} | {meta['release_decade']}s | {meta['genre']} | Rating: {meta['avg_rating']})")
    print("=" * 80)
    print("Finding candidates with matching Multi-Decade Reception Trajectories...\n")

    for rank, r in enumerate(recs, 1):
        dec_match_tag = "[SAME RELEASE DECADE]" if r['same_release_decade'] else f"[{r['release_decade']}]"
        print(f"Rank {rank}: {r['movie']} ({r['release_year']}) {dec_match_tag}")
        print(f"  * Composite Trajectory Score : {r['score']}")
        print(f"  * Trajectory Alignment Type  : {r['trend_type']}")
        print(f"  * Decade Z (2020s) Similarity : {r['sim_2020s']} (Weight: 1.0 - Modern Confidence)")
        print(f"  * Decade Y (2010s) Similarity : {r['sim_2010s']} (Weight: 0.5 - Decayed Relevance)")
        print(f"  * Genre: {r['genre']} | Rating: {r['avg_rating']}/5.0")
        print("-" * 80)

def main():
    parser = argparse.ArgumentParser(description="Multi-Decade Temporal Trajectory Recommender")
    parser.add_argument('--movie', type=str, help="Simulate user liking a specific movie")
    parser.add_argument('--interactive', action='store_true', help="Run interactive movie query loop")
    args = parser.parse_args()

    recommender = TemporalTrajectoryRecommender()

    # Always generate/update the official Excel artifact
    recommender.export_excel_report()

    if args.movie:
        print_user_recommendation(recommender, args.movie)
    elif args.interactive:
        print("\nEnter movie name to get trajectory recommendations (type 'exit' to quit):")
        while True:
            try:
                query = input("\nMovie Title > ").strip()
                if query.lower() in ['exit', 'quit', 'q']:
                    break
                if not query:
                    continue
                print_user_recommendation(recommender, query)
            except (KeyboardInterrupt, EOFError):
                break
    else:
        # Default run: Print representative sample recommendations
        sample_targets = ['Harakiri', 'Interstellar', 'The Godfather', 'Everything Everywhere All at Once']
        for st in sample_targets:
            if st in recommender.movies:
                print_user_recommendation(recommender, st, top_k=3)

if __name__ == '__main__':
    main()
