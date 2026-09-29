# 🔍 MMDS Part C: Similarity and Distance Analysis

This folder contains the complete implementation, mathematical pipelines, and generated deliverables for **MMDS Part C: Similarity and Distance Analysis**.

---

## 📋 Overview: Unsupervised LSH Review Discovery & Decade Dynamics

To discover similar reviews and near-duplicate textual sentiments across 61,786 user reviews without brute-force $O(N^2)$ comparisons:

1. **$k$-Shingling**:
   - Converts natural language review texts into character 4-shingles ($k=4$) to capture local phrase structure and stylistic vocabulary.
2. **MinHash Signatures**:
   - Compresses high-dimensional shingle sets into compact signatures of length $n = 100$ using universal linear hash permutations:
     $$h_{a,b}(x) = (a \cdot x + b) \pmod p$$
3. **Locality-Sensitive Hashing (LSH)**:
   - Divides signatures into $b = 20$ bands with $r = 5$ rows per band ($b \cdot r = 100$).
   - Theoretical similarity collision threshold:
     $$t \approx \left(\frac{1}{b}\right)^{1/r} = \left(\frac{1}{20}\right)^{1/5} \approx 0.549$$
   - Maps candidates with S-curve collision probability $P = 1 - (1 - s^r)^b$.
4. **Post-Hoc Decade and Cross-Movie Grouping**:
   - Evaluates collided pairs across:
     - **Same Movie Hits**: Intra-movie consensus across time.
     - **Same Release Era Cross-Movie Hits**: Independent films released in the same decade triggering identical critical reception decades later.
     - **Cross-Era Temporal Echoes**: Modern films echoing vintage classic critiques.

---

## 📁 Folder Contents

| File | Description |
|:---|:---|
| [`similarity_analysis.py`](file:///c:/Users/keert/OneDrive%20-%20Amrita%20vishwa%20vidyapeetham/Amrita/Sem7/Projects/MMDS/similarity_analysis/similarity_analysis.py) | Full Python implementation of 4-shingling, MinHashing, banded LSH collision discovery, and pairwise distance calculations. |
| [`complete_movie_pairwise_similarity_analysis.xlsx`](file:///c:/Users/keert/OneDrive%20-%20Amrita%20vishwa%20vidyapeetham/Amrita/Sem7/Projects/MMDS/similarity_analysis/complete_movie_pairwise_similarity_analysis.xlsx) | Multi-sheet Excel workbook containing LSH Collision Pairs, Release Decade Hit Matrix, Watch vs. Release Era Hits, Top Affinities, and S-Curve Theory. |

---

## 🚀 How to Run

```bash
python similarity_analysis/similarity_analysis.py
```
