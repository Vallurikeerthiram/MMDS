# 🎯 MMDS Part E: Option 3 — Clustering & BFR/CURE Applicability

This folder contains the complete implementation, data analysis, and technical report for **MMDS Part E (Pattern Discovery) — Option 3: Clustering**.

---

## 📋 Overview of Tasks

1. **Group Similar Records**:
   - **Movie-Level Clustering**: 185 unique catalogued films grouped into coherent cinematic cohorts based on a 30-dimensional multimodal feature space (release year, runtime, star rating distributions, text-derived latent aesthetic themes via TF-IDF + TruncatedSVD, one-hot primary genres, and regional production scale).
   - **Review-Level Clustering**: Individual user review records ($N=2,500$ sample) clustered into 4 audience reception typologies (Casual High-Praise, In-Depth Critique, Literary Classicist Appreciations, and Critical Dissenters).
2. **Discuss BFR / CURE Applicability**:
   - Rigorous mathematical and empirical comparison of **BFR (Bradley-Fayyad-Reina)** and **CURE (Clustering Using REpresentatives)** algorithms on large-scale cinematic data.
   - Streaming simulation of BFR demonstrating **9.64× memory compression** using Discard Set (DS), Compressed Set (CS), and Retained Set (RS).
   - CURE simulation with shrunk representative points ($\alpha = 0.20, c=4$) demonstrating capture of non-spherical, elongated cluster geometries.

---

## 📁 Folder Contents

| File | Description |
|:---|:---|
| [`clustering_engine.py`](file:///c:/Users/keert/OneDrive%20-%20Amrita%20vishwa%20vidyapeetham/Amrita/Sem7/Projects/MMDS/clustering/clustering_engine.py) | Full Python implementation of K-Means, BFR stream clustering simulation, CURE representative shrinkage, and evaluation metrics. |
| [`clustering_results.xlsx`](file:///c:/Users/keert/OneDrive%20-%20Amrita%20vishwa%20vidyapeetham/Amrita/Sem7/Projects/MMDS/clustering/clustering_results.xlsx) | Multi-sheet Excel workbook containing Movie Clusters, Cohort Profiles, Optimal $k$ Search, BFR Memory Compression Metrics, CURE Representative Coordinates, and Algorithm Benchmarks. |
| [`bfr_cure_applicability_analysis.md`](file:///c:/Users/keert/OneDrive%20-%20Amrita%20vishwa%20vidyapeetham/Amrita/Sem7/Projects/MMDS/clustering/bfr_cure_applicability_analysis.md) | In-depth technical report discussing results, observations, mathematical proofs, and algorithmic suitability. |

---

## 🚀 How to Run

Execute the clustering pipeline and regenerate all Excel deliverables:

```bash
python clustering/clustering_engine.py
```

### Key Output Metrics:
- **Optimal Partition**: $k = 3$ (Silhouette Score: $0.1574$, Davies-Bouldin: $2.0749$)
- **BFR Compression Ratio**: $9.64\times$ (Absorbed 153/185 points into Discard Set)
- **CURE Shrunk Points**: 12 representative points across 3 clusters ($\alpha = 0.20$)
