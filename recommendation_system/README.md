# 🎬 MMDS Part E: Option 2 — Recommendation System with Temporal Dynamics

This folder contains the complete implementation, mathematical architecture, and generated deliverables for **MMDS Part E (Pattern Discovery) — Option 2: Recommendation System**.

---

## 📋 Overview: Multi-Decade Temporal Trajectory Alignment (TTA)

Standard Item-Item Collaborative Filtering assumes static user-item affinities. This engine models the temporal evolution of cinematic reception across decades:

1. **Release Decade Context (Decade $X$)**:
   - Calculates release era proximity: $S_{release}(A, B) = \exp(-|X_A - X_B| / 30.0)$.
   - Movies born in the same theatrical era receive vintage peer weighting.
2. **Multi-Era Review Reception (Decades $Y$ and $Z$)**:
   - Aggregates review text corpora across distinct viewing decades: Decade $Y$ (2010s) and Decade $Z$ (2020s).
   - Computes era-specific cosine similarity via TF-IDF vectorization and rating concordance.
3. **Recency-Weighted Temporal Decay**:
   - Applies temporal decay weights: $w_{2020s} = 1.0$ (recent high confidence), $w_{2010s} = 0.5$ (older decayed memory).
4. **Trajectory Alignment Multiplier ($\gamma$)**:
   - **Permanent Multi-Decade Twin** ($\gamma = 1.25$): High similarity in *both* 2010s and 2020s ("Trends fell the same").
   - **Modern Emerging Match** ($\gamma = 0.95$): High similarity in 2020s only ("Weight reduced, still recommended").
   - **Faded Historic Match** ($\gamma = 0.70$): High similarity in 2010s only (Decayed relevance).
   - **Uncorrelated Trajectory** ($\gamma = 0.55$).
5. **Liked Movie Inference Engine**:
   - Simulates user liking Movie $A$ today, querying candidate catalog $B$, ranking top recommendations, and explaining trajectory dynamics.

---

## 📁 Folder Contents

| File | Description |
|:---|:---|
| [`recommendation_engine.py`](file:///c:/Users/keert/OneDrive%20-%20Amrita%20vishwa%20vidyapeetham/Amrita/Sem7/Projects/MMDS/recommendation_system/recommendation_engine.py) | Full implementation of the Multi-Decade Temporal Trajectory Recommender with interactive CLI query capabilities. |
| [`temporal_item_recommendations.xlsx`](file:///c:/Users/keert/OneDrive%20-%20Amrita%20vishwa%20vidyapeetham/Amrita/Sem7/Projects/MMDS/recommendation_system/temporal_item_recommendations.xlsx) | Multi-sheet Excel workbook containing Top 5 Recommendations for all 185 films, Trajectory Analysis breakdown, User Liked Simulations, and Full Utility Matrix. |

---

## 🚀 How to Run

### Batch Generation (Updates Excel Artifacts)
```bash
python recommendation_system/recommendation_engine.py
```

### Query Single Movie Recommendation
```bash
python recommendation_system/recommendation_engine.py --movie "Harakiri"
```

### Interactive Command-Line Mode
```bash
python recommendation_system/recommendation_engine.py --interactive
```
