# Mining Temporal Evolution in Global Cinema (1924–2024) 🎬

A large-scale data mining study analyzing a century of cinematic production trajectories, cross-linguistic genre shifts, and contemporary digital audience consumption patterns using the **Mining of Massive Datasets (MMDS)** analytical framework.

---

## 📌 Abstract

Over the past century, cinema has transitioned from regional studio systems to an interconnected global streaming ecosystem. While traditional media analytics treat viewer evaluations as static observations, this study investigates the macro-level evolutionary trajectories of cinematic production and audience reception across a 100-year span (1924–2024). Leveraging a curated dataset of **61,786 validated user interactions** across **185 internationally acclaimed films**, we model: (1) how cinematic production attributes—such as genres, runtimes, countries of origin, and languages—have evolved across historical eras, and (2) how contemporary digital audiences (2012–2024) discover and evaluate historic versus modern cinema by analyzing the temporal delta between a film's **theatrical release year** and its **actual consumption date**. Applying core massive data mining methodologies—including high-dimensional vector distance metrics, stream sampling simulations, and network link analysis—this research uncovers key patterns in global cultural convergence, genre transitions, and viewer taste drift at scale.

---

## 🎯 Problem Statement

Traditional film analytics fail to capture the dual-dynamic nature of cinema:
1. **The Production Shift (1924–2024):** How film industries across different countries and languages have evolved their storytelling modes over time (e.g., Japan’s transition from mid-century samurai historical epics to modern animation; South Korea's ascendancy in high-tension psychological thrillers; changing audience tolerance for epic runtimes across different decades).
2. **The Audience Consumption Shift (Release Year vs. Watch Year):** How modern viewers interact with cinema retrospectively. The dataset allows us to measure the temporal gap between when a film was produced ($T_{\text{release}} \in [1924, 2024]$) and when a modern digital viewer experienced and reviewed it ($T_{\text{watch}} \in [2012, 2024]$). This reveals which historical eras and foreign-language movements maintain long-term cultural authority, and how re-watch behaviors vary across genres.

---

## 📊 Dataset Architecture & Ingestion Pipeline

The dataset was constructed by joining and cleaning two public Letterboxd corpora:
* **User Review & Interaction Stream:** Raw user reviews, star ratings, and diary logging dates across top-rated cinema (`letterboxd_250movie_reviews.csv`).
* **Film Metadata Catalog:** Film-level attributes covering genres, runtimes, languages, countries, and release years (`letterboxd_movies_dataset.csv`).

### Data Cleaning & Curation
* **URL Slug & Title Normalization:** Automated title resolution mapping URL slugs (e.g., `8-half` $\rightarrow$ *8½*) to canonical film records.
* **Rating Parsing:** Normalized Unicode star strings (`★` and `½`) to uniform continuous floats on a $[0.5, 5.0]$ scale.
* **Pruning Artifacts & Redundant Features:** Dropped rows with missing data or scraping UI artifacts (`Show All…` buttons, emotional theme tags). Removed derived/redundant columns (such as URL slugs, calculated ages, and decade categories) to avoid multicollinearity.
* **Final Deliverable:** **61,786 complete, non-null records** across **17 authentic core features**, stored in `data/letterboxd_final_dataset.csv` (67.1 MB) and Apache Parquet format (38.0 MB).

---

## 🗂️ Dataset Schema (17 Core Features)

| # | Attribute | Type | Description / Range |
|:---:|:---|:---:|:---|
| 1 | `review_id` | String | Unique synthetic primary key (`REV_000001` to `REV_061786`). |
| 2 | `movie_title` | String | Canonical film title (185 distinct films). |
| 3 | `rating` | Float | User star rating on a $[0.5, 5.0]$ scale (Mean = 4.49). |
| 4 | `review_date` | Date | Timestamp of user viewing/review (`1896` to `2024`, 99.9% in 2012–2024). |
| 5 | `watch_status` | Categorical | Logged user viewing state (`Watched`, `Rewatched`). |
| 6 | `review_text` | String | Unstructured natural language review text written by users. |
| 7 | `review_word_count` | Integer | Length of review in words ($[1, 10182]$, Median = 79). |
| 8 | `review_char_count` | Integer | Length of review in characters ($[1, 91885]$, Median = 441). |
| 9 | `release_year` | Integer | Theatrical release year ($[1924, 2024]$, spanning 76 distinct years). |
| 10 | `runtime` | Float | Official duration in minutes ($[40, 422]$, Mean = 134.5 min). |
| 11 | `genres` | String | Multi-label comma-separated genre string (116 combinations). |
| 12 | `primary_genre` | Categorical | Dominant genre category (16 standard film genres). |
| 13 | `genre_count` | Integer | Number of genres tagged to the film ($[1, 3]$, Mean = 2.1). |
| 14 | `country` | Categorical | Production country of origin (23 unique countries). |
| 15 | `country_category` | Categorical | Regional group (`Asian`, `English_Speaking`, `European`, `Other`). |
| 16 | `language` | Categorical | Primary spoken language of the film (19 languages). |
| 17 | `production_scale` | Categorical | Industry budget/distribution scale (`Independent`, `Studio`, `Epic`). |

---

## 🔬 Coursework Mapping (MMDS Syllabus)

* **Part C: Similarity and Distance Analysis:**
  * Computing high-dimensional Cosine and Jaccard distances between cinematic eras across genre, runtime, and country vectors.
  * Extracting $k$-shingles from review texts to detect near-duplicate reviews and verify how critical reception vocabulary shifts between modern and historical films using MinHashing and Locality-Sensitive Hashing (LSH).
* **Part D: Stream & Link Analysis:**
  * *Stream Processing:* Modeling incoming reviews over `review_date` as a continuous data stream, applying Reservoir Sampling and Bloom filters to detect burst sentiment shifts.
  * *Network Link Analysis:* Constructing a multi-partite graph connecting Countries $\leftrightarrow$ Eras $\leftrightarrow$ Genres, applying PageRank/HITS to identify structural cultural hubs over time.
* **Part E: Pattern Discovery:**
  * *Market Basket Analysis / Frequent Itemsets (Apriori):* Mining co-occurring pattern shifts (e.g., `{Era: 1960s, Country: Japan} $\Rightarrow$ {Genre: History}`).
  * *Clustering / Dimensionality Reduction:* Grouping films and viewing cohorts via BFR/CURE and SVD to uncover latent cinematic eras.

---

## 📁 Repository Structure

```
├── data/
│   ├── letterboxd_final_dataset.csv       # Final clean dataset (61,786 rows, 17 cols)
│   ├── letterboxd_final_dataset.parquet   # Optimized columnar storage
│   ├── letterboxd_cleaned_dataset.csv     # Intermediate cleaned dataset
│   ├── letterboxd_unified_dataset.csv     # Raw merged dataset
│   └── letterboxd_movies_dataset.csv      # Catalog metadata (16k+ films)
├── scripts/
│   ├── consolidate_datasets.py            # Initial merge and title normalization
│   ├── clean_final_dataset.py             # Pruning scraper artifacts & text normalization
│   ├── column_profile.py                  # Distinct counts and range profiler
│   └── check_user_rows.py                 # Data audit & verification utilities
├── Assignment Pattern.docx                # Academic assignment specification
└── README.md                              # Project documentation
```
