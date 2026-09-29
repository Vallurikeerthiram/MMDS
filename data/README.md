# 📊 Data Directory: Letterboxd Cinematic Dataset

This directory houses the raw, intermediate, and finalized datasets utilized across all MMDS modules.

---

## 📁 Dataset Catalog

| File Name | Format | Records | Size | Description |
|:---|:---:|:---:|:---:|:---|
| [`letterboxd_final_dataset.csv`](file:///c:/Users/keert/OneDrive%20-%20Amrita%20vishwa%20vidyapeetham/Amrita/Sem7/Projects/MMDS/data/letterboxd_final_dataset.csv) | CSV | 61,786 | ~70 MB | **Production Gold Standard Dataset** containing 17 clean, standardized columns. |
| [`letterboxd_final_dataset.parquet`](file:///c:/Users/keert/OneDrive%20-%20Amrita%20vishwa%20vidyapeetham/Amrita/Sem7/Projects/MMDS/data/letterboxd_final_dataset.parquet) | Parquet | 61,786 | ~40 MB | High-performance columnar format with Snappy compression for Spark/PySpark pipelines. |
| `letterboxd_movies_dataset.csv` | CSV | 16,000+ | ~3.5 MB | Film catalog metadata (theatrical release year, runtime, production countries, languages, and genres). |
| `letterboxd_unified_dataset.csv` | CSV | 61,786 | ~116 MB | Pre-deduplication merged corpus. |
| `letterboxd_cleaned_dataset.csv` | CSV | 61,786 | ~91 MB | Intermediate cleaned dataset prior to final field profiling. |
| `letterboxd_250movie_reviews.csv`| CSV | - | ~98 MB | Raw scrape corpus of top-rated film reviews. |

---

## 📑 Feature Dictionary (`letterboxd_final_dataset.csv`)

| Index | Column Name | Type | Description & Domain |
|:---:|:---|:---:|:---|
| 1 | `review_id` | String | Unique review identifier (`REV_000001` to `REV_061786`). |
| 2 | `movie_title` | String | Official film title (185 distinct films). |
| 3 | `rating` | Float | Star rating on a $[0.5, 5.0]$ scale (Mean = 4.49). |
| 4 | `review_date` | Date | Timestamp of viewing/review (`1896` to `2024`, 99.9% in 2012–2024). |
| 5 | `watch_status` | Categorical | Logged user viewing state (`Watched`, `Rewatched`). |
| 6 | `review_text` | String | Natural language review text written by users. |
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
