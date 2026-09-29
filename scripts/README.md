# 🛠️ Data Processing, Cleaning & Diagnostic Scripts

This directory contains Python utilities developed for data ingestion, auditing, schema harmonization, text cleaning, and temporal sample diagnostics.

---

## 📋 Script Inventory

### 1. Data Cleaning & Harmonization
- [`consolidate_datasets.py`](file:///c:/Users/keert/OneDrive%20-%20Amrita%20vishwa%20vidyapeetham/Amrita/Sem7/Projects/MMDS/scripts/consolidate_datasets.py): Merges raw Letterboxd review scrapes with catalog metadata, resolving title discrepancies and unifying schemas.
- [`clean_final_dataset.py`](file:///c:/Users/keert/OneDrive%20-%20Amrita%20vishwa%20vidyapeetham/Amrita/Sem7/Projects/MMDS/scripts/clean_final_dataset.py): Normalizes review dates, prunes scraper artifacts, filters malformed rows, computes word/character lengths, and exports the final CSV and Parquet files.

### 2. Quality Audits & Profiling
- [`column_profile.py`](file:///c:/Users/keert/OneDrive%20-%20Amrita%20vishwa%20vidyapeetham/Amrita/Sem7/Projects/MMDS/scripts/column_profile.py): Generates comprehensive statistics: unique counts, data types, min/max bounds, and null counts.
- [`audit_empty_cells.py`](file:///c:/Users/keert/OneDrive%20-%20Amrita%20vishwa%20vidyapeetham/Amrita/Sem7/Projects/MMDS/scripts/audit_empty_cells.py): Scans for whitespace-only strings, NaNs, and incomplete entries.
- [`audit_data.py`](file:///c:/Users/keert/OneDrive%20-%20Amrita%20vishwa%20vidyapeetham/Amrita/Sem7/Projects/MMDS/scripts/audit_data.py): Verifies total record counts and duplicate review IDs.
- [`check_user_rows.py`](file:///c:/Users/keert/OneDrive%20-%20Amrita%20vishwa%20vidyapeetham/Amrita/Sem7/Projects/MMDS/scripts/check_user_rows.py): Inspects review date distributions across decades and user behavior patterns.
- [`check_movies.py`](file:///c:/Users/keert/OneDrive%20-%20Amrita%20vishwa%20vidyapeetham/Amrita/Sem7/Projects/MMDS/scripts/check_movies.py): Diagnostic verification of movie titles across catalog metadata.

### 3. Sampling & Similarity Diagnostics
- [`test_temporal_similarity.py`](file:///c:/Users/keert/OneDrive%20-%20Amrita%20vishwa%20vidyapeetham/Amrita/Sem7/Projects/MMDS/scripts/test_temporal_similarity.py): Computes exact Jaccard and Cosine metrics between early (2012) and modern (2024) reviews of the same movie.
- [`find_temporal_pair.py`](file:///c:/Users/keert/OneDrive%20-%20Amrita%20vishwa%20vidyapeetham/Amrita/Sem7/Projects/MMDS/scripts/find_temporal_pair.py): Identifies candidate pairs of reviews separated by a decade or more.
- [`pull_random_samples.py`](file:///c:/Users/keert/OneDrive%20-%20Amrita%20vishwa%20vidyapeetham/Amrita/Sem7/Projects/MMDS/scripts/pull_random_samples.py): Generates reproducible random samples for validation.

---

## 🚀 Execution Example

```bash
# Profile the final clean dataset
python scripts/column_profile.py

# Run temporal similarity check between early and modern reviews
python scripts/test_temporal_similarity.py
```
