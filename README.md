# Letterboxd Unified Dataset 🎬

This dataset is a unified collection of Letterboxd movie reviews, ratings, and metadata.

## Overview
Initially, two separate datasets were found and merged to create a comprehensive view of movie reviews, incorporating user ratings, movie titles, genres, and metadata (like release year, runtime, and country of origin). 

## Data Cleaning Process
During the initial analysis of the merged dataset, we noticed some missing and empty values across certain columns:
- **Genres & Primary Genre**: ~14.8% missing
- **Ratings (`numeric_rating`, `raw_rating`)**: ~3.9% missing
- **Watch Status**: ~3.5% missing
- **Review Text & Dates**: Negligible missing rows (~5 rows)

Since these missing values were not present for all rows and incomplete data can interfere with analysis and machine learning tasks, **all rows containing any missing values were dropped**.

### Dataset Files
- **`data/letterboxd_unified_dataset.csv`**: The original merged dataset containing missing values (90,016 rows, 34 columns).
- **`data/letterboxd_cleaned_dataset.csv`**: The intermediate cleaned dataset with all `NaN` rows dropped (71,072 rows, 34 columns).
- **`data/letterboxd_final_dataset.csv`**: The finalized dataset with all heuristic flags, scraper artifacts (`Show All…`), and redundant columns removed (61,786 complete rows, 17 authentic core columns). Also exported as `data/letterboxd_final_dataset.parquet`.

## Next Steps / Potential Use Cases
This cleaned dataset can be used for a variety of tasks, such as:
1. **Sentiment Analysis**: Predicting movie ratings based on review text using NLP.
2. **Recommendation Systems**: Recommending similar movies based on genres, decades, and user ratings.
3. **Exploratory Data Analysis**: Exploring rating trends across different decades and genres.
