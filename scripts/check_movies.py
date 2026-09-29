import pandas as pd

movies = pd.read_csv('data/letterboxd_movies_dataset.csv')
print("Unique movies in catalog:", len(movies))

# Let's inspect the movies in our dataset
df = pd.read_csv('data/letterboxd_final_dataset.csv')
unique_titles = df['movie_title'].unique()
print("Unique movies in our final dataset:", len(unique_titles))

# Check which movies have 'Show All'
show_all_movies = df[df['genres'].astype(str).str.contains('Show All', case=False, na=False)]['movie_title'].unique()
print("Movies with 'Show All':", len(show_all_movies), show_all_movies.tolist())

# Check movies with theme tags
theme_tags = ['Dreamlike', 'Gripping', 'Religious faith', 'Brutal', 'Terrifying', 'patriotism']
theme_movies = df[df['primary_genre'].isin(theme_tags)]['movie_title'].unique()
print("Movies with theme tags:", len(theme_movies), theme_movies.tolist())
