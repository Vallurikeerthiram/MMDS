import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('data/letterboxd_final_dataset.csv')
sub = df[df['movie_title'] == 'Harakiri'].sort_values('review_date')

print("--- Earliest Reviews of Harakiri ---")
for idx, r in sub.head(5).iterrows():
    if len(r['review_text'].split()) >= 20:
        print(f"ID: {r['review_id']} | Date: {r['review_date']} | Words: {r['review_word_count']} | Rating: {r['rating']}")
        print(f"Text: {r['review_text'][:160]}...\n")

print("--- Latest Reviews of Harakiri ---")
for idx, r in sub.tail(5).iterrows():
    if len(r['review_text'].split()) >= 20:
        print(f"ID: {r['review_id']} | Date: {r['review_date']} | Words: {r['review_word_count']} | Rating: {r['rating']}")
        print(f"Text: {r['review_text'][:160]}...\n")
