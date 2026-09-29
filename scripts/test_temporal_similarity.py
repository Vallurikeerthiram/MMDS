import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('data/letterboxd_final_dataset.csv')
r1 = df[df['review_id'] == 'REV_000058'].iloc[0]
r2 = df[df['review_id'] == 'REV_000080'].iloc[0]

def get_char_shingles(text, k=3):
    clean = ''.join(c.lower() for c in text if c.isalnum() or c.isspace())
    clean = ' '.join(clean.split())
    return set(clean[i:i+k] for i in range(len(clean) - k + 1))

s1 = get_char_shingles(r1['review_text'], k=3)
s2 = get_char_shingles(r2['review_text'], k=3)
inter = s1.intersection(s2)
union = s1.union(s2)

j_sim = len(inter) / len(union)
j_dist = 1.0 - j_sim

# Word-level
w1 = set(r1['review_text'].lower().split())
w2 = set(r2['review_text'].lower().split())
wj_sim = len(w1.intersection(w2)) / len(w1.union(w2))
wj_dist = 1.0 - wj_sim

# TF-IDF Cosine
vec = TfidfVectorizer(stop_words='english')
tfidf = vec.fit_transform([r1['review_text'], r2['review_text']])
cos_sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
cos_dist = 1.0 - cos_sim

print("=== RECORD 1 (Early Review: 2012) ===")
print("ID:", r1['review_id'], "| Date:", r1['review_date'], "| Rating:", r1['rating'])
print("Text:", r1['review_text'])

print("\n=== RECORD 2 (Recent Review: 2024) ===")
print("ID:", r2['review_id'], "| Date:", r2['review_date'], "| Rating:", r2['rating'])
print("Text:", r2['review_text'])

print("\n=== SHINGLING RESULTS (k=3) ===")
print(f"|S1| = {len(s1)}, |S2| = {len(s2)}")
print(f"|S1 ∩ S2| = {len(inter)}, |S1 ∪ S2| = {len(union)}")
print("Sample shared shingles (first 10):", sorted(list(inter))[:10])

print("\n=== METRIC VALUES ===")
print(f"Char 3-Shingle Jaccard Similarity = {j_sim:.4f}")
print(f"Char 3-Shingle Jaccard Distance   = {j_dist:.4f}")
print(f"Word Bag Jaccard Similarity       = {wj_sim:.4f}")
print(f"Word Bag Jaccard Distance         = {wj_dist:.4f}")
print(f"TF-IDF Cosine Similarity         = {cos_sim:.4f}")
print(f"TF-IDF Cosine Distance           = {cos_dist:.4f}")
