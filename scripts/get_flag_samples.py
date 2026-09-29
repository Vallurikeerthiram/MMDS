import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('data/letterboxd_final_dataset.csv')

def get_sample(target_flag):
    # Only target_flag is True, others False
    sub = df[df[target_flag] == True]
    other_flags = [f for f in ['mentions_format', 'mentions_audio', 'mentions_visuals', 'mentions_theatrical'] if f != target_flag]
    for of in other_flags:
        sub = sub[sub[of] == False]
    sub = sub[(sub['review_word_count'] >= 25) & (sub['review_word_count'] <= 80)]
    return sub.iloc[0]

flags = ['mentions_format', 'mentions_audio', 'mentions_visuals', 'mentions_theatrical']

for fl in flags:
    r = get_sample(fl)
    print(f"=== Sample where {fl} = True (Others = False) ===")
    print(f"Review ID: {r['review_id']}")
    print(f"Movie: {r['movie_title']} (Release Year: {int(r['release_year'])})")
    print(f"Rating: {r['rating']} stars | Date: {r['review_date']}")
    print(f"Flags -> format: {r['mentions_format']}, audio: {r['mentions_audio']}, visuals: {r['mentions_visuals']}, theatrical: {r['mentions_theatrical']}")
    print(f"Review Text:\n\"{r['review_text']}\"\n")

# Sample where ALL 4 are True
sub_all = df[df['mentions_format'] & df['mentions_audio'] & df['mentions_visuals'] & df['mentions_theatrical']]
sub_all = sub_all[(sub_all['review_word_count'] >= 40) & (sub_all['review_word_count'] <= 120)]
r_all = sub_all.iloc[0]
print(f"=== Sample where ALL 4 FLAGS ARE TRUE ===")
print(f"Review ID: {r_all['review_id']}")
print(f"Movie: {r_all['movie_title']} (Release Year: {int(r_all['release_year'])})")
print(f"Rating: {r_all['rating']} stars | Date: {r_all['review_date']}")
print(f"Flags -> format: {r_all['mentions_format']}, audio: {r_all['mentions_audio']}, visuals: {r_all['mentions_visuals']}, theatrical: {r_all['mentions_theatrical']}")
print(f"Review Text:\n\"{r_all['review_text']}\"")
