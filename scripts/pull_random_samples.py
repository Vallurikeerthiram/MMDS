import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('data/letterboxd_final_dataset.csv')

flags = ['mentions_format', 'mentions_audio', 'mentions_visuals', 'mentions_theatrical']

for fl in flags:
    print('======================================================================')
    print(f'>>> 5 RANDOM SAMPLES WHERE {fl} = TRUE <<<')
    print('======================================================================')
    sample = df[df[fl] == True].sample(5, random_state=42)
    for idx, (_, r) in enumerate(sample.iterrows(), 1):
        txt = str(r['review_text']).replace('\n', ' ')
        if len(txt) > 220:
            txt = txt[:220] + '...'
        print(f"{idx}. [{r['review_id']}] Movie: \"{r['movie_title']}\" (Release Year: {int(r['release_year'])}) | Rating: {r['rating']} stars | Date: {r['review_date']}")
        print(f"   Flags: Format={r['mentions_format']}, Audio={r['mentions_audio']}, Visuals={r['mentions_visuals']}, Theatrical={r['mentions_theatrical']}")
        print(f"   Review Snippet: \"{txt}\"\n")

print('======================================================================')
print('>>> 5 RANDOM SAMPLES WHERE ALL 4 FLAGS ARE TRUE <<<')
print('======================================================================')
all_true = df[df['mentions_format'] & df['mentions_audio'] & df['mentions_visuals'] & df['mentions_theatrical']]
sample_all = all_true.sample(5, random_state=42)
for idx, (_, r) in enumerate(sample_all.iterrows(), 1):
    txt = str(r['review_text']).replace('\n', ' ')
    if len(txt) > 220:
        txt = txt[:220] + '...'
    print(f"{idx}. [{r['review_id']}] Movie: \"{r['movie_title']}\" (Release Year: {int(r['release_year'])}) | Rating: {r['rating']} stars | Date: {r['review_date']}")
    print(f"   Flags: Format={r['mentions_format']}, Audio={r['mentions_audio']}, Visuals={r['mentions_visuals']}, Theatrical={r['mentions_theatrical']}")
    print(f"   Review Snippet: \"{txt}\"\n")
