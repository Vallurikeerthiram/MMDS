"""
Locality-Sensitive Hashing (LSH) and MinHash Pipeline for Text Corpora
======================================================================
MMDS Part C: Shingling, MinHashing, and Banded LSH Implementation.

This module implements:
1. k-Shingle Extraction (Character & Word level).
2. Universal Hash Function Generation: h_i(x) = (a_i * x + b_i) mod p.
3. MinHash Signature Matrix generation for documents.
4. Banded Locality-Sensitive Hashing (LSH):
   - Parameters: n hash functions, b bands, r rows per band (n = b * r).
   - Candidate pair generation via hash collisions in buckets.
5. Theoretical S-Curve Analysis:
   - S-curve formula: P(collision) = 1 - (1 - s^r)^b
   - Empirical vs. Theoretical threshold approximation: t ~= (1/b)^(1/r).
6. False Positive / False Negative trade-off evaluation.
"""

import os
import random
import re
import numpy as np
import pandas as pd

def clean_text(text):
    if not isinstance(text, str):
        return ""
    clean = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    return ' '.join(clean.split())

def generate_shingles(text, k=5):
    """Generate k-character shingles from text."""
    clean = clean_text(text)
    if len(clean) < k:
        return set([clean]) if clean else set()
    return set(clean[i:i+k] for i in range(len(clean) - k + 1))

def compute_exact_jaccard(set1, set2):
    union = set1.union(set2)
    if not union:
        return 1.0
    return len(set1.intersection(set2)) / len(union)

class MinHashLSH:
    def __init__(self, num_perm=100, num_bands=20, prime=4294967311, seed=42):
        """
        num_perm (n): Number of hash functions.
        num_bands (b): Number of bands.
        r = n // b: Rows per band.
        """
        assert num_perm % num_bands == 0, "num_perm must be divisible by num_bands"
        self.n = num_perm
        self.b = num_bands
        self.r = num_perm // num_bands
        self.prime = prime
        self.threshold = (1.0 / self.b) ** (1.0 / self.r)
        
        # Universal hashing coefficients: h(x) = (a*x + b) % prime
        rng = np.random.default_rng(seed)
        self.a_coeffs = rng.integers(1, self.prime - 1, size=self.n, dtype=np.int64)
        self.b_coeffs = rng.integers(0, self.prime - 1, size=self.n, dtype=np.int64)

    def compute_signature(self, shingles_set):
        """Compute MinHash signature vector of length n."""
        if not shingles_set:
            return np.zeros(self.n, dtype=np.int64)
        
        # Map shingles to 32-bit integer hashes
        shingle_hashes = np.array([hash(s) & 0xFFFFFFFF for s in shingles_set], dtype=np.int64)
        
        # Vectorized min-hashing across all n hash functions
        # signature[i] = min_{x in shingles} ((a_i * x + b_i) % prime)
        signature = np.full(self.n, np.iinfo(np.int64).max, dtype=np.int64)
        for i in range(self.n):
            a = self.a_coeffs[i]
            b = self.b_coeffs[i]
            hashed_values = (a * shingle_hashes + b) % self.prime
            signature[i] = np.min(hashed_values)
            
        return signature

    def estimate_jaccard(self, sig1, sig2):
        """Estimate Jaccard similarity from MinHash signatures: Sim = fraction of matching components."""
        return np.mean(sig1 == sig2)

    def get_band_buckets(self, doc_id, signature):
        """Hash each band of r rows to a bucket ID."""
        band_keys = []
        for band_idx in range(self.b):
            start = band_idx * self.r
            end = start + self.r
            band_slice = tuple(signature[start:end])
            bucket_hash = hash((band_idx, band_slice))
            band_keys.append((band_idx, bucket_hash))
        return band_keys

    def s_curve_prob(self, s):
        """Theoretical probability that two docs with Jaccard similarity s become candidate pairs."""
        return 1.0 - (1.0 - s ** self.r) ** self.b

def main():
    print("=" * 80)
    print("LOCALITY-SENSITIVE HASHING (LSH) & MINHASH SYSTEM VERIFICATION")
    print("=" * 80)

    # LSH Config: n=100, b=20, r=5 -> threshold ~= 0.549
    lsh = MinHashLSH(num_perm=100, num_bands=20, seed=42)
    print(f"MinHash Signatures (n) : {lsh.n}")
    print(f"Number of Bands (b)    : {lsh.b}")
    print(f"Rows per Band (r)      : {lsh.r}")
    print(f"Calculated Threshold t : (1/{lsh.b})^(1/{lsh.r}) = {lsh.threshold:.4f} (~{lsh.threshold*100:.1f}%)")

    # Display Theoretical S-Curve Table
    s_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 0.95]
    print("\n" + "-" * 80)
    print("THEORETICAL S-CURVE CANDIDATE PROBABILITIES: P(collision) = 1 - (1 - s^r)^b")
    print("-" * 80)
    print(f"{'Similarity (s)':<18}{'P(Candidate)':<18}{'Bucket Filter Status':<30}")
    print("-" * 80)
    curve_data = []
    for s in s_vals:
        p = lsh.s_curve_prob(s)
        status = "Filtered Out (>99.9% rejected)" if p < 0.01 else ("Transition Region" if p < 0.90 else "Captured as Candidate Pair")
        curve_data.append({'similarity': s, 'p_candidate': round(p, 4), 'status': status})
        print(f"{s:<18.2f}{p:<18.4f}{status:<30}")

    # Load Sample Documents from dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'letterboxd_final_dataset.csv')
    df = pd.read_csv(data_path)

    # Pick 4 diverse reviews:
    # 1 & 2: Same film (Harakiri), different reviews
    # 3: Parasite review
    # 4: Harakiri review #1 slightly mutated (synthetic near-duplicate)
    r1 = df[df['review_id'] == 'REV_000058'].iloc[0]
    r2 = df[df['review_id'] == 'REV_000080'].iloc[0]
    r3 = df[df['movie_title'] == 'Parasite'].iloc[0]
    
    # Synthetic near-duplicate of r1
    text1 = r1['review_text']
    text2 = r2['review_text']
    text3 = r3['review_text']
    text4 = text1 + " Truly an absolute masterpiece of cinema history."

    docs = {
        'Doc_1 (Harakiri Early)': text1,
        'Doc_2 (Harakiri Recent)': text2,
        'Doc_3 (Parasite Review)': text3,
        'Doc_4 (Doc_1 Near-Duplicate)': text4
    }

    # Extract shingles and compute MinHash signatures
    shingle_dict = {}
    sig_dict = {}
    for name, text in docs.items():
        s = generate_shingles(text, k=4)
        shingle_dict[name] = s
        sig_dict[name] = lsh.compute_signature(s)

    # Compare Pairwise Exact vs MinHash
    print("\n" + "=" * 80)
    print("EMPIRICAL COMPARISON: EXACT JACCARD VS MINHASH ESTIMATION")
    print("=" * 80)
    
    names = list(docs.keys())
    pair_results = []
    
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d1, d2 = names[i], names[j]
            exact_j = compute_exact_jaccard(shingle_dict[d1], shingle_dict[d2])
            est_j = lsh.estimate_jaccard(sig_dict[d1], sig_dict[d2])
            
            # Check LSH candidate collision
            b1 = lsh.get_band_buckets(d1, sig_dict[d1])
            b2 = lsh.get_band_buckets(d2, sig_dict[d2])
            collided = any(b1[k] == b2[k] for k in range(lsh.b))
            
            pair_results.append({
                'Pair': f"{d1} vs {d2}",
                'Exact Jaccard': round(exact_j, 4),
                'MinHash Est': round(est_j, 4),
                'Abs Error': round(abs(exact_j - est_j), 4),
                'LSH Candidate': 'YES' if collided else 'NO (Pruned)'
            })

    pair_df = pd.DataFrame(pair_results)
    print(pair_df.to_string(index=False))

    # Export LSH verification metrics
    lsh_csv = os.path.join(script_dir, 'lsh_s_curve_metrics.csv')
    pd.DataFrame(curve_data).to_csv(lsh_csv, index=False)
    print(f"\n[OK] LSH S-Curve metrics saved to: {lsh_csv}")

if __name__ == '__main__':
    main()
