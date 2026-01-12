# ======================================
# cosine_similarity.py
# ======================================

import numpy as np
import math

# -------------------------------
# STEP 1: Load Vocabulary
# -------------------------------
id_to_word = {}

with open("vocab.txt", "r") as f:
    for line in f:
        word, idx, freq = line.strip().split(",")
        id_to_word[int(idx)] = word

# -------------------------------
# STEP 2: Load Embeddings
# Change file if needed:
# embeddings_cbow.csv OR embeddings_skipgram.csv
# -------------------------------
embeddings = {}

with open("embeddings_cbow.csv", "r") as f:   # <-- change to embeddings_skipgram.csv if required
    next(f)  # skip header
    for line in f:
        parts = line.strip().split(",")
        word_id = int(parts[0])
        vector = np.array(list(map(float, parts[1:])))
        embeddings[word_id] = vector

# -------------------------------
# STEP 3: Cosine Similarity Function
# -------------------------------
def cosine_similarity(v1, v2):
    dot = np.dot(v1, v2)
    norm1 = math.sqrt(np.dot(v1, v1))
    norm2 = math.sqrt(np.dot(v2, v2))
    return dot / (norm1 * norm2 + 1e-8)

# -------------------------------
# STEP 4: Top-K Similar Words
# -------------------------------
def get_top_k(word, k=5):
    query_id = None
    for idx, w in id_to_word.items():
        if w == word:
            query_id = idx
            break

    if query_id is None:
        return []

    query_vec = embeddings[query_id]
    similarities = []

    for idx, vec in embeddings.items():
        if idx != query_id:
            sim = cosine_similarity(query_vec, vec)
            similarities.append((id_to_word[idx], sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:k]

# -------------------------------
# STEP 5: Query Words (Cybersecurity)
# -------------------------------
query_words = [
    "login",
    "malware",
    "authentication",
    "encryption",
    "access"
]

# -------------------------------
# STEP 6: Save Results
# -------------------------------
with open("similarity_results.txt", "w") as f:
    for word in query_words:
        f.write(f"Query word: {word}\n")
        results = get_top_k(word)
        for i, (w, s) in enumerate(results, start=1):
            f.write(f"{i}. {w} ({s:.4f})\n")
        f.write("\n")

print("Cosine similarity results saved to similarity_results.txt")
