
cyber_corpus = """
A security team monitors logs for suspicious login attempts, abnormal data exfiltration,
and malware indicators. Multi-factor authentication and least-privilege policies reduce
account takeover risk. Incident response includes containment, forensic imaging, and
patch deployment to close vulnerabilities. Regular audits check encryption, backup
integrity, and compliance with access control standards.
"""
import string

cyber_corpus = cyber_corpus.lower()

cyber_corpus = cyber_corpus.translate(str.maketrans('', '', string.punctuation))

tokens = cyber_corpus.split()

print(tokens)

from collections import Counter

word_freq = Counter(tokens)

vocab = {}
for idx, word in enumerate(word_freq.keys()):
    vocab[word] = idx

for word, idx in vocab.items():
    print(word, idx, word_freq[word])

with open("vocab.txt", "w") as f:
    for word, idx in vocab.items():
        f.write(f"{word},{idx},{word_freq[word]}\n")

print("Vocabulary file saved as vocab.txt")


tokens   
vocab    

W = 4  # context window size
cbow_data = []

for i in range(len(tokens)):
    target_word = tokens[i]
    target_id = vocab[target_word]

    context_ids = []

    
    for j in range(i - W, i + W + 1):
        if j != i and j >= 0 and j < len(tokens):
            context_ids.append(vocab[tokens[j]])

    
    if len(context_ids) > 0:
        cbow_data.append((context_ids, target_id))

with open("cbow_dataset.csv", "w") as f:
    f.write("context_ids,target_id\n")
    for context, target in cbow_data:
        f.write(f"{context},{target}\n")

skipgram_data = []

for i in range(len(tokens)):
    target_word = tokens[i]
    target_id = vocab[target_word]

    for j in range(i - W, i + W + 1):
        if j != i and j >= 0 and j < len(tokens):
            context_word = tokens[j]
            context_id = vocab[context_word]
            skipgram_data.append((target_id, context_id))

with open("skipgram_dataset.csv", "w") as f:
    f.write("target_id,context_id\n")
    for target, context in skipgram_data:
        f.write(f"{target},{context}\n")
