# =========================================
# train_skipgram.py
# =========================================

import tensorflow as tf
import numpy as np

# -------------------------------
# STEP 1: Load Vocabulary
# -------------------------------
vocab = {}
id_to_word = {}

with open("vocab.txt", "r") as f:
    for line in f:
        word, idx, freq = line.strip().split(",")
        idx = int(idx)
        vocab[word] = idx
        id_to_word[idx] = word

vocab_size = len(vocab)
print("Vocabulary size:", vocab_size)

# -------------------------------
# STEP 2: Load Skip-gram Dataset
# -------------------------------
targets = []
contexts = []

with open("skipgram_dataset.csv", "r") as f:
    next(f)  # skip header
    for line in f:
        target_id, context_id = line.strip().split(",")
        targets.append(int(target_id))
        contexts.append(int(context_id))

# -------------------------------
# STEP 3: One-Hot Encoding
# -------------------------------
def one_hot(index, size):
    vec = np.zeros(size)
    vec[index] = 1
    return vec

X = np.array([one_hot(t, vocab_size) for t in targets])
y = np.array([one_hot(c, vocab_size) for c in contexts])

print("Input shape:", X.shape)
print("Output shape:", y.shape)

# -------------------------------
# STEP 4: Train with Learning Rates
# -------------------------------
learning_rates = [0.01, 0.001, 0.0001]
epochs = 100
embedding_dim = 10

for lr in learning_rates:
    print(f"\nTraining Skip-gram with learning rate = {lr}")

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(
            embedding_dim,
            activation=None,
            input_shape=(vocab_size,)
        ),
        tf.keras.layers.Dense(
            vocab_size,
            activation="softmax"
        )
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy"
    )

    history = model.fit(X, y, epochs=epochs, verbose=0)

    # -------------------------------
    # STEP 5: Save Loss
    # -------------------------------
    loss_file = f"loss_skipgram_lr_{lr}.txt"
    with open(loss_file, "w") as f:
        f.write("epoch,loss\n")
        for i, loss in enumerate(history.history["loss"], start=1):
            f.write(f"{i},{loss}\n")

    print(f"Saved loss to {loss_file}")

    # -------------------------------
    # STEP 6: Save Embeddings (best LR)
    # -------------------------------
    if lr == 0.001:
        embedding_weights = model.layers[0].get_weights()[0]

        with open("embeddings_skipgram.csv", "w") as f:
            f.write(
                "word_id," +
                ",".join([f"e{i+1}" for i in range(embedding_dim)]) +
                "\n"
            )
            for word_id in range(vocab_size):
                vector = embedding_weights[word_id]
                f.write(
                    f"{word_id}," +
                    ",".join(map(str, vector)) +
                    "\n"
                )

        print("Saved embeddings_skipgram.csv (LR = 0.001)")

print("Skip-gram training complete")

