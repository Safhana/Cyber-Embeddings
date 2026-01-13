
import tensorflow as tf
import numpy as np
import ast


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

contexts = []
targets = []

with open("cbow_dataset.csv", "r") as f:
    next(f)
    for line in f:
        context_str, target_id = line.strip().rsplit(",", 1)
        context_ids = ast.literal_eval(context_str)
        targets.append(int(target_id))
        contexts.append(context_ids)

def one_hot(idx, size):
    v = np.zeros(size)
    v[idx] = 1
    return v

X, y = [], []

for context, target in zip(contexts, targets):
    context_vectors = [one_hot(i, vocab_size) for i in context]
    X.append(np.mean(context_vectors, axis=0))   # CBOW average
    y.append(one_hot(target, vocab_size))

X = np.array(X)
y = np.array(y)

learning_rates = [0.01, 0.001, 0.0001]
epochs = 100
embedding_dim = 10

for lr in learning_rates:
    print(f"\nTraining CBOW with learning rate = {lr}")

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(embedding_dim, input_shape=(vocab_size,)),
        tf.keras.layers.Dense(vocab_size, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy"
    )

    history = model.fit(X, y, epochs=epochs, verbose=0)

    
    with open(f"loss_cbow_lr_{lr}.txt", "w") as f:
        f.write("epoch,loss\n")
        for i, loss in enumerate(history.history["loss"], 1):
            f.write(f"{i},{loss}\n")

    if lr == 0.001:
        weights = model.layers[0].get_weights()[0]
        with open("embeddings_cbow.csv", "w") as f:
            f.write("word_id," + ",".join([f"e{i+1}" for i in range(embedding_dim)]) + "\n")
            for wid in range(vocab_size):
                f.write(str(wid) + "," + ",".join(map(str, weights[wid])) + "\n")

print("CBOW training complete")
