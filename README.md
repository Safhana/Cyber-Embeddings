# Custom Word Embeddings for Cybersecurity

This project implements **CBOW** and **Skip-gram** architectures to create 10-dimensional word embeddings using a specialized Cybersecurity corpus.

###  The Approach Tried

#### 1. Data Pipeline (`dataset_builder.py`)

* **Preprocessing:** Standardized the raw corpus by converting to lowercase and stripping punctuation.
* **Tokenization:** Generated a sequential list of tokens and a unique vocabulary mapping.
* **Windowing Logic:** Utilized a sliding window () to capture context.
* **CBOW:** Aggregated neighbor words to predict the center word.
* **Skip-gram:** Used the center word to predict individual neighbors.

#### 2. Model Architecture (`train_cbow.py` & `train_skipgram.py`)

The models were built using **Keras/TensorFlow** with the following layers:

* **Input:** One-hot encoded vectors (Size = Vocabulary Size).
* **Embedding Layer (Hidden):** A dense layer with **10 neurons**, representing the learned dimensions ( to ).
* **Output Layer:** A dense layer with **Softmax activation** to produce a probability distribution across the vocabulary.
* **Optimizer:** Adam (Adaptive Moment Estimation).

#### 3. Hyperparameter Experiments

Conducted training experiments across three different learning rates to evaluate convergence efficiency:

* **0.01 (Optimized):** Provided the best balance between speed and stability.
* **0.001 (Standard):** Produced slower convergence within the 100-epoch limit.
* **0.0001 (Slow):** Insufficient for meaningful weight updates in a small corpus.

### Performance & Results

#### Training Convergence

Our logs show that the **0.01 Learning Rate** was significantly more effective for this specific cybersecurity text:

* **Initial Loss:** 3.81
* **Final Loss:** **0.77** (after 100 epochs)
* **Status:** Successful Convergence

#### Similarity Evaluation (`cosine_similarity.py`)

Using **Cosine Similarity**, we queried words to find their nearest neighbors in the 10-D space.

* **Findings:** The model successfully grouped related concepts. For example, the query for **"malware"** returned terms like **"indicators"** and **"suspicious"** with high similarity scores.


