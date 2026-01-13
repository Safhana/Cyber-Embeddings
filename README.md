# Custom Word Embeddings for Cybersecurity

This project implements **CBOW** and **Skip-gram** architectures to create 10-dimensional word embeddings using a specialized Cybersecurity corpus.

###  The Approach Tried
1. **Preprocessing:** Took a raw cybersecurity paragraph, removed punctuation, and tokenized it into a vocabulary.
2. **Dataset Generation:** - **CBOW:** Created context-target pairs using a sliding window of W=4.
   - **Skip-gram:** Created target-context pairs.
3. **Model Architecture:** A 2-layer Neural Network with an **Embedding Layer (10-D)** and a **Softmax Output Layer**.
4. **Optimization:** Tested three learning rates (0.01, 0.001, 0.0001) using the **Adam Optimizer**.
5. **Evaluation:** Used **Cosine Similarity** to measure the mathematical distance between word vectors to find similar terms.

### 📈 Results
The best performance was achieved with a **Learning Rate of 0.01**, where the loss converged from **3.81 to 0.77** within 100 epochs.



