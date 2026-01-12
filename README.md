# Custom Word Embeddings for Cybersecurity

This project implements **CBOW** and **Skip-gram** architectures to create 10-dimensional word embeddings using a specialized Cybersecurity corpus.

## 📊 Experimental Results (100 Epochs)
| Learning Rate | Initial Loss | Final Loss | Status |
| :--- | :--- | :--- | :--- |
| **0.01** | 3.80 | **0.77** | **Best Performance** |
| 0.001 | 3.81 | 3.57 | Under-fitted |
| 0.0001 | 3.81 | 3.79 | No Learning |

## 🚀 How to Run
1. Run `python Datasets.py` to generate the vocabulary and windowed datasets.
2. Run `python Train_cbow.py` to train the model and save the 10-D vectors.
3. Run `python cosine_similarity.py` to find related security terms.

## 🧠 Key Concepts
- **Dimension size:** 10 (e1 to e10)
- **Optimizer:** Adam (Adaptive Learning Rates)
- **Domain:** Cybersecurity terminology (Firewall, Malware, Authentication)
