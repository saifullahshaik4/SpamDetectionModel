# Spam Email Detection with Gaussian Naive Bayes

This project implements a fully custom **Gaussian Naive Bayes classifier** to detect spam emails using the [UCI Spambase dataset](https://archive.ics.uci.edu/dataset/94/spambase). It incorporates advanced preprocessing, feature selection, parameter tuning, and ensemble learning to achieve over **91% accuracy** on the test set.

---

## Project Highlights

- **Custom Naive Bayes Implementation**
  - Built from scratch using only NumPy.
  - Stable handling of zero variance using a minimum standard deviation.
  - Inference based on log-probabilities to prevent underflow.

- **Advanced Preprocessing**
  - Log1p transformation to reduce skew.
  - Robust IQR-based scaling.
  - Outlier clipping to ±3 IQR.

- **Feature Selection**
  - Uses a signal-to-noise ratio to rank features by discriminative power.
  - Enables top-k feature selection for optimized performance.

- **Hyperparameter Optimization**
  - Grid search over combinations of `min_std`, `prior_smoothing`, and `top_k` features.
  - Evaluation based on validation accuracy.

- **Ensemble Learning**
  - Combines 7 Naive Bayes models trained on different feature subsets.
  - Final predictions are made by majority voting.

- **Evaluation**
  - Achieves:
    - Accuracy: **0.9124**
    - Precision: **0.8628**
    - Recall: **0.9246**
    - F1 Score: **0.8926**
  - Includes confusion matrix visualizations.

---

## File Structure

---

## How to Run

1. Clone this repository.
2. Open `notebook/SpamEmailDetection.ipynb` in Google Colab or Jupyter.
3. Ensure the following libraries are installed:
   - `numpy`
   - `pandas`
   - `scikit-learn`
   - `matplotlib`
4. Run all cells to train and evaluate the models.

---

## Dataset

UCI Spambase Dataset:  
[https://archive.ics.uci.edu/dataset/94/spambase](https://archive.ics.uci.edu/dataset/94/spambase)

---

## Course Information

This project was developed for **CS 445/545: Machine Learning (Spring 2025)** at Portland State University.
