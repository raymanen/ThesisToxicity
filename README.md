# Comparative Toxicity Classification

Empirical comparison of three text classifiers on toxic comment detection under natural class imbalance.

## Aim
Compare **TF-IDF + Logistic Regression**, **TF-IDF + Linear SVM**, and a fine-tuned **DistilBERT** on toxicity classification, and study how each model benefits from more training data.

## Data
- **Civil Comments** dataset (Borkan et al., 2019), loaded via the Hugging Face `datasets` library.
- Binary label: `toxic = 1 if toxicity >= 0.5 else 0` (≈ 8% positive).
- Fixed nested training subsets of **25k / 100k / 400k**, a frozen **20k validation** set, and the official **test** split.

## Input / Output
- **Input:** raw comment text (string).
- **Output:** model score in `[0, 1]` (or SVM decision value) and a binary prediction `{0, 1}` using a per-model threshold tuned on validation.

## Metrics
- Primary: **PR-AUC / Average Precision**
- Secondary: **Macro-F1**

## How to run
1. Open `ToxicityThesisLastVersion.ipynb` in **Google Colab** (GPU runtime recommended for DistilBERT).
2. Mount Google Drive — the notebook reads/writes under `MyDrive/ToxicityThesis/` (creates `data/splits`, `models`, `predictions`, `metrics`, `figures`, `tables`, `logs` on first run).
3. Run the cells **in order**. Section 6 builds and freezes the splits to `data/splits/*.parquet`; later sections load them back and train each model.
4. Trained models, predictions, metrics, and figures are saved to the matching subfolders for inspection.

See Markdowns, code cells, and results inline in the notebook.
