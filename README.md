# Consumer Complaints Multi‑Class Text Classification

## Overview
This project builds and evaluates six supervised NLP classification models (three neural networks + three transformers) on the **CFPB Consumer Complaint Database**. The task is to predict the financial product category from a complaint narrative. The work was completed as part of an NLP assessment, focusing on performance comparison, trade‑offs, and ethical considerations.

## Repository Contents
- `Consumer_Complaint_Classification.ipynb` – Main Jupyter notebook with full pipeline: data loading, preprocessing, model training (6 models × 2 configurations), evaluation, and visualisations.
- `class_distribution.png` – Bar chart of product class distribution in the test set.
- `confusion_matrix.png` – Normalised confusion matrix for the best model (DistilBERT Config1).
- `README.md` – This file.

## Dataset
- **Source:** Consumer Financial Protection Bureau (CFPB) – Consumer Complaint Database.
- **Rows:** ~162,000 after cleaning.
- **Input:** Free‑text complaint narrative (preprocessed: lowercasing, punctuation/URL removal, stopword removal, lemmatisation).
- **Target:** Five product categories – credit_reporting, debt_collection, mortgages_and_loans, credit_card, retail_banking.
- **Class imbalance:** credit_reporting accounts for 56% of samples; macro and weighted F1 used as primary metrics.

## Models & Configurations

### Neural Networks (3 models, 2 configs each)
- **MLP** – Simple feedforward network (embedding → flatten → dense layers).
- **1‑D CNN** – Conv1D + global max pooling for local n‑gram patterns.
- **BiLSTM** – Bidirectional LSTM for long‑range dependencies.

Configurations varied embedding dimension, hidden units, filters/kernel sizes, and dropout rates.

### Transformers (3 models, 2 configs each)
- **DistilBERT** – Lightweight distilled BERT.
- **ALBERT** – Parameter‑efficient via factorised embeddings.
- **DeBERTa‑small** – Disentangled attention (required fp16=False due to compatibility).

All transformers fine‑tuned with **200 training samples** (subset due to GPU quota constraints), batch size 2, 1 epoch, max length 64.

## Results (Test Set – 24,337 samples)

| Model          | Config | Accuracy | Macro F1 | Weighted F1 |
|----------------|--------|----------|----------|--------------|
| MLP            | 2      | 0.88     | 0.84     | 0.88         |
| 1‑D CNN        | 2      | 0.87     | 0.82     | 0.86         |
| BiLSTM         | 2      | 0.86     | 0.81     | 0.85         |
| DistilBERT     | 1      | 0.86     | 0.81     | 0.86         |
| ALBERT         | 1      | 0.85     | 0.80     | 0.84         |
| DeBERTa‑small  | 1      | 0.10     | 0.04     | 0.02         |

- **Best overall:** MLP Config2 (88% accuracy, 0.88 weighted F1).
- **Transformers** performed on par with neural nets despite tiny training subset, except DeBERTa which failed (insufficient data / fp16 incompatibility).
- **Confusion matrix** shows main errors between `debt_collection` and `credit_card`.

## Visualisations
- `class_distribution.png` – Test set class distribution (stratified).
- `confusion_matrix.png` – Normalised confusion matrix for DistilBERT Config1.

## How to Run the Notebook
1. **Open in Google Colab** (recommended) or run locally with Jupyter.
2. **Upload the dataset** `Consumer Complaints Dataset for NLP.csv` when prompted (or place it in the same directory).
3. **Run all cells** sequentially.  
   - Neural network training will use the full training set (~113k samples).  
   - Transformer training uses a **200‑sample subset** (to respect memory limits).  
   - Final evaluation uses the full test set (24k samples).
  
## Ethical Considerations
Potential harms: Misrouting serious complaints could delay regulatory action; bias against minority product classes.

Safeguards implemented: Stratified splitting, macro F1 monitoring, human‑in‑the‑loop for low‑confidence predictions, logging, and model cards.

Full discussion in notebook (section d).

References

Kim, Y. (2014). Convolutional neural networks for sentence classification. EMNLP.

Sanh, V. et al. (2019). DistilBERT. NeurIPS EMC2 Workshop.

Lan, Z. et al. (2020). ALBERT. ICLR.

He, P. et al. (2021). DeBERTa. ICLR.

Mitchell, M. et al. (2019). Model cards for model reporting. FAccT.

Author
Rezene Ghebrehiwot – NLP_ Final_Report, St Mary's University, 03/04/2026

License
This project is licensed under the terms of the [MIT License](LICENSE)

text

---

**Instructions:**
1. Create a new file in your GitHub repository called `README.md`.
2. Copy the above content into it.
3. Replace `[Your Name]`, `[University Name]`, `[Date]` with your details.
4. If your results differ (e.g., CNN/BiLSTM numbers), update the table accordingly.
5. Commit and push the file.

You can also add a **badge** for the Colab link (optional). For example:
```markdown
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/Consumer_Complaint_Classification.ipynb)

### Dependencies
Install required libraries:
```bash
pip install pandas numpy scikit-learn tensorflow transformers datasets accelerate matplotlib seaborn nltk
