# Investigating BERT: An Ablation Study in Sentiment Classification

A systematic inference-time ablation study examining the contribution of 
attention heads and positional encodings to BERT's performance on binary 
sentiment classification.

## Overview

BERT-base contains 144 attention heads and relies on positional embeddings 
to encode word order. But how much do individual heads actually matter? And 
how much does word order contribute to sentiment prediction? This project 
investigates both questions by selectively disabling components at inference 
time — without retraining — and measuring the impact on classification 
accuracy on the SST-2 benchmark.

## Research Questions

1. Which attention heads contribute meaningfully to model performance, and 
   does importance vary across early, middle, and late layers?
2. To what extent does positional information influence sentiment 
   classification — can BERT still classify sentiment from token identity 
   alone?

## Dataset & Model

- **Dataset:** SST-2 (Stanford Sentiment Treebank), GLUE benchmark — binary 
  sentiment classification of movie review excerpts
- **Model:** `textattack/bert-base-uncased-SST-2` — BERT-base fine-tuned on 
  SST-2, ~110M parameters, baseline accuracy ~92%
- **Evaluation subset:** 200 validation samples (CPU constraint)

## Technologies

- Python 3.x
- PyTorch — model inference and hook registration
- Hugging Face Transformers & Datasets — model and data loading
- scikit-learn — accuracy, F1, classification report
- SciPy — paired t-tests, Bonferroni correction, Cohen's d
- Matplotlib, Seaborn — visualization

## Project Structure
```
notebooks/   - main Jupyter notebook (full ablation pipeline)
```

## What the Notebook Covers

- BERT architecture overview: multi-head attention, positional encoding, 
  and the mathematical decomposition of attention score interactions
- Baseline evaluation on SST-2 validation set
- **Experiment 1 — Attention Head Ablation:** forward hook-based zeroing of 
  9 representative heads (layers 0, 5, 11 × heads 0, 5, 11); layer-wise 
  importance comparison
- **Experiment 2 — Positional Encoding Ablation:** custom embedding forward 
  pass replacement that zeros positional embeddings while preserving token 
  and segment embeddings
- Statistical analysis: paired t-tests, Cohen's d effect size, Bonferroni 
  correction for multiple comparisons
- Visualization: per-head accuracy drop bar charts, layer-depth importance, 
  component importance summary

## Key Results

| Component                        | Accuracy Drop |
|----------------------------------|---------------|
| Positional encoding removed      | −13.4%        |
| Most critical head (L5-H11)      | −1.9%         |
| Average across 9 ablated heads   | ~−1.3%        |

Positional encoding is far more critical than any individual attention head. 
Multi-head attention contains substantial redundancy — no single head proves 
indispensable. Middle-layer heads (Layer 5) show the highest sensitivity, 
consistent with their role in encoding semantic structure relevant to 
sentiment.

## Notes

Final exam project for the Deep Learning course at SoftUni.
