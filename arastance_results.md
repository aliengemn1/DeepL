# Arastance Model Results & Comparison

This report summarizes the performance of SVM, CNN, LSTM, BiLSTM, and AraBERT models on the Arastance dataset. The metrics used are **Accuracy**, **Precision**, **Recall**, and **F1-score**.

## Model Performance Chart

![Arastance Model Comparison](arastance_model_comparison.png)

## Metrics Table

| Model | Accuracy | F1 Macro | F1 Weighted | Precision Macro | Precision Weighted | Recall Macro | Recall Weighted |
|---|---|---|---|---|---|---|---|
| SVM | 0.7561 | 0.6790 | 0.7364 | 0.7712 | 0.7658 | 0.6451 | 0.7561 |
| CNN | 0.7683 | 0.6961 | 0.7576 | 0.7250 | 0.7570 | 0.6810 | 0.7683 |
| LSTM | 0.5549 | 0.2379 | 0.3960 | 0.1850 | 0.3079 | 0.3333 | 0.5549 |
| BiLSTM | 0.6951 | 0.6270 | 0.6977 | 0.6690 | 0.7264 | 0.6193 | 0.6951 |
| AraBERT | 0.7744 | 0.7028 | 0.7626 | 0.7471 | 0.7704 | 0.6799 | 0.7744 |

---

## How to Interpret

- **Accuracy**: Overall correctness of the model.
- **Precision**: How many selected items are relevant.
- **Recall**: How many relevant items are selected.
- **F1-score**: Harmonic mean of precision and recall.
- **Macro avg**: Average metric across classes, treating all classes equally.
- **Weighted avg**: Average metric across classes, weighted by support (number of true instances per class).
