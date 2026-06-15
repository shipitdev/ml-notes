
---

tags: [machine-learning, cross-validation, model-evaluation, k-fold, stratified-k-fold, sampling-bias, overfitting] source: https://www.youtube.com/watch?v=7062skdX05Y related:

- "[[GridSearchCV and RandomizedSearchCV]]"
- "[[Precision, Recall, F1 Score]]"
- "[[R² and Adjusted R²]]"

---

# 🌳 Cross-Validation Frameworks & Sampling Mechanics

## ⚡ TL;DR

A single `train_test_split` is fragile — shift the `random_state` and your accuracy metric can swing wildly, proving it reflects the shuffle, not the model. **Cross-Validation (CV)** fixes this by rotating which data block acts as the test fold across $K$ sequential loops, then averaging the results into a stable **Mean Accuracy Profile**. For class-imbalanced data, **Stratified K-Fold** additionally forces every fold to preserve the original class ratio.

---

## 📌 The Big Picture: Bypassing the Random State Trap

```
[ Standard 1-Pass Split ]                   [ K-Fold Cross-Validation ]
┌──────────────────────────┐               ┌──────────────────────────────────────┐
│ Train (70%): shuffled    │               │ Fold 1: [TEST][Train][Train][Train]  │
├──────────────────────────┤  ──────────►  │ Fold 2: [Train][TEST][Train][Train]  │
│ Test  (30%): leftover    │               │ Fold 3: [Train][Train][TEST][Train]  │
└──────────────────────────┘               └──────────────────────────────────────┘
 (fluctuates wildly with random_state)      (averages folds → stable metric)
```

Changing `random_state=0` → `random_state=50` can swing accuracy from 85% → 87%. That swing is a data artifact, not a model property. CV eliminates this by ensuring **every row is used for both training and validation** across the full pipeline.

---

## 📐 Mathematical Framework

### 1. Leave-One-Out CV (LOOCV)

Sets $K = N$ — each loop holds out exactly one row: $$\mathcal{D}_{\text{train}}^{(i)} = \mathcal{D} \setminus {X_i}, \quad \mathcal{D}_{\text{val}}^{(i)} = {X_i}$$ $$\text{LOOCV Error} = \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}\big(f(X_i;, \theta^{(i)}),, y_i\big)$$

> ⚠️ Nearly identical training sets across folds → highly correlated models → **high variance** in scores despite low bias.

### 2. K-Fold Partitioning

Splits $\mathcal{D}$ into $K$ equal, mutually exclusive blocks. Each fold $k$: $$\text{Validation size} = \frac{N}{K}$$ $$\mathcal{D}_{\text{val}}^{(k)} = \mathcal{D}_k, \quad \mathcal{D}_{\text{train}}^{(k)} = \mathcal{D} \setminus \mathcal{D}_k$$

### 3. Stratified Class Alignment

For binary labels with global positive ratio $p = \frac{|{y=1}|}{N}$, every fold $k$ is forced to satisfy: $$\frac{\big|{y_j = 1 \mid X_j \in \mathcal{D}_k}\big|}{|\mathcal{D}_k|} \approx p$$

Ensures minority classes appear in every fold at the same rate as the full dataset.

---

## 🧮 Numerical Walkthrough (1,000-Row Matrix, K=5)

|Pass|Test Block|Train Block|Output|
|---|---|---|---|
|Fold 1|Rows 1–200|Rows 201–1000|Accuracy₁|
|Fold 2|Rows 201–400|Rows 1–200 ∪ 401–1000|Accuracy₂|
|Fold 3|Rows 401–600|Rows 1–400 ∪ 601–1000|Accuracy₃|
|Fold 4|Rows 601–800|Rows 1–600 ∪ 801–1000|Accuracy₄|
|Fold 5|Rows 801–1000|Rows 1–800|Accuracy₅|

**Final metric** = mean of all 5 accuracy scores → stable, bias-resistant performance estimate.

---

## 💻 Python Pipeline

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

def execute_cross_validation_audit_pipeline(X, y, fold_count=5):
    base_estimator = LogisticRegression(random_state=42)

    # Standard K-Fold
    kfold_engine = KFold(n_splits=fold_count, shuffle=True, random_state=42)
    kfold_scores = cross_val_score(base_estimator, X, y, cv=kfold_engine, scoring='accuracy')

    # Stratified K-Fold (preserves class ratios)
    stratified_engine = StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=42)
    stratified_scores = cross_val_score(base_estimator, X, y, cv=stratified_engine, scoring='accuracy')

    comparison_summary = pd.DataFrame({
        'Fold_Index': range(1, fold_count + 1),
        'Standard_KFold_Accuracy': kfold_scores,
        'Stratified_KFold_Accuracy': stratified_scores
    })

    return comparison_summary, kfold_scores.mean(), stratified_scores.mean()
```

### Expected Output (85% / 15% imbalanced dataset)

```
 Fold_Index  Standard_KFold_Accuracy  Stratified_KFold_Accuracy
          1                 0.862500                   0.854167
          2                 0.829167                   0.850000
          3                 0.870833                   0.858333
          4                 0.841667                   0.854167
          5                 0.845833                   0.850000
---------------------------------------------------------------------
Standard K-Fold Global Mean:    85.00%
Stratified K-Fold Global Mean:  85.33%  ← tighter variance across folds
```

---

## 🌍 Real-World Use Cases

### Simple — Marketing Response Classifiers

Validate demographic response predictors across small consumer groups. Stratification ensures minority high-income buyers appear in every fold.

### Advanced — Financial Stock Trend Classification

Validate multi-axis sequential features predicting long-range price changes.

> ⚠️ **Critical Stress Point — Data Leakage:** Never apply standard K-Fold or Stratified K-Fold to time-series data. Shuffling mixes future records into past training loops — the model memorises the future and collapses on live deployment. Use **`TimeSeriesSplit`** instead (rolling walk-forward windows only).

---

## ⚠️ Gotchas

> [!warning] Never Scale Before Splitting Fitting `StandardScaler` on the full dataset before CV causes **data leakage** — the scaler has already seen the test fold's statistics. Always fit the scaler inside the training fold only, never on the full matrix before splitting.

> [!warning] Time-Series Invalidation Standard K-Fold shuffles rows, destroying temporal ordering. Use `TimeSeriesSplit` for any sequential data (price logs, sensor streams, trading signals).

> [!warning] High CV Score ≠ No Overfitting If preprocessing (scaling, encoding) is applied globally before folds are created, high CV scores can still reflect leakage rather than genuine generalisation.

---

## 📊 CV Strategy Comparison

|Framework|Complexity|Bias / Variance Risk|Best For|
|---|---|---|---|
|**LOOCV**|$O(N)$ — retrains $N$ times|Low bias, **high variance**|Small high-stakes datasets (<100 rows)|
|**Standard K-Fold**|Medium — $K$ loops|Risk of sampling bias on imbalanced data|Large, balanced class distributions|
|**Stratified K-Fold**|Medium — $K$ loops + label tracking|Low bias, stable variance|Imbalanced classification (fraud, anomaly detection)|
|**TimeSeriesSplit**|Low-to-medium — expanding window|Prevents look-ahead leakage|Sequential data, trading pipelines, time-series|

---

## ❓ Active Recall Questions

1. Why does fitting `StandardScaler` on the full dataset _before_ creating CV folds constitute data leakage?
2. How does choosing $K=5$ vs $K=20$ affect the bias-variance trade-off in your validation scores?
3. What causes a model to score high on CV training folds but fail on live production data?
4. Why does LOOCV produce high-variance scores despite using nearly the full dataset for training in each fold?

---

## 🔗 Related Notes

- [[GridSearchCV and RandomizedSearchCV]]
- [[Precision, Recall, F1 Score]]
- [[R² and Adjusted R²]]
- [[TimeSeriesSplit]]