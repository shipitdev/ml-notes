# 📊 Machine Learning: Imbalanced Data — Under-Sampling & Over-Sampling Foundations

---

## ⚡ TL;DR

A dataset is **imbalanced** when one target class heavily outnumbers the other (e.g., 990 legitimate transactions vs. 10 fraudulent ones). If you train a model on raw imbalanced data, it will cheat to achieve high accuracy by simply guessing the majority class every time — failing completely to catch the rare minority events.

- **Under-Sampling** fixes this by **deleting records from the majority class** until it matches the small size of the minority class.
- **Over-Sampling** fixes this by **replicating or synthetically creating new examples of the minority class** until it matches the size of the majority class.

> **Baseline Rule:** Under-Sampling should only be used if you possess a massive dataset with millions of rows, because deleting data chips away at available training information.

---

## 📌 1. The Big Picture: The Minority Bias Bottleneck

In balanced data setups, algorithms assume an equal cost of misclassification. However, inside high-stakes domains (card fraud, medical tumour detection), your target class is highly skewed.

If a dataset contains 284,315 normal transactions and only 492 fraud cases, a completely broken model that flags _every single transaction_ as "Normal" will achieve a **99.82% global accuracy score**. Global accuracy hides the model's absolute failure to isolate fraud.

Handling imbalanced data functions as an architectural realignment that forces the algorithm to treat both error dimensions with equal mathematical priority.

---

## 📌 2. Real-World Use-Cases

**Simple Application — E-Commerce Customer Conversion:** Balancing data tracking users who visit a site vs. the small percentage who actually purchase, ensuring the model isolates the exact conversion triggers.

**Production Architecture — Autonomous Credit Card Fraud Prevention Engine:** Processing streaming card swipe features (`Amount`, temporal variables, and PCA components) to flag fraudulent transactions.

> **Stress Point:** If an engineer blindly deploys an under-sampling routine on a small or medium-sized dataset, the model will throw away thousands of legitimate customer profiles. It will create clean boundaries for fraud but completely forget how standard consumer spending shifts across macro conditions — triggering massive spikes in false alarms for legitimate users.

---

## 📌 3. Key Questions

1. Why does under-sampling a dataset introduce a severe risk of **Information Loss** inside your production modelling pipeline?
2. If your model achieves 99% accuracy on an imbalanced dataset but returns an F1-score of exactly 0.0 for the minority class, what has happened inside the confusion matrix?
3. In what explicit data volume conditions is Under-Sampling a safe, statistically valid architectural choice compared to Over-Sampling?

---

## 📌 4. Mathematical Framework & Resampling Logic

---

### The Class Disruption Ratio

Given a binary dependent label vector Y containing majority class records C₀ and minority records C₁:

$$\text{Ratio} = \frac{|C_1|}{|C_0| + |C_1|}$$

**Imbalance Threshold:** When Ratio < 0.05 (less than 5%), standard cross-validation steps break down, requiring direct resampling interventions.

---

### Under-Sampling Matrix Reduction

Under-sampling extracts the minority class count and forces the majority class array to match its length exactly via random stochastic selection:

$$|C_{0,\text{resampled}}| = |C_1| \qquad \Longrightarrow \qquad \text{Total Data} = 2 \times |C_1|$$

**The Core Mechanism:** Algorithms like **`NearMiss`** use distance metrics (Euclidean distance) to choose which majority class records to keep — aiming to preserve records that sit close to or far from the minority class boundary.

---

## 📌 5. Numerical Walkthrough — The Credit Card Case Study

### Baseline Ingestion Dimensions

- Total Dataset Rows (N): **284,807 records**
- Majority Class (`Class: 0` — Normal): **284,315 rows**
- Minority Class (`Class: 1` — Fraud): **492 rows**

---

### Execute NearMiss Under-Sampling

1. The engine isolates all 492 fraudulent records and keeps them completely intact.
2. It targets the normal transaction database (284,315 records).
3. The sampler pulls exactly **492 records** out of the normal pool, discarding the remaining 283,823 records.

---

### Post-Resampling Dimensions

|Metric|Before|After|
|:--|:-:|:-:|
|**Total Rows**|284,807|**984**|
|**Class 0 (Normal)**|284,315|**492**|
|**Class 1 (Fraud)**|492|**492**|
|**Class Balance**|99.8% / 0.2%|**50% / 50%**|

---

## 📌 6. Python Production Under-Sampling Pipeline

### 💻 Python Code

```python
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import NearMiss

def execute_under_sampling_pipeline(X, y):
    """
    Tracks class distribution profiles, applies NearMiss down-sampling,
    and validates the perfectly balanced output dimensions.
    """
    initial_distribution = Counter(y)
    print(f"Initial Dataset Class Balance Matrix: {initial_distribution}")

    # 1. Instantiate the NearMiss Under-Sampling engine
    # version=1 balances by looking at the average distance to the closest minority points
    under_sampler = NearMiss(version=1)

    # 2. Fit and resample the feature spaces cleanly
    X_balanced, y_balanced = under_sampler.fit_resample(X, y)

    balanced_distribution = Counter(y_balanced)

    return X_balanced, y_balanced, balanced_distribution

# =====================================================================
# SYSTEM SAMPLING VERIFICATION ENGINE
# =====================================================================
if __name__ == "__main__":
    # Generate a heavily imbalanced synthetic classification matrix (99% vs 1%)
    X_raw, y_raw = make_classification(
        n_samples=5000,
        n_features=5,
        weights=[0.99, 0.01],
        random_state=42
    )

    X_res, y_res, final_balance = execute_under_sampling_pipeline(X_raw, y_raw)

    print("=====================================================================")
    print("POST-SAMPLING PIPELINE DIMENSIONS SUMMARY")
    print("=====================================================================")
    print(f"Resampled Dataset Shape    : {X_res.shape}")
    print(f"Final Class Balance Matrix : {final_balance}")
    print("---------------------------------------------------------------------")

    assert final_balance[0] == final_balance[1], "Pipeline Error: Imbalance remains uncorrected."
    print("Pipeline Balancing Assertions: Passed Successfully.")
```

### 📉 Printed Output Verification

```text
Initial Dataset Class Balance Matrix: Counter({0: 4940, 1: 60})
=====================================================================
POST-SAMPLING PIPELINE DIMENSIONS SUMMARY
=====================================================================
Resampled Dataset Shape    : (120, 5)
Final Class Balance Matrix : Counter({0: 60, 1: 60})
---------------------------------------------------------------------
Pipeline Balancing Assertions: Passed Successfully.
```

---

## 📌 7. Gotchas & Misconceptions

- **The Accuracy Metric Trap:** When working on an imbalanced dataset, **never report global accuracy to your team or stakeholders.** If your data is split 99-to-1, a totally broken model can reach 99% accuracy easily. You must use **Precision, Recall, F1-Score, and the Area Under the Precision-Recall Curve (AUPRC)** to track true performance.
- **The Validation Splitting Blunder:** Never run an under-sampling or over-sampling algorithm on your _entire dataset_ before performing your train-test split. If you resample first, your validation set will be artificially balanced, causing your cross-validation metrics to lie. **Action:** Always isolate your test set _first_, then apply resampling exclusively to the training set.

---

## 📌 8. Imbalance Correction Taxonomy Matrix

|Strategy|Dimension Change|Primary Risk|Best Deployment Case|
|:--|:--|:--|:--|
|**Under-Sampling (Down-Sampling)**|Row contraction — discards majority class rows to match minority class size.|**Severe Information Loss:** Throws away valuable data, can cause underfitting.|Vast data environments — millions of rows where training on the full set is too slow.|
|**Over-Sampling (Up-Sampling)**|Row expansion — replicates or synthetically generates new minority class rows.|**Overfitting Risk:** Simple duplication can cause the model to memorise minority rows verbatim.|Scarce data environments — small datasets where you cannot afford to discard any rows.|

---

## 📌 9. Active Recall Questions

1. Why does performing a random under-sampling sweep across a small dataset heavily degrade the model's ability to generalise to unseen test data?
2. If you are deploying an XGBoost model on an imbalanced dataset, how can you use the internal parameter `scale_pos_weight` to balance the classes without dropping any rows?
3. How does the `NearMiss` distance logic select which majority class rows to keep compared to a simple random down-sampler?

---

## 🔗 Related Notes

- [[ML — ROC and AUC Curve Intuition]]
- [[ML — Classification Performance Metrics Part 1]]
- [[ML — Performance Metrics Multi-Class Classification]]
- [[ML — Dimensionality Reduction VIF vs PCA]]
- [[ML — KMeans Clustering & Elbow Method]]
- [[Feature Engineering — Missing Data Mechanisms & Continuous Imputation (Day 1)]]
- [[Data Science — Unified EDA, Feature Engineering & Feature Selection Roadmap]]

---

## 📹 Recommended Video Resource

[Krish Naik — Tutorial 45: Handling Imbalanced Dataset using Python Part 1](https://www.youtube.com/watch?v=YMPMZmlH5Bo)

---

_Tags: #machine-learning #imbalanced-data #under-sampling #over-sampling #nearmiss #smote #fraud-detection #precision-recall #f1-score #class-imbalance #imblearn #sklearn #python #data-science_