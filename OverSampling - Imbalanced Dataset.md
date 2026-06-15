# 📊 Machine Learning: Imbalanced Data — Over-Sampling & SMOTETomek Frameworks (Part 2)

---

## ⚡ TL;DR

When dealing with a heavily unbalanced dataset, you cannot use Under-Sampling if your data pool is small — deleting records throws away valuable training context.

- **Random Over-Sampling** directly clones or duplicates existing minority class rows verbatim until they match the size of the majority pool.
- **SMOTETomek** is an advanced hybrid framework that combines **SMOTE** (Synthetic Minority Over-sampling Technique) and **Tomek Links**. Instead of copying rows exactly, SMOTE draws vector lines between adjacent minority points and generates completely new, synthetic data points along those paths. Then, Tomek Links find and drop overlapping points right on the class borders — clearing away noise so your classifier can draw a perfectly stable boundary.

---

## 📌 1. The Big Picture: Over-Sampling without Memorisation

While under-sampling contracts a dataset's row count, over-sampling expands your spatial data room to achieve a balanced 1:1 target distribution across your parameters.

The underlying mathematical challenge is ensuring your model doesn't overfit. If you copy minority labels verbatim, a non-linear algorithm like a Decision Tree or Random Forest will grow hyper-specific branches that perfectly match those exact coordinates. The model looks flawless on paper but completely breaks when deployed to live production streams. Hybrid methods like `SMOTETomek` avoid this pitfall by interpolating unique vector coordinates and cleaning up border confusion.

---

## 📌 2. Real-World Use-Cases

**Simple Application — App User Subscription Conversion:** Scaling up target labels tracking the small percentage of freemium users who upgrade to paid premium tiers, ensuring marketing classifiers don't just default to predicting "no upgrade" for everyone.

**Production Architecture — High-Frequency Credit Card Fraud Prevention API:** Ingesting high-leverage transaction metrics (`Amount`, temporal records, and PCA vectors) to flag fraud attempts across real-time credit pipelines.

> **Stress Point:** Running an over-sampler expands your dataset to over 567,000 active rows. If you apply your over-sampling transformation to your _entire dataset_ before performing a train-test split, synthetic variations of your test data will leak straight into your training loops — creating a massive **Data Leakage** flaw that causes your validation metrics to look perfect while masking real-world system failures.

---

## 📌 3. Key Questions

1. Why does simple random over-sampling introduce a severe risk of overfitting for tree-based ensemble models, and how does SMOTE's vector interpolation fix this flaw?
2. If your dataset contains a majority class size of 1,000 and a minority class size of 100, what will the total row count become if you initialise a `RandomOverSampler` with `sampling_strategy=0.5`?
3. Why does running a hybrid engine like `SMOTETomek` create cleaner decision boundaries for a classifier compared to running standard SMOTE alone?

---

## 📌 4. Mathematical Framework & Resampling Logic

---

### SMOTE Synthetic Vector Interpolation

To create a brand-new, realistic minority instance, SMOTE isolates a minority coordinate vector X_i and locates its k-nearest neighbours inside that same minority class space using Euclidean distance. It randomly selects one neighbour (X_nn) and calculates a new point along the vector line connecting them using a random scaling multiplier (λ):

$$X_{\text{new}} = X_i + \lambda \cdot (X_{nn} - X_i) \qquad \lambda \in [0, 1]$$

**The Operational Logic:** This ensures that the newly generated data point is a completely unique vector that falls naturally within the geometric clusters of the minority class.

---

### Tomek Links Boundary Trimming

A Tomek Link is a pair of data points (X_i, X_j) that are each other's absolute nearest neighbours but **belong to opposite classes**. The formal condition:

$$\nexists ; X_k ; \text{such that} ; d(X_i, X_k) < d(X_i, X_j) ; \text{or} ; d(X_j, X_k) < d(X_i, X_j)$$

**The Trimming Action:** When a Tomek Link is detected, `SMOTETomek` removes the majority class instance (or both points) from the link — stripping away ambiguity along the decision boundary and allowing the classifier to draw a clean separation line between categories.

---

## 📌 5. Numerical Walkthrough — The Resampling Scaling Track

### Baseline Ingested Data Dimensions

- Normal Transactions Class (0): **284,315 rows**
- Fraudulent Transactions Class (1): **492 rows**
- Total Combined Rows (N): **~284,807 rows**

---

### Execute SMOTETomek Over-Sampling Pipeline

|Step|Action|Rows After|
|:--|:--|:-:|
|**1. SMOTE Generation**|Generates synthetic fraud records via neighbour interpolation until minority count matches majority (284,315).|~568,630|
|**2. Tomek Links Cleanup**|Scans boundaries and drops overlapping, noisy row links to clean up class borders.|~567,562|
|**3. Final Balance**|Class 0 and Class 1 perfectly balanced.|283,781 each|

---

## 📌 6. Python Production Over-Sampling Sandbox

### 💻 Python Code

```python
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import RandomOverSampler

def execute_oversampling_optimization_pipeline(X, y):
    """
    Executes hybrid SMOTETomek and adaptive RandomOverSampler adjustments
    to correct imbalanced datasets without information loss.
    """
    initial_distribution = Counter(y)
    print(f"Original Dataset Shape Profile: {initial_distribution}")

    # 1. Hybrid SMOTETomek Resampling (SMOTE + Tomek Links)
    # Automatically scales minority class to match majority, then prunes border links
    smote_tomek = SMOTETomek(random_state=42)
    X_smt, y_smt = smote_tomek.fit_resample(X, y)
    print(f"SMOTETomek Resampled Balance : {Counter(y_smt)}")

    # 2. Controlled RandomOverSampler (Target Ratio = 0.5)
    # sampling_strategy=0.5 means minority class scales to 50% of the majority class size
    ros_controlled = RandomOverSampler(sampling_strategy=0.5, random_state=42)
    X_ros, y_ros = ros_controlled.fit_resample(X, y)
    print(f"Controlled ROS Balance (0.5) : {Counter(y_ros)}")

    return X_smt, y_smt, X_ros, y_ros

# =====================================================================
# OVER-SAMPLING PREPROCESSING EVALUATION
# =====================================================================
if __name__ == "__main__":
    np.random.seed(42)

    # Construct a heavily skewed classification matrix (98% vs 2%)
    X_raw, y_raw = make_classification(
        n_samples=2000,
        n_features=4,
        weights=[0.98, 0.02],
        random_state=42
    )

    X_smt, y_smt, X_ros, y_ros = execute_oversampling_optimization_pipeline(X_raw, y_raw)

    print("=====================================================================")
    print("RESAMPLING ENGINE EXECUTION SUCCESSFUL")
    print("=====================================================================")

    # Verify Controlled ROS mathematics
    assert Counter(y_ros)[1] == int(Counter(y_ros)[0] * 0.5), "Controlled sampling strategy failed."
    print("Pipeline Balancing Assertions: Passed Successfully.")
```

### 📉 Printed Output Verification

```text
Original Dataset Shape Profile: Counter({0: 1956, 1: 44})
SMOTETomek Resampled Balance : Counter({0: 1954, 1: 1954})  <- Balanced and trimmed border links
Controlled ROS Balance (0.5) : Counter({0: 1956, 1: 978})   <- Minority scaled to 50% of 1956
=====================================================================
RESAMPLING ENGINE EXECUTION SUCCESSFUL
=====================================================================
Pipeline Balancing Assertions: Passed Successfully.
```

---

## 📌 7. Gotchas & Misconceptions

- **The `sampling_strategy` Ratio Confusion:** A common pitfall when using a float value (like `0.5`) in `sampling_strategy` is thinking it defines the final dataset composition percentage. It doesn't. The float specifies the **desired ratio of the minority class size relative to the majority class size**. If your majority class contains 1,000 rows, setting `sampling_strategy=0.5` forces the minority class to scale up until it contains exactly **500 rows**.
- **The Resource Allocation Explosion:** Over-sampling synthesises thousands of new records, exponentially inflating the size of your dataset. Training a complex model like XGBoost on a massively over-sampled dataset will dramatically increase training times and memory consumption — unlike under-sampling which speeds up processing by shrinking data volume.

---

## 📌 8. Resampling Strategy Reference Matrix

|Framework|Data Modification|Overfitting Risk|Best Deployment Case|
|:--|:--|:--|:--|
|**RandomOverSampler**|Row multiplication — directly copies existing minority rows to match target counts.|**High** — models quickly memorise the exact duplicate coordinates verbatim.|Low-resource environments: fast and efficient for small datasets where data loss cannot be tolerated.|
|**SMOTETomek (Hybrid)**|Row synthesis + noise trimming — synthesises new coordinates and deletes overlapping border rows.|**Low** — creates realistic, unique data points and clarifies class boundaries.|Production ingestion engines: ideal for high-dimensional, complex arrays like credit fraud tracking.|

---

## 📌 9. Active Recall Questions

1. How does the `k_neighbors` hyperparameter inside the SMOTE configuration control the spatial distribution of newly synthesised data points?
2. Why does running a hybrid method like `SMOTETomek` help prevent a classifier from overfitting compared to using standard random over-sampling?
3. What is the operational danger of running a train-test cross-validation split _after_ applying an over-sampling transformation to your data?

---

## 🔗 Related Notes

- [[ML — Imbalanced Data Under-Sampling & Over-Sampling Foundations]]
- [[ML — ROC and AUC Curve Intuition]]
- [[ML — Classification Performance Metrics Part 1]]
- [[ML — Hyperparameter Optimisation XGBoost RandomizedSearchCV]]
- [[ML — Ensemble Methods Bagging & Random Forest]]
- [[Data Science — Unified EDA, Feature Engineering & Feature Selection Roadmap]]

---

## 📹 Recommended Video Resource

[Krish Naik — Tutorial 46: Handling Imbalanced Dataset using Python Part 2](https://www.youtube.com/watch?v=OJedgzdipC0)

---

_Tags: #machine-learning #imbalanced-data #smote #smotetomek #tomek-links #over-sampling #random-over-sampler #synthetic-data #class-imbalance #data-leakage #fraud-detection #imblearn #sklearn #python #data-science_