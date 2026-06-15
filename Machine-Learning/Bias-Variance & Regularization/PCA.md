
---

tags: [machine-learning, unsupervised-learning, dimensionality-reduction, pca, feature-transformation, curse-of-dimensionality] source: https://www.youtube.com/watch?v=OFyyWcw2cyM related:

- "[[K-Means Clustering]]"
- "[[Euclidean and Manhattan Distance]]"
- "[[VIF]]"
- "[[Silhouette Coefficient Score]]"

---

# 🌳 Principal Component Analysis (PCA) — Dimensionality Reduction

## ⚡ TL;DR

**PCA** compresses high-dimensional feature spaces into fewer dimensions while preserving as much variance as possible. It does this by rotating your coordinate axes into brand-new directions called **Principal Components (PCs)** — each one orthogonal to the last, ordered by how much variance they capture. The result: fewer features, faster models, and visualisable data — at the cost of interpretability.

---

## 📌 The Big Picture: Compressing the Feature Space

As feature count grows, models degrade and compute costs explode — this is the **Curse of Dimensionality**.

```
[ High-Dimensional 30D Grid ]                  [ Low-Dimensional PCA Space ]
┌──────────────────────────────┐              ┌──────────────────────────────┐
│ Radius, Texture, Area...     │ ──────────►  │ PC1 (captures ~70% variance) │
│ [30 correlated features]     │  Linear      │ PC2 (captures ~20% variance) │
└──────────────────────────────┘  Transform   └──────────────────────────────┘
```

PCA rotates the coordinate space to align with the main spread of the data cloud, then projects it down to fewer dimensions — e.g., compressing 30 features → 2 PCs while retaining ~90% of variance.

---

## 📐 Mathematical Framework

### Step 1 — Mean Centering & Standardisation

PCA is variance-driven. Features with large numeric ranges dominate if unscaled. Z-score first: $$Z = \frac{X_i - \mu}{\sigma}$$

### Step 2 — Covariance Matrix ($\Sigma$)

Captures how every feature varies relative to all others ($M \times M$ symmetric matrix): $$\Sigma = \frac{1}{N-1} Z^T Z$$

### Step 3 — Eigendecomposition

Decomposes the covariance matrix into directions and magnitudes: $$\Sigma, \mathbf{v} = \lambda, \mathbf{v}$$

|Term|Role|
|---|---|
|**Eigenvectors ($\mathbf{v}$)**|Direction of each new Principal Component axis|
|**Eigenvalues ($\lambda$)**|Amount of variance captured along each axis|

### Step 4 — Project into Lower-Dimensional Space

Stack the top $k$ eigenvectors into projection matrix $\mathbf{W}_k$, then transform: $$\mathbf{X}_{\text{PCA}} = Z, \mathbf{W}_k$$

---

## 🧮 Numerical Walkthrough (Breast Cancer Dataset)

**Input:** 569 patient profiles × 30 cell attributes (`mean radius`, `perimeter`, `compactness`…)

1. Pass through `StandardScaler` → all 30 columns at mean $= 0$, std $= 1$
2. Fit PCA with `n_components=2` → extracts top 2 eigenvectors
3. **Matrix shape:** `(569, 30)` → **`(569, 2)`** — 93% of columns removed

**Variance preserved:**

- PC1: 44.27%
- PC2: 18.97%
- **Total: 63.24%**

---

## 💻 Python Pipeline

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def execute_pca_dimension_pipeline(raw_data, target_components=2):
    # 1. Standardize (mandatory — PCA is variance-driven)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(raw_data)

    # 2. Fit PCA and compress
    pca_engine = PCA(n_components=target_components, random_state=42)
    compressed_matrix = pca_engine.fit_transform(scaled_features)

    # 3. Extract variance profile
    explained_variance_ratio = pca_engine.explained_variance_ratio_
    total_variance_preserved = np.sum(explained_variance_ratio)

    return compressed_matrix, explained_variance_ratio, total_variance_preserved

if __name__ == "__main__":
    cancer_dataset = load_breast_cancer()
    X_raw = cancer_dataset.data

    X_transformed, variance_ratios, total_info = execute_pca_dimension_pipeline(X_raw, target_components=2)

    print(f"Original Shape:           {X_raw.shape}")
    print(f"Compressed Shape:         {X_transformed.shape}")
    print(f"Variance from PC1:        {variance_ratios[0]*100:.2f}%")
    print(f"Variance from PC2:        {variance_ratios[1]*100:.2f}%")
    print(f"Total Preserved Variance: {total_info*100:.2f}%")
```

### Expected Output

```
Original Shape:           (569, 30)
Compressed Shape:         (569, 2)
Variance from PC1:        44.27%
Variance from PC2:        18.97%
Total Preserved Variance: 63.24%
```

---

## 🌍 Real-World Use Cases

### Simple — Customer Segmentation Dashboards

Compress multi-axis shopping behaviours (cart frequency, return rates, navigation trails) into a 2D scatter plot for product teams to spot buyer cohorts instantly.

### Advanced — Quantitative Feature Compression Pipeline

Compress 1,000 correlated technical indicators + market microstructure signals into ~10 principal components to feed a live trading model.

> ⚠️ **Stress Point:** PCs are abstract linear combinations of original features — you lose the direct meaning of columns like `Mean Radius`. Debugging a false alarm becomes significantly harder when you can't trace it back to a specific physical input.

---

## ⚠️ Gotchas

> [!warning] Always Scale Before PCA PCA maximises variance. Without `StandardScaler`, large-scale columns (e.g., `Annual Income`) dominate the eigenvector directions and bury small-scale but important features (e.g., `Cell Smoothness`).

> [!warning] PCA is Unsupervised — No Label Awareness PCA only sees the feature matrix $\mathbf{X}$, never the target labels $y$. Directions of maximum variance don't necessarily align with class separation. PCA helps clean decision boundaries but provides no guarantee.

> [!warning] Loss of Interpretability Principal Components are mathematical constructs. Once transformed, you can no longer say "this feature is `mean radius`" — they become abstract combinations of all original inputs.

---

## 📊 PCA vs Feature Selection

|Criterion|Feature Selection (Lasso / VIF)|PCA|
|---|---|---|
|**Logic**|Drops redundant/low-impact original columns|Rotates entire space into new artificial axes|
|**Matrix Effect**|Narrows column count; values unchanged|Changes the data values themselves|
|**Interpretability**|✅ High — columns keep original meaning|❌ Low — PCs are abstract combinations|
|**Best For**|When interpretability matters|When compression speed > interpretability|

---

## ❓ Active Recall Questions

1. Why does PC2 always capture less variance than PC1?
2. If your dataset has 10 features, what is the maximum number of Principal Components that can be generated?
3. How do you use a **Scree Plot** to decide how many components to retain in production?
4. Why is `StandardScaler` a mandatory prerequisite before running PCA — not just a best practice?

---

## 🔗 Related Notes

- [[K-Means Clustering]]
- [[DBSCAN Clustering]]
- [[Silhouette Coefficient Score]]
- [[Euclidean and Manhattan Distance]]
- [[VIF]]