---
tags: [machine-learning, unsupervised-learning, clustering, k-means, elbow-method]
source: https://www.youtube.com/watch?v=AWKCCK5YHsE
related:
  - "[[PCA]]"
  - "[[Euclidean and Manhattan Distance]]"
  - "[[WCSS and Elbow Method]]"
---

# 🌳 K-Means Clustering & The Elbow Method

## ⚡ TL;DR

**K-Means** is an unsupervised algorithm that groups unlabelled data into **K clusters** based on spatial proximity. It iteratively assigns points to the nearest centroid and recalculates centroid positions until stable. The **Elbow Method** helps find the optimal K by tracking WCSS error drop across K values.

---

## 📌 The Big Picture

- No labels ($y$) needed — only a raw feature matrix ($\mathbf{X}$)
- Projects records into N-dimensional space and groups by **spatial proximity**
- Centroids shift dynamically until cluster boundaries lock in place

---

## 🔄 The 5-Step Algorithm

### Step 1 — Define $K$
Choose how many clusters (centroids) to create.

### Step 2 — Random Initialization
Place $K$ centroids randomly in feature space:
$$\mu_k \in \mathbb{R}^d$$

### Step 3 — Assign Points (Euclidean Distance)
Each point $X_i$ is assigned to its nearest centroid $c_i$:
$$c_i = \arg\min_{k} \|X_i - \mu_k\|^2 = \arg\min_{k} \sqrt{\sum_{j=1}^{d}(X_{i,j} - \mu_{k,j})^2}$$

### Step 4 — Recalculate Centroids (Mean Shift)
Each centroid moves to the geometric mean of its assigned points:
$$\mu_k = \frac{1}{|S_k|} \sum_{i \in S_k} X_i$$

### Step 5 — Convergence
Repeat Steps 3–4 until **zero points change cluster assignment**.

---

## 📐 The Elbow Method (Choosing K)

### WCSS Formula
$$\text{WCSS} = \sum_{k=1}^{K} \sum_{i \in S_k} \|X_i - \mu_k\|^2$$

### Intuition
| K | WCSS Behaviour |
|---|---------------|
| K = 1 | Massive error — one centroid covers everything |
| K increases | Error drops sharply |
| K = N | WCSS = 0 — every point is its own centroid |
| **Elbow Point** | Error stops dropping sharply → optimal K |

> 💡 **The Elbow Point** is the inflection where the curve flattens — balancing simplicity vs. accuracy.

---

## 💻 Python Pipeline

```python
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def execute_kmeans_optimization_engine(data_matrix, max_k_test=10):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_matrix)

    wcss_scores = []
    for k in range(1, max_k_test + 1):
        kmeans_sweep = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        kmeans_sweep.fit(scaled_data)
        wcss_scores.append(kmeans_sweep.inertia_)

    optimized_kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42, n_init=10)
    optimized_kmeans.fit(scaled_data)

    wcss_profile_df = pd.DataFrame({
        'K_Value': range(1, max_k_test + 1),
        'WCSS_Score': wcss_scores
    })
    return wcss_profile_df, optimized_kmeans.labels_
```

### Expected Output
```
K_Value  WCSS_Score
      1  298.000000   ← Base error
      2  112.435129   ← Sharp drop
      3   14.892311   ← Elbow point ✅
      4   12.983211   ← Flattening
      5   11.109341
      6    9.431024
```

---

## 🌍 Real-World Use Cases

### Simple — E-Commerce Segmentation
Cluster shoppers by age + spending velocity → targeted marketing tiers.

### Advanced — Crypto Trading Regime Detection
Cluster live order book parameters (OBI, volatility, volume depth) into regimes:
- `Standard Liquid Loop`
- `High-Imbalance Liquidation Cascade`
- `Low-Volume Drift`

> ⚠️ **Stress Point:** Fixed $K$ fails under real-time regime shifts. Background WCSS variance monitoring can auto-trigger $K$ recalculation.

---

## ⚠️ Gotchas

> [!warning] Always Scale Your Features
> K-Means uses Euclidean distance. A column with range 10–1000 will dominate a column with range 0.01–0.05. Always use `StandardScaler` before clustering.

> [!warning] Random Init Trap
> Random centroid placement can cause convergence on a local minimum. Always use `init='k-means++'` to spread initial centroids across the data space.

---

## 📊 K-Means vs Hierarchical Clustering

| Criterion | K-Means | Hierarchical |
|---|---|---|
| **Method** | Centroid-driven partitioning | Connectivity-driven tree building |
| **K Required?** | Yes — set upfront | No — outputs a Dendrogram |
| **Scaling** | Linear — handles large datasets | Quadratic — slow on large data |

---

## ❓ Active Recall Questions

1. Why does WCSS = 0 when K = N?
2. How do outliers distort centroid recalculation during the mean-shift step?
3. Why can't K-Means reliably separate concentric circular clusters?
4. What flaw does `K-Means++` initialization fix vs. pure random init?

---

## 🔗 Related Notes
- [[PCA]]
- [[Euclidean and Manhattan Distance]]
- [[WCSS and Elbow Method]]