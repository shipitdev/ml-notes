
---

tags: [machine-learning, unsupervised-learning, validation, silhouette-coefficient, clustering-metrics] source: https://www.youtube.com/watch?v=DpRPd274-0E related:

- "[[K-Means Clustering]]"
- "[[Hierarchical Clustering]]"
- "[[DBSCAN Clustering]]"
- "[[WCSS and Elbow Method]]"

---

# 🌳 Silhouette Coefficient Score — Unsupervised Cluster Validation

## ⚡ TL;DR

The **Silhouette Coefficient** validates unsupervised cluster quality when you have no ground-truth labels to check against. It measures two things per data point: **Cohesion** (how close it is to its own cluster) and **Separation** (how far it is from the nearest neighbouring cluster). The final score is normalised to $[-1, +1]$. Scores near $+1$ = tight, well-separated clusters. Negative scores = misclassified points.

---

## 📌 The Big Picture: Beyond the Elbow Method

The Elbow Method (WCSS) only measures internal cluster error — it will always drop as $K$ increases, hitting $0$ when $K = N$. It cannot tell you if clusters are actually distinct from each other.

The Silhouette Coefficient fills this gap by enforcing a **Separation Space** check — balancing how tightly bound a cluster is internally against how far it sits from neighbouring groups.

---

## 📐 Mathematical Framework

For every data point $X_i$:

### Step 1 — Intra-Cluster Distance: $a_i$ (Cohesion)

Average distance between $X_i$ and all other points in its **own** cluster $C_I$: $$a_i = \frac{1}{|C_I| - 1} \sum_{j \in C_I,, j \neq i} d(X_i, X_j)$$

> Divides by $|C_I| - 1$ to exclude the self-distance $d(X_i, X_i) = 0$.

### Step 2 — Inter-Cluster Distance: $b_i$ (Separation)

Minimum average distance from $X_i$ to all points in any **neighbouring** cluster $C_K$: $$b_i = \min_{K \neq I} \left( \frac{1}{|C_K|} \sum_{j \in C_K} d(X_i, X_j) \right)$$

> Targets the **closest** neighbouring cluster — the one most at risk of accidental merging.

### Step 3 — Silhouette Score: $s_i$

$$s_i = \frac{b_i - a_i}{\max(a_i,, b_i)}$$

---

## 📊 Score Range Interpretation

|Score|Meaning|
|---|---|
|$s_i \to +1.0$|Point sits in a dense, well-separated cluster ✅|
|$s_i \approx 0.0$|Point sits on the boundary between two clusters ⚠️|
|$s_i \to -1.0$|Point is closer to a neighbouring cluster than its own → **misclassified** ❌|

---

## 🧮 Parameter Sweep Results (Worked Example)

|K|Global Silhouette Score|Observation|
|---|---|---|
|2|0.70|High score but uneven cluster block thickness → size imbalance|
|3|0.58|Cluster 0 shows a **negative blade** → misclassified boundary points|
|**4**|**0.65**|Rebounded score + uniform block thickness + **zero negative points** ✅|
|5|0.57|Score drops again with new negative outliers|

> 💡 **Conclusion:** Cross-referencing Elbow Method + Silhouette plots confirms **K=4** as the true optimised cluster count.

---

## 💻 Python Pipeline

```python
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples

def run_silhouette_validation_engine(X_raw, cluster_range=[2, 3, 4, 5]):
    # 1. Standardize features (mandatory for Euclidean distance stability)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    validation_logs = []

    for k in cluster_range:
        kmeans_model = KMeans(n_clusters=k, init='k-means++', random_state=10, n_init=10)
        labels = kmeans_model.fit_predict(X_scaled)

        global_avg = silhouette_score(X_scaled, labels)       # single global score
        sample_values = silhouette_samples(X_scaled, labels)  # per-point scores
        negative_outliers_count = np.sum(sample_values < 0)

        validation_logs.append({
            'K_Clusters': k,
            'Global_Average_Silhouette': global_avg,
            'Negative_Outliers_Count': negative_outliers_count
        })

    return pd.DataFrame(validation_logs)
```

### Expected Output

```
 K_Clusters  Global_Average_Silhouette  Negative_Outliers_Count
          2                   0.704285                        0
          3                   0.588200                        5  ← misclassified boundary points!
          4                   0.650518                        0  ← optimal: zero boundary errors ✅
          5                   0.574522                        3
```

---

## 📈 Reading Silhouette Plots

### Block Thickness

Width of each cluster's silhouette shape = number of points in that cluster. Massive size gaps between blocks signal cluster imbalance.

### Negative Blades

Any shape extending **below the zero line** = points with $s_i < 0$ = misclassified rows. Even one cluster with a negative blade is a red flag, regardless of the global average score.

---

## 🌍 Real-World Use Cases

### Simple — Streaming Content Audience Profiling

Validate subscription tier clusters (genre affinity + watch-time) are cleanly separated before deploying content recommendation carousels.

### Advanced — Quant Regime Mapping Engine

Monitor rolling Silhouette scores on live market microstructure clusters (`OBI`, `Volatility`, `Volume Depth`).

> ⚠️ **Stress Point:** When liquidity shifts during macro events, cluster boundaries blur. Set an automated alert to trigger a model rebuild when the rolling Silhouette score drops below **+0.5**.

---

## ⚠️ Gotchas

> [!warning] Spherical Geometry Bias Silhouette uses Euclidean distance and assumes **spherical/circular** cluster shapes. Complex non-linear shapes (interlocking crescents, concentric rings) will receive poor scores even when clustering is correct. Use domain knowledge to interpret the score accordingly.

> [!warning] Always Scale Features Cohesion and separation are both distance-based. Without `StandardScaler`, large-scale columns dominate the $a_i$ and $b_i$ calculations, producing meaningless scores.

> [!warning] High Global Score ≠ Good Clustering A high global average can mask individual clusters with negative blades. Always inspect `silhouette_samples` per cluster, not just the global average.

---

## 📊 Elbow Method vs Silhouette Coefficient

|Metric|Mathematical Core|Optimises For|Best Used For|
|---|---|---|---|
|**Elbow Method (WCSS)**|Sum of squared distances to centroids|Cohesion only|Fast initial baseline for $K$ selection|
|**Silhouette Coefficient**|Normalised ratio of $a_i$ vs $b_i$|Cohesion + Separation|Validating boundary separation + catching misclassified points|

---

## ❓ Active Recall Questions

1. Why does a global Silhouette score near $0.0$ indicate points are sitting on a cluster boundary?
2. Why is a high global average score insufficient if one cluster displays a negative blade?
3. How does swapping Euclidean distance for Manhattan distance alter the $a_i$ and $b_i$ calculations?
4. Why will Silhouette penalise a model that perfectly clusters two interlocking crescent-moon shapes?

---

## 🔗 Related Notes

- [[K-Means Clustering]]
- [[Hierarchical Clustering]]
- [[DBSCAN Clustering]]
- [[WCSS and Elbow Method]]
- [[Euclidean and Manhattan Distance]]