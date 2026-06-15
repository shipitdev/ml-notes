
---

tags: [machine-learning, unsupervised-learning, clustering, hierarchical-clustering, dendrogram, agglomerative] source: https://www.youtube.com/watch?v=0jPGHniVVNc related:

- "[[K-Means Clustering]]"
- "[[WCSS and Elbow Method]]"
- "[[Euclidean and Manhattan Distance]]"

---

# 🌳 Hierarchical Clustering & Dendrogram Mechanics

## ⚡ TL;DR

**Hierarchical Clustering** is an unsupervised algorithm that builds clusters **bottom-up** — no need to guess $K$ upfront. Every point starts as its own cluster, then the two closest are merged repeatedly until one master cluster remains. The full merge history is visualised as a **Dendrogram**. The optimal cluster count is found by cutting through the **longest uncrossed vertical line** in the tree.

---

## 📌 The Big Picture: Agglomerative Tree Assembly

- Operates **bottom-up (Agglomerative)** — no random centroid placement
- Every merge is tracked at its exact distance and plotted on the **Dendrogram's vertical axis**
- You choose cluster granularity **after** training, not before

---

## 🔄 The 5-Step Algorithm

### Step 1 — Initialise Each Point as Its Own Cluster

$$\mathcal{C} = \big{{X_1}, {X_2}, \dots, {X_N}\big}$$

### Step 2 — Build Proximity Matrix

Compute Euclidean Distance between every cluster pair ($N \times N$ matrix): $$d(C_a, C_b) = |X_a - X_b|_2 = \sqrt{\sum_{j=1}^{d}(X_{a,j} - X_{b,j})^2}$$

### Step 3 — Merge Closest Pair

Find the two nearest clusters and combine them: $$C_{\text{new}} = C_a \cup C_b \quad \Big| \quad \mathcal{C} = \big(\mathcal{C} \setminus {C_a, C_b}\big) \cup {C_{\text{new}}}$$

### Step 4 — Update Proximity Matrix (Linkage Rule)

|Linkage Method|Distance Measured Between Clusters|
|---|---|
|**Single**|Two _closest_ points across clusters|
|**Complete**|Two _furthest_ points across clusters|
|**Ward's** ✅|Minimises total internal variance (Scikit-Learn default)|

### Step 5 — Converge

Repeat Steps 3–4 until **all points merge into one master cluster**. Each link adds a horizontal bar to the Dendrogram at the height of its merge distance.

---

## 📐 Reading the Dendrogram: The Threshold Cut

To find the optimal number of clusters without guessing:

1. Find the **longest vertical lines** — these represent large distance jumps between merges
2. Isolate lines **not crossed by any horizontal branch**
3. Draw a **horizontal threshold line** through those clean vertical branches
4. Count how many vertical lines the threshold slices → **that count = optimal K**

> 💡 Cutting at the top of the Dendrogram = **1 cluster**. Cutting at the very bottom = **N clusters**.

---

## 💻 Python Pipeline

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram

def execute_hierarchical_clustering_pipeline(data_matrix, n_target_clusters=2):
    # 1. Standardize features (mandatory for stable distance metrics)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data_matrix)

    # 2. Compute full linkage tree matrix via SciPy
    linkage_matrix = linkage(scaled_features, method='ward', metric='euclidean')

    # 3. Extract final cluster labels via Scikit-Learn
    clustering_engine = AgglomerativeClustering(
        n_clusters=n_target_clusters,
        metric='euclidean',
        linkage='ward'
    )
    cluster_labels = clustering_engine.fit_predict(scaled_features)

    return linkage_matrix, cluster_labels
```

### Expected Output

```
   Cluster_A  Cluster_B  Distance  Sample_Count
         4.0        5.0  0.197990           2.0  ← First closest pair merged
         0.0        1.0  0.212132           2.0  ← Second closest pair merged
         2.0        3.0  0.282843           2.0  ← Third closest pair merged
         6.0        7.0  4.062019           4.0  ← Distance scales up
         8.0        9.0  4.502390           6.0  ← Final master merge

Final Data Group Label Assignments: [1 1 0 0 0 0]
```

---

## 🌍 Real-World Use Cases

### Simple — E-Commerce Category Trees

Group retail items by user profile similarity into multi-level navigation trees: `Apparel → T-Shirts → Waist-Length Cropped Tees`

### Advanced — Threat Intelligence IP Profiling

Cluster server IP metrics (`RequestFrequency`, `PayloadSize`, `FailureRate`) to detect:

- Automated security probes
- Administrative bypass footprints

> ⚠️ **Stress Point:** $O(N^3)$ scaling makes this impractical for large streaming logs. Use K-Means or DBSCAN for real-time filtering; reserve Hierarchical Clustering for deep analysis on small anomalous batches.

---

## ⚠️ Gotchas

> [!warning] Always Scale Your Features Hierarchical Clustering uses Euclidean distance exclusively. Without `StandardScaler`, large-scale columns will completely dominate distance calculations and destroy cluster boundaries.

> [!warning] No `.predict()` for New Data Unlike K-Means, `AgglomerativeClustering` in Scikit-Learn **cannot classify new incoming points**. New data requires a full tree rebuild from scratch — making it a poor fit for live production routing APIs.

---

## 📊 K-Means vs Hierarchical Clustering

|Criterion|K-Means|Hierarchical (Agglomerative)|
|---|---|---|
|**Method**|Centroid-driven partitioning|Connectivity-driven tree building|
|**K Required Upfront?**|✅ Yes|❌ No — outputs a Dendrogram|
|**Computational Scaling**|$O(N)$ — linear, fast|$O(N^3)$ — cubic, memory-heavy|
|**Live Production Fit**|✅ High — `.predict()` on new data|❌ Low — full rebuild required|

---

## ❓ Active Recall Questions

1. Why does `AgglomerativeClustering` lack a `.predict()` method for unseen records in Scikit-Learn?
2. How does **Ward's Linkage** prevent the **Chaining** problem that Single Linkage suffers from?
3. If the Dendrogram's longest uncrossed vertical branch spans distance heights 1.5 → 4.0, what does a threshold cut at height 3.0 reveal about the data structure?
4. Why does Hierarchical Clustering scale at $O(N^3)$ vs K-Means at $O(N)$?

---

## 🔗 Related Notes

- [[K-Means Clustering]]
- [[WCSS and Elbow Method]]
- [[Euclidean and Manhattan Distance]]
- [[DBSCAN Clustering]]
- [[Silhouette Coefficient Score]]