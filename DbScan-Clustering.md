
---

tags: [machine-learning, unsupervised-learning, clustering, dbscan, density-based, outlier-detection] source: https://www.youtube.com/watch?v=C3r7tGRe2eI related:

- "[[K-Means Clustering]]"
- "[[Hierarchical Clustering]]"
- "[[Silhouette Coefficient Score]]"
- "[[Euclidean and Manhattan Distance]]"

---

# 🌳 DBSCAN — Density-Based Spatial Clustering

## ⚡ TL;DR

**DBSCAN** groups data by local point density rather than distance to a fixed center. It handles arbitrary non-linear shapes (rings, curves, winding paths) that K-Means cannot, and automatically flags scattered outliers as **noise** (label `-1`) without you needing to set $K$ upfront. Two parameters control everything: **$\epsilon$** (scan radius) and **MinPoints** (density threshold).

---

## 📌 The Big Picture: Density-Driven Aggregation

- Classical clustering uses flat linear boundaries → fails on curved or concentric data
- DBSCAN scans like an **expanding wave**, tracing density chains across the coordinate space
- Dense trails get grouped; stray background points are cleanly isolated as noise

---

## 🔷 The Three Point Classifications

Every data point is classified into one of three types:

### I. Core Point (Dense Cluster Anchor)

Has at least `MinPoints` neighbors within radius $\epsilon$: $$\big|N_\epsilon(X_i)\big| \ge \text{MinPoints}$$

### II. Border Point (Cluster Edge Layer)

Doesn't meet density threshold itself, but falls within the $\epsilon$-radius of a Core Point: $$\big|N_\epsilon(X_j)\big| < \text{MinPoints} \quad \text{and} \quad \exists, X_c \in N_\epsilon(X_j) \text{ s.t. } X_c \text{ is a Core Point}$$

### III. Noise Point (Isolated Outlier)

Fails both tests — not dense enough, not touching any Core Point: $$\big|N_\epsilon(X_k)\big| < \text{MinPoints} \quad \text{and} \quad X_k \notin N_\epsilon(X_c) \text{ for any Core Point } X_c$$

> 🏷️ Noise points receive label **`-1`** in Scikit-Learn and are excluded from all clusters.

---

## 📐 Mathematical Framework

### Epsilon-Neighborhood Scan

All points within scan radius $\epsilon$ of point $X_i$: $$N_\epsilon(X_i) = {X_j \in \mathcal{D} \mid \text{dist}(X_i, X_j) \le \epsilon}$$

### Two Hyperparameters

|Parameter|Role|
|---|---|
|**$\epsilon$ (eps)**|Scan radius — geometric boundary around each point|
|**MinPoints**|Minimum neighbours required to establish a dense core|

---

## 🧮 Numerical Walkthrough (Mall Customer Dataset)

**Setup:** Features = `Annual Income` + `Spending Score` | $\epsilon = 3$, MinPoints $= 4$

**Execution:**

1. Loop through each customer row, scan neighbourhood within radius 3
2. High-density coordinates chain together into cluster groups
3. Outliers failing the density check are filtered out immediately

**Output (`model.labels_`):**

- Label **`-1`** → noisy outlier rows
- Labels **`0` to `8`** → 9 valid density clusters

---

## 💻 Python Pipeline

```python
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def execute_dbscan_density_engine(X_raw, eps_radius=0.5, min_samples_target=4):
    # 1. Standardize features (mandatory for Euclidean distance stability)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # 2. Fit DBSCAN density scanner
    dbscan_model = DBSCAN(eps=eps_radius, min_samples=min_samples_target, metric='euclidean')
    dbscan_model.fit(X_scaled)
    extracted_labels = dbscan_model.labels_

    # 3. Parse cluster counts
    unique_labels = set(extracted_labels)
    core_clusters_count = len(unique_labels) - (1 if -1 in unique_labels else 0)
    outliers_count = np.sum(extracted_labels == -1)

    # 4. Silhouette validation (only valid if >1 cluster found)
    validation_score = -1.0
    if core_clusters_count > 1:
        validation_score = silhouette_score(X_scaled, extracted_labels)

    return extracted_labels, core_clusters_count, outliers_count, validation_score
```

### Expected Output

```
Total Unique Density Clusters Located: 2
Isolated Noise Points Detected (-1):   18 rows  ← stray outliers auto-isolated
Calculated Silhouette Quality Score:    0.7342  ← high score = tight, clean separation
```

---

## 🌍 Real-World Use Cases

### Simple — Urban Foot-Traffic Hotspots

Cluster raw mobile coordinate logs to find high-density shopper zones in a city. Stray travellers are automatically filtered as path noise.

### Advanced — Network Intrusion Detection

Monitor network vectors (`PayloadEntropy`, `RequestIntervals`, `FailedAuthAttempts`) to detect DDoS waves or port-scan footprints.

> ⚠️ **Stress Point:** Real-world network density shifts drastically between peak hours and off-hours. A fixed $\epsilon$ will either merge unrelated traffic groups or break valid clusters into noise. Consider adaptive models like **HDBSCAN** or **OPTICS** for shifting densities.

---

## ⚠️ Gotchas

> [!warning] Fixed-Radius Trap (Uniform Density Assumption) DBSCAN assumes clusters share roughly uniform density. If one cluster is tightly packed and another is loosely scattered, no single $\epsilon$ works for both. Use **OPTICS** or **HDBSCAN** for variable-density data.

> [!warning] No `.predict()` for New Data Like Hierarchical Clustering, Scikit-Learn's DBSCAN **cannot classify new incoming points** in isolation. Cluster assignment depends on continuous density chains — new records require a full rescan of the entire matrix.

> [!warning] Always Scale Features DBSCAN uses Euclidean distance. Without `StandardScaler`, large-scale columns dominate the $\epsilon$ scan and destroy boundary accuracy.

---

## 📊 K-Means vs Hierarchical vs DBSCAN

|Criterion|K-Means|Hierarchical|DBSCAN|
|---|---|---|---|
|**Grouping Method**|Shifting centroids via group averages|Bottom-up pair merging into a tree|Expanding density chains across space|
|**Boundary Shape**|Flat linear partitions|Nested folder tree|Arbitrary non-linear shapes ✅|
|**Noise Handling**|❌ Outliers distort centroids|❌ Outliers clutter the tree|✅ Auto-labelled `-1` and excluded|
|**Parameters**|$K$ (cluster count)|Linkage rule / threshold cut|$\epsilon$ + MinPoints|
|**`.predict()` Support**|✅ Yes|❌ No|❌ No|

---

## ❓ Active Recall Questions

1. Why does Scikit-Learn's DBSCAN assign label `-1` to certain data rows?
2. How does the **Curse of Dimensionality** ($M \to \infty$) make choosing a stable $\epsilon$ increasingly unreliable?
3. A point has enough neighbours to meet `min_samples` — what additional condition determines if it's a Core Point vs a Border Point?
4. Why does a single fixed $\epsilon$ fail on datasets with clusters of vastly different densities?

---

## 🔗 Related Notes

- [[K-Means Clustering]]
- [[Hierarchical Clustering]]
- [[Silhouette Coefficient Score]]
- [[Euclidean and Manhattan Distance]]
- [[OPTICS Clustering]]
- [[HDBSCAN]]