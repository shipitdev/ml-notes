
---
## ⚡ TL;DR

- **Euclidean Distance** calculates the **shortest, straight-line distance** between two points — like a bird flying directly from point A to point B, cutting diagonally across the map.
- **Manhattan Distance** calculates the distance by moving strictly along **grid-like, perpendicular paths** — like a taxi driving through the streets of Manhattan, forced to follow the grid lines rather than cutting through buildings.

---

## 📌 1. The Big Picture

Distance metrics quantify how similar or dissimilar two data points are. If the calculated distance between P₁ and P₂ is very small, the model assumes those two records share highly similar characteristics.

Choosing the wrong metric can significantly degrade model performance:

- **Euclidean distance squares errors** — highly sensitive to high-leverage outliers.
- **Manhattan distance uses absolute differences** — far more robust when dealing with high-dimensional datasets or sparse binary matrices.

---

## 📌 2. Real-World Use-Cases

**Simple Application — Flight Path vs. Warehouse Logistics:**

- **Euclidean:** Calculating the direct, unobstructed air-traffic path between two airports across a map.
- **Manhattan:** Calculating the path a robotic forklift must take to navigate around rows of shelves inside a grid-planned storage facility.

**Production Architecture — High-Dimensional E-Commerce User Segmentation API:** Ingesting user feature profiles (spending frequency, average cart values, active categories) to group users into dynamic behaviour clusters via K-Means.

> **Stress Point:** As the number of features grows (M > 100), the straight-line Euclidean distance between any two users begins to converge and look near-identical — rendering the clustering model ineffective. Switching to Manhattan Distance separates user clusters cleanly based on total absolute coordinate jumps.

---

## 📌 3. Mathematical Framework & Geometric Derivations

Consider two points P₁ and P₂ inside an N-dimensional continuous feature space:

$$P_1 = (x_1, y_1, z_1, \dots) \qquad P_2 = (x_2, y_2, z_2, \dots)$$

---

### Euclidean Distance — L₂ Norm

Derived directly from the **Pythagorean Theorem** (AC² = AB² + BC²). Computes the square root of the sum of squared coordinate differences across all dimensions:

$$\text{Distance}_{\text{Euclidean}} = \sqrt{\sum_{i=1}^{N}(P_{2,i} - P_{1,i})^2} = \sqrt{(x_2-x_1)^2 + (y_2-y_1)^2 + \dots}$$

**Geometric Behaviour:** Measures the length of the straight line connecting the two points diagonally across space.

---

### Manhattan Distance — L₁ Norm

Restricts movement to strict, axis-parallel lines. Sums the **absolute differences** of the coordinates across all dimensions — completely bypassing square roots and exponents:

$$\text{Distance}_{\text{Manhattan}} = \sum_{i=1}^{N}|P_{2,i} - P_{1,i}| = |x_2-x_1| + |y_2-y_1| + \dots$$

**Geometric Behaviour:** Measures the total distance travelled along perpendicular paths, matching grid-like city block layouts.

---

## 📌 4. Numerical Walkthrough

Two house feature points in 2D space:

- **House 1 (P₁):** 2 Rooms, $100k Value → coordinates (2, 1)
- **House 2 (P₂):** 6 Rooms, $400k Value → coordinates (6, 4)

---

### Step 1: Compute Euclidean Distance (L₂)

1. Isolate differences: Δx = (6 − 2) = 4 and Δy = (4 − 1) = 3
2. Square each: Δx² = 16 and Δy² = 9
3. Sum: 16 + 9 = 25
4. Take the square root:

$$\text{Distance}_{\text{Euclidean}} = \sqrt{25} = \mathbf{5.00}$$

---

### Step 2: Compute Manhattan Distance (L₁)

1. Isolate absolute differences: |Δx| = |6 − 2| = 4 and |Δy| = |4 − 1| = 3
2. Sum the absolute steps:

$$\text{Distance}_{\text{Manhattan}} = 4 + 3 = \mathbf{7.00}$$

**Result Analysis:** The Manhattan grid path (7.00) is structurally longer than the straight-line Euclidean shortcut (5.00), as it is constrained by axis-parallel movements.

---

## 📌 5. Python Simulation Sandbox

### 💻 Python Code

```python
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

def execute_distance_metric_pipeline(p1, p2):
    """
    Computes both Euclidean (L2) and Manhattan (L1) distance metrics
    from scratch and validates against Scikit-Learn's outputs.
    """
    array_1 = np.array(p1).reshape(1, -1)
    array_2 = np.array(p2).reshape(1, -1)

    # 1. Manual Math Implementations
    euclidean_manual = np.sqrt(np.sum((array_2 - array_1) ** 2))
    manhattan_manual = np.sum(np.abs(array_2 - array_1))

    # 2. Production Scikit-Learn Implementations
    euclidean_sklearn = pairwise_distances(array_1, array_2, metric='euclidean')[0][0]
    manhattan_sklearn = pairwise_distances(array_1, array_2, metric='manhattan')[0][0]

    results_df = pd.DataFrame({
        'Metric_Norm':       ['Euclidean (L2)', 'Manhattan (L1)'],
        'Manual_Calculation': [euclidean_manual, manhattan_manual],
        'Sklearn_Output':     [euclidean_sklearn, manhattan_sklearn]
    })

    return results_df

# =====================================================================
# SYSTEM DISTANCE VERIFICATION ENGINE
# =====================================================================
if __name__ == "__main__":
    house_1_coords = [2, 1]
    house_2_coords = [6, 4]

    metrics_summary = execute_distance_metric_pipeline(house_1_coords, house_2_coords)

    print("=====================================================================")
    print("GEOMETRIC DISTANCE EVALUATION SUMMARY")
    print("=====================================================================")
    print(metrics_summary.to_string(index=False))
    print("=====================================================================")

    assert metrics_summary['Manual_Calculation'].iloc[0] == 5.0, "Euclidean failure."
    assert metrics_summary['Manual_Calculation'].iloc[1] == 7.0, "Manhattan failure."
    print("Mathematical Consistency Asserts: Passed Successfully.")
```

### 📉 Printed Output Verification

```text
=====================================================================
GEOMETRIC DISTANCE EVALUATION SUMMARY
=====================================================================
   Metric_Norm  Manual_Calculation  Sklearn_Output
Euclidean (L2)                 5.0             5.0
Manhattan (L1)                 7.0             7.0
=====================================================================
Mathematical Consistency Asserts: Passed Successfully.
```

---

## 📌 6. Gotchas & Misconceptions

- **The Scale Domination Trap:** Distance metrics are deeply vulnerable to feature scale differences. If `RoomCount` ranges from 1 to 5 and `LotArea` ranges from 1,000 to 5,000, any distance formula will be completely dominated by `LotArea`, rendering the room feature useless. **Action:** Always scale your data using `StandardScaler` or `MinMaxScaler` before passing features to distance-based algorithms like KNN or K-Means.
- **The Outlier Sensitivity Fallacy:** Because Euclidean distance squares its differences, an extreme outlier creates a massive jump in the total score. Manhattan distance scales linearly, making it far more robust and stable when working with noisy real-world datasets.

---

## 📌 7. Distance Metric Comparison Taxonomy

|Evaluation Criterion|Euclidean Distance (L₂ Norm)|Manhattan Distance (L₁ Norm)|
|:--|:--|:--|
|**Geometric Path**|Straight-line diagonal — absolute shortest path between two points.|Perpendicular grid — movement restricted strictly to axis-parallel lines.|
|**Outlier Sensitivity**|**High** — squaring the differences amplifies errors from extreme points.|**Low & Robust** — absolute differences scale linearly, reducing outlier impact.|
|**Dimensionality Behaviour**|Degrades significantly as features increase (M > 50) — distances begin looking uniform.|Maintains better contrast and separation in high-dimensional feature spaces.|
|**Target Algorithms**|Standard KNN, K-Means Clustering, Support Vector Machines.|Sparse matrix problems, text analytics, high-dimensional deep learning, Lasso.|

---

## 📌 8. Active Recall Questions

1. How does the calculation of a model's gradient step change when swapping out an L₂ norm (Euclidean) for an L₁ norm (Manhattan) inside a loss function?
2. Why does the geometric distance between any two data points in a hyper-dimensional space (M → ∞) begin to look completely identical when using Euclidean distance?
3. In cross-validation configurations, how can you test whether a dataset performs better under an L₁ or L₂ distance metric using hyperparameter tuning loops?

---

## 🔗 Related Notes

- [[ML — Ensemble Methods Bagging & Random Forest]]
- [[ML — Hyperparameter Optimisation XGBoost RandomizedSearchCV]]
- [[ML — AdaBoost Adaptive Boosting Framework]]
- [[Feature Engineering — Feature Scaling & Gaussian Transformations (Day 6)]]
- [[Feature Engineering — Feature Scaling Why We Need It]]
- [[Data Science — Unified EDA, Feature Engineering & Feature Selection Roadmap]]

---

## 📹 Recommended Video Resource

[Krish Naik — Euclidean Distance and Manhattan Distance](https://www.youtube.com/watch?v=p3HbBlcXDTE)

---

_Tags: #machine-learning #distance-metrics #euclidean-distance #manhattan-distance #l1-norm #l2-norm #knn #k-means #curse-of-dimensionality #feature-scaling #pythagorean-theorem #geometry #sklearn #python #data-science_