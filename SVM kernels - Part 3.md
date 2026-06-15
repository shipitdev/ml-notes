
---

tags: [machine-learning, classification, svm, kernel-trick, polynomial-kernel, rbf, dimensionality] source: https://www.youtube.com/watch?v=8bFKyb77vp0 related:

- "[[Support Vector Machines]]"
- "[[SVM Math Intuition Part 2]]"
- "[[GridSearchCV and RandomizedSearchCV]]"
- "[[PCA]]"

---

# 📐 SVM Kernels — Intuition (Part 3)

## ⚡ TL;DR

When data isn't linearly separable (e.g., a circle of points inside a ring), no straight hyperplane works. **Kernels** solve this by projecting data into a higher-dimensional space where it _becomes_ linearly separable — then the standard SVM margin optimisation runs as normal. The **Kernel Trick** computes this projection implicitly, without ever building the high-dimensional matrix, saving massive memory and compute.

---

## 📌 The Big Picture: Kernel as a Dimension Lifter

```
[ 2D Input — Inseparable ]             [ Higher-Dim Space — Separable ]
  ●  ○  ○  ●                            ○  ○  ○   ← lifted
  ○  ●  ●  ○    ──────────►             ●  ●  ●   ← base layer
  ●  ○  ○  ●
(no line works)                        (flat hyperplane slices cleanly)
```

The hyperplane found in high-dim space projects **back** to the original 2D space as a **curved, non-linear boundary** (circle, ellipse, arbitrary curve).

---

## 📐 Mathematical Framework: Polynomial Expansion

For 2D input $[X_1, X_2]$, a degree-2 polynomial kernel internally computes: $$\mathbf{X}_{\text{new}} = [X_1,; X_2,; X_1^2,; X_2^2,; X_1 X_2]$$

This lifts the data from 2D → 5D, creating cross-feature interactions the original space couldn't capture.

### Numerical Example

|Original Input|Transformation|High-Dim Feature Vector|
|---|---|---|
|$[1, 2]$|$[X_1, X_2, X_1^2, X_2^2, X_1 X_2]$|$[1, 2, 1, 4, 2]$|
|$[3, 0]$|$[X_1, X_2, X_1^2, X_2^2, X_1 X_2]$|$[3, 0, 9, 0, 0]$|

Points that were inseparable in 2D become cleanly split when viewed through these new dimensions.

---

## 🔑 The Kernel Trick (Why It Doesn't Crash Your Machine)

Explicitly building the high-dimensional matrix is expensive. Kernels compute the **dot product in high-dimensional space** without constructing that space:

$$K(X_i, X_j) = \phi(X_i)^T \phi(X_j)$$

The function $\phi(\cdot)$ maps to high dimensions, but the kernel evaluates the inner product directly from the original inputs. For the RBF kernel this dimension is _infinite_ — yet it runs in constant time per pair.

---

## 📊 Kernel Comparison Matrix

|Kernel|Formula|Best Used When|
|---|---|---|
|**Linear**|$X^T Y$|Data is already linearly separable; fast on high-dim text data|
|**Polynomial**|$(X^T Y + 1)^d$|Feature interactions matter (e.g., $X_1 \cdot X_2$ terms)|
|**RBF** ✅ default|$e^{-\gamma \|X - Y\|^2}$|Complex non-linear clusters; best general-purpose choice|
|**Sigmoid**|$\tanh(\kappa X^T Y + c)$|Neural-network-style activation boundaries|

> 💡 **Default workflow:** Start with RBF. Tune `C` and `gamma` via `GridSearchCV`. Switch to Linear if data is known to be linearly separable (e.g., high-dim text with TF-IDF features).

---

## 💻 Python Pipeline

```python
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Polynomial kernel — degree controls boundary complexity
model = make_pipeline(
    StandardScaler(),
    SVC(kernel='poly', degree=3, C=1.0)
)
model.fit(X_train, y_train)

# Tune kernel type, degree, C, and gamma together
param_grid = {
    'svc__kernel': ['linear', 'poly', 'rbf'],
    'svc__C': [0.1, 1, 10],
    'svc__degree': [2, 3, 4],       # only used by poly
    'svc__gamma': ['scale', 'auto'] # only used by rbf/poly
}
gs = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
gs.fit(X_train, y_train)
print(gs.best_params_)
```

---

## 🌍 Real-World Use Cases

### Simple — Non-Linear Consumer Segmentation

Classify demographics that cluster in rings or arcs (e.g., buyers who spend at extremes but not in the mid-range) — no straight line separates them.

### Advanced — Network Intrusion Detection

Map raw protocol telemetry into multi-dimensional feature space. The Kernel Trick keeps RAM usage flat even as the implicit feature space grows enormously.

---

## ⚠️ Gotchas

> [!warning] High Degree = Overfitting Polynomial kernels with high `degree` memorise training noise instead of learning structure. The decision boundary becomes absurdly complex. Always cross-validate `degree` — don't increase it to chase training accuracy.

> [!warning] Always Scale Before Kernels RBF and Polynomial kernels compute distances and dot products. Unscaled features with large ranges dominate both, destroying the kernel's geometry. `StandardScaler` inside the pipeline is non-negotiable.

> [!warning] RBF's Gamma Parameter `gamma` controls how far the influence of a single training point reaches. High `gamma` → tight, wiggly boundary → overfitting. Low `gamma` → smooth boundary → underfitting. Tune alongside `C`.

---

## ❓ Active Recall Questions

1. Why doesn't the RBF kernel (which projects to infinite dimensions) exhaust system memory?
2. How does increasing the `degree` hyperparameter in a Polynomial Kernel affect the shape of the decision boundary?
3. In what scenario would you choose a Linear kernel over RBF for an SVM?
4. What does the `gamma` parameter control in the RBF kernel, and what happens at extreme values?

---

## 🔗 Related Notes

- [[Support Vector Machines]]
- [[SVM Math Intuition Part 2]]
- [[GridSearchCV and RandomizedSearchCV]]
- [[PCA]]
- [[Cross Validation]]