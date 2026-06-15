
---

tags: [machine-learning, classification, regression, svm, support-vector-machine, kernel-trick, margin-optimization] source: https://www.youtube.com/watch?v=H9yACitf-KM related:

- "[[Logistic Regression]]"
- "[[GridSearchCV and RandomizedSearchCV]]"
- "[[Cross Validation]]"

---

# 🌳 Support Vector Machines (SVM)

## ⚡ TL;DR

**SVM** doesn't just find _a_ boundary between classes — it finds the _best_ one: the **hyperplane with the maximum margin** between the two class edges. Only the closest points to the boundary (**Support Vectors**) define it. For non-linearly separable data, the **Kernel Trick** projects points into higher dimensions where a flat hyperplane _can_ split them.

---

## 📌 The Big Picture: Maximising the Margin

```
[ Poor Hyperplane ]                       [ Optimal Hyperplane (SVM) ]
┌──────────────────────────┐             ┌──────────────────────────┐
│  ●  ● │ ▲  ▲  ▲          │             │  ●  ●  ‖      ‖  ▲  ▲   │
│    ●  │ ▲  ▲             │  ────────►  │    ●   ‖ MAX  ‖  ▲      │
│  ●  ● │ ▲  ▲             │             │  ●  ●  ‖ MARGIN‖  ▲  ▲  │
└──────────────────────────┘             └──────────────────────────┘
  (small margin → fragile)                 (wide margin → generalises)
```

- **Hyperplane:** The decision boundary separating classes
- **Support Vectors:** The data points _closest_ to the hyperplane — the only points that define it. Moving any other point leaves the boundary unchanged
- **Margin:** The gap between support vectors on either side. Wider = more robust to unseen data

---

## 📐 Mathematical Framework

### Optimisation Objective

Maximise the margin by minimising the weight vector magnitude, subject to correct classification of all training points: $$\min_{\mathbf{w},, b} \frac{1}{2} |\mathbf{w}|^2$$

The weight vector $\mathbf{w}$ ends up defined **entirely by the support vectors** — all other points are irrelevant to the boundary.

### The C Parameter (Hard vs Soft Margin)

|C Value|Effect|
|---|---|
|**High C**|Hard margin — penalises misclassifications heavily, smaller margin, risk of overfitting|
|**Low C**|Soft margin — allows some misclassifications, wider margin, better generalisation|

---

## 🧮 The Kernel Trick: Handling Non-Linear Data

When no straight line can separate the classes in the original space, kernels project points into a higher-dimensional space where a hyperplane _can_ split them — without the memory cost of actually constructing those extra dimensions.

**Intuition:** A 2D circle of points inside a ring of points can't be linearly separated. Project to 3D → the inner circle rises above the outer ring → a flat plane slices cleanly between them.

### Common Kernel Types

|Kernel|Best For|
|---|---|
|**Linear**|Linearly separable, high-dimensional data (e.g., text)|
|**RBF** (Radial Basis Function)|Non-linear, moderate-dimensional data — most common default|
|**Polynomial**|Curved boundaries; risk of overfitting on noisy data|
|**Sigmoid**|Neural-network-style boundaries|

> 💡 Always use `GridSearchCV` to cross-validate kernel choice — the wrong kernel causes severe over- or under-fitting.

---

## 💻 Python Pipeline

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

# 1. Pipeline: scale first (mandatory), then classify
model = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0))

# 2. Fit
model.fit(X_train, y_train)

# 3. Predict
predictions = model.predict(X_test)

# 4. Tune C and kernel via cross-validation
param_grid = {
    'svc__C': [0.1, 1, 10, 100],
    'svc__kernel': ['linear', 'rbf', 'poly']
}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
```

---

## 🌍 Real-World Use Cases

### Simple — Email Malice Detection

Classify email logs as `Standard` vs `Malicious` by finding the widest margin between the two feature clusters.

### Advanced — Network Anomaly Detection

Ingest high-dimensional security telemetry (`PacketSize`, `Frequency`, `Entropy`) to isolate threat clusters.

> ⚠️ **Stress Point:** Malicious traffic that mimics standard traffic in 2D space is not linearly separable. An RBF or Polynomial kernel is required to project the data into a space where the threat clusters become separable.

---

## ⚠️ Gotchas

> [!warning] Always Scale Features SVM optimises distances. A feature with values in the millions will dominate the margin calculation over a feature in the 0–1 range. Always use `StandardScaler` inside a pipeline before fitting.

> [!warning] Wrong Kernel = Broken Model Linear kernel on curved data → severe underfitting. Polynomial kernel on noisy data → severe overfitting. Never hard-code the kernel — tune with `GridSearchCV`.

> [!warning] Slow on Large Datasets SVM training scales roughly as $O(N^2)$ to $O(N^3)$ with the number of samples. For datasets above ~100k rows, consider `LinearSVC` (much faster) or switch to gradient-boosted trees.

---

## 📊 SVM vs Logistic Regression

|Criterion|SVM|Logistic Regression|
|---|---|---|
|**Objective**|Maximise margin between classes|Maximise log-likelihood of labels|
|**Decision Boundary**|Defined only by support vectors|Influenced by all training points|
|**Non-Linear Data**|Kernel Trick projects to higher dimensions|Requires manual feature engineering|
|**Probabilistic Output**|No native probability (`predict_proba` needs `probability=True`)|Native class probabilities|
|**Scaling Sensitivity**|✅ High — must scale|✅ High — must scale|

---

## ❓ Active Recall Questions

1. Why does moving a non-support-vector data point leave the SVM hyperplane unchanged?
2. How does the **C parameter** trade off between hard margin (perfect training fit) and soft margin (better generalisation)?
3. In what specific scenario would you choose a Linear kernel over an RBF kernel?
4. Why does the Kernel Trick avoid the memory cost of explicitly building higher-dimensional feature spaces?

---

## 🔗 Related Notes

- [[Logistic Regression]]
- [[GridSearchCV and RandomizedSearchCV]]
- [[Cross Validation]]
- [[Decision Trees]]