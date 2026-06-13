
---
## ⚡ TL;DR

**GridSearchCV** is the exhaustive counterpart to RandomizedSearchCV. Instead of sampling randomly, it evaluates **every single combination** in your parameter grid using k-fold cross-validation — guaranteeing the mathematically optimal configuration within your defined search space.

Use it when your parameter space is small (≤ 3 hyperparameters) and you need certainty over speed.

---

## 📌 1. The Big Picture: Why Exhaustive Search Exists

RandomizedSearchCV is fast but probabilistic — it may miss the true optimal combination simply because it never sampled it. GridSearchCV removes that uncertainty entirely. Every combination is tested, every score is recorded, and the winner is mathematically guaranteed to be the best within your grid.

The cost is computational. A grid with 4 parameters × 5 values each = 5⁴ = 625 combinations × 5-fold CV = **3,125 model fits**. Add one more parameter and you're at 15,625. This is why GridSearchCV is reserved for smaller, well-understood search spaces — not open-ended exploration.

---

## 📌 2. Real-World Use-Cases

**Simple Application — Logistic Regression Threshold Tuning:** Testing combinations of regularisation strength `C` and solver type across a binary classification problem where you can afford exhaustive evaluation.

**Production Architecture — SVM Kernel Optimisation Pipeline:** When deploying a Support Vector Machine on tabular financial data, the kernel type (`rbf`, `linear`, `poly`) combined with `C` and `gamma` form a small, discrete grid. GridSearchCV exhaustively finds the exact kernel configuration that maximises precision on rare fraud events — where missing the true optimal combination is not acceptable.

> **Stress Point:** Grid search on an SVM with an `rbf` kernel scales as O(n²) with training data size. On datasets above 50,000 rows, even a small 3-parameter grid becomes computationally expensive. Always benchmark on a data subset before committing to full-scale grid search.

---

## 📌 3. Key Questions

1. If GridSearchCV guarantees the best combination within your grid, why might the selected parameters still underperform on unseen production data?
2. What is the mathematical relationship between `n_splits` in cross-validation and total model fits in GridSearchCV?
3. Why does GridSearchCV's exhaustive nature make it **less** suitable as the parameter space grows, even though it is more thorough?

---

## 📌 4. Mathematical Framework

---

### 1. Total Fits Formula

For a grid with $k$ parameters, each containing $n_i$ values, and $f$-fold cross-validation:

$$\text{Total Fits} = f \times \prod_{i=1}^{k} n_i$$

**Example:** 3 parameters with [5, 4, 3] values, 5-fold CV:

$$\text{Total Fits} = 5 \times (5 \times 4 \times 3) = 5 \times 60 = 300 \text{ fits}$$

---

### 2. Cross-Validation Score Per Combination

For each parameter combination $\theta$, GridSearchCV computes the mean validation score across all $f$ folds:

$$\text{CV Score}(\theta) = \frac{1}{f} \sum_{i=1}^{f} \text{Score}(\theta, \text{fold}_i)$$

The winning configuration $\theta^*$ is:

$$\theta^* = \arg\max_{\theta} \text{ CV Score}(\theta)$$

---

### 3. Regularisation Parameter C (for SVM / Logistic Regression)

A common GridSearchCV target — `C` controls the penalty for misclassification:

$$\text{Objective} = \frac{1}{2} |w|^2 + C \sum_{i=1}^{n} \max(0, 1 - y_i(w \cdot x_i + b))$$

- **Low C** → larger margin, more misclassifications tolerated → underfitting risk
- **High C** → tight margin, few misclassifications tolerated → overfitting risk

**Tuning Range:** `C` ∈ [0.001, 0.01, 0.1, 1, 10, 100]

---

## 📌 5. Grid vs. Randomised — Decision Framework

|Dimension|GridSearchCV|RandomizedSearchCV|
|:--|:--|:--|
|**Search method**|Every combination|Random sample of `n_iter` combinations|
|**Guarantee**|Best within grid|Probabilistic best|
|**Computation**|Exponential|Linear (controlled by `n_iter`)|
|**Best for**|≤ 3 parameters, small grids|≥ 4 parameters, large value ranges|
|**Risk**|Computationally locked on large grids|May miss optimal combination|
|**Typical use**|Final fine-tuning after random search narrows the space|Initial broad exploration|

> **Pro workflow:** Run RandomizedSearchCV first to identify the promising region of parameter space → then run GridSearchCV on a tightened grid around those values. You get speed **and** certainty.

---

## 📌 6. Numerical Walkthrough: Logistic Regression Tuning

### The Parameter Grid

```python
param_grid = {
    "C":       [0.001, 0.01, 0.1, 1, 10, 100],
    "penalty": ["l1", "l2"],
    "solver":  ["liblinear", "saga"]
}
# Total combinations: 6 × 2 × 2 = 24
# With 5-fold CV: 24 × 5 = 120 fits
```

**Step 1:** Define the full parameter grid — every value you want evaluated.

**Step 2:** Mount GridSearchCV with `refit=True` so the best estimator is automatically retrained on the full training set.

**Step 3:** After fitting, inspect `.best_params_`, `.best_score_`, and `.cv_results_` for the full score matrix.

```python
winning_config = {
    "C":       1,
    "penalty": "l2",
    "solver":  "liblinear"
}
# Best CV Score: 0.8742
```

---

## 📌 7. Python Implementation

### 💻 Python Code

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

def execute_grid_search_pipeline(X, y):
    """
    Exhaustively searches every hyperparameter combination using GridSearchCV
    and returns the best estimator with its winning configuration.
    """
    # 1. Scale features — critical for regularisation-sensitive models
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Define exhaustive parameter grid
    param_grid = {
        "C":       [0.001, 0.01, 0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
        "solver":  ["liblinear", "saga"]
    }

    # 3. Mount the exhaustive search engine
    grid_search = GridSearchCV(
        estimator=LogisticRegression(random_state=42, max_iter=1000),
        param_grid=param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        refit=True,
        verbose=0
    )

    grid_search.fit(X_scaled, y)

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


# =====================================================================
# PIPELINE EXECUTION VERIFICATION
# =====================================================================
if __name__ == "__main__":
    X_mock, y_mock = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        random_state=42
    )

    best_model, best_params, best_score = execute_grid_search_pipeline(X_mock, y_mock)

    print("=====================================================================")
    print("GRIDSEARCHCV EXHAUSTIVE OPTIMISATION SUMMARY")
    print("=====================================================================")
    print(f"Optimal C (Regularisation)       : {best_params['C']}")
    print(f"Optimal Penalty                  : {best_params['penalty']}")
    print(f"Optimal Solver                   : {best_params['solver']}")
    print("---------------------------------------------------------------------")
    print(f"Best Cross-Validation ROC-AUC    : {best_score:.4f}")
    print("=====================================================================")
```

### 📉 Printed Output Verification

```text
=====================================================================
GRIDSEARCHCV EXHAUSTIVE OPTIMISATION SUMMARY
=====================================================================
Optimal C (Regularisation)       : 1
Optimal Penalty                  : l2
Optimal Solver                   : liblinear
---------------------------------------------------------------------
Best Cross-Validation ROC-AUC    : 0.8742
=====================================================================
```

---

## 📌 8. Gotchas & Misconceptions

- **The Data Leakage Trap:** If you scale your features before splitting into train/test, then pass the full dataset into GridSearchCV, the scaler has already seen the test data. Always fit the scaler inside a `Pipeline` object so scaling is applied only within each CV fold — never to the held-out validation data.
- **The Boundary Problem:** If GridSearchCV selects a parameter sitting at the edge of your grid (e.g. `C=100` when your grid goes up to 100), the true optimum likely lies beyond your boundary. Extend the grid in that direction and re-run.
- **`refit=True` Is Not Optional in Production:** Without `refit=True`, GridSearchCV does not retrain on the full dataset after finding the best params. You get the winning configuration but not a production-ready model.

---

## 📌 9. Active Recall Questions

1. You run GridSearchCV and the best `C` value is `100` — the highest in your grid. What should you do next and why?
2. Why must feature scaling be placed **inside** a `Pipeline` when used with GridSearchCV, rather than applied to `X` before the search?
3. GridSearchCV with 5 parameters × 6 values each and 5-fold CV — how many total model fits will execute? Is this advisable?

---

## 🔗 Related Notes

- [[ML — XGBoost HyperParameter Optimisation with RandomizedSearchCV]]
- [[ML — Decision Trees & Entropy Beginners Guide]]
- [[ML — Ensemble Methods Bagging & Random Forest]]
- [[ML — Classification Performance Metrics Part 1]]
- [[ML — ROC and AUC Curve Intuition]]

---

## 📹 Recommended Video Resource

[Krish Naik — GridSearchCV vs RandomizedSearchCV](https://www.youtube.com/results?search_query=krish+naik+gridsearchcv)

---

_Tags: #machine-learning #gridsearchcv #hyperparameter-tuning #cross-validation #logistic-regression #sklearn #exhaustive-search #regularisation #python #data-science_