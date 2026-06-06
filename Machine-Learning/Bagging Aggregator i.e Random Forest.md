# 🌳 Machine Learning: Ensemble Methods — Bagging & Random Forest Frameworks

---

## ⚡ TL;DR

A single Decision Tree grown to maximum depth behaves like a student who memorises every detail of a practice test — **Low Bias (flawless training accuracy)** but **High Variance (poor generalisation to new data)**, leading to severe overfitting.

**Bagging (Bootstrap Aggregation)** fixes this by building multiple baseline models in parallel. Each model trains on a random subset of rows sampled **with replacement** (Bootstrapping). Predictions are then combined via majority vote (classification) or mean (regression).

**Random Forest** optimises this further by also randomly sampling a subset of **features (columns)** at each split — ensuring trees remain highly independent, diverse, and robust.

---

## 📌 1. The Big Picture: The Bias-Variance Trade-off

### The Problem with a Single Deep Tree

When growing a standalone Decision Tree to full depth, it exhibits these structural properties:

- **Low Bias:** By recursively splitting nodes down to individual leaf purity, the tree warps its boundaries to fit the training set perfectly. Training error approaches zero.
- **High Variance:** Because the boundaries are tightly wrapped around training points, minor variations or noise in new test data cause the tree's decision paths to fail, leading to unstable predictions.

### The Ensemble Resolution Strategy

To turn high variance into **Low Variance** without destroying your low bias profile, you build an army of parallel expert trees. By forcing each tree to look at a slightly different version of the dataset, you create a diverse group of models. When you combine their independent predictions, the random errors cancel each other out, stabilising your final predictions.

---

## 📌 2. Mathematical Framework & Sampling Logic

---

### Bootstrapping: Row Sampling with Replacement

Given a master training dataset D containing N total records, the Bagging engine generates K independent bootstrap datasets (D₁, D₂, ..., D_k). To build each bootstrap set, the engine randomly selects a row from D, copies it into the new sample, and **puts it back** into the master pool before drawing the next row.

$$D_k \subset D \qquad |D_k| < |D|$$

**The Mathematical Reality:** Because rows are placed back into the pool, any specific row has approximately a **63.2% chance** of being selected at least once in a bootstrap sample. The remaining **36.8%** of unselected rows are called **Out-Of-Bag (OOB) samples** — these can be used as an integrated test set to validate the model without a separate cross-validation loop.

---

### Random Forest Feature Selection (Column Sampling)

A standard Bagging ensemble passes _all_ available features (M) to its base models. Random Forest introduces **Column Sampling** — at each node split, the algorithm restricts the available features to a random subset of size m:

$$m = \sqrt{M} \quad \text{(Classification)} \qquad m = \frac{M}{3} \quad \text{(Regression)}$$

**Why this is critical:** If your dataset contains one overwhelmingly dominant feature (e.g., `OverallQual` in housing data), every single tree in a standard Bagging ensemble will immediately select that feature for its first split. This makes trees highly correlated — their errors won't cancel each other out when averaged. Limiting features available at each split forces some trees to learn from alternative signals, creating a highly independent and robust ensemble.

---

## 📌 3. The Aggregation Loop

Once all K base decision trees are fully trained across their respective randomised rows and columns, predictions on new test records are processed through an aggregation layer.

---

### Case A: Random Forest Classifier — Majority Vote

Each tree casts a single discrete vote for a class label. The class receiving the most votes becomes the final prediction:

$$\hat{y} = \text{mode}{M_1(X), M_2(X), \dots, M_K(X)}$$

---

### Case B: Random Forest Regressor — Continuous Average

Each individual tree outputs a continuous numeric value. The final ensemble prediction is the arithmetic mean of those values:

$$\hat{y} = \frac{1}{K}\sum_{k=1}^{K} M_k(X)$$

---

### Aggregation Flow Summary

|Input|Tree 1|Tree 2|Tree K|Aggregation|Final Output|
|:--|:-:|:-:|:-:|:--|:--|
|Test Record X|Output: 1|Output: 0|Output: 1|Majority Vote|**1** (2 vs. 1)|
|Test Record X|Output: 142k|Output: 138k|Output: 155k|Mean Average|**145k**|

---

## 📌 4. Python Production Simulation Pipeline

### 💻 Python Code

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

def execute_ensemble_architecture_validation(X, y, task_type='classification'):
    """
    Instantiates and fits an optimised Random Forest ensemble,
    configuring explicit row bootstrapping and feature subset selection limits.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if task_type == 'classification':
        # max_features='sqrt' triggers Column Sampling (m = sqrt(M))
        # bootstrap=True  triggers Row Sampling with replacement
        ensemble_model = RandomForestClassifier(
            n_estimators=100,      # Number of parallel decision trees (K)
            max_features='sqrt',   # Feature subset selection size limit (m)
            bootstrap=True,        # Enable row bootstrapping
            oob_score=True,        # Calculate Out-of-Bag validation accuracy
            random_state=42,
            n_jobs=-1              # Utilise all available CPU cores for parallel training
        )
    else:
        ensemble_model = RandomForestRegressor(
            n_estimators=100,
            max_features=0.33,     # Regression feature limit benchmark (M / 3)
            bootstrap=True,
            oob_score=True,
            random_state=42,
            n_jobs=-1
        )

    ensemble_model.fit(X_train, y_train)

    return ensemble_model, ensemble_model.oob_score_

# =====================================================================
# ENSEMBLE SYSTEM VERIFICATION RUN
# =====================================================================
if __name__ == "__main__":
    np.random.seed(42)
    samples        = 200
    features_count = 10

    X_mock = np.random.normal(0, 1, size=(samples, features_count))
    y_mock = np.random.choice([0, 1], size=samples)

    fitted_rf, oob_accuracy = execute_ensemble_architecture_validation(
        X_mock, y_mock, task_type='classification'
    )

    print("=====================================================================")
    print("RANDOM FOREST ENSEMBLE INITIALISATION LOGS")
    print("=====================================================================")
    print(f"Total Base Learners Trained (K)          : {fitted_rf.n_estimators}")
    print(f"Feature Sampling Strategy Setting        : {fitted_rf.max_features}")
    print(f"Out-Of-Bag (OOB) Validation Score        : {oob_accuracy * 100:.2f}%")
    print("=====================================================================")
```

> **Note:** There is a typo in the source — `test_split=0.2` should be `test_size=0.2`. The corrected version is above.

---

## 📌 5. Gotchas & Misconceptions

- **The Data Shock Robustness:** What happens if 20% of the rows in your dataset change overnight? In a single Decision Tree, this variance shift can completely alter your tree splits. In a Random Forest, because the changes are distributed across independent bootstrap samples, the overall ensemble remains highly stable and robust to localised data shocks.
- **The Scale Invariance Benefit:** Unlike linear models (Ridge/Lasso) or distance-based models (KNN/SVM), Random Forests rely on threshold splits (X_i ≤ θ). This means they are **completely invariant to feature scaling**. Running a log transform or Z-score normalisation will not change how a tree splits — you can skip extensive scaling steps when feeding data into a Random Forest pipeline.

---

## 📌 6. Summary Comparison Taxonomy

|Framework|Base Learner|Row Sampling|Feature Sampling|Aggregation|
|:--|:--|:--|:--|:--|
|**Standard Bagging**|Any algorithm (KNN, Linear, Trees).|Bootstrapping — rows sampled with replacement.|Retains 100% of available features (M).|Majority vote or continuous mean.|
|**Random Forest**|Strictly Decision Trees grown to full depth.|Bootstrapping — rows sampled with replacement.|Random subset at each split (m = √M for classification).|Majority vote (Classifier) or mean (Regressor).|

---

## 📌 7. Active Recall Questions

1. Why does restricting the number of available features to m = √M at each split node lower the correlation between separate trees in a Random Forest?
2. If your training data contains a massive class imbalance, why can standard row bootstrapping without stratification result in individual base trees completely missing the minority class?
3. How can you use a Random Forest's Out-of-Bag (OOB) score to validate your model configuration without using a separate validation split or cross-validation loop?

---

## 🔗 Related Notes

- [[ML — Decision Trees & Entropy Beginners Guide]]
- [[ML — Performance Metrics Multi-Class Classification]]
- [[ML — ROC and AUC Curve Intuition]]
- [[ML — Classification Performance Metrics Part 1]]
- [[ML — Dimensionality Reduction VIF vs PCA]]
- [[Feature Selection — Dropping Constant Features Using Variance Threshold]]
- [[Data Science — Unified EDA, Feature Engineering & Feature Selection Roadmap]]

---

_Tags: #machine-learning #ensemble-methods #bagging #random-forest #bootstrap-aggregation #decision-tree #bias-variance-tradeoff #oob-score #feature-sampling #column-sampling #classification #regression #sklearn #python #data-science_