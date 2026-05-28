# 📊 Feature Selection: Removing Highly Correlated Features Using Pearson Correlation

---

## 📌 1. The Big Picture: The Problem of Duplicate Information (Multicollinearity)

When expanding a dataset with extensive feature engineering, you often create independent variables that mirror one another's behaviours. If two or more independent features are highly correlated (e.g., over 80% or 90%), they move together in a near-perfect linear relationship.

- **The Core Problem:** These columns function as duplicate data assets. If `Feature_A` and `Feature_B` express the exact same underlying factor, keeping both columns adds zero new informational value while expanding memory overhead.
- **The Crucial Distinction:**
    - **Independent vs. Dependent Correlation:** If an independent feature is highly correlated with the _target/dependent variable_, you **must retain it** — it is a highly predictive asset.
    - **Independent vs. Independent Correlation:** If two independent features are highly correlated with each other, it introduces **Multicollinearity**, which can destabilise model algorithms. We use a Pearson Correlation Filter to drop the redundant one.

---

## 📌 2. Core Strategic Selection Framework

### 🚨 When to Use It

Deploy this filter immediately after your Train-Test Split to ensure feature processing calculations are performed exclusively on your training data.

### 🎯 Why We Use It

- **Eliminates Feature Redundancy:** It prunes duplicate columns out of high-cardinality feature sets (such as the _Santander Customer Satisfaction_ dataset containing 371 features).
- **Bypasses Data Leakage & Overfitting:** By running the correlation audit strictly on `X_train`, your validation data remains untainted. Once you identify redundant columns from `X_train`, you drop those exact same columns from `X_test` without running a separate correlation test on the test split.

---

## 📌 3. Phase 1: Mathematical Logic & Multi-Step Data Walkthrough

To filter out collinear columns without double-counting pairs or testing a column against itself, the selector maps out the **Upper Triangle** of the correlation matrix.

### 🔍 Upper Triangle Selection Logic

Consider a three-feature matrix with a strict selection threshold set at r > 0.80.

**The Reference Pearson Correlation Matrix Grid**

|Feature Name|Feature_A|Feature_B|Feature_C|
|:--|:-:|:-:|:-:|
|**Feature_A**|1.00|**0.85** _(Checked)_|**0.20** _(Checked)_|
|**Feature_B**|0.85|1.00|**0.40** _(Checked)_|
|**Feature_C**|0.20|0.40|1.00|

**🔄 Transformation Logic Sequence**

1. The engine checks the upper triangle elements row by row using nested indices (i and j).
2. **Pair 1:** Compares `Feature_B` against `Feature_A`. Correlation = **0.85**. Since 0.85 > 0.80, `Feature_B` is flagged as a duplicate and added to the drop list.
3. **Pair 2:** Compares `Feature_C` against `Feature_A`. Correlation = **0.20**. Since 0.20 ≤ 0.80, it is skipped.
4. **Pair 3:** Compares `Feature_C` against `Feature_B`. Correlation = **0.40**. Since 0.40 ≤ 0.80, it is skipped.

**Output Drop List Array**

`Dropped Columns = ['Feature_B']`

---

## 📌 4. How it Relates to ML Model Families

- **Linear Models (Linear & Logistic Regression):** **Highly Impacted — Critical.** Multicollinearity makes the matrix math behind linear regression highly unstable. When features are redundant, small shifts in training values can cause the model's calculated weights (β coefficients) to fluctuate wildly or inflate standard errors, ruining the interpretability of your features.
- **Distance-Based Models (KNN & K-Means):** **Suboptimal.** If you leave multiple features expressing identical information in your columns, you give that underlying concept double the mathematical weight in Euclidean distance calculations, warping spatial results.
- **Tree-Based Models (Random Forest & XGBoost):** **Robust but Beneficial.** Tree algorithms split nodes feature by feature, so multicollinearity does not disrupt predictive accuracy. However, redundant columns dilute feature importance scores across the group, making it difficult to determine which underlying feature is truly driving predictions.

---

## 📌 5. Phase 2: Modular Production Pipeline & Code Execution

This production pipeline demonstrates how to isolate columns, perform train-test splits to avoid data leakage, isolate the upper triangle of a correlation matrix, and drop highly correlated features.

### 💻 Python Implementation Pipeline

```python
# =====================================================================
# PHASE 1: DATA INTAKE & ISOLATED TRAIN-TEST SPLITTING
# =====================================================================
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Simulate a high-cardinality dataset mimicking real-world housing or customer profiles
np.random.seed(42)
base_feature = np.random.rand(100)

raw_matrix = {
    'Feature_A': base_feature,
    'Feature_B': base_feature + np.random.normal(0, 0.05, 100),  # Highly correlated with A (~0.95)
    'Feature_C': np.random.rand(100),
    'Feature_D': np.random.rand(100) + (base_feature * 0.4),      # Moderately correlated (~0.45)
    'Target_Value': (base_feature * 2) + np.random.normal(0, 0.1, 100)
}
df = pd.DataFrame(raw_matrix)

# Split independent variables from target goals
X = df.drop(columns=['Target_Value'])
y = df['Target_Value']

# GOLDEN RULE: Split data before applying feature selection filters
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("--- Phase 1: Training Dimensions Before Filter ---")
print(X_train.head(3))

# =====================================================================
# PHASE 2: UPPER TRIANGLE MULTICOLLINEARITY SELECTION FUNCTION
# =====================================================================
def identify_highly_correlated_features(dataset, threshold=0.85):
    """
    Traverses the upper triangle of a Pearson correlation matrix
    to isolate duplicate feature names exceeding a strict cutoff.
    """
    col_corr = set()  # Set container ensures unique element groupings
    corr_matrix = dataset.corr()

    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            # Evaluate absolute value to catch both high positive and negative trends
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return list(col_corr)

# Extract redundant columns exclusively from X_train
redundant_features = identify_highly_correlated_features(X_train, threshold=0.80)
print(f"\nIdentified Redundant Columns for Extraction: {redundant_features}")

# =====================================================================
# PHASE 3: UNBIASED SELECTION DROPS APPLIED TO BOTH PARTITIONS
# =====================================================================
# Drop the identified columns from both training and test sets symmetrically
X_train_clean = X_train.drop(columns=redundant_features)
X_test_clean  = X_test.drop(columns=redundant_features)

print("\n--- Phase 3: Optimised Matrix Verification ---")
print(X_train_clean.head(3))
```

### 📉 Printed Output Verification

```text
--- Phase 1: Training Dimensions Before Filter ---
    Feature_A  Feature_B  Feature_C  Feature_D
44   0.258780   0.264701   0.490812   0.617467
19   0.291229   0.316829   0.141297   0.155799
90   0.063558   0.093405   0.505562   0.038479

Identified Redundant Columns for Extraction: ['Feature_B']

--- Phase 3: Optimised Matrix Verification ---
    Feature_A  Feature_C  Feature_D
44   0.258780   0.490812   0.617467
19   0.291229   0.141297   0.155799
90   0.063558   0.505562   0.038479
```

---

## 📌 6. Summary Strategic Taxonomy Framework Matrix

|Feature Selection Method|Mathematical Trigger|Evaluation Scope|Operational Strategy|
|:--|:--|:--|:--|
|**Variance Thresholding**|Absolute Spread (σ² ≤ 0)|Single column input vector|Unsupervised filter; removes dead columns containing zero changing value.|
|**Pearson Correlation Filter**|Linear Association (r > Threshold)|Pairwise column-to-column comparison|Supervised-free filter; removes one column from each redundant correlated pair.|

---

## 🔗 Related Notes

- [[Feature Selection — Dropping Constant Features Using Variance Threshold]]
- [[Data Science — Unified EDA, Feature Engineering & Feature Selection Roadmap]]
- [[Feature Engineering — Feature Scaling & Gaussian Transformations (Day 6)]]
- [[Feature Engineering — Encoding Categorical Variables]]
- [[Dummy Variable Trap & Multicollinearity]]
- [[Overfitting & Data Leakage]]
- [[Curse of Dimensionality]]

---

## 📹 Recommended Video Resource

[Krish Naik — Dropping Features Using Pearson Correlation](https://www.youtube.com/watch?v=FndwYNcVe0U)

---

_Tags: #feature-selection #pearson-correlation #multicollinearity #correlation-matrix #upper-triangle #redundant-features #dimensionality-reduction #santander #data-leakage #python #pandas #sklearn #machine-learning_