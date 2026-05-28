# 📊 Feature Selection: Dropping Constant Features Using Variance Threshold

---
## 📌 1. The Big Picture: The Informational Value of Spread

When building real-world machine learning models, engineers face the challenge of **High Cardinality** and the **Curse of Dimensionality**. If a dataset expands to hundreds of columns (such as the _Santander Customer Satisfaction_ data featuring 371 variables), training times spike and models overfit on background noise.

The earliest, most foundational step in feature selection is identifying and discarding **Constant Features**. If a column contains the exact same value across every single row (e.g., every customer has a `0` or a `1` in a status code box), that column possesses exactly **zero mathematical variance**.

- **The Informational Reality:** A variable with zero variance provides zero predictive split power. It contains no unique patterns, holds no informational entropy, and cannot help an optimisation model differentiate between output classes.
- **Unsupervised Characteristic:** Because this method calculates variance strictly on input features (X) without looking at the outcome labels (y), it operates as a completely **unsupervised filter selection mechanism**.

---

## 📌 2. Core Strategic Framework: Variance Thresholding

### 🚨 When to Use It

Deploy this filter immediately after completing your initial Feature Engineering cycle (handling missing labels, scaling, encoding) and right after executing your **Train-Test Split**.

### 🎯 Why We Use It

- **Eliminates Structural Redundancy:** It trims completely dead feature axes out of your data arrays, lightening your matrix footprint before running heavy compute models.
- **Bypasses Data Leakage:** By executing `.fit()` exclusively on your training set (`X_train`), you prevent any out-of-sample data properties from leaking into your feature validation boundaries.

---

## 📌 3. Phase 1: Mathematical Logic & Multi-Step Data Walkthrough

To uncover how Scikit-Learn's `VarianceThreshold` identifies constant features under the hood, let's track a raw 4-column matrix containing hidden constant patterns.

---

### 📊 Data Walkthrough

**Input (Raw Feature Observations Matrix)**

- Columns `A` and `B` contain varied, changing numerical measurements.
- Column `C` is hardcoded to a constant `0` for all rows.
- Column `D` is hardcoded to a constant `1` for all rows.

|Row_ID|Column_A|Column_B|Column_C|Column_D|
|:--|:-:|:-:|:-:|:-:|
|0|1.2|4.5|0|1|
|1|2.4|1.1|0|1|
|2|0.8|6.2|0|1|
|3|3.1|2.9|0|1|

**🔄 Intermediate Mathematical Check (Variance Auditing)**

The selector computes the mathematical variance for each independent column array:

- Variance(A) = **0.869** → Exceeds Threshold (> 0) → **KEEP**
- Variance(B) = **3.742** → Exceeds Threshold (> 0) → **KEEP**
- Variance(C) = **0.000** → Matches Zero Condition (≤ 0) → **DROP**
- Variance(D) = **0.000** → Matches Zero Condition (≤ 0) → **DROP**

The system generates boolean support flags via `.get_support()`: `[True, True, False, False]` — where `True` = retain, `False` = flag for extraction.

**Output (Dimension-Optimised Feature State Matrix)**

Columns `C` and `D` are extracted and dropped entirely from the final production layout.

|Row_ID|Column_A|Column_B|
|:--|:-:|:-:|
|0|1.2|4.5|
|1|2.4|1.1|
|2|0.8|6.2|
|3|3.1|2.9|

---

## 📌 4. How it Relates to ML Model Families

- **Linear Models (Linear & Logistic Regression):** **Highly Impacted.** Dropping zero-variance dimensions eliminates uninformative intercept terms. It keeps optimisation algorithms from wasting training steps calculating weights for dead axes that do not shift model predictions.
- **Distance-Based Models (KNN & K-Means):** **Highly Impacted.** Distance calculations evaluate geometric coordinates across dimensions. Retaining features that stay constant adds zero value while expanding the distance grid, directly triggering dimensionality problems and slowing down spatial queries.
- **Tree-Based Models (Random Forest & LightGBM):** **Neutral to Positive.** Decision tree architectures naturally skip over zero-variance columns because they can never compute a valid impurity split on a constant value. However, stripping these columns out before training reduces memory usage, speeding up the construction of deep forest estimators.

---

## 📌 5. Phase 2: Modular Production Pipeline & Code Execution

This script sets up an end-to-end processing pipeline that isolates columns, executes train-test splits to enforce proper data boundaries, calculates variance bounds, and drops constant columns.

### 💻 Python Implementation Pipeline

```python
# =====================================================================
# PHASE 1: DATA INTAKE & ISOLATED SPLIT CONFIGURATION
# =====================================================================
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

# Simulate high-cardinality data mimicking Santander customer blocks
np.random.seed(42)
raw_matrix = {
    'Var_Changing_A': np.random.randn(100),
    'Var_Changing_B': np.random.uniform(10, 50, 100),
    'Var_Constant_C': np.zeros(100),       # Constant column (Variance = 0)
    'Var_Constant_D': np.ones(100),        # Constant column (Variance = 0)
    'Target_Label':   np.random.choice([0, 1], size=100)
}
df = pd.DataFrame(raw_matrix)

# Split independent variables from labels
X = df.drop(columns=['Target_Label'])
y = df['Target_Label']

# Golden Rule: Train-Test split FIRST to prevent data leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("--- Phase 1: Shape of Train Splits Before Selection ---")
print(f"X_train Shape: {X_train.shape} | Column count: {len(X_train.columns)}")

# =====================================================================
# PHASE 2: VARIANCE THRESHOLD FITTING & PROOF
# =====================================================================
# Initialize selector with a strict zero-variance floor threshold
selector = VarianceThreshold(threshold=0.0)

# Fit exclusively on X_train to discover variance patterns safely
selector.fit(X_train)

# Query boolean support flags (True = Keep, False = Drop)
boolean_mask = selector.get_support()
print("\n--- Phase 2: Generated Boolean Support Mask Array ---")
print(boolean_mask)

# =====================================================================
# PHASE 3: DYNAMIC CONSTANT EXTRACTION & FILTER DROPS
# =====================================================================
# Isolate column labels that match the False mask criteria
constant_columns = [col for col in X_train.columns if col not in X_train.columns[boolean_mask]]
print(f"\nIdentified Dead Constant Columns for Drop: {constant_columns}")

# Execute structural drops across both Train and Test partitions
X_train_filtered = X_train.drop(columns=constant_columns)
X_test_filtered  = X_test.drop(columns=constant_columns)

print("\n--- Phase 3: Final Matrix State Shape After Selection ---")
print(f"Final X_train Shape: {X_train_filtered.shape}")
print(X_train_filtered.head(3))
```

### 📉 Printed Output Verification

```text
--- Phase 1: Shape of Train Splits Before Selection ---
X_train Shape: (70, 4) | Column count: 4

--- Phase 2: Generated Boolean Support Mask Array ---
[ True  True False False]

Identified Dead Constant Columns for Drop: ['Var_Constant_C', 'Var_Constant_D']

--- Phase 3: Final Matrix State Shape After Selection ---
Final X_train Shape: (70, 2)
    Var_Changing_A  Var_Changing_B
44       -0.093331       17.409395
19       -1.412304       48.657731
90        0.241962       41.403487
```

---

## 📌 6. Summary Strategic Taxonomy Framework Matrix

|Filter Selection Flavour|Parametric Cutoff|Best Architectural Use-Case|Core Engineering Risk|
|:--|:-:|:--|:--|
|**Strict Constant Selection**|`threshold=0.0`|Dropping absolute dead space columns from highly sparse grids.|Low risk; zero theoretical configuration errors.|
|**Quasi-Constant Selection**|`threshold=0.01` or `0.10`|Removing near-constant columns where 99% of values are identical.|**Information Loss:** Risk of dropping a rare indicator feature that correlates highly with an outcome label.|

---

## 🔗 Related Notes

- [[Data Science — Unified EDA, Feature Engineering & Feature Selection Roadmap]]
- [[Feature Engineering — Feature Scaling & Gaussian Transformations (Day 6)]]
- [[Feature Engineering — Unified Scaling Architecture (Day 6b)]]
- [[Feature Engineering — Advanced Categorical Encoding Taxonomy (Day 4)]]
- [[Feature Engineering — Missing Data Mechanisms & Continuous Imputation (Day 1)]]
- [[Curse of Dimensionality]]
- [[Overfitting & Data Leakage]]

---

## 📹 Recommended Video Resource

[Krish Naik — Dropping Constant Features Using Variance Threshold](https://www.youtube.com/watch?v=uMlU2JaiOd8)

---

_Tags: #feature-selection #variance-threshold #constant-features #quasi-constant #dimensionality-reduction #unsupervised-filter #santander #curse-of-dimensionality #data-leakage #python #sklearn #machine-learning_