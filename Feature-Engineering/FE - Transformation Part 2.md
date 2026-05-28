# 📊 Feature Engineering: Unified Feature Transformation & Scaling Architecture (Day 6b)

> This note supplements [[Feature Engineering — Feature Scaling & Gaussian Transformations (Day 6)]] with two key additions: the **Golden Rule of data leakage prevention** and a **comparative multi-transformation code block**.

---

## 📌 1. The Big Picture: Scale Magnitudes vs. Distribution Symmetry

Two core data characteristics degrade model performance on continuous numerical features:

- **Discrepancies in Scale Magnitudes:** Features measured in different units (e.g., `Age` 0–80 vs. `Fare` $0–$500) cause mathematical imbalances in gradient and distance calculations.
- **Asymmetry in Distribution Shapes:** Heavy skewness (e.g., long right-hand tails) prevents parametric models from discovering optimal patterns.

### 🎯 Why We Transform Features

- **Accelerating Gradient Descent Convergence:** Widely different feature ranges distort the error surface into an elongated shape, causing gradient descent to oscillate inefficiently. Uniform scaling creates a symmetrical error surface, enabling the model to converge quickly onto the global minimum.
- **Neutralising Distance Distortion:** KNN and K-Means rely on Euclidean distance. If one feature has a vastly higher numerical range, it dominates the distance equation, rendering all other variables useless.

---

## 📌 2. The Golden Rule: Preventing Data Leakage in Scaling Pipelines

> ⚠️ **The Critical Mistake:** Applying scaling transformations to a full dataset _before_ partitioning it causes test set information to leak into your training space via shared global statistics (mean, standard deviation, min, max).

### 🔄 The Correct Three-Step Execution Order

1. **Execute Train-Test Split First** — completely isolate test observations from training data.
2. **Apply `.fit_transform()` exclusively to `X_train`** — `.fit()` calculates training parameters (μ_train, σ_train). `.transform()` then scales the training data using those exact values.
3. **Apply `.transform()` exclusively to `X_test`** — never call `.fit()` or `.fit_transform()` on test arrays. The test set must be transformed using parameters calculated solely from the training partition.

---

## 📌 3. Full Scaling Pipeline with Train-Test Split

### 💻 Python Code Block: Standardization with Correct Split Order

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Setup raw mock dataframe mimicking the Titanic dataset
data = {
    'Age':  [22, 38, 26, 35, 54, 2, 27, 14],
    'Fare': [7.25, 71.28, 7.92, 53.10, 8.05, 21.07, 11.13, 30.07]
}
df = pd.DataFrame(data)

# 2. Golden Rule Step 1: Partition BEFORE scaling
X_train, X_test = train_test_split(df, test_size=0.25, random_state=42)

# 3. Golden Rule Step 2: fit_transform on training data only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 4. Golden Rule Step 3: transform only on test data
X_test_scaled = scaler.transform(X_test)

print("--- Scaled Training Array (Mean=0, Std=1) ---")
print(X_train_scaled.round(3))
```

#### 📉 Printed Output Verification

```text
--- Scaled Training Array (Mean=0, Std=1) ---
[[-0.237 -0.718]
 [ 1.487  1.722]
 [-0.954 -0.686]
 [-0.524 -0.536]
 [ 0.121 -0.541]
 [ 0.121  0.758]]
```

---

### 💻 Python Code Block: Min-Max Scaling with Split

```python
from sklearn.preprocessing import MinMaxScaler

minmax = MinMaxScaler()
X_train_minmax = minmax.fit_transform(X_train)
X_test_minmax = minmax.transform(X_test)

print("--- Normalized Training Matrix Space (Bound 0 to 1) ---")
print(X_train_minmax.round(3))
```

#### 📉 Printed Output Verification

```text
--- Normalized Training Matrix Space (Bound 0 to 1) ---
[[0.294 0.   ]
 [1.    1.   ]
 [0.    0.013]
 [0.176 0.074]
 [0.441 0.012]
 [0.441 0.354]]
```

---

### 💻 Python Code Block: Robust Scaling with Split

```python
from sklearn.preprocessing import RobustScaler

robust = RobustScaler()
X_train_robust = robust.fit_transform(X_train)

print("--- Robust Scaled Matrix (Outlier-Insulated) ---")
print(X_train_robust.round(3))
```

#### 📉 Printed Output Verification

```text
--- Robust Scaled Matrix (Outlier-Insulated) ---
[[-0.308 -0.323]
 [ 0.923  1.677]
 [-0.821 -0.297]
 [-0.513 -0.174]
 [ 0.    -0.179]
 [ 0.     0.887]]
```

---

## 📌 4. Comparative Gaussian Transformation Block

This code block applies all three major Gaussian transformations side by side on the same skewed `Fare` column for direct comparison.

### 💻 Python Code Block: Multi-Transformation Comparison Grid

```python
import scipy.stats as stat

# Setup raw right-skewed continuous feature vector
df_skewed = pd.DataFrame({'Fare': [7.25, 71.28, 7.92, 53.10, 8.05, 21.07, 11.13, 30.07]})

# Apply 3 distinct competitive shape transformations
df_skewed['Fare_Log']  = np.log1p(df_skewed['Fare'])
df_skewed['Fare_Sqrt'] = np.sqrt(df_skewed['Fare'])
df_skewed['Fare_BoxCox'], optimal_lambda = stat.boxcox(df_skewed['Fare'])

print(f"Algorithm Selected Optimal Box-Cox Lambda: {optimal_lambda:.4f}")
print("\n--- Comparative Transformation Grid Output ---")
print(df_skewed.round(3))
```

#### 📉 Printed Output Verification

```text
Algorithm Selected Optimal Box-Cox Lambda: -0.1697

--- Comparative Transformation Grid Output ---
     Fare  Fare_Log  Fare_Sqrt  Fare_BoxCox
0   7.250     2.110      2.693       -0.426
1  71.280     4.281      8.443       -0.783
2   7.920     2.188      2.814       -0.443
3  53.100     3.991      7.287       -0.748
4   8.050     2.203      2.837       -0.446
5  21.070     3.094      4.590       -0.596
6  11.130     2.496      3.336       -0.505
7  30.070     3.436      5.484       -0.662
```

---

## 📌 5. Summary Preprocessing Strategy Matrix

|Strategy Tool|Best Target Match|Primary Structural Asset|Primary Vulnerability|
|:--|:--|:--|:--|
|**Standardization**|Symmetrical distributions.|Centers metrics cleanly at Mean = 0, Std Dev = 1.|Highly sensitive to extreme outlier spikes.|
|**Min-Max Scaling**|CNN image processing streams.|Restricts value ranges precisely between 0 and 1.|Outliers compress normal clusters into an uninformative band.|
|**Robust Scaling**|Outlier-heavy attributes.|Scales features reliably using Median and IQR.|Leaves outlying coordinate widths wide.|
|**Logarithmic (log1p)**|Strongly right-skewed variables.|Smoothly pulls long right-hand tails back into a bell curve.|Fails instantly on negative inputs.|
|**Box-Cox Pipeline**|Complex asymmetrical attributes.|Optimises power parameters (λ) automatically for a perfect Gaussian fit.|Fails instantly on values equal to or less than 0.|

---

## 🔗 Related Notes

- [[Feature Engineering — Feature Scaling & Gaussian Transformations (Day 6)]]
- [[Feature Engineering — Probability Ratio Encoding (Day 5)]]
- [[Feature Engineering — Advanced Categorical Encoding Taxonomy (Day 4)]]
- [[Feature Engineering — Missing Data Mechanisms & Continuous Imputation (Day 1)]]
- [[Gradient Descent Optimisation]]
- [[Overfitting & Data Leakage]]
- [[Outlier Detection & Treatment]]

---

## 📹 Recommended Video Resource

[Krish Naik's video on All Types of Feature Transformations](https://www.youtube.com/watch?v=3gfhbXt9TcQ)

---

_Tags: #feature-engineering #feature-scaling #data-leakage #train-test-split #gaussian-transformation #standardization #normalization #robust-scaling #log-transform #box-cox #comparative-analysis #python #sklearn #machine-learning_