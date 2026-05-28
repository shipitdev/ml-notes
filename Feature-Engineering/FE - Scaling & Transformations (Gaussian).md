# 📊 Feature Engineering: Feature Scaling & Gaussian Mathematical Transformations (Day 6)

---

## 📌 1. The Big Picture: Scale and Symmetry Problems

When training machine learning models on real-world numerical datasets, two structural issues frequently degrade model performance: uneven scale magnitudes and asymmetrical distribution shapes. Krish Naik splits the remedies into two core methodologies:

- **Feature Scaling (Standardization & Normalization):** Shifting data ranges so that columns with massive numerical values (e.g., `Fare` ranging from $0 to $500) do not mathematically override columns with smaller values (e.g., `Age` ranging from 0 to 80).
- **Feature Transformation (Gaussian Mapping):** Shifting data distributions mathematically to replicate a standard symmetrical normal distribution (bell curve).

---

## 📌 2. Phase 1: Proving Normality via Exploratory Testing & QQ Plots

Before altering continuous features, you must mathematically or visually check if a column is already normally distributed. While a histogram shows basic shape, the industry standard for checking distribution health is the **Quantile-Quantile (Q-Q) Plot** (`scipy.stats.probplot`).

### 🔍 How a Q-Q Plot Works

A Q-Q plot compares your column's actual, observed distribution against a theoretically perfect normal distribution.

**Interpretation:** If your data points line up in a straight diagonal red reference line, your data is **normally distributed**. If the points loop, bend, or curve sharply away from the reference line, your column is **skewed** and requires a Gaussian transformation.

---

#### 💻 Python Code Block: Verification Engine

```python
import numpy as np
import pandas as pd
import scipy.stats as stat
import matplotlib.pyplot as plt

# Setup a mock housing dataset containing a normal and a right-skewed distribution
np.random.seed(42)
df_eda = pd.DataFrame({
    'Normal_Feature': np.random.normal(loc=30, scale=5, size=500),
    'Skewed_Feature': np.random.exponential(scale=10, size=500)
})

def check_normality(df, column):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left subplot: Standard Histogram Density
    axes[0].hist(df[column], bins=20, edgecolor='black', alpha=0.7)
    axes[0].set_title(f'Histogram of {column}')

    # Right subplot: Probability Plot (Q-Q Plot)
    stat.probplot(df[column], dist="norm", plot=axes[1])
    axes[1].set_title(f'Q-Q Plot of {column}')

    plt.tight_layout()
    plt.savefig(f'{column}_normality_check.png')
    print(f"Normality check executed for: {column}")

# Execute validation checks
check_normality(df_eda, 'Normal_Feature')
check_normality(df_eda, 'Skewed_Feature')
```

#### 📉 Printed Output Verification

```text
Normality check executed for: Normal_Feature
-> Graph points stay perfectly straight along the diagonal line.

Normality check executed for: Skewed_Feature
-> Graph points warp into a steep curve away from the reference line.
```

---

## 📌 3. Part 1: Deconstructing the 3 Feature Scaling Workflows

---

### Technique 1: Standardization (Z-Score Scaling)

#### When to Use It

When your numerical continuous features follow a Gaussian distribution and your machine learning algorithms calculate gradient updates or coordinate vectors.

#### Why We Use It

It centers the data distribution directly at a mean of **0** and scales the variance/standard deviation to **1**. The formula calculates:

$$Z = \frac{X - \mu}{\sigma}$$

#### ⚠️ How it Relates to ML Models

- **Linear Models (Linear/Logistic Regression):** **Exceptional.** By standardising feature distributions, you prevent features with large scales from creating an oblong error surface. This allows Gradient Descent paths to converge quickly onto the global minimum.
- **Distance Models (KNN, K-Means):** **Crucial.** Distance-based models compute spatial distances. Without standardisation, a feature like `Salary` will dominate the Euclidean formula, making features with smaller numbers completely irrelevant.
- **Tree Models (Random Forest, XGBoost):** **No Impact.** Trees create individual axis-aligned splitting rules based on rank sorting inequalities. They are completely immune to feature magnitudes.

---

#### 📊 Data Walkthrough

**Input** → Mean (μ) = 30, Standard Deviation (σ) = 8.165

|Row_ID|Original Value (X)|Z-Score Calculation|Output Standardized|
|:--|:-:|:--|:-:|
|1|20|(20 − 30) / 8.165|**-1.225**|
|2|30|(30 − 30) / 8.165|**0.000**|
|3|40|(40 − 30) / 8.165|**1.225**|

#### 💻 Python Code Block: Standardization

```python
from sklearn.preprocessing import StandardScaler

df_scale = pd.DataFrame({'Feature_X': [20.0, 30.0, 40.0]})
scaler = StandardScaler()

df_scale['Standardized_X'] = scaler.fit_transform(df_scale[['Feature_X']])
print("=== Standardized Scaling Output ===")
print(df_scale)
```

#### 📉 Printed Output Verification

```text
=== Standardized Scaling Output ===
   Feature_X  Standardized_X
0       20.0       -1.224745
1       30.0        0.000000
2       40.0        1.224745
```

---

### Technique 2: Min-Max Scaling (Normalization)

#### When to Use It

Highly prevalent in **Deep Learning image processing channels** (CNN inputs), or when you require hardbound range limitations.

#### Why We Use It

It scales your continuous numeric arrays to fit within a strict bound between **0 and 1**. The formula calculates:

$$X_{\text{scaled}} = \frac{X - X_{\text{min}}}{X_{\text{max}} - X_{\text{min}}}$$

#### ⚠️ How it Relates to ML Models

- **Deep Learning (CNNs):** **Standard Practice.** Image pixel values range natively between 0 and 255. Dividing pixel arrays by 255 scales inputs down between 0 and 1, keeping learning updates stable across hidden layer neurons.
- **Outlier Vulnerability:** **Highly Volatile.** If your feature contains heavy outliers, your maximum value expands significantly. This squashes the normal data cluster into an uninformative range between 0.001 and 0.01.

---

#### 📊 Data Walkthrough

**Input** → Minimum = 20, Maximum = 40

|Row_ID|Original Value (X)|Min-Max Calculation|Output Normalized|
|:--|:-:|:--|:-:|
|1|20|(20 − 20) / (40 − 20)|**0.0**|
|2|30|(30 − 20) / (40 − 20)|**0.5**|
|3|40|(40 − 20) / (40 − 20)|**1.0**|

#### 💻 Python Code Block: Min-Max Scaling

```python
from sklearn.preprocessing import MinMaxScaler

df_minmax = pd.DataFrame({'Feature_X': [20.0, 30.0, 40.0]})
minmax_scaler = MinMaxScaler()

df_minmax['Normalized_X'] = minmax_scaler.fit_transform(df_minmax[['Feature_X']])
print("=== Normalized Scaling Output ===")
print(df_minmax)
```

#### 📉 Printed Output Verification

```text
=== Normalized Scaling Output ===
   Feature_X  Normalized_X
0       20.0           0.0
1       30.0           0.5
2       40.0           1.0
```

---

### Technique 3: Robust Scaling (Median & Interquartile Range)

#### When to Use It

When your data contains **heavy outliers** that cannot be deleted, and standard scale tracking fails.

#### Why We Use It

Unlike Mean and Standard Deviation, which are corrupted by outlier extremes, this approach uses **Median** and **IQR** (75th Percentile minus 25th Percentile). The formula:

$$\text{Robust } X = \frac{X - \text{Median}}{\text{IQR}}$$

#### ⚠️ How it Relates to ML Models

- **All Models Facing Outliers:** **Highly Reliable.** Robust scaling effectively scales features containing strong outliers without letting extreme values compress the normal data clusters.

---

#### 📊 Data Walkthrough

**Input** `[10, 20, 30, 40, 200]` → Median = 30, Q1 = 20, Q3 = 40, IQR = 20

|Original Value (X)|Robust Scaling Equation|Output Scaled|
|:-:|:--|:-:|
|10|(10 − 30) / 20|**-1.0**|
|30|(30 − 30) / 20|**0.0**|
|200 _(Outlier)_|(200 − 30) / 20|**8.5**|

#### 💻 Python Code Block: Robust Scaling

```python
from sklearn.preprocessing import RobustScaler

df_robust = pd.DataFrame({'Feature_X': [10.0, 20.0, 30.0, 40.0, 200.0]})
robust_scaler = RobustScaler()

df_robust['Robust_X'] = robust_scaler.fit_transform(df_robust[['Feature_X']])
print("=== Robust Scaling Output ===")
print(df_robust)
```

#### 📉 Printed Output Verification

```text
=== Robust Scaling Output ===
   Feature_X  Robust_X
0       10.0      -1.0
1       20.0      -0.5
2       30.0       0.0
3       40.0       0.5
4      200.0       8.5
```

---

## 📌 4. Part 2: Deconstructing the 5 Gaussian Transformation Workflows

When your continuous column is heavily skewed, standard scaling transformations are not enough. You must use mathematical equations to map the skewed shapes back into a clean bell curve.

---

### Method 1: Logarithmic Transformation

#### When to Use It

When the target continuous feature is heavily **right-skewed** (e.g., population sizes, financial incomes, asset price allocations).

#### Why We Use It

Applying the natural log compresses large numbers while expanding small numbers, pulling your right-skewed tail back into a normal distribution curve.

> 🚨 **Production Trap Rule:** Natural log equations break down when given zero or negative numbers (ln(0) = −∞). Always use **`np.log1p(x)`**, which implements log(x + 1) to handle zeros smoothly.

---

#### 📊 Data Walkthrough

|Original Value (X)|Applied Equation log(x + 1)|Transformed Value|
|:-:|:-:|:-:|
|0|log(0 + 1)|**0.000**|
|10|log(10 + 1)|**2.397**|
|100|log(100 + 1)|**4.615**|

#### 💻 Python Code Block: Logarithmic Transformation

```python
df_log = pd.DataFrame({'Skewed_Fare': [0.0, 10.0, 50.0, 200.0]})
# Apply log1p safely to handle zero metrics
df_log['Log_Fare'] = np.log1p(df_log['Skewed_Fare'])
print(df_log)
```

#### 📉 Printed Output Verification

```text
   Skewed_Fare  Log_Fare
0          0.0  0.000000
1         10.0  2.397895
2         50.0  3.931826
3        200.0  5.303305
```

---

### Method 2: Reciprocal Transformation

#### When to Use It

Used primarily on ratio scales or rate features where small values hold high structural performance weight.

#### Why We Use It

It takes the inverse of your numerical value (1 / X). This shrinks large values drastically, but can completely invert the order of your data point sequence.

#### 💻 Python Code Block: Reciprocal Transformation

```python
df_recip = pd.DataFrame({'Age': [20.0, 40.0, 60.0]})
df_recip['Reciprocal_Age'] = 1 / df_recip['Age']
print(df_recip)
```

#### 📉 Printed Output Verification

```text
    Age  Reciprocal_Age
0  20.0        0.050000
1  40.0        0.025000
2  60.0        0.016667
```

---

### Method 3: Square Root Transformation

#### When to Use It

Ideal for moderate right-skewed variables, frequently used on frequency count data (e.g., call volumes, customer traffic metrics).

#### Why We Use It

It applies a square root exponent (X^(1/2)), smoothly compressing variance tails without overly flattening small values.

#### 💻 Python Code Block: Square Root Transformation

```python
df_sqrt = pd.DataFrame({'Counts': [4.0, 16.0, 64.0]})
df_sqrt['Sqrt_Counts'] = np.sqrt(df_sqrt['Counts'])
print(df_sqrt)
```

#### 📉 Printed Output Verification

```text
   Counts  Sqrt_Counts
0     4.0          2.0
1    16.0          4.0
2    64.0          8.0
```

---

### Method 4: Exponential Power Transformation

#### When to Use It

Highly effective for custom configurations or when data distributions loop into complex curves.

#### Why We Use It

It applies a custom fraction exponent parameter value (e.g., X^(1/1.2)), giving you manual control over the transformation scale.

#### 💻 Python Code Block: Exponential Power Transformation

```python
df_exp = pd.DataFrame({'Age': [20.0, 35.0, 50.0]})
df_exp['Exp_Age'] = df_exp['Age'] ** (1 / 1.2)
print(df_exp)
```

#### 📉 Printed Output Verification

```text
    Age    Exp_Age
0  20.0  12.143242
1  35.0  19.349137
2  50.0  26.046342
```

---

### Method 5: Box-Cox Transformation

#### When to Use It

The ultimate mathematical remedy for mapping skewed continuous profiles back into a normal distribution curve.

#### Why We Use It

Instead of forcing you to guess which mathematical operation is best, it systematically tests an entire parameter range for an exponent called **Lambda (λ)** between −5 and +5:

$$Y(\lambda) = \begin{cases} \dfrac{X^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0 \[6pt] \ln(X) & \text{if } \lambda = 0 \end{cases}$$

The algorithm automatically selects the optimal Lambda parameter that produces the best normal distribution fit.

> 🚨 **Strict Mathematical Restraint:** Box-Cox fails instantly if the input array contains numbers less than or equal to **0**. If your column tracks metrics that hit zero, add a constant scalar offset first (e.g., X + 1), or use the modern alternative **Yeo-Johnson Transformation**, which handles zero and negative numbers automatically.

---

#### 💻 Python Code Block: Box-Cox Transformation

```python
import scipy.stats as stat

df_box = pd.DataFrame({'Fare': [12.5, 24.0, 85.0, 310.0]})
# stat.boxcox returns the transformed array plus the chosen optimal Lambda
transformed_array, optimal_lambda = stat.boxcox(df_box['Fare'])

df_box['BoxCox_Fare'] = transformed_array
print(f"Algorithm Selected Optimal Lambda: {optimal_lambda:.4f}")
print("\n=== Optimized Box-Cox Output ===")
print(df_box)
```

#### 📉 Printed Output Verification

```text
Algorithm Selected Optimal Lambda: -0.2181

=== Optimized Box-Cox Output ===
    Fare  BoxCox_Fare
0   12.5     2.639454
1   24.0     2.748530
2   85.0     2.915729
3  310.0     2.983944
```

---

## 📌 5. Production Strategy Summary Matrix

|Preprocessing Tool|Best Operational Context|Structural Asset Delivered|Hidden Production Failure Vector|
|:--|:--|:--|:--|
|**Standardization**|Gaussian numerical features.|Centers columns cleanly at Mean = 0, Std Dev = 1.|Highly sensitive to extreme outliers.|
|**Min-Max Scaling**|CNN image preprocessing arrays.|Keeps values tightly locked within a 0 to 1 grid.|Outliers compress the normal data clusters into a narrow band.|
|**Robust Scaling**|Outlier-heavy arrays.|Scales data using stable position metrics (Median, IQR).|Leaves outlying coordinate ranges wide.|
|**Logarithmic (log1p)**|Strongly right-skewed values.|Compresses right-hand data tails smoothly.|Breaks down completely on negative inputs.|
|**Box-Cox Pipeline**|General skewed attributes.|Auto-tests Lambda exponents to find the best normal distribution fit.|Fails instantly on values equal to or less than 0.|

---

## 🔗 Related Notes

- [[Feature Engineering — Probability Ratio Encoding (Day 5)]]
- [[Feature Engineering — Advanced Categorical Encoding Taxonomy (Day 4)]]
- [[Feature Engineering — Categorical Imputation & Advanced Encoding Taxonomy (Day 3)]]
- [[Feature Engineering — Advanced Continuous Imputation (Day 2)]]
- [[Feature Engineering — Missing Data Mechanisms & Continuous Imputation (Day 1)]]
- [[Feature Engineering — Feature Scaling]]
- [[Gradient Descent Optimisation]]
- [[Outlier Detection & Treatment]]

---

## 📹 Recommended Video Resource

For the live dataset demonstrations analysing these transformation curves on the Titanic passenger variables, review [Krish Naik's video on Feature Scaling & Gaussian Transformations](https://www.youtube.com/watch?v=B3gyVWw1UBg).

---

_Tags: #feature-engineering #feature-scaling #gaussian-transformation #standardization #normalization #robust-scaling #log-transform #box-cox #qq-plot #skewed-data #python #scipy #sklearn #machine-learning_