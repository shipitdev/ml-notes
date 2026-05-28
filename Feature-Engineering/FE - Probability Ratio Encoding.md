# 📊 Feature Engineering: Probability Ratio Encoding for Binary Classifications (Day 5)

---

## 📌 1. The Big Picture: The Mathematics of "Odds"

When your nominal categorical features contain a large number of unique text choices (high cardinality, such as `Cabin blocks` or `Postal Zones`), using standard One-Hot Encoding introduces a high risk of feature explosion.

- **The Goal:** Convert a high-cardinality text column into a single numerical column while capturing how strongly each word choice links to a successful binary outcome.
- **The Strategy:** We calculate the **Odds Ratio**. For each category name, we find the probability that it leads to a positive outcome (`Target = 1`) and divide it by the probability that it leads to a negative outcome (`Target = 0`). This maps text values directly onto a continuous scale representing numerical likelihood.

---

## 📌 2. Phase 1: Proving Target Distributions Across Classes (Hypothesis Validation)

Before transforming your features, you must write exploratory code to prove that your category classes actually hold significantly different probabilities relative to your target variable. If all categories carry roughly identical success rates, this encoding method adds no real predictive value.

### 🔍 Validation Test

Using a sample of the Titanic dataset, we group passengers by their extracted `Cabin_Block` (the first letter of their cabin code) and check the mean of the `Survived` column. This tells us the raw baseline success probabilities for each cabin environment.

---

#### 💻 Python Code Block: Verification Engine

```python
import numpy as np
import pandas as pd

# Setup raw Titanic sample data
data = {
    'Cabin': ['A10', 'B22', 'A35', 'B45', 'A12', 'B80', 'B11', 'A02'],
    'Survived': [1, 0, 1, 0, 0, 1, 0, 1]
}
df = pd.DataFrame(data)

# Extract the general deck group block code (First Letter)
df['Cabin_Block'] = df['Cabin'].str[0]

print("--- Class Distribution Verification ---")
prob_df = df.groupby(['Cabin_Block'])['Survived'].mean().reset_index()
prob_df.rename(columns={'Survived': 'Probability_Survived'}, inplace=True)
print(prob_df)
```

#### 📉 Printed Output Verification

```text
--- Class Distribution Verification ---
  Cabin_Block  Probability_Survived
0           A                  0.75
1           B                  0.25
```

**Conclusion:** The code confirms a clear probability divergence. A passenger in `Cabin_Block A` has a 75% survival probability, while `Cabin_Block B` drops to 25%. This confirms that the variable holds actionable signal for a Probability Ratio map.

---

## 📌 3. Phase 2: The Step-by-Step Probability Ratio Workflow

Once your distribution check passes, the encoding pipeline executes 4 quick steps:

1. Calculate the probability of survival: P(Survived) for each class.
2. Calculate the probability of dying: P(Died) = 1 − P(Survived).
3. Compute the Ratio: P(Survived) / P(Died).
4. Map these ratios back onto the text column.

### ⚠️ How it Relates to ML Models

- **Linear & Logistic Models:** **Exceptional.** This encoding technique directly matches the logarithmic log-odds assumption behind logistic regression models. It structures features into a clean monotonic curve, helping linear algorithms converge with high mathematical precision.
- **Distance Models (KNN):** **Safe.** It compresses an expanding categorical dimension matrix down into a dense, normalised continuous 1D line feature.
- **Tree Models:** Highly effective, though standard decision trees can find these probability splits naturally if given enough depth.
- **🚨 The Production Core Vulnerability — Target Data Leakage:** Because your feature transformations are calculated using your target outcome columns, any small sample sizes can cause mathematical breakdowns. For instance, if a category appears only once and that passenger survived, P(Died) = 0. This forces a **division by zero**, resulting in a ratio of **Infinity**, leading to severe model overfitting.

---

## 📌 4. Multi-Step Data Walkthrough Matrix

### Input (Raw Data Matrix)

- Block `A`: 4 records total → 3 Survived, 1 Died.
- Block `B`: 4 records total → 1 Survived, 3 Died.

|Passenger_ID|Cabin_Block|Survived (Target)|
|:--|:--|:-:|
|101|`A`|1|
|102|`B`|0|
|103|`A`|1|
|104|`B`|0|
|105|`A`|0|
|106|`B`|1|
|107|`B`|0|
|108|`A`|1|

### 🔄 Intermediate Mathematical Logic Check

**Block A:**

$$P(\text{Survived}) = \frac{3}{4} = 0.75 \qquad P(\text{Died}) = 1 - 0.75 = 0.25 \qquad \text{Ratio} = \frac{0.75}{0.25} = \mathbf{3.0}$$

**Block B:**

$$P(\text{Survived}) = \frac{1}{4} = 0.25 \qquad P(\text{Died}) = 1 - 0.25 = 0.75 \qquad \text{Ratio} = \frac{0.25}{0.75} = \mathbf{0.333}$$

### Output (Final Encoded Feature State Matrix)

|Passenger_ID|Cabin_Block_Encoded|Survived (Target)|
|:--|:-:|:-:|
|101|**3.000**|1|
|102|**0.333**|0|
|103|**3.000**|1|
|104|**0.333**|0|
|105|**3.000**|0|
|106|**0.333**|1|
|107|**0.333**|0|
|108|**3.000**|1|

---

## 📌 5. Modular Preprocessing Implementation Code

This multi-module pipeline structures the transformation steps cleanly into isolated, maintainable phases.

### 💻 Python Implementation Pipeline

```python
# --- PHASE 2: CALCULATING PROBABILITIES & ODDS RATIO MAP ---
prob_df['Probability_Died'] = 1 - prob_df['Probability_Survived']

# Compute the probability ratio
prob_df['Probability_Ratio'] = prob_df['Probability_Survived'] / prob_df['Probability_Died']

# Convert the lookup metrics into a standalone dictionary map
ratio_encoder_map = prob_df.set_index('Cabin_Block')['Probability_Ratio'].to_dict()
print("Generated Probability Ratio Lookup Dictionary:")
print(ratio_encoder_map)

# --- PHASE 3: FINAL FEATURE MATRIX TRANSFORMATION ---
df['Cabin_Block_Encoded'] = df['Cabin_Block'].map(ratio_encoder_map)

print("\n=== FINAL PROCESSED SHIFT MATRIX ===")
print(df[['Cabin_Block', 'Survived', 'Cabin_Block_Encoded']])
```

### 📉 Printed Output Verification

```text
Generated Probability Ratio Lookup Dictionary:
{'A': 3.0, 'B': 0.3333333333333333}

=== FINAL PROCESSED SHIFT MATRIX ===
  Cabin_Block  Survived  Cabin_Block_Encoded
0           A         1             3.000000
1           B         0             0.333333
2           A         1             3.000000
3           B         0             0.333333
4           A         0             3.000000
5           B         1             0.333333
6           B         0             0.333333
7           A         1             3.000000
```

---

## 📌 6. Summary Strategic Framework Matrix

|Target Encoding Alternative|Best Application Match|Primary Operational Risk|Primary ML Benefit|
|:--|:--|:--|:--|
|**Mean Target Encoding**|Continuous/Regression target tracking.|Severe data leakage on small sample partitions.|Maps category options directly onto target probability trends.|
|**Probability Ratio Encoding**|Strict **Binary Classification** targets (0 or 1).|**Division by Zero:** Explodes to infinity if a class contains zero negative samples.|Fits perfectly into the log-odds matrix formulations used by logistic models.|

---

## 🔗 Related Notes

- [[Feature Engineering — Advanced Categorical Encoding Taxonomy (Day 4)]]
- [[Feature Engineering — Categorical Imputation & Advanced Encoding Taxonomy (Day 3)]]
- [[Feature Engineering — Missing Data Mechanisms & Continuous Imputation (Day 1)]]
- [[Feature Engineering — Advanced Continuous Imputation (Day 2)]]
- [[Feature Engineering — Feature Scaling]]
- [[Overfitting & Data Leakage]]
- [[Logistic Regression & Log-Odds]]

---

## 📹 Recommended Video Resource

For the live, full-length streaming session demonstrating how to manage edge-case category probabilities manually across complex records, review [Krish Naik's video on Probability Ratio Encoding](https://www.youtube.com/watch?v=daknXSGbTpo).

---

_Tags: #feature-engineering #probability-ratio-encoding #target-encoding #binary-classification #log-odds #logistic-regression #data-leakage #high-cardinality #titanic #python #pandas #machine-learning_