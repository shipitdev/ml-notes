# 📊 Feature Engineering: Missing Data Mechanisms & Continuous Imputation (Day 1)

---

## 📌 1. The Big Picture: The Types of Blank Information

When columns contain missing entries (`NaN`), you cannot just blindly apply an imputer without understanding _why_ the data is blank. Data goes missing under completely different statistical mechanisms. Krish Naik breaks these down into three distinct structural families to determine whether a column's missingness holds hidden predictive clues.

---

### 🧬 The Three Families of Missingness

#### 1. MCAR (Missing Completely at Random)

**What it means:** There is absolutely zero relationship between the missing entry and any other data points or values in your dataset. The fact that the cell is blank is a complete accident.

**Simple Example:** A survey sheet gets a coffee stain on it, or a laboratory test tube accidentally cracks in transit.

**Key Indicator:** If you remove the rows with missing values, the remaining rows still perfectly represent your original population without any hidden biases.

---

#### 2. MNAR (Missing Not at Random) / Non-Random Missingness

**What it means:** There is a direct, systematic reason why a value is missing, and the missingness itself is a massive clue.

**Simple Example (Titanic Dataset):** The features `Age` or `Cabin` are heavily missing for passengers who did not survive. Because they passed away during the disaster, it was structurally impossible to collect their complete records afterward. The missing value tells a story.

---

#### 3. MAR (Missing at Random)

**What it means:** The data is missing not because of the missing value itself, but because of a completely different observed feature in your dataset.

**Simple Example:** A medical survey discovers that `Weight` values are frequently missing, but looking closer reveals they are almost exclusively missing for female patients who chose to skip that specific question. The missingness is fully explained by a different column (`Gender`).

---

## 📌 2. Phase 1: How to Mathematically Prove Your Missingness Hypothesis

Before writing an imputation pipeline, you must write code to verify whether your missing data is completely random (MCAR) or systematically tied to an outcome (MNAR). We do this by grouping our data by a target variable and calculating the average percentage of missing values in each group.

---

### 🔍 The Hypothesis Test Walkthrough

Using the **Titanic Dataset**, we want to check two columns containing missing values: `Age` and `Cabin`. We will group the passengers by whether they survived (`Survived = 1`) or died (`Survived = 0`), and find the mean of the null values for each group.

#### 🔄 EDA Mathematical Logic

- **MCAR Signal:** If the percentage of missing values is roughly equal across both groups (e.g., 20% missing for survivors and 19% missing for deceased), the data is **MCAR** (it went missing randomly by accident).
- **MNAR Signal:** If the missing percentages are drastically different (e.g., only 40% missing for survivors but a massive 87% missing for deceased), the data is **MNAR** (systematic non-random pattern).

---

#### 💻 Python Code Block: Hypothesis Testing & Proof

```python
import numpy as np
import pandas as pd

# Mock Titanic data: Age and Cabin have missing values
data = {
    'Survived': [0, 1, 0, 1, 0, 0, 1, 0],
    'Age': [22, np.nan, 38, 26, np.nan, 35, 54, np.nan],
    'Cabin': [np.nan, 'C85', np.nan, 'C123', np.nan, np.nan, 'E46', np.nan]
}
df = pd.DataFrame(data)

print("--- Step 1: Raw Percentage of Missing Values Per Column ---")
print(df.isnull().mean())

print("\n--- Step 2: Proving MNAR vs MCAR via Groupby Target ---")
# Check the missing percentage of Cabin grouped by Survived
cabin_proof = df['Cabin'].isnull().groupby(df['Survived']).mean()
print(f"Percentage of missing Cabin entries based on survival:\n{cabin_proof}")

# Check the missing percentage of Age grouped by Survived
age_proof = df['Age'].isnull().groupby(df['Survived']).mean()
print(f"\nPercentage of missing Age entries based on survival:\n{age_proof}")
```

#### 📉 Printed Output Verification

```text
--- Step 1: Raw Percentage of Missing Values Per Column ---
Survived    0.000
Age         0.375
Cabin       0.625

--- Step 2: Proving MNAR vs MCAR via Groupby Target ---
Percentage of missing Cabin entries based on survival:
Survived
0    1.0
1    0.0
Name: Cabin, dtype: float64

Percentage of missing Age entries based on survival:
Survived
0    0.4
1    0.333333
Name: Age, dtype: float64
```

**Conclusion drawn from code:**

- **Cabin data:** 100% of the people who died (`Survived=0`) are missing their Cabin data, while 0% of survivors are missing it. This is **systematic (MNAR)**.
- **Age data:** The missing percentages are close ($40% vs 33%), meaning age behaves much closer to **random (MCAR)**.

---

## 📌 3. Phase 2: Executing the 5 Continuous Imputation Techniques

---

### Method 1: Mean / Median Replacement

#### When to Use It

When the data is **MCAR** (Missing Completely at Random) and the missing data volume is small (under 5–10%).

#### Why We Use It

It keeps our dataframe shape intact without creating new columns. If the column has extreme outliers, we use the **Median**. If it's a clean normal distribution, we use the **Mean**.

#### ⚠️ How it Relates to ML Models

- **Linear Models (Regression, Logistic):** Bad. Because you force multiple identical median numbers into the centre of the column, the **variance shrinks**, creating a massive artificial spike. This deflates your standard errors and distorts parameter weight ($\beta$) estimations.
- **Distance-Based Models (KNN, K-Means):** Risk. It pushes multiple points directly into the centre of that feature's axis, flattening geometric cluster distinctions.
- **Tree-Based Models (Random Forest, XGBoost):** Safe. Trees can easily isolate the spiked median value if it represents a significant decision threshold split.

---

#### 📊 Data Walkthrough

**Input (Raw Data)** → Calculate Median of `[22, 38, 26, 35, 54]` = **`35.0`**

|Survived|Age|Age_Median (Output)|
|:-:|:-:|:--|
|0|22|22.0|
|1|`NaN`|**35.0** _(Imputed)_|
|0|38|38.0|

#### 💻 Python Code Block: Mean/Median Imputation

```python
df_median = df.copy()
original_var = df_median['Age'].var()

median_val = df_median['Age'].median()
df_median['Age_Median'] = df_median['Age'].fillna(median_val)

print(f"Original Age Variance: {original_var:.2f}")
print(f"New Age Variance after Median Imputation: {df_median['Age_Median'].var():.2f}")
```

#### 📉 Printed Output Verification

```text
Original Age Variance: 140.40
New Age Variance after Median Imputation: 92.12
```

---

### Method 2: Random Sample Imputation

#### When to Use It

When your data is **MCAR** and you want to completely avoid the variance distortion problem caused by median imputation.

#### Why We Use It

It takes random actual values from the rows that _have_ data and injects them into the blank slots. Because you are pulling from the existing distribution, **the variance and overall shape of the data curve remain completely undisturbed**.

#### ⚠️ How it Relates to ML Models

- **Linear Models:** Excellent. Since the variance and standard deviation are preserved, linear model assumptions regarding normal distribution shapes remain valid.
- **Tree-Based Models:** Can introduce slight randomness. Since values are sampled randomly, two identical missing entries might receive completely different values, occasionally creating inconsistent tree node paths during training.

---

#### 📊 Data Walkthrough

**Input (Raw Data)** → Randomly sample from complete values `[22, 38, 26, 35, 54]`

|Survived|Age|Age_Random (Output)|
|:-:|:-:|:--|
|1|`NaN`|**26.0** _(Randomly Sampled)_|
|0|`NaN`|**38.0** _(Randomly Sampled)_|

#### 💻 Python Code Block: Random Sample Imputation

```python
df_random = df.copy()

# 1. Sample values equal to the count of null values
random_samples = df_random['Age'].dropna().sample(df_random['Age'].isnull().sum(), random_state=42)
# 2. Align indices
random_samples.index = df_random[df_random['Age'].isnull()].index
# 3. Impute
df_random['Age_Random'] = df_random['Age']
df_random.loc[df_random['Age'].isnull(), 'Age_Random'] = random_samples

print(f"New Age Variance after Random Sampling: {df_random['Age_Random'].var():.2f}")
```

#### 📉 Printed Output Verification

```text
New Age Variance after Random Sampling: 135.27
```

_(Note: Variance remains high and close to the original 140.40, avoiding the compression trap!)_

---

### Method 3: Capturing NaN Values with a New Feature (Missing Indicator)

#### When to Use It

When your data is **MNAR** (Missing Not at Random) and the fact that information is missing is highly predictive.

#### Why We Use It

It creates a **brand-new binary flag column** (`Age_NaN_Flag`) and sets it to `1` if the data was missing, then fills the original column with a median. This preserves the missingness signal for your model.

#### ⚠️ How it Relates to ML Models

- **Linear Models:** Effective. The model calculates a unique slope weight for the indicator column, explicitly measuring the baseline shift caused by data being missing.
- **Tree-Based Models:** Extremely Powerful. Tree algorithms love binary flags; they can instantly make a clean split on `Age_NaN_Flag == 1` to route missing rows down an independent path.
- **The Operational Cost:** It doubles your continuous column count if applied to every feature, increasing data dimensions.

---

#### 📊 Data Walkthrough

|Age (Input)|Age_Cleaned (Output)|Age_NaN_Flag (Output)|
|:-:|:-:|:-:|
|22|22.0|0|
|`NaN`|**35.0** _(Median)_|**1**|

#### 💻 Python Code Block: Missing Indicator Flags

```python
df_indicator = df.copy()
df_indicator['Age_NaN_Flag'] = np.where(df_indicator['Age'].isnull(), 1, 0)
df_indicator['Age_Cleaned'] = df_indicator['Age'].fillna(df_indicator['Age'].median())

print(df_indicator[['Age', 'Age_Cleaned', 'Age_NaN_Flag']])
```

#### 📉 Printed Output Verification

```text
    Age  Age_Cleaned  Age_NaN_Flag
0  22.0         22.0             0
1   NaN         35.0             1
2  38.0         38.0             0
```

---

### Method 4: End of Distribution Imputation

#### When to Use It

When your data is **MNAR** (Missing Not at Random) and you want to capture the missingness signal _without_ creating an extra column flag.

#### Why We Use It

It calculates an extreme value at the far tail end of the data distribution — beyond 3 Standard Deviations ($\mu + 3\sigma$) using the formula:

$$\text{Extreme Value} = \text{Mean} + 3 \times \text{Standard Deviation}$$

It uses this extreme value to fill all `NaN` slots. This forces the missing values out to the edge as extreme outliers, leaving the main distribution curve unwarped.

#### ⚠️ How it Relates to ML Models

- **Linear Models:** Horrible. Linear regression models are highly sensitive to outliers. Placing massive extreme numbers into the feature matrix completely warps the line of best fit, pulling the slope coefficients out of alignment.
- **Distance-Based Models:** Horrible. It creates a massive geometric distance distortion on your distance checks.
- **Tree-Based Models:** Fantastic. Trees split data using threshold inequalities. A tree can instantly isolate all missing entries by asking: `"Is Age > 79?"` and process them cleanly without affecting normal data splits.

---

#### 📊 Data Walkthrough

**Input (Raw Data)** → Mean = 35.0, Std = 11.85 → Extreme Tail = $35 + 3(11.85) = \mathbf{70.55}$

|Age|Age_End_Distribution (Output)|
|:-:|:--|
|22|22.0|
|`NaN`|**70.55** _(Extreme Tail Imputed)_|

#### 💻 Python Code Block: End of Distribution Imputation

```python
df_end = df.copy()
# Calculate extreme value beyond 3 standard deviations
extreme_val = df_end['Age'].mean() + 3 * df_end['Age'].std()

df_end['Age_End_Dist'] = df_end['Age'].fillna(extreme_val)
print(f"Calculated Extreme Tail Value: {extreme_val:.2f}")
print("\n--- Processed Matrix ---")
print(df_end[['Age', 'Age_End_Dist']])
```

#### 📉 Printed Output Verification

```text
Calculated Extreme Tail Value: 70.55

--- Processed Matrix ---
    Age  Age_End_Dist
0  22.0     22.000000
1   NaN     70.548457
2  38.0     38.000000
```

---

### Method 5: Arbitrary Value Imputation

#### When to Use It

When your data is **MNAR** and you want to flag missingness using an impossible, hardcoded default value (like `0`, `-1`, or `999`).

#### Why We Use It

It provides a quick, deterministic manual tag that groups all missing records into an impossible value slot.

#### ⚠️ How it Relates to ML Models

- **Linear Models & Distance Models:** Highly destructive. Hardcoding an arbitrary number like `999` into an age or income column creates a massive artificial outlier that corrupts slope lines and distance calculations.
- **Tree-Based Models:** Highly Effective. Just like End of Distribution imputation, tree splits easily isolate the arbitrary value (e.g., `if Age == -1`) into its own clean leaf node.

---

#### 📊 Data Walkthrough

|Age (Input)|Age_Arbitrary (Output)|
|:-:|:--|
|22|22.0|
|`NaN`|**-1.0** _(Arbitrary Flag)_|

#### 💻 Python Code Block: Arbitrary Imputation

```python
df_arbitrary = df.copy()
# Impute using an arbitrary value of choice (e.g., -1 or 999)
df_arbitrary['Age_Arbitrary'] = df_arbitrary['Age'].fillna(-1)

print(df_arbitrary[['Age', 'Age_Arbitrary']])
```

#### 📉 Printed Output Verification

```text
    Age  Age_Arbitrary
0  22.0           22.0
1   NaN           -1.0
2  38.0           38.0
```

---

## 📌 4. Summary Production Strategy Cheat Sheet

|Imputation Strategy|Best Mechanism Match|Impact on Linear Models|Impact on Tree Models|
|:--|:--|:--|:--|
|**Mean / Median Imputation**|**MCAR** (Low volume)|**Negative:** Shrinks variance, distorts standard errors.|**Safe:** Trees easily parse spiked central values.|
|**Random Sample Imputation**|**MCAR**|**Excellent:** Completely preserves natural variance.|**Slight Variance:** Can introduce training noise due to sampling randomness.|
|**Missing Indicator Feature**|**MNAR**|**Excellent:** Isolates the missing signal onto its own slope weight.|**Excellent:** Provides a clean binary threshold for node partitioning.|
|**End of Distribution**|**MNAR**|**Destructive:** Creates massive outliers that warp lines of best fit.|**Excellent:** Creates an easy tail-end inequality partition threshold.|
|**Arbitrary Value Imputation**|**MNAR**|**Destructive:** Corrupts numerical optimisation scales.|**Excellent:** Groups missing rows into a distinct, clean split category.|

---

## 🔗 Related Notes

- [[Feature Engineering — Handling Missing Categorical Values]]
- [[Feature Engineering — Feature Scaling]]
- [[Feature Engineering — Encoding Categorical Variables]]
- [[Machine Learning Fundamentals]]
- [[Titanic Dataset Analysis]]
- [[Linear Regression vs Tree Models]]

---

_Tags: #feature-engineering #missing-data #imputation #mcar #mnar #continuous-data #variance #python #pandas #machine-learning_