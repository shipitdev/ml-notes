# 📊 Feature Engineering: Advanced Categorical Encoding Taxonomy (Day 4)

---

## 📌 1. The Big Picture: Transforming Words without Dimensional Explosion

When dealing with categorical variables, standard **One-Hot Encoding** works well for low-cardinality features, but crashes systems when features contain dozens or hundreds of unique choices (high cardinality).

- **The Day 4 Goal:** Master advanced techniques that convert text strings into dense, predictive numerical features. We focus on techniques that preserve structural relationships, keep our columns locked to a **single feature dimensions grid**, and leverage our **target labels** to engineer maximum predictive accuracy.

---

## 📌 2. Deconstructing the 4 Core Encoding Workflows

---

### Technique 1: Ordinal Number Encoding (Manual Label Assignment)

#### When to Use It

When the text options possess an inherent, undisputed real-world ranking system or sequence (e.g., student grades `A, B, C` → `1, 2, 3`, time sequences like days of the week, or customer feedback loops like `Poor, Satisfactory, Excellent`).

#### Why We Use It

We explicitly build a manual lookup dictionary to map the climbing text options directly onto a ladder of integers. This tells the downstream model the exact chronological or structural sequence of the feature without generating any additional feature columns.

#### ⚠️ How it Relates to ML Models

- **Linear Models:** Assumes a uniform distance between rank steps. It forces the optimisation engine to believe the gap between step 1 → 2 is mathematically identical to the step from 2 → 3.
- **Tree Models:** **Excellent.** Decision tree architectures love rank intervals; they can instantly make a cut split on `Feature_Rank <= 2` to separate structural classes cleanly.
- **Distance Models:** Safe, but distance metrics will be fully dictated by your hardcoded step spacing values.

---

#### 📊 Data Walkthrough

**Input (Raw Chronological Transactions Data)**

|Transaction_ID|Day_of_Week|
|:--|:--|
|101|`Monday`|
|102|`Thursday`|
|103|`Sunday`|

**🔄 Transformation Logic**

Construct a strict, sequence-enforced manual dictionary lookup frame:

```python
weekday_map = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Sunday': 7}
```

**Output (Encoded Array Space)**

|Transaction_ID|Day_Encoded|
|:--|:-:|
|101|**1**|
|102|**4**|
|103|**7**|

---

#### 💻 Python Code Block: Manual Ordinal Sequence Mapping

```python
import pandas as pd

# 1. Initialize mock chronological sequence series
df_ordinal = pd.DataFrame({'Day_of_Week': ['Monday', 'Thursday', 'Sunday', 'Monday']})
print("--- Step 1: Raw Categorical Feature ---")
print(df_ordinal)

# 2. Construct manual ranking index dictionary
weekday_map = {
    'Monday': 1, 'Tuesday': 2, 'Wednesday': 3,
    'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7
}

# 3. Vector map the values instantly
df_ordinal['Day_Encoded'] = df_ordinal['Day_of_Week'].map(weekday_map)
print("\n--- Step 2: Final Processed Ordinal Matrix ---")
print(df_ordinal)
```

#### 📉 Printed Output Verification

```text
--- Step 1: Raw Categorical Feature ---
  Day_of_Week
0      Monday
1    Thursday
2      Sunday
3      Monday

--- Step 2: Final Processed Ordinal Matrix ---
  Day_of_Week  Day_Encoded
0      Monday            1
1    Thursday            4
2      Sunday            7
3      Monday            1
```

---

### Technique 2: Count or Frequency Encoding

#### When to Use It

When dealing with nominal variables (names with no order) that contain a massive number of unique choices (high cardinality, like `Zip Codes`, `Device Models`, or `Country Codes`).

#### Why We Use It

Instead of blowing up your dataset with hundreds of one-hot binary columns, it collapses the feature down into a single numerical column by replacing every country name or zip code with the **exact number of times it appears across your training data**. This captures the popularity or prevalence of each category cleanly.

#### ⚠️ How it Relates to ML Models

- **Linear Models:** Effective at dimension control, but can blur patterns if a high-volume category lacks real predictive correlation to your outcome target.
- **Tree Models:** Efficient. Trees easily use frequency thresholds to split high-prevalence records from rare outliers.
- **Distance Models:** **Vulnerable to Count Collisions.** If two completely unrelated categories (e.g., `India` and `Germany`) happen to appear exactly 100 times in your rows, they both receive an encoded value of `100`. A distance model will view these points as completely identical, permanently erasing their unique characteristics.

---

#### 📊 Data Walkthrough

**Input (Raw High-Cardinality Geographies)**

|User_ID|Country|
|:--|:--|
|0|`USA`|
|1|`Cuba`|
|2|`USA`|
|3|`India`|
|4|`USA`|

**🔄 Transformation Logic**

1. Run frequency value counts: `USA` = 3, `Cuba` = 1, `India` = 1.
2. Form dictionary map: `{'USA': 3, 'Cuba': 1, 'India': 1}`.

**Output (Dimension-Compressed Matrix)**

|User_ID|Country_Encoded|
|:--|:-:|
|0|**3**|
|1|**1**|
|2|**3**|
|3|**1**|
|4|**3**|

---

#### 💻 Python Code Block: Count/Frequency Re-mapping

```python
# Setup raw country list imitating census records
df_freq = pd.DataFrame({'Country': ['USA', 'Cuba', 'USA', 'India', 'USA']})

# 1. Compute value frequencies and cast to dictionary lookup
country_counts_map = df_freq['Country'].value_counts().to_dict()
print(f"Calculated Frequency Lookup Map: {country_counts_map}")

# 2. Map frequencies back onto the raw feature space
df_freq['Country_Encoded'] = df_freq['Country'].map(country_counts_map)
print("\n--- After Count Frequency Encoding Matrix ---")
print(df_freq)
```

#### 📉 Printed Output Verification

```text
Calculated Frequency Lookup Map: {'USA': 3, 'Cuba': 1, 'India': 1}

--- After Count Frequency Encoding Matrix ---
  Country  Country_Encoded
0     USA                3
1    Cuba                1
2     USA                3
3   India                1
4     USA                3
```

---

### Technique 3: Target-Guided Ordinal Encoding

#### When to Use It

When handling high-cardinality nominal blocks where standard mapping fails, and you want to ensure your category transformations align perfectly with the target outcome.

#### Why We Use It

It links your categorical column values directly to the target variable. It calculates the mean survival rate (or target average) for each category, sorts those categories from lowest to highest mean, and assigns an ascending ordinal integer rank (0, 1, 2, 3). This guarantees a clean **monotonic relationship** between your text values and your target goal.

#### ⚠️ How it Relates to ML Models

- **Linear Models:** **Exceptional.** By organising the category integers to follow the target's mean curve, it creates a clean linear trend line, significantly improving slope calculation accuracy.
- **Tree Models:** Highly optimised split nodes; cuts boundaries based on target density.
- **Risk Across All Fields:** High danger of **Data Leakage and Overfitting.** Because you use the target variable to engineer the training feature, any rare categories can overfit heavily, causing the model to memorise the training set instead of generalising to unseen data.

---

#### 📊 Data Walkthrough

**Input (Raw Titanic Deck Blocks + Binary Target Survival Output)**

|Passenger_ID|Cabin_Block|Survived (Target)|
|:--|:--|:-:|
|1|`M` (Missing)|0|
|2|`C`|1|
|3|`M` (Missing)|0|
|4|`E`|1|
|5|`M` (Missing)|1|

**🔄 Transformation Logic & Hypothesis Proof**

**Step 1:** Compute target group means per category block:

$$\text{Mean for M} = \frac{0 + 0 + 1}{3} = \mathbf{0.33} \qquad \text{Mean for C} = \frac{1}{1} = \mathbf{1.00} \qquad \text{Mean for E} = \frac{1}{1} = \mathbf{1.00}$$

**Step 2:** Sort ascending by target mean value: `['M', 'C', 'E']`

**Step 3:** Enumerate array sequence to assign climbing index weights: `{'M': 0, 'C': 1, 'E': 2}`

**Output (Target-Ranked Vector Matrix)**

|Passenger_ID|Cabin_Block_Target_Ranked|Survived (Target)|
|:--|:-:|:-:|
|1|**0**|0|
|2|**1**|1|
|3|**0**|0|
|4|**2**|1|
|5|**0**|1|

---

#### 💻 Python Code Block: Target-Guided Rank Pipeline

```python
# Create raw training frame linking cabin blocks to survival outcomes
df_target = pd.DataFrame({
    'Cabin': ['M', 'C', 'M', 'E', 'M'],
    'Survived': [0, 1, 0, 1, 1]
})

print("--- Proving Group Target Means ---")
print(df_target.groupby(['Cabin'])['Survived'].mean())

# 1. Group categories by target mean performance and extract the sorted index
ordered_labels = df_target.groupby(['Cabin'])['Survived'].mean().sort_values().index
print(f"\nSorted Monotonic Labels Index: {list(ordered_labels)}")

# 2. Build ordinal rank integer map via dictionary comprehension
target_ordinal_map = {k: i for i, k in enumerate(ordered_labels, 0)}
print(f"Generated Target Guided Map: {target_ordinal_map}")

# 3. Map values back onto original feature grid
df_target['Cabin_Guided_Rank'] = df_target['Cabin'].map(target_ordinal_map)
print("\n--- After Target-Guided Ordinal Encoding Matrix ---")
print(df_target)
```

#### 📉 Printed Output Verification

```text
--- Proving Group Target Means ---
Cabin
C    1.000000
E    1.000000
M    0.333333
Name: Survived, dtype: float64

Sorted Monotonic Labels Index: ['M', 'C', 'E']
Generated Target Guided Map: {'M': 0, 'C': 1, 'E': 2}

--- After Target-Guided Ordinal Encoding Matrix ---
  Cabin  Survived  Cabin_Guided_Rank
0     M         0                  0
1     C         1                  1
2     M         0                  0
3     E         1                  2
4     M         1                  0
```

---

### Technique 4: Mean Encoding (Target Encoding)

#### When to Use It

Similar to Target-Guided Ordinal Encoding, use this when you want to capture target insights within a single high-cardinality column, but prefer using the **exact target mean floats** directly rather than converting them into integer ranks (0, 1, 2).

#### Why We Use It

It replaces each text category name directly with its raw target mean calculation percentage or decimal. This packs a huge amount of predictive information into a single feature without altering your dataset's shape.

#### ⚠️ How it Relates to ML Models

- **Linear Models:** Highly linear. It fits perfectly into linear formulations because the feature scale directly mirrors the target probability trend.
- **Tree Models:** Extremely effective at isolating variance splits.
- **The Core Vulnerability:** **Extreme Data Leakage Risk.** If a category appears only once or twice in your rows, its mean encoded value will perfectly match the target column for those specific rows. This allows your model to easily memorise the training labels, leading to severe overfitting.

---

#### 📊 Data Walkthrough

**Input (Raw Features + Dependent Binary Target)**

|Passenger_ID|Cabin_Block|Survived (Target)|
|:--|:--|:-:|
|1|`M`|0|
|2|`C`|1|
|3|`M`|0|
|4|`E`|1|
|5|`M`|1|

**🔄 Transformation Logic**

Calculate each group's raw target mean decimal: `M = 0.33`, `C = 1.00`, `E = 1.00`. Replace the category words directly with these floats.

**Output (Mean Encoded Matrix Space)**

|Passenger_ID|Cabin_Mean_Encoded|Survived (Target)|
|:--|:-:|:-:|
|1|**0.33**|0|
|2|**1.00**|1|
|3|**0.33**|0|
|4|**1.00**|1|
|5|**0.33**|1|

---

#### 💻 Python Code Block: Direct Mean Encoding

```python
df_mean = df_target.copy()

# 1. Map each category directly to its calculated target mean float dictionary
mean_encoder_map = df_mean.groupby(['Cabin'])['Survived'].mean().to_dict()
print(f"Calculated Mean Target Map: {mean_encoder_map}")

# 2. Swap the category words for the raw mean values
df_mean['Cabin_Mean_Encoded'] = df_mean['Cabin'].map(mean_encoder_map)
print("\n--- After Direct Mean Target Encoding Matrix ---")
print(df_mean[['Cabin', 'Survived', 'Cabin_Mean_Encoded']])
```

#### 📉 Printed Output Verification

```text
Calculated Mean Target Map: {'C': 1.0, 'E': 1.0, 'M': 0.3333333333333333}

--- After Direct Mean Target Encoding Matrix ---
  Cabin  Survived  Cabin_Mean_Encoded
0     M         0            0.333333
1     C         1            1.000000
2     M         0            0.333333
3     E         1            1.000000
4     M         1            0.333333
```

---

## 📌 3. Summary Production Strategy Cheat Sheet Matrix

|Encoding Strategy Vector|Best Cardinality Match|Impact on Linear Models|Impact on Distance Models|Core Engineering Threat|
|:--|:--|:--|:--|:--|
|**Ordinal Number Encoding**|Low to Medium Ordinal scales|**Moderate:** Assumes spacing intervals are uniform.|Safe, but fully reliant on step values.|Distorts non-linear categorical trends.|
|**Count / Frequency Encoding**|High Cardinality Nominal fields|**Safe:** Collapses wide dimension grids efficiently.|**Poor:** Vulnerable to category count collisions.|Blurs unique class differences if row frequencies match.|
|**Target-Guided Ordinal**|High Cardinality Nominal fields|**Excellent:** Enforces a clean, monotonic trend alignment.|Good profile grouping.|**High Data Leakage Danger:** Prone to overfitting.|
|**Mean / Target Encoding**|High Cardinality Nominal fields|**Excellent:** Directly copies target probability trends.|Good profile grouping.|**High Data Leakage Danger:** Prone to overfitting on small sample sizes.|

---

## 🔗 Related Notes

- [[Feature Engineering — Categorical Imputation & Advanced Encoding Taxonomy (Day 3)]]
- [[Feature Engineering — Encoding Categorical Variables]]
- [[Feature Engineering — Missing Data Mechanisms & Continuous Imputation (Day 1)]]
- [[Feature Engineering — Advanced Continuous Imputation (Day 2)]]
- [[Feature Engineering — Feature Scaling]]
- [[Dummy Variable Trap & Multicollinearity]]
- [[Overfitting & Data Leakage]]

---

_Tags: #feature-engineering #categorical-encoding #ordinal-encoding #frequency-encoding #target-encoding #mean-encoding #high-cardinality #data-leakage #overfitting #titanic #python #pandas #machine-learning_