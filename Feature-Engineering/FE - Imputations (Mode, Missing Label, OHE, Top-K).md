# 📊 Feature Engineering: Categorical Imputation & Advanced Encoding Taxonomy (Day 3)

---

## 📌 1. The Big Picture: Words, Blanks, and Matrix Constraints

In our previous sessions, we focused primarily on continuous numeric features. Today, we cross the threshold into **Categorical Data Engineering**. Categorical variables hold labels or words rather than direct numerical measurements. Handling missing text elements and preparing valid numeric data blocks requires separate mathematical paradigms.

- **The Categorical Missingness Challenge:** You cannot calculate a column mean or mathematical standard deviation on string labels like `"Attached"`, `"Detached"`, or `"Unknown"`. We must look to frequency metrics or structural text grouping vectors.
- **The Encoding Challenge:** Machine learning models are linear/non-linear algebra optimisation calculators. Passing text words directly into an array triggers code execution crashes. We must use targeted nominal and ordinal mapping schemas to turn words into numbers without introducing mathematical anomalies.

---

## 📌 2. Part 1: Handling Missing Values in Categorical Features

When categorical attributes (e.g., `Garage_Type`, `Fireplace_Quality`) contain empty `NaN` cells, we deploy two primary workflows depending on how heavily the missing observations populate the variable.

---

### Technique A: Frequent Category Imputation (Mode Imputation)

#### When to Use It

When the missing records populate a small fraction of the column (typically under 5% to 10%) **and** the feature space is heavily dominated by one single, highly popular text choice.

#### Why We Use It

It is computationally fast and does not expand data dimensions. It runs on a basic probability assumption: if data went missing at random, it most likely belongs to the most common category already observed.

#### ⚠️ How it Relates to ML Models

- **Linear Models:** Can mask variance signals. Artificially spiking the count of your most popular class inflates its significance, skewing the intercept or default baseline category estimation.
- **Distance Models:** Safe for low volume, but clusters may compress heavily towards the dominant category coordinate space.
- **Tree Models:** Highly stable. Tree splits can isolate or pool this category effortlessly.

---

#### 📊 Data Walkthrough

**Input (Raw Ames Housing Data Subset)**

|House_ID|Bsmt_Quality|
|:--|:--|
|101|`Gd` (Good)|
|102|`TA` (Typical)|
|103|`NaN` _(Missing)_|
|104|`Gd` (Good)|

**🔄 Transformation Logic & Hypothesis Proof**

1. Run frequency check: `Gd` appears 2 times, `TA` appears 1 time. The **Mode** is **`Gd`**.
2. Replace all occurrences of `NaN` directly with the string label `Gd`.

**Output (Imputed Matrix)**

|House_ID|Bsmt_Quality_Cleaned|
|:--|:--|
|101|`Gd`|
|102|`TA`|
|103|**`Gd`** _(Imputed)_|
|104|`Gd`|

---

#### 💻 Python Code Block: Phase 1 (Mode Imputation)

```python
import numpy as np
import pandas as pd

# Create a mock dataset mimicking housing data
data = {'Bsmt_Quality': ['Gd', 'TA', np.nan, 'Gd', np.nan, 'Gd', 'TA']}
df = pd.DataFrame(data)

print("--- Step 1: Proving Class Popularity (Value Counts) ---")
print(df['Bsmt_Quality'].value_counts())

# Calculate the mode value safely
mode_label = df['Bsmt_Quality'].mode()[0]
print(f"\nCalculated Column Mode: '{mode_label}'")

# Execute Frequent Category Imputation
df['Bsmt_Quality_Mode'] = df['Bsmt_Quality'].fillna(mode_label)
print("\n--- Step 2: Final Imputed Categorical Matrix ---")
print(df)
```

#### 📉 Printed Output Verification

```text
--- Step 1: Proving Class Popularity (Value Counts) ---
Gd    3
TA    2
Name: Bsmt_Quality, dtype: int64

Calculated Column Mode: 'Gd'

--- Step 2: Final Imputed Categorical Matrix ---
  Bsmt_Quality Bsmt_Quality_Mode
0           Gd                Gd
1           TA                TA
2          NaN                Gd
3           Gd                Gd
4          NaN                Gd
5           Gd                Gd
6           TA                TA
```

---

### Technique B: Replacing NaN with a New Label (Missing / New Category Imputation)

#### When to Use It

When the volume of missing data is high (e.g., over 10% to 20%) **or** when the fact that information is blank holds its own predictive value (MNAR profile). For instance, a missing text value for `Fireplace_Quality` usually just means the house doesn't have a fireplace.

#### Why We Use It

Instead of forcing empty entries into a fake category that distorts the real distribution ratios, this method replaces `NaN` cells with a fresh text category name like **`"Missing"`** or **`"Unknown"`**. This completely preserves the text boundaries and turns missingness into an active signal.

#### ⚠️ How it Relates to ML Models

- **Linear Models:** Excellent. The model calculates a clean independent slope parameter weight for the `"Missing"` category, logging it as a unique baseline condition.
- **Tree Models:** Exceptional. Decision trees love distinct partitions. The tree can instantly split a node on `Feature == "Missing"` to isolate those records down their own path.

---

#### 📊 Data Walkthrough

**Input (Raw Features with Missing Fireplace data)**

|House_ID|Fireplace_Quality|
|:--|:--|
|201|`Ex` (Excellent)|
|202|`NaN` _(No Fireplace)_|
|203|`TA` (Typical)|

**🔄 Transformation Logic**

Overwrite every instance of `NaN` with the explicit descriptive placeholder word `"Missing"`.

**Output (Transformed Feature Space)**

|House_ID|Fireplace_Quality_Cleaned|
|:--|:--|
|201|`Ex`|
|202|**`Missing`** _(New Category Flag)_|
|203|`TA`|

---

#### 💻 Python Code Block: Phase 2 (New Category Imputation)

```python
# Reusing our dataset to demonstrate New Category Imputation
df_new_cat = df.copy()

# Replace all null elements with the custom label 'Missing'
df_new_cat['Bsmt_Quality_MissingLabel'] = df_new_cat['Bsmt_Quality'].fillna('Missing')

print("--- After Replacing NaN with an Independent Category Label ---")
print(df_new_cat[['Bsmt_Quality', 'Bsmt_Quality_MissingLabel']])
```

#### 📉 Printed Output Verification

```text
--- After Replacing NaN with an Independent Category Label ---
  Bsmt_Quality Bsmt_Quality_MissingLabel
0           Gd                        Gd
1           TA                        TA
2          NaN                   Missing
3           Gd                        Gd
4          NaN                   Missing
5           Gd                        Gd
6           TA                        TA
```

---

## 📌 3. Part 2: Categorical Feature Encoding Techniques

Once your categorical features are fully imputed and hold no null entries, they must be translated into clean numerical arrays before being passed to a model.

---

### Technique A: Standard One-Hot Encoding (Low Cardinality)

#### When to Use It

When a categorical nominal column contains a small handful of unique choices (low cardinality, typically under 10 options) like `Sex` (Male, Female) or `Embarked` (S, C, Q) from the Titanic dataset.

#### Why We Use It

It builds an individual binary vector column for each unique choice. To eliminate perfect linear dependency amongst features, we strictly pass the parameter **`drop_first=True`**. This drops the first column to yield exactly M−1 features, completely bypassing the **Dummy Variable Trap**.

#### ⚠️ How it Relates to ML Models

- **Linear Models (Crucial Rule):** You **must** drop the first column (`drop_first=True`). If you don't, the columns become perfectly predictable from each other (multicollinearity). This breaks matrix math operations, meaning your linear regression equations cannot resolve a stable line of best fit.
- **Distance Models:** Safe, but keep an eye on column counts. Each new binary column adds a fresh axis, expanding dimensions.
- **Tree Models:** Suboptimal if cardinality climbs. Split criteria struggle to isolate variance when features are splintered into hundreds of sparse binary columns.

---

#### 📊 Data Walkthrough

**Input (Raw Nominal Vector)**

|Passenger_ID|Embarked|
|:--|:--|
|1|`S`|
|2|`C`|
|3|`Q`|
|4|`S`|

**🔄 Transformation Logic (M−1 System Drop Rules)**

We have 3 classes: `C`, `Q`, `S`. We select `C` as our baseline anchor column and drop it. We create binary status indicators for the remaining features: `Embarked_Q` and `Embarked_S`. If a row was originally `C`, it scales to `[0, 0]` natively.

**Output (One-Hot Encoded Data Block)**

|Passenger_ID|Embarked_Q|Embarked_S|
|:--|:-:|:-:|
|1|0|**1**|
|2|0|0|
|3|**1**|0|
|4|0|**1**|

---

#### 💻 Python Code Block: Phase 3 (Standard One-Hot Encoding Execution)

```python
# Setup low-cardinality nominal dataset
titanic_data = {'Passenger_ID': [1, 2, 3, 4], 'Embarked': ['S', 'C', 'Q', 'S']}
df_titanic = pd.DataFrame(titanic_data)

print("--- Raw Categorical Feature Block ---")
print(df_titanic)

# Perform Standard OHE dropping the first class flag to bypass the dummy variable trap
df_ohe = pd.get_dummies(df_titanic, columns=['Embarked'], drop_first=True, dtype=int)
print("\n--- Processed M-1 Dummy Variable Matrix Space ---")
print(df_ohe)
```

#### 📉 Printed Output Verification

```text
--- Raw Categorical Feature Block ---
   Passenger_ID Embarked
0             1        S
1             2        C
2             3        Q
3             4        S

--- Processed M-1 Dummy Variable Matrix Space ---
   Passenger_ID  Embarked_Q  Embarked_S
0             1           0           1
1             2           0           0
2             3           1           0
3             4           0           1
```

---

### Technique B: One-Hot Encoding with Many Categories (Top-K High Cardinality)

#### When to Use It

When a nominal feature contains dozens or hundreds of unique classes (high cardinality, like the Mercedes-Benz dataset where feature `X1` has over 50 unique engineering part codes).

#### Why We Use It

Standard One-Hot Encoding on high-cardinality features causes a dangerous column explosion. This creates an extremely wide, mostly empty dataset (matrix sparsity) that triggers the **Curse of Dimensionality** and slows down your system. Popularised by the winning ensemble selectors of the **KDD Cup Orange Challenge**, this technique calculates column frequencies, extracts **only the top K most frequent classes** (usually K=10), and builds dummy columns exclusively for them. Any rare, long-tail entries receive a row profile of all zeros `[0, 0, 0...]` to filter out background noise.

#### ⚠️ How it Relates to ML Models

- **Linear Models:** Highly Recommended for high cardinality. Enforces a clean, manual geometric cap on feature dimensions, keeping your optimisation equations lightweight and fast.
- **Distance Models:** Prevents distance metrics from being corrupted by hundreds of rare, uninformative binary dimensions.
- **Tree Models:** Excellent focus. Allows decision tree architectures to isolate primary data categories easily without getting bogged down by a long tail of rare outliers.

---

#### 📊 Data Walkthrough

**Input (Raw High-Cardinality Custom Part Codes − Limit K=2)**

|Vehicle_ID|Part_Code|
|:--|:--|
|0|`at`|
|1|`av`|
|2|`at`|
|3|`n` _(Rare Part)_|
|4|`at`|
|5|`av`|

**🔄 Transformation Logic**

1. Extract counts: `at` = 3, `av` = 2, `n` = 1.
2. Select the top 2 labels: `['at', 'av']`.
3. Create binary columns _only_ for those top 2 options. The rare option `n` naturally gets assigned `0` across both columns.

**Output (Dimension-Controlled Matrix)**

|Vehicle_ID|Part_Code_at|Part_Code_av|
|:--|:-:|:-:|
|0|**1**|0|
|1|0|**1**|
|2|**1**|0|
|3|0|0|
|4|**1**|0|
|5|0|**1**|

---

#### 💻 Python Code Block: Phase 4 (Top-K High-Cardinality Multi-Category Encoding)

```python
# Create mock high-cardinality dataset matching the Mercedes-Benz project layout
mercedes_data = {'Part_Code': ['at', 'av', 'at', 'n', 'at', 'av', 'n', 'at', 'av', 'z']}
df_mercedes = pd.DataFrame(mercedes_data)

print("--- Raw High-Cardinality Parts Field ---")
print(df_mercedes['Part_Code'].value_counts())

# Set our Top-K limit parameter to the top 2 most frequent items
K = 2
top_k_labels = df_mercedes['Part_Code'].value_counts().head(K).index.tolist()
print(f"\nTop-{K} Most Frequent Categories Isolated: {top_k_labels}")

# Map binary status indicators for our top isolated categories exclusively
for label in top_k_labels:
    df_mercedes[f'Part_{label}'] = np.where(df_mercedes['Part_Code'] == label, 1, 0)

print("\n--- Final Dimension-Capped Preprocessed Feature Matrix ---")
print(df_mercedes)
```

#### 📉 Printed Output Verification

```text
--- Raw High-Cardinality Parts Field ---
at    4
av    3
n     2
z     1
Name: Part_Code, dtype: int64

Top-2 Most Frequent Categories Isolated: ['at', 'av']

--- Final Dimension-Capped Preprocessed Feature Matrix ---
  Part_Code  Part_at  Part_av
0        at        1        0
1        av        0        1
2        at        1        0
3         n        0        0
4        at        1        0
5        av        0        1
6         n        0        0
7        at        1        0
8        av        0        1
9         z        0        0
```

---

## 📌 4. Summary Production Strategy Cheat Sheet Matrix

|Strategy Vector|Category Family|Best When...|Primary Architectural Impact|
|:--|:--|:--|:--|
|**Frequent Category (Mode)**|Imputation|Missing cells are **< 5–10%** and one choice clearly dominates.|Fast, keeps column structure, but skews data ratios if overused.|
|**Missing Label Flag**|Imputation|Missing cells are **substantial (> 10%)** or carry hidden predictive value.|Keeps your data distributions intact; converts missingness into a distinct category.|
|**Standard One-Hot (OHE)**|Encoding|Column cardinality is **low (< 10–15 unique options)**.|Simple, but requires `drop_first=True` to prevent the dummy variable trap in linear models.|
|**Top-K Selection Encoding**|Encoding|Column cardinality is **extremely high** (Dozens or hundreds of options).|Enforces a hard structural cap on dimensionality, protecting models from data sparsity.|

---

## 🔗 Related Notes

- [[Feature Engineering — Handling Missing Categorical Values]]
- [[Feature Engineering — Encoding Categorical Variables]]
- [[Feature Engineering — Missing Data Mechanisms & Continuous Imputation (Day 1)]]
- [[Feature Engineering — Advanced Continuous Imputation (Day 2)]]
- [[Feature Engineering — Feature Scaling]]
- [[Machine Learning Fundamentals]]
- [[Dummy Variable Trap & Multicollinearity]]

---

## 📹 Recommended Video Resource

For the live, full-length dataset demonstrations using the Ames Housing and Mercedes-Benz manufacturing records, review [Krish Naik's Feature Engineering Live Stream — Day 3](https://www.youtube.com/watch?v=dmt2R4A3W1Q).

---

_Tags: #feature-engineering #categorical-imputation #one-hot-encoding #high-cardinality #mode-imputation #missing-labels #dummy-variable-trap #ames-housing #mercedes-benz #python #pandas #machine-learning_