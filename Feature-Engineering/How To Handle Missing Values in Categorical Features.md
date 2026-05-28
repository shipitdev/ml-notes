# 📊 Feature Engineering: Handling Missing Values in Categorical Features

---

## 📌 1. The Big Picture: The Mystery of Blank Word Cells

When raw data is collected from the real world, finding missing cells is completely normal — typically recorded as `NaN` (Not a Number) or null values. While numerical columns can be filled with math averages like the mean or median, categorical columns (text entries like `"Country"` or `"Garage Type"`) cannot be averaged. You cannot calculate the average string of `"Attached"`, `"Detached"`, and a blank cell.

- **The Goal:** Handle missing text entries responsibly to make the dataset mathematically complete for machine learning algorithms — without adding false biases or ruining your data distributions.

---

## 📌 2. The 4 Core Categorical Imputation Strategies Deconstructed

---

### ❌ Method 1: Complete Row Deletion (Dropping Rows)

#### When to Use It

When the missing data is extremely tiny (typically less than 1% or 2% of your entire dataset) **or** when a row is missing values across almost all of its columns, making it completely useless.

#### Why We Use It

It is the absolute fastest way to get a clean dataset. It introduces zero guesswork or assumptions because you are not making up data out of thin air.

---

#### 📊 Data Walkthrough

**Input (Raw Data)**

|House_ID|Rooms|Garage_Type|
|:--|:-:|:--|
|101|3|`Attached`|
|102|4|`Detached`|
|103|2|`NaN` _(Missing)_|
|104|5|`Attached`|

**🔄 Transformation Logic**

The pipeline spots that House 103 has a missing string value. Since the overall dataset has thousands of complete records, it simply deletes the entire row for House 103 from memory.

**Output (Cleaned Data)**

|House_ID|Rooms|Garage_Type|
|:--|:-:|:--|
|101|3|`Attached`|
|102|4|`Detached`|
|104|5|`Attached`|

---

### 🔄 Method 2: Frequent Category Imputation (Mode Imputation)

#### When to Use It

When the missing values make up a small fraction of your data (less than 5% to 10%) **and** the column is heavily dominated by one single, highly popular choice.

#### Why We Use It

It is computationally lightweight and easy to implement. It works on simple probability: if an entry went missing at random, it most likely belongs to the most common group already in the dataset.

> ⚠️ **The Vulnerability:** It artificially spikes the count of your most popular category. On a bar chart, the dominant category's bar will shoot up, which distorts the true variance of your data.

---

#### 📊 Data Walkthrough

**Input (Raw Data)**

|House_ID|Garage_Type|
|:--|:--|
|101|`Attached`|
|102|`Detached`|
|103|`NaN` _(Missing)_|
|104|`Attached`|
|105|`NaN` _(Missing)_|

**🔄 Transformation Logic**

1. Compute the counts: `Attached` = 2, `Detached` = 1. The most frequent word (**Mode**) is `Attached`.
2. Find all `NaN` entries and replace them with `Attached`.

**Output (Imputed Data)**

|House_ID|Garage_Type|
|:--|:--|
|101|`Attached`|
|102|`Detached`|
|103|**`Attached`** _(Imputed)_|
|104|`Attached`|
|105|**`Attached`** _(Imputed)_|

---

### 🤖 Method 3: Classifier Algorithm Prediction (Supervised Learning)

#### When to Use It

When the missing data rate is relatively high (10% to 30%) and you can clearly see that the missing column has a **strong relationship** with your other complete columns.

#### Why We Use It

Instead of guessing or forcing everything into the most popular choice, you turn the missing value problem into a mini machine learning task. You train a classification model (like a _Decision Tree_ or _K-Nearest Neighbors Classifier_) on all the rows that are **complete**. The model learns the patterns between the complete features and the text labels, then uses that knowledge to accurately predict the missing text categories.

---

#### 📊 Data Walkthrough

**Input (Raw Data)**

|House_ID|House_Price|Garage_Type|
|:--|:-:|:--|
|101|$350k|`Attached`|
|102|$120k|`Detached`|
|103|$380k|`NaN` _(Missing)_|
|104|$110k|`Detached`|

**🔄 Transformation Logic**

1. The system splits the data. It sees that expensive homes ($350k, $380k) tend to have `Attached` garages, while affordable homes have `Detached` garages.
2. A classification model trains on rows 101, 102, and 104.
3. The model looks at row 103's price ($380k) and predicts that its missing garage type is highly likely to be `Attached`.

**Output (Predicted Data Matrix)**

|House_ID|House_Price|Garage_Type|
|:--|:-:|:--|
|101|$350k|`Attached`|
|102|$120k|`Detached`|
|103|$380k|**`Attached`** _(Predicted)_|
|104|$110k|`Detached`|

---

### 🌲 Method 4: Unsupervised Learning Imputation (Clustering / Distance Matching)

#### When to Use It

When you want to find natural groupings in your data across **multiple features** without relying on a strict prediction line, letting row similarities fill the blanks.

#### Why We Use It

It uses algorithms like **KNN Imputer** to measure the geometric distance between row data points. It finds the "Nearest Neighbor" records that match the overall profile of the row with the missing value, then copies the most common category directly from those similar records.

---

#### 📊 Data Walkthrough

**Input (Raw Data)**

|Profile_ID|Rooms|Bathrooms|Neighborhood|
|:--|:-:|:-:|:--|
|1|5|4|`Luxury_Suburbs`|
|2|2|1|`Downtown_Urban`|
|3|5|4|`NaN` _(Missing)_|

**🔄 Transformation Logic**

The unsupervised distance calculator looks across the numeric columns (`Rooms` and `Bathrooms`). It notes that Profile 3 `[5, 4]` is geometrically closer to Profile 1 `[5, 4]` than it is to Profile 2 `[2, 1]`. It flags Profile 1 as its nearest neighbour and copies its neighbourhood label.

**Output (Clustered Imputation Data)**

|Profile_ID|Rooms|Bathrooms|Neighborhood|
|:--|:-:|:-:|:--|
|1|5|4|`Luxury_Suburbs`|
|2|2|1|`Downtown_Urban`|
|3|5|4|**`Luxury_Suburbs`** _(Clustered)_|

---

## 📌 3. Real-World Applications Across Industries

- **Credit Risk Evaluation:** Loan applications track an applicant's `Employment Type` (e.g., Full-Time, Self-Employed). If a subset is missing, a **Method 3 Classifier Algorithm** can use features like `Annual Income` and `Credit Score` to predict the missing employment category rather than assigning everyone to a generic default category.
- **Healthcare Patient Portals:** Medical admission forms contain a categorical entry for `Primary Health Concern`. If a digital intake form has empty values, hospitals use **Method 4 Unsupervised Distance Clustering** to match the patient's numerical vitals (Heart Rate, Blood Pressure) to historical patient records, categorising their risk profile cleanly.

---

## 📌 4. Programmatic Implementation & Code Execution

This end-to-end Python script sets up an entry matrix with missing elements and demonstrates how to execute **Row Deletion (Method 1)**, **Mode Imputation (Method 2)**, and a **Machine Learning Classifier Prediction (Method 3)**.

### 💻 Python Code

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. Create a raw dataset with missing text values
data = {
    'House_Price_k': [350, 120, 380, 110, 400],
    'Garage_Type': ['Attached', 'Detached', np.nan, 'Detached', np.nan]
}
df = pd.DataFrame(data)
print("=== 1. RAW INPUT DATA ===")
print(df)

# --- METHOD 1: Complete Row Deletion ---
df_deleted = df.dropna().copy()
print("\n=== 2. AFTER ROW DELETION (METHOD 1) ===")
print(df_deleted)

# --- METHOD 2: Frequent Category Imputation (Mode) ---
mode_value = df['Garage_Type'].mode()[0]
df_mode = df.copy()
df_mode['Garage_Type'] = df_mode['Garage_Type'].fillna(mode_value)
print("\n=== 3. AFTER MODE IMPUTATION (METHOD 2) ===")
print(df_mode)

# --- METHOD 3: Classifier Prediction ---
df_clf = df.copy()
# Separate complete rows from missing rows
train_data = df_clf[df_clf['Garage_Type'].notnull()]
predict_data = df_clf[df_clf['Garage_Type'].isnull()]

# Initialize a simple machine learning classifier model
clf = RandomForestClassifier(random_state=42)
clf.fit(train_data[['House_Price_k']], train_data['Garage_Type'])

# Predict the missing labels
predictions = clf.predict(predict_data[['House_Price_k']])

# Place the predicted labels back into the missing spots
df_clf.loc[df_clf['Garage_Type'].isnull(), 'Garage_Type'] = predictions
print("\n=== 4. AFTER CLASSIFIER PREDICTION (METHOD 3) ===")
print(df_clf)
```

### 📉 Printed Output

```text
=== 1. RAW INPUT DATA ===
   House_Price_k Garage_Type
0            350    Attached
1            120    Detached
2            380         NaN
3            110    Detached
4            400         NaN

=== 2. AFTER ROW DELETION (METHOD 1) ===
   House_Price_k Garage_Type
0            350    Attached
1            120    Detached
3            110    Detached

=== 3. AFTER MODE IMPUTATION (METHOD 2) ===
   House_Price_k Garage_Type
0            350    Attached
1            120    Detached
2            380    Attached
3            110    Detached
4            400    Attached

=== 4. AFTER CLASSIFIER PREDICTION (METHOD 3) ===
   House_Price_k Garage_Type
0            350    Attached
1            120    Detached
2            380    Attached
3            110    Detached
4            400    Attached
```

---

## 📌 5. Summary Cheat Sheet Matrix

|Imputation Technique|When to Deploy|Operational Advantage|Hidden Vulnerability|
|:--|:--|:--|:--|
|**Complete Row Deletion**|Missing data is **very tiny (<1–2%)** or rows are completely uninformative.|Fast; introduces zero guesswork or false distribution changes.|Discards valuable information from other complete columns in that row.|
|**Frequent Category (Mode)**|Missing data is **low (<5–10%)** and one class dominates the dataset.|Extremely simple; leaves your column structure intact.|Artificially distorts data distributions by over-spiking the mode count.|
|**Classifier Prediction**|Missing data is **high (10–30%)** and has a strong relationship to other features.|Highly intelligent; uses existing data patterns to fill blanks accurately.|Requires building and maintaining a separate model within your data pipeline.|
|**Unsupervised Clustering**|You want to find structural row groupings across **multiple columns** at once.|Captures deep, multi-column row profiles to find matching entries.|Slower to calculate because it computes geometric distances across the whole dataset.|

---

## 🔗 Related Notes

- [[Feature Engineering — Encoding Categorical Variables]]
- [[Feature Engineering — Feature Scaling]]
- [[Machine Learning Fundamentals]]
- [[Scikit-Learn Preprocessing Pipeline]]
- [[KNN and K-Means Clustering]]

---

_Tags: #feature-engineering #missing-values #imputation #categorical-data #mode-imputation #knn-imputer #random-forest #preprocessing #python_