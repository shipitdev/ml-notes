# 📊 EDA Visualisation Playbook: Advanced Housing Dataset (Ames, Iowa)

---

## ⚡ TL;DR

EDA visualisation is not about making pretty charts — it is a structured diagnostic process. Every chart answers a specific question about your data's structure, distribution, relationships, or anomalies. This playbook maps each chart type to its exact purpose, the parameters to inspect, and the patterns you might find — so you always know what you are looking for before you look.

---

## 📌 1. The Big Picture: The 4 Questions Every EDA Answers

Before plotting anything, anchor every chart to one of these four diagnostic questions:

1. **How is this single feature distributed?** → Univariate plots
2. **How do two features relate to each other?** → Bivariate plots
3. **How do multiple features relate simultaneously?** → Multivariate plots
4. **Are there anomalies, outliers, or missing data patterns?** → Diagnostic plots

---

## 📌 2. Phase 1: Understanding Your Target Variable (SalePrice)

The target variable `SalePrice` is the most important column in the entire dataset. Before touching any feature, understand its distribution completely.

---

### Chart 1: Histogram + KDE of SalePrice

**How to plot it:**

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(df['SalePrice'], kde=True, bins=50)
plt.title('SalePrice Distribution')
plt.xlabel('Sale Price ($)')
plt.show()
```

**Purpose:** Understand the shape of your target variable before building any model.

**Parameters to inspect:**

- The peak (mode) of the distribution — where do most house prices cluster?
- The tail — does it trail off to the right (right-skewed)?
- The width — is there a wide or narrow spread of prices?

**Patterns you might find:**

| Pattern                             | What it Means                                                                           | Action                                             |
| :---------------------------------- | :-------------------------------------------------------------------------------------- | :------------------------------------------------- |
| **Right-skewed tail** (most common) | Most houses sell in the $100k–$250k range but luxury estates stretch the axis far right | Apply `np.log1p()` transformation before modelling |
| **Roughly bell-shaped**             | Prices are normally distributed — ideal for linear models                               | No transformation needed                           |
| **Bimodal (two peaks)**             | Two distinct market segments exist in the data (e.g., low-income vs. luxury)            | Consider segmenting or using a non-linear model    |

**What to note:** SalePrice in the Ames dataset is almost always right-skewed. After applying `np.log1p()`, the distribution should resemble a bell curve — verify this by plotting again.

---

### Chart 2: Q-Q Plot of SalePrice

**How to plot it:**

```python
import scipy.stats as stats

stats.probplot(df['SalePrice'], dist="norm", plot=plt)
plt.title('Q-Q Plot: SalePrice vs Normal Distribution')
plt.show()
```

**Purpose:** Mathematically verify whether SalePrice follows a normal distribution — a key assumption for linear regression.

**Parameters to inspect:**

- Do the data points follow the red diagonal reference line?
- Where do the points bend away from the line?

**Patterns you might find:**

|Pattern|What it Means|
|:--|:--|
|**Points hug the diagonal line**|SalePrice is normally distributed — proceed confidently|
|**Points curve upward at the right tail**|Right-skewed distribution — apply log transformation|
|**S-shaped curve**|Heavy tails on both ends — consider robust scaling|

---

## 📌 3. Phase 2: Univariate Analysis — Individual Feature Distributions

Plot every numerical feature individually to understand its shape before considering its relationship to anything else.

---

### Chart 3: Grid of Histograms for All Numerical Features

**How to plot it:**

```python
df.hist(figsize=(20, 15), bins=30, edgecolor='black')
plt.suptitle('Distribution of All Numerical Features', y=1.02)
plt.tight_layout()
plt.show()
```

**Purpose:** Get a bird's-eye diagnostic view of every numerical column in one sweep. This is the fastest way to spot anomalies across the entire dataset.

**Parameters to inspect for each subplot:**

- Shape — is it symmetrical, right-skewed, left-skewed, or flat?
- Spikes — are there isolated towers separated by gaps?
- Zero walls — is there a massive bar at 0 dwarfing everything else?
- Range — what are the minimum and maximum values?

**5 Patterns to hunt for:**

| Visual Pattern              | Feature Examples                        | Diagnosis                                    | Action                          |
| :-------------------------- | :-------------------------------------- | :------------------------------------------- | :------------------------------ |
| **Flat rectangle**          | `Id`                                    | Sequential counter — no predictive value     | Drop immediately                |
| **Isolated integer spikes** | `MSSubClass`, `OverallQual`             | Nominal categorical code dressed as a number | Cast to string                  |
| **Long right tail**         | `LotArea`, `GrLivArea`, `SalePrice`     | Log-normal skew — outliers will dominate OLS | Apply `np.log1p()`              |
| **Massive zero wall**       | `PoolArea`, `3SsnPorch`, `LowQualFinSF` | Zero-inflated sparse feature                 | Create binary `HasFeature` flag |
| **Dual peaks**              | `YearRemodAdd`, `GarageYrBlt`           | Two mixed populations                        | Create boolean separation flag  |

---

### Chart 4: Bar Chart for Categorical Features

**How to plot it:**

```python
categorical_cols = df.select_dtypes(include='object').columns

fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(20, 20))
axes = axes.flatten()

for i, col in enumerate(categorical_cols):
    df[col].value_counts().plot(kind='bar', ax=axes[i], edgecolor='black')
    axes[i].set_title(col)
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
```

**Purpose:** Understand the class balance and cardinality of every categorical feature.

**Parameters to inspect:**

- How many unique categories does the feature have?
- Is one category dominating (e.g., 95% of rows are one label)?
- Are there rare categories with very few rows?

**Patterns you might find:**

|Pattern|What it Means|Action|
|:--|:--|:--|
|**One bar towers over all others**|Near-constant feature — very low variance|Consider dropping or using as a binary flag|
|**Many categories with equal heights**|High-cardinality balanced feature|Use Target Encoding or Top-K OHE|
|**Many rare categories with tiny bars**|Long-tail categories|Group rare categories into an "Other" bucket|
|**Only 2–3 bars**|Low-cardinality nominal feature|Use standard One-Hot Encoding (drop_first=True)|

---

## 📌 4. Phase 3: Bivariate Analysis — How Features Relate to SalePrice

Now connect each feature to your target variable one at a time.

---

### Chart 5: Scatter Plot — Continuous Feature vs SalePrice

**How to plot it:**

```python
key_features = ['GrLivArea', 'TotalBsmtSF', 'GarageArea', 'LotArea']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, col in enumerate(key_features):
    axes[i].scatter(df[col], df['SalePrice'], alpha=0.4, edgecolors='none')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('SalePrice')
    axes[i].set_title(f'{col} vs SalePrice')

plt.tight_layout()
plt.show()
```

**Purpose:** Identify linear relationships, outliers, and non-linear patterns between continuous features and the target.

**Parameters to inspect:**

- Direction — does SalePrice increase as the feature increases?
- Strength — are the points tightly packed around an imaginary line, or scattered?
- Outliers — are there isolated points far from the main cluster?
- Curvature — does the relationship curve upward (suggesting a log transformation would help)?

**Patterns you might find:**

|Pattern|What it Means|
|:--|:--|
|**Tight upward diagonal cloud**|Strong positive linear relationship — high predictive power|
|**Fan shape widening to the right**|Heteroscedasticity — log transformation of SalePrice will help|
|**Two or three isolated dots far from the cluster**|High-leverage outliers — flag for potential removal|
|**Flat horizontal cloud**|No relationship with SalePrice — consider dropping|
|**Curved upward band**|Non-linear relationship — log transform the feature|

> **Key outliers to watch in the Ames dataset:** `GrLivArea` typically has 2–3 huge houses (>4000 sq ft) that sold for surprisingly low prices. These are known outliers that should be removed before modelling.

---

### Chart 6: Box Plot — Categorical Feature vs SalePrice

**How to plot it:**

```python
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

cat_features = ['Neighborhood', 'OverallQual', 'ExterQual', 'KitchenQual', 'BsmtQual', 'GarageFinish']

for i, col in enumerate(cat_features):
    sns.boxplot(x=df[col], y=df['SalePrice'], ax=axes[i], order=sorted(df[col].dropna().unique()))
    axes[i].set_title(f'{col} vs SalePrice')
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
```

**Purpose:** Compare the median SalePrice and spread across different categories. Essential for ordinal quality features.

**Parameters to inspect:**

- Median line position inside each box — does it climb as quality improves?
- Box height (IQR) — wide boxes mean high price variance within that category
- Whiskers — how far do the tails extend?
- Diamond or circle outlier dots — extreme prices within a category

**Patterns you might find:**

|Pattern|What it Means|
|:--|:--|
|**Boxes staircase upward left-to-right**|Strong ordinal relationship — use Label Encoding preserving the order|
|**All boxes at the same height**|The category has no relationship with SalePrice — low predictive value|
|**One box dramatically higher than others**|That specific category commands a strong price premium|
|**Wide boxes with many outlier dots**|High variance within category — may need sub-grouping|

> **What to look for specifically:** `OverallQual` should show a very clean staircase pattern. `Neighborhood` will show dramatic price differences between areas like NridgHt vs. MeadowV.

---

### Chart 7: Violin Plot — Ordinal Quality vs SalePrice

**How to plot it:**

```python
sns.violinplot(x='OverallQual', y='SalePrice', data=df, palette='viridis')
plt.title('Overall Quality vs SalePrice')
plt.show()
```

**Purpose:** A richer version of the box plot — shows the full distribution shape within each category, not just the median and quartiles.

**Parameters to inspect:**

- Width of the violin at different price points — where does the density cluster?
- Symmetry — is the distribution within each quality rating symmetric or skewed?
- Tail length — do high-quality homes have a long upper price tail?

**Patterns you might find:**

|Pattern|What it Means|
|:--|:--|
|**Violins get taller and wider as quality increases**|Higher quality homes have both higher and more variable prices|
|**Very thin violins**|Few observations in that quality tier — be cautious about drawing conclusions|
|**Long upper tail on high-quality violins**|Top-rated homes have luxury outliers pulling the price ceiling up|

---

## 📌 5. Phase 4: Multivariate Analysis — Relationships Between Multiple Features

---

### Chart 8: Correlation Heatmap

**How to plot it:**

```python
import numpy as np

corr_matrix = df.select_dtypes(include=[np.number]).corr()

plt.figure(figsize=(18, 14))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Show only lower triangle
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=False,
    cmap='coolwarm',
    center=0,
    linewidths=0.5,
    vmin=-1, vmax=1
)
plt.title('Pearson Correlation Matrix — All Numerical Features')
plt.show()
```

**Purpose:** Identify which features correlate strongly with SalePrice (high predictive value) and which features correlate strongly with each other (multicollinearity risk).

**Parameters to inspect:**

- The `SalePrice` row/column — which features show the darkest red (positive correlation)?
- Feature-to-feature dark cells — which pairs of features are nearly identical?
- Blue cells — which features inversely correlate with SalePrice?

**Patterns you might find:**

|Pattern|What it Means|Action|
|:--|:--|:--|
|**Feature has r > 0.5 with SalePrice**|Strong predictor — keep it|Prioritise in feature selection|
|**Two features have r > 0.85 with each other**|Multicollinearity — they duplicate information|Drop one or combine them|
|**Feature has r ≈ 0.0 with SalePrice**|Weak predictor — likely low value|Consider dropping|
|**Feature has r < −0.3 with SalePrice**|Inverse relationship — higher value = lower price|Keep it — still predictive|

> **Strong correlates to expect:** `OverallQual`, `GrLivArea`, `GarageCars`, `GarageArea`, `TotalBsmtSF` should all show strong positive correlation with SalePrice. `GarageArea` and `GarageCars` will likely show high correlation with each other — a multicollinearity pair.

---

### Chart 9: Pairplot of Top Correlated Features

**How to plot it:**

```python
top_features = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(df[top_features], diag_kind='kde', plot_kws={'alpha': 0.4})
plt.suptitle('Pairplot — Top Correlated Features vs SalePrice', y=1.02)
plt.show()
```

**Purpose:** See all pairwise relationships between your top features simultaneously. The diagonal shows each feature's distribution; every off-diagonal cell is a scatter plot.

**Parameters to inspect:**

- The SalePrice row — do the scatter plots show upward trends?
- The diagonal KDE curves — are any features strongly skewed?
- Off-diagonal feature-feature cells — are any features near-perfectly linearly related?

---

### Chart 10: Grouped Bar Chart — Neighbourhood vs Mean SalePrice

**How to plot it:**

```python
neighbourhood_price = df.groupby('Neighborhood')['SalePrice'].mean().sort_values(ascending=False)

plt.figure(figsize=(14, 6))
neighbourhood_price.plot(kind='bar', color='steelblue', edgecolor='black')
plt.title('Mean SalePrice by Neighbourhood')
plt.xlabel('Neighbourhood')
plt.ylabel('Mean Sale Price ($)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

**Purpose:** Understand geographic price variation — one of the strongest drivers in any housing dataset.

**Parameters to inspect:**

- How large is the price gap between the most and least expensive neighbourhoods?
- Are there clear tiers (luxury vs. mid-range vs. affordable)?
- Are any neighbourhoods surprisingly close together in price?

---

## 📌 6. Phase 5: Missing Data Visualisation

---

### Chart 11: Missing Value Heatmap

**How to plot it:**

```python
import missingno as msno

msno.matrix(df, figsize=(18, 8), sparkline=False)
plt.title('Missing Value Pattern Matrix')
plt.show()
```

Or without the missingno library:

```python
missing = df.isnull().mean().sort_values(ascending=False)
missing = missing[missing > 0]

plt.figure(figsize=(12, 6))
missing.plot(kind='bar', color='tomato', edgecolor='black')
plt.title('Percentage of Missing Values per Feature')
plt.ylabel('Missing Ratio')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

**Purpose:** Map which columns have missing values and whether the missingness is random or systematic.

**Parameters to inspect:**

- Which features have the highest missing percentage?
- Do multiple features show missing values on the same rows? (suggests MNAR — missing not at random)
- Are features with high missingness related to each other?

**Patterns you might find:**

|Pattern|What it Means|Action|
|:--|:--|:--|
|**PoolQC, MiscFeature missing > 99%**|House simply doesn't have that feature|Replace NaN with `"None"` string|
|**Alley, Fence missing ~93%**|Most houses have no alley/fence access|Replace NaN with `"None"`|
|**GarageType, GarageFinish missing ~5%**|Small missing chunk — likely no garage|Group impute with `"None"` or `0`|
|**LotFrontage missing ~17%**|Genuine missing measurement|Impute using neighbourhood median|

> **Key insight for the Ames dataset:** Most missing values in quality/condition columns (like `PoolQC`, `FireplaceQu`, `GarageQual`) are not truly missing — they mean the feature doesn't exist. Replace with `"None"` rather than imputing a fake value.

---

### Chart 12: Missing Value Correlation Heatmap (MNAR Proof)

**How to plot it:**

```python
# Check if missingness in one column relates to SalePrice
df['PoolQC_missing'] = df['PoolQC'].isnull().astype(int)

sns.boxplot(x='PoolQC_missing', y='SalePrice', data=df)
plt.title('Does PoolQC Missingness Relate to SalePrice?')
plt.xticks([0, 1], ['Not Missing', 'Missing'])
plt.show()
```

**Purpose:** Prove whether missing data is MCAR (random) or MNAR (systematic). If houses with missing `PoolQC` sell at significantly different prices, the missingness carries predictive signal.

---

## 📌 7. Phase 6: Outlier Detection

---

### Chart 13: Box Plots for Outlier Detection

**How to plot it:**

```python
numerical_cols = ['GrLivArea', 'LotArea', 'TotalBsmtSF', 'GarageArea', 'SalePrice']

fig, axes = plt.subplots(1, len(numerical_cols), figsize=(20, 5))

for i, col in enumerate(numerical_cols):
    axes[i].boxplot(df[col].dropna())
    axes[i].set_title(col)
    axes[i].set_ylabel('Value')

plt.suptitle('Box Plots for Outlier Detection')
plt.tight_layout()
plt.show()
```

**Purpose:** Visualise the IQR bounds and identify data points that sit beyond 1.5× IQR from the quartiles.

**Parameters to inspect:**

- The box height — how wide is the interquartile range?
- The whisker length — how far do typical values extend?
- Individual dots beyond the whiskers — how many outliers are there, and how extreme are they?

**Patterns you might find:**

|Pattern|What it Means|
|:--|:--|
|**Many dots far above the upper whisker**|Right-skewed distribution with extreme values — log transform|
|**1–3 isolated dots very far from the box**|Specific outlier records — investigate individually|
|**Symmetric box with few outlier dots**|Well-behaved feature — no transformation needed|

---

### Chart 14: Scatter Plot with IQR Boundary Lines

**How to plot it:**

```python
Q1 = df['GrLivArea'].quantile(0.25)
Q3 = df['GrLivArea'].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR

plt.figure(figsize=(10, 6))
plt.scatter(df['GrLivArea'], df['SalePrice'], alpha=0.4, label='Data Points')
plt.axvline(x=upper_bound, color='red', linestyle='--', label=f'IQR Upper Bound ({upper_bound:.0f})')
plt.xlabel('GrLivArea (sq ft)')
plt.ylabel('SalePrice ($)')
plt.title('GrLivArea vs SalePrice with IQR Outlier Boundary')
plt.legend()
plt.show()
```

**Purpose:** Visually confirm which specific data points are outliers and whether they follow the main trend or contradict it.

> **Critical outlier pattern in the Ames dataset:** There are 2 houses with `GrLivArea > 4000 sq ft` that sold for under $200k. These are legitimate measurement errors or distress sales and should be removed before model training.

---

## 📌 8. EDA Visualisation Sequence Checklist

Use this as your step-by-step workflow when starting any housing EDA:

- [ ] **Step 1:** Plot histogram + KDE of `SalePrice` → check for right skew
- [ ] **Step 2:** Apply `np.log1p()` to SalePrice and re-plot → verify bell curve
- [ ] **Step 3:** Plot Q-Q plot of transformed SalePrice → verify normality
- [ ] **Step 4:** Plot `df.hist()` grid for all numerical columns → identify anomaly patterns
- [ ] **Step 5:** Plot bar charts for all categorical columns → check class balance
- [ ] **Step 6:** Plot scatter plots of top numerical features vs SalePrice → identify linear relationships and outliers
- [ ] **Step 7:** Plot box plots of quality/condition features vs SalePrice → verify ordinal staircase patterns
- [ ] **Step 8:** Plot correlation heatmap → identify top predictors and multicollinearity pairs
- [ ] **Step 9:** Plot pairplot of top 6–8 correlated features → deep-dive relationships
- [ ] **Step 10:** Plot missing value bar chart → identify and categorise null patterns
- [ ] **Step 11:** Plot box plots for outlier detection on key features → flag extreme values
- [ ] **Step 12:** Remove known outliers and re-check scatter plots → confirm clean data

---

## 📌 9. Summary: Chart-to-Purpose Quick Reference

|Chart Type|Primary Purpose|Key Question Answered|
|:--|:--|:--|
|**Histogram + KDE**|Single feature distribution shape|Is this feature normally distributed or skewed?|
|**Q-Q Plot**|Normality verification|Does this feature follow a normal distribution?|
|**Grid of Histograms**|Multi-feature bird's-eye audit|Which features have anomalous shapes?|
|**Bar Chart**|Categorical class balance|How are categories distributed? Are any dominant?|
|**Scatter Plot**|Continuous feature vs SalePrice|Is there a linear relationship? Are there outliers?|
|**Box Plot (categorical)**|Category vs SalePrice comparison|Does SalePrice differ significantly across categories?|
|**Violin Plot**|Distribution shape within categories|What does the full price distribution look like per category?|
|**Correlation Heatmap**|Feature-to-feature relationships|Which features predict SalePrice? Which are redundant?|
|**Pairplot**|Multi-feature pairwise relationships|How do the top features relate to each other and SalePrice?|
|**Missing Value Bar Chart**|Null pattern diagnosis|Which columns are missing data and by how much?|
|**Box Plot (outlier)**|Outlier detection|Which records sit beyond IQR bounds?|

---

## 🔗 Related Notes

- [[ML — Advanced Numerical Feature Taxonomies]]
- [[ML — Histogram Pattern Recognition & Feature Engineering Actions]]
- [[Feature Engineering — Missing Data Mechanisms & Continuous Imputation (Day 1)]]
- [[Feature Engineering — Handling Missing Categorical Values]]
- [[Feature Engineering — Feature Scaling & Gaussian Transformations (Day 6)]]
- [[Feature Selection — Removing Highly Correlated Features Using Pearson Correlation]]
- [[Feature Selection — Dropping Constant Features Using Variance Threshold]]
- [[Data Science — Unified EDA, Feature Engineering & Feature Selection Roadmap]]

---

_Tags: #eda #visualisation #housing-dataset #ames-iowa #kaggle #histogram #scatter-plot #correlation-heatmap #box-plot #missing-values #outlier-detection #seaborn #matplotlib #python #data-science #beginners-guide_