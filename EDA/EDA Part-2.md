# 📊 Machine Learning Foundations: Exploratory Data Analysis (EDA) — Part 2

Exploratory Data Analysis (EDA) leverages geometric layouts to isolate patterns, check data distribution splits, and verify classification boundaries. Using the standard **Iris dataset** ($150 \times 5$ data matrix mapping `sepal_length`, `sepal_width`, `petal_length`, `petal_width` to target label `species`), these notes break down the architectural frameworks for programmatic univariate, bivariate, and multivariate analysis.

---
## 📌 1. Univariate Analysis (1D Dimensional Space Plotting)

**Univariate Analysis** examines exactly **one numerical feature column** at a time to evaluate its geometric distribution and check whether classes can be isolated along a single directional axis.

### 🛠️ The 1D Horizontal Matrix Hack (`np.zeros_like`)

When plotting a single variable (e.g., `sepal_length`) on a scatter plot, there is no second axis. To prevent visualization libraries from throwing formatting exceptions, data engineers keep the Y-axis fixed at `0` using a matching zero array.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load Target Tabular Matrix
df = pd.read_csv('[https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv](https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv)')

# Split Row Categories Based on Class Labels
df_setosa = df.loc[df['species'] == 'setosa']
df_virginica = df.loc[df['species'] == 'virginica']
df_versicolor = df.loc[df['species'] == 'versicolor']

# Plot 1D Scatter Matrices along a flat baseline
plt.plot(df_setosa['sepal_length'], np.zeros_like(df_setosa['sepal_length']), 'o', label='Setosa')
plt.plot(df_virginica['sepal_length'], np.zeros_like(df_virginica['sepal_length']), 'o', label='Virginica')
plt.plot(df_versicolor['sepal_length'], np.zeros_like(df_versicolor['sepal_length']), 'o', label='Versicolor')

plt.xlabel('Sepal Length')
plt.legend()
plt.show()
```
### 🚨 Critical Downstream Bottleneck: The Overlap Problem

* **Observation:** While `Setosa` data points cluster tightly on the lower end of the spectrum, the `Virginica` and `Versicolor` observations blend heavily across the middle coordinates.
* **Pipeline Impact:** Because the features overlap, drawing a clean, single-point linear boundary on this 1D chart is mathematically impossible. A model relying on this single feature would generate high training error rates. To resolve this, we must expand our feature dimensions.

---

## 📌 2. Bivariate Analysis (2D Multi-Feature Interaction Mapping)
Bivariate Analysis scales up our geometric views by charting exactly two distinct feature columns simultaneously across an (X, Y) coordinate plane.

### 🐍 Seaborn FacetGrid Matrix Routine
To visualize how features interact while maintaining distinct colors for each target class, we map our coordinates using Seaborn's `FacetGrid` pipeline:

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Instantiate the facet tracking layout grid categorized by target label
sns.FacetGrid(df, hue="species", height=5) \
   .map(plt.scatter, "petal_length", "sepal_width") \
   .add_legend()

plt.show()
```
### 🧠 Algorithmic Strategy Mapping From 2D Spaces
* **The Symmetrical Isolation State (Setosa):** Looking at the `petal_length` vs. `sepal_width` scatter plot, `Setosa` separates cleanly into its own isolated cluster. This indicates it is linearly separable. Simple distance or hyper-plane algorithms (such as Support Vector Machines with Linear Kernels or basic Logistic Regression) can find a clean split-point effortlessly.
* **The Interlocking Clustering State (Versicolor / Virginica):** The coordinates for `Versicolor` and `Virginica` sit closer together, creating minor overlap zones at their boundaries. A straight linear line would misclassify points near the border.
* **The Engineering Action:** For these overlapping zones, you must select non-linear estimators capable of charting curved decision boundaries, such as Random Forests, Gradient Boosting (XGBoost), or Support Vector Machines with RBF Kernels.

---

## 📌 3. Multivariate Analysis (High-Dimensional Feature Dashboarding)
When your data matrix contains more than two features, you can no longer display them on a single flat scatter chart. To work around this visual limitation, data engineers generate paired matrix grids to systematically inspect every combination of features.

### 🛠️ High-Dimensional Dashboard Activation (`sns.pairplot`)
```python
# Generate a complete grid dashboard across all numerical attributes
sns.pairplot(df, hue="species", height=3)
plt.show()
```
### 🔍 How to Read and Filter a Pair Plot Matrix Dashboard

* **Symmetrical Grid Layout:** If your data frame has 4 unique numerical feature columns, `sns.pairplot` produces an exact 4x4 dashboard grid containing 16 independent charts.
* **The Diagonal Density Tracks:** Where a feature intersects with itself (e.g., Row 1, Column 1 tracking `sepal_length` vs. `sepal_length`), a scatter plot would simply form a meaningless straight line. The pair plot automatically replaces these charts with univariate **Probability Density Functions (PDF / Kernel Density Estimations)** or **Histograms**. These curves clearly show the data's shape, variance, and overlap zones.
* **The Off-Diagonal Spatial Coordinates:** Every chart outside the main diagonal presents a 2D bivariate scatter plot mapping those specific cross-features (e.g., tracking `petal_length` against `petal_width`).

---

### 🎯 Feature Selection Protocol: Targeting Dominant Separability

By analyzing the entire grid matrix, you can identify which feature pairs create the cleanest separation between classes:
* **Low Separability Pairs (sepal_length vs. sepal_width):** The scatter plots for these features show heavily mixed clouds of data points where all three classes overlap, making it difficult for an algorithm to find clear boundaries.
* **High Separability Pairs (petal_length vs. petal_width):** These features show clean, well-defined clusters with minimal overlap between categories.
* **The Engineering Action:** The pair plot reveals that `petal_length` and `petal_width` are the most predictive features in the dataset. If you need to streamline your model's computational footprint, you can safely drop the less predictive `sepal` columns and train your model using only the `petal` attributes with little to no loss in accuracy.