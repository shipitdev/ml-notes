# 📉 Machine Learning Practical: Multiple Linear Regression Architecture

## 📌 1. Theoretical Evolution: Moving Beyond Single Features

In basic simple linear setups, we chart our data constraints using a single input line across a 2-dimensional plane.

### 📐 Recapping Simple Linear Regression
The simple standard baseline model relies on exactly one independent variable to map a prediction:
$$y = \beta_0 + \beta_1 x_1$$
* $\beta_0$ = The baseline intercept constant.
* $\beta_1$ = The direct single feature slope coefficient.
* $x_1$ = The standalone independent column attribute (e.g., `Size of a house`).

### 🚀 Scaling to Multiple Linear Regression
In production settings, real-world targets are influenced by multiple forces simultaneously. To model these interactions, we scale our equation to support a multi-dimensional feature space:
$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_3 + \dots + \beta_n x_n$$



* **The Hyperplane Concept:** Instead of drawing a flat 1D line on a 2D grid, multiple linear regression crafts a multi-dimensional **Hyperplane** through a high-dimensional cloud of coordinates.
* **The Coefficient Dynamics:** Every individual $\beta_j$ weight coefficient represents the exact unit change expected in target $y$ when its associated feature $x_j$ shifts by one unit, assuming all other feature vectors in the data matrix remain absolutely static.

---

## 📌 2. The Operational Case Study & Feature Matrix Design

We implement this pipeline using a dataset tracking performance metrics across **50 Startup Companies** to forecast absolute **Profitability Levels**.

### 📊 The Raw Column Archetype
* **Independent Feature Variables ($X$):** `R&D Spend`, `Administration Spend`, `Marketing Spend`, and `State`.
* **Dependent Target Variable ($y$):** `Profit`.

---

## 📌 3. Managing Categorical Inputs & Resolving the Dummy Variable Trap

Computers cannot perform raw linear optimization checks directly on text strings (like `New York`, `California`, or `Florida`). We must transform these attributes using **One-Hot Encoding**.

### 🚨 The Dummy Variable Trap Mechanics
If a text column contains 3 distinct geographical values, generating 3 independent dummy boolean columns (`NY`, `CA`, `FL`) introduces severe **Multicollinearity**. 

Because the final category can be perfectly predicted by the presence or absence of the first two, the columns become linearly dependent. This breaks down standard matrix math solvers during gradient descent routines.

### 🛡️ The Engineering Fix: `drop_first=True`
To maintain full mathematical integrity, we apply the **$n-1$ rule**. If you have 3 distinct string categories, you generate exactly **2 dummy columns**:
* If `Florida = 1` and `New York = 0` $\rightarrow$ The startup is in Florida.
* If `Florida = 0` and `New York = 1` $\rightarrow$ The startup is in New York.
* If `Florida = 0` and `New York = 0` $\rightarrow$ By process of deduction, the startup must be in California.

---

## 📌 4. Step-by-Step Python Engineering Pipeline

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 1. Ingest Data Matrix
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1]  # Extract R&D Spend, Administration, Marketing Spend, State
y = dataset.iloc[:, -1]   # Extract target Profit column

# 2. Convert text states using one-hot encoding & eliminate the dummy trap
states = pd.get_dummies(X['State'], drop_first=True)

# 3. Restructure feature grid spaces
X = X.drop('State', axis=1)    # Safely drop the original string text column
X = pd.concat([X, states], axis=1)  # Concatenate clean binary encoding columns

# 4. Isolate train/test partitions using an 80/20 distribution split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 5. Fit model weights using Scikit-Learn's core linear framework
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# 6. Gather out-of-sample predictions
y_pred = regressor.predict(X_test)
```
## 📌 5. Evaluation Math Blueprint: The $R^2$ Score Metric

To evaluate how well our high-dimensional hyperplane fits our unseen test points, we use the **Coefficient of Determination ($R^2$ Score)**:

$$R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{mean}}}$$

---

### 🛠️ Deconstructing the Sum of Squares Fractions

#### A. Residual Sum of Squares ($SS_{\text{res}}$)
Tracks the variance left unexplained by your model by squaring the vertical distance from each real data point to your fitted hyperplane:
$$SS_{\text{res}} = \sum_{i=1}^{n} \left(y_i - \hat{y}_i\right)^2$$

#### B. Total Sum of Squares ($SS_{\text{mean}}$)
Establishes a worst-case validation baseline by matching our data against a simple horizontal line drawn at the dataset's target mean ($\bar{y}$):
$$SS_{\text{mean}} = \sum_{i=1}^{n} \left(y_i - \bar{y}\right)^2$$

---

### 📊 Practical Interpretation Metrics

* **The Scale Limits:** $R^2$ output states generally score between **0** and **1**.
* **Closer to 1.0 State:** Indicates an exceptionally accurate model fit. A higher score means your engineered hyperplane significantly reduces error compared to the baseline mean model.
* **Our Final Pipeline Score:** Executing `r2_score(y_test, y_pred)` on our test coordinates returns a solid score of **0.9347**. This confirms that our regularized linear model successfully explains **93.47%** of all variance within our startup profit targets!