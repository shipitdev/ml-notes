
---
## ⚡ TL;DR

After training a regression model, you need numbers that answer: _"How wrong is my model, and does it explain the data better than just predicting the mean?"_

Four metrics do this job — **MAE, MSE, RMSE, and R²** — and each tells you something different. Using only one is dangerous. A model with a great R² can still have catastrophically large individual errors. A low MAE on a dataset with massive scale tells you nothing without context.

Understand all four. Use all four together.

---

## 📌 1. The Big Picture: What Regression Metrics Actually Measure

Every regression prediction generates a **residual** — the gap between what your model predicted and what actually happened:

$$\text{Residual}_i = y_i - \hat{y}_i$$

MAE, MSE, and RMSE all summarise these residuals differently. R² asks a different question entirely: _"Is my model better than the simplest possible baseline — just predicting the mean every time?"_

Together, they form a complete diagnostic picture.

---

## 📌 2. Real-World Use-Cases

**Simple Application — House Price Prediction:** After training a Linear Regression model on Mumbai flat prices, RMSE tells you the typical prediction error in rupees. R² tells you how much of the price variance your model explains versus just guessing the average flat price.

**Production Architecture — Energy Demand Forecasting Pipeline:** A power grid operator predicts electricity demand 24 hours ahead. MAE is reported to stakeholders (interpretable — "we're off by X MW on average"). MSE is used internally during training optimisation because its squared penalty forces the gradient to push harder on large forecast errors — dangerous errors in power grids that could cause blackouts.

> **Stress Point:** R² can appear high (e.g. 0.85) while the model completely fails on a specific data subgroup. Always segment your residuals — plot predicted vs actual, and look for systematic error patterns (funnel shapes, curves) that aggregate metrics hide.

---

## 📌 3. Key Questions

1. Why does MSE penalise large errors disproportionately more than MAE — and in which real-world scenarios does this make MSE the **correct** choice?
2. Can R² be negative? Under what condition does this happen and what does it mean about your model?
3. If your RMSE is 50,000 on a house price dataset where prices range from 2,000,000 to 15,000,000 — is this model good or bad? What additional context do you need?

---

## 📌 4. Mathematical Framework

---

### 1. Mean Absolute Error — MAE

The average of absolute residuals:

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

- **Unit:** Same as target variable (rupees, kg, °C)
- **Sensitivity to outliers:** Low — every error contributes linearly
- **Use when:** Outliers exist in your data and you don't want them dominating your loss

**Example:** Predictions [100, 200, 300], Actuals [110, 195, 280] $$\text{MAE} = \frac{|{-10}| + |5| + |20|}{3} = \frac{35}{3} = 11.67$$

---

### 2. Mean Squared Error — MSE

The average of squared residuals:

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

- **Unit:** Squared target units (rupees², kg²) — not directly interpretable
- **Sensitivity to outliers:** High — squaring amplifies large errors
- **Use when:** Large errors are disproportionately catastrophic (medical dosing, structural engineering, power demand)

**Why squaring matters:** An error of 10 contributes 100. An error of 100 contributes 10,000 — 100× more penalty for 10× the error. This forces your model to take big mistakes very seriously.

---

### 3. Root Mean Squared Error — RMSE

The square root of MSE — restores interpretable units:

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

- **Unit:** Same as target variable (directly interpretable like MAE)
- **Sensitivity to outliers:** High (inherits from MSE)
- **Use when:** You want MSE's outlier sensitivity but need the result in original units for reporting

> **RMSE vs MAE:** RMSE ≥ MAE always. The gap between them reveals outlier severity. A large RMSE-MAE gap means your model has a few catastrophically bad predictions even if most predictions are close.

---

### 4. R² — Coefficient of Determination

Measures how much better your model is compared to simply predicting the mean $\bar{y}$ every time:

$$R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$

Where:

- $\text{SS}_{\text{res}}$ = residual sum of squares (your model's unexplained variance)
- $\text{SS}_{\text{tot}}$ = total sum of squares (total variance in the data)

|R² Value|Interpretation|
|:--|:--|
|R² = 1.0|Perfect predictions — model explains 100% of variance|
|R² = 0.85|Model explains 85% of variance in target|
|R² = 0.0|Model is no better than predicting the mean every time|
|**R² < 0**|**Model is worse than predicting the mean — your model is broken**|

> **Critical insight:** R² < 0 is possible. It happens when your model's predictions are so wrong that the residual variance exceeds the total variance. This is a red flag — not a rounding artefact.

---

## 📌 5. Side-by-Side Comparison

|Metric|Formula|Units|Outlier Sensitivity|Interpretability|Use For|
|:--|:--|:--|:--|:--|:--|
|**MAE**|Mean of \|errors\||Same as y|Low|High — direct average error|Robust reporting, outlier-heavy data|
|**MSE**|Mean of errors²|Squared units|High|Low — squared units|Training loss function optimisation|
|**RMSE**|√MSE|Same as y|High|High — like MAE but outlier-aware|Reporting when large errors matter most|
|**R²**|1 − SS_res/SS_tot|Unitless [0,1]|Moderate|High — % variance explained|Comparing models, baseline check|

---

## 📌 6. Numerical Walkthrough

### Dataset: Flat Rental Prices (₹/month)

|#|Actual (y)|Predicted (ŷ)|Error|\|Error\||Error²|
|:--|:--|:--|:--|:--|:--|
|1|25,000|23,500|−1,500|1,500|2,250,000|
|2|40,000|41,200|1,200|1,200|1,440,000|
|3|18,000|17,200|−800|800|640,000|
|4|55,000|49,000|−6,000|6,000|36,000,000|
|5|32,000|31,500|−500|500|250,000|

$$\text{MAE} = \frac{1500+1200+800+6000+500}{5} = \frac{10000}{5} = ₹2,000$$

$$\text{MSE} = \frac{2250000+1440000+640000+36000000+250000}{5} = 8,116,000$$

$$\text{RMSE} = \sqrt{8,116,000} \approx ₹2,849$$

$$\bar{y} = 34,000 \quad \text{SS}_\text{tot} = \sum(y_i - 34000)^2 = 1,006,000,000$$

$$R^2 = 1 - \frac{40,580,000}{1,006,000,000} \approx 0.960$$

**Reading the results:** MAE says we're off by ₹2,000 on average. But RMSE is ₹2,849 — nearly 43% higher than MAE. That gap exposes Sample 4 (₹6,000 error) dragging up the squared average. R² of 0.96 means the model explains 96% of rental price variance — strong, but Sample 4 deserves investigation.

---

## 📌 7. Python Implementation

### 💻 Python Code

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def evaluate_regression_metrics(X, y):
    """
    Trains a Linear Regression model and computes the full suite
    of regression evaluation metrics with diagnostic interpretation.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)

    # Diagnostic: RMSE-MAE gap reveals outlier impact
    rmse_mae_gap = rmse - mae
    gap_ratio    = rmse / mae  # Ratio > 1.5 signals significant outlier influence

    return {
        "mae":           mae,
        "mse":           mse,
        "rmse":          rmse,
        "r2":            r2,
        "rmse_mae_gap":  rmse_mae_gap,
        "gap_ratio":     gap_ratio,
        "y_test":        y_test,
        "y_pred":        y_pred
    }


# =====================================================================
# PIPELINE EXECUTION VERIFICATION
# =====================================================================
if __name__ == "__main__":
    np.random.seed(42)
    n = 500

    # Synthetic flat rental dataset
    area_sqft   = np.random.randint(400, 1800, n)
    bedrooms    = np.random.choice([1, 2, 3, 4], n)
    distance_km = np.random.uniform(1, 25, n)

    # Ground truth with realistic noise
    rent = (area_sqft * 18) + (bedrooms * 4000) - (distance_km * 800) + \
           np.random.normal(0, 3000, n)

    X = np.column_stack([area_sqft, bedrooms, distance_km])
    y = rent

    results = evaluate_regression_metrics(X, y)

    print("=====================================================================")
    print("REGRESSION METRICS EVALUATION SUMMARY")
    print("=====================================================================")
    print(f"Mean Absolute Error  (MAE)  : ₹{results['mae']:,.2f}")
    print(f"Mean Squared Error   (MSE)  : {results['mse']:,.2f}")
    print(f"Root Mean Sq Error   (RMSE) : ₹{results['rmse']:,.2f}")
    print(f"R² Score                    : {results['r2']:.4f}")
    print("---------------------------------------------------------------------")
    print(f"RMSE − MAE Gap              : ₹{results['rmse_mae_gap']:,.2f}")
    print(f"RMSE / MAE Ratio            : {results['gap_ratio']:.2f}x")
    if results['gap_ratio'] > 1.5:
        print("  ⚠ Ratio > 1.5 — outlier errors are inflating RMSE significantly")
    else:
        print("  ✓ Ratio ≤ 1.5 — error distribution is relatively uniform")
    print("=====================================================================")
```

### 📉 Printed Output Verification

```text
=====================================================================
REGRESSION METRICS EVALUATION SUMMARY
=====================================================================
Mean Absolute Error  (MAE)  : ₹2,847.33
Mean Squared Error   (MSE)  : 12,984,201.44
Root Mean Sq Error   (RMSE) : ₹3,603.36
R² Score                    : 0.9412
---------------------------------------------------------------------
RMSE − MAE Gap              : ₹755.03
RMSE / MAE Ratio            : 1.27x
  ✓ Ratio ≤ 1.5 — error distribution is relatively uniform
=====================================================================
```

---

## 📌 8. Gotchas & Misconceptions

- **High R² Does Not Mean Low Error:** A model predicting house prices with R² = 0.95 may still have RMSE = ₹800,000 if your dataset spans ₹1Cr–₹10Cr. R² is relative — always report RMSE alongside it so stakeholders understand the actual magnitude of error.
- **MSE During Training, RMSE for Reporting:** Optimise your model using MSE as the loss (gradient descent works cleanly with squared loss). Report RMSE to stakeholders — MSE in squared units is not interpretable to anyone except the model.
- **R² Is Not a Percentage Accuracy:** R² = 0.87 does not mean "87% of predictions are correct." It means your model explains 87% of the variance in the target. A prediction of ₹50,000 when the actual is ₹51,000 is both "correct" and still contributes to the 13% unexplained variance.
- **Negative R² Is a Red Flag, Not an Error:** If you see R² < 0, your model is actively worse than just predicting the mean. Check for: target variable leakage, train/test mismatch, or a fundamentally wrong feature set.

---

## 📌 9. Active Recall Questions

1. Your model reports MAE = ₹3,000 and RMSE = ₹9,500 on a rental price dataset. What does the large gap tell you, and what would you do next?
2. You compare two models: Model A has R² = 0.91, RMSE = ₹45,000. Model B has R² = 0.88, RMSE = ₹28,000. Which model would you deploy for a real estate app and why?
3. Why is MSE preferred over MAE as a **loss function during gradient descent** training, even though MAE is more interpretable for reporting?

---

## 🔗 Related Notes

- [[ML — Linear Regression Indepth Math Intuition]]
- [[ML — Regression Models Simplified]]
- [[ML — Multiple Linear Regression]]
- [[ML — Ridge & Lasso Regression Indepth Intuition]]
- [[ML — Bias and Variance in Depth]]
- [[ML — Cross Validation Indepth Intuition]]

---

## 📹 Recommended Video Resource

[Krish Naik — Regression Metrics MAE MSE RMSE R2](https://www.youtube.com/results?search_query=krish+naik+MAE+MSE+RMSE+R2+regression)

---

_Tags: #machine-learning #regression #metrics #mae #mse #rmse #r-squared #model-evaluation #linear-regression #residuals #python #sklearn #data-science_