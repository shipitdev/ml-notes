# 17. Logistic Regression Details Pt1: Coefficients
🔗 https://www.youtube.com/watch?v=vN5cNN2-HWE

## The Big Idea
This video zooms into exactly what a Logistic Regression coefficient means, how to interpret it in plain terms, and how we test whether it's statistically significant (via the Wald test).

## Flow of the Video

### 1. Simple example: predicting heart disease from a single 0/1 variable
- Say our only predictor is "genotype" — coded 0 (no mutation) or 1 (has mutation) — predicting whether someone has heart disease.
- The fitted equation:
```
log(odds of disease) = intercept + coefficient × genotype
```

### 2. Interpreting the intercept
- When genotype = 0 (no mutation), the equation becomes: log(odds) = intercept.
- So the **intercept is simply the log(odds) of disease for the baseline group** (genotype = 0).

### 3. Interpreting the coefficient
- When genotype = 1 (has mutation): log(odds) = intercept + coefficient.
- So the **coefficient is the *change* in log(odds)** when moving from genotype=0 to genotype=1.
- This matches video #15 exactly: the coefficient = **log(Odds Ratio)** comparing the "mutation" group to the "no mutation" group.
- To get an actual Odds Ratio, exponentiate: `Odds Ratio = e^coefficient`.
  - Simple example: if coefficient = 0.7, Odds Ratio = e^0.7 ≈ 2.0 → having the mutation roughly **doubles** the odds of heart disease.

![How Coefficients Map to the Sigmoid Curve|697](./images/logistic_coefficients.svg)

### 4. How confident are we in this coefficient? The Wald test
- Just like ordinary regression coefficients have standard errors, so does the logistic regression coefficient.
- We compute a **Wald statistic**: `z = coefficient / standard error`.
- This z-value tells us how many standard errors the coefficient is away from 0 (i.e., away from "no effect").
- A large |z| → a small p-value → we're confident the coefficient (and thus the Odds Ratio) is genuinely different from 1 (no effect), not just noise.

![The Wald Test for Coefficient Significance](./images/wald_test.svg)

### 5. Watch out: huge standard errors happen with small/sparse data
- If a category has very few data points (e.g., very few people with the mutation), the standard error can be enormous, making even a large coefficient statistically insignificant.
- This is a common practical pitfall in logistic regression with small sample sizes or rare categories.

## Key Takeaways (Quick Recall)
- Intercept = log(odds) for the baseline (reference) group.
- Coefficient = change in log(odds) when moving to the comparison group = log(Odds Ratio).
- Exponentiate the coefficient to get the actual Odds Ratio: e^coefficient.
- Wald test (z = coefficient / standard error) checks if a coefficient is statistically significant.
- Small sample sizes can inflate standard errors, making real effects look statistically weak.
