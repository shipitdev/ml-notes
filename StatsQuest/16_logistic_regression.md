# 16. StatQuest: Logistic Regression
🔗 https://www.youtube.com/watch?v=yIYKR4sgzI8

## The Big Idea
Logistic Regression is used when the thing you're predicting is a **category** (like Yes/No, Has Disease/Doesn't), not a continuous number. Instead of fitting a straight line to raw data, it fits an S-shaped curve to the **probability** of the category — built on top of everything learned about log(odds).

## Flow of the Video

### 1. Why plain Linear Regression doesn't work for Yes/No outcomes
- Simple example: predicting "Obese" (Yes=1, No=0) from weight.
- If you fit a straight line to these 0/1 dots, the line will happily predict values like -0.3 or 1.7 for certain weights — but probabilities can only range from 0 to 1! A straight line has no way to respect that boundary.

### 2. The fix: model probability with an S-shaped curve
- Instead of a straight line, Logistic Regression fits a curve that:
  - Approaches 0 for very low values of the predictor,
  - Approaches 1 for very high values,
  - Smoothly transitions through the middle.
- This S-shaped curve is called the **sigmoid (logistic) function**.

### 3. Connecting back to log(odds) (videos #14-15)
- Under the hood, Logistic Regression actually fits a **straight line to log(odds)**, not to probability directly (remember: log(odds) stretches probability's 0-to-1 range out to -∞ to +∞, so a straight line finally makes sense there).
- Formula:
```
log(odds) = intercept + slope × x
```
- Then, to get back to an interpretable probability, we convert log(odds) back to probability using:
```
p = e^(log odds) / (1 + e^(log odds))
```
- This conversion is exactly what turns the straight line (in log-odds space) into the S-curve (in probability space).

### 4. How do we know the curve is "best"?
- Unlike Linear Regression's "minimize the sum of squared residuals" approach, Logistic Regression can't use squared residuals in quite the same way (because outcomes are 0/1, not continuous).
- Instead, it uses **Maximum Likelihood** — finding the S-curve that makes the actually-observed 0s and 1s as probable as possible under the model. (Full details are in the next video!)

### 5. What the output tells us
- Coefficients from Logistic Regression represent **log(Odds Ratios)** (as covered in video #15) — how much the log-odds of the outcome shift per unit increase in a predictor.
- We can convert a coefficient back to an actual Odds Ratio by exponentiating it: `Odds Ratio = e^(coefficient)`.

## Key Takeaways (Quick Recall)
- Logistic Regression predicts categories (e.g., Yes/No) by modeling **probability** with an S-shaped (sigmoid) curve.
- Internally, it fits a straight line to **log(odds)**, then converts back to probability.
- Fitting is done via **Maximum Likelihood**, not least squares.
- Coefficients = log(Odds Ratios); exponentiate them to get interpretable Odds Ratios.
