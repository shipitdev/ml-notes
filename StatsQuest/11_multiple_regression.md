# 11. Multiple Regression, Clearly Explained!!!
🔗 https://www.youtube.com/watch?v=zITIFTsivN8

## The Big Idea
Multiple Regression is the same least-squares idea as simple Linear Regression, except now we use **more than one input variable** to predict the outcome — and we need a small tweak to R² so that adding more variables doesn't unfairly inflate our "goodness of fit" score.

## Flow of the Video

### 1. From one predictor to many
- Simple Linear Regression: predict weight using just height → fits a **line** (2D).
- Multiple Regression: predict weight using height **and** waist size (two predictors) → fits a **flat plane** through a 3D cloud of points instead of a line through a 2D scatterplot.
- With even more predictors, we can't visualize it anymore, but the math generalizes the same way — we're fitting a flat "plane" in higher dimensions.

### 2. Still minimizing Sum of Squared Residuals
- Exactly like before: for any candidate plane, measure the vertical distance (residual) from each point to the plane, square them, sum them up (SSR), and find the plane that minimizes this sum.
- Each predictor now gets its own coefficient (slope), plus one overall intercept.

### 3. Same R² formula, same idea
```
R² = ( SS(mean) - SS(fit) ) / SS(mean)
```
- Compares our multi-predictor plane's error to the naive "just guess the average" baseline — same as simple regression.

### 4. The catch: R² always goes up when you add more predictors
- Even totally useless/random predictors will nudge R² up slightly, just by chance, because the fitting process has more flexibility to reduce error.
- This makes plain R² **misleading** when comparing models with different numbers of predictors — a model with more variables will almost always look "better" even if those variables are junk.

### 5. The fix: Adjusted R²
- Adjusted R² **penalizes** the score for every extra predictor added, so it only goes up if a new variable meaningfully improves the fit (more than random chance would predict).
- Used to fairly compare models that have different numbers of predictors.

### 6. How do we know which predictors are actually useful?
- Just like simple regression, each predictor's coefficient comes with its own **p-value**, showing whether that specific variable's contribution is statistically significant, or could just be noise.
- We can also compare two whole models (e.g., with vs. without a certain variable) using an **F-test** to see if adding variables meaningfully improved the fit.

## Key Takeaways (Quick Recall)
- Multiple Regression = same least-squares fitting, just with more than one predictor (fits a plane/hyperplane instead of a line).
- R² formula is unchanged, but plain R² always increases with more predictors — even useless ones.
- **Adjusted R²** penalizes for extra predictors, giving a fairer comparison between models.
- Each predictor gets its own coefficient and p-value to judge whether it's meaningfully contributing.
