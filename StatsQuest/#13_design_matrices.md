# 13. Design Matrices For Linear Models, Clearly Explained!!!
🔗 https://www.youtube.com/watch?v=CqLGvwi-5Pc

## The Big Idea
A "design matrix" is just the organized table of 0s, 1s, and numeric values that we feed into a linear model so it can be solved with matrix algebra. Understanding this table is the key to understanding how software actually fits regressions, t-tests, and ANOVAs behind the scenes.

## Flow of the Video

### 1. What a design matrix looks like
- Every row = one observation (one data point/person).
- Every column = one predictor (including a special column of all 1s for the intercept).
- Simple example: predicting weight from height for 3 people:

![Predicting Weight from Height Design Matrix](./images/design_matrix_height.svg)

- This matrix, multiplied by a column of unknown coefficients (intercept, slope), and compared against the actual weights, is exactly the system of equations that least-squares regression solves.

### 2. Design matrix for categorical variables (dummy coding)
- Recall from the last video: groups get turned into 0/1 dummy columns.
- Simple example: 3 diets (A, B, C), with Diet A as baseline:

![Dummy Coded Categorical Design Matrix](./images/design_matrix_dummy.svg)

- This is literally the design matrix used to run the "ANOVA-as-regression" from the previous video.

### 3. Combining continuous and categorical predictors
- A design matrix can mix numeric columns (like height) with dummy columns (like diet group) in the same table — this is how models handle both types of variables together (this generalizes into things like ANCOVA).

### 4. Why the design matrix matters
- Once your data is arranged into this matrix form, fitting a linear model becomes pure matrix algebra:
```
Coefficients = (XᵀX)⁻¹ Xᵀy
```
  (X = design matrix, y = the outcome variable column). You don't need to memorize this formula — the point is that **the design matrix is what makes this equation solvable** for any type of regression, t-test, or ANOVA, all using the exact same underlying method.

![The Design Matrix Equation: Y = X Beta + e](./images/design_matrix_eq.svg)

- Different ways of coding categorical variables (e.g., choosing a different "baseline" group, or using different coding schemes like "sum coding") change the design matrix, and therefore change how you interpret the coefficients — but they don't change the model's overall predictions.

## Key Takeaways (Quick Recall)
- Design matrix = table of predictor values (rows = observations, columns = predictors, including a column of 1s for the intercept).
- Categorical variables become 0/1 dummy columns within the design matrix.
- Continuous and dummy columns can be mixed in the same design matrix.
- The whole point: once data is in this matrix format, the *same* matrix algebra solves regression, t-tests, and ANOVA alike.
