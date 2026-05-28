---
tags:
  - machine-learning
  - regression
  - regularization
  - feature-engineering
cssclasses:
  - Ridge & Lasso Regression Indepth Intuition
aliases:
---

# Regularization: Ridge vs. Lasso Regression

When a machine learning model suffers from **high error variance (overfitting)**, its weight coefficients ($\beta$) have typically grown too large and complex because they are tuning themselves to random statistical noise in the training set. 

**Regularization** fixes this by adding a penalty term directly to the loss function, forcing the model to keep its weights small and controlled.

---

## 1. The Core Formulas

### Ordinary Least Squares (OLS) Baseline
Before regularization, standard linear regression only cares about minimizing the squared errors:
$$\text{Loss}_{OLS} = \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2$$

### Ridge Regression ($L_2$ Regularization)
Adds a penalty proportional to the **squared magnitude** of the coefficients:
$$\text{Loss}_{Ridge} = \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2 + \alpha \sum_{j=1}^{m} \beta_j^2$$

### Lasso Regression ($L_1$ Regularization)
Adds a penalty proportional to the **absolute value** of the coefficients:
$$\text{Loss}_{Lasso} = \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2 + \alpha \sum_{j=1}^{m} |\beta_j|$$

> [!INFO] What is $\alpha$?
> $\alpha$ (sometimes written as $\lambda$) is a hyperparameter that controls how severely you penalize the model. 
> * If $\alpha = 0$, the penalty is disabled, and you are back to standard **OLS Regression**.
> * If $\alpha \rightarrow \infty$, the penalty dominates, forcing all weights down toward zero, resulting in a flat line (high bias).

---

## 2. The Geometric "Why" (Circles vs. Diamonds)

The best way to understand why Ridge and Lasso behave differently is to look at their geometry in a 2D coordinate space where the axes represent two feature weights ($\beta_1$ and $\beta_2$).

[Image of Ridge vs Lasso geometric constraints showing circular and diamond boundaries]

### Ridge ($L_2$) Geometry $\rightarrow$ The Smooth Circle
Because the Ridge penalty squares the weights ($\beta_1^2 + \beta_2^2 \le t$), its constraint boundary forms a **perfect circle** (or hypersphere in higher dimensions). 

As the OLS loss contours expand outward to find the best fit, they hit the smooth edge of this circle. Because there are no sharp corners, the intersection point almost never lands exactly on an axis line. Consequently, the weights are shrunken uniformly and get incredibly close to zero, but they **never become exactly $0.0$**. Every feature stays alive.

### Lasso ($L_1$) Geometry $\rightarrow$ The Pointy Diamond
Because the Lasso penalty uses absolute values ($|\beta_1| + |\beta_2| \le t$), its constraint boundary forms a **sharp, pointy diamond** (or polyhedron). 

When the circular OLS loss contours expand, they are mathematically much more likely to clip one of the sharp corners of the diamond. Because the corners of a diamond sit directly on the axis lines, one of