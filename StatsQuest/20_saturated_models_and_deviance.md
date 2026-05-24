# 20. Saturated Models and Deviance
🔗 https://www.youtube.com/watch?v=C4N3_XJJ-jU

## The Big Idea
A "Saturated Model" is a theoretical, perfect model that fits the training data exactly (one parameter per data point) — it's not useful for prediction, but it's a crucial *reference point* for measuring how good our actual, simpler model is. That comparison is called **Deviance**.

## Flow of the Video

### 1. What is a Saturated Model?
- Simple example: instead of a single logistic regression line for all patients, imagine giving **every single patient their own personal parameter**, perfectly matching their observed outcome.
- This model has as many parameters as data points → it fits the training data **perfectly** (log-likelihood is as high as mathematically possible).
- It's obviously overfit and useless for predicting *new* patients — but it serves as the theoretical "gold standard" ceiling for log-likelihood.

### 2. Deviance: comparing our real model to the saturated model
```
Deviance = -2 × ( LL(fitted model) - LL(saturated model) )
```
- This measures how far our actual, useful model falls short of the theoretical perfect fit.
- **Lower deviance = better fit** (our model's log-likelihood is closer to the saturated model's maximum possible log-likelihood).
- This is directly analogous to "Sum of Squared Residuals" in linear regression — deviance plays the same "how much is left unexplained" role, just built from log-likelihoods instead of squared errors.

### 3. Two useful versions of deviance
- **Null Deviance**: deviance of the simplest possible model (just an intercept, i.e., ignoring all predictors — same as the "overall probability" baseline from the last video) compared to the saturated model.
- **Residual Deviance**: deviance of our actual fitted model (with predictors included) compared to the saturated model.
- Comparing Null Deviance to Residual Deviance tells us how much our predictors actually improved the fit — the bigger the drop from Null to Residual Deviance, the more our predictors are helping.

### 4. Using Deviance for a formal significance test
- The **difference** between Null Deviance and Residual Deviance follows a **Chi-squared distribution** (with degrees of freedom equal to the number of predictors added).
- This gives us a p-value: are our predictors improving the model significantly more than random chance would suggest?
- This is the same underlying test introduced conceptually in the previous video (Pt 3: R-squared and p-value) — Deviance is just the formal, standard way statistical software actually reports and computes it.

### 5. Deviance as a general-purpose "goodness of fit" tool
- Because Deviance is built from log-likelihoods (not squared residuals), it generalizes beyond logistic regression to any model fit via Maximum Likelihood — this is why you'll see "deviance" reported in many types of statistical models, not just logistic regression.

## Key Takeaways (Quick Recall)
- Saturated Model = a theoretical "perfect fit" model (one parameter per data point) used only as a reference ceiling, never for real predictions.
- Deviance = -2 × (LL(our model) − LL(saturated model)) → lower deviance = better fit, plays the same role as SSR does in linear regression.
- Null Deviance = deviance of the intercept-only model; Residual Deviance = deviance of our full fitted model.
- The drop from Null to Residual Deviance, tested via a Chi-squared distribution, tells us if our predictors are significantly helpful.
- Deviance generalizes the "goodness of fit" idea to any Maximum-Likelihood-based model, not just logistic regression.
