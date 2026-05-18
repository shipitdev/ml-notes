# 6. Machine Learning Fundamentals: Bias and Variance
🔗 https://www.youtube.com/watch?v=EuBBz3bI-aA

## The Big Idea
Every model makes errors for one of two fundamental reasons: it's **too simple** to capture the real pattern (Bias), or it's **too sensitive** to the specific quirks of the training data (Variance). A good model finds the balance between the two.

## Flow of the Video

### 1. Setting up the example
- Simple example: we have data points showing a wavy relationship between some input (e.g., dosage of a drug) and an outcome (e.g., effectiveness).
- We want to fit a line/curve that predicts effectiveness from dosage.

### 2. Option A: a straight line (too simple)
- Fit a straight line through the wavy data.
- It consistently misses the curves — it's **not flexible enough** to capture the true relationship.
- This systematic, consistent inability to capture the pattern is called **Bias**.
- **High Bias** = the model makes strong assumptions (like "the relationship must be a straight line") that don't match reality → **underfitting**.

### 3. Option B: a super wiggly curve that touches every single point (too flexible)
- Fit an extremely flexible curve that passes through every training data point exactly.
- On the training data, it looks perfect (zero error!).
- But if you gave this model a **new/different training set** from the same real-world process, the wiggly curve fit would look completely different each time — it's chasing noise, not the real signal.
- This sensitivity to the specific training data is called **Variance**.
- **High Variance** = the model changes a lot depending on which exact training data it saw → **overfitting**.

### 4. Visualizing the difference
- Imagine fitting your model on many different training sets (drawn from the same real-world process) and overlaying all the resulting curves:
  - **High Bias model**: all the fitted lines look similar to each other, but they're all similarly *wrong* (consistently missing the true pattern).
  - **High Variance model**: the fitted curves look wildly different from each other every time (inconsistent), even though each one fits *its own* training data very well.

### 5. The trade-off
- As you make a model more flexible (to reduce bias), it tends to pick up more noise (increasing variance) — and vice versa.
- The goal is to find the **sweet spot**: a model flexible enough to capture the real trend, but not so flexible that it starts memorizing noise.
- This is why we use **Cross Validation** (video #2!) — testing on unseen data reveals whether a model is overfitting (great on training, bad on testing = high variance) or underfitting (mediocre on both = high bias).

## Key Takeaways (Quick Recall)
- **Bias** = error from an overly simple model that can't capture the true pattern → underfitting.
- **Variance** = error from an overly flexible model that reacts too much to the specific training data → overfitting.
- Ideal model: low bias AND low variance — but there's usually a trade-off between the two.
- Cross validation (comparing training vs. testing performance) is how we detect which problem we have.
