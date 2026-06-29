
---

tags: [machine-learning, bias, variance, overfitting, underfitting, model-evaluation] source: https://youtu.be/EuBBz3bI-aA author: StatQuest with Josh Starmer related:

- "[[Cross Validation]]"
- "[[Decision Trees]]"
- "[[Regularization]]"

---

# 🎯 Bias and Variance — Machine Learning Fundamentals

## ⚡ One-Line Summary

**Bias** = your model is too simple and misses the pattern. **Variance** = your model is too complex and memorises the noise. Good ML is finding the sweet spot between the two.

---

## 🧠 The Core Idea

When you train a model, two things can go wrong:

|Problem|What's happening|Also called|
|---|---|---|
|**High Bias**|Model is too simple — can't fit the data|Underfitting|
|**High Variance**|Model is too complex — fits training noise|Overfitting|

Think of it like drawing a line through a scatterplot:

- A flat horizontal line has **high bias** — it ignores the actual trend
- A squiggly line through every single point has **high variance** — it follows noise, not signal

---

## 📌 Bias — What It Means

**Bias** measures how far your model's average predictions are from the correct answer.

- A **high-bias** model makes the same type of mistake over and over
- It performs poorly on **both** training data and new test data
- The model hasn't learned enough from the data

> 🔑 **Simple rule:** If your model does badly even on the data it trained on → high bias

---

## 📌 Variance — What It Means

**Variance** measures how much your model's predictions change when you train it on different data samples.

- A **high-variance** model performs great on training data but terribly on new data
- It's memorised the specific training examples rather than learning the general pattern

> 🔑 **Simple rule:** If your model does great on training data but badly on test data → high variance

---

## ⚖️ The Tradeoff

You can't eliminate both at the same time — reducing one tends to increase the other:

```
Complexity  →  →  →  →  →  →  →  →  →  →

Simple model                    Complex model
[High Bias]  ←──── Sweet Spot ────►  [High Variance]
[Low Variance]                        [Low Bias]
(Underfitting)                        (Overfitting)
```

**Goal:** find the model complexity where total error (bias² + variance + irreducible noise) is minimised.

---

## 🔍 How to Spot Each Problem

### Diagnosing with Train vs Test Error

|Scenario|Training Error|Test Error|Diagnosis|
|---|---|---|---|
|Both high|High|High|High Bias (underfitting)|
|Big gap|Low|High|High Variance (overfitting)|
|Both low|Low|Low|✅ Good fit|

---

## 🛠️ How to Fix Each

### Fixing High Bias

- Use a more complex model (more layers, more features, higher polynomial degree)
- Add more relevant features
- Reduce regularisation strength

### Fixing High Variance

- Get **more training data** (most reliable fix)
- Use **regularisation** (Ridge, Lasso, Dropout)
- Use a simpler model
- Use **cross-validation** to detect overfitting early
- **Ensemble methods** (Random Forests, Bagging) — average out variance across many models

---

## 💡 Concrete Examples

|Model|Typical Problem|
|---|---|
|Linear regression on curved data|High Bias|
|Very deep decision tree on small dataset|High Variance|
|Random Forest|Reduces Variance vs single tree|
|Ridge/Lasso regression|Reduces Variance via regularisation|

---

## ❓ Quick Recall Questions

1. Your model gets 99% accuracy on training data and 62% on test data. What's the problem?
2. Your model gets 65% on training and 63% on test. What's the problem?
3. Name two techniques that specifically reduce variance.
4. Why does getting more data help with variance but not with bias?

---

## 🔗 Related Notes

- [[Cross Validation]]
- [[Decision Trees]]
- [[Regularization]]
- [[Random Forests]]