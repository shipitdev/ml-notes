
---

tags: [machine-learning, classification, naive-bayes, bayes-theorem, probabilistic-models, text-classification] source: https://www.youtube.com/watch?v=jS1CKhALUBQ related:

- "[[Logistic Regression]]"
- "[[Cross Validation]]"
- "[[Decision Trees]]"

---

# 🧠 Naïve Bayes Classifier — Probabilistic Classification

## ⚡ TL;DR

**Naïve Bayes** predicts a class by multiplying the prior probability of that class by the individual likelihoods of each feature occurring within it. The "Naïve" part: it assumes all features are **independent** of each other — almost never true in reality, but this simplification makes the math fast and highly effective for high-dimensional text and log data.

---

## 📌 The Big Picture: Bayes Theorem → Classification

Goal: given features $X_1, X_2, \dots, X_n$, find the most probable class $Y$.

### Core Formula

$$P(Y \mid X) = \frac{P(X \mid Y) \cdot P(Y)}{P(X)}$$

### Expanded Multi-Feature Form

$$P(Y \mid X_1, X_2, \dots, X_n) \propto P(Y) \cdot \prod_{i=1}^{n} P(X_i \mid Y)$$

|Term|Name|Meaning|
|---|---|---|
|$P(Y \mid X)$|**Posterior**|Probability of class $Y$ given features $X$|
|$P(X \mid Y)$|**Likelihood**|Probability of seeing feature $X$ given class $Y$|
|$P(Y)$|**Prior**|Base rate of class $Y$ in the training data|
|$P(X)$|**Evidence**|Normalising constant — same for all classes, so often dropped|

---

## 🧮 Numerical Walkthrough: "Play Tennis" Case Study

**Dataset:** 14 records. Features: `Outlook` (Sunny/Overcast/Rainy) + `Temperature` (Hot/Mild/Cool). Target: Play (Yes/No).

**Test case:** Outlook = Sunny, Temperature = Hot → Predict?

### Step 1 — Priors

$$P(\text{Yes}) = \frac{9}{14}, \quad P(\text{No}) = \frac{5}{14}$$

### Step 2 — Likelihoods

|Feature|Condition|$P(\text{cond} \mid \text{Yes})$|$P(\text{cond} \mid \text{No})$|
|---|---|---|---|
|Outlook|Sunny|$2/9$|$3/5$|
|Temperature|Hot|$2/9$|$2/5$|

### Step 3 — Proportional Scores

$$\text{Score(Yes)} = \frac{2}{9} \cdot \frac{2}{9} \cdot \frac{9}{14} \approx 0.031$$ $$\text{Score(No)} = \frac{3}{5} \cdot \frac{2}{5} \cdot \frac{5}{14} \approx 0.086$$

### Step 4 — Normalise

$$P(\text{Yes} \mid \text{Today}) = \frac{0.031}{0.031 + 0.086} \approx 0.27$$ $$P(\text{No} \mid \text{Today}) = 1 - 0.27 = 0.73$$

**Result:** $0.73 > 0.27$ → model predicts **Not Play** ❌🎾

---

## 💻 Python Pipeline

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# GaussianNB: for continuous features following a normal distribution
# MultinomialNB: for discrete counts (e.g., word frequencies in text)
# BernoulliNB: for binary features (word present/absent)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Get class probabilities (not just the predicted label)
y_proba = model.predict_proba(X_test)
```

---

## 🌍 Real-World Use Cases

### Simple — Spam Filtering

Predict Spam / Not Spam by computing the likelihood of flagged words (`win`, `money`, `urgent`) given each class.

### Advanced — Real-Time Log Triage

Stream server logs and classify tokens into `Normal` / `Warning` / `Critical` silos using token frequency likelihoods.

> ⚠️ **Stress Point:** When input features are heavily correlated (e.g., `New` almost always followed by `York`), Naïve Bayes underestimates their combined weight. For dense feature dependencies, switch to Logistic Regression or gradient-boosted models.

---

## ⚠️ Gotchas

> [!warning] Zero Frequency Problem If a feature value never appears in training for a given class, $P(X_i \mid Y) = 0$, which zeroes out the entire posterior. Fix: **Laplace Smoothing** — add 1 to all counts before computing likelihoods so no probability ever hits zero.

> [!warning] Numerical Underflow Multiplying many small probabilities together collapses toward zero and triggers floating-point errors. Fix: convert to **log-probabilities** — multiplication becomes addition: $$\log P(Y \mid X) \propto \log P(Y) + \sum_{i=1}^{n} \log P(X_i \mid Y)$$ Scikit-Learn handles this internally.

> [!warning] Independence Assumption Features are almost never truly independent in real data. Naïve Bayes still works well in practice (especially NLP), but accuracy degrades on datasets with strongly correlated features.

---

## 📊 Which Naïve Bayes Variant to Use?

|Variant|Feature Type|Typical Use Case|
|---|---|---|
|`GaussianNB`|Continuous, normally distributed|Medical diagnostics, sensor data|
|`MultinomialNB`|Discrete counts|Word frequency in text (spam, topic classification)|
|`BernoulliNB`|Binary (0/1)|Word presence/absence in short documents|

---

## ❓ Active Recall Questions

1. Why does the independence assumption actually _help_ Naïve Bayes scale in high-dimensional datasets?
2. How does Laplace Smoothing prevent a single unseen word from zeroing out the entire posterior probability?
3. Why is Naïve Bayes considered a **generative** model, while Logistic Regression is **discriminative**?
4. How would you handle a test record containing a feature value that never appeared in training?

---

## 🔗 Related Notes

- [[Logistic Regression]]
- [[Cross Validation]]
- [[Decision Trees]]
- [[Precision, Recall, F1 Score]]