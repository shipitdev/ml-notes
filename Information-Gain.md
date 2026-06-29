
---

tags: [machine-learning, decision-trees, classification, gini-impurity, information-gain, entropy] source: https://youtu.be/YtebGVx-Fxw author: StatQuest with Josh Starmer related:

- "[[Bias and Variance]]"
- "[[Random Forests]]"
- "[[Cross Validation]]"

---

# 🌳 Decision Trees — Clearly Explained

## ⚡ One-Line Summary

A Decision Tree asks a series of yes/no questions about your data, splitting it step by step, until it can make a confident prediction. Each question is chosen to create the **purest possible groups**.

---

## 🧠 The Core Idea

A Decision Tree works like a flowchart of questions:

```
                 [Chest Pain?]
                /             \
              Yes              No
           [Blocked          [Good
           Arteries?]        Circulation?]
           /      \           /      \
         Yes       No       Yes       No
        [Heart  [Heart    [Heart   [Heart
        Disease] Disease] Disease] Healthy]
```

At each **node**, the tree asks one question and splits the data into two groups. It keeps splitting until the groups are pure (all one class) or it hits a stopping rule.

---

## 📌 Key Vocabulary

|Term|What it means|
|---|---|
|**Root Node**|The very first question (top of the tree)|
|**Branch**|A yes/no path from a question|
|**Leaf Node**|A final answer (no more splits)|
|**Depth**|How many levels of questions|
|**Impurity**|How mixed a group is (bad = mixed, good = pure)|

---

## 📐 How It Picks the Best Question: Gini Impurity

**Gini Impurity** measures how mixed a group is. Lower is better (purer).

$$\text{Gini} = 1 - \sum_{k} p_k^2$$

where $p_k$ is the proportion of class $k$ in the group.

- **Gini = 0** → perfectly pure (all one class) ✅
- **Gini = 0.5** → maximally mixed (50/50 split) ❌

### Example

A group of 4 people: 3 sick, 1 healthy: $$\text{Gini} = 1 - \left[\left(\frac{3}{4}\right)^2 + \left(\frac{1}{4}\right)^2\right] = 1 - [0.5625 + 0.0625] = 0.375$$

**The tree picks whichever question produces the lowest weighted Gini across both child groups.**

---

## 📐 Alternative: Information Gain (Entropy)

Instead of Gini, some trees use **Entropy** + **Information Gain**:

$$\text{Entropy} = -\sum_k p_k \log_2(p_k)$$

- Entropy = 0 → pure group
- Entropy = 1 → maximally mixed (for binary classes)

$$\text{Information Gain} = \text{Entropy(parent)} - \text{Weighted Average Entropy(children)}$$

The tree picks the split with the **highest information gain**.

> 💡 Gini and Entropy usually give very similar results. Scikit-Learn uses Gini by default (`criterion='gini'`).

---

## 🛑 Stopping & Pruning

A tree that splits until every leaf is pure will **overfit** (high variance). Two ways to fix this:

### Pre-Pruning (set limits before growing)

- `max_depth` — limit how many levels deep the tree goes
- `min_samples_leaf` — require a minimum number of samples in each leaf
- `min_samples_split` — require a minimum to even attempt a split

### Post-Pruning (grow then trim)

- Grow the full tree, then remove branches that don't improve performance on validation data

---

## 💻 Python Quickstart

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

model = DecisionTreeClassifier(
    criterion='gini',   # or 'entropy'
    max_depth=4,        # prevents overfitting
    min_samples_leaf=5,
    random_state=42
)
model.fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test)}")

# Visualise the tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plot_tree(model, feature_names=feature_names, class_names=class_names, filled=True)
plt.show()
```

---

## ⚠️ Gotchas

> [!warning] Overfitting by Default An unconstrained decision tree will grow until every leaf is pure — perfectly memorising training data. Always set `max_depth` or `min_samples_leaf`.

> [!warning] Instability Decision trees are sensitive to small changes in data — a slightly different training set can produce a completely different tree. This is the main motivation for **Random Forests** (averaging many trees reduces this variance).

> [!warning] Doesn't Need Feature Scaling Unlike SVM or KNN, decision trees split on thresholds, not distances. No need for `StandardScaler`.

---

## 📊 Decision Tree vs Random Forest

||Decision Tree|Random Forest|
|---|---|---|
|**Variance**|High (unstable)|Low (averaged)|
|**Interpretability**|✅ Easy to visualise|❌ Black box|
|**Accuracy**|Moderate|Higher|
|**Speed**|Fast|Slower|

---

## ❓ Quick Recall Questions

1. What does a Gini Impurity of 0 mean?
2. How does Information Gain decide which feature to split on?
3. What hyperparameter would you tune first to stop a decision tree from overfitting?
4. Why are random forests better than a single decision tree?

---

## 🔗 Related Notes

- [[Bias and Variance]]
- [[Random Forests]]
- [[Cross Validation]]
- [[Naive Bayes Classifier]]