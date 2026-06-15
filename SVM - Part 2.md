
---

tags: [machine-learning, classification, svm, support-vector-machine, optimization, hinge-loss, soft-margin, regularization] source: https://www.youtube.com/watch?v=Js3GLb1xPhc related:

- "[[Support Vector Machines]]"
- "[[Logistic Regression]]"
- "[[ROC and AUC Curve]]"
- "[[GridSearchCV and RandomizedSearchCV]]"

---

# 📐 SVM — Mathematical Intuition (Part 2)

## ⚡ TL;DR

Part 1 established _what_ SVM does — maximise the margin. Part 2 derives _how_. The margin width equals $\frac{2}{|\mathbf{W}|}$, so maximising it means **minimising** $|\mathbf{W}|$. Real-world data is never perfectly separable, so a **Slack Variable** $\zeta_i$ is introduced to allow controlled misclassification, penalised by the **C parameter**. The full objective: minimise weight magnitude _and_ total slack penalty simultaneously.

---

## 📌 From Logistic Regression to SVM

Logistic Regression finds _a_ separating hyperplane. SVM finds the _best_ one by enforcing hard margin constraints:

$$\mathbf{W}^T\mathbf{X} + B = 0 \quad \text{(hyperplane)}$$

$$\mathbf{W}^T\mathbf{X} + B \ge +1 \quad \text{(positive class boundary)}$$ $$\mathbf{W}^T\mathbf{X} + B \le -1 \quad \text{(negative class boundary)}$$

By using $+1$ and $-1$ (not $1$ and $0$), the margin becomes a **symmetric** zone of width $\frac{2}{|\mathbf{W}|}$ centred on the hyperplane. Support Vectors are the only points that sit exactly on the $\pm 1$ planes — all other points are irrelevant to the boundary.

---

## 📐 Mathematical Optimization Engine

### Margin Distance Formula

$$\text{Margin} = \frac{2}{|\mathbf{W}|}$$

Maximising margin → minimising $|\mathbf{W}|$. Since $|\mathbf{W}|$ is non-differentiable at zero, we minimise $\frac{1}{2}|\mathbf{W}|^2$ instead (same solution, easier calculus).

### Primal Objective Function (Soft Margin)

$$\min_{\mathbf{W},, B} \left( \frac{1}{2}|\mathbf{W}|^2 + C \sum_{i=1}^{n} \zeta_i \right)$$

|Term|Name|Role|
|---|---|---|
|$\frac{1}{2}\|\mathbf{W}\|^2$|Weight regularisation|Maximises margin width|
|$\zeta_i$|Slack variable|Distance of misclassified point across margin|
|$C$|Regularisation constant|Penalises slack — controls overfitting vs. margin width|

---

## 🔧 The C Parameter: Soft Margin Trade-Off

|C Setting|Behaviour|Risk|
|---|---|---|
|**Very high C**|Penalises all errors heavily → tiny margin, forces near-perfect training fit|Overfitting|
|**Low C**|Tolerates some misclassifications → wide, stable margin|Slightly higher training error|
|**Tuned C**|Balanced generalisation on unseen data|Use `GridSearchCV` to find it|

> 💡 Never hard-code $C$. Always cross-validate. Cranking $C$ to maximise training accuracy routinely destroys generalisation on live data.

---

## 🧮 Hinge Loss Connection

The slack penalty $C \sum \zeta_i$ is equivalent to the **Hinge Loss**: $$\mathcal{L}_{\text{hinge}} = \max(0,; 1 - y_i(\mathbf{W}^T X_i + B))$$

- If a point is correctly classified with margin ≥ 1: loss = 0
- If a point violates the margin: loss > 0, scaled by how far it penetrates

Scikit-Learn uses `LibLinear` or `SMO` solvers to minimise this objective numerically — you never need to solve it by hand.

---

## 💻 Python Pipeline

```python
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

model = make_pipeline(
    StandardScaler(),
    SVC(kernel='linear', C=1.0)
)
model.fit(X_train, y_train)

# Tune C via cross-validation
param_grid = {'svc__C': [0.01, 0.1, 1, 10, 100]}
gs = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
gs.fit(X_train, y_train)
print(gs.best_params_)
```

---

## 📊 SVM Optimization Terms Reference

|Term|Formula|Engineering Role|
|---|---|---|
|**Hyperplane**|$\mathbf{W}^T\mathbf{X} + B = 0$|Final decision boundary|
|**Margin**|$\frac{2}{\|\mathbf{W}\|}$|Buffer zone between classes|
|**Support Vectors**|Points on $\pm 1$ planes|Sole definers of the boundary|
|**Slack ($\zeta_i$)**|Error distance of misclassified points|Allows noise in non-separable data|
|**C**|Penalty on $\sum \zeta_i$|Generalisation / overfitting dial|

---

## ⚠️ Gotchas

> [!warning] Don't Forget the Bias Term B The hyperplane doesn't always pass through the origin. $B$ shifts the entire margin plane to correctly centre it between the support vectors. Assuming $B = 0$ misplaces the boundary on non-origin-centred data.

> [!warning] High C ≠ Better Model Forcing zero training error by inflating $C$ shrinks the margin to near-zero, making the model brittle to any variation in production data. Regularisation exists for a reason.

> [!warning] Primal vs Dual Formulation The primal form above minimises $|\mathbf{W}|^2$. In practice, Scikit-Learn often solves the **dual form** (in terms of Lagrange multipliers), which is what enables the Kernel Trick. The two forms yield identical solutions.

---

## ❓ Active Recall Questions

1. Why does maximising the margin $\frac{2}{|\mathbf{W}|}$ require _minimising_ $|\mathbf{W}|^2$ rather than $|\mathbf{W}|$ directly?
2. Why does SVM use $\pm 1$ class labels instead of $1$ and $0$?
3. How does the slack variable $\zeta_i$ differ from a standard classification error — what additional geometric information does it carry?
4. What is the relationship between SVM's Hinge Loss and the C parameter?

---

## 🔗 Related Notes

- [[Support Vector Machines]]
- [[Logistic Regression]]
- [[ROC and AUC Curve]]
- [[GridSearchCV and RandomizedSearchCV]]
- [[Cross Validation]]