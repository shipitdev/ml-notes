# 📊 Machine Learning: Logistic Regression — Geometric & Optimization Foundations (Part 2)

---

## 📌 1. The Big Picture: From Visual Curves to Geometric Optimization

In Part 1, we established why standard OLS linear equations fail when applied to discrete binary targets, and introduced the **Sigmoid Function** as an activation layer to squash unconstrained scores into a valid probability range between 0 and 1.

Now we look at the underlying geometric framework of **Logistic Regression** to understand how the algorithm optimises its parameters. Rather than viewing optimisation as a black-box concept, we define it geometrically: finding an optimal decision boundary (hyperplane) in a multi-dimensional space that maximises correct class separation while insulating the model from the distorting effects of extreme data outliers.

---

## 📌 2. The Geometric Architecture of the Decision Hyperplane

To separate two distinct classes in space, Logistic Regression defines a linear boundary known as a **Hyperplane**.

### The Plane Equation

In a two-dimensional feature space, this boundary is a simple line. In higher dimensions, it becomes a multi-dimensional plane defined by the vector equation:

$$w^T x + b = 0$$

Where:

- **w^T** = The transposed weight vector coefficients perpendicular to the hyperplane (defining its orientation).
- **x** = The input feature vector instance coordinates.
- **b** = The bias scalar coefficient (defining the intercept/spatial offset from the origin).

### Directed Distance from the Boundary

For any given observation point x_i in our dataset, we can calculate its shortest directed distance to the decision boundary. Assuming a unit-normalised weight vector, the algebraic dot product provides a scale proportional to that geometric distance:

$$\text{Distance} \propto w^T x_i$$

- If a point lies on the **positive side** of the plane: w^T x_i > 0
- If a point lies on the **negative side** of the plane: w^T x_i < 0
- If a point sits **directly on the boundary**: w^T x_i = 0

---

## 📌 3. The Sign Product Matrix Rule: y_i × (w^T x_i)

To automate parameter optimisation, the model needs a way to evaluate whether its current hyperplane correctly separates the classes. This is achieved using the **Sign Product Matrix rule** — multiplying the actual ground-truth label (y_i) by the calculated directed distance (w^T x_i).

For this geometric framework, binary target labels use an explicit sign scheme: **y_i ∈ {+1, −1}** (where +1 represents the positive class and −1 represents the negative class).

---

### The 4 Error and Success States

|State|Ground Truth (y_i)|Hyperplane Side (w^T x_i)|Sign Product|Result|
|:--|:-:|:--|:-:|:--|
|**Correct Positive**|+1|Positive side (> 0)|(+1) × (+ve)|**> 0** — Correct|
|**Correct Negative**|−1|Negative side (< 0)|(−1) × (−ve)|**> 0** — Correct|
|**Misclassified Positive**|+1|Negative side (< 0)|(+1) × (−ve)|**< 0** — Error|
|**Misclassified Negative**|−1|Positive side (> 0)|(−1) × (+ve)|**< 0** — Error|

### The Raw Optimisation Objective

This sign product rule reveals a key insight: **for any correctly classified point, the product y_i(w^T x_i) is always positive.** For any misclassified point, it is always negative.

Therefore, our initial optimisation goal is to find a weight vector w that maximises the sum of these products across all training instances:

$$\max_w \sum_{i=1}^{n} y_i (w^T x_i)$$

---

## 📌 4. The Outlier Vulnerability Trap: Why Raw Distance Optimisation Fails

While maximising the raw sum of signed distances seems mathematically sound, using this objective function directly makes the model **highly vulnerable to extreme outliers**.

### Outlier Scenario Analysis

Imagine evaluating a potential decision boundary against a dataset of 6 points:

- **Points 1 to 5 (Local Clean Clusters):** These 5 instances are misclassified by the current plane. They sit close to the boundary on the wrong side, each returning a negative sign product of **−2**.
- **Point 6 (Extreme Positive Outlier):** This single point sits incredibly far to the right, deep in the positive region. It is correctly classified and returns a massive positive sign product of **+500**.

### Calculating the Global Cost Sum

$$\text{Global Sum} = (-2) + (-2) + (-2) + (-2) + (-2) + 500 = +490$$

**The Problem:** The optimisation engine sees a massive positive score of +490 and accepts this hyperplane as highly optimal. In reality, this boundary is terrible — it misclassifies 5 valid data points just to correctly capture a single extreme outlier far away.

---

## 📌 5. The Sigmoid Solution: Neutralising Outlier Distortion

To prevent extreme distances from dominating the objective function, Logistic Regression passes its raw distance metrics through the non-linear **Sigmoid Function** before calculating the sum.

```
[ Raw Distance Vector: z = wᵀx ]
              │
              ▼
[ Sigmoid Function: 1 / (1 + e⁻ᶻ) ]
              │
              ▼
[ Squashed Probabilities Bounded Strictly in [0.0, 1.0] ]
```

### The Transformed Objective Function

$$\max_w \sum_{i=1}^{n} \sigma \left( y_i (w^T x_i) \right)$$

### How Sigmoid Absorbs Outliers

The Sigmoid curve levels out asymptotically, flattening as inputs head toward positive or negative infinity:

- A standard point with a sign product score of **+2** maps to approximately **0.88**.
- An extreme outlier with a sign product score of **+500** is squashed tightly to **1.00**.

This squashing mechanism removes the disproportionate influence of extreme distances. The single outlier's contribution is capped at `1.00`, preventing it from overriding the combined weights of the local data clusters. The final decision boundary remains robust and centred between the classes.

---

## 📌 6. Summary Strategic Taxonomy Framework Matrix

|Optimisation Component|Raw Distance Formulation|Sigmoid Activated Formulation|
|:--|:--|:--|
|**Mathematical Equation**|max_w Σ y_i (w^T x_i)|max_w Σ σ(y_i (w^T x_i))|
|**Output Range**|**Infinite:** Values scale unchecked from −∞ to +∞.|**Constrained:** Values locked within [0.0, 1.0].|
|**Outlier Resilience**|**Highly Vulnerable:** A single extreme distance corrupts the entire boundary.|**Highly Robust:** The asymptotic curve flattens extreme values, neutralising distortion.|
|**Downstream Metric**|Raw geometric distance tracking scores.|Stable log-likelihood posterior probability outcomes.|

---

## 🔗 Related Notes

- [[ML — Logistic Regression In-Depth Intuition Part 1]]
- [[ML — Classification Performance Metrics Part 1]]
- [[Feature Engineering — Feature Scaling & Gaussian Transformations (Day 6)]]
- [[Feature Selection — Removing Highly Correlated Features Using Pearson Correlation]]
- [[Data Science — Unified EDA, Feature Engineering & Feature Selection Roadmap]]

---

## 📹 Recommended Video Resource

[Krish Naik — Logistic Regression In-Depth Intuition Part 2](https://www.youtube.com/watch?v=uFfsSgQgerw)

---

_Tags: #machine-learning #logistic-regression #hyperplane #decision-boundary #sigmoid-function #sign-product #outliers #optimisation #geometric-intuition #gradient-descent #binary-classification #data-science_