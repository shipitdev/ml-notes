# 📊 Machine Learning: Performance Metrics for Classification Problems (Part 1)

---

## 📌 1. The Big Picture: Class Labels vs. Production Thresholds

When evaluating classification algorithms in machine learning, measuring structural model validity requires a deep understanding of output types. Classification tasks generally return results via two approaches:

1. **Direct Class Labels:** Assigns data entries directly into absolute discrete categories (e.g., `Class A` vs. `Class B`).
2. **Class Probabilities:** Returns a raw continuous float between `0.0` and `1.0` representing the probability that an observation belongs to the positive class.

### The Role of the Decision Threshold

By default, most library implementations (such as Scikit-Learn's `.predict()`) use a hardbound **Decision Threshold of 0.5**:

- Probability > 0.5 → **Positive Class (1)**
- Probability ≤ 0.5 → **Negative Class (0)**

In high-stakes industries like healthcare or finance, relying blindly on a static `0.5` boundary can cause catastrophic failures. For instance, when diagnosing cancer, the threshold should be manually lowered (e.g., to `0.2` or `0.3`) to catch potential positive cases early, even if it introduces more false alarms. Adjusting this threshold directly impacts your downstream performance metrics.

---

## 📌 2. The Trap of Accuracy: Balanced vs. Imbalanced Datasets

A common mistake is selecting an inappropriate evaluation metric for the underlying dataset distribution.

- **Balanced Datasets:** Target classes exhibit relatively uniform proportions (e.g., 50:50, 60:40, or up to 70:30). Under these conditions, **Accuracy** serves as an effective metric.
- **Imbalanced Datasets:** Target classes display a stark asymmetrical distribution (e.g., 80:20 or 90:10). In extreme scenarios like credit card fraud detection or rare disease screening, the minority class might make up less than 1% of records.

### The Core Structural Vulnerability: Blind Majority Guessing

If you train a model on **900 healthy patient records** and **100 cancer patient records**, a broken model can simply predict "healthy" for every single row:

$$\text{Accuracy} = \frac{900}{1000} = 90% 
$$

Despite a **90% accuracy score**, this model is completely useless — it misses 100% of the active cancer cases. For imbalanced scenarios, accuracy must be replaced by **Recall**, **Precision**, and the **F-Beta Score**.

---

## 📌 3. The Anatomy of a Confusion Matrix

To break down classification errors systematically, we construct a 2x2 table known as the **Confusion Matrix**. This matrix contrasts real-world ground truths against model prediction vectors.

### The Confusion Matrix Layout

||** Predicted Positive (1)**|**Predicted Negative (0)**|
|:--|:-:|:-:|
|**Actual Positive (1)**|True Positive (TP)|False Negative (FN)|
|**Actual Negative (0)**|False Positive (FP)|True Negative (TN)|

### The 4 Core Quadrant Definitions

|Quadrant|Actual|Predicted|Error Type|
|:--|:-:|:-:|:--|
|**True Positive (TP)**|1|1|Correct|
|**True Negative (TN)**|0|0|Correct|
|**False Positive (FP)**|0|1|Type 1 Error|
|**False Negative (FN)**|1|0|Type 2 Error|

---

## 📌 4. Core Metric Explanations and Formulas

---

### Metric 1: Accuracy

**Core Definition:** The overall ratio of correct predictions to total records evaluated.

$$\text{Accuracy} = \frac{TP + TN}{TP + FP + FN + TN}$$

---

### Metric 2: Recall (Sensitivity / True Positive Rate)

**Core Definition:** Measures the proportion of actual positive records the model correctly identified. It answers: _Out of all real positive cases, how many did we capture?_

$$\text{Recall} = \frac{TP}{TP + FN}$$

---

### Metric 3: Precision (Positive Predictive Value)

**Core Definition:** Measures the purity of the positive predictions made by the model. It answers: _Out of all cases the model flagged as positive, how many are actually true positives?_

$$\text{Precision} = \frac{TP}{TP + FP}$$

---

## 📌 5. Real-World Case Studies: Precision vs. Recall Optimization

---

### Use-Case A: Email Spam Detection — Optimize for Precision

- **Positive Class (1):** Spam Email
- **Negative Class (0):** Legitimate Inbox Email (Ham)

|Error Type|What Happens|Consequence|
|:--|:--|:--|
|**False Negative (FN)**|A spam email lands in your inbox.|Minor annoyance for the user.|
|**False Positive (FP)**|A critical business email is flagged as spam.|Disastrous — user misses vital information.|

**Engineering Decision:** Minimize False Positives at all costs → **Optimize for Precision**

---

### Use-Case B: Medical Cancer Screening — Optimize for Recall

- **Positive Class (1):** Malignant Tumor Present
- **Negative Class (0):** Healthy / Benign

|Error Type|What Happens|Consequence|
|:--|:--|:--|
|**False Positive (FP)**|A healthy patient is flagged as having cancer.|Short-term stress; cleared during follow-up testing.|
|**False Negative (FN)**|A patient with a tumor is told they are healthy.|Fatal — patient misses the window for life-saving treatment.|

**Engineering Decision:** Minimize False Negatives at all costs → **Optimize for Recall**

---

## 📌 6. Advanced Balancing: The F-Beta Score Framework

When a business application requires balancing both performance axes simultaneously, we need a unified metric: the **F-Beta Score**.

### The Generalized F-Beta Equation

$$F_{\beta} = (1 + \beta^2) \times \frac{\text{Precision} \times \text{Recall}}{(\beta^2 \times \text{Precision}) + \text{Recall}}$$

The parameter **Beta** controls the weight distribution between the two metrics:

---

#### 1. Beta = 1.0 — The Standard F1 Score

Setting Beta = 1 reduces the formula to the standard harmonic mean:

$$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

**Application:** Used when False Positives and False Negatives carry equal operational weight.

---

#### 2. Beta < 1.0 — The F0.5 Score

Squashing Beta down to `0.5` reduces the mathematical weight of Recall.

**Application:** Used when **Precision is more critical than Recall** (e.g., Spam Detection).

---

#### 3. Beta > 1.0 — The F2 Score

Increasing Beta to `2.0` gives more mathematical weight to Recall.

**Application:** Used when **Recall is more critical than Precision** (e.g., medical diagnoses, system failure alerts).

---

## 📌 7. Strategic Optimization Matrix

|Metric Framework|Best Dataset Type|Primary Error Minimised|Industry Use-Case|
|:--|:--|:--|:--|
|**Global Accuracy**|Balanced class arrays|Minimises all errors uniformly.|Standard demographic profiling.|
|**Isolate Precision**|Imbalanced class arrays|Minimises False Positives (Type 1).|Spam detection, fraud alerts.|
|**Isolate Recall**|Imbalanced class arrays|Minimises False Negatives (Type 2).|Healthcare screenings, failure alerts.|
|**Harmonic F1 Score**|Balanced error demands|Pairs Precision and Recall equally.|General classification tuning.|

---

## 🔗 Related Notes

- [[Statistics — Practical Implementation of Hypothesis Testing]]
- [[Statistical Inference — Choosing the Right Hypothesis Test]]
- [[Data Science — Unified EDA, Feature Engineering & Feature Selection Roadmap]]
- [[Imbalanced Datasets and SMOTE]]
- [[Overfitting and Data Leakage]]

---

## 📹 Recommended Video Resource

[Krish Naik — Performance Metrics for Classification Problems Part 1](https://www.youtube.com/watch?v=aWAnNHXIKww)

---

_Tags: #machine-learning #classification #confusion-matrix #accuracy #precision #recall #f1-score #f-beta #imbalanced-data #decision-threshold #type1-error #type2-error #data-science_