# 📊 StatQuest #05: Sensitivity, Specificity, Precision & Recall — The Complete Picture

> **Video:** [The Sensitivity, Specificity, Precision, Recall Sing-a-Long!!!](https://www.youtube.com/watch?v=j-EB6RsqjNs)
> **Series:** StatQuest Machine Learning Playlist
> **Core Idea:** Four metrics, each looking at the confusion matrix from a different angle. Together they give you the complete story of your classifier's performance.

---

## 🎯 The Core Intuition (Plain English First)

All four metrics — Sensitivity, Specificity, Precision, Recall — come from the same confusion matrix. But each one asks a **different question** about a different slice of that matrix.

Think of it like this:

- **Sensitivity / Recall:** "How good are we at finding what we're looking for?"
- **Specificity:** "How good are we at correctly ignoring what we're not looking for?"
- **Precision:** "When we say we found something, how often are we actually right?"

The reason we need all four is that no single metric tells the full story, especially with imbalanced data.

---

## 📌 Where This Fits in the Big Picture

This video consolidates and clarifies the terminology from Note #04, adding **Precision** as a fourth metric. These four numbers are used constantly throughout machine learning — in every model evaluation, every paper, every competition leaderboard. Getting them crystal clear now pays dividends for the entire playlist.

---

## 🧩 Step-by-Step Conceptual Walkthrough

### The Reference Confusion Matrix

Using a consistent example throughout:

```
               Predicted: Positive   Predicted: Negative
Actual: Positive  [  TP = 100     |      FN = 25        ]
Actual: Negative  [  FP = 10      |      TN = 865       ]
```

Total: 1000 data points. 125 actually positive, 875 actually negative.

---

### Metric 1: Sensitivity (= Recall = True Positive Rate)

**The Question:** Of all the things that are ACTUALLY positive, how many did we correctly identify?

**Look at:** The "Actual Positive" row only.

$$\text{Sensitivity} = \frac{TP}{TP + FN} = \frac{100}{100 + 25} = \frac{100}{125} = \mathbf{80\%}$$

We correctly found 80% of all actual positives.

**When it matters:** Medical screening (don't miss sick people), fraud detection (don't miss fraud).

---

### Metric 2: Specificity (= True Negative Rate)

**The Question:** Of all the things that are ACTUALLY negative, how many did we correctly clear?

**Look at:** The "Actual Negative" row only.

$$\text{Specificity} = \frac{TN}{TN + FP} = \frac{865}{865 + 10} = \frac{865}{875} = \mathbf{98.9\%}$$

We correctly cleared 98.9% of actual negatives.

**When it matters:** Spam filters (don't block legitimate emails), drug testing (don't falsely accuse clean athletes).

---

### Metric 3: Precision (= Positive Predictive Value)

**The Question:** Of all the things we PREDICTED as positive, how many were actually positive?

**Look at:** The "Predicted Positive" column only.

$$\text{Precision} = \frac{TP}{TP + FP} = \frac{100}{100 + 10} = \frac{100}{110} = \mathbf{90.9\%}$$

When we predicted "positive", we were right 90.9% of the time.

**When it matters:** Information retrieval (are your search results relevant?), recommendation systems (are your recommendations actually good?).

---

### Metric 4: Recall (= Sensitivity = True Positive Rate)

Recall is exactly the same as Sensitivity — just a different name used in different contexts.

$$\text{Recall} = \frac{TP}{TP + FN} = \mathbf{80\%}$$

**Note:** "Recall" is the term used in information retrieval and machine learning. "Sensitivity" is the term used in medicine. Same formula, same meaning.

---

### The Key Difference: Precision vs. Sensitivity/Recall

| Metric                   | What it looks at          | Denominator                       |
| :----------------------- | :------------------------ | :-------------------------------- |
| **Sensitivity / Recall** | Actual Positive row       | TP + FN (all actual positives)    |
| **Precision**            | Predicted Positive column | TP + FP (all predicted positives) |

**Sensitivity** is about **completeness** — did you find all the positives?
**Precision** is about **accuracy** — were your positive predictions correct?

---

### The F1 Score — Balancing Precision and Recall

When you want a single number that balances Precision and Recall:

$$F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

This is the **harmonic mean** of Precision and Recall. It's lower than the arithmetic average — which means it's only high if **both** Precision AND Recall are high.

---

## 📐 Summary of All Formulas

$$\text{Sensitivity} = \text{Recall} = \text{TPR} = \frac{TP}{TP + FN}$$

$$\text{Specificity} = \text{TNR} = \frac{TN}{TN + FP}$$

$$\text{Precision} = \text{PPV} = \frac{TP}{TP + FP}$$

$$F1 = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

---

## 💡 Key BAM Moments

- BAM! **Precision and Recall are often in tension.** If you predict more things as positive (lower threshold), Recall goes up but Precision goes down (more false positives dilute your predictions). This is the Precision-Recall trade-off.
- BAM! With imbalanced classes, **Precision becomes very important**. If only 1% of data is positive and you blindly predict everything as positive, you get 100% Recall but near-0% Precision.
- BAM! **Specificity looks at the negative class; Precision also uses the negative class (FP)** — but from a different angle. Specificity asks "of all negatives, how many did we clear?" Precision asks "of all our positive predictions, how many were actually positive?"
- BAM! The **F-beta score** generalises F1. Beta > 1 weights Recall more; Beta < 1 weights Precision more.

---

## ❓ Active Recall Questions

1. What is the formula for Precision? What slice of the confusion matrix does it use?
2. What is the key difference between Precision and Sensitivity/Recall?
3. When would you prioritise Precision over Recall? Give a real-world example.
4. What is the F1 score and when would you use it?
5. Why is the harmonic mean used for F1 instead of the arithmetic mean?

---

## 🔗 Related Notes

- [[StatQuest 04 — Sensitivity and Specificity]]
- [[StatQuest 07 — ROC and AUC]]
- [[ML — Classification Performance Metrics Part 1]]
- [[ML — ROC and AUC Curve Intuition]]
- [[ML — Imbalanced Data Under-Sampling & Over-Sampling Foundations]]

---

_Tags: #statquest #machine-learning #sensitivity #specificity #precision #recall #f1-score #confusion-matrix #classification-metrics #precision-recall-tradeoff_
