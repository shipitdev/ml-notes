# 📊 StatQuest #02: Machine Learning Fundamentals — Cross Validation

> **Video:** [Machine Learning Fundamentals: Cross Validation](https://www.youtube.com/watch?v=fSytzGwwBVw)
> **Series:** StatQuest Machine Learning Playlist
> **Core Idea:** How do you know if your model is actually learning, or just memorising? Cross Validation gives you a reliable way to measure real performance.

---

## 🎯 The Core Intuition (Plain English First)

Suppose you're a teacher and you want to test if your students actually understood the lesson — not just memorised the practice problems. You wouldn't give them the **exact same questions** they practised on. You'd give them **new questions** on the same topic.

Machine learning has the same problem. If you test a model on the same data it trained on, it could score 100% just by memorising. That tells you nothing about whether it actually learned.

**Cross Validation** is the solution. It's a smart way of testing your model on data it hasn't seen — even when your dataset is small and you can't afford to set aside a large separate test set.

---

## 📌 Where This Fits in the Big Picture

Cross Validation is the **second fundamental concept** in the playlist — immediately after the intro. Without it, you can't reliably evaluate any model or compare different approaches. Every algorithm you'll study from here depends on this concept for fair evaluation.

---

## 🧩 Step-by-Step Conceptual Walkthrough

### Step 1: The Problem with Simple Train/Test Split

The naive approach: randomly split your data into 80% for training, 20% for testing.

This works, but has a weakness: your performance estimate depends heavily on **which 20%** you happened to put in the test set. You might get lucky or unlucky with that split. With a small dataset, 20% test = very few examples → noisy, unreliable estimate.

### Step 2: K-Fold Cross Validation — The Core Method

Instead of one split, do **K splits**:

1. Divide your data into K equal-sized groups called **folds** (typically K = 10).
2. **Round 1:** Train on folds 2–10. Test on fold 1. Record the performance score.
3. **Round 2:** Train on folds 1, 3–10. Test on fold 2. Record the performance score.
4. **Round 3:** Train on folds 1–2, 4–10. Test on fold 3. Record the performance score.
5. ...continue until every fold has been the test set exactly once.
6. **Final score** = average of all K test scores.

This way every data point is used for testing exactly **once** and for training K−1 times. Much more reliable.

### Step 3: Why K = 10 Is Common

- **Too small K (e.g., K = 2):** Each training set is too small → model doesn't learn well → underestimates true performance.
- **Too large K (e.g., K = N, called Leave-One-Out CV):** Very thorough, very slow. Each round leaves out just one data point.
- **K = 5 or 10:** Practical sweet spot used in almost all real applications.

### Step 4: What Does the Score Tell You?

The average CV score tells you how well your model is **expected to perform on new, unseen data**. This is a much more honest estimate than training accuracy.

The **spread** of scores across folds also matters:

- Scores very consistent → model is stable (low variance)
- Scores all over the place → model is unstable (high variance, sensitive to which data it sees)

### Step 5: Using Cross Validation for Model Selection

Cross Validation isn't just for evaluation — it's also used for:

- **Comparing models:** Which model has the better CV score?
- **Tuning hyperparameters:** Which setting of K (in KNN) or depth (in Decision Trees) gives the best CV score?
- The winner is the model/setting with the best average CV performance.

---

## 📐 The Math (After the Intuition)

**K-Fold CV Score:**

$$\text{CV Score} = \frac{1}{K}\sum_{k=1}^{K}\text{Score}_k$$

Where Score_k is whatever metric you're using (accuracy, RMSE, AUC, etc.) computed on fold k when it was the test set.

---

## 💡 Key BAM Moments

- BAM! The golden rule: **the test set must never influence training in any way**. This includes feature scaling — fit your scaler on training data only, then transform the test fold using those parameters. Never fit on the test fold.
- BAM! If you use CV to select your best model, and then report that CV score as your final performance — you've slightly overfit to the validation folds. The truly honest final score comes from a **separate held-out test set** you've never touched.
- BAM! Cross validation is computationally expensive — you train K models instead of one. With K = 10 and a slow algorithm, this takes 10× longer.

---

## ❓ Active Recall Questions

1. What is the main problem with just using a single train/test split?
2. In 10-Fold Cross Validation with 1000 data points, how many data points are in each fold?
3. How many times does each data point appear in the test set across all K rounds?
4. What does the variance in CV scores across folds tell you about the model?
5. Why is it wrong to report a CV score as your final performance if you used CV to select the model?

---

## 🔗 Related Notes

- [[StatQuest 01 — Gentle Introduction to Machine Learning]]
- [[StatQuest 03 — The Confusion Matrix]]
- [[StatQuest 06 — Bias and Variance]]
- [[ML — Hyperparameter Optimisation XGBoost RandomizedSearchCV]]

---

\_Tags: #statquest #machine-learning #cross-validation #k-fold #model-evaluation #overfitting #train-test-split #hyperparameter-tuning
