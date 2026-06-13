# 📊 Statistical Inference: Choosing the Right Hypothesis Test (T-Test, Chi-Square, ANOVA, Correlation)

---

## 📌 1. The Big Picture: Mapping Features to Statistical Proving Grounds

In data science, we don't just guess if features are related or if experimental groups differ significantly. We use **Hypothesis Testing** to make data-driven decisions based on mathematical probability.

The absolute key to passing technical statistics interviews and building correct validation models is knowing exactly **which statistical test to apply to which dataset configuration**. You do not choose a test based on a gut feeling — you choose it strictly by auditing the **Data Types** (Categorical vs. Continuous) and the **Number of Features** you are evaluating simultaneously.

---

## 📌 2. The Sandbox Data Architecture

To understand when to use each test, consider this baseline mock dataset:

|Row_ID|Gender (Categorical)|Age_Group (Categorical)|Weight (Continuous)|Height (Continuous)|
|:--|:--|:--|:-:|:-:|
|1|`M`|`Elderly`|70 kg|1.4 m|
|2|`F`|`Adult`|65 kg|1.2 m|
|3|`M`|`Adult`|65 kg|1.4 m|
|4|`F`|`Child`|20 kg|1.0 m|
|5|`M`|`Adult`|75 kg|1.3 m|
|6|`M`|`Elderly`|80 kg|1.3 m|

---

## 📌 3. Core Hypothesis Foundations & The Decision Boundary

Every scenario below relies on setting up two opposing claims:

- **Null Hypothesis (H₀):** States that there is **no difference**, no association, or no effect present. It assumes everything matches standard baseline expectations.
- **Alternative Hypothesis (H₁):** States that there is a **genuine difference**, relationship, or structural shift.
- **Significance Cutoff Threshold (α = 0.05):** If your test calculates a **P-Value less than 0.05**, the result is highly surprising under baseline assumptions. You **Reject the Null Hypothesis** and accept the Alternative claim.

---

## 📌 4. The 5 Core Testing Scenarios: When to Use What

---

### Scenario 1: One Categorical Feature → One-Sample Proportion Test

#### The Setup

You are auditing exactly **one categorical text column** (e.g., `Gender`).

#### The Objective

Find out if the proportion of categories inside your sample matches a known background baseline target or population metric.

#### Real-World Interview Question

_"Is there a significant difference in the baseline proportion of Male vs. Female applicants checking into this platform compared to the national average?"_

#### The Operational Path

Run a **One-Sample Proportion Test**.

---

### Scenario 2: Two Categorical Features → Chi-Square Test

#### The Setup

You are comparing **two categorical text columns** at the same time (e.g., `Gender` AND `Age_Group`).

#### The Objective

Find out if there is a dependent association between these two text labels, or if they are completely independent of one another.

#### Real-World Interview Question

_"Is a customer's purchasing category classification dependent on, or associated with, their geographic location label?"_

#### The Operational Path

Run a **Chi-Square (χ²) Test of Independence**.

---

### Scenario 3: One Continuous Variable → T-Test

#### The Setup

You are evaluating exactly **one continuous numerical variable** (e.g., `Height` or `Weight`).

#### The Objective

Calculate your sample's mathematical average and determine if it significantly deviates from a known historical reference value or population baseline target.

#### Real-World Interview Question

_"The average height calculated from our current sample group is 1.3 meters. Does this represent a statistically significant deviation from the historical target mean of 1.5 meters?"_

#### The Operational Path

Run a **T-Test** (specifically a One-Sample T-Test).

> **Extension:** If you are comparing the means of _two independent numeric groups_ (e.g., Weight of Males vs. Weight of Females), use a **Two-Sample Independent T-Test**.

---

### Scenario 4: Two Continuous Variables → Correlation Test

#### The Setup

You are evaluating **two continuous numerical variables** simultaneously (e.g., `Weight` AND `Height`).

#### The Objective

Map the linear strength, direction, and trend between these two numeric scales without categorising them into groups.

#### Real-World Interview Question

_"As a person's numerical weight increases, does their recorded height trend upward in a reliable linear relationship?"_

#### The Operational Path

Run a **Pearson / Spearman Correlation Test**. This yields a correlation metric ranging smoothly between -1.0 and +1.0.

---

### Scenario 5: One Continuous Variable vs. One High-Cardinality Categorical Feature (3+ Classes) → ANOVA Test

#### The Setup

You are comparing **one continuous numerical column** (e.g., `Height`) across **one categorical column that contains three or more unique label options** (e.g., `Age_Group` featuring _Elderly_, _Adult_, and _Child_).

#### The Objective

Find out if the average values across these three or more independent groups differ significantly from one another.

#### Why Not Use Multiple T-Tests?

If you try to run multiple individual T-Tests to compare every pair of groups, your probability of making a false-positive calculation error increases drastically. ANOVA solves this by comparing the variance between all groups simultaneously.

#### Real-World Interview Question

_"Does the average height change significantly when comparing Elderly vs. Adult vs. Child demographic classifications?"_

#### The Operational Path

Run an **ANOVA (Analysis of Variance) Test**.

---

## 📌 5. The Unified Test Selection Cheat Sheet Matrix

|Input Feature 1|Input Feature 2|Group Count|Analytical Objective|Recommended Test|
|:--|:--|:--|:--|:--|
|**Categorical**|None|1 column|Check proportion balance against baseline targets.|**One-Sample Proportion Test**|
|**Categorical**|**Categorical**|2 columns|Audit dependency associations between labels.|**Chi-Square (χ²) Test**|
|**Continuous**|None|1 column|Check sample average against a reference value.|**One-Sample T-Test**|
|**Continuous**|**Categorical**|2 distinct labels|Compare averages between two distinct groups.|**Two-Sample Independent T-Test**|
|**Continuous**|**Continuous**|2 numeric columns|Track linear trends and structural direction.|**Pearson / Spearman Correlation**|
|**Continuous**|**Categorical**|3 or more distinct labels|Compare averages across three or more groups at once.|**ANOVA (Analysis of Variance)**|

---

## 🔗 Related Notes

- [[Feature Selection — Dropping Constant Features Using Variance Threshold]]
- [[Feature Selection — Removing Highly Correlated Features Using Pearson Correlation]]
- [[Data Science — Unified EDA, Feature Engineering & Feature Selection Roadmap]]
- [[Feature Engineering — Missing Data Mechanisms & Continuous Imputation (Day 1)]]
- [[Probability Distributions & Normal Curve]]
- [[P-Value & Statistical Significance]]

---

## 📹 Recommended Video Resource

[Krish Naik — Tutorial 32: All About P Value, T-Test, Chi-Square Test, ANOVA Test and When to Use What](https://www.youtube.com/watch?v=YrhlQB3mQFI)

---

_Tags: #statistics #hypothesis-testing #t-test #chi-square #anova #pearson-correlation #p-value #null-hypothesis #statistical-inference #machine-learning #data-science #interviews_