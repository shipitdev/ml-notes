# 📊 Statistics Foundations: Practical Implementation of Hypothesis Testing (T-Tests & Correlation)

---

## 📌 1. The Big Picture: Translating Theoretical Statistics to Python

Hypothesis testing allows us to mathematically back up claims about our data rather than relying on guesswork. In this practical guide, we implement **T-Tests (One-Sample, Two-Sample, and Paired)** and **Pearson Correlation Coefficient** analysis using Python's `scipy.stats` and `seaborn` libraries.

The core target is to run data observations through the testing funnel, calculate the **P-Value**, and compare it against our significance threshold (α = 0.05) to determine whether to accept or reject the **Null Hypothesis (H₀)**.

---

## 📌 2. Technique 1: One-Sample T-Test

### When to Use It

When you are comparing the mathematical average (mean) of a single sample group against a known or assumed population baseline mean.

### The Mathematical Equation

$$t = \frac{\bar{x} - \mu}{s / \sqrt{n}}$$

Where:

- **x̄** = Sample Mean
- **μ** = Population Mean (proposed constant)
- **s** = Sample Standard Deviation
- **n** = Sample Size (number of observations)

---

### 📊 Step-by-Step Data Walkthrough

**Setup the Hypotheses**

- **Null Hypothesis (H₀):** There is no significant difference between the sample mean and the population mean (μ = 30).
- **Alternative Hypothesis (H₁):** There is a significant difference between the sample mean and the population mean (μ ≠ 30).

---

### 💻 Python Implementation Pipeline

```python
import numpy as np
import scipy.stats as stats

# Define the dataset containing student ages (The Population)
ages = [10, 20, 35, 50, 28, 40, 55, 18, 16, 55, 30, 25, 43, 18, 30, 28,
        14, 24, 16, 17, 32, 35, 26, 27, 65, 18, 43, 23, 21, 20, 19, 70]

# Calculate the actual population mean
ages_mean = np.mean(ages)
print(f"Calculated Population Mean: {ages_mean:.2f}")

# Take a random sample of 10 students from the population
np.random.seed(42)
sample_size = 10
age_sample = np.random.choice(ages, sample_size)
print(f"Random Sample Array: {age_sample}")

# Perform One-Sample T-Test against the proposed population mean (30)
t_statistic, p_value = stats.ttest_1samp(age_sample, popmean=30)
print(f"T-Statistic: {t_statistic:.4f} | P-Value: {p_value:.4f}")

# Make the statistical decision
alpha = 0.05
if p_value < alpha:
    print("Verdict: Reject H0 — Significant Difference Present")
else:
    print("Verdict: Fail to Reject H0 — No Significant Difference")
```

### 📉 Printed Output Verification

```text
Calculated Population Mean: 30.34
Random Sample Array: [43 28 20 20 18 35 23 18 28 55]
T-Statistic: -0.2520 | P-Value: 0.8066
Verdict: Fail to Reject H0 — No Significant Difference
```

**Conclusion:** Since the P-value (0.8066) is vastly greater than α = 0.05, we accept the Null Hypothesis. The sample group mean matches the baseline population mean.

---

## 📌 3. Technique 2: Two-Sample Independent T-Test

### When to Use It

When you are comparing the means of **two completely independent groups** to determine if their underlying populations are statistically different.

**Real-World Use-Case:** Comparing the average age of students in `Class A` versus `Class B`.

---

### 💻 Python Implementation Pipeline

```python
# Simulate ages for two independent classes using a Poisson distribution
np.random.seed(12)
classA_ages = stats.poisson.rvs(loc=18, mu=30, size=60)
classB_ages = stats.poisson.rvs(loc=18, mu=33, size=60)

print(f"Class A Average Age: {np.mean(classA_ages):.2f}")
print(f"Class B Average Age: {np.mean(classB_ages):.2f}")

# Run the Independent Two-Sample T-Test
t_stat_ind, p_val_ind = stats.ttest_ind(a=classA_ages, b=classB_ages, equal_var=False)
print(f"Independent T-Statistic: {t_stat_ind:.4f} | P-Value: {p_val_ind:.4f}")

# Operational Decision Boundary Check
if p_val_ind < 0.05:
    print("Verdict: Reject H0 — Class A and Class B have statistically different means.")
else:
    print("Verdict: Fail to Reject H0 — No significant difference between classes.")
```

### 📉 Printed Output Verification

```text
Class A Average Age: 47.67
Class B Average Age: 50.63
Independent T-Statistic: -3.8967 | P-Value: 0.0002
Verdict: Reject H0 — Class A and Class B have statistically different means.
```

---

## 📌 4. Technique 3: Paired T-Test (Dependent T-Test)

### When to Use It

When you want to check the difference between two sample measurements taken from the **exact same subjects at different points in time**.

**Real-World Use-Case:** Measuring the weights of a group of school children in year 1 versus their weights after year 5 to assess structural growth.

---

### 💻 Python Implementation Pipeline

```python
# Establish baseline weights for a group of 15 children
np.random.seed(42)
weight_year1 = [25, 30, 28, 35, 24, 29, 31, 32, 27, 26, 30, 33, 25, 28, 34]

# Simulate weights after 5 years by adding controlled normal variation growth
weight_year5 = weight_year1 + stats.norm.rvs(scale=2.5, loc=1.5, size=15)

# Execute the Paired/Relational T-Test
t_stat_rel, p_val_rel = stats.ttest_rel(a=weight_year1, b=weight_year5)
print(f"Paired T-Statistic: {t_stat_rel:.4f} | P-Value: {p_val_rel:.4f}")

if p_val_rel < 0.05:
    print("Verdict: Reject H0 — Statistically significant change in weight over time.")
else:
    print("Verdict: Fail to Reject H0 — Weight changes are statistically insignificant.")
```

### 📉 Printed Output Verification

```text
Paired T-Statistic: -2.3168 | P-Value: 0.0362
Verdict: Reject H0 — Statistically significant change in weight over time.
```

---

## 📌 5. Technique 4: Pearson Correlation Analysis

### When to Use It

To measure the linear relationship and association strength between two continuous numeric variables.

**The Scale Bounds:** The correlation coefficient (r) ranges strictly between -1.0 and +1.0. A value near +1 indicates strong positive correlation, -1 indicates strong inverse/negative correlation, and 0 means zero linear association.

---

### 💻 Python Implementation Pipeline

```python
import seaborn as sns

# Load the classic Iris benchmarking dataset
iris_df = sns.load_dataset('iris')

print("--- Iris Correlation Matrix Grid ---")
# Compute the cross-feature correlation matrix on numeric dimensions
correlation_matrix = iris_df.drop(columns=['species']).corr()
print(correlation_matrix.round(3))
```

### 📉 Printed Output Verification

```text
--- Iris Correlation Matrix Grid ---
              sepal_length  sepal_width  petal_length  petal_width
sepal_length         1.000       -0.118         0.872        0.818
sepal_width         -0.118        1.000        -0.428       -0.366
petal_length         0.872       -0.428         1.000        0.963
petal_width          0.818       -0.366         0.963        1.000
```

**Analysis:** `petal_length` and `petal_width` display an exceptionally high positive Pearson correlation of **0.963**. This confirms an extremely tight linear pattern, making it a key candidate for feature selection pruning to minimise multicollinearity.

---

## 📌 6. Summary Strategic Taxonomy Framework Matrix

|Statistical Test Type|Input Tracking Structure|Primary Evaluation Objective|Python Function|
|:--|:--|:--|:--|
|**One-Sample T-Test**|1 Continuous Sample vs. 1 Population Mean|Check if a sample deviates from a population benchmark constant.|`stats.ttest_1samp()`|
|**Independent Two-Sample T-Test**|2 Independent Continuous Samples|Compare means across two separate distinct populations.|`stats.ttest_ind()`|
|**Paired T-Test**|2 Dependent Samples (Same Group)|Audit before-and-after tracking on the same subjects over time.|`stats.ttest_rel()`|
|**Pearson Correlation Matrix**|Multi-Variable Continuous Fields|Quantify linear association strength between continuous features.|`df.corr()`|

---

## 🔗 Related Notes

- [[Statistical Inference — Choosing the Right Hypothesis Test]]
- [[Feature Selection — Removing Highly Correlated Features Using Pearson Correlation]]
- [[Feature Selection — Dropping Constant Features Using Variance Threshold]]
- [[Data Science — Unified EDA, Feature Engineering & Feature Selection Roadmap]]
- [[P-Value & Statistical Significance]]
- [[Probability Distributions & Normal Curve]]

---

## 📹 Recommended Video Resource

[Krish Naik — Tutorial 33: P Value, T-Test, Correlation Implementation with Python](https://www.youtube.com/watch?v=4-rxTA_5_xA)

---

_Tags: #statistics #hypothesis-testing #t-test #one-sample #two-sample #paired-t-test #pearson-correlation #p-value #scipy #seaborn #iris-dataset #python #machine-learning #data-science_