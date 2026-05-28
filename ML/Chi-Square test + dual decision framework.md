# 📊 Statistics Foundations: Practical Chi-Square (χ²) Test Implementation in Python

---

## 📌 1. The Big Picture: Categorical Dependency Tracking

When our dataset contains multiple **categorical variables** (e.g., labels, words, or classes), we cannot use a T-test or ANOVA since those metrics are built strictly for continuous averages. Instead, we look to the **Chi-Square (χ²) Test of Independence**.

- **The Core Objective:** To determine whether there is a statistically significant correlation or dependent association between two distinct categorical features derived from a single population sample.
- **Independence Definition:** If two variables are independent, knowing the classification of one variable provides zero predictive information about the other variable's outcome distribution.

---

## 📌 2. The Core Mathematical Anatomy of Chi-Square

The Chi-Square test functions by directly contrasting your **Observed Values** (the real-world data numbers collected in your matrix) against the **Expected Values** (the numbers you would mathematically expect if the two features were completely independent).

### The Chi-Square Formula

$$\chi^2 = \sum \frac{(O - E)^2}{E}$$

Where:

- **O** = Observed Frequency count inside a contingency cell
- **E** = Expected Frequency count computed under the assumption of independence

### Calculating Degrees of Freedom (dof)

For a contingency table, the degrees of freedom represent the number of cell categories that can vary independently:

$$dof = (\text{Number of Rows} - 1) \times (\text{Number of Columns} - 1)$$

---

## 📌 3. Phase 1: Proving Class Dependencies via Contingency Tables

Before executing code, we structure our categorical columns into a matrix called a **Contingency Table** (or Cross-Tabulation).

### 🔍 Sandbox Case Study Layout

We use the classic `tips` dataset from Seaborn. We want to test whether a customer's **Gender (`sex`)** has an active, dependent relationship with whether they smoke **(`smoker`)**.

- **Null Hypothesis (H₀):** `sex` and `smoker` are completely independent. There is no structural association between a customer's gender and their smoking choice.
- **Alternative Hypothesis (H₁):** `sex` and `smoker` are dependent. A significant structural association is present.

---

### 📊 Data Walkthrough

**Input (Raw Observed Frequency Contingency Table)**

Running `pd.crosstab()` maps out our absolute real-world counts:

|sex \ smoker|Yes|No|Row Totals|
|:--|:-:|:-:|:-:|
|**Male**|60|97|**157**|
|**Female**|33|54|**87**|
|**Column Totals**|**93**|**151**|**Grand Total (N) = 244**|

**🔄 Intermediate Mathematical Expected Counts Logic**

To compute the Expected Value (E) for any given cell under perfect independence conditions:

$$E = \frac{\text{Row Total} \times \text{Column Total}}{\text{Grand Total } (N)}$$

Calculated expected values:

$$E[\text{Male, Yes}] = \frac{157 \times 93}{244} = \mathbf{59.840} \qquad E[\text{Male, No}] = \frac{157 \times 151}{244} = \mathbf{97.160}$$

$$E[\text{Female, Yes}] = \frac{87 \times 93}{244} = \mathbf{33.160} \qquad E[\text{Female, No}] = \frac{87 \times 151}{244} = \mathbf{53.840}$$

**Output (Expected Frequency Contingency Matrix)**

|sex \ smoker|Yes|No|
|:--|:-:|:-:|
|**Male**|**59.840**|**97.160**|
|**Female**|**33.160**|**53.840**|

---

## 📌 4. Phase 2: Modular Production Pipeline & Code Execution

### 💻 Python Implementation Pipeline

```python
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

# 1. Load the production tips dataset from Seaborn
df_tips = sns.load_dataset('tips')

print("--- Step 1: Raw Categorical Feature Head Samples ---")
print(df_tips[['sex', 'smoker']].head(4))

# 2. Build the structural Contingency Cross-Tabulation table
contingency_table = pd.crosstab(df_tips['sex'], df_tips['smoker'])
print("\n--- Step 2: Observed Frequency Matrix Block ---")
print(contingency_table)

# Isolate the core numpy array of absolute counts
observed_values = contingency_table.values
print(f"\nObserved Values Extracted Array:\n{observed_values}")

# 3. Execute the Chi-Square Contingency Test
# stats.chi2_contingency returns: chi2 stat, p-value, dof, and expected array
chi2_stat, p_val, dof, expected_values = stats.chi2_contingency(contingency_table)

print("\n=====================================================================")
print("STATISTICAL RESULTS COMPILATION METRICS")
print("=====================================================================")
print(f"Calculated Chi-Square Statistic : {chi2_stat:.6f}")
print(f"Calculated P-Value              : {p_val:.6f}")
print(f"Calculated Degrees of Freedom   : {dof}")
print(f"Calculated Expected Matrix Array:\n{expected_values.round(3)}")

# 4. Method A: Validation via Critical Value Threshold
alpha = 0.05
critical_value = stats.chi2.ppf(q=1 - alpha, df=dof)
print(f"\nStatistical Critical Value Threshold: {critical_value:.4f}")

print("\n--- Decision Path A: Critical Value Evaluation ---")
if chi2_stat >= critical_value:
    print("Verdict: Reject H0 — Significant Association Present. Features Dependent.")
else:
    print("Verdict: Fail to Reject H0 — Features Independent. Zero Association.")

# 5. Method B: Validation via P-Value
print("\n--- Decision Path B: P-Value Evaluation ---")
if p_val <= alpha:
    print("Verdict: Reject H0 — Significant Association Present. Features Dependent.")
else:
    print("Verdict: Fail to Reject H0 — Features Independent. Zero Association.")
```

### 📉 Printed Output Verification

```text
--- Step 1: Raw Categorical Feature Head Samples ---
      sex smoker
0  Female     No
1    Male     No
2    Male     No
3    Male     No

--- Step 2: Observed Frequency Matrix Block ---
smoker  Yes   No
sex             
Male     60   97
Female   33   54

Observed Values Extracted Array:
[[60 97]
 [33 54]]

=====================================================================
STATISTICAL RESULTS COMPILATION METRICS
=====================================================================
Calculated Chi-Square Statistic : 0.001935
Calculated P-Value              : 0.964915
Calculated Degrees of Freedom   : 1
Calculated Expected Matrix Array:
[[ 59.84  97.16]
 [ 33.16  53.84]]

Statistical Critical Value Threshold: 3.8415

--- Decision Path A: Critical Value Evaluation ---
Verdict: Fail to Reject H0 — Features Independent. Zero Association.

--- Decision Path B: P-Value Evaluation ---
Verdict: Fail to Reject H0 — Features Independent. Zero Association.
```

---

## 📌 5. Double-Verification Dual Decision Frameworks

In operational production scenarios, there are two alternate ways to check your final hypothesis verdict. Both yield the exact same statistical answer.

### Approach 1: The Chi-Square Statistic vs. Critical Value

You compare your calculated χ² score directly against the distribution's **Critical Value Cutoff Floor**.

- **The Rule:** If your calculated χ² score is greater than or equal to the critical value, it has landed past the critical threshold — reject the null hypothesis.
- **Our Case:** 0.001935 < 3.8415 → **Fail to Reject H₀**

### Approach 2: The P-Value vs. Alpha

You compare your calculated P-value directly against your **Significance Level Alpha (α)**.

- **The Rule:** If P ≤ α, the chance of getting your data results randomly is beneath your threshold limit — reject the null hypothesis.
- **Our Case:** 0.964915 > 0.05 → **Fail to Reject H₀**

---

## 📌 6. Summary Strategic Taxonomy Framework Matrix

|Validation Method|Input Variable Combination|Metric Contrast|Primary Feature Selection Use|
|:--|:--|:--|:--|
|**Pearson Correlation Matrix**|Continuous vs. Continuous|Linear deviations and slope mapping.|Trimming multicollinear independent variables to prevent matrix destabilisation.|
|**Chi-Square (χ²) Test**|Categorical vs. Categorical|Observed frequencies vs. Expected probability metrics.|Dropping categorical features that hold zero dependency to your categorical target label.|

---

## 🔗 Related Notes

- [[Statistics — Practical Implementation of Hypothesis Testing (T-Tests & Correlation)]]
- [[Statistical Inference — Choosing the Right Hypothesis Test]]
- [[Feature Selection — Removing Highly Correlated Features Using Pearson Correlation]]
- [[Feature Selection — Dropping Constant Features Using Variance Threshold]]
- [[Data Science — Unified EDA, Feature Engineering & Feature Selection Roadmap]]
- [[P-Value & Statistical Significance]]

---

## 📹 Recommended Video Resource

[Krish Naik — Tutorial 33: Chi-Square Test Implementation with Python — Hypothesis Testing Part 2](https://www.youtube.com/watch?v=w5iKu1IrTJQ)

---

_Tags: #statistics #chi-square #hypothesis-testing #categorical-data #contingency-table #p-value #degrees-of-freedom #independence-test #scipy #seaborn #tips-dataset #python #machine-learning #data-science_