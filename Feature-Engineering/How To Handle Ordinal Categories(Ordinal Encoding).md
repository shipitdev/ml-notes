# 📊 Feature Engineering: Ordinal Encoding (Label Encoding)

> [!info] The Big Picture
> Some categorical variables naturally contain a real-world ranking or progression.
>
> These are called:
>
> - **Ordinal Categories**
>
> Unlike nominal categories, ordinal labels possess meaningful order relationships that machine learning models should preserve.

---

# 📌 1. Respecting the Natural Hierarchy

## 🎯 The Core Goal

Convert categorical text labels into integers while preserving their natural ranking structure.

---

## ❌ The Problem with One-Hot Encoding

Suppose we have an education-level feature:

```text
High School
Bachelors
Masters
PhD
```

If we apply **One-Hot Encoding**:

- the dataset expands into multiple columns
- the ordering relationship disappears

The model no longer understands:

```text
PhD > Masters > Bachelors > High School
```

---

## ✅ The Solution — Ordinal Encoding

We manually assign increasing integer values to preserve hierarchy.

Example:

```python
{
    'High School': 1,
    'Bachelors': 2,
    'Masters': 3,
    'PhD': 4
}
```

This ensures the model recognizes structural progression.

---

# 📌 2. Core Strategic Application Framework

## 🚨 When to Use It

Use Ordinal Encoding when categories possess:

- clear sequential ordering
- natural ranking
- mathematical progression

---

## 📚 Common Examples

### 🎓 Academic Grades

```text
Fail < C < B < A
```

---

### 😊 Satisfaction Surveys

```text
Poor < Good < Excellent
```

---

### 🏢 Corporate Hierarchies

```text
Intern < Associate < Manager < Director
```

---

### 📅 Ordered Time Structures

```text
Monday → Tuesday → Wednesday
```

---

# 📌 3. Why We Use Ordinal Encoding

## ✅ Preserves Semantic Information

The model understands:

- higher-ranked groups carry larger importance
- ordinal relationships matter mathematically

This is especially useful for:

- linear models
- tree-based splitting logic
- ranking systems

---

## ✅ Prevents Feature Space Explosion

Unlike One-Hot Encoding:

- `1` categorical column remains:
  - exactly `1` numerical column

This avoids:

- Curse of Dimensionality
- sparse matrices
- memory inefficiency

---

# 📌 4. Real-World Applications

## 💼 HR & Fintech Recruitment Analytics

Suppose we want to predict candidate salary.

Feature:

```text
Education Level
```

Ordinal mapping:

```python
{
    'High School': 1,
    'Bachelors': 2,
    'Masters': 3,
    'PhD': 4
}
```

### ✅ Why This Helps

The model can now learn:

```text
Higher education level → potentially higher salary
```

---

## 🏨 Customer Experience Systems

Hotels and restaurants collect reviews like:

```text
Poor
Satisfactory
Excellent
```

Ordinal encoding converts these qualitative rankings into sortable numerical signals.

---

# 📌 5. Walkthrough Example

## 🎯 Objective

We want to encode the `Day_of_Week` feature to help predict customer shopping behavior.

---

## 🛠️ Step-by-Step Transformation

### 📥 Raw Dataset

| Transaction_ID | Day_of_Week |
|---|---|
| 001 | `Monday` |
| 002 | `Thursday` |
| 003 | `Sunday` |
| 004 | `Tuesday` |
| 005 | `Monday` |

---

## 🔄 Step 1 — Create Ordinal Mapping Dictionary

```python
weekday_map = {
    'Monday': 1,
    'Tuesday': 2,
    'Wednesday': 3,
    'Thursday': 4,
    'Friday': 5,
    'Saturday': 6,
    'Sunday': 7
}
```

---

## 🔄 Step 2 — Replace Text with Integer Rankings

Each categorical label is replaced with its corresponding ordinal value.

---

## 📤 Final Encoded Output

| Transaction_ID | Day_of_Week_Encoded |
|---|---|
| 001 | `1` |
| 002 | `4` |
| 003 | `7` |
| 004 | `2` |
| 005 | `1` |

---

# 📌 6. Programmatic Implementation

> [!example] Python Implementation Pipeline

```python
import datetime
import pandas as pd

# 1. Create sample date list
base_date = datetime.datetime.today()

date_list = [
    base_date - datetime.timedelta(days=x)
    for x in range(0, 5)
]

# 2. Initialize DataFrame
df = pd.DataFrame({'Date': date_list})

# 3. Extract day names
df['Day_of_Week'] = df['Date'].dt.day_name()

print("=== 1. RAW INPUT DATA ===")
print(df[['Date', 'Day_of_Week']])

# 4. Create ordinal hierarchy dictionary
weekday_map = {
    'Monday': 1,
    'Tuesday': 2,
    'Wednesday': 3,
    'Thursday': 4,
    'Friday': 5,
    'Saturday': 6,
    'Sunday': 7
}

print(f"\nOrdinal Mapping Rules: {weekday_map}")

# 5. Apply ordinal mapping
df['Day_Encoded'] = df['Day_of_Week'].map(weekday_map)

print("\n=== 2. FINAL PROCESSED MATRIX ===")
print(df[['Day_of_Week', 'Day_Encoded']])
```

---

# 📉 Output Verification

```text
=== 1. RAW INPUT DATA ===
                        Date Day_of_Week
0 2026-05-25 12:43:10.123456      Monday
1 2026-05-24 12:43:10.123456      Sunday
2 2026-05-23 12:43:10.123456    Saturday
3 2026-05-22 12:43:10.123456      Friday
4 2026-05-21 12:43:10.123456    Thursday

Ordinal Mapping Rules:
{
    'Monday': 1,
    'Tuesday': 2,
    'Wednesday': 3,
    'Thursday': 4,
    'Friday': 5,
    'Saturday': 6,
    'Sunday': 7
}

=== 2. FINAL PROCESSED MATRIX ===
  Day_of_Week  Day_Encoded
0      Monday            1
1      Sunday            7
2    Saturday            6
3      Friday            5
4    Thursday            4
```

---

# 📌 7. Advantages & Disadvantages

## ✅ Advantages

### 🧠 Preserves Structural Semantics

Ordinal Encoding keeps the natural ordering alive.

The model understands:

```text
Higher encoded value = higher rank
```

---

### ⚡ Computationally Efficient

- Extremely fast vectorized mapping
- Zero additional feature columns
- Minimal memory usage

---

# ⚠️ Disadvantages — The Uniform Distance Assumption

## ❌ The Hidden Problem

Ordinal Encoding assumes:

```text
Every step distance is equal
```

Example:

```text
1 → 2 == 3 → 4
```

But in reality:

- the jump from:
  - `Bachelors → Masters`
may be small

while:

- the jump from:
  - `Masters → PhD`
may be massive

---

## 🧠 Why This Matters

Ordinal Encoding compresses all transitions into:

```text
+1 increments
```

which may flatten complex real-world nonlinear relationships.

---

# 📌 8. Summary Cheat Sheet

| Category Type | Recommended Encoding | Core Logic | Feature Space Impact |
|---|---|---|---|
| **Nominal Categories** | One-Hot / Count Encoding | No ordering assumptions | Expands or compresses depending on method |
| **Ordinal Categories** | **Ordinal Encoding** | Preserve ranking hierarchy | Remains exactly `1` column |

---

# 🧠 Final Intuition

> [!tip] Key Insight
> Ordinal Encoding is fundamentally about:
>
> - preserving hierarchy
> - preserving ranking
> - preserving structured progression
>
> while still maintaining:
>
> - low memory cost
> - compact feature dimensions
> - efficient model processing
>
> It is ideal whenever category labels contain meaningful real-world order. > 
> [!tip] 

