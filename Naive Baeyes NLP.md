
---

tags: [machine-learning, nlp, naive-bayes, text-classification, bag-of-words, sentiment-analysis, tf-idf] source: https://www.youtube.com/watch?v=temQ8mHpe3k related:

- "[[Naive Bayes Classifier]]"
- "[[Bag of Words and TF-IDF]]"
- "[[Transformers and Attention]]"

---

# 🧠 Naïve Bayes for Text Data (NLP)

## ⚡ TL;DR

The Naïve Bayes math doesn't change for NLP — but the **preprocessing pipeline does**. Raw text must be converted into numerical vectors before the classifier can touch it. Once vectorised (via Bag of Words or TF-IDF), every word column is treated as an independent feature and the standard posterior calculation runs as normal. Key traps: zero-frequency words collapse the entire probability, and the model is blind to word order and context.

---

## 📌 The Big Picture: Strings → Probability Vectors

```
Raw Text → [Stop Word Removal] → [Bag of Words / TF-IDF Matrix] → [Naïve Bayes Classifier]
```

1. **Preprocessing:** Strip stop words (`the`, `is`, `a`) — they carry no sentiment signal
2. **Vectorisation:** Transform cleaned tokens into a frequency matrix (rows = sentences, columns = vocabulary words)
3. **Probability Mapping:**

$$P(\text{Good} \mid \text{Sentence}) \propto P(\text{Good}) \cdot \prod_{i=1}^{n} P(\text{Word}_i \mid \text{Good})$$

---

## 🧮 Numerical Walkthrough: Sentiment Pipeline

**Input sentence:** "The food is bad" **After stop word removal:** `["food", "bad"]`

### Likelihood Lookup (class = Good, 2/5 training reviews are Good)

|Word|$P(\text{word} \mid \text{Good})$|$P(\text{word} \mid \text{Bad})$|
|---|---|---|
|`food`|$2/4 = 0.5$|$1/3 \approx 0.33$|
|`bad`|$0/4 = 0.0$ ← 💀|$2/3 \approx 0.67$|

### Posterior Scores

$$\text{Score(Good)} = 0.4 \times 0.5 \times 0.0 = \mathbf{0.0}$$ $$\text{Score(Bad)} = 0.6 \times 0.33 \times 0.67 \approx \mathbf{0.13}$$

**Result:** Single zero-frequency word (`bad` never seen in Good reviews) collapses Score(Good) entirely → model predicts **Bad** ✅ — but only by accident. This is the zero-frequency flaw.

---

## 💻 Python Pipeline

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# MultinomialNB: designed for discrete word count features
# CountVectorizer: builds Bag of Words matrix + removes stop words
model_pipeline = Pipeline([
    ('vectorizer', CountVectorizer(stop_words='english')),
    ('classifier', MultinomialNB())
])

train_text = [
    "the food is delicious",
    "best service ever",
    "the food is bad",
    "worst experience"
]
labels = [1, 1, 0, 0]  # 1=Good, 0=Bad

model_pipeline.fit(train_text, labels)

prediction = model_pipeline.predict(["food was terrible"])
print(f"Sentiment: {'Good' if prediction[0] == 1 else 'Bad'}")
```

> 💡 Use `TfidfVectorizer` instead of `CountVectorizer` to down-weight common words that appear frequently across all classes (reducing noise).

---

## 🌍 Real-World Use Cases

### Simple — Restaurant Feedback Sentiment

Sort incoming reviews into Positive / Negative. Words like `delicious` push toward Positive; `bad` pushes toward Negative.

### Advanced — Support Ticket Triage

Route thousands of inbound tickets to engineering teams based on token frequency: tickets with `API` + `Error` → Backend team; `UI` + `crash` → Frontend team.

> ⚠️ **Stress Point — Zero Frequency:** A single new word not seen in training zeroes out the full posterior for that class. Always use **Laplace Smoothing** (`alpha=1.0` in `MultinomialNB`, which is the default) so no probability ever hits zero.

---

## ⚠️ Gotchas

> [!warning] Context Blindness Naïve Bayes treats sentences as unordered word bags. "The food is **good**, not bad" and "The food is **bad**, not good" generate the same vector. For sarcasm, negation, or grammatical context → use **LSTMs** or **Transformers**.

> [!warning] Class Imbalance Bias If 90% of training reviews are Positive, the prior $P(\text{Good}) = 0.9$ heavily skews predictions. Even reviews with negative words may get labelled Good. Fix: balance your training set or use `class_weight` adjustments.

> [!warning] Zero Frequency Collapse One unseen word zeroes the entire posterior. Laplace Smoothing (`MultinomialNB(alpha=1.0)`) adds a pseudocount of 1 to every word-class combination, eliminating hard zeros.

---

## 📊 Naïve Bayes (NLP) vs Transformers

|Criterion|Naïve Bayes (Classical NLP)|Transformers (Modern NLP)|
|---|---|---|
|**Text Representation**|Bag of Words / TF-IDF sparse matrix|Dense embeddings (Word2Vec / BERT)|
|**Context Awareness**|None — scrambled token list|Full — word order, grammar, long-range dependencies|
|**Compute Cost**|Extremely low — CPU, milliseconds|High — requires GPU/TPU|
|**Best For**|Fast baselines, short text, high-volume triage|Sarcasm, negation, complex semantics|

---

## ❓ Active Recall Questions

1. Why must stop words be removed before building the Bag of Words matrix for sentiment analysis?
2. How does Laplace Smoothing prevent a single unseen word from crashing the entire classification?
3. Why is `MultinomialNB` preferred over `GaussianNB` when using `CountVectorizer` output?
4. What architecture would you upgrade to if Naïve Bayes fails on reviews where context and word order matter?

---

## 🔗 Related Notes

- [[Naive Bayes Classifier]]
- [[Bag of Words and TF-IDF]]
- [[Transformers and Attention]]
- [[Logistic Regression]]
- [[Cross Validation]]