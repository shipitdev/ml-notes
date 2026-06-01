**Course:** Harvard Statistics 110  
**Tags:** #stats110 #probability #inclusion-exclusion #birthday-problem #axioms #unit1  
**Status:** #complete  
**Links:** [[Stats110_Lectures_1_2]] | [[Stats110_Stars_and_Bars]] | [[Stats110_Lecture_4 — Conditional Probability]]

---

> [!NOTE] What this lecture covers Four big ideas, all connected:
> 
> 1. The **birthday problem** — deriving it cleanly using the complement
> 2. **Axioms of probability** — the non-naive foundation
> 3. **Inclusion-exclusion** — the tool for counting unions of overlapping events
> 4. The **matching problem** — inclusion-exclusion applied to a famous puzzle

---

## 1 · The Birthday Problem — Full Derivation

> [!QUESTION] How many people do you need for P(two share a birthday) > 50%? Answer: **only 23.** Most people guess ~180.

### Setup

- $k$ people, 365 days, all birthdays equally likely (naive probability)
- Total equally-likely outcomes: $365^k$

### Strategy — always count the complement

$$P(\text{at least one shared birthday}) = 1 - P(\text{all birthdays distinct})$$

### Counting the complement

$$P(\text{all distinct}) = \frac{365 \times 364 \times 363 \times \cdots \times (365 - k + 1)}{365^k}$$

The numerator: first person takes one day, second person must avoid that one, etc. This is exactly $\frac{365!}{(365-k)!}$ divided by $365^k$.

$$P(\text{at least one match}) = 1 - \frac{365!}{(365-k)! \cdot 365^k}$$

### Key values

|$k$ (people)|$P(\text{shared birthday})$|
|---|---|
|10|11.7%|
|20|41.1%|
|**23**|**50.7%** ← crosses 50%|
|30|70.6%|
|40|89.1%|
|57|99.0%|
|70|99.9%|

> [!TIP] Why so low? With 23 people there are $\binom{23}{2} = 253$ pairs. Each pair has ~1/365 chance of a match. 253 "chances" compounds fast. The intuition: you're not checking one person against one birthday — you're checking every pair.

> [!SUCCESS] The master trick from this problem **Count the complement.** Any problem asking for "at least one" almost always goes: $$P(\text{at least one}) = 1 - P(\text{none})$$ P(none) is usually much easier to compute.

---

## 2 · Axioms of Probability

Blitzstein now upgrades from the naive definition (equally likely outcomes) to the full definition that works for _everything_.

### Kolmogorov's three axioms

> [!DEFINITION] The axioms A function $P$ is a valid probability measure if it satisfies:
> 
> **Axiom 1:** $P(A) \geq 0$ for all events $A$
> 
> **Axiom 2:** $P(S) = 1$ (the sample space has probability 1)
> 
> **Axiom 3:** For _countably many disjoint_ events $A_1, A_2, \ldots$: $$P\left(\bigcup_{i=1}^{\infty} A_i\right) = \sum_{i=1}^{\infty} P(A_i)$$

That's it. All of probability theory follows from these three rules.

### Properties derived from the axioms

These are NOT additional axioms — they are theorems. Blitzstein derives them from Axioms 1–3:

|Property|Formula|Proof sketch|
|---|---|---|
|Complement rule|$P(A^c) = 1 - P(A)$|$A \cup A^c = S$ (disjoint), so $P(A) + P(A^c) = P(S) = 1$|
|Empty set|$P(\emptyset) = 0$|$P(S^c) = 1 - P(S) = 0$|
|Monotonicity|If $A \subseteq B$ then $P(A) \leq P(B)$|$B = A \cup (B \cap A^c)$, these are disjoint, apply Axiom 3|
|Inclusion-exclusion (2 events)|$P(A \cup B) = P(A) + P(B) - P(A \cap B)$|Split $A \cup B$ into disjoint parts, apply Axiom 3|
|Union bound|$P(A \cup B) \leq P(A) + P(B)$|Since $P(A \cap B) \geq 0$|

> [!WARNING] Common confusion: Axiom 3 requires DISJOINT events You can only add probabilities directly when events have no overlap. If events overlap, you need inclusion-exclusion to correct for double-counting.

---

## 3 · Inclusion-Exclusion — The Core Idea

### The 2-event case

When $A$ and $B$ overlap, naively adding $P(A) + P(B)$ counts $A \cap B$ twice.

$$\boxed{P(A \cup B) = P(A) + P(B) - P(A \cap B)}$$

**Venn diagram logic:**

```
Region          | Counted in P(A) | Counted in P(B) | Net after formula
A only          |       1         |       0         |      1  ✓
A ∩ B           |       1         |       1         | 2 − 1 = 1  ✓
B only          |       0         |       1         |      1  ✓
```

### The 3-event case

Three overlapping sets have 7 distinct regions (remember: $2^3 - 1$). Track how many times each region gets counted:

$$P(A \cup B \cup C) = P(A) + P(B) + P(C)$$ $$\quad - P(A \cap B) - P(A \cap C) - P(B \cap C)$$ $$\quad + P(A \cap B \cap C)$$

> [!TIP] How to remember the signs
> 
> - **Add** all singles (1 at a time)
> - **Subtract** all pairs (2 at a time)
> - **Add** all triples (3 at a time)
> - **Subtract** all quadruples (4 at a time) — if applicable
> - Signs alternate: $+, -, +, -, \ldots$

### The general formula

> [!DEFINITION] Inclusion-exclusion for $n$ events $$P!\left(\bigcup_{i=1}^n A_i\right) = \sum_{k=1}^{n} (-1)^{k+1} \sum_{1 \leq i_1 < \cdots < i_k \leq n} P(A_{i_1} \cap \cdots \cap A_{i_k})$$

In plain English: sum over all non-empty subsets of the events, adding for odd-sized subsets and subtracting for even-sized subsets.

**Total number of terms:** $\binom{n}{1} + \binom{n}{2} + \cdots + \binom{n}{n} = 2^n - 1$

### Why does it work?

For any point $x$ that lies in exactly $m$ of the events $A_1, \ldots, A_n$, inclusion-exclusion counts $x$ exactly:

$$\binom{m}{1} - \binom{m}{2} + \binom{m}{3} - \cdots + (-1)^{m+1}\binom{m}{m} = 1$$

This is a consequence of the binomial theorem: $(1-1)^m = 0$ implies $\binom{m}{0} - \binom{m}{1} + \binom{m}{2} - \cdots = 0$, rearranged gives the sum above equals 1.

---

## 4 · The Matching Problem (de Montmort, 1713)

### Problem statement

Shuffle cards labeled $1, 2, \ldots, n$. Card $i$ is a **match** if card $i$ ends up in position $i$. What is $P(\text{at least one match})$?

### Setup with inclusion-exclusion

Let $A_i$ = event that card $i$ is in position $i$.

We want $P(A_1 \cup A_2 \cup \cdots \cup A_n)$.

### Computing each intersection probability

**Singles:** Fix card $i$ in position $i$. The remaining $n-1$ cards can go anywhere: $$P(A_i) = \frac{(n-1)!}{n!} = \frac{1}{n}$$

**Pairs:** Fix cards $i$ and $j$. The remaining $n-2$ cards can go anywhere: $$P(A_i \cap A_j) = \frac{(n-2)!}{n!} = \frac{1}{n(n-1)}$$

**$k$ cards:** Fix any $k$ specific cards in place: $$P(A_{i_1} \cap \cdots \cap A_{i_k}) = \frac{(n-k)!}{n!}$$

### Applying inclusion-exclusion

There are $\binom{n}{k}$ ways to choose which $k$ cards to fix. Each gives the same probability $\frac{(n-k)!}{n!}$:

$$P(\geq 1 \text{ match}) = \sum_{k=1}^{n} (-1)^{k+1} \binom{n}{k} \frac{(n-k)!}{n!}$$

Simplify $\binom{n}{k} \cdot (n-k)! = \frac{n!}{k!(n-k)!} \cdot (n-k)! = \frac{n!}{k!}$:

$$= \sum_{k=1}^{n} (-1)^{k+1} \frac{1}{k!} = 1 - \frac{1}{2!} + \frac{1}{3!} - \frac{1}{4!} + \cdots + \frac{(-1)^{n+1}}{n!}$$

### The beautiful limit

Recall the Taylor series: $e^{-1} = 1 - 1 + \frac{1}{2!} - \frac{1}{3!} + \cdots = \sum_{k=0}^{\infty} \frac{(-1)^k}{k!}$

So as $n \to \infty$:

$$P(\geq 1 \text{ match}) \to 1 - e^{-1} = 1 - \frac{1}{e} \approx 0.6321$$

> [!TIP] Convergence is shockingly fast

|$n$|$P(\geq 1 \text{ match})$|
|---|---|
|1|1.0000|
|2|0.5000|
|3|0.6667|
|4|0.6250|
|5|0.6333|
|6|0.6319|
|**∞**|**0.6321**|

By $n = 5$ or $6$, it's essentially already the limiting value!

### Derangements

The complement — $P(\text{no matches})$ — counts **derangements**: permutations where no element is in its natural position.

$$P(\text{no matches}) = \sum_{k=0}^{n} \frac{(-1)^k}{k!} \approx e^{-1} \approx 0.3679$$

The number of derangements of $n$ items: $D_n = n! \cdot \sum_{k=0}^{n} \frac{(-1)^k}{k!} \approx \frac{n!}{e}$

---

## 5 · How to Apply Inclusion-Exclusion — The Algorithm

> [!TIP] Step-by-step procedure
> 
> 1. **Identify the events.** Define $A_i$ = "bad event $i$ occurs" (usually what you want to count).
> 2. **Find the structure.** Are all events symmetric (same $P(A_i)$, same $P(A_i \cap A_j)$, etc.)? If yes, exploit symmetry.
> 3. **Compute each level.** $P(A_i)$, $P(A_i \cap A_j)$, $P(A_i \cap A_j \cap A_k)$, etc.
> 4. **Multiply by the count.** $\binom{n}{k}$ terms at level $k$ if symmetric.
> 5. **Alternate signs and sum.**

### Symmetric case shortcut

When all $k$-fold intersections have the same probability $p_k$:

$$P!\left(\bigcup A_i\right) = \binom{n}{1} p_1 - \binom{n}{2} p_2 + \binom{n}{3} p_3 - \cdots$$

This was exactly the matching problem: all $p_k = (n-k)!/n!$.

---

## 6 · Worked Examples

### 🟢 Easy — two events

> $P(A) = 0.4$, $P(B) = 0.5$, $P(A \cap B) = 0.2$. Find $P(A \cup B)$.

$$P(A \cup B) = 0.4 + 0.5 - 0.2 = \mathbf{0.7}$$

---

### 🟡 Medium — three events, symmetric

> 3 fair dice are rolled. What is $P(\text{at least one 6})$?

Let $A_i$ = event die $i$ shows 6.

$P(A_i) = 1/6$, $P(A_i \cap A_j) = 1/36$, $P(A_1 \cap A_2 \cap A_3) = 1/216$

$$P(\geq 1 \text{ six}) = 3 \cdot \frac{1}{6} - 3 \cdot \frac{1}{36} + 1 \cdot \frac{1}{216} = \frac{1}{2} - \frac{1}{12} + \frac{1}{216} = \frac{108 - 18 + 1}{216} = \frac{91}{216} \approx 0.421$$

_Faster via complement:_ $1 - (5/6)^3 = 1 - 125/216 = 91/216$. ✓

> [!TIP] Complement vs. inclusion-exclusion For "at least one" with independent events, the complement is always cleaner. Use inclusion-exclusion when events are NOT independent or when you need the full breakdown.

---

### 🟡 Medium — three events, non-symmetric

> In a class of 40 students: 20 like math, 18 like physics, 15 like chemistry. 10 like both math and physics, 8 like both math and chemistry, 6 like both physics and chemistry, 5 like all three. How many like at least one subject?

$$|M \cup P \cup C| = 20 + 18 + 15 - 10 - 8 - 6 + 5 = \mathbf{34}$$

Six students like none of the three subjects.

---

### 🔴 Hard — matching problem variant

> 4 married couples are seated randomly in a row of 8 chairs. What is $P(\text{at least one couple sits adjacent})?$

Let $A_i$ = event couple $i$ sits adjacent ($i = 1, 2, 3, 4$).

**$P(A_i)$:** Treat couple $i$ as a block → 7 objects, 7! arrangements, 2 internal. Favorable = $7! \times 2$. Total = $8!$. $$P(A_i) = \frac{2 \times 7!}{8!} = \frac{2}{8} = \frac{1}{4}$$

**$P(A_i \cap A_j)$:** Two blocks → 6 objects, $6! \times 2^2$ arrangements. $$P(A_i \cap A_j) = \frac{4 \times 6!}{8!} = \frac{4}{56} = \frac{1}{14}$$

**$P(A_i \cap A_j \cap A_k)$:** Three blocks → $5! \times 2^3$: $$P(A_i \cap A_j \cap A_k) = \frac{8 \times 5!}{8!} = \frac{8}{336} = \frac{1}{42}$$

**$P(A_1 \cap A_2 \cap A_3 \cap A_4)$:** All 4 blocks → $4! \times 2^4$: $$P = \frac{16 \times 4!}{8!} = \frac{384}{40320} = \frac{1}{105}$$

**Inclusion-exclusion:**

$$P(\geq 1 \text{ adjacent couple}) = \binom{4}{1}\frac{1}{4} - \binom{4}{2}\frac{1}{14} + \binom{4}{3}\frac{1}{42} - \binom{4}{4}\frac{1}{105}$$

$$= 1 - \frac{6}{14} + \frac{4}{42} - \frac{1}{105} = 1 - \frac{3}{7} + \frac{2}{21} - \frac{1}{105}$$

$$= \frac{105 - 45 + 10 - 1}{105} = \frac{69}{105} = \frac{23}{35} \approx \mathbf{0.657}$$

---

## 7 · Common Mistakes

> [!WARNING] Mistake 1 — Adding probabilities when events overlap Axiom 3 allows addition ONLY for disjoint events. If events overlap, you MUST use inclusion-exclusion. Test: can the same outcome be in both $A$ and $B$? If yes → use IE.

> [!WARNING] Mistake 2 — Forgetting the alternating sign The sign pattern is $+, -, +, -, \ldots$. Singles are added, pairs subtracted, triples added. Writing them all with the same sign is the most common error on exams.

> [!WARNING] Mistake 3 — Not using symmetry When all $P(A_i)$ are equal, all $P(A_i \cap A_j)$ are equal, etc. — use symmetry! You only need to compute one representative from each level, then multiply by $\binom{n}{k}$. Computing each term separately is slow and error-prone.

> [!WARNING] Mistake 4 — Forgetting to define events clearly Before applying IE, write out "Let $A_i = $ [exact description]." Sloppy definitions lead to sloppy intersections.

---

## 8 · Summary of Key Results

|Result|Formula|
|---|---|
|Birthday problem|$P(\geq 1 \text{ match}) = 1 - \frac{365!}{(365-k)! \cdot 365^k}$|
|Complement rule|$P(A^c) = 1 - P(A)$|
|IE for 2 events|$P(A \cup B) = P(A) + P(B) - P(A \cap B)$|
|IE for 3 events|$P(A \cup B \cup C) = \sum P - \sum P(\cap) + P(\cap\cap)$|
|Matching: $P(\geq 1)$|$1 - \frac{1}{2!} + \frac{1}{3!} - \cdots \to 1 - 1/e$|
|Derangement count|$D_n \approx n!/e$|

---

## 9 · Flashcards

**Q:** What is the fastest way to compute $P(\text{at least one of } A_1, \ldots, A_n)$?  
**A:** If events are independent: complement $1 - P(\text{none}) = 1 - P(A_1^c) \cdots P(A_n^c)$. If not independent: use inclusion-exclusion.

---

**Q:** State the inclusion-exclusion formula for 3 events.  
**A:** $P(A \cup B \cup C) = P(A)+P(B)+P(C) - P(A \cap B)-P(A \cap C)-P(B \cap C) + P(A \cap B \cap C)$

---

**Q:** In the matching problem, why does $P(A_i \cap A_j) = \frac{(n-2)!}{n!}$?  
**A:** Fix cards $i$ and $j$ in their correct positions. The remaining $n-2$ cards arrange freely: $(n-2)!$ ways. Total arrangements: $n!$.

---

**Q:** What is the limit of $P(\text{at least one match})$ in the matching problem as $n \to \infty$?  
**A:** $1 - 1/e \approx 0.6321$. The series $1 - 1/2! + 1/3! - \ldots$ is the Taylor expansion of $1 - e^{-1}$.

---

**Q:** A point lies in exactly $m$ of the events. How many times does inclusion-exclusion count it?  
**A:** Exactly once. The alternating sum $\binom{m}{1} - \binom{m}{2} + \cdots = 1$ for all $m \geq 1$.

---

**Q:** How many terms are in the inclusion-exclusion formula for $n$ events?  
**A:** $2^n - 1$ (all non-empty subsets of $n$ events).

---

## 10 · Practice Problems

**P1.** $P(A) = 0.6$, $P(B) = 0.7$, $P(A \cup B) = 0.9$. Find $P(A \cap B)$.

> [!EXAMPLE]- Solution P1 Rearrange IE: $P(A \cap B) = P(A) + P(B) - P(A \cup B) = 0.6 + 0.7 - 0.9 = \mathbf{0.4}$

---

**P2.** A fair coin is flipped 5 times. What is $P(\text{at least one head})$?

> [!EXAMPLE]- Solution P2 Complement: $P(\text{at least one head}) = 1 - P(\text{all tails}) = 1 - (1/2)^5 = 1 - 1/32 = \mathbf{31/32}$

---

**P3.** Students take exams in Math (M), English (E), and Science (S). $P(M) = 0.9, P(E) = 0.8, P(S) = 0.7$. Exams are independent. What is $P(\text{passes all three})$? $P(\text{fails at least one})$?

> [!EXAMPLE]- Solution P3 Independence: $P(\text{all}) = 0.9 \times 0.8 \times 0.7 = 0.504$  
> Complement: $P(\text{fails at least one}) = 1 - 0.504 = \mathbf{0.496}$

---

**P4 (Hard).** A hat contains 5 cards labeled 1–5. Cards are drawn one at a time without replacement. What is $P(\text{no card drawn in its own position})$? (i.e. no derangement match)

> [!EXAMPLE]- Solution P4 $P(\text{no match}) = \sum_{k=0}^{5} \frac{(-1)^k}{k!} = 1 - 1 + \frac{1}{2} - \frac{1}{6} + \frac{1}{24} - \frac{1}{120}$  
> $= \frac{60 - 60 + 30 - 10 + \frac{5}{2} - \frac{1}{2}}{120}... $  
> $= \frac{44}{120} = \frac{11}{30} \approx \mathbf{0.3667}$  
> (Compare: $1/e \approx 0.3679$ — already very close at $n=5$!)

---

## 11 · Connection Map

```
Lecture 3 — Inclusion-Exclusion
        │
        ├──► Union bound (loose upper bound on P(A∪B))
        │         Used in: theoretical bounds, Boole's inequality
        │
        ├──► Complement rule (from axioms)
        │         P(at least one) = 1 − P(none) — most-used trick
        │
        ├──► Bonferroni inequalities (later)
        │         IE truncated after k terms gives alternating bounds
        │
        ├──► Derangements → Poisson limit (Lecture 11)
        │         D_n / n! → 1/e → Poisson(1) distribution
        │
        └──► Conditional probability (Lecture 4)
                  P(A|B) = P(A∩B)/P(B) — needs intersection calculations
                  IE gives us the intersections
```

---

## 📚 Resources

- **Blitzstein & Hwang** Ch. 1, Section 1.6 (inclusion-exclusion)  
    http://probabilitybook.net
- **Strategic Practice 2:** http://stat110.net — problems on inclusion-exclusion
- **Lecture 3 video:** Harvard Stats 110 YouTube playlist

---

_Previous:_ [[Stats110_Lectures_1_2]]  
_Next:_ [[Stats110_Lecture_4 — Conditional Probability & Bayes Rule]]