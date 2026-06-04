---
course: Stat 110
lecture: 3
topic: Axioms of Probability, Matching Problem, and Inclusion-Exclusion
tags: [probability, combinatorics, story-proof, notes]
---

# Lecture 3: The Axioms of Probability & The Inclusion-Exclusion Machinery

## 📌 Core Mindset Shift
Do not think of probability as purely grinding equations. Think of it as **assigning total weights to a universe of possibilities**. 
When using pure Permutations and Combinations (PnC), every complex counting operation follows a chronological timeline: **First select the conditions, AND THEN shuffle the remaining universe.**

---

## 1. Formal Foundations: The Axioms of Probability
Any valid mathematical probability system must satisfy the three **Kolmogorov Axioms**. Let $S$ be the total sample space.

1. **Non-negativity:** $P(A) \ge 0$ for any event $A$.
2. **Total Certainty:** $P(S) = 1$ (The probability of *something* happening in the universe is 100%).
3. **Countable Additivity:** If $A_1, A_2, A_3, \dots$ are mutually exclusive (disjoint) events, then:
   $$P\left(\bigcup_{i=1}^{\infty} A_i\right) = \sum_{i=1}^{\infty} P(A_i)$$

> [!danger] The "OR" Addition Trap
> You can **only** add probabilities directly ($P(A) + P(B)$) if the events are completely disjoint (they cannot physically happen at the same time). If they overlap, simple addition over-counts the universe.

---

## 2. The Core Mechanic: "Choose, and then Shuffle"
When calculating favorable outcomes for overlapping events, use this two-step English timeline framework to construct your PnC math.

### The Timeline Breakdown
1. **The Choice ($\binom{n}{k}$):** You select exactly $k$ items or positions out of $n$ available options to satisfy your specific constraint.
2. **The Shuffle ($(n-k)!$):** > [!todo] Why do we multiply?
   > We multiply because of the **"And Then..."** principle. For *every single unique choice* you made in step 1, the remaining $n-k$ elements retain their freedom to shuffle in $(n-k)!$ unique ways, ballooning the total branch of possibilities.

$$\text{Total Favorable Arrangements} = \underbrace{\binom{n}{k}}_{\text{Choose the slots}} \times \underbrace{(n-k)!}_{\text{Shuffle the rest}}$$

---

## 3. The Inclusion-Exclusion Principle
When events overlap, we execute an alternating dance of addition and subtraction to systematically clean up over-counting.

### The General Formula
$$P\left(\bigcup_{i=1}^n A_i\right) = \sum_{i} P(A_i) - \sum_{i<j} P(A_i \cap A_j) + \sum_{i<j<k} P(A_i \cap A_j \cap A_k) - \dots + (-1)^{n-1} P(A_1 \cap \dots \cap A_n)$$

### The Intuition
* **Add Singles:** We look at each event individually. This drastically over-counts the overlaps.
* **Subtract Pairs:** We strip away the intersections where two things happen at once. Now we have under-counted.
* **Add Triplets:** We restore the intersections where three things happen at once.

---

## 4. Deep-Dive: The Matching / Card-Countdown Problem
**Problem Statement:** You have a deck of $n$ unique cards labeled $1$ to $n$. You shuffle the deck and deal them face up one by one while counting out loud: *"1, 2, 3, \dots, n."* What is the probability that **at least one card matches your voice count**?

Let $A_i$ be the event that Card $i$ lands in Slot $i$. We want to find $P(A_1 \cup A_2 \cup \dots \cup A_n)$.

### Step-by-Step Inclusion-Exclusion Construction



### Block 1: The Singles ($\sum P(A_i)$)
* **English Context:** Choose exactly **1** slot out of $n$ to be a match, *and then* let the remaining $n-1$ cards shuffle completely through the remaining $n-1$ slots.
* **PnC Count:** $\binom{n}{1} \times (n-1)! = n \times (n-1)! = n!$
* **Probability Weight:** Divide by total sample space ($n!$):
  $$\text{Term Value} = \frac{n!}{n!} = 1 \implies \frac{1}{1!}$$

### Block 2: The Pairs ($\sum P(A_i \cap A_j)$)
* **English Context:** Choose exactly **2** slots out of $n$ to simultaneously hold matching cards, *and then* let the remaining $n-2$ cards shuffle freely through the remaining $n-2$ slots.
* **PnC Count:** $\binom{n}{2} \times (n-2)! = \frac{n!}{2!(n-2)!} \times (n-2)! = \frac{n!}{2!}$
* **Probability Weight:** Divide by total sample space ($n!$):
  $$\text{Term Value} = \frac{\frac{n!}{2!}}{n!} = \frac{1}{2!}$$

### Block 3: The Triplets ($\sum P(A_i \cap A_j \cap A_k)$)
* **English Context:** Choose exactly **3** slots out of $n$ to simultaneously hold matching cards, *and then* let the remaining $n-3$ cards shuffle freely.
* **PnC Count:** $\binom{n}{3} \times (n-3)! = \frac{n!}{3!(n-3)!} \times (n-3)! = \frac{n!}{3!}$
* **Probability Weight:**
  $$\text{Term Value} = \frac{\frac{n!}{3!}}{n!} = \frac{1}{3!}$$

### The Final Alternating Sequence
Assembling the blocks under the alternating signs of Inclusion-Exclusion yields:
$$P(\text{At least one match}) = \frac{1}{1!} - \frac{1}{2!} + \frac{1}{3!} - \frac{1}{4!} + \dots + (-1)^{n-1}\frac{1}{n!}$$

> [!tip] The Infinite Convergence ($n \to \infty$)
> This sequence is the Taylor series expansion for $1 - e^{-1}$. Therefore, regardless of whether your deck has 52 cards or 10,000,000 cards, the probability of getting at least one match converges rapidly to:
> $$P(\text{Match}) \approx 1 - \frac{1}{e} \approx 0.63212 \ (63.2\%)$$

---

## 🧠 Diagnostic Checklist for Revision
When solving an assignment question, ask yourself these consecutive questions:
- [ ] Am I trying to count the probability of an "OR" condition ($A \cup B \cup C$)?
- [ ] Do these events overlap? Can they happen at the same time? If yes, I **must** use Inclusion-Exclusion.
- [ ] When building the intersection terms, did I **Choose** the locked elements first using combinations?
- [ ] Did I **Multiply** by the remaining factorial shuffles to capture the full branch of possibilities?