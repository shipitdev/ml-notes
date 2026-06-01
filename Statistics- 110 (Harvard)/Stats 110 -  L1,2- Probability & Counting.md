**Course:** Harvard Statistics 110 — Introduction to Probability  
**Instructor:** Joe Blitzstein  
**Tags:** #stats110 #probability #counting #unit1  
**Status:** #complete  
**Links:** [[Stats 110 MOC]] | [[Lecture 3 — Birthday Problem & Axioms]]

---

## 🗺️ Overview

> [!NOTE] What these lectures cover Lectures 1 & 2 build the **entire foundation** of probability theory.
> 
> - What probability _means_
> - How to define a sample space and events
> - The **naive definition** of probability
> - Counting techniques: multiplication rule, permutations, combinations
> - The **sampling table** — the cheat sheet for all counting problems
> - First look at the Birthday Problem

---

## 1 · What Is Probability?

Two competing philosophies — the math is identical, but the interpretation differs:

|School|Definition|Works for|
|---|---|---|
|**Frequentist**|Long-run frequency of an event over repeated trials|Repeatable experiments (coin flips, dice rolls)|
|**Bayesian**|Degree of belief, updated with evidence|One-off events ("70% chance of rain tomorrow")|

> [!TIP] Blitzstein's stance Stats 110 mostly uses frequentist language, but the tools are universal. Don't get stuck on philosophy — focus on the math.

---

## 2 · Sample Space & Events

### Definitions

> [!DEFINITION] **Sample space** $S$ = the set of _all possible outcomes_ of an experiment.  
> **Event** $A$ = any subset of $S$, i.e., $A \subseteq S$.  
> **Probability** $P$ = a function mapping events to $[0, 1]$.

### The Pebble World 🪨

Think of each outcome as a **pebble** with a weight.

- Total weight of all pebbles = **1**
- $P(A)$ = total weight of pebbles _inside_ event $A$

This is the entire intuition of probability. Every calculation you do is just asking: _which pebbles are in the event, and how heavy are they?_

### Examples of Sample Spaces

|Experiment|Sample Space $S$|
|---|---|
|Flip one coin|${H, T}$|
|Roll one die|${1, 2, 3, 4, 5, 6}$|
|Flip two coins|${HH, HT, TH, TT}$|
|Pick a random number in $[0,1]$|All real numbers in $[0,1]$ — uncountably infinite|

> [!WARNING] Watch out The sample space can be finite, countably infinite, or uncountably infinite. The tools we use differ depending on this.

---

## 3 · The Naive Definition of Probability

> [!DEFINITION] Naive Definition $$P(A) = \frac{|A|}{|S|} = \frac{\text{number of outcomes in } A}{\text{total number of outcomes}}$$ **Valid only when all outcomes are equally likely.**

### When can you use it?

✅ Fair coin, fair die, random card draws, random arrangements of people  
❌ Loaded dice, biased coins, non-uniform distributions

> [!CAUTION] This is a big assumption! Most of Stats 110 is about building the **non-naive** framework. But for Lectures 1–2, assume equally likely outcomes unless told otherwise.

---

## 4 · Counting — The Multiplication Rule

To use naive probability, you need to count outcomes reliably. The master tool:

> [!FORMULA] Multiplication Rule If experiment 1 has $n_1$ outcomes, and for each outcome, experiment 2 has $n_2$ outcomes, …, then: $$\text{Total outcomes} = n_1 \times n_2 \times \cdots \times n_k$$

### Example

Choosing an outfit: 3 shirts × 4 pants × 2 shoes = **24** possible outfits.

Visualised as a **tree diagram**: each branch multiplies the choices.

```
Shirt 1 ── Pants 1 ── Shoes A  →  outcome 1
         |          └── Shoes B  →  outcome 2
         ├── Pants 2 ── Shoes A  →  outcome 3
         ...
Shirt 2 ── ...
Shirt 3 ── ...
```

24 leaves total = 3 × 4 × 2.

---

## 5 · Permutations — Ordered Arrangements

### Arranging all n objects

> [!FORMULA] $n$ Factorial $$n! = n \times (n-1) \times (n-2) \times \cdots \times 1$$ **Why?** First slot: $n$ choices. Second slot: $n-1$ (one used). Third: $n-2$. Multiply all.

|$n$|$n!$|
|---|---|
|1|1|
|3|6|
|5|120|
|7|5,040|
|10|3,628,800|
|20|2,432,902,008,176,640,000|

Note: $0! = 1$ by convention (there is exactly 1 way to arrange nothing).

### Choosing and ordering k from n (no replacement)

> [!FORMULA] Permutations $P(n, k)$ $$P(n, k) = \frac{n!}{(n-k)!} = n \times (n-1) \times \cdots \times (n-k+1)$$

**Example:** Gold, silver, bronze from 10 athletes: $$P(10, 3) = 10 \times 9 \times 8 = 720$$

---

## 6 · Combinations — The Binomial Coefficient

When **order does NOT matter** (a committee is the same regardless of who was listed first):

> [!FORMULA] Binomial Coefficient — "n choose k" $$\binom{n}{k} = \frac{n!}{k!,(n-k)!}$$

### The intuition behind the formula

1. Start with $P(n, k)$ = all ordered selections = $\frac{n!}{(n-k)!}$
2. Each group of $k$ items was counted $k!$ times (once for every ordering)
3. Divide out the overcounting:

$$\binom{n}{k} = \frac{P(n,k)}{k!} = \frac{n!}{k!,(n-k)!}$$

### The key question to ask every time

> [!TIP] Does order matter?
> 
> - "Choose President, VP, Secretary from 10 people" → **order matters** → $P(n,k)$
> - "Choose a committee of 3 from 10 people" → **order doesn't matter** → $\binom{n}{k}$

### Useful values to know

$$\binom{n}{0} = 1 \qquad \binom{n}{1} = n \qquad \binom{n}{n} = 1 \qquad \binom{n}{k} = \binom{n}{n-k}$$

The last identity is elegant: choosing $k$ things to include is the same as choosing $n-k$ things to _exclude_.

---

## 7 · The Sampling Table 📊

> [!INFO] Memorise this — it categorises every counting problem Ask two questions: (1) Does order matter? (2) With or without replacement?

||**Order matters**|**Order doesn't matter**|
|---|---|---|
|**With replacement**|$n^k$|$\binom{n+k-1}{k}$|
|**Without replacement**|$\dfrac{n!}{(n-k)!}$|$\dbinom{n}{k}$|

> [!NOTE] The top-right cell "With replacement, order doesn't matter" (stars and bars) appears rarely in the first half of Stats 110. The other three cells are your daily workhorses.

---

## 8 · Worked Examples

### 🟢 Easy — 3-letter words

> How many 3-letter "words" can be formed from ${A, B, C, D, E}$?

**(a) With replacement** (letters can repeat):  
Each slot has 5 choices, independently → $5^3 = \mathbf{125}$

**(b) Without replacement** (no letter repeats):  
$5 \times 4 \times 3 = \mathbf{60}$ — order matters, no repeats → $P(5,3)$

---

### 🟡 Medium — Adjacent seating

> 10 students sit randomly in a row of 10 chairs. What is $P(\text{Alice and Bob sit next to each other})$?

**Total arrangements:** $10!$

**Favorable:** Treat Alice+Bob as a single "super-person":

- 9 objects to arrange: $9!$ ways
- Alice and Bob can swap within the block: $2!$ ways
- Favorable = $9! \times 2$

$$P = \frac{9! \times 2}{10!} = \frac{2}{10} = \mathbf{\frac{1}{5}}$$

> [!TIP] The "gluing" trick Whenever two items must be adjacent, treat them as a single block, count arrangements of blocks, then multiply by internal arrangements of the block. Works repeatedly throughout the course.

---

### 🔴 Hard — Full-suit poker hand

> From a 52-card deck, what is $P(\text{a 5-card hand contains all 4 suits})$?

**Total 5-card hands:** $\binom{52}{5} = 2{,}598{,}960$

**Favorable:** All 4 suits must appear in 5 cards → exactly one suit appears twice.

|Step|Count|
|---|---|
|Choose which suit appears twice|$\binom{4}{1} = 4$|
|Choose 2 cards from that suit|$\binom{13}{2} = 78$|
|Choose 1 card from each of the other 3 suits|$13^3 = 2{,}197$|

$$\text{Favorable} = 4 \times 78 \times 2{,}197 = 685{,}464$$

$$P = \frac{685{,}464}{2{,}598{,}960} \approx \mathbf{0.2637}$$

About 26.4% of 5-card hands span all four suits.

---

## 9 · The Birthday Problem — Preview

> [!QUESTION] How many people do you need in a room for P(two share a birthday) > 50%?

The answer: **only 23 people.** Most people guess ~180. Let's see why.

**Strategy — count the complement:**

$$P(\text{at least one shared birthday}) = 1 - P(\text{all birthdays different})$$

$$P(\text{all different}) = \frac{365}{365} \times \frac{364}{365} \times \frac{363}{365} \times \cdots \times \frac{365 - n + 1}{365}$$

With $n = 23$: $P(\text{all different}) \approx 0.4927$, so $P(\text{shared}) \approx \mathbf{0.5073}$.

|People $n$|P(shared birthday)|
|---|---|
|10|11.7%|
|23|**50.7%** ← crosses 50%|
|40|89.1%|
|57|99.0%|
|70|99.9%|

> [!TIP] Why is it so low? With 23 people, there are $\binom{23}{2} = 253$ pairs. Each pair has a 1/365 chance of sharing a birthday. 253 "lottery tickets", each with ~0.27% odds — it adds up fast. (We'll formalise this in Lecture 3.)

---

## 10 · Common Pitfalls & Exam Tips

> [!WARNING] Pitfall 1 — Forgetting the "equally likely" assumption The naive formula $P(A) = |A|/|S|$ only works when outcomes are equally likely. Always check this before applying it.

> [!WARNING] Pitfall 2 — Confusing permutations and combinations Ask: "Would switching the order create a different outcome?" If yes → permutation. If no → combination.

> [!WARNING] Pitfall 3 — Double-counting When counting favorable outcomes, make sure you're not counting the same outcome twice. The "overcounting then dividing" technique (as in the combination formula) is the systematic fix.

> [!TIP] Exam strategy Always state your sample space explicitly. Blitzstein rewards clear reasoning over just a final answer.

---

## 11 · Key Formulas Summary

|Name|Formula|When to use|
|---|---|---|
|Multiplication rule|$n_1 \times n_2 \times \cdots \times n_k$|Sequential independent choices|
|Permutations (all)|$n!$|Arrange all $n$ objects in order|
|Permutations (k of n)|$\frac{n!}{(n-k)!}$|Order matters, no replacement|
|Combinations|$\binom{n}{k} = \frac{n!}{k!(n-k)!}$|Order doesn't matter, no replacement|
|Naive probability|$P(A) = \frac{\|A\|}{\|S\|}$|Equally likely outcomes only|

---

## 12 · Practice Problems

Try these before looking at solutions. Cover the answer and work it yourself.

**P1 (Easy).** A lock has a 3-digit code (digits 0–9, repetition allowed). How many possible codes are there? What if repetition is not allowed?

> [!EXAMPLE]- Solution P1 **With repetition:** $10^3 = 1000$  
> **Without repetition:** $10 \times 9 \times 8 = 720$

---

**P2 (Medium).** A group of 5 men and 5 women sit randomly in a row of 10 chairs. What is $P(\text{men and women alternate})$?

> [!EXAMPLE]- Solution P2 Alternating means either MWMWMWMWMW or WMWMWMWMWM → 2 patterns.  
> Within each pattern: $5!$ ways to arrange the men, $5!$ ways for the women.  
> Favorable = $2 \times 5! \times 5!$  
> Total = $10!$  
> $$P = \frac{2 \times 5! \times 5!}{10!} = \frac{2 \times 120 \times 120}{3{,}628{,}800} = \frac{28{,}800}{3{,}628{,}800} \approx \mathbf{0.00794}$$  
> Less than 1% — alternating seating is rare by chance!

---

**P3 (Hard).** How many ways can you distribute 12 identical cookies among 4 kids such that each kid gets at least 1 cookie?

> [!EXAMPLE]- Solution P3 **Stars and bars with a constraint.**  
> First, give each kid 1 cookie (satisfying the "at least 1" condition). Now distribute the remaining $12 - 4 = 8$ cookies freely among 4 kids, with no restriction.  
> This is "with replacement, order doesn't matter": $\binom{8 + 4 - 1}{4 - 1} = \binom{11}{3} = \mathbf{165}$

---

## 13 · Connections to Later Topics

```
Counting (L1-2)
    │
    ├──► Conditional Probability (L3-5) — need counting to compute P(A∩B)/P(B)
    │
    ├──► Binomial Distribution (L7) — C(n,k) is the binomial coefficient
    │
    ├──► Birthday Problem full solution (L3) — complement counting
    │
    └──► Hypergeometric Distribution (L7) — sampling without replacement
```

---

## 14 · Flashcards

Use these in Anki or Obsidian's spaced repetition plugin.

**Q:** What is the naive definition of probability and when does it apply?  
**A:** $P(A) = |A|/|S|$. Only when all outcomes are equally likely.

---

**Q:** What is the difference between $P(n,k)$ and $\binom{n}{k}$?  
**A:** $P(n,k)$ counts ordered selections (order matters). $\binom{n}{k}$ counts unordered subsets (order doesn't matter). $\binom{n}{k} = P(n,k)/k!$.

---

**Q:** How do you count arrangements where two specific items must be adjacent?  
**A:** Glue them into a single block. Count arrangements of $(n-1)$ objects, then multiply by $2!$ for the internal swap.

---

**Q:** Why does $\binom{n}{k} = \binom{n}{n-k}$?  
**A:** Choosing $k$ items to include is equivalent to choosing $n-k$ items to exclude.

---

**Q:** With 23 people, what is approximately $P(\text{two share a birthday})$?  
**A:** About 50.7%. Use complement: $1 - \prod_{i=0}^{22} \frac{365-i}{365}$.

---

## 📚 References & Resources

- **Textbook:** Blitzstein & Hwang, _Introduction to Probability_ (2nd ed.) — Chapter 1  
    Free online: http://probabilitybook.net
- **YouTube:** Stats 110 Playlist — Lectures 1 & 2  
    https://www.youtube.com/watch?v=KbB0FoqQLps
- **edX course:** https://www.edx.org/course/introduction-to-probability-0
- **Practice problems:** http://stat110.net (Strategic Practice sets 1 & 2)

---

_Next:_ [[Stats110_Lecture_3 — Birthday Problem, Axioms of Probability]]