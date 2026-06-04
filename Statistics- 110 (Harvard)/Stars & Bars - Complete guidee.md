**Course:** Harvard Statistics 110  
**Topic:** Sampling with replacement, order doesn't matter — $\binom{n+k-1}{k}$  
**Tags:** #stats110 #counting #stars-and-bars #unit1 #sampling-table  
**Links:** [[Stats110_Lectures_1_2]] | [[Stats110_Lecture_3 — Birthday Problem & Axioms]]  
**Difficulty:** ⭐⭐⭐ (the hardest cell of the sampling table)

---

> [!DANGER] Why people struggle here The other three cells of the sampling table feel natural. This one doesn't — because you're counting arrangements of _identical_ objects, and your brain keeps wanting to treat them as distinct. The fix is a single reframing trick called **stars and bars**, and once it clicks, this formula becomes mechanical.

---

## 0 · The sampling table — where we are

||**Order matters**|**Order doesn't matter**|
|---|---|---|
|**With replacement**|$n^k$|$\Large\binom{n+k-1}{k}$ ← **you are here**|
|**Without replacement**|$\dfrac{n!}{(n-k)!}$|$\dbinom{n}{k}$|

> [!TIP] The two diagnostic questions Before touching any formula, ask:
> 
> 1. **Does order matter?** (is arrangement A,B different from B,A?)
> 2. **With or without replacement?** (can the same item/category appear more than once?)
> 
> If your answers are **No + Yes** → you need $\binom{n+k-1}{k}$.

---

## 1 · The problem in plain English

You want to distribute **$k$ identical items** across **$n$ distinct bins**.

- Bins can receive **any amount**, including zero
- Items are **identical** — only the count per bin matters
- There is **no order** — bin 1 gets 3, bin 2 gets 1 is one outcome, not six

**Classic disguises in problem statements:**

| Phrasing                                          | $n$             | $k$        |
| ------------------------------------------------- | --------------- | ---------- |
| Distribute 10 identical apples to 4 kids          | 4 kids          | 10 apples  |
| Choose 5 donuts from 3 flavours, repeats ok       | 3 flavours      | 5 donuts   |
| Buy 6 stamps from 4 denominations                 | 4 denominations | 6 stamps   |
| How many ways to roll 3 dice (unordered results)? | 6 face values   | 3 rolls    |
| Non-negative integer solutions to $x_1+x_2+x_3=8$ | 3 variables     | 8 total    |
| Distribute 12 identical cookies to 4 kids         | 4 kids          | 12 cookies |

> [!NOTE] The keyword radar Watch for: _identical, indistinguishable, any amount, flavours/types, denominations, non-negative integer solutions, unordered._ Any of these is a hint toward stars & bars.

---

## 2 · The core idea — the stars and bars reframing

> [!INFO] The key insight Every way to distribute $k$ identical items into $n$ bins can be written as a unique sequence of **$k$ stars** and **$(n-1)$ bars**. This is a perfect 1-to-1 mapping.

### How it works

- **★** = one item
- **|** = divider between adjacent bins
- Read off the count per bin by counting stars between each pair of bars

### Example: 4 cookies into 3 jars

|Distribution|Sequence|Read as|
|---|---|---|
|Jar A=4, B=0, C=0|`★★★★` \| \||4 before first bar, 0 between bars, 0 after|
|Jar A=2, B=1, C=1|`★★` \| `★` \| `★`|2, then 1, then 1|
|Jar A=0, B=3, C=1|\| `★★★` \| `★`|0, then 3, then 1|
|Jar A=1, B=0, C=3|`★` \| \| `★★★`|1, then 0, then 3|
|Jar A=0, B=0, C=4|\| \| `★★★★`|0, then 0, then 4|

> [!TIP] The n−1 bars rule You always need **exactly $n-1$ bars** to create $n$ sections (just like $n-1$ cuts to slice a rope into $n$ pieces). For 3 jars you need 2 bars. For 5 jars you need 4 bars.

---

## 3 · Deriving the formula step by step

**Step 1:** Any valid distribution = a sequence of $k$ stars and $(n-1)$ bars.

**Step 2:** Total length of the sequence = $k + (n-1) = k + n - 1$ symbols.

**Step 3:** The sequence is fully determined by choosing _which positions are stars_. The bars fill the rest automatically.

**Step 4:** Choosing $k$ positions from $k+n-1$ total positions (order doesn't matter among stars) =

$$\boxed{\binom{n+k-1}{k} = \frac{(n+k-1)!}{k!,(n-1)!}}$$

### Why not divide by something extra?

Because we're choosing positions for _identical_ stars — there's no overcounting to remove. Position 2 being a star is the same regardless of "which star" goes there.

### The equivalent form

$$\binom{n+k-1}{k} = \binom{n+k-1}{n-1}$$

Both are valid — you can choose the $k$ star positions _or_ the $(n-1)$ bar positions. Pick whichever makes the arithmetic easier.

---

## 4 · Worked examples — simple to hard

### 🟢 Easy 1 — direct application

> How many ways can you distribute 6 identical balls into 4 boxes?

- Items $k = 6$, bins $n = 4$
- Formula: $\binom{4+6-1}{6} = \binom{9}{6} = \binom{9}{3} = \mathbf{84}$

---

### 🟢 Easy 2 — flavour selection

> An ice cream shop has 5 flavours. You want 3 scoops (repeats allowed, order doesn't matter — it's one bowl). How many combinations?

- Flavours = bins $n = 5$, scoops = items $k = 3$
- $\binom{5+3-1}{3} = \binom{7}{3} = \mathbf{35}$

---

### 🟡 Medium 1 — integer solutions

> How many non-negative integer solutions are there to $x_1 + x_2 + x_3 = 10$?

This is exactly distributing 10 identical units among 3 variables (bins).

- $n = 3$, $k = 10$
- $\binom{3+10-1}{10} = \binom{12}{10} = \binom{12}{2} = \mathbf{66}$

---

### 🟡 Medium 2 — coins into piles

> In how many ways can you make change for $1 using pennies (1¢), nickels (5¢), dimes (10¢), and quarters (25¢)?

Wait — this is **not** a simple stars and bars problem because denominations have different values and we're constrained to a total of 100¢. This is a constrained version requiring careful case analysis or generating functions.

> [!WARNING] Don't blindly apply the formula Stars and bars counts ways to distribute a fixed **number** of items. If items have different **values** and you're fixing a **sum of values**, the formula doesn't directly apply without adjustment.

---

### 🟡 Medium 3 — with a constraint (at least 1 in each bin)

> Distribute 10 identical candies among 5 kids such that each kid gets **at least 1**.

**Strategy:** Give each kid 1 candy first (use 5). Now distribute the remaining $10 - 5 = 5$ candies freely.

- $n = 5$, $k = 5$ (remaining)
- $\binom{5+5-1}{5} = \binom{9}{5} = \mathbf{126}$

> [!TIP] The "at least 1" trick Subtract 1 from each bin upfront to convert "at least 1" → "at least 0", then apply the formula to the remainder: $$\binom{n + (k-n) - 1}{k-n} = \binom{k-1}{k-n}$$

---

### 🔴 Hard 1 — unordered dice

> You roll a standard 6-sided die 5 times. How many distinct **unordered** outcome sets are possible?

Unordered = identical rolls into 6 face-value bins. The set {1,1,3,5,5} is the same as {5,1,3,5,1}.

- $n = 6$ (faces), $k = 5$ (rolls)
- $\binom{6+5-1}{5} = \binom{10}{5} = \mathbf{252}$

Compare to ordered: $6^5 = 7{,}776$ — very different!

---

### 🔴 Hard 2 — constrained integer solutions

> How many non-negative integer solutions are there to $x_1 + x_2 + x_3 + x_4 = 12$ with $x_1 \leq 3$?

**Strategy — complement:**

Total (no constraint): $\binom{12+4-1}{12} = \binom{15}{12} = 455$

Subtract cases where $x_1 \geq 4$: substitute $y_1 = x_1 - 4$, so $y_1 + x_2 + x_3 + x_4 = 8$:

$$\binom{8+4-1}{8} = \binom{11}{8} = 165$$

$$\text{Answer} = 455 - 165 = \mathbf{290}$$

> [!TIP] Constraint trick For **upper bound** constraints, use inclusion-exclusion or complement. For **lower bound** constraints, substitute $y_i = x_i - \text{lower bound}$ and reduce $k$ accordingly.

---

### 🔴 Hard 3 — Stars 110 exam style

> A professor gives out 15 identical extra-credit points to 6 students. How many ways can the points be distributed if no student receives more than 5 points?

**Inclusion-exclusion:** Let $A_i$ = event that student $i$ gets more than 5 points (i.e. $\geq 6$).

Total ways (no constraint): $\binom{6+15-1}{15} = \binom{20}{15} = 15{,}504$

$|A_i|$: substitute $y_i = x_i - 6$, distribute remaining $15-6=9$ among 6: $$\binom{6+9-1}{9} = \binom{14}{9} = 2{,}002$$

There are $\binom{6}{1} = 6$ choices of which student, so $6 \times 2{,}002 = 12{,}012$.

$|A_i \cap A_j|$: two students each get $\geq 6$, remaining $15-12=3$: $$\binom{6+3-1}{3} = \binom{8}{3} = 56$$

There are $\binom{6}{2} = 15$ pairs, so $15 \times 56 = 840$.

$|A_i \cap A_j \cap A_k|$: three students, remaining $15-18 < 0$ → impossible = 0.

By inclusion-exclusion:

$$\text{Answer} = 15{,}504 - 12{,}012 + 840 - 0 = \mathbf{4{,}332}$$

---

## 5 · The formula in the full context of the sampling table

### When to use which formula — decision tree

```
Start here: are the k items IDENTICAL (indistinguishable)?
│
├── NO (items are distinct)
│     Does order matter?
│     ├── YES → nᵏ  (e.g. passwords, sequences)
│     └── NO  → C(n,k)  (e.g. committees, hands)
│
└── YES (items are identical)
      Does order matter?
      ├── YES → nᵏ  (each position is still a distinct choice)
      └── NO  → C(n+k−1, k)  ← stars and bars
```

### Side-by-side contrast

|Question|Formula|Why|
|---|---|---|
|Choose 3 people from 10 for a committee|$\binom{10}{3} = 120$|distinct people, no repeat, no order|
|Choose 3 scoops from 10 flavours, repeats ok|$\binom{12}{3} = 220$|identical scoops, repeats, no order|
|Arrange 3 people from 10 for ranked positions|$\frac{10!}{7!} = 720$|distinct people, no repeat, order matters|
|Roll a die 3 times, record ordered results|$6^3 = 216$|with replacement, order matters|
|Roll a die 3 times, count unordered result sets|$\binom{8}{3} = 56$|with replacement, no order|

---

## 6 · Common mistakes and how to avoid them

> [!WARNING] Mistake 1 — using C(n,k) when repetition is allowed **C(n,k)** counts choosing $k$ from $n$ _without_ repetition.  
> If items can repeat, you must use $\binom{n+k-1}{k}$.  
> Test: "Can the same category appear more than once?" → Yes → stars and bars.

> [!WARNING] Mistake 2 — confusing n and k $n$ = number of **bins/categories/groups** (what you're distributing _into_)  
> $k$ = number of **items** (what you're distributing)  
> Mnemonic: **n = Number of bins, k = Kit of items to distribute**

> [!WARNING] Mistake 3 — applying stars and bars when items are NOT identical If the problem says "distribute a red ball, a blue ball, and a green ball among 3 boxes", the balls are distinct. Stars and bars does not apply — you'd use $3^3 = 27$ (each ball independently chooses a box).

> [!WARNING] Mistake 4 — forgetting the constraint adjustment "At least $m$ per bin" → subtract $m$ from each bin upfront, reduce $k$ by $n \times m$, then apply formula.  
> If $k - n \times m < 0$, the answer is 0 (impossible).

> [!WARNING] Mistake 5 — confusing "non-negative" vs "positive" integer solutions Non-negative ($x_i \geq 0$) → direct stars and bars with $k$ total  
> Positive ($x_i \geq 1$) → subtract 1 per variable, apply stars and bars with $k - n$ total

---

## 7 · The constraint toolkit

|Constraint|What to do|
|---|---|
|$x_i \geq 0$ (no constraint)|Direct: $\binom{n+k-1}{k}$|
|$x_i \geq 1$ (each bin non-empty)|Substitute $y_i = x_i - 1$, use $k' = k - n$: $\binom{n+k'-1}{k'}$|
|$x_i \geq m$ (at least $m$ per bin)|Substitute $y_i = x_i - m$, use $k' = k - nm$: $\binom{n+k'-1}{k'}$|
|$x_i \leq m$ (at most $m$ per bin)|Inclusion-exclusion on $A_i = {x_i > m}$|
|Mixed constraints|Combine the above techniques|

---

## 8 · Integer solutions cheat sheet

The equation $x_1 + x_2 + \cdots + x_n = k$ is the purest form of stars and bars.

|Constraint|Formula|Example ($n=3$, $k=7$)|
|---|---|---|
|$x_i \geq 0$|$\binom{n+k-1}{k}$|$\binom{9}{7} = 36$|
|$x_i \geq 1$|$\binom{k-1}{n-1}$|$\binom{6}{2} = 15$|
|$x_i \geq 2$|$\binom{k-n-1}{n-1}$ ... substitute $y_i=x_i-2$|$k'=7-6=1$: $\binom{3}{2}=3$|

> [!TIP] The $\binom{k-1}{n-1}$ shortcut For $x_i \geq 1$ (positive integers), the formula simplifies nicely to $\binom{k-1}{n-1}$ — place $n-1$ dividers among $k-1$ gaps between $k$ items lined up in a row. This is equivalent but often faster to compute.

---

## 9 · Flashcards

**Q:** What does stars and bars count?  
**A:** The number of ways to distribute $k$ identical items into $n$ distinct bins, where bins can be empty. Answer: $\binom{n+k-1}{k}$.

---

**Q:** What is the stars and bars sequence for "Jar A gets 2, Jar B gets 0, Jar C gets 3"?  
**A:** ★★ | | ★★★ (2 stars, bar, 0 stars, bar, 3 stars)

---

**Q:** Why $(n-1)$ bars for $n$ bins?  
**A:** You need $n-1$ dividers to split a line into $n$ sections — same as $n-1$ cuts to divide a rope into $n$ pieces.

---

**Q:** How many non-negative integer solutions to $x_1 + x_2 + x_3 + x_4 = 5$?  
**A:** $\binom{4+5-1}{5} = \binom{8}{5} = \binom{8}{3} = 56$

---

**Q:** You want to distribute 8 identical chocolates to 3 friends such that each gets at least 1. How many ways?  
**A:** Give 1 to each first (uses 3), distribute remaining 5 freely: $\binom{3+5-1}{5} = \binom{7}{5} = \binom{7}{2} = 21$

---

**Q:** What is the key diagnostic question to identify a stars and bars problem?  
**A:** Are the $k$ items **identical/indistinguishable** and does **order not matter**? If yes → $\binom{n+k-1}{k}$.

---

**Q:** $\binom{n+k-1}{k}$ equals what other expression?  
**A:** $\binom{n+k-1}{n-1}$ — choose bar positions instead of star positions. Both give the same answer.

---

## 10 · Practice problems (with collapsible solutions)

**P1.** A bakery has 4 types of muffins. You buy a box of 6. How many different boxes are possible? (Only the count per type matters.)

> [!EXAMPLE]- Solution P1 $n = 4$ types (bins), $k = 6$ muffins (items)  
> $\binom{4+6-1}{6} = \binom{9}{6} = \binom{9}{3} = \mathbf{84}$

---

**P2.** Find the number of non-negative integer solutions to $a + b + c + d + e = 15$.

> [!EXAMPLE]- Solution P2 5 variables (bins), sum = 15 (items)  
> $\binom{5+15-1}{15} = \binom{19}{15} = \binom{19}{4} = \frac{19 \times 18 \times 17 \times 16}{4!} = \mathbf{3876}$

---

**P3.** How many ways can you distribute 9 identical stickers among 4 students such that each student gets at least 2?

> [!EXAMPLE]- Solution P3 Give 2 to each student first: uses $4 \times 2 = 8$ stickers. Remaining: $9 - 8 = 1$.  
> Distribute 1 sticker freely among 4 students: $\binom{4+1-1}{1} = \binom{4}{1} = \mathbf{4}$

---

**P4.** A vending machine has 5 slots, each stocked with an unlimited supply of one candy type. You buy 4 candies. How many distinct (unordered) selections exist?

> [!EXAMPLE]- Solution P4 $n = 5$ slots, $k = 4$ purchases  
> $\binom{5+4-1}{4} = \binom{8}{4} = \mathbf{70}$

---

**P5 (Hard).** How many ways can 3 dice (6-sided) show a total of 9, counting only the unordered set of face values?

> [!EXAMPLE]- Solution P5 We need non-negative integers $x_1 \leq x_2 \leq x_3$ (die values 1–6) summing to 9.  
> This is harder than pure stars and bars because (a) values are between 1 and 6, and (b) they must be ordered (unordered set).  
> **Direct enumeration** of triples $(x_1, x_2, x_3)$ with $x_1 \leq x_2 \leq x_3$, each in ${1,...,6}$, summing to 9:  
> (1,2,6), (1,3,5), (1,4,4), (2,2,5), (2,3,4), (3,3,3) → **6 ways**  
> _Note: this problem has tight enough constraints that enumeration beats formula. Stars and bars gives a starting count; upper-bound constraints then require inclusion-exclusion._

---

**P6 (Hard).** How many ways can you choose 7 items from 4 categories if category 1 must contribute at most 3 items and all others are unrestricted?

> [!EXAMPLE]- Solution P6 **Total** (no constraint): $\binom{4+7-1}{7} = \binom{10}{7} = 120$  
> **Subtract** cases where category 1 contributes $\geq 4$: let $y_1 = x_1 - 4$, remaining $= 7-4 = 3$ items among 4 categories:  
> $\binom{4+3-1}{3} = \binom{6}{3} = 20$  
> **Answer:** $120 - 20 = \mathbf{100}$

---

## 11 · Connection map

```
Stars & Bars
     │
     ├──► Integer solutions (Stats 110 psets, combinatorics)
     │         x₁ + x₂ + ... + xₙ = k, non-negative
     │
     ├──► Inclusion-exclusion (Lecture 4–5)
     │         Adding upper/lower bound constraints
     │
     ├──► Multinomial coefficient (later)
     │         Distributing k DISTINCT items into n bins
     │         → n! / (n₁! n₂! ... nₖ!) instead
     │
     ├──► Poisson approximation (Lecture 11)
     │         Stars and bars motivates "many small contributions"
     │
     └──► Generating functions (advanced)
               Stars and bars is the coefficient of xᵏ in (1/(1-x))ⁿ
```

---

## 12 · One-line summary to tattoo on your brain

> [!SUCCESS] The rule **Stars and bars** = arrange $k$ identical stars and $n-1$ bars in a row.  
> Count = ways to pick which $k$ of the $k+n-1$ positions are stars = $\binom{n+k-1}{k}$.

---

## 📚 Resources

- **Blitzstein & Hwang**, _Introduction to Probability_ Ch. 1 — Stars and Bars section  
    Free: http://probabilitybook.net
- **Strategic Practice 1**, Harvard Stats 110 — Problems on counting  
    http://stat110.net
- **3Blue1Brown** — "Stars and Bars" visual explainer on YouTube