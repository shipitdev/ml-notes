# 19. Logistic Regression Details Pt 3: R-squared and p-value
🔗 https://www.youtube.com/watch?v=xxFYro8QuXA

## The Big Idea
Just like Linear Regression has R² and a p-value to judge how good the model is, Logistic Regression needs its own versions of these — but since it's built on likelihood (not sum of squares), the formulas look a bit different, using "log-likelihoods" instead.

## Flow of the Video

### 1. Recap: three reference log-likelihoods we need
- **LL(overall probability)**: the log-likelihood if we ignore the predictor entirely and just use the overall proportion (e.g., overall % of people who are obese) to predict everyone — this is our "dumb baseline," similar to "just guess the mean" in linear regression.
- **LL(fitted model)**: the log-likelihood of our actual fitted logistic curve (from the previous video).
- **LL(saturated model)**: the log-likelihood of a "perfect" model that predicts each individual outcome exactly (this concept is expanded fully in the next video).

### 2. McFadden's Pseudo R²
```
Pseudo R² = ( LL(overall probability) - LL(fitted model) ) / ( LL(overall probability) - LL(saturated model) )
```
- This mirrors the structure of ordinary R² (video #10): numerator = how much better our model is vs. the dumb baseline; denominator = how much better a "perfect" model would be vs. the dumb baseline.
- Result ranges roughly 0 to 1: higher = our model captures more of the possible improvement over just guessing.
- Note: Pseudo R² values in logistic regression are typically much smaller than R² values you'd see in linear regression — this is normal and expected, not a red flag by itself.

### 3. Testing significance: is our model better than the baseline by chance?
- We compare **2×(LL(fitted model) − LL(overall probability))** to a **Chi-squared distribution**.
- This tells us the p-value: the probability that we'd see this much improvement over the baseline model purely due to random chance if the predictor(s) had no real effect.
- Small p-value → our predictor(s) are genuinely improving the model's ability to explain the outcome.

### 4. Why this all connects to "Deviance"
- The quantities above (differences between log-likelihoods, multiplied by -2) are closely related to a measure called **Deviance** — which becomes the star of the very next video.

## Key Takeaways (Quick Recall)
- Logistic Regression's R² (McFadden's Pseudo R²) compares log-likelihoods instead of sum-of-squares.
- Formula: (LL(baseline) − LL(fitted)) / (LL(baseline) − LL(saturated)).
- Pseudo R² values run smaller than typical linear regression R² — that's normal.
- Significance testing uses a Chi-squared test on 2×(difference in log-likelihoods) between fitted model and baseline.
- These log-likelihood differences are the seed of the concept of "Deviance," covered next.
