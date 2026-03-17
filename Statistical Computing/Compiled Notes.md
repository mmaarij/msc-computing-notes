# Week 1: Fundamental Concepts & Discrete Distributions

## Definitions

- **Population:** The complete collection of all individuals or items under consideration in a statistical study.
- **Sample:** A subset of the population from which information is actually collected.

### Parameters vs Statistics

- **Population Parameters** (constants, usually unknown):
    
    - $\mu$ → population mean
    - $\sigma$ → population standard deviation
- **Sample Statistics** (random variables):
    
    - $\bar{x}$ → sample mean
    - $s$ → sample standard deviation

* * *

## Measures of Centrality and Variation

### Measures of Centrality (Centre)

1.  **Mean ($\bar{x}$):** Arithmetic average

    $$\bar{x} = \frac{\sum x_i}{n}$$
    
2.  **Median:** Middle value when data is ordered.
3.  **Mode:** Most frequently occurring value.

### Measures of Variation (Spread)

- **Range:** $\text{Max} - \text{Min}$. Very sensitive to outliers.
- **Sample Variance ($s^2$):**

    $$s^2 = \frac{\sum (x_i - \bar{x})^2}{n - 1}$$

- **Sample Standard Deviation ($s$):**

    $$s = \sqrt{\frac{\sum (x_i - \bar{x})^2}{n - 1}}$$
    

* * *

## Probability Theory Basics

### Core Rules

- **Sample Space ($\Omega$):** Set of all possible outcomes. $P(\Omega) = 1$

- **Empty Set ($\emptyset$):** Impossible event. $P(\emptyset) = 0$

- **Probability Bounds:** $0 \le P(A) \le 1$

- **Complement Rule:** $P(A^c) = 1 - P(A)$. Notation: $A^c$, $\bar{A}$, or $A'$.

### Combining Events

- **Intersection (AND):** $A \cap B$
- **Union (OR):** $A \cup B$

#### Addition Rules

- **General Addition Rule:** $P(A \cup B) = P(A) + P(B) - P(A \cap B)$
    
- **Disjoint (Mutually Exclusive) Events:**
    
    - Cannot occur together
    - $A \cap B = \emptyset$
    - $P(A \cup B) = P(A) + P(B)$
        

* * *

## Conditional Probability and Independence

### Conditional Probability

Probability that $B$ occurs given that $A$ has occurred:

$$P(B \mid A) = \frac{P(A \cap B)}{P(A)}$$

### Multiplication Law

$$P(A \cap B) = P(B \mid A) \times P(A)$$

### Independence

Two events are independent if one does not affect the other:

- $P(B \mid A) = P(B)$, or
- $P(A \cap B) = P(A) \times P(B)$
    

* * *

## Advanced Probability Theorems

### Bayes' Theorem

Used to reverse conditional probabilities:

$$P(A \mid B) = \frac{P(B \mid A) \times P(A)}{P(B)}$$

### Law of Total Probability

If $A_1, A_2, \dots, A_n$ partition the sample space:

$$P(B) = \sum_{i=1}^{n} P(B \mid A_i) \times P(A_i)$$

* * *

## Discrete Random Variables

### Definition

A **Random Variable ($X$)** is a numerical model for a measurement.

- **Discrete RV:** Takes a finite or countably infinite number of values.
    
- **Bernoulli RV:** Simplest discrete RV.  
    Takes value:
    
    - 1 for success
    - 0 for failure

### Probability Mass Function (pmf)

$$f(x) = P(X = x)$$

### Expected Value (Mean)

The long-run average or centre of gravity:

$$E(X) = \mu = \sum x \cdot P(X = x)$$

*Example (Fair die):*

$$E(X) = 3.5$$

* * *

## Cumulative Distribution Function (CDF)

$$F(x) = P(X \le x)$$

- For discrete RVs, the CDF has a **step shape**.
- **At least rule:** $P(X \ge k) = 1 - P(X < k)$
    

* * *

## Discrete Probability Distributions

### Binomial Distribution

Used for the number of successes in $n$ trials.

#### Assumptions (Always state in exams)

1.  Fixed number of trials ($n$)
2.  Constant probability of success ($p$)
3.  Trials are independent

#### Model

$$P(X = x) = \binom{n}{x} p^x (1 - p)^{n-x}$$

#### Parameters

- **Mean:** $\mu = np$

- **Standard Deviation:**

    $$\sigma = \sqrt{np(1 - p)}$$
    

* * *

### Poisson Distribution

Used for counting arrivals in a fixed interval of time or space.

#### Assumptions

1.  Probability proportional to interval size
2.  Probability of two or more arrivals in a very small interval is negligible
3.  Non-overlapping intervals are independent

#### Model

$$P(X = x) = \frac{e^{-\alpha t} (\alpha t)^x}{x!}$$

- $\alpha$ = average rate per unit
- $t$ = length of interval

#### Key Property (Very Exam Important)

$$Rate = \lambda = E(X) = Var(X) = \alpha t$$

* * *

# Week 2: Continuous, Sampling & Hypothesis Testing

## Continuous Random Variables

### Definition

A continuous random variable can take values anywhere in a continuum, such as height, temperature, or sales.

- **Density Function ($f(x)$):**  
    A curve where the area under the curve between two points represents probability.
    
- **Total Area:**  
    The total area under $f(x)$ is always 1:

    $$\int_{-\infty}^{\infty} f(x)\,dx = 1$$
    

### Uniform Distribution

The simplest continuous distribution where probability is constant between $a$ and $b$.

- **PDF:**

    $$f(x) = \frac{1}{b - a}, \quad a \le x \le b$$
    

* * *

## The Normal Distribution

### Properties

- Defined by **Mean ($\mu$)** and **Variance ($\sigma^2$)**.
- Notation: $X \sim N(\mu, \sigma^2)$
    

### Empirical Rule (68, 95, 99.7)

- 68% of data lies within $\mu \pm 1\sigma$
- 95% of data lies within $\mu \pm 2\sigma$
- 99.7% of data lies within $\mu \pm 3\sigma$

* * *

## Sampling Distributions

### Central Limit Theorem (CLT)

Regardless of the population distribution, if sample size $n$ is large, the distribution of the sample mean $\bar{X}$ is approximately normal.

- **Mean of $\bar{X}$:** $E(\bar{X}) = \mu$

- **Variance of $\bar{X}$:**

    $$\text{Var}(\bar{X}) = \frac{\sigma^2}{n}$$

- **Standard Error:** $\frac{\sigma}{\sqrt{n}}$
    
- **Z Statistic:**

    $$Z = \frac{\bar{X} - \mu}{\sigma / \sqrt{n}}$$
    

* * *

## Hypothesis Testing Basics

### Core Concepts

- **Null Hypothesis ($H_0$):**  
    Assumed true. Always contains equality ($=$, $\le$, $\ge$).
    
- **Alternative Hypothesis ($H_1$):**  
    The claim we seek evidence for. Always contains inequality ($\ne$, $<$, $>$).
    

### Errors

- **Type I Error ($\alpha$):**  
    Rejecting $H_0$ when it is actually true.
    
- **Type II Error ($\beta$):**  
    Failing to reject $H_0$ when it is actually false.
    

### The p-value

The probability of observing a result at least as extreme as the one obtained, assuming $H_0$ is true.

- **Decision Rule:**  
    Reject $H_0$ if
    
    $$\text{p-value} < \alpha$$
    

* * *

## Confidence Intervals

### Definition

An interval constructed around $\bar{x}$ where we are reasonably confident the true population mean $\mu$ lies.

- **Interpretation:**  
    In repeated sampling, 95% of such intervals would contain $\mu$.

### Formula (Known $\sigma$ or Large $n$)

$$\bar{x} \pm z_{\alpha/2} \frac{\sigma}{\sqrt{n}}$$

### Example: Cola Cans

- $\bar{x} = 299.64$
- $n = 100$
- $\sigma = 1.2$

Resulting interval:

$$[299.40, 299.88]$$

Since 300 is not in the interval, reject $H_0$.

* * *

## Hypothesis Tests for Proportions

Used for categorical data.

### Two Approaches

| Test | Method | Best For |
|------|--------|---------|
| `prop.test` | Normal ($\chi^2$) approximation | Large samples ($n \ge 30$) |
| `binom.test` | Exact binomial probabilities | Small samples or when exact results are needed |

### Example: Thanos Snap

- $H_0: p = 0.5$
- $H_1: p \ne 0.5$
- Observed: 64 vanished out of 100

**R Code (Approximate):**

```r
prop.test(64, 100, p = 0.5)
```

- p-value = 0.0069  
    Reject $H_0$.

**R Code (Exact — preferred for small samples):**

```r
binom.test(64, 100, p = 0.5)
```

- p-value = 0.0105  
    Reject $H_0$.

### `binom.test` Arguments

```r
binom.test(x, n, p = 0.5, alternative = "two.sided")
```

- `x` — number of successes observed
- `n` — number of trials
- `p` — hypothesised probability of success under $H_0$
- `alternative` — `"two.sided"`, `"less"`, or `"greater"`

### When to Use Each

- Use **`binom.test`** when $n$ is small (roughly $n < 30$), or when you need exact p-values.
- Use **`prop.test`** for large samples; it also supports comparing two proportions (`prop.test(c(x1, x2), c(n1, n2))`).

* * *

## One Sample t-Test

Used when population variance $\sigma^2$ is unknown.

- Uses Student's t distribution
- Degrees of freedom: $df = n - 1$

### Assumptions

1.  Data is numeric and continuous.
2.  Data is normally distributed.

**Normality Test: Shapiro-Wilk**

- If p-value > 0.05, assume normality.

### Example: Corrib River Radiation

- $H_0: \mu \ge 5$
- $H_1: \mu < 5$

**R Code:**

```r
t.test(corrib, mu = 5, alternative = "less")
```

- p-value = 0.002  
    Reject $H_0$. Water is safe.

* * *

## Comparing Two Means: Independent Samples

Used to compare two separate groups.

### Steps

1.  **Check Normality:**  
    Shapiro-Wilk test on both groups.
    
2.  **Check Variances:**
    
    - Levene's Test (robust)
    - Bartlett's Test (requires normality)
    
    If p-value > 0.05, assume equal variances.
    
3.  **Run t-Test:**  
    Choose the appropriate variant based on the variance test result.

### Which t-Test to Use

| Situation | Test | R Code |
|-----------|------|--------|
| **Unequal variances** (or unsure) | Welch Two Sample t-test — **does NOT assume equal variances** | `t.test(x, y, alternative = ...)` |
| **Equal variances confirmed** | Pooled (Student's) t-test — assumes equal variances | `t.test(x, y, var.equal = TRUE, alternative = ...)` |

> **Default in R:** `t.test()` uses Welch's test (`var.equal = FALSE`) — safe to use in all cases.

**R Code (Welch — unequal/unknown variances):**

```r
t.test(x, y, alternative = "less")
```

**R Code (Pooled — equal variances confirmed):**

```r
t.test(x, y, var.equal = TRUE, alternative = "less")
```

* * *

## Comparing Two Means: Paired Samples

Used when observations are dependent or matched.

### Logic

Performs a one sample t-test on the differences between paired observations.

### Example: Diet Study

- $H_1: \mu_{\text{diff}} > 0$

**R Code:**

```r
t.test(before, after, paired = TRUE, alternative = "greater")
```

- p-value = 0.02  
    Reject $H_0$. Diet worked.

### Warning

Using an independent t-test on paired data is incorrect and can increase the chance of a Type II error.

* * *

# Week 3: Enumerative Data Analysis & MLE

## Enumerative Data Analysis (Chi-Squared)

### Qualitative vs Quantitative

Previously we analysed **quantitative data** (height, weight, marks).

Now we analyse **qualitative (categorical) data**:

- Data consists of **counts / frequencies**
- Examples: Eye colour, Yes/No, Defective/Not defective

We compare **Observed vs Expected frequencies**.

---

### The Chi-Squared Distribution ($\chi^2$)

- Not symmetric
- Right skewed
- Range: $0 \rightarrow \infty$
- Depends on **degrees of freedom (df)**
- As df increases, it becomes more Normal shaped
- Right tail area = significance level $\alpha$

---

## Chi-Squared Goodness-of-Fit Test

### Purpose

Tests whether observed categorical data matches a claimed distribution.

#### Test Statistic

$$\chi^2 = \sum \frac{(O - E)^2}{E}$$

Where:

- $O$ = Observed frequency  
- $E$ = Expected frequency  
- $df = k - 1$  

Large $\chi^2$ means observed differs strongly from expected.

---

## M&Ms Example

**Claim ($H_0$):**

30% Brown, 20% Yellow, 20% Red, 10% Orange, 10% Green, 10% Blue  

**Hypotheses:**

- $H_0$: Distribution matches claim  
- $H_1$: Distribution differs  

Reject $H_0$ if $\chi^2_{calc} > \chi^2_{critical}$.

---

### R Code

```r
chocolate <- c(67, 36, 43, 24, 23, 7)
probs <- c(0.3, 0.2, 0.2, 0.1, 0.1, 0.1)
chisq.test(chocolate, p = probs)
```

---

## Chi-Squared Test of Independence

### Purpose

Tests whether two categorical variables are related.

#### Hypotheses

- $H_0$: Variables are independent  
- $H_1$: Variables are dependent  

---

### Expected Counts Formula

For contingency table:

$$E_{ij} = \frac{(\text{Row Total})(\text{Column Total})}{\text{Grand Total}}$$

Degrees of freedom:

$$df = (r - 1)(c - 1)$$

---

### Assumptions

1. Categorical variables  
2. Independent observations  
3. Rule of 5:
   - At least 80% of expected counts $\ge$ 5  
   - No expected count $<$ 1  

If violated, combine categories or use Fisher's test.

---

### Fisher's Exact Test

Used for small samples.

```r
# Independent
wilcox.test(group_A, group_B, alternative = "two.sided")

#Paired
wilcox.test(group_A, group_B, alternative = "two.sided", paired = TRUE)
```

---

### Mann-Whitney U Test / Wilcoxon Test
Used for median

```r
wilcox.test
```

#### Effect Size: Statistical vs Practical Significance

#### The Problem with Large Samples

- **Statistical significance** tells you if a difference exists.  
- **Practical importance** tells you if the difference matters.  
- With very large $n$, even tiny differences can produce small p values.  
- Example: A 2 second improvement may be statistically significant but practically useless.

---

#### The Solution: Effect Size

Effect size measures the **magnitude** of a difference.

---

#### Chi-Squared Tests: Phi Coefficient

For $2 \times 2$ tables:

$$\phi = \sqrt{\frac{\chi^2}{n}}$$

**Guidelines:**

- 0.1 small  
- 0.3 medium  
- 0.5 large  

---

#### t Tests: Cohen's d

Used when comparing two means.

$$d = \frac{\bar{x}_1 - \bar{x}_2}{s}$$

For independent samples, use the pooled standard deviation.

**Guidelines:**

- 0.2 small  
- 0.5 medium  
- 0.8+ large  

## Maximum Likelihood Estimation

### The Core Idea

How do we find the "best" parameters such as $\mu$ or $\lambda$?

MLE finds the parameter that makes your data most likely.

- **Fisher's Principle:** Choose parameter $\theta$ that makes the observed data most probable.  
- Goal: Find $\theta$ that maximizes $P(\text{data} \mid \theta)$ i.e. the Likelihood Function $L(\theta)$.

---

### MLE Step by Step

#### Likelihood Function

Write the probability of the entire dataset.

If observations are independent:

$$L(\theta) = \prod f(x_i \mid \theta)$$

---

#### Log-Likelihood

Take the natural log:

$$\ell(\theta) = \sum \ln \big(f(x_i \mid \theta)\big)$$

Why?

- Differentiating a product is messy  
- Differentiating a sum is easier  
- Logs turn products into sums  

---

#### Differentiate

Find derivative with respect to $\theta$:

$$\frac{d\ell}{d\theta}$$

---

#### Solve

Set derivative equal to 0 and solve for $\theta$.

This gives the MLE estimate.

---

## MLE Examples

### Poisson Distribution (Horse Kicks)

- Data: Counts of deaths by horse kicks (von Bortkiewicz data)  
- Model:

$$X \sim \text{Poisson}(\lambda)$$

#### MLE Result

$$\hat{\lambda}_{MLE} = \frac{1}{n} \sum x_i = \bar{x}$$

Takeaway: For Poisson, the MLE for $\lambda$ is the **sample mean**.

---

### Normal Distribution

We estimate two parameters: $\mu$ and $\sigma^2$.

---

#### Estimating the Mean

$$\hat{\mu} = \bar{x}$$

Takeaway: MLE mean equals the sample mean.

---

#### Estimating the Variance

$$\hat{\sigma}^2_{MLE} = \frac{1}{n} \sum (x_i - \bar{x})^2$$

#### Bias Issue

- MLE divides by $n$ → biased (underestimates variance)  
- Sample variance:

$$s^2 = \frac{1}{n - 1} \sum (x_i - \bar{x})^2$$

Uses Bessel's correction and is unbiased.

Conclusion: For large $n$, difference is negligible.

---

## R Implementation

For complex models, solve numerically.

Note: R minimizes functions, so use the **negative log-likelihood**.

```r
library(stats4)

# 1. Define Negative Log-Likelihood
nloglik <- function(lambda) {
    return(-sum(dpois(data, lambda, log = TRUE)))
}

# 2. Run Optimizer
fit <- mle(nloglik, start = list(lambda = 1))
summary(fit)
```

---

# Week 4: Advanced MLE & Numerical Methods

## Complex MLE & The Need for Optimization

### When Math Fails (The Gamma Distribution)

- The Gamma distribution models right-skewed data, for example insurance claims.  

- It uses two parameters: $\alpha$ (shape) and $\beta$ (scale).  

- **The Problem:** When you take the derivative of the Gamma log-likelihood and set it equal to 0, there is no simple closed-form solution. You cannot solve it by hand.  

- **The Solution:** Numerical optimization. We use a computer to find where the derivative is approximately zero, which corresponds to the peak of the likelihood.

---

## Numerical Optimization Methods

When we cannot find the maximum likelihood mathematically, we use algorithms to walk uphill to the peak.

### Gradient Ascent / Descent

- **How it works:** Finds the direction of the steepest slope, the gradient, and takes a step in that direction.  

- **Pros:** Simple to implement; only needs first derivatives.  

- **Cons:** Slow, linear convergence; choosing the right step size is tricky.

### Newton's Method

- **How it works:** Uses curvature, the Hessian matrix of second derivatives, to fit a quadratic curve and jump straight to its maximum.  

- **Pros:** Very fast, quadratic convergence; fewer, smarter steps.  

- **Cons:** Fails if the Hessian matrix is not invertible or near saddle points; computationally expensive because it requires second derivatives.

### BFGS (Quasi-Newton)

- **How it works:** Achieves Newton-like speed without computing second derivatives. It approximates the Hessian matrix using previous gradient information.  

- **Pros:** Fast, robust, and requires no second derivatives. This is the default in R's `optim()` and `mle()`.

### Nelder-Mead (Simplex)

- **How it works:** Uses no derivatives. It constructs a simplex, a geometric shape of points, that reflects and shrinks over the surface to find the peak.  

- **Pros:** Extremely robust; works on non-smooth functions and poor starting values.  

- **Cons:** Slow; struggles in high-dimensional problems with many parameters.

---

## MLE Optimization in R

### The Negative Log-Likelihood Trick

- R's optimization functions such as `optim()` and `nlm()` are designed to minimize, not maximize.  

- To compute the Maximum Likelihood Estimate, we minimize the Negative Log-Likelihood.  

- If $\ell(\theta)$ is the log-likelihood, we minimize $-\ell(\theta)$.

### Using `log=TRUE`

- When computing likelihoods in R, always use `log=TRUE` inside density functions, for example `dgamma(x, shape, scale, log=TRUE)`.  

- This computes the log-probability directly, which is more numerically stable than computing a very small probability and then taking its logarithm.

### Optimization Pitfalls

- **Local Maxima:** The algorithm may converge to a smaller local peak instead of the global maximum.  

- **Solution:** Try multiple starting values. If all runs converge to the same point, you likely found the global maximum. If not, the likelihood may be multimodal.  

- **Check Convergence:** In R, `optim()$convergence == 0` indicates successful convergence. Any non-zero value indicates failure.

---

## Why We Love MLE (Theoretical Properties)

Even when computed numerically, MLE has excellent theoretical properties.

1. **Consistency:** As sample size $n \to \infty$, $\hat{\theta} \to \theta$.  

2. **Equivariance:** If $\hat{\theta}$ is the MLE of $\theta$, then $g(\hat{\theta})$ is the MLE of $g(\theta)$.  

3. **Asymptotic Normality:** For large samples, $\hat{\theta} \approx \mathcal{N}\left(\theta, \frac{1}{I(\theta)}\right)$, where $I(\theta)$ is the Fisher Information.  

4. **Asymptotic Efficiency:** For large samples, the MLE achieves the minimum possible variance among regular estimators.

---

## The Likelihood Ratio Test (LRT)

### Concept

Used to compare two nested models to determine whether additional parameters significantly improve model fit.

- $H_0$ (Restricted Model): Parameters are fixed, for example a fair coin with $p = 0.5$.  

- $H_1$ (Unrestricted Model): Parameters are estimated using MLE, for example $p = \hat{p}$.

### The Test Statistic

$$\Lambda = -2 \left[ \ell(\hat{\theta}_0) - \ell(\hat{\theta}) \right]$$

- $\ell(\hat{\theta})$: Log-likelihood of the unrestricted model.  
- $\ell(\hat{\theta}_0)$: Log-likelihood of the restricted model.

### The Distribution

- Under $H_0$, $\Lambda \sim \chi^2_{df}$  

- Degrees of freedom $df$ equal the number of restrictions imposed under $H_0$.

### Profile Likelihood & Confidence Intervals

- Since the LRT statistic follows a $\chi^2$ distribution asymptotically, we can invert the test to construct confidence intervals without assuming normality.  

- In R, `confint(fit)` computes profile likelihood confidence intervals.

---

# Week 5: Generalised Linear Models

## Linear Regression Review and Diagnostics

* Unlike machine learning which focuses on prediction, this module focuses on inference to understand relationships, quantify uncertainty, and compare competing models.
* **Linear Model:**

    $$y = \beta_0 + \beta_1 x + \epsilon$$

  where $\epsilon \sim N(0, \sigma^2)$.
* Interpreting the summary output:

  * The p-value tests the null hypothesis $H_0: \beta = 0$.
  * While often informally described as the probability that the coefficient occurred by chance, it precisely measures the probability of observing an estimate this far from zero if the true coefficient were actually zero.
  * The F-statistic tests whether all slope coefficients are simultaneously zero ($H_0: \beta_1 = \dots = \beta_p = 0$).

* Diagnostic plots are critical for checking assumptions:

  * **Residuals vs Fitted:** Looks for non-linear patterns; should display random scatter.
  * **Normal Q-Q:** Checks if errors are normally distributed; points should follow the diagonal line.
  * **Scale-Location:** Checks for constant variance (homoscedasticity); a funnel shape indicates a violation.
  * **Residuals vs Leverage:** Identifies highly influential points pulling the regression line.

---

## Limitations of Linear Regression

* Linear regression assumes the response $Y$ is continuous and unbounded.
* It assumes a direct linear relationship between predictors and the mean response.
* It assumes errors are normally distributed with constant variance.
* Forcing binary data (pass/fail), count data, or positive right-skewed data into a linear model leads to invalid predictions and violated assumptions.

---

## Introduction to Generalised Linear Models (GLMs)

* Introduced by Nelder and Wedderburn (1972), GLMs unify various regression models under one framework.
* A GLM consists of three core components:

  * **Random Component:** Specifies that the response $Y_i$ follows a distribution from the exponential family (e.g., Normal, Binomial, Poisson, Gamma).
  * **Systematic Component:** The linear predictor combining the predictors.

      $$\eta_i = \beta_0 + \beta_1 x_{i1} + \dots + \beta_p x_{ip}$$

  * **Link Function:** Connects the mean $\mu_i$ to the linear predictor $\eta_i$.

      $$g(\mu_i) = \eta_i$$

### Common GLM Families

* **Normal:** Uses the Identity link ($g(\mu) = \mu$) for continuous, roughly symmetric data.
* **Binomial:** Uses the Logit link for binary responses or proportions.
* **Poisson:** Uses the Log link ($g(\mu) = \log \mu$) for count data.
* **Gamma:** Uses the Log link for positive, right-skewed continuous data.

---

## Logistic Regression

* Used when the response $Y$ is binary, modelling the probability of success $p = P(Y=1)$.
* It applies the logit (log-odds) transformation to ensure predictions remain within the valid $[0, 1]$ bounds.
  * **Logit Link:**

      $$\log\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 x_1 + \dots + \beta_p x_p$$

> **Note on Probability and Odds Conversions:**
> Converting between probability and odds is essential for interpretation.
>
> * **Odds Formula:**
>
>     $$\text{odds} = \frac{p}{1-p}$$
>
> * **Probability Formula:**
>
>     $$p = \frac{\text{odds}}{1+\text{odds}}$$

### Interpreting Coefficients

* Logistic regression coefficients are interpreted as Odds Ratios.
  * **Odds Ratio:**

      $$e^{\beta_j}$$

* If $e^{\beta_j} > 1$: Increasing $x_j$ by 1 unit increases the odds of success.
* If $e^{\beta_j} < 1$: Increasing $x_j$ by 1 unit decreases the odds of success.
* If $e^{\beta_j} = 1$ ($\beta_j = 0$): The predictor has no effect.
* To compute confidence intervals directly on the odds ratio scale in R, use `exp(confint(model))`.

---

## Nested Models and Interactions

* **Likelihood Ratio Test (LRT):**

  * Used to formally compare a reduced model ($H_0$) against a full model ($H_1$) to determine if adding predictors significantly improves the fit.
  * In R, this is executed using the `anova()` function with a Chi-Squared test.
    `anova(model1, model2, test="Chisq")`

* **Interactions and R Formula Syntax (`+`, `*`, `:`):**

  * An interaction implies that the effect of one predictor depends on the level of another predictor.
  * **`+` (Plus):** Adds independent main effects to the model without an interaction (e.g., `y ~ x1 + x2`).
  * **`*` (Asterisk):** Specifies an interaction *and* automatically includes the corresponding main effects (e.g., `y ~ x1 * x2` is identical to `y ~ x1 + x2 + x1:x2`).
  * **`:` (Colon):** Specifies *only* the interaction term between predictors without adding the main effects.

---

## Fitting GLMs

* GLMs are fitted using Maximum Likelihood Estimation (MLE).
* The objective is to find the parameter values $\hat{\beta}$ that minimize the negative log-likelihood.
* **Iteratively Reweighted Least Squares (IRLS):**

  * This is the specific numerical algorithm used to optimize the log-likelihood for GLMs.
  * R's `glm()` function uses IRLS, which converges more accurately and efficiently for these specific models than general-purpose optimizers like `mle()`.

---

# Week 6: Generalised Linear Models (Continued)

## Model Evaluation and Deviance

* Understanding Maximum Likelihood Estimation (MLE) provides a principled framework for comparing models, deriving standard errors, and evaluating overall model fit.

* **Model Deviance:**

    $$D = -2[l(\hat{\beta}) - l(\text{saturated model})]$$

* A "saturated" model fits perfectly by using $n$ parameters for $n$ observations; deviance measures how much worse the fitted model is in comparison.

* For Generalised Linear Models (GLMs), deviance plays the same role that the Residual Sum of Squares (RSS) plays in standard linear regression.

* **Residual Deviance:** Represents the amount of unexplained variation remaining after fitting the model.

* If a model fits well, the residual deviance should be approximately equal to the residual degrees of freedom ($n - p$, where $p$ is the number of estimated parameters).

---

## Model Comparison Tools

### Likelihood Ratio Test (LRT)

> The LRT formula ($\Lambda = -2[\ell(\hat{\theta}_0) - \ell(\hat{\theta})]$) and its $\chi^2$ distribution were introduced in Week 4. Its application to GLMs via `anova()` was covered in Week 5. See those sections for the full details.

* In the GLM context, LRT compares two **nested models** where the reduced model is a special case of the full model. A large $\Lambda$ (small p-value) indicates the extra parameters significantly improve the fit.

### Information Criteria (AIC and BIC)

* Used for comparing **non-nested models** or for automated model selection. Both criteria penalise adding extra parameters to prevent overfitting.

* **Akaike Information Criterion (AIC):**

    $$AIC = -2l(\hat{\beta}) + 2k$$

* **Bayesian Information Criterion (BIC):**

    $$BIC = -2l(\hat{\beta}) + k \log n$$

* For both metrics, a lower value indicates a better model. BIC applies a stronger penalty for complexity in large datasets, thereby favouring simpler models.

---

## Poisson Regression for Count Data

* Designed for response variables that are non-negative integer counts (e.g., number of insurance claims, hospital admissions).

* Applying standard linear regression to count data causes invalid predictions (negative counts), violates normality assumptions, and fails to capture that variance increases with the mean.

* **Poisson Distribution Properties:** The expected mean and variance are identical ($E(Y) = \lambda$, $Var(Y) = \lambda$).

* The Poisson GLM connects the log of the mean to the linear predictors.

* **Systematic Component:**

    $$\log(\lambda_i) = \beta_0 + \beta_1 x_{i1} + \dots + \beta_p x_{ip}$$

### Offsets

* Used when modelling a rate (e.g., claims per policyholder) rather than a raw total count across groups of differing sizes.

* Using `offset(log(exposure))` in the model formula constrains the coefficient of the exposure variable to exactly 1.

* **Offset Formula:**

    $$\log\left(\frac{\lambda_i}{Exposure_i}\right) = \beta_0 + \beta_1 x_{i1} + \dots$$

---

## Handling Categorical Predictors in R

* **Dummy Coding (N-1 Bits):** When dealing with categorical data, avoid standard "one-hot encoding" (e.g., representing 4 districts as 1000, 0100, 0010, 0001). Instead, R uses an $n-1$ bit representation to avoid multicollinearity. For a 4-level variable, level 1 acts as the baseline (000), and three indicator variables represent the rest (100, 010, 001).

* **Orthogonal Polynomials:** For ordered factors, R automatically uses orthogonal polynomial contrasts (`.L` for linear, `.Q` for quadratic, `.C` for cubic trends). If standard dummy coding is preferred, it must be explicitly forced using `contr.treatment`.

* **Formula Implementation:** Ensure categorical data is explicitly cast before modelling to trigger correct dummy coding (e.g., wrapping variables in `as.factor()` directly within the formula or dataset).

---

## Variance Functions and Overdispersion

* In standard linear regression, variance is assumed constant. In GLMs, variance is a function of the mean.

* **Variance Function:**

    $$Var(Y_i) = \phi \cdot V(\mu_i)$$

* For the Poisson and Binomial families, the dispersion parameter $\phi$ is theoretically fixed at 1.

* **Overdispersion:** Occurs frequently in real-world data when the observed variance is strictly greater than the mean ($Var(Y) > \mu$).

* Ignoring overdispersion results in standard errors that are too small, p-values that trigger false positives, and artificially narrow confidence intervals.

### Checking for Overdispersion

* **Dispersion Statistic ($\hat{\phi}$):**

    $$\hat{\phi} = \frac{\text{Residual Deviance}}{\text{Residual df}}$$

* **Evaluation Rules of Thumb:**

    * $\hat{\phi} \approx 1$: The model fits the variance assumption perfectly.
    * $\hat{\phi} \approx 2$: The fit is questionable; exploration of alternative models is recommended.
    * $\hat{\phi} \ge 3$: Hard cut-off. The Poisson model is drastically overconfident, its p-values cannot be trusted, and it must be discarded in favour of an overdispersion-tolerant model.

---


## Dealing with Overdispersion

### Quasi-Poisson Model

* Relaxes the rigid assumption by estimating the dispersion parameter $\phi$ directly from the data.

* The resulting coefficients remain completely identical to the standard Poisson model, but the standard errors are widened by multiplying them by $\sqrt{\hat{\phi}}$.

* Because it relies on quasi-likelihood rather than true maximum likelihood, no AIC value is generated for model comparison.

### Negative Binomial Regression

* A fully parametric alternative specifically designed for overdispersed count data.

* Introduces a dedicated dispersion parameter $r$.

* **Negative Binomial Variance:**

    $$Var(Y) = \mu + \frac{\mu^2}{r}$$

* Unlike Quasi-Poisson, it provides a proper log-likelihood, enabling the use of AIC for direct model comparison.

* Implemented in R using `glm.nb()` from the `MASS` package.

---

## Positive Continuous Responses (Gamma GLM)

* Used when the response variable is strictly positive ($Y > 0$), exhibits a right-skewed distribution, and possesses a variance that increases with the mean (e.g., claim financial amounts, hospital lengths of stay).

* **Gamma Variance:** Grows quadratically with the mean ($\mu^2 / \alpha$).

* **Link Function:** While the canonical link for the Gamma distribution is the inverse ($1/\mu$), the **log link** ($\log(\mu)$) is almost always preferred in practice.

* The log link guarantees strictly positive predictions and preserves a clean, multiplicative interpretation of the coefficients (identical to Poisson interpretation).

* Must be explicitly specified in R: `family = Gamma(link = "log")`.

---

## Quick Reference: Coefficient Interpretation

* For log-link models, coefficients must be exponentiated to evaluate their effect on the interpretable scale.

| Model | Link Function | Interpretation of Exponentiated Coefficient ($e^{\hat{\beta}_j}$) |
| --- | --- | --- |
| **Logistic** | Logit | **Odds ratio:** The odds of $Y=1$ multiply by $e^{\hat{\beta}_j}$ per 1-unit increase in $X_j$. |
| **Poisson** | Log | **Rate ratio:** The expected count multiplies by $e^{\hat{\beta}_j}$ per 1-unit increase in $X_j$. |
| **Gamma** | Log | **Mean ratio:** The expected continuous positive response multiplies by $e^{\hat{\beta}_j}$. |
| **Linear** | Identity | **Additive effect:** The expected response increases directly by the raw coefficient $\hat{\beta}_j$. |

# Week 7: Density Estimation

## Motivation and Introduction to Density Estimation

* In previous weeks, we relied on known mathematical density functions (e.g., Normal, Gamma, Chi-Squared) to describe data, replacing parameters like $\mu$ and $\sigma^2$ with sample estimates.
* This standard approach is called the parametric approach.
* However, real-world sample data often does not resemble any known standard distribution, making the parametric approach invalid.
* Density estimation aims to approximate the probability density function (pdf) of a random variable directly from the data, allowing the data to "speak for themselves" through non-parametric procedures.

## Histograms and Local Averages

* Histograms act as a basic form of density estimation by counting and plotting the number of observations in defined bins.
* Histograms have limitations: the bins must be defined, and the resulting shape can be very blocky depending on the chosen bin width.
* We can formally estimate the density $f_X(x)$ using a local average of points falling within a small window $\delta x$.

* **Local Average Estimate:**

    $$\hat{f}_{X}(x)=\frac{1}{n\delta x}\sum_{i=1}^{n}I(|x-x_{i}|<\delta x/2)$$

* Because the local average uses a strict rectangular function $I(\cdot)$, the resulting plot remains quite blocky.

## Kernel Density Estimation (KDE)

* To fix the blocky nature of local averages, we replace the rectangular indicator function $I(\cdot)$ with a tapering kernel function $K(\cdot)$, producing a locally weighted average.

* **Kernel Density Estimator:**

    $$\hat{f}_{X}(x)=\frac{1}{nh}\sum_{i=1}^{n}K\left(\frac{x-x_{i}}{h}\right)$$

* In this formula, $h$ acts as the width of the window and is known as the bandwidth.
* A valid kernel density estimate must be a true density (non-negative and integrating to 1), which is guaranteed if the chosen kernel $K$ is itself a valid probability density function.
* **PDF Properties:** A valid density must satisfy $f(x) \ge 0$ and $\int f(x)\,dx = 1$.
* **KDE Interpretation:** KDE can be viewed as a sum of small "bumps" (kernels) centred at each data point; the final estimate is the average of these contributions.

### Choice of Kernel

* **Rectangular Kernel:** Corresponds to the basic moving average ($K(t) = 1/2$ for $|t| < 1$).
* **Triangular Kernel:** Produces a less blocky result than the rectangular kernel.
* **Normal (Gaussian) Kernel:** A highly popular choice based on the standard normal distribution.
* **Epanechnikov Kernel:** Theoretically the most efficient kernel.
* Efficiency refers to how accurately the kernel estimates the true underlying density while minimizing variance; all the above kernels have high efficiency (greater than 90%).

### Choice of Bandwidth

* The bandwidth $h$ controls the smoothness of the estimate; too small creates an "undersmoothed" jagged line, while too large creates an "oversmoothed" flat line.

* **Silverman's Rule of Thumb:**

    $$h \approx 1.06 \sigma n^{-1/5}$$

* The rule of thumb assumes the true underlying distribution is Normal, and R uses a pragmatic variant of this ($h = 0.9 \min(s, R/1.34) n^{-1/5}$) as its default `bw="nrd0"`.
* **Cross-Validation:** An alternative method that minimizes the integrated square error by leaving out one observation at a time.
* In R, unbiased cross-validation is called using `bw.ucv`, and biased cross-validation is called using `bw.bcv`.

* **Cross-Validation Formula:** Bandwidth can be chosen by minimising:

    $$M(h) = \int \hat{f}(x)^2\,dx - \frac{2}{n} \sum_{i=1}^{n} \hat{f}_{-i}(x_i)$$

### Edge Effects

* KDE performs poorly near boundaries (e.g. data $\ge 0$).
* Can be improved using transformations (e.g. log transform).

### Special Data Types

* **Periodic data** (e.g. angles): require periodic kernels.
* **Positive-only data:** often handled via transformations.

## Computational Complexity

* A naive implementation of KDE evaluates the kernel at $m$ grid points for every $n$ observation, resulting in $O(nm)$ operations, which is very slow for large datasets.
* KDE is fundamentally a convolution of the kernel and a binned histogram.
* By using the Fast Fourier Transform (FFT), convolutions can be computed in $O(m \log m)$ operations, making it independent of the sample size $n$.
* R's built-in `density()` function utilizes this FFT method internally for rapid performance.

## Applications (Bayesian Classification)

* Beyond displaying data, density estimation can be used to construct non-parametric classifiers.
* Using Bayes' theorem, we can classify the probability of a condition $C$ given a test measurement $X$.

* **Posterior Probability (Bayes Classifier):**

    $$P(C|X) = \frac{f(X|C)P(C)}{f(X)}$$

* The densities $f(X|C)$ (e.g., blood pressure given diseased vs. healthy) can be estimated directly using KDE.
* **Mixture vs. Marginal Denominator:** The denominator $f(X)$ can be calculated in two ways.
  * *Mixture:* Theoretically correct, built from the conditional densities using the Law of Total Probability: $p f_1(x) + (1-p) f_0(x)$.
  * *Marginal:* The density estimated directly from all combined patient data.
* **Mixture vs. Marginal Insight:** Theoretically equivalent, but in practice they differ slightly due to smoothing/estimation error.

## Multivariate Densities

* Density estimation scales to higher dimensions; for 2D data, a two-dimensional kernel is placed at each data point, and the results are averaged.
* In R, 2D density estimation is handled by the `kde2d` function from the `MASS` package.
* By default, `kde2d` selects bandwidths for each dimension independently using the rule of thumb; more principled matrix-valued bandwidth selection is available via the `ks` package.
* **Multivariate Bandwidth Detail:** Separate bandwidths for each dimension provide basic flexibility; a full bandwidth matrix (e.g. the `ks` package) allows for correlation-aware smoothing.

### R Functions to Know

* `density()` -- uses FFT internally
* `bw.nrd0` -- rule of thumb
* `bw.ucv`, `bw.bcv` -- cross-validation
* `geom_density()` -- ggplot
* `kde2d()` -- 2D KDE in `MASS`

## Collinearity Review

* Collinearity occurs when there is a strong correlation between two predictor variables, violating the desire for independent explanatory variables.
* High collinearity leads to unstable coefficients, difficult interpretation, and high variance in standard errors (making p-values unreliable).
* When creating dummy variables for categorical data, including all categories creates "perfect multicollinearity" because one category can be perfectly predicted by the absence of the others.
* The solution is to always exclude one level to serve as the baseline/reference category.

## Complete Exam Quick Reference Table

| Concept / Test | Formula or R Function | Use Case | Key Exam Notes |
|---------------|----------------------|----------|---------------|
| **Sample Mean** | $\bar{x} = \frac{\sum x_i}{n}$ | Estimate $\mu$ | Centre of data |
| **Sample Variance** | $s^2 = \frac{\sum (x_i - \bar{x})^2}{n - 1}$ | Spread | Uses $n-1$ |
| **Sample Std Dev** | $s = \sqrt{\frac{\sum (x_i - \bar{x})^2}{n - 1}}$ | Spread in units | Root of variance |
| **Addition Rule** | $P(A \cup B) = P(A) + P(B) - P(A \cap B)$ | Combine events | Avoid double counting |
| **Complement Rule** | $P(A^c) = 1 - P(A)$ | At least one questions | Often simplifies |
| **Conditional Prob** | $P(B \mid A) = \frac{P(A \cap B)}{P(A)}$ | Given info | Order matters |
| **Independence** | $P(A \cap B) = P(A)P(B)$ | Check independence | Only if unrelated |
| **Bayes Theorem** | $P(A \mid B) = \frac{P(B \mid A)P(A)}{P(B)}$ | Reverse conditional | Common trap |
| **Binomial Mean** | $\mu = np$ | Expected successes | Fixed $n,p$ |
| **Binomial SD** | $\sigma = \sqrt{np(1-p)}$ | Spread | Memorise |
| **Poisson Mean** | $\mu = \lambda$ | Arrivals | Mean = variance |
| **Uniform PDF** | $f(x)=\frac{1}{b-a}$ | Constant density | Area = probability |
| **Z Statistic** | $Z = \frac{\bar{X}-\mu}{\sigma/\sqrt{n}}$ | Mean tests | Known $\sigma$ |
| **Confidence Interval** | $\bar{x} \pm z_{\alpha/2}\frac{\sigma}{\sqrt{n}}$ | Estimate mean | Check if $\mu_0$ inside |
| **Shapiro Wilk Test** | `shapiro.test(x)` | Normality | $H_0$: Normal |
| **Levene Test** | `leveneTest()` | Compare variances | Robust |
| **Bartlett Test** | `bartlett.test()` | Compare variances | Needs normality |
| **One Sample t Test** | `t.test(x, mu=...)` | Mean vs constant | Unknown $\sigma$ |
| **Independent t Test** | `t.test(x, y)` | Two groups, unequal/unknown variances | Welch default; no equal variance assumption |
| **Independent t Test (Equal Var)** | `t.test(x, y, var.equal=TRUE)` | Two groups, equal variances confirmed | Pooled; only after Levene/Bartlett p > 0.05 |
| **Paired t Test** | `t.test(x, y, paired=TRUE)` | Before vs after | Uses differences |
| **Proportion Test** | `prop.test(x, n)` | Test proportion | Large samples ($n \ge 30$), normal approx |
| **Exact Binomial Test** | `binom.test(x, n, p=...)` | Test proportion exactly | Small samples; exact binomial p-value |
| **Chi Square Statistic** | $\chi^2 = \sum \frac{(O - E)^2}{E}$ | Categorical tests | Large = big difference |
| **Goodness of Fit** | `chisq.test(x, p=probs)` | Match distribution | $df = k-1$ |
| **Independence Test** | `chisq.test(matrix)` | Relationship test | $df = (r-1)(c-1)$ |
| **Fisher Exact Test** | `fisher.test(matrix)` | Small samples | Use if counts < 5 |
| **Effect Size (Phi)** | $\phi = \sqrt{\frac{\chi^2}{n}}$ | Strength of association | 0.1 small, 0.3 med, 0.5 large |
| **Cohen's d** | $d = \frac{\bar{x}_1 - \bar{x}_2}{s}$ | Effect size for mean differences | 0.2 small, 0.5 medium, 0.8 large. |
| **Likelihood** | $L(\theta)=\prod f(x_i|\theta)$ | Parameter estimation | Maximise |
| **Log Likelihood** | $\ell(\theta)=\log L(\theta)$ | Simplify math | Turns product into sum |
| **MLE Normal Mean** | $\hat{\mu}=\bar{x}$ | Estimate mean | Same as sample mean |
| **MLE Normal Variance** | $\hat{\sigma}^2=\frac{1}{n}\sum (x_i-\bar{x})^2$ | Estimate variance | Biased |
| **BFGS** | `optim(method="BFGS")` | General-purpose MLE optimization | Fast, robust, no second derivatives required |
| **Nelder-Mead** | `optim(method="Nelder-Mead")` | Non-smooth likelihoods | Very robust but slower, weak in high dimensions |
| **Negative Log-Likelihood** | $-\sum \log f(x_i \mid \theta)$ | Convert maximization to minimization | R minimizes by default |
| **Convergence Check** | `fit$convergence == 0` | Verify optimizer success | 0 indicates successful convergence |
| **Equivariance (MLE)** | If $\hat{\theta}$ is MLE, then $g(\hat{\theta})$ is MLE of $g(\theta)$ | Transformations of parameters | Core theoretical property |
| **LRT Statistic** | $\Lambda = -2[\ell(\hat{\theta}_0) - \ell(\hat{\theta})]$ | Compare nested models | Based on log-likelihood difference |
| **LRT Distribution** | $\Lambda \sim \chi^2_{df}$ | Compute p-values | df equals number of restrictions |
| **Profile Confidence Intervals** | `confint(fit)` | Construct CIs via LRT | Does not rely on normal approximation |
| **Linear F-Statistic** | $H_0: \beta_1 = \dots = \beta_p = 0$ | Overall model significance | Tests if at least one predictor matters |
| **GLM Setup** | $g(\mu_i) = \eta_i$ | Linking mean to predictors | Connects distribution to linear equation |
| **Logit Link** | $\log\left(\frac{p}{1-p}\right) = \eta$ | Logistic regression link | Bounds predictions to $[0,1]$ |
| **Odds** | $\text{odds} = \frac{p}{1-p}$ | Probability to Odds | Ratio of success to failure |
| **Probability from Odds** | $p = \frac{\text{odds}}{1+\text{odds}}$ | Odds to Probability | Reverses the odds calculation |
| **Odds Ratio** | $e^{\beta_j}$ | Interpreting logistic coefficients | $>1$ increases odds, $<1$ decreases odds |
| **Model Comparison (LRT)** | `anova(mod1, mod2, test="Chisq")` | Full vs Reduced model | Tests if added variables improve fit |
| **GLM Fitting Algorithm** | Iteratively Reweighted Least Squares (IRLS) | Optimization in `glm()` | More efficient than standard `mle()` |
| **Formula: +** | `y ~ x1 + x2` | Adding main effects | Adds independent predictors |
| **Formula: *** | `y ~ x1 * x2` | Main effects + interaction | Shortcut for `x1 + x2 + x1:x2` |
| **Formula: :** | `y ~ x1:x2` | Interaction term only | The effect of one depends on the other |
| **Model Deviance** | $D = -2[l(\hat{\beta}) - l(\text{saturated})]$ | Measure GLM fit | Like RSS for GLMs |
| **AIC** | $AIC = -2l(\hat{\beta}) + 2k$ | Compare non-nested models | Lower is better |
| **BIC** | $BIC = -2l(\hat{\beta}) + k\log n$ | Compare non-nested models | Stronger penalty than AIC |
| **Poisson Regression** | `glm(y ~ x, family=poisson)` | Count data | Mean $=$ Variance ($\lambda$) |
| **Offset (Rate Modelling)** | `offset(log(exposure))` | Model rates not totals | Constrains exposure coefficient to 1 |
| **Dispersion Statistic** | $\hat{\phi} = \frac{\text{Residual Deviance}}{\text{Residual df}}$ | Check overdispersion | $\hat{\phi} \ge 3$: discard Poisson |
| **Quasi-Poisson** | `glm(y ~ x, family=quasipoisson)` | Overdispersed counts | No AIC; SEs multiplied by $\sqrt{\hat{\phi}}$ |
| **Negative Binomial** | `glm.nb(y ~ x)` from `MASS` | Overdispersed counts | Has AIC; variance $= \mu + \mu^2/r$ |
| **Gamma GLM** | `glm(y ~ x, family=Gamma(link="log"))` | Positive right-skewed data | Variance $\propto \mu^2$ |
| **Rate Ratio (Poisson/Gamma)** | $e^{\hat{\beta}_j}$ | Interpret log-link coefficients | $>1$ increases rate/mean, $<1$ decreases |
| **Dummy Coding** | `as.factor(x)` in formula | Categorical predictors | R uses $n-1$ bits; avoids multicollinearity |
| **KDE** | $\hat{f}(x)=\frac{1}{nh}\sum K\!\left(\frac{x-x_i}{h}\right)$ | Non-parametric density estimation | Kernel + bandwidth; must integrate to 1 |
| **Silverman's Rule** | $h \approx 1.06\,\sigma\,n^{-1/5}$ | Default bandwidth | Assumes Normal; R: `bw.nrd0` |
| **Cross-Validation (BW)** | `bw.ucv` / `bw.bcv` | Data-driven bandwidth | Minimises integrated square error |
| **Epanechnikov Kernel** | Optimal efficiency kernel | KDE kernel choice | Most efficient; all common kernels $>90\%$ |
| **`density()`** | `density(x)` | Compute KDE in R | Uses FFT; $O(m \log m)$ |
| **Bayes Classifier** | $P(C \mid X) = \frac{f(X \mid C)P(C)}{f(X)}$ | Non-parametric classification | Estimate $f(X \mid C)$ via KDE |
| **2D KDE** | `kde2d()` from `MASS` | Bivariate density | Separate bandwidths per dimension |
| **Collinearity** | Correlated predictors | Unstable coefficients | Drop one dummy level as baseline |
---
