# Measuring noise

We consider measuring the noise in evaluation, with some specific considerations for LLMs evals.

For evaluation, there are $N$ examples $\{(x_1, y_1), \ldots, (x_N, y_N)\}$,
the model makes prediction $\hat{y}(x)$, possibly random, and we get a binary result
$\text{R}(\hat{y}, y) \in \{0, 1\}$, 0 for incorrect and 1 for correct.
We may also compute an average $\text{acc} = \frac1{N} \sum_i R(\hat{y}_i, y_i)$ 

## Noise from stochastic prediction
Model predictions can be random when sampling from the LLM.
While this adds to potential confusion and can be problematic, it is straightforward to deal with.
This prediction noise can be reduced to epsilon by averaging and can be estimated from a few samples.
For LLMs, reducing the temperature or fixing the seed are two other methods.
Having large enough data size predictably controls this noise as well:
since variance of the sum of independent variables is the sum the variances of each variable
$\text{Var}(\text{acc}) = \frac1{N^2} \sum_i \text{Var}(R(\hat{y}_i, y_i)) = \frac1{N^2} \sum_i p_i (1-p_i)$,
where $p_i$ is the probability to be correct on the example $i$.
This is 0 for deterministic predictions and upperbounded by $\frac1{4N}$ if the model outputs a correct answer with probability 1/2. Sampling noise can nevertheless be a problem sometime, but measuring it by drawing a few samples should be sufficient to verify if this is a problem or not. 
<!-- sampling is only part of it -->

On [CRUXEval](https://crux-eval.github.io/eval-arena/model_CRUXEval-output.html#model_table), 10 samples are used to estimate this noise which is reported as std,
and typically the sampling variance is $ < 0.025 \frac1{N}$, which is less than the other noise we need to worry about.

So let's assume the predictions are deterministic. We have model A and B, and we are trying to determine is A is better than B.
Statistics have a lot say about this problem. While the answers are not exactly the same depending on the method and assumptions, the sign test and bootstrap methods agree with each other.   

## Sign test
The textbook recommendation where you have correctness judgements is the sign test or McNemar's test.
When comparing model A vs. model B, the null-hypothesis is that model A and B are actually equal and each have 1/2 chance of winning against the other. The question is if the observed results are likely to happen under this null-hypothesis.
Let $A$ be the number of times model A won against model B and vice versa, and let's assume $A > B$. 
The p-values is then $\text{Pr}[X \leq B \lor X \geq A]$ for $X \sim \text{binomial}(A+B, 0.5)$ with $E[X] = \frac1{2}(A + B)$ and $\text{Var}[X] = \frac1{4} (A+B)$. The question is how likely is $X$ to be as extreme as $A$ or $B$ as observed. When $A+B$ is large and reasonably larger than $A-B$, the question is just how likely $\text{Pr}[ |Z| \geq \frac{A-B}{\sqrt{A+B}}]$ for standard normal $Z$ (i.e. two sided).

## Bootstrap
Bootstrap is a general method with the insight/assumption that the particular examples we measured on are not that special, and you can reach equally valid conclusions by resampling them with replacement.
The natural question here is how likely are we to still observe $A > B$ on bootstraped samples. This also has a good approximation when $A + B > 20$.
Let $X_A, X_B, X_0 \sim \text{multinomial}(N, p_A, p_B, p_0)$ for $p_A = A / N, p_B = B / N$ and the tie probability $p_0 = 1 - p_A - p_B$.
The p-value is $\text{Pr}[X_A - X_B \leq 0]$ where $E[X_A - X_B] = A - B$ and $\text{Var}[X_A - X_B] = N \left(p_a (1-p_a) + p_b (1-p_b) + 2 p_a p_b\right) = N (p_a + p_b - (p_a - p_b)^2) \approx A + B$.
The question is how likely is $\text{Pr}[ Z \geq \frac{A-B}{\sqrt{A+B}}]$ for standard normal $Z$.

The bootstrap test gives a p-value around half the sign test due to being one-sided and ignoring $(p_a-p_b)^2$. The main difference is one-tailed vs two-tailed depending if you assume $A > B$. In both cases, it is only for $\frac{A-B}{\sqrt{A+B}} \in [1, 2.5]$ that the significance level might be questionable.

In restrospect, the main value of the experiment was to establish that $A + B \geq 20$ and $A + B \geq |A - B| +  8$ for all model pairs and all benchmarks tested.
This leads to simple behavior predictable from theory. So instead of conducting their own tests, users of these benchmarks can just ask if what they are comparing might be an exception, and if not, they can just interpret the results based on the aggregate behavior that is true for all model pairs so far. If they suspect an exception, they should verify that their $A+B$ is indeed very small.

## Previous work measuring uncertainty
Of the leaderboards, [Chatbot arena](https://chat.lmsys.org/) and [CRUXEval](https://crux-eval.github.io/) computed confidence intervals using bootstrap
and they both did this by having a reference and then compute the intervals relative to the reference. While that is the only way (that I know) to show confidence levels in a linear figure, that method produces intervals that are too large on most pairs, but especially those that are far from the reference. Both of their plots show larger and larger intervals just by being further from the reference. So some of pairs with overlapping intervals might actually be different enough when tested pairwise.