# Measuring noise

We consider measuring the noise in evaluation by adopting classic methods and with some specific considerations for LLMs evals.

For correctness evaluation, there are $N$ examples $\{(x_1, y_1), \ldots, (x_N, y_N)\}$,
the model makes prediction $\hat{y}(x)$, possibly random, and we get a binary correctness judgement $R(\hat{y}, y) \in \{0, 1\}$, 0 for incorrect and 1 for correct. $R$ can be checking for equality or running tests on $\hat{y}$ or asking another model, but we assume a binary outcome here.
We may also compute then get the mean $\text{acc} = \frac1{N} \sum_i R(\hat{y}_i, y_i)$ 

We have model A and B, and we are trying to determine is A is better than B.
Statistics have a lot say about this problem. While the answers are not exactly the same depending on the method and assumptions, the sign test asks if the observed results are likely when the outcomes are random, and bootstrap asks if the observed results is reliable when the examples are random. It is resassuring that they give about the same answer.

## Sign test
The textbook recommendation where you have binary outcomes is the sign test or McNemar's test. Here is a translation for the application to model comparison.
When comparing model A vs. model B, the null-hypothesis is that model A and B are equal and each model has 1/2 chance of being better (win) on each example. A wins if A gets a question correct but B does not, tie if A and B are both correct or incorrect. The question is if the observed results are likely to happen under this null-hypothesis.
Let $A$ be the number of times model A wins against model B and vice versa, and let's assume $A > B$. 
The p-values is then $\text{Pr}[X \leq B \lor X \geq A]$ for $X \sim \text{binomial}(A+B, 0.5)$ with $E[X] = \frac1{2}(A + B)$ and $\text{Var}[X] = \frac1{4} (A+B)$. The question is how likely is $X$ to be as extreme as $A$ or $B$ as observed. When $A+B$ is large, the normal approximation is accurate and reduces to how likely $\text{Pr}[ |Z| \geq \frac{A-B}{\sqrt{A+B}}]$ for standard normal $Z$ (i.e. two sided). If we assume an improvement should happen, we may also ask how likely is an observation from the null hypothesis to be better than $A$ (one-sided) and get exactly half the p-value.

<!-- This null hypotehsis is equivalent to predicting from the mixture model where predictions of A and B are used with probability 1/2 each. This method can be implemented to produce variance and higher accuracies. -->

## Bootstrap
Bootstrap is a general method with the insight/assumption that the particular examples we measured on are not that special, and you should be able to reach equally valid conclusions by resampling your given examples with replacement.
We can compute this by actually drawing the samples, but we can also understand what bootstrap depends on by deriving the formula for a similar question as for the sign test.
Let $A$ be the number of times model A wins against model B and vice versa, and let's assume $A > B$. The question is how likely are we to still observe $A > B$ on bootstraped samples. This also has a good approximation when $A + B > 20$.
Let $X_A, X_B, X_0 \sim \text{multinomial}(N, p_A, p_B, p_0)$ for $p_A = A / N, p_B = B / N$ and the tie probability $p_0 = 1 - p_A - p_B$.
The p-value is $\text{Pr}[X_A - X_B \leq 0]$ where $E[X_A - X_B] = A - B$ and $\text{Var}[X_A - X_B] = N \left(p_a (1-p_a) + p_b (1-p_b) + 2 p_a p_b\right) = N (p_a + p_b - (p_a - p_b)^2) \approx A + B$.
The question is how likely is $\text{Pr}[ Z \geq \frac{A-B}{\sqrt{A+B}}]$ for standard normal $Z$, which gives about the same result as the one-sided sign test (slightly smaller due to $(p_a-p_b)^2$).


The distinction between one-sided and two-sided only matters for somewhat questionable differences when $\frac{A-B}{\sqrt{A+B}} \approx 1 \sim2.5$, and in any event can be converted with a factor of 2 for symmetric distributions. You may also get confidence intervals, where the 95% intervals are approximately $\pm 2\sigma$ with $\sigma=\sqrt{A+B}$.

In restrospect, the main value of the experiment in eval-arena was to establish that $A + B \geq 20$ and $A + B \geq |A - B| +  8$ for all model pairs and all benchmarks tested.
This leads to simple behavior predictable from theory. So instead of conducting their own tests, users of these benchmarks can just ask if what they are comparing might be an exception, and if not, they can just interpret the results based on the aggregate behavior that is true for all model pairs so far. If they suspect an exception, they should verify that their $A+B$ is indeed small.


## Noise from stochastic LLM prediction
You can get iid (identical and independent) predictions from LLMs, which can add some potential confusion for reproducible vs. statistically significant, so is discussed here.
In LLMs, you can draw iid sample they are interestingly different from each other. In the physical world, because you cannot reset the state, once you do an experiment, you cannot ask for another iid sample: did a treatment work on a patient? or did the student solve a particular problem? So usually statistical tests do not consider the ability to draw more iid samples.

You can certainly draw more samples to estimate a lowerbound in the noise level, but this is still insufficient. If we assume the outputs are deterministic, so that A always beats B in 51 examples, and always loses in 49 examples, then these results are perfectly reproducible. We still don't want to conclude that A is better than B. If you believe the validity of bootstrapping, then this is just because your results can easily change if a slightly different set of examples were used instead. 


If this iid prediction noise is worrisome, it can be reduced by averaging and can be estimated from a few samples.
For LLMs, reducing the temperature or fixing the seed are two other methods.
Having large enough data size predictably control this noise as well:
since variance of the sum of independent variables is the sum the variances of each variable
$\text{Var}(\text{acc}) = \frac1{N^2} \sum_i \text{Var}(R(\hat{y}_i, y_i)) = \frac1{N^2} \sum_i p_i (1-p_i)$,
where $p_i$ is the probability to be correct on the example $i$.
This is 0 for deterministic predictions and upperbounded by $\frac1{4N}$ if the model outputs a correct answer with probability 1/2.
<!-- sampling is only part of it -->

On [CRUXEval](https://crux-eval.github.io/eval-arena/model_CRUXEval-output.html#model_table), 10 samples are used to estimate the iid sample noise which is reported as std,
and typically the sampling variance is $ < 0.025 \frac1{N}$, which is quite a bit less than the noise we study here.

## Special examples

For a problem requiring a long answer where guessing correctly is unlikely, answering even 1 problem might be significant and interesting. For example, the problem can ask for the proof of an important open problem and the test checks the proof. If a model (or someone) solves such a hard problem then we should not object to the sample size of 1.
My experience as a student lead me to intuitively believe that you can get a lot of useful signals by testing on a few hard problems with full answers (like Math or algorithm contests, interviews etc.), especially if there is a range of difficulty levels so you can get some signal for most outcomes. If we need to consider this, the hypothesis testing framework can be extended to model weights and tie probabilities, but bootstrap is probably unfixable when the sample set is small and carefully constructed. 

We probably don't need to consider this yet. In the empirical data, all pairs of models have enough noisy inconsistencies where model A beats B on 2 hard examples, but then B beats A on 10 easy examples. If A is so good that it didn't make any mistakes on the long answer, why did it fail on many easier ones. Second, problems solvable by mediocore models or where reference solutions can be found on the internet (i.e. training data) is unlikely to deserve special deference.


## Other works measuring uncertainty
Of the leaderboards, [Chatbot arena](https://chat.lmsys.org/) and [CRUXEval](https://crux-eval.github.io/) computed confidence intervals using bootstrap
and they both did this by having a reference and then compute the intervals relative to the reference . While that is the only way (that I know) to show confidence levels in a linear figure, that method produces intervals that are too large on most pairs, but especially those that are far from the reference. Both of their plots show larger and larger intervals just by being further from the reference (figure `visualize_bootstrap_scores` for chatbot arena). So some of their pairs with overlapping intervals are like not actually overlapping when tested pairwise.
