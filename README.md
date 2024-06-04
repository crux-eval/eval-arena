# Eval Arena

There are many evaluation sets for LLMs and code generation in particular. For development we want to know if the gains are real or just noise, and how much we should trust each benchmark and particular examples. In this work we use matchup data on thousands model pairs to measure the noise on several popular execution-based code generation benchmarks (HumanEval, MBPP, DS-1000, CRUXEval, LiveCodeBench).

The main data can be found at [https://crux-eval.github.io/eval-arena](https://crux-eval.github.io/eval-arena), with some highlights below.

## Findings

### Several benchmarks have higher noise than commonly assumed.
The smaller evaluation set tend to have larger noise. The most popular evaluation in this area is HumanEval+ containing 164 examples.
According to our analysis, a difference of **6.7% is the minimum required** to achieved a p-value of 0.05 across all pairwise comparisons.
You still need a 5% difference to get a p-value of 0.20, which is more than dicey. MBPP+ needs at at least 4.2%.
These thresholds can be found in the `p5_min` column of the [summary page](https://crux-eval.github.io/eval-arena). 
Many popular papers in the area contain at some results that do not meet the bars for statistical significance: [HumanEval](https://arxiv.org/pdf/2107.03374), [mbpp](https://arxiv.org/pdf/2108.07732),
[reflexion](https://arxiv.org/pdf/2303.11366), [self-debugging](https://arxiv.org/pdf/2304.05128), [Coder-Reviewer](https://arxiv.org/pdf/2211.16490) are just a few examples.

If the general perception is that you get the right ranking anyways, here are some counter-examples. According to [HumanEval+](https://evalplus.github.io/leaderboard.html), some larger `Claude-3`, `CodeGen2-`, `code-` models are worse than smaller ones. For most other model pairs we do not know the ground-truth, but some are likely in wrong order as well.

### Measuring and reading off the p-value and noise
Understandbly, most people have better things to do than calculating p-values.
Fortunately, according to our pairwise data, p-values on these datasets are predictable from the accuracy difference alone and you can read it off.
This can be seen visually from [this figure](https://crux-eval.github.io/eval-arena/model_humaneval+.html#fig_accs_and_pvalues).
The easiest way is just to use the `p5_min` or `p5_max` values in the summary table. If more accuracy is desired, then you can get an estimate from [this difference vs. inconsistency figure](https://crux-eval.github.io/eval-arena/model_humaneval+.html#fig_diff_vs_sum). Convert the percent difference to number of examples `|#A_win - #B_win|` in the dataset and look at which region of the plot it falls in. For example, a difference of 6 examples is not significant at a 0.2 level on HumanEval, whereas a difference of 18 is significant at a 0.05 level. If you also compute `|#A_win + #B_win|`, then you'd know exactly what the p-value is. 

The key observation from these thousands of comparisons is that for any pair of models A and B, there is always disagreements where `pred_A(x) != pred_B(x)` on more than 20 examples `x`. This rules out highly significant results at small number of examples. If A beats B 10 times straights without losing, that is more significant than if A won 55 times and B won 45 times out of 100 matches for the same difference of 10. This also means that chi-2 is accurate for all pairs and behaviors are predictable.

#### The null-hypothesis is reasonable
The p-value is the probability that data from the null-hypothesis is more extreme than the ones observed. Here the null is just predicting from model A or B with probably 1/2 each, which is a method that can implemented to easily produce results with low p-values without making any real improvements. 

### Mistakes in datasets
Evaluation set with mistakes are intuitive annoying.
One way to use data to detect mistakes is to find examples that are anti-correlated with the overall model quality. We find some suspect problems on MBPP+ [here](https://crux-eval.github.io/eval-arena/ex_mbpp+.html#suspect). I manually inspected the 10 most suspect examples, and 7 of them are indeed problematic. 

* Reference answer is wrong: [459](https://crux-eval.github.io/eval-arena/evalplus/Mbpp/459.html)
* Assert contradicts instruction: [581](https://crux-eval.github.io/eval-arena/evalplus/Mbpp/581.html), [102](https://crux-eval.github.io/eval-arena/evalplus/Mbpp/102.html), [615](https://crux-eval.github.io/eval-arena/evalplus/Mbpp/615.html)
* Underspecified: [558](https://crux-eval.github.io/eval-arena/evalplus/Mbpp/558.html), [559](https://crux-eval.github.io/eval-arena/evalplus/Mbpp/559.html), [87](https://crux-eval.github.io/eval-arena/evalplus/Mbpp/87.html), 

While it is tempting to correct all mistakes, weird mistakes can be useful for detecting cheating models, which may output wrong and unintuitive reference solution instead of following the prompt. On [615](https://crux-eval.github.io/eval-arena/evalplus/Mbpp/615.html), CodeGen, CohereForAI--c4ai-command, codegemma-7b outputs the unintuitive reference answer whereas most other models outputs the more intuitive answer judged as incorrect.

Increasing data size without degrading average quality clearly improves the statistical power whereas a higher quality dataset of the same size does not seem to have a measurable effect. HumanEval+ has the same ranking issues as HumanEval whereas MBPP+ seems to rank all model series correctly where bigger models are better within the same series. On the other hand, it is clear that HumanEval+ has higher quality compared to MBPP+ (HumanEval has few tau-, which all passes manual inspection anyway).

### Unsolved examples and maximum accuracy

### Getting more signal
Use more evals, increase data size, breaking down the tests to get more information per example, example level modelling such as IRT or reweighting.

## Usage 

## Contributors

TODO