# [Eval Arena](https://crux-eval.github.io/eval-arena)

For evaluating methods or developing models, we want to know if the gains are real or noise, and how much we can trust each evaluation benchmark. In this work we use matchup data on thousands of model pairs to measure the noise and errors on several popular code generation benchmarks evaluated by executing tests (HumanEval, MBPP, DS-1000, CRUXEval, LiveCodeBench). 

The main results is [https://crux-eval.github.io/eval-arena](https://crux-eval.github.io/eval-arena): a summary table for all benchmarks and reports aggregated by models or by example for each benchmark. The method to run comparisons on all model pairs of each benchmark (hence arena, like chatbot arena but using benchmark data). The interactive figures helped convince me that the noise across all matchups behaves predictably and their measurements as provided are meaningful. Example level evaluation data is released in `data/`, which might be use for developing better evaluation metrics or better leaderboards.

## Main findings

### Popular benchmarks are noisier than assumed by popular papers.
The smaller evaluation set tend to have larger noise. The most popular evaluation in this area is HumanEval+ containing 164 examples.
According to our analysis, a difference of **6.7% is the minimum required** to achieved a p-value of 0.05 across all pairwise comparisons.
Even if our bar is lower, we still need a 5% difference to get a p-value of 0.20.
<!-- MBPP+ needs at at least 4.2% to achieve the p-value of 0.05. -->
These thresholds can be found in the `p5_min` column of the [summary page](https://crux-eval.github.io/eval-arena). 
Many popular papers in the area contain results that have low statistical significance: [HumanEval](https://arxiv.org/pdf/2107.03374), [mbpp](https://arxiv.org/pdf/2108.07732), [StarCoder 2](https://arxiv.org/pdf/2402.19173), [CodeLlamma](https://arxiv.org/pdf/2308.12950) 
[reflexion](https://arxiv.org/pdf/2303.11366), [self-debugging](https://arxiv.org/pdf/2304.05128), [Coder-Reviewer](https://arxiv.org/pdf/2211.16490).

If the general perception is that you always get the right rankings anyways, here are some counter-examples. According to [HumanEval+](https://evalplus.github.io/leaderboard.html), some larger models are worse than smaller models for the series of models `Claude-3-`, `CodeGen2-`, `code-`. This is not surprising given the noise level involved. Inconsistencies in the eval sections of StarCoder2 or CodeLlamma can also be explained by the noise level.

**The null-hypothesis.**
The p-value is the probability that data from the null-hypothesis is more extreme than the one observed. It should capture that nothing interesting is happening. Our particular null-hypothesis is that model A and B has 0.5 chance of winning against each other. This is equivalent to predicting from the mixture model where predictions of A and B are used with probability 1/2 each. This method can be implemented to produce variance and higher accuracies by chance as long as A and B makes enough distinct predictions `#A_win + #B_win`. 


### Reading p-value and noise level from eval-arena
Understandably, most people have better priorities than calculating p-values.
Fortunately, according to our pairwise data, p-values on these datasets are predictable from the accuracy difference alone and you can estimate it from eval-arena data.
This can be seen visually from [this figure](https://crux-eval.github.io/eval-arena/model_humaneval+.html#fig_accs_and_pvalues).
The easiest way is to use the `p5_min` or `p5_max` values in the summary table. If more accuracy is desired, then you can get an estimate from [this difference vs. inconsistency figure](https://crux-eval.github.io/eval-arena/model_humaneval+.html#fig_diff_vs_sum). Convert the percent difference to number of examples `|#A_win - #B_win|` in the dataset and look at which region of the plot it falls in. For example, a difference of 6 examples is not significant at a 0.2 level on HumanEval, whereas a difference of 12 is significant at a 0.2 level and a difference of 18 is significant at the 0.05 level.


**Details about testing.** 
Let $A$ be the number of times model A won against model B and vice versa.
The key observation from these thousands of comparisons is that for any pair of models A and B, there is always disagreements where $A + B > \sim 20$. This fact rules out significant results with a small $A-B$. If A beats B 10 times  without losing at all, that is more significant than if A won 55 times and B won 45 times out of 100 matches for the same difference of 10. The actual p-values are computed exactly as $1 - \text{Pr}[B < X < A]$ for $X \sim \text{bionom}(A+B, 0.5)$.

Since there is always enough disagreements $A+B$, this simple theory is well-justified and the $\chi^2$ approximations is accurate for all pairs. An acurate and interpretable test is then $(|A-B| - 1)^2 / (A + B) > \chi^2_{\alpha}$ for desired level $\alpha$, the resulting parabolas are plotted in [the difference vs. inconsistency figure](https://crux-eval.github.io/eval-arena/model_humaneval+.html#fig_diff_vs_sum). 
Different treatment of ties and bootstrap all yields similar answers.

### Errors and cheats
Evaluation set with mistakes can be annoying and it is very tempting to correct them.
One way to detect mistakes is to find examples that are anti-correlated with the overall model quality. MBPP/+ is a promising source of suspect problems, [here](https://crux-eval.github.io/eval-arena/ex_mbpp+.html#suspect) is the list. I manually inspected the 10 most suspect examples, and 7 of them are indeed wrong: 

* Reference answer is wrong: [459](https://crux-eval.github.io/eval-arena/evalplus/Mbpp/459.html)
* Example contradicts instruction: [581](https://crux-eval.github.io/eval-arena/evalplus/Mbpp/581.html), [102](https://crux-eval.github.io/eval-arena/evalplus/Mbpp/102.html), [615](https://crux-eval.github.io/eval-arena/evalplus/Mbpp/615.html)
* Underspecified: [558](https://crux-eval.github.io/eval-arena/evalplus/Mbpp/558.html), [559](https://crux-eval.github.io/eval-arena/evalplus/Mbpp/559.html), [87](https://crux-eval.github.io/eval-arena/evalplus/Mbpp/87.html), 

Mistakes can be useful for detecting cheating, which may output the reference solution even when it is wrong. On Mbpp/[615](https://crux-eval.github.io/eval-arena/evalplus/Mbpp/615.html) and [459](https://crux-eval.github.io/eval-arena/evalplus/Mbpp/459.html), `codegen-6b`, `CohereForAI--c4ai-command`, `codet5p-6b` outputs the incorrect reference solution whereas most other models outputs the more natural answer. This provides some evidence that most models have not memorized the eval set.

Increasing data size without degrading average quality clearly improves the statistical power whereas a higher quality dataset of the same size does not seem to have a measurable effect. HumanEval+ has the same ranking issues as HumanEval whereas MBPP+ seems to rank all model series correctly where the bigger model is better within each series of models `X-7B, X-13B, X-34B, etc`. So the bigger size of MBPP+ seems to overcome the much higher intuitive quality of HumanEval over MBPP (For example, HumanEval has fewer tau- examples, which all passes manual inspection anyway). This can also be seen visually by comparing [HumanEval+](https://crux-eval.github.io/eval-arena/model_humaneval+.html#fig_accs_and_pvalues) vs. [mbpp+](https://crux-eval.github.io/eval-arena/model_mbpp+.html#fig_accs_and_pvalues) where mbpp has a narrower band of insignificant comparisons.

### Accuracy, ELO, win rate
In [results tables](https://crux-eval.github.io/eval-arena/model_humaneval+.html#model_table), we provide pass1, average win-rate over all other models (used by [BigCode](https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard)), and Elo (Bradly-Terry coefficients following [Chatbot Arena](https://chat.lmsys.org/)). **We recommend raw accuracy (or pass1)** -- since accuracy has better stability as measured by tau-correlation on subsamples of the benchmark (following figure 4 of [this paper](https://aclanthology.org/2021.acl-long.346.pdf)). Unless other metrics are measurably better, accuracy has the clear advantage that it does not depend on other models.

### Difficulty levels and hard problems
In the `by examples` section, we provide a list of examples that are solved by 0 model and solved by only 1 model ([example](https://crux-eval.github.io/eval-arena/ex_humaneval+.html#nosolve)). These are useful for understanding the ability of leading models as well as the quality of the datasets. Whether a benchmark is truly saturated depends on if the remaining examples still contain signal and not just on the raw accuracy. We also provide histograms of accuracies and minimum Elo required to solve examples. On this metric, [LiveCodeBench](https://crux-eval.github.io/eval-arena/ex_lcb_codegen.html#hist) stands out for having a lot of hard examples and a fairly even distribution of difficulties (maybe since it was constructed with a mixture of difficulties).

For problem requiring a long answer where guessing correctly is unlikely, answering even 1 problem might be significant and interesting. For example, the problem can ask for the proof of an important open problem and the test checks the proof. Surely if a model or someone solves it we should not object to the sample size of 1.
This case only seems to be theoretical for now,
since all problems solvable by mediocore models are not this type and there are no cases where any model A strictly outperforms model B on a few problems.  

### Improving evaluation
Since it is difficult to collect more evaluations, using more evaluation benchmarks is a good way to improve confidence. Even when individual benchmarks yield low significance, consistent improvements across benchmarks can be significant.

For the test based benchmarks, a solution passing a weak test is still different from a solution failing it, thus running indepedent tests may also help yield more information per example instead of focusing on complete correctness. This is similar to the tech interview approach, where often only 1 or 2 questions are asked but you may get more information than just correct vs. incorrect.

It may also be possible to model individual problems better, such as the probability of each outcome or the difficulty level using [item response model](https://eacl2024irt.github.io/). However our initial attempts at this failed to improve stability or significance levels.

## Usage 

The raw example level evaluation data is stored in `data/*` in this format:
```
{"benchmark_id":"humaneval", "model":"code-llama-multi-34b", "example_id":"HumanEval\/4", "pass1":1}
{"benchmark_id":"CRUXEval-input", "model":"phind", "example_id":"CRUXEval-input\/0", "pass1":0.8}
```

To generate the summaries and figures
```
python run_arena.py
```

Additional models and evaluations are welcome.


## Citation

```
@misc{evalarena,
  title = {{E}val-{A}rena: noise and errors by leaderboard showdowns},
  author = {Sida I. Wang and Alex Gu and Jiawei Liu and Yuxiang Wei and Naman Jain and Yuhang Lai and Sten Sootla and Ofir Press and Baptiste Rozière and Gabriel Synnaeve},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/crux-eval/eval-arena}}
}
```

## License

The majority of Eval-Arena is licensed under MIT, however portions of the project are available under separate license terms:

https://github.com/xlang-ai/DS-1000/blob/main/LICENSE

https://github.com/google-research/google-research/blob/master/LICENSE
