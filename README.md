# [Eval Arena](https://crux-eval.github.io/eval-arena)

For evaluating methods or developing models, we want to know if the gains are real or noise, and if individual evaluation examples are informative or trustworthy. In this work, we match up thousands of model pairs from > 1M example level results on several LLM evaluation benchmarks. These include code generation datasets,
<ul>
      <li><a href="https://evalplus.github.io/">EvalPlus versions of MBPP/+, HumanEval/+</a> </li>
      <li><a href="https://livecodebench.github.io/leaderboard.html">LiveCodeBench</a></li>
      <li><a href="https://crux-eval.github.io/">CRUXEval</a></li>
      <li><a href="https://ds1000-code-gen.github.io/">DS1000</a></li>
</ul>

As well as popular LLM evaluations such as agi-english, arc_challenge, GSM-8k, hellaswag, mmlu, nq, piqa, siqa, and tqa (see [this paper for data descriptions](https://arxiv.org/pdf/2406.10229)).

 We find that all comparisons are highly inconsistent where a better model is also worse on some examples than a worse model.  As seen in the [pairwise results](https://crux-eval.github.io/eval-arena/model_humaneval+.html#fig_pvalue_vs_diff), the noise level thus has low dependence on particular model pairs, and can be measured meaningfully for each benchmark. We may expect that a good enough model can solve all easy enough examples, but actually models and examples are plausibly independent of each other as seen in the [example level leaderboard](https://crux-eval.github.io/eval-arena/ex_v_model_mbpp+.html). 

For practical references, the results on measuring noise level, model quality and benchmark quality can be found at [https://crux-eval.github.io/eval-arena](https://crux-eval.github.io/eval-arena). We provide performance results by models (accuracy, win rates, ELO, all pairwise comparisons) and by examples (solved by 0 or 1 models, suspect examples and distribution of difficulties). The method is to run comparisons on all model pairs for all benchmarks (hence arena, inspired by chatbot arena but using static eval benchmarks). The example level evaluation data is released in `data/`, which might be useful for developing better evaluation metrics / leaderboards.

See [noise.md](noise.md) for technical details.

## Findings

### Popular benchmarks are noisier than assumed by popular papers.
We measure the noise using both p-values and standard deviations, which can help ["increase the rigor of the conclusions drawn from data"](https://projecteuclid.org/journals/annals-of-applied-statistics/volume-15/issue-3/The-ASA-presidents-task-force-statement-on-statistical-significance-and/10.1214/21-AOAS1501.full).
The most popular evaluation in this area is HumanEval/+ containing 164 examples.
According to our analysis, a difference of **6.7% is the minimum required** to achieve a p-value of 0.05 across all pairwise comparisons.
Even if our bar is lower, we still need a 5% difference to get a p-value of 0.20.
<!-- MBPP+ needs at least 4.2% to achieve the p-value of 0.05. -->
These thresholds can be found in the `p5_min` column of the [summary page](https://crux-eval.github.io/eval-arena). 
Many popular papers in the area contain some results that have low statistical significance: [HumanEval](https://arxiv.org/pdf/2107.03374), [mbpp](https://arxiv.org/pdf/2108.07732), [StarCoder 2](https://arxiv.org/pdf/2402.19173), [CodeLlama](https://arxiv.org/pdf/2308.12950) 
[self-debugging](https://arxiv.org/pdf/2304.05128), [Coder-Reviewer](https://arxiv.org/pdf/2211.16490).
This is not meant to single out these works, since reporting on a common set of benchmarks is beneficial and low significance level does not mean wrong. However, the results can be better interpreted knowing the noise level for each benchmark. The benchmark builders should audit/measure noise rather than users.  

There seems to be a perception that we always get the right rankings anyways. For some counter-examples on [HumanEval+](https://evalplus.github.io/leaderboard.html), some larger models are worse than smaller models for the series of models `Claude-3-`, `CodeGen2-`, `code-`. This is less surprising given the noise level involved. Inconsistencies in the eval sections of StarCoder2 or CodeLlamma can also be explained by the noise level.

**The null-hypothesis.**
The p-value is the probability that data from the null-hypothesis is more extreme than the one observed. It should capture that there is no real difference between the pair. Our particular null-hypothesis is that model A and B has 0.5 chance of winning against each other (i.e. sign test or McNemar's test). 


### Reading p-value and noise level from eval-arena
Understandably, most people have better priorities than calculating p-values.
Fortunately, p-values on these benchmarks are predictable from the accuracy difference alone which mostly depends on the benchmark.
This can be seen visually from [this figure](https://crux-eval.github.io/eval-arena/model_humaneval+.html#fig_accs_and_pvalues).
The easiest way is to use the `p5_min` or `p5_max` values in the summary table. If other p-values are desired, then we can get an estimate from the [p-values vs difference figure](https://crux-eval.github.io/eval-arena/model_humaneval+.html#fig_pvalue_vs_diff). 
For example, a 4% difference is unlikely to be significant even at the 0.2 level on HumanEval, whereas a 10% difference is significant at the 0.05 level.
The [difference vs. sum figure](https://crux-eval.github.io/eval-arena/model_humaneval+.html#fig_diff_vs_sum) is the most informative and shows that the noise across matchups is fairly consistent.

An alternative metric is the implied stdev of the null hypothesis, which can be estimated from the `std(A) = p5  / (sqrt(2) * 1.96)`. This is the std of the null hypothesis mixing the prediction of `A` and `B` with equal probability. The `sqrt(2)` is due to taking `std(acc(A) - acc(B))` and `1.96 std` corresponds to `p <= 5`.

### Signal to noise ratio

While the above is a good measurement of noise, evaluation benchmarks also differ in their signal.
We measured the [measured signal to noise](https://crux-eval.github.io/eval-arena/signal_noise.html) by `(acc(A) - acc(B)) / std(acc(A) - acc(B))` where `A` is the bigger model and `B` is the smaller model of the same series. We use size ranging from 7B to 70B normalized for each doubling `log2(size(A)/size(B))`.
While the main factor is the size of the dataset, mmlu, tqa, and gsm8k produced high signal to noise ratios whereas siqa and piqa produced low signal to noise for their sizes
(i.e. per sqrt(size)).


**Details on testing.** 
See [noise.md](noise.md) for technical information about testing and modeling noise.
Let $A$ be the number of times model A won against model B and vice versa.
The key observation from these thousands of comparisons is that for any pair of models A and B, there is always disagreements where $A + B > \sim 20$. That is, model A is never strictly better than model B on all examples across all benchmarks and all model pairs (if A and B had somewhat similar accuracies). This fact rules out significant results with a small $A-B$. If A beats B 10 times  without losing at all, that is more significant than if A won 55 times and B won 45 times out of 100 matches for the same difference of 10. Assuming $A \geq B$, the p-values are computed as $\text{Pr}[X \leq B \lor X \geq A]$  for $X \sim \text{binom}(A+B, 0.5)$.

<!-- Since there is always enough disagreements $A+B$, this simple theory is well-justified and the $\chi^2$ approximations is accurate for all pairs. An accurate and interpretable test is then $(|A-B| - 1)^2 / (A + B) > \chi^2_{\alpha}$ for desired level $\alpha$, the resulting parabolas are plotted in [the difference vs. inconsistency figure](https://crux-eval.github.io/eval-arena/model_humaneval+.html#fig_diff_vs_sum). 
Different treatment of ties and bootstrap gave similar answers.  -->


### Errors and contamination
While errors in evaluation set are irritating, errors and idiosyncrasies provide some signal about contamination as well.
We test a method for detecting errors by finding examples that are anti-correlated with the overall model quality. MBPP/MBPP+ seems to be the most promising source of suspect problems, [here](https://crux-eval.github.io/eval-arena/ex_mbpp+.html#suspect) is the list. I manually inspected the 10 most suspect examples, and 7 of them are indeed wrong: 

* Reference answer is wrong: [459](https://crux-eval.github.io/eval-arena/evalplus/Mbpp/459.html)
* Example contradicts instruction: [581](https://crux-eval.github.io/eval-arena/evalplus/Mbpp/581.html), [102](https://crux-eval.github.io/eval-arena/evalplus/Mbpp/102.html), [615](https://crux-eval.github.io/eval-arena/evalplus/Mbpp/615.html)
* Underspecified: [558](https://crux-eval.github.io/eval-arena/evalplus/Mbpp/558.html), [559](https://crux-eval.github.io/eval-arena/evalplus/Mbpp/559.html), [87](https://crux-eval.github.io/eval-arena/evalplus/Mbpp/87.html), 

These errors can provide some evidence when models memorize the incorrect reference solution. On Mbpp/[615](https://crux-eval.github.io/eval-arena/evalplus/Mbpp/615.html) and [459](https://crux-eval.github.io/eval-arena/evalplus/Mbpp/459.html), `codegen-6b`, `CohereForAI--c4ai-command`, `codet5p-6b` outputs the incorrect reference solution whereas most other models outputs the correct answer. On [581](https://crux-eval.github.io/eval-arena/evalplus/Mbpp/581.html), Claude and GPT both gave a less wrong answer than the reference. This provides some evidence that models have not memorized many of the answers. A few wrong or underspecified problems may provide evidence if a model is memorizing the evaluation data (likely post-training) instead of making correct predictions.

Increasing data size without degrading average quality clearly improves the statistical power whereas a higher quality dataset of the same size does not seem to have a measurable effect (yet). HumanEval+ has the same ranking issues as HumanEval whereas MBPP+ seems to rank all model series correctly where the bigger model is better within each series of models `X-7B, X-13B, X-34B, etc`. So the bigger size of MBPP+ seems to overcome the much higher intuitive quality of HumanEval over MBPP (For example, HumanEval has fewer tau- examples, which all pass manual inspection anyway). This can also be seen visually by comparing [HumanEval+](https://crux-eval.github.io/eval-arena/model_humaneval+.html#fig_accs_and_pvalues) vs. [mbpp+](https://crux-eval.github.io/eval-arena/model_mbpp+.html#fig_accs_and_pvalues) where mbpp has a narrower band of statistically insignificant comparisons.

### Accuracy, win rate, ELO
In [results tables](https://crux-eval.github.io/eval-arena/model_humaneval+.html#model_table), we provide pass1, average win-rate over all other models (used by [BigCode](https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard)), and Elo (Bradly-Terry coefficients following [Chatbot Arena](https://chat.lmsys.org/)). **We recommend raw accuracy (or pass1)** -- since all metrics are highly correlated for our setting and since accuracy actually has better stability as measured by tau-correlation between subsamples of benchmarks (following figure 4 of [this paper](https://aclanthology.org/2021.acl-long.346.pdf)). Unless a more complex metric is measurably better, accuracy has a clear advantage in that it does not depend on other models.

### Difficulty levels and hard problems
In the `by examples` section, we provide a list of examples that are solved by 0 model and solved by only 1 model ([example](https://crux-eval.github.io/eval-arena/ex_humaneval+.html#nosolve)). These are useful for understanding the ability of leading models as well as the quality of the datasets. Whether a benchmark is truly saturated depends on if the remaining examples still contain signal and not just on the raw accuracy. We also provide histograms of accuracies and minimum Elo required to solve examples. On this metric, [LiveCodeBench](https://crux-eval.github.io/eval-arena/ex_lcb_codegen.html#hist) stands out for having a lot of hard examples and a fairly even distribution of difficulties (maybe since it was constructed with a mixture of difficulties).


### Improving evaluation
Since it is difficult to collect more high quality evaluations, using more evaluation benchmarks is a good way to improve confidence. Even when individual benchmarks yield low significance, consistent improvements across benchmarks can be significant.

For test based benchmarks, a solution passing a weak test is still different from a solution failing it, thus running indepedent tests may also help yield more information per example instead of focusing on complete correctness. This is similar to the tech interview approach, where often only 1 or 2 questions are asked but we may get more information than just correct vs. incorrect, especially with a progressive question with easy to hard parts.

It may also be possible to model individual problems better, such as the probability of each outcome or the difficulty level using [item response model](https://eacl2024irt.github.io/). However our initial attempts did not improve stability or significance levels, maybe because most examples give [very noisy](https://crux-eval.github.io/eval-arena/ex_v_model_humaneval+.html) signals, where most few examples are noisy, and they are not all errors.

## Usage 

The raw example level evaluation data is stored in `data/*` in this format:
```
{"benchmark_id":"humaneval", "model":"code-llama-multi-34b", "example_id":"HumanEval\/4", "pass1":1}
{"benchmark_id":"CRUXEval-input", "model":"phind", "example_id":"CRUXEval-input\/0", "pass1":0.8}
```

To generate the summaries and figures, install requirements and set `OUTPUT_PATH` then
```
python run_arena.py
```

Additional models and evaluations are welcome.


## Contributors and citation 

If you find this work helpful, consider citing us.

```
@misc{evalarena,
  title = {{E}val-{A}rena: noise and errors for LLM evaluations},
  author = {Sida I. Wang and Alex Gu and Lovish Madaan and Dieuwke Hupkes and Jiawei Liu and Yuxiang Wei and Naman Jain and Yuhang Lai and Sten Sootla and Ofir Press and Baptiste Rozière and Gabriel Synnaeve},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/crux-eval/eval-arena}}
}
```

Thanks for helpful discussions with Ari Holtzman, Nicolas Usunier, and Pedro Rodriguez. Chatbot arena provided the inspiration to test all model pairs and summarizing the results in a understandable way.


## License

The majority of Eval-Arena is licensed under MIT, however portions of the project are available under separate license terms:

https://github.com/xlang-ai/DS-1000/blob/main/LICENSE

https://github.com/google-research/google-research/blob/master/LICENSE
