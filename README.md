# Eval Arena

There are many evaluation sets for LLMs and code generation. For evaluating methods or developing models, we want to know if the gains are real or noise, and how much we can trust each benchmark. In this work we use matchup data on thousands of model pairs to measure the noise on several popular code generation benchmarks evaluated by executing tests (HumanEval, MBPP, DS-1000, CRUXEval, LiveCodeBench). 

The main results is [https://crux-eval.github.io/eval-arena](https://crux-eval.github.io/eval-arena): a summary table for all benchmarks and reports aggregated by models or by example for each benchmark. The main method is running all pairwise comparisons (hence arena, like chatbot arena but using benchmark data). The interactive figures helped convince me that the noise across all matchups behaves predictably and the measurements are meaningful.

## Findings

### Popular benchmarks are noisier than assumed by popular papers.
The smaller evaluation set tend to have larger noise. The most popular evaluation in this area is HumanEval+ containing 164 examples.
According to our analysis, a difference of **6.7% is the minimum required** to achieved a p-value of 0.05 across all pairwise comparisons.
You still need a 5% difference to get a p-value of 0.20, which is more than dicey. MBPP+ needs at at least 4.2% to achieve the p-value of 0.05.
These thresholds can be found in the `p5_min` column of the [summary page](https://crux-eval.github.io/eval-arena). 
Many popular papers in the area contain some results that have low statistical significance: [HumanEval](https://arxiv.org/pdf/2107.03374), [mbpp](https://arxiv.org/pdf/2108.07732),
[reflexion](https://arxiv.org/pdf/2303.11366), [self-debugging](https://arxiv.org/pdf/2304.05128), [Coder-Reviewer](https://arxiv.org/pdf/2211.16490) are just a few examples.

If the general perception is that you get the right ranking anyways, here are some counter-examples. According to [HumanEval+](https://evalplus.github.io/leaderboard.html), some larger models are worse than smaller models for the series `Claude-3-`, `CodeGen2-`, `code-`.

**The null-hypothesis.**
The p-value is the probability that data from the null-hypothesis is more extreme than the one observed. It should capture that nothing interesting is happening -- and our particular null-hypothesis is that model A and B have equal win probability against each other. This is equivalent to predicting from the mixture model where predictions of A and B are used with probability 1/2 each. This method can be implemented to produce variance and higher accuracies by chance without making any real improvements. Model B can be a similar model as A, using a different prompt, or even just another sample from A.

### Reading p-value and noise level from eval-arena
Understandably, most people have better priorities than calculating p-values.
Fortunately, according to our pairwise data, p-values on these datasets are predictable from the accuracy difference alone and you can estimate it from eval-arena data.
This can be seen visually from [this figure](https://crux-eval.github.io/eval-arena/model_humaneval+.html#fig_accs_and_pvalues).
The easiest way is just to use the `p5_min` or `p5_max` values in the summary table. If more accuracy is desired, then you can get an estimate from [this difference vs. inconsistency figure](https://crux-eval.github.io/eval-arena/model_humaneval+.html#fig_diff_vs_sum). Convert the percent difference to number of examples `|#A_win - #B_win|` in the dataset and look at which region of the plot it falls in. For example, a difference of 6 examples is not significant at a 0.2 level on HumanEval, whereas a difference of 12 is significant at a 0.2 level and a difference of 18 is significant at the 0.05 level. If you also compute `|#A_win + #B_win|`, then you'd know exactly what the p-value is.

The key observation from these thousands of comparisons is that for any pair of models A and B, there is always disagreements where `|#A_win + #B_win|` on more than 20 examples. This fact rules out significant results with a small `|#A_win - #B_win|`. If A beats B 10 times  without losing at all, that is more significant than if A won 55 times and B won 45 times out of 100 matches for the same difference of 10. This also means that chi-2 is accurate for all pairs and behaviors are predictable from theory.


### Errors and cheats
Evaluation set with mistakes can be annoying.
One way to use data to detect mistakes is to find examples that are anti-correlated with the overall model quality. MBPP/+ is a promising source of suspect problems, [here](https://crux-eval.github.io/eval-arena/ex_mbpp+.html#suspect) is the list. I manually inspected the 10 most suspect examples, and 7 of them are indeed wrong: 

* Reference answer is wrong: [459](https://crux-eval.github.io/eval-arena/evalplus/Mbpp/459.html)
* Example contradicts instruction: [581](https://crux-eval.github.io/eval-arena/evalplus/Mbpp/581.html), [102](https://crux-eval.github.io/eval-arena/evalplus/Mbpp/102.html), [615](https://crux-eval.github.io/eval-arena/evalplus/Mbpp/615.html)
* Underspecified: [558](https://crux-eval.github.io/eval-arena/evalplus/Mbpp/558.html), [559](https://crux-eval.github.io/eval-arena/evalplus/Mbpp/559.html), [87](https://crux-eval.github.io/eval-arena/evalplus/Mbpp/87.html), 

While it is tempting to correct mistakes, mistakes can be useful for detecting cheating, which may output the reference solution even when it is wrong. On Mbpp/[615](https://crux-eval.github.io/eval-arena/evalplus/Mbpp/615.html) and [459](https://crux-eval.github.io/eval-arena/evalplus/Mbpp/459.html), `codegen-6b`, `CohereForAI--c4ai-command`, `codet5p-6b` outputs the incorrect reference solution whereas most other models outputs the more natural answer. This provides some evidence that most models have not memorized the eval set and some may have tried.

Increasing data size without degrading average quality clearly improves the statistical power whereas a higher quality dataset of the same size does not seem to have a measurable effect. HumanEval+ has the same ranking issues as HumanEval whereas MBPP+ seems to rank all model series correctly where the bigger model is better within each series of models `X-7B, X-13B, X-34B, etc`. So the bigger size of MBPP+ seems to overcome the much higher intuitive quality of HumanEval+ over MBPP+ (For example, HumanEval has fewer tau- examples, which all passes manual inspection anyway). This can also be seen visually by comparing [HumanEval+](https://crux-eval.github.io/eval-arena/model_humaneval+.html#fig_accs_and_pvalues) vs. [mbpp+](https://crux-eval.github.io/eval-arena/model_mbpp+.html#fig_accs_and_pvalues) which has a much narrower band of high p-values.

### Accuracy, ELO, win rate
In [results tables](https://crux-eval.github.io/eval-arena/model_humaneval+.html#model_table), we provide pass1, average win-rate over all other models (used by [BigCode](https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard)), and Elo (Bradly-Terry coefficients following [Chatbot Arena](https://chat.lmsys.org/)). This might be helpful to facilitate comparisons across leaderboards but we recommend raw accuracy (or pass1) -- since accuracy does not depend on other models and also has better stability as measured by tau-correlation on subsamples of the benchmark (following figure 4 of [this paper](https://aclanthology.org/2021.acl-long.346.pdf)). Elo may also have some advantage where it is put into an uniform scale that can help comparisons across benchmarks.

### Difficulty levels and hard problems
In the `by examples` section, we provide a list of examples that are solved by 0 model and solved by only 1 model ([example](https://crux-eval.github.io/eval-arena/ex_humaneval+.html#nosolve)). These are useful for understanding the ability of leading models as well as the quality of the datasets. Whether a benchmark is truly saturated depends on if the remaining examples still contain signal and not just on the raw accuracy. We also provide histograms of accuracies and minimum elo required to solve examples. On this metric, [LiveCodeBench](https://crux-eval.github.io/eval-arena/ex_lcb_codegen.html#hist) stands out for having a lot of hard examples and a fairly even distribution of difficulties, maybe because it was constructed with a mixture of difficulties.

For problem requiring a long answer where guessing correctly is unlikely, answering even 1 problem might be significant and interesting. For example, the problem can ask for the proof of an important open problem and the test checks the proof. Surely if a model or someone solves it we should not object to the sample size of 1. All problems that are solved by mediocore models are not this type. The few problems not solved by any models are candidates which might be worth monitoring for evaluation.
While solving a very hard problem can be highly informative, they most likely would not be solved and do not yield much information on average.

### Ideas for improving evaluation
Since it is difficult to collect more evaluations, using more evaluation benchmarks is a good way to improve confidence. Even when individual benchmarks yield low significance, consistent improvements across benchmarks can be significant.

For the test based benchmarks, a solution passing a weak test is still different from a solution failing it, thus running indepedent tests may also help yield more information per example instead of focusing on complete correctness. This is similar to the tech interview approach, where often only 1 or 2 questions are asked but you may get more information than just correct vs. incorrect.

It may also be possible to model individual problems better, such as the probability of each outcome or the difficulty level using [item response model](https://eacl2024irt.github.io/). However our initial attempts at this failed to improve stability or significance levels, possibly due to a high level of noise.

## Usage 

The raw example level evaluation data is stored in `data/*` in this format:
```
{"benchmark_id":"CRUXEval-input", "model":"phind", "example_id":"CRUXEval-input\/0", "pass1":0.8}
```

To generate the summaries and figures
```
python run_arena.py
```

Additional models and evaluations are welcome.


## Citation

```
@misc{eval_arena,
  title = {Eval-Arena: measuring noise and detecting errors using leaderboard data},
  authors = {Sida I. Wang and Alex Gu and Jiawei Liu and Yuxiang Wei and Naman Jain and Yuhang Lai and Ofir Press and Baptiste Rozière and Gabriel Synnaeve}
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