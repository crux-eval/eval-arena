## Eval arena

There are many evaluations of varying quality and noise. For development we want to predict the performance head-to-head with other models and aggregate the information from many evaluations on different aspects. We want to know if the gains are real or just noise, and how much we should trust each example and each benchmark. Model developers should not need to worry if their improved benchmark numbers are just due to noise.
We aim to provide some helpful information on when differences in evaluation is real or not (aggregate results), and use example level results to detect important examples.

| Aggregate results | Example level results     |
| :---        |    :---- |
| [HumanEval+](https://crux-eval.github.io/eval-arena/agg_humaneval+.html)   | [HumanEval+](https://crux-eval.github.io/eval-arena/ex_humaneval+.html)  |
| [MBPP+](https://crux-eval.github.io/eval-arena/agg_mbpp+.html) | [MBPP+](https://crux-eval.github.io/eval-arena/ex_mbpp+.html) |
| [CRUXEval-input](https://crux-eval.github.io/eval-arena/agg_CRUXEval-input.html) | [CRUXEval-input](https://crux-eval.github.io/eval-arena/ex_CRUXEval-input.html) |
| [CRUXEval-output](https://crux-eval.github.io/eval-arena/agg_CRUXEval-output.html) | [CRUXEval-output](https://crux-eval.github.io/eval-arena/ex_CRUXEval-output.html) |