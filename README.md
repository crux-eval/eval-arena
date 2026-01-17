# Measuring all the noises of LLM evals

<p align="left">
    <a href="https://arxiv.org/abs/2512.21326"><img src="https://img.shields.io/badge/arXiv-2512.21326-b31b1b.svg?style=for-the-badge">
</p>

<p align="left">
    🧐&nbsp;<a href="#-about">About</a>
    | 🚀&nbsp;<a href="#-quick-start">Quick Start</a>
    | 📝&nbsp;<a href="#-citation">Citation</a>
    | 🙏&nbsp;<a href="#-acknowledgements">Acknowledgements</a>
</p>

## 🧐 About

This code measures the prediction noise, data noise, and total noise on many
LLMs/agents and evals.
These measurements allow us to estimate the statistical significance of any results on these evals.
Using paired analysis, we find that the model predictions are responsible for more noises than the data, thus we can usually detect effect sizes 2 times smaller on unrelated models and 6+ times smaller on related models by reducing the prediction noise.

The reference noise measurements for many evals are [here](https://all-the-noises.github.io/main/index.html),
which links interactive figures such as [noises vs. accuracy](https://all-the-noises.github.io/highk_temp0.7/model_math500_cot.html),
and [the predictions heatmaps](https://all-the-noises.github.io/highk_temp0.7/ex_v_model_acc_math500_cot.html).
For results based on one prediction per example, use the total standard error shown under `SE(A-B)`.
For the potential of noise reduction, the remaining data standard error is shown under `SE_x(A-B)`.

### How?
* Since LLMs generates independent and diverse samples, we can draw multiple samples per question to measure the noise components.
* [estimators.py](estimators.py) contains the Paired and Unpaired estimators to do this.
* By measuring the noise of many pairs of models, some clear patterns emerge.


The original Eval-Arena docs are at [doc/eval-arena-findings.md](doc/eval-arena-readme.md).

## 🚀 Quick Start

To generate the static summaries and figures, install requirements and set `OUTPATH`

```bash
 python -u run_arena.py data="data/vllm_evals/highk_temp0.7.jsonl" \
    out_dir=${OUTPATH}/highk_temp0.7 \
    max_diff=0.2 recompute=True
```

To view the results,

```bash
cd ${OUTPATH}/highk_temp0.7
python -m http.server
```


### Data

The question level metrics are stored in this format:

```json
{"benchmark_id":"humaneval", "model":"code-llama-multi-34b", "example_id":"HumanEval4", "pass1":1, "correct":2, "count":2}
{"benchmark_id":"CRUXEval-input", "model":"phind", "example_id":"CRUXEval-input0", "pass1":0.8, "correct":4, "count":5}
```

`benchmark_id`, `model` and `example_id` should together be unique. `pass1` is the ratio of correct results out of `count` attempts.

Data contributions are welcome via pull requests to [data](https://github.com/all-the-noises/data).

The datasets used to produce the results is in [this release](https://github.com/crux-eval/eval-arena/releases/tag/data-12-25-25). The corresponding runs are at [`submit_all.sh`](./submit_all.sh)

This data is visualized by these [heatmaps](https://all-the-noises.github.io/highk_temp0.7/ex_v_model_acc_math500_cot.html), linked under details/data of each eval from the [main table](https://all-the-noises.github.io/).

## 📝 Citation

```bibtex
@misc{wang2025allthenoises,
      title={Measuring all the noises of LLM Evals},
      author={Sida Wang},
      year={2025},
      eprint={2512.21326},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2512.21326},
}
```
<!-- For the data, cite [eval-arena](doc/eval-arena-readme.md#contributors-and-citation). -->

## 🙏 Acknowledgements
I thank Sean O’Brien, Lovish Madaan, Dieuwke Hupkes, Alex Gu, Jiawei
Liu, Yuhang Lai, and Sten Sootla for making question-level data available for analysis. I am extremely grateful
to Evan Miller, Nicolas Usunier, Zach Rait, Yuxiang Wei, Jannik Kossen, and Ari Holtzman for
valuable discussions and feedback; Pedro Rodriguez, Ofir Press, Naman Jain, Baptiste Rozière,
Gabriel Synnaeve, Dawn Song, and Zijian Wang for their advice and support. The all-pairs approach
is inspired by Chatbot Arena and the clarity of Miller (2024) greatly helped.
