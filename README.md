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
You can get the reference noise measurements for many evals [here](https://all-the-noises.github.io/main/index.html),
which links to interactive figures such as [noises vs. accuracy](https://all-the-noises.github.io/highk_temp0.7/model_math500_cot.html), 
and [the predictions heatmaps](https://all-the-noises.github.io/highk_temp0.7/ex_v_model_acc_math500_cot.html).

### Why?
* These measurements allow us to assess the statistical significance of any results on these evals.
* Shows that we can detect much smaller effects when prediction noise > data noise. Typically this allows us to detect 1/2 the effect size, but could be <1/6 on related models and even less on related checkpoints.
* When prediction noise is too high, modelling the data is pointless, so we need to know how much noise is due to the model predictions vs. the eval data.

### How?
* Since LLMs can draw independent and diverse samples, we can draw multiple samples to measure the prediction noise directly.
* [estimators.py](estimators.py) contains the core Paired and Unpaired estimators to do this.


The original Eval-Arena docs are at [doc/eval-arena-findings.md](doc/eval-arena-findings.md).

## 🚀 Quick Start

To generate the static summaries and figures, install requirements and set `OUTPUT_PATH` 

```
 python -u run_arena.py data="data/vllm_evals/highk_temp0.7.jsonl" \
    out_dir=${OUTPUT_PATH}/highk_temp0.7 \
    max_diff=0.2 recompute=True
```

To view the results,

```
cd ${OUTPUT_PATH}/highk_temp0.7
python -m http.server
```


### Data

The example level evaluation data is stored in this format:

```
{"benchmark_id":"humaneval", "model":"code-llama-multi-34b", "example_id":"HumanEval4", "pass1":1, "correct":2, "count":2}
{"benchmark_id":"CRUXEval-input", "model":"phind", "example_id":"CRUXEval-input0", "pass1":0.8, "correct":4, "count":5}
```

The dataset used to produce the results can be found in the [release](https://github.com/crux-eval/eval-arena/releases/tag/data-12-25-25). Corresponding to [`submit_all.sh`](./submit_all.sh)


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

For findings in [eval-arena](doc/eval-arena-readme.md#contributors-and-citation):

## 🙏 Acknowledgements
I thank Sean O’Brien, Lovish Madaan, Dieuwke Hupkes, Alex Gu, Jiawei
Liu, Yuhang Lai, and Sten Sootla for making question-level data available for analysis. I am extremely grateful
to Evan Miller, Nicolas Usunier, Zach Rait, Yuxiang Wei, Jannik Kossen, and Ari Holtzman for
valuable discussions and feedback; Pedro Rodriguez, Ofir Press, Naman Jain, Baptiste Rozière,
Gabriel Synnaeve, Dawn Song, and Zijian Wang for their advice and support. The all-pairs approach
is inspired by Chatbot Arena and the clarity of Miller (2024) greatly helped.
