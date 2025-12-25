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

## About

We measures the prediction noise, data noise, and total noise on many
LLMs/agents and evals.
You can get the reference noise measurements for each eval [here](https://all-the-noises.github.io/main/index.html),
which links to interactive figures such as [the noises vs. accuracy](https://all-the-noises.github.io/highk_temp0.7/model_math500_cot.html), 
and [the predictions overview](https://all-the-noises.github.io/highk_temp0.7/ex_v_model_acc_math500_cot.html).

Since LLMs can draw independent and diverse samples, we can measure their prediction noise directly given multiple samples.
[estimators.py](estimators.py) contains the core Paired and Unpaired estimators to do this.

These measurements allow us to assess significance of results and to detect much smaller effects in
controlled experiments when prediction noise > data noise.

Eval-Arena docs are at [doc/eval-arena-findings.md](doc/eval-arena-findings.md).

## Quick Start

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

The example level evaluation data is stored in this format:

```
{"benchmark_id":"humaneval", "model":"code-llama-multi-34b", "example_id":"HumanEval4", "pass1":1, "correct":2, "count":2}
{"benchmark_id":"CRUXEval-input", "model":"phind", "example_id":"CRUXEval-input0", "pass1":0.8, "correct":4, "count":5}
```

## Citation
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

## License

The majority of Eval-Arena is licensed under MIT, however portions of the project are available under separate license terms:

https://github.com/xlang-ai/DS-1000/blob/main/LICENSE

https://github.com/google-research/google-research/blob/master/LICENSE