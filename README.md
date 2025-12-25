# [Eval Arena](https://crux-eval.github.io/eval-arena)

## Usage 

The raw example level evaluation data is stored in `data/*` in this format:
```
{"benchmark_id":"humaneval", "model":"code-llama-multi-34b", "example_id":"HumanEval\/4", "pass1":1, "count": 2, "correct": 2}
{"benchmark_id":"CRUXEval-input", "model":"phind", "example_id":"CRUXEval-input\/0", "pass1":0.8, "count": 5, "correct": 4}
```

To generate the summaries and figures, install requirements and set `OUTPUT_PATH` then
```
 python -u run_arena.py data="data/vllm_evals/highk_temp0.7.jsonl" \
    out_dir=${OUTPATH}/${NAME}/highk_temp0.7 \
    max_diff=0.2 recompute=True
```

Additional models and evaluations are welcome.


## Contributors and citation 

If you find this work helpful, consider citing us.

```
@misc{evalarena,
  title = {{E}val-{A}rena: noise and errors on LLM evaluations},
  author = {Sida I. Wang and Alex Gu and Lovish Madaan and Dieuwke Hupkes and Jiawei Liu and Yuxiang Wei and Naman Jain and Yuhang Lai and Sten Sootla and Ofir Press and Baptiste Rozière and Gabriel Synnaeve},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/crux-eval/eval-arena}}
}
```

Thanks for the helpful discussions with Ari Holtzman, Nicolas Usunier, and Pedro Rodriguez. Chatbot arena provided the inspiration to test all model pairs and summarizing the results in a understandable way.


## License

The majority of Eval-Arena is licensed under MIT, however portions of the project are available under separate license terms:

https://github.com/xlang-ai/DS-1000/blob/main/LICENSE

https://github.com/google-research/google-research/blob/master/LICENSE