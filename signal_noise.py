from collections import defaultdict
import json, math, glob
import numpy as np
import pandas as pd
import scipy.stats as stats
import plotly.express as px
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

import arena

def generate_summary(eval_results: pd.DataFrame):
    benchmarks = set(eval_results['benchmark_id'])
    results = {}
    for bid in benchmarks:
        result = eval_results[eval_results['benchmark_id'] == bid]
        battles = arena.pass1_to_battle(result)
        summary = arena.battle_summary(battles)
        results[bid] = summary

    return results

records = []
for fname in glob.glob(f"raw-data/*_hf.jsonl"):
    with open(fname, 'rt') as f:
        records.extend([json.loads(l) for l in f.readlines()])
eval_results = pd.DataFrame(records)
print(eval_results)

results = generate_summary(eval_results)

pairs = {
    'Qwen1.5-': ['110B', '72B', '32B', '14B', '7B', '4B', '1.8B', '0.5B'],
    # 'llama_': ['65B', '33B', '13B', '07B'],
    # 'deepseek-llm-': ['67b-base', '7b-base'],
    'llama2_': ['70B', '13B', '07B'],
    # 'Mixtral-': ['8x22B-v0.1', '8x7B-v0.1'],
    # 'Meta-Llama-3-': ['70B', '8B'],
    # 'gemma-': ['7b', '2b'],
}

def enumerate_all():
    for base in pairs:
        models = pairs[base]
        for m1, m2 in zip(models[:-1], models[1:]):
            yield base + m1, base + m2

def combine(results):
    dfs = []
    for r in results:
        # print(r)
        dfr = results[r]
        # display(dfr)
        dfr['benchmark_id'] = r
        dfr = dfr.set_index(['model_a', 'model_b'])
        dfr = dfr.loc[enumerate_all()]
        dfs.append(dfr)
    return pd.concat(dfs, axis=0)
df = combine(results)
    
import re
r1 = re.compile(r'(.*)[-_x](\d+\.?\d?)[bB]')
def f(x):
    print(x.name)
    def sizeB(name: str):
        match = r1.search(name)
        return float(match.group(2))
    def model_name(name: str):
        match = r1.search(name)
        return match.group(1)
        
    logsize = np.log2(sizeB(x.name[0])) - np.log2(sizeB(x.name[1]))
    name = model_name(x.name[0])
    return pd.Series({'signal_noise': x['diff'] / np.sqrt(x['sum']),
                      'logsize': logsize,
                      'sum': x['sum'],
                      'model_family': name,
                      'model_pair': ' vs. '.join(x.name),
                      **x})

df = df.apply(f, axis=1)
df.sort_values(by='total', inplace=True, ascending=False)
fig = px.strip(df, x='benchmark_id', y='signal_noise', color='model_family', hover_name='model_pair')

OUTPUT_PATH = 'gh-pages/'
print('generating summary table...')

with open(f'{OUTPUT_PATH}/signal_noise.html', 'w') as f:
    f.write(fig.to_html(full_html=True))