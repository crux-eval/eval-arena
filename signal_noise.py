import logging
import re

import numpy as np
import pandas as pd
import plotly.express as px

import arena
from utils import load_jsonl_files

logger = logging.getLogger(__name__)

pairs_default = {
    # 'Qwen1.5-': ['110B', '72B', '32B', '14B', '7B', '4B', '1.8B', '0.5B'],
    'Qwen1.5-': ['72B', '32B', '7B'],
    'llama_': ['65B', '33B', '07B'],
    'deepseek-llm-': ['67b-base', '7b-base'],
    'llama2_': ['70B', '13B', '07B'],
    'Mixtral-': ['8x22B-v0.1', '8x7B-v0.1'],
    'Meta-Llama-3-': ['70B', '8B'],
}

# model names are specified manually for each eval
pairs_specific = {
    'DS1000': {
        'meta-llama-Llama-3-': ['70b-chat-hf', '8b-chat-hf'],
        'meta-llama-Llama-3-': ['70B', '8B'],
        'Qwen-Qwen1.5-': ['72B-Chat', '32B-Chat'],
        'Qwen-Qwen1.5-': ['72B', '32B', '7B'],
        'deepseek-ai-deepseek-coder-': ['33b-base', '6.7b-base']	
    },
    'lcb_codegen': {
        'LLama3-': ['70b-Ins', '8b-Ins'],
        'LLama3-': ['70b-Base', '8b-Base'],
        'DSCoder-': ['33b-Ins', '6.7b-Ins'],
        'DSCoder-': ['33b-Base', '6.7b-Base'],
    },
    'mbpp': {
        'deepseek-coder-': ['33b-instruct', '6.7b-instruct'],
        'wizardcoder-': ['34b', '7b'],
    },
    'CRUXEval-output': {
        'deepseek-base-': ['33b', '6.7b'],
        'deepseek-instruct-': ['33b', '6.7b'],
        'codellama-': ['34b', '13b', '7b'],
        'codellama-': ['34b+cot', '13b+cot', '7b+cot'],
    },
    'humaneval+': {
        'deepseek-coder-': ['33b-instruct', '6.7b-instruct'], 
        'opencodeinterpreter-ds-': ['33b', '6.7b']
    },
    'safim': {
        'deepseek-coder-': ['33b', '6.7b', '1.3b'],
        'codellama-': ['13b', '7b'], # '34b' does not have FIM, and this is a FIM benchmark
        'wizardcoder-': ['33b', '15b', '3b'],
        'codegen-': ['16b', '6b', '2b']
    }
}
pairs_specific['mbpp+'] = pairs_specific['mbpp']
pairs_specific['humaneval'] = {**pairs_default, **pairs_specific['humaneval+']}
pairs_specific['CRUXEval-input-T0.2'] = pairs_specific['CRUXEval-output']
pairs_specific['CRUXEval-input-T0.8'] = pairs_specific['CRUXEval-output']
pairs_specific['CRUXEval-output-T0.2'] = pairs_specific['CRUXEval-output']
pairs_specific['CRUXEval-output-T0.8'] = pairs_specific['CRUXEval-output']


def model_pairs(bid: str):
    if bid in pairs_specific:
        pairs = pairs_specific[bid]
    else:
        pairs = pairs_default

    for base in pairs:
        models = pairs[base]
        for m1, m2 in zip(models[:-1], models[1:]):
            yield base + m1, base + m2


re_modelsize = re.compile(r'(.*)[-_x](\d+\.?\d?)[bB]')
def agg_signal_noise(x):
    def sizeB(name: str):
        match = re_modelsize.search(name)
        return float(match.group(2))
    def model_name(name: str):
        match = re_modelsize.search(name)
        return match.group(1)
        
    logsize = np.log2(sizeB(x.name[0])) - np.log2(sizeB(x.name[1]))
    name = model_name(x.name[0])
    return pd.Series({
        'signal to noise': x['sum(A-B)'] / np.sqrt(x['sum(A!=B)']) / logsize,
        'norm. signal to noise': x['sum(A-B)'] / np.sqrt(x['sum(A!=B)']) / logsize / np.sqrt(x['total']),
        'logsize': logsize,
        'sum(A!=B)': x['sum(A!=B)'],
        'model_family': name,
        'model_pair': ' vs. '.join(x.name),
        **x,
    })


def generate_all_summary(eval_results: pd.DataFrame):
    benchmarks = set(eval_results['benchmark_id'])
    results = {}
    for bid in benchmarks:
        result = eval_results[eval_results['benchmark_id'] == bid]
        battles = arena.BattleSummary.pass1_to_battle(result)
        summary = arena.BattleSummary.battle_summary(battles)
        results[bid] = summary
    return results


def signal_to_noise(bid: str, df):
    df['benchmark_id'] = bid
    df = df.set_index(['model_a', 'model_b'])
    pairs = list(model_pairs(bid))
    try:
        df = df.loc[pairs]
    except KeyError:
        logger.warning(f'The data does not have all specified model pairs: {pairs}')
        return None
    return df.apply(agg_signal_noise, axis=1)


if __name__ == '__main__':
    records = load_jsonl_files('data/*.jsonl')
    eval_results = pd.DataFrame(records)

    def combine(results):
        dfs = []
        for r in results:
            dfs.append(signal_to_noise(r, results[r]))
        return pd.concat(dfs, axis=0)

    results = generate_all_summary(eval_results)
    df = combine(results)
    df = df.apply(agg_signal_noise, axis=1)
    fig = px.scatter(df, x='benchmark_id', y='signal to noise', color='model_family', symbol='model_family', hover_name='model_pair')
    # fig = px.box(df, x='benchmark_id', y='signal to noise', hover_name='model_pair')
    fig.update_layout(
        width=800, height=600,
        xaxis = dict(tickangle=45),
    )
    df.sort_values(by='total', inplace=True, ascending=False)
    fig.update_xaxes(categoryorder='mean descending')

    OUTPUT_PATH = 'gh-pages/'
    logger.info('Generating signal noise table...')

    PLOTLY_CONFIGS = dict(full_html=False, include_plotlyjs="cdn")

    with open(f'{OUTPUT_PATH}/signal_noise.html', 'w') as f:
        f.write(fig.to_html(**PLOTLY_CONFIGS))