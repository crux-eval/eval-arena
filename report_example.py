from collections import defaultdict
import json, math, glob
import numpy as np
import pandas as pd
import scipy.stats as stats
import plotly.express as px
from tqdm import tqdm
import math
import os
from report_agg import result_table, pass1_to_battle

display = print

def get_anchor(benchmark_id: str, example_id: str):
    # supporting {'humaneval+', 'CRUXEval-input', 'mbpp+', 'CRUXEval-output'}
    def get_link():
        if benchmark_id in ['humaneval', 'humaneval+', 'mbpp', 'mbpp+']:
            dir, id = example_id.split('/') # expecting HumanEval/93 and Mbpp/622 etc.
            return f'https://crux-eval.github.io/eval-arena/evalplus/{dir}/{id}.html'
        elif benchmark_id in ['CRUXEval-input', 'CRUXEval-output']:
            id = example_id.replace(benchmark_id + '/', '')
            return f'https://crux-eval.github.io/demo.html?id={int(id) + 1}'
        else:
            return ''
    return f'<a href="{get_link()}">{example_id}</a>'

def gen_example_table(result, all_stats):
    records = []
    ids = set(result['example_id']) 
    len_data = len(set(result['example_id']))
    print(np.mean(all_stats['elo']))
    
    for current_id in list(ids):
        example_data = result[result['example_id'] == current_id][['model', 'pass1']]
        ex = example_data.merge(all_stats[['model', 'elo']], left_on = 'model', right_on = 'model')
        # fit_data['result'] = fit_data['result']
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression()
        ex['correct'] = np.where(ex['pass1'] > 0, 1, 0)
        model_elos = ex[ex['correct'] == 1]['elo']
        # print(model_elos.describe())
        r = model_elos.describe().to_dict()
        r['example_id'] = current_id
        r['models'] = ex[ex['correct'] == 1]['model'].to_numpy()
        r['acc'] = len(ex[ex['correct'] == 1]) / len(ex)
        records.append(r)

    return pd.DataFrame(records)


def get_example_level_results(benchmark_id):
    result = eval_results[eval_results['benchmark_id'] == benchmark_id]
    battles = pass1_to_battle(result)
    battles_no_ties = battles[battles["winner"].str.contains("model_")]
    all_stats = result_table(battles_no_ties, result)
    example_table = gen_example_table(result, all_stats)
    example_table['example_link'] = example_table['example_id'].apply(lambda x: get_anchor(benchmark_id, x))

    outputs = {}
    outputs['result table'] = all_stats.sort_values(by='elo', ascending=False).to_html(float_format='%10.3f')
    outputs['fig_min_elo_solve'] = px.histogram(example_table, x='min', marginal='rug', title='min ELO to solve').to_html(full_html=False)
    outputs['table_histogram_accs'] = px.histogram(example_table, x='acc', marginal='rug', title='accuracy on examples').to_html(full_html=False)

    no_solve = example_table[example_table['count'] == 0]
    outputs['list_no_solve'] = sorted(no_solve['example_link'].to_list())
    one_solve = example_table[example_table['count'] == 1]
    display(one_solve)
    one_solve['model'] = one_solve['models'].apply(lambda x: x[0])
    one_solve = one_solve.sort_values(by='max', ascending=False)
    one_solve = one_solve[['example_link', 'model', 'max']]
    display(one_solve)
    outputs['table_one_solve'] = one_solve.to_html(escape=False)

    elo75 = all_stats['elo'].quantile(0.75)
    print(elo75)
    list_suspect = example_table[example_table['max'] < elo75]
    outputs['table_suspect'] = list_suspect[['example_link', 'models', 'max']].to_html(escape=False)

    print(outputs.keys())
    return outputs

records = []
for fname in glob.glob(f"data/*.jsonl"):
    with open(fname, 'rt') as f:
        records.extend([json.loads(l) for l in f.readlines()])

eval_results = pd.DataFrame(records)
print(eval_results.describe())

def gen_report(benchmark_id: str):
    outputs = get_example_level_results(benchmark_id)
    from jinja2 import Template
    template_path = r"examplelevel_template.html"
    output_path = rf"crux-eval.github.io/eval-arena/ex_{benchmark_id}.html"
    with open(output_path, "w", encoding="utf-8") as output_file:
        with open(template_path) as template_file:
            j2_template = Template(template_file.read())
            output_file.write(j2_template.render({'benchmark_id': benchmark_id, 'outputs': outputs}))

for b in ['CRUXEval-output', 'CRUXEval-input', 'humaneval+', 'mbpp+', 'lcb_codegen']:
    gen_report(b)
    # outputs['fig_unique_solves'] = px.histogram(one_solve, x='model').update_xaxes(categoryorder='total descending')
