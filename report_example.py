from collections import defaultdict
import json, math, glob
import numpy as np
import pandas as pd
import scipy.stats as stats
import plotly.express as px
from tqdm import tqdm
import math
import os
from arena import result_table, pass1_to_battle, example_table

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
    link = get_link()
    if link != '':
        return f'<a href="{get_link()}">{example_id}</a>'
    else:
        return example_id

def get_example_level_results(benchmark_id):
    result = eval_results[eval_results['benchmark_id'] == benchmark_id]
    battles = pass1_to_battle(result)
    battles_no_ties = battles[battles["winner"].str.contains("model_")]
    all_stats = result_table(battles_no_ties, result)
    ex_table = example_table(result, all_stats)
    ex_table['example_link'] = ex_table['example_id'].apply(lambda x: get_anchor(benchmark_id, x))

    outputs = {}
    outputs['result table'] = all_stats.sort_values(by='elo', ascending=False).to_html(float_format='%10.3f')
    outputs['fig_min_elo_solve'] = px.histogram(ex_table, x='min_elo', marginal='rug', title='min ELO to solve').to_html(full_html=False)
    outputs['table_histogram_accs'] = px.histogram(ex_table, x='acc', marginal='rug', title='accuracy on examples').to_html(full_html=False)

    no_solve = ex_table[ex_table['num_solved'] == 0]
    outputs['list_no_solve'] = sorted(no_solve['example_link'].to_list())
    one_solve = ex_table[ex_table['num_solved'] == 1]
    display(one_solve)
    one_solve['model'] = one_solve['models'].apply(lambda x: x[0])
    one_solve = one_solve.sort_values(by='min_elo', ascending=False)
    one_solve = one_solve[['example_link', 'model', 'min_elo']]
    display(one_solve)
    outputs['table_one_solve'] = one_solve.to_html(escape=False, float_format='%10.3f', index=False)

    elo75 = all_stats['elo'].quantile(0.75)
    print(elo75)
    list_suspect = ex_table.sort_values(by='tau', ascending=True).head(10)
    outputs['table_suspect'] = list_suspect[['example_link', 'acc', 'tau']].to_html(escape=False, float_format='%10.3f', index=False)
    print(benchmark_id, 'anti-correlated prop', np.mean(ex_table['tau'] <= 0))
    print(ex_table['tau'].describe())

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

for b in set(eval_results['benchmark_id']):
    gen_report(b)
    # outputs['fig_unique_solves'] = px.histogram(one_solve, x='model').update_xaxes(categoryorder='total descending')
