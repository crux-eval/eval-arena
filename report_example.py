import numpy as np
import pandas as pd
import plotly.express as px
import math
from arena import model_table, pass1_to_battle, example_table

def get_anchor(benchmark_id: str, example_id: str):
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


def get_example_level_results(benchmark_id, result):
    battles = pass1_to_battle(result)
    battles_no_ties = battles[battles["winner"].str.contains("model_")]
    all_stats = model_table(battles_no_ties, result)
    ex_table = example_table(result, all_stats)
    ex_table['example_link'] = ex_table['example_id'].apply(lambda x: get_anchor(benchmark_id, x))

    outputs = {}
    outputs['result table'] = all_stats.sort_values(by='elo', ascending=False).to_html(float_format='%10.3f')
    outputs['fig_min_elo_solve'] = px.histogram(ex_table, x='min_elo', marginal='rug', title='min ELO to solve').to_html(full_html=False)
    outputs['table_histogram_accs'] = px.histogram(ex_table, x='acc', marginal='rug', title='accuracy on examples').to_html(full_html=False)

    no_solve = ex_table[ex_table['num_solved'] == 0]
    outputs['list_no_solve'] = sorted(no_solve['example_link'].to_list())
    one_solve = ex_table[ex_table['num_solved'] == 1]
    pd.options.mode.chained_assignment = None 
    one_solve['model'] = one_solve['models'].apply(lambda x: x[0])
    one_solve = one_solve.sort_values(by='min_elo', ascending=False)
    one_solve = one_solve[['example_link', 'model', 'min_elo']]
    outputs['table_one_solve'] = one_solve.to_html(escape=False, float_format='%10.3f', index=False)

    list_suspect = ex_table.sort_values(by='tau', ascending=True).head(10)
    outputs['table_suspect'] = list_suspect[['example_link', 'acc', 'tau']].to_html(escape=False, float_format='%10.3f', index=False)
    print(benchmark_id, 'anti-correlated prop', np.mean(ex_table['tau'] <= 0))

    return outputs


def gen_example_report(benchmark_id: str, raw_results: pd.DataFrame, OUTPUT_PATH):
    outputs = get_example_level_results(benchmark_id, raw_results)
    from jinja2 import Template
    template_path = r"templates/template_example.html"
    output_path = rf"{OUTPUT_PATH}/ex_{benchmark_id}.html"
    with open(output_path, "w", encoding="utf-8") as output_file:
        with open(template_path) as template_file:
            j2_template = Template(template_file.read())
            output_file.write(j2_template.render({'benchmark_id': benchmark_id, 'outputs': outputs}))