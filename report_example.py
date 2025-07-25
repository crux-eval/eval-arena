import numpy as np
import pandas as pd
import plotly.express as px
from arena import ArenaResult
from jinja2 import Template

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

def fig_example_vs_model(result, all_stats, ex_table):
    df = result[['model', 'example_id', 'pass1']].merge(ex_table[['example_id', 'acc']], on='example_id')
    df = df.merge(all_stats[['model', 'pass1']], on='model', suffixes=['_ex', '_model'])
    df.sort_values(by=['acc', 'example_id', 'pass1_model', 'model'], inplace=True)
    fig = px.scatter(df, y='example_id', x='model', color='pass1_ex',
                     opacity=0.75,
                     color_continuous_scale=["red", "yellow", "green"],
                     hover_data=['acc', 'model', 'example_id'])
    fig.update_xaxes(autorange="reversed")
    fig.update_traces(marker={'symbol': 'square'})
    fig.update_layout(
            width=900, height=1200,
            xaxis = dict(side ="top"),
        )
    return fig

def get_example_level_results(benchmark_id, ares: ArenaResult):
    all_stats = ares.model_table 
    ex_table = ares.example_table
    ex_table['example_link'] = ex_table['example_id'].apply(lambda x: get_anchor(benchmark_id, x))

    outputs = {}
    outputs['result table'] = all_stats.sort_values(by='elo', ascending=False).to_html(classes="number-table", float_format='%10.3f')
    plotly_configs = dict(full_html=False, include_plotlyjs="cdn")
    outputs['fig_min_elo_solve'] = px.histogram(ex_table, x='min_elo', marginal='rug', title='min ELO to solve').to_html(**plotly_configs)
    outputs['table_histogram_accs'] = px.histogram(ex_table, x='acc', marginal='rug', title='accuracy on examples').to_html(**plotly_configs)

    no_solve = ex_table[ex_table['num_solved'] == 0]
    outputs['list_no_solve'] = sorted(no_solve['example_link'].to_list())
    one_solve = ex_table[ex_table['num_solved'] == 1]
    pd.options.mode.chained_assignment = None 
    one_solve['model'] = one_solve['models'].apply(lambda x: x[0])
    one_solve = one_solve.sort_values(by='min_elo', ascending=False)
    one_solve = one_solve[['example_link', 'model', 'min_elo']]
    outputs['table_one_solve'] = one_solve.to_html(escape=False, classes="number-table", float_format='%10.3f', index=False)

    list_suspect = ex_table.sort_values(by='tau', ascending=True).head(10)
    outputs['table_suspect'] = list_suspect[['example_link', 'acc', 'tau']].to_html(escape=False, classes="number-table", float_format='%10.3f', index=False)
    print(benchmark_id, 'anti-correlated prop', np.mean(ex_table['tau'] <= 0))

    outputs['fig_example_vs_model'] = fig_example_vs_model(ares.input_table, all_stats, ex_table)
    return outputs

def gen_example_report(benchmark_id: str, ares: ArenaResult, OUTPUT_PATH):
    outputs = get_example_level_results(benchmark_id, ares)
    template_path = r"templates/template_example.html"
    output_path = rf"{OUTPUT_PATH}/ex_{benchmark_id}.html"
    with open(output_path, "w", encoding="utf-8") as output_file:
        with open(template_path) as template_file:
            j2_template = Template(template_file.read())
            output_file.write(j2_template.render({'benchmark_id': benchmark_id, 'outputs': outputs}))

    with open(f'{OUTPUT_PATH}/ex_v_model_{benchmark_id}.html', 'wt') as f:
        f.write(outputs['fig_example_vs_model'].to_html())
    