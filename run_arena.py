import json, glob
import pandas as pd

from jinja2 import Template

import arena
from report_example import gen_example_report
from report_model import gen_model_report

def generate_summary(eval_results: pd.DataFrame):
    benchmarks = set(eval_results['benchmark_id'])
    records = []
    for bid in benchmarks:
        result = eval_results[eval_results['benchmark_id'] == bid]
        battles = arena.pass1_to_battle(result)
        summary = arena.battle_summary(battles)
        agg_results = arena.model_table(battles, result)
        ex = arena.example_table(result, agg_results)

        data_sz = int(summary.iloc[0]['total'])
        min_p5 = int(summary[summary['pvalue'] < 0.05]['diff'].abs().min())
        max_p5 = int(summary[summary['pvalue'] > 0.05]['diff'].abs().max())
        min_dist = int(summary['sum'].abs().min())
        r = {
            'benchmark_id': bid,
            'size': data_sz,
            'p5_min': min_p5,
            'p5_max': max_p5,
            'min_dist': min_dist,
            'no_solve': (ex['acc'] == 0).to_numpy().sum(),
            'tau-': (ex['tau'] < 0).to_numpy().sum(),
        }
        records.append(r)

    summary_count = pd.DataFrame(records).sort_values(by='benchmark_id')
    def link_detail(bid):
        l1 = f"""by <a href="model_{bid}.html">models </a> | """
        l2 = f"""<a href="ex_{bid}.html"> examples </a>"""
        return l1 + l2
    summary_count['link to details'] = summary_count['benchmark_id'].apply(link_detail)

    def normalize(counts, includes):
        percent = pd.DataFrame(counts)
        for c in includes:
            percent[c] = percent[c] / percent['size']
        return percent

    includes_cols = ['benchmark_id', 'size', 'p5_min', 'p5_max', 'no_solve', 'tau-', 'link to details']
    percent_cols = ['p5_min', 'p5_max', 'no_solve', 'tau-']
    summary_percent = normalize(summary_count, percent_cols)

    template_path = r"templates/summary.html"
    output_path = rf"crux-eval.github.io/eval-arena/index.html"
    with open(output_path, "w", encoding="utf-8") as output_file:
        with open(template_path) as template_file:
            j2_template = Template(template_file.read())
            output_file.write(j2_template.render({
                'count_table': summary_count[includes_cols].to_html(escape=False, index=False),
                'percent_table': summary_percent[includes_cols].to_html(
                    escape=False,
                    index=False,
                    formatters={
                        'p5_min': '{:.1%}'.format,
                        'p5_max': '{:.1%}'.format,
                        'min_dist': '{:.1%}'.format,
                        'no_solve': '{:.1%}'.format,
                        'tau-': '{:.1%}'.format,
                    }),
            }))


records = []
for fname in glob.glob(f"data/*.jsonl"):
    with open(fname, 'rt') as f:
        records.extend([json.loads(l) for l in f.readlines()])
eval_results = pd.DataFrame(records)

print('generating summary table...')
generate_summary(eval_results)
benchmarks = set(eval_results['benchmark_id'])

OUTPUT_PATH = 'crux-eval.github.io/eval-arena'
for bid in benchmarks:
    print(f'processing {bid}...')
    raw_results = eval_results[eval_results['benchmark_id'] == bid] 
    gen_example_report(bid, raw_results, OUTPUT_PATH)
    gen_model_report(bid, raw_results, OUTPUT_PATH)