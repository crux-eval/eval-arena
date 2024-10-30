import os
import json, glob
from dataclasses import dataclass, field
from typing import Dict, Any, Iterator, Optional

import pandas as pd
from jinja2 import Template
from omegaconf import OmegaConf
from pathlib import Path

import arena
from report_example import gen_example_report
from report_model import gen_model_report
from signal_noise import signal_to_noise

def summarize_benchmark(result: pd.DataFrame):
    benchmarks = set(result['benchmark_id'])
    assert len(benchmarks) == 1
    bid = benchmarks.pop()
    battles = arena.pass1_to_battle(result)
    summary = arena.battle_summary(battles)
    agg_results = arena.model_table(battles, result)
    ex = arena.example_table(result, agg_results)
    print(summary)

    r = {
        'benchmark_id': bid,
        'size': int(summary.iloc[0]['total']),
        'p5_min': int(summary[summary['pvalue'] < 0.05]['diff'].abs().min()),
        'p5_max': int(summary[summary['pvalue'] > 0.05]['diff'].abs().max()),
        'min_dist': int(summary['sum'].abs().min()),
        'no_solve': (ex['acc'] == 0).to_numpy().sum(),
        'tau-': (ex['tau'] < 0).to_numpy().sum(),
    }

    sig_to_noise = signal_to_noise(bid, summary)
    r['sig_noise'] = sig_to_noise['signal to noise'].median() if sig_to_noise is not None else float('nan')
    return r


def write_summary_table(summary_count: pd.DataFrame, output_path: Path):
    summary_count = summary_count.sort_values(by='benchmark_id')

    def link_detail(bid):
        l1 = f"""by <a href="model_{bid}.html">models </a> """
        l2 = f"""<a href="ex_{bid}.html"> examples </a>"""
        l3 = f"""<a href="ex_v_model_{bid}.html"> data </a>"""
        return l1 + '|' + l2 + '|' + l3
    summary_count['link to details'] = summary_count['benchmark_id'].apply(link_detail)

    def normalize(counts, includes):
        percent = counts.copy(deep=True)
        for c in includes:
            percent[c] = percent[c] / percent['size']
        return percent

    includes_cols = ['benchmark_id', 'size', 'p5_min', 'p5_max', 'no_solve', 'tau-', 'sig_noise', 'link to details']
    percent_cols = ['p5_min', 'p5_max', 'no_solve', 'tau-']
    summary_percent = normalize(summary_count, percent_cols)

    template_path = r"templates/summary.html"

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
                        'sig_noise': '{:.2f}'.format,
                    }),
            }))


@dataclass
class ReportArgs:
    out_dir: Optional[str] = 'gh-pages/'
    data: str = "data/*.jsonl"
    recompute: bool = True # generate results for all data and summary line
    write_summary: bool = True # use results in out_dir/tmp to generate the summary table

def run_arena(args: ReportArgs):
    records = []
    for fname in glob.glob(args.data):
        with open(fname, 'rt') as f:
            records.extend([json.loads(l) for l in f.readlines()])
    eval_results = pd.DataFrame(records)
    print(eval_results)

    benchmarks = set(eval_results['benchmark_id'])
    print('included benchmarks: ', benchmarks)

    tmp_dir = Path(args.out_dir) / 'tmp'
    os.makedirs(tmp_dir, exist_ok=True)

    if args.recompute:
        for bid in benchmarks:
            print(f'processing {bid}...')
            result_bid = eval_results[eval_results['benchmark_id'] == bid] 
            summary = summarize_benchmark(result_bid)
            print(pd.DataFrame([summary]))
            pd.DataFrame([summary]).to_json(tmp_dir / f'summary-{bid}.jsonl', orient='records', lines=True)

            gen_example_report(bid, result_bid, args.out_dir)
            gen_model_report(bid, result_bid, args.out_dir)
    
    if args.write_summary:
        records = []
        for fname in glob.glob(f'{tmp_dir}/summary-*.jsonl'):
            with open(fname, 'rt') as f:
                records.extend([json.loads(l) for l in f.readlines()])
        write_summary_table(pd.DataFrame(records), Path(args.out_dir) / 'index.html')


if __name__ == "__main__":
    cli_args = OmegaConf.from_cli()
    default_cfg = OmegaConf.structured(ReportArgs())
    args = OmegaConf.merge(default_cfg, cli_args)
    args = OmegaConf.to_object(args)
    print(args)
    run_arena(args)
