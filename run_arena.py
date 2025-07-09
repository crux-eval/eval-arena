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
from report_model import gen_model_report, write_summary_table
from signal_noise import signal_to_noise


@dataclass
class ReportArgs:
    out_dir: Optional[str] = 'gh-pages/'
    data: str = "data/*.jsonl"
    recompute: bool = True # generate results for all data and summary line
    write_summary: bool = True # use results in out_dir/tmp to generate the summary table


def summarize_benchmark(result: pd.DataFrame):
    benchmarks = set(result['benchmark_id'])
    assert len(benchmarks) == 1
    bid = benchmarks.pop()
    battles = arena.pass1_to_battle(result)
    summary = arena.battle_summary(battles)
    agg_results = arena.model_table(battles, result)
    ex = arena.example_table(result, agg_results)
    print(summary)

    close_pairs = summary[summary["pvalue"] > 1e-3]

    r = {
        'benchmark_id': bid,
        'size': int(summary.iloc[0]['total']),
        'models': len(set(summary["model_a"])),
        'total_pairs': len(summary),
        'close_pairs': len(close_pairs),
        'std(A-B)': close_pairs["std(A-B)"].describe().to_dict(),
        'corr(A,B)': close_pairs["corr(A,B)"].describe().to_dict(),
        'p5_min': int(summary[summary['pvalue'] < 0.05]['diff'].abs().min()),
        'p5_max': int(summary[summary['pvalue'] > 0.05]['diff'].abs().max()),
        'min_dist': int(summary['sum'].abs().min()),
        'no_solve': (ex['acc'] == 0).to_numpy().sum(),
        'tau-': (ex['tau'] < 0).to_numpy().sum(),
    }

    sig_to_noise = signal_to_noise(bid, summary)
    r['sig_noise'] = sig_to_noise['signal to noise'].median() if sig_to_noise is not None else float('nan')
    return r


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
        print(records)
        # Copy custom.css to the output directory
        css_src = Path("templates/custom.css")
        css_dst = Path(args.out_dir) / "static" / "custom.css"
        os.makedirs(Path(args.out_dir) / "static" , exist_ok=True)
        with open(css_src, "rb") as src_file, open(css_dst, "wb") as dst_file:
            dst_file.write(src_file.read())

        df_summary = pd.DataFrame(records)
        df_summary.to_csv(Path(args.out_dir) / 'summary.csv')
        write_summary_table(pd.DataFrame(df_summary), Path(args.out_dir) / 'index.html')


if __name__ == "__main__":
    cli_args = OmegaConf.from_cli()
    default_cfg = OmegaConf.structured(ReportArgs())
    args = OmegaConf.merge(default_cfg, cli_args)
    args = OmegaConf.to_object(args)
    print(args)
    run_arena(args)
