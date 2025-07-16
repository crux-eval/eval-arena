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
from report_model import gen_model_report, write_data_tables, write_summary_table
from signal_noise import signal_to_noise


@dataclass
class ReportArgs:
    out_dir: Optional[str] = "gh-pages/"
    data: str = "data/*.jsonl"
    recompute: bool = True # generate results for all data and summary line
    write_summary: bool = True # use results in out_dir/tmp to generate the summary table

def setup_output(args: ReportArgs):
    # Copy custom.css to the output directory
    css_src = Path("templates/custom.css")
    css_dst = Path(args.out_dir) / "static" / "custom.css"
    os.makedirs(Path(args.out_dir) / "static" , exist_ok=True)
    with open(css_src, "rb") as src_file, open(css_dst, "wb") as dst_file:
        dst_file.write(src_file.read())
    
    tmp_dir = Path(args.out_dir) / "tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(Path(args.out_dir) / "data" , exist_ok=True)

def run_arena(args: ReportArgs):
    records = []
    for fname in glob.glob(args.data):
        with open(fname, "rt") as f:
            records.extend([json.loads(l) for l in f.readlines()])
    eval_results = pd.DataFrame(records)
    print(eval_results)

    benchmarks = set(eval_results["benchmark_id"])
    print("included benchmarks: ", benchmarks)
    tmp_dir = Path(args.out_dir) / "tmp"

    if args.recompute:
        for bid in benchmarks:
            print(f"processing {bid}...")
            result_bid = eval_results[eval_results["benchmark_id"] == bid] 
            arena_res: arena.ArenaResult = arena.summarize_benchmark(result_bid)

            sig_to_noise = signal_to_noise(bid, arena_res.summary)
            summary_stats = arena_res.summary_stats 
            summary_stats["sig_noise"] = sig_to_noise["signal to noise"].median() if sig_to_noise is not None else float("nan")

            print(pd.DataFrame([summary_stats]))
            pd.DataFrame([summary_stats]).to_json(tmp_dir / f"summary-{bid}.jsonl", orient="records", lines=True)

            gen_model_report(bid, arena_res, args.out_dir)
            gen_example_report(bid, arena_res, args.out_dir)
            write_data_tables(bid, arena_res, args.out_dir)
    
    if args.write_summary:
        records = []
        for fname in glob.glob(f"{tmp_dir}/summary-*.jsonl"):
            with open(fname, "rt") as f:
                records.extend([json.loads(l) for l in f.readlines()])
        print(records)
        df_summary = pd.DataFrame(records)
        df_summary.to_csv(Path(args.out_dir) / "summary.csv")
        write_summary_table(pd.DataFrame(df_summary), Path(args.out_dir) / "index.html")


if __name__ == "__main__":
    cli_args = OmegaConf.from_cli()
    default_cfg = OmegaConf.structured(ReportArgs())
    args = OmegaConf.merge(default_cfg, cli_args)
    args = OmegaConf.to_object(args)
    print(args)
    setup_output(args)
    run_arena(args)
