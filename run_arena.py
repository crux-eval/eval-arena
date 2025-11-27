import logging
import os
from pathlib import Path

from omegaconf import OmegaConf
import pandas as pd

import arena
from arena import ReportArgs
from report_example import gen_example_report
from report_model import gen_model_report, write_data_tables, write_summary_table
from signal_noise import signal_to_noise
from utils import load_jsonl_files, check_data, fill_count

logger = logging.getLogger(__name__)


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
    records = load_jsonl_files(args.data)
    eval_results = pd.DataFrame(records)
    eval_results = fill_count(eval_results)
    check_data(eval_results)
    logger.info(f"Loaded {len(eval_results)} evaluation results")

    benchmarks = set(eval_results["benchmark_id"])
    logger.info(f"Included benchmarks: {benchmarks}")
    tmp_dir = Path(args.out_dir) / "tmp"

    if args.recompute:
        for bid in benchmarks:
            logger.info(f"Processing {bid}...")
            result_bid = eval_results[eval_results["benchmark_id"] == bid]
            arena_res: arena.ArenaResult = arena.summarize_benchmark(result_bid, args)

            sig_to_noise = signal_to_noise(bid, arena_res.summary)
            summary_stats = arena_res.summary_stats
            summary_stats["sig_noise"] = sig_to_noise["signal to noise"].median() if sig_to_noise is not None else float("nan")

            logger.info(f"Summary stats for {bid}:\n{pd.DataFrame([summary_stats])}")
            pd.DataFrame([summary_stats]).to_json(tmp_dir / f"summary-{bid}.jsonl", orient="records", lines=True)

            gen_model_report(bid, arena_res, args.out_dir)
            gen_example_report(bid, arena_res, args.out_dir)
            write_data_tables(bid, arena_res, args.out_dir)

    if args.write_summary:
        records = load_jsonl_files(f"{tmp_dir}/summary-*.jsonl")
        logger.info(f"Loaded {len(records)} summary records")
        df_summary = pd.DataFrame(records)
        df_summary.to_csv(Path(args.out_dir) / "summary.csv")
        write_summary_table(pd.DataFrame(df_summary), Path(args.out_dir) / "index.html", include_var_components=args.include_var_components)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    cli_args = OmegaConf.from_cli()
    default_cfg = OmegaConf.structured(ReportArgs())
    args = OmegaConf.merge(default_cfg, cli_args)
    args = OmegaConf.to_object(args)
    logger.info(f"Running with args: {args}")
    setup_output(args)
    run_arena(args)