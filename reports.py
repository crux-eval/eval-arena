import logging
import os
from pathlib import Path

from jinja2 import Template
import numpy as np
import pandas as pd
import plotly.express as px

from arena import ArenaResult
from figures import (
    fig_accs_and_pvalues,
    fig_cov_baseline,
    fig_diff_vs_sum,
    fig_example_vs_model,
    fig_marginals,
    fig_pass_at_k,
)

PLOTLY_CONFIGS = dict(full_html=False, include_plotlyjs="cdn")
logger = logging.getLogger(__name__)

# Helper Functions
# =============================================================================

def _summary_stats(s, f=2, percent=True):
    if s["count"] == 0:
        return "n=0"
    return f"""{s["mean"]:.2g}±{s["std"]:.2g} | [{s["min"]:.2g}--{s["max"]:.2g}] | n={int(s["count"])}"""

def _format_stats_badge(s):
    s_percent = dict(s)
    for st in ["mean", "std", "min", "max"]:
        if s["count"] != 0:
            s_percent[st] = 100 * s[st]
    summary = _summary_stats(s)
    mean = s["mean"]
    mean_str = "N/A" if mean is None else f"{100*mean:.2g}"
    return f"""<span class="tooltip" data-tooltip="{summary}">{mean_str}</span>"""

def _get_anchor(benchmark_id: str, example_id: str):
    """
    link to the actual questions and outputs on selected benchmarks for easier inspection
    """
    def get_link():
        if benchmark_id in ["humaneval", "humaneval+", "mbpp", "mbpp+"]:
            if "/" in example_id:
                dir, id = example_id.split("/") # expecting HumanEval/93 and Mbpp/622 etc
            else:
                return ""
            return f"https://all-the-noises.github.io/evalplus/{dir}/{id}.html"
        elif benchmark_id in ["CRUXEval-input", "CRUXEval-output"]:
            id = example_id.replace(benchmark_id + "/", "")
            return f"https://crux-eval.github.io/demo.html?id={int(id) + 1}"
        else:
            return ""
    link = get_link()
    if link != "":
        return f"""<a href="{get_link()}">{example_id}</a>"""
    else:
        return example_id

# Section Generation Functions
# =============================================================================

def get_sections(res: ArenaResult, benchmark_id):
    summary = res.summary

    sections = {
        "fig_accs_and_pvalues": fig_accs_and_pvalues(benchmark_id, summary).to_html(**PLOTLY_CONFIGS),
        "fig_diff_vs_sum": fig_diff_vs_sum(benchmark_id, summary).to_html(**PLOTLY_CONFIGS),
        "fig_cov_baseline": fig_cov_baseline(benchmark_id, summary, res.input_table).to_html(**PLOTLY_CONFIGS),
        "fig_marginals": fig_marginals(benchmark_id, res.input_table, res.model_table, res.example_table, xkey="rank").to_html(**PLOTLY_CONFIGS),
        "fig_pass_at_k": fig_pass_at_k(benchmark_id, res.input_table).to_html(**PLOTLY_CONFIGS),
        "model_table": res.model_table.to_html(
            index=False,
            classes="number-table",
            formatters={
                "pass1": lambda x: f"{100*x:.3g}",
                "pass@count": lambda x: f"{100*x:.3g}",
                "win_rate": lambda x: f"{100*x:.3g}",
                "SE(A)": lambda x: f"{100*x:.2g}",
                "SE_x(A)": lambda x: f"{100*x:.2g}",
                "SE_pred(A)": lambda x: f"{100*x:.2g}",
                "count": lambda x: f"{x:.2g}",
        }),
    }
    return sections


def get_example_level_results(benchmark_id, ares: ArenaResult):
    all_stats = ares.model_table
    ex_table = ares.example_table
    ex_table["example_link"] = ex_table["example_id"].apply(lambda x: _get_anchor(benchmark_id, x))

    outputs = {}
    outputs["result table"] = all_stats.sort_values(by="pass1", ascending=False).to_html(classes="number-table", float_format="%10.3f")
    plotly_configs = dict(full_html=False, include_plotlyjs="cdn")
    outputs["fig_min_rating_solve"] = px.histogram(ex_table, x="min_pass1_of_model", marginal="rug", title="min pass1 to solve").to_html(**plotly_configs)
    outputs["table_histogram_accs"] = px.histogram(ex_table, x="pass1_of_ex", marginal="rug", title="accuracy on examples").to_html(**plotly_configs)

    no_solve = ex_table[ex_table["num_solved"] == 0]
    outputs["list_no_solve"] = sorted(no_solve["example_link"].to_list())
    one_solve = ex_table[ex_table["num_solved"] == 1]
    pd.options.mode.chained_assignment = None
    one_solve["model"] = one_solve["models"].apply(lambda x: x[0])
    one_solve = one_solve.sort_values(by="min_pass1_of_model", ascending=False)
    one_solve = one_solve[["example_link", "model", "min_pass1_of_model"]]
    outputs["table_one_solve"] = one_solve.to_html(escape=False, classes="number-table", float_format="%10.3f", index=False)

    list_suspect = ex_table.sort_values(by="tau", ascending=True).head(10)
    outputs["table_suspect"] = list_suspect[["example_link", "pass1_of_ex", "tau"]].to_html(escape=False, classes="number-table", float_format="%10.3f", index=False)
    logger.info(f"{benchmark_id} anti-correlated prop: {np.mean(ex_table['tau'] <= 0):.3f}")

    outputs["fig_example_vs_model"] = fig_example_vs_model(ares.input_table, all_stats, ex_table)
    outputs["fig_example_vs_model_acc"] = fig_example_vs_model(ares.input_table, all_stats, ex_table, use_acc_as_position=True)
    return outputs

# write utilties
# =============================================================================

def write_figures(sections: dict, OUTPUT_PATH):
    sections_dir = Path(OUTPUT_PATH) / "sections"
    os.makedirs(sections_dir, exist_ok=True)

    # Write each section to its own file
    for section_name, section_content in sections.items():
        section_path = sections_dir / f"{section_name}.html"
        with open(section_path, "w", encoding="utf-8") as output_file:
            output_file.write(section_content)


def write_data_tables(ares: ArenaResult, OUTPUT_PATH):
    """Write data tables (CSVs) to the tables directory."""
    data_path = Path(OUTPUT_PATH) / "tables"
    os.makedirs(data_path, exist_ok=True)
    ares.input_table.to_csv(data_path / "input.csv", index=True)
    ares.model_table.to_csv(data_path / "model.csv")
    ares.example_table.to_csv(data_path / "example.csv")
    ares.summary.to_csv(data_path / "summary.csv")


def write_directory_index(benchmark_id: str, OUTPUT_PATH):
    """Generate raw_index.html with directory listing of all files."""
    base_path = Path(OUTPUT_PATH)

    # Get tables
    table_files = []
    if (base_path / "tables").exists():
        table_files = sorted([f.name for f in (base_path / "tables").iterdir() if f.is_file()])

    # Get sections (figures)
    section_files = []
    if (base_path / "sections").exists():
        section_files = sorted([f.name for f in (base_path / "sections").iterdir() if f.is_file()])

    # Get root HTML files (reports)
    report_files = sorted([f.name for f in base_path.iterdir() if f.is_file() and f.suffix == '.html'])

    template_path=r"templates/template_data.html"
    output_path = Path(OUTPUT_PATH) / "raw_index.html"
    with open(output_path, "w", encoding="utf-8") as output_file:
        with open(template_path) as template_file:
            j2_template = Template(template_file.read())
            output_file.write(j2_template.render({
                "benchmark_id": benchmark_id,
                "table_files": table_files,
                "section_files": section_files,
                "report_files": report_files,
            }))


def write_sections_index(out_dir: Path):
    """
    Generate ${OUTPATH}/raw_index.html listing all section files across benchmarks.
    """
    sections_index_dir = out_dir / "raw_index.html"
    sections_index_dir.mkdir(parents=True, exist_ok=True)

    html = ['<!DOCTYPE html><html><head>',
            '<meta charset="utf-8">',
            '<title>Sections Index</title>',
            '<link rel="stylesheet" href="../static/css/bulma.min.css">',
            '<link rel="stylesheet" href="../static/css/custom.css">',
            '</head><body>',
            '<section class="section"><div class="container">',
            '<h1 class="title">All Sections</h1>']

    # Find all benchmark directories with sections
    for benchmark_dir in sorted(out_dir.iterdir()):
        sections_path = benchmark_dir / "sections"
        if sections_path.is_dir():
            html.append(f'<h2 class="subtitle" style="margin-top: 1.5rem;">{benchmark_dir.name}</h2><ul>')
            for f in sorted(sections_path.iterdir()):
                if f.is_file() and f.suffix == '.html':
                    html.append(f'<li><a href="../{benchmark_dir.name}/sections/{f.name}">{f.stem}</a></li>')
            html.append('</ul>')

    html.append('</div></section></body></html>')

    index_path = sections_index_dir / "index.html"
    with open(index_path, "w", encoding="utf-8") as f:
        f.write('\n'.join(html))


def write_summary_table(summary_count: pd.DataFrame, output_path: Path, include_var_components: bool = False):
    summary_count = summary_count.sort_values(by="benchmark_id")

    def link_detail(bid):
        links = []
        links.append(f"""<a href="{bid}/model.html">models </a> """)
        links.append(f"""<a href="{bid}/ex.html"> examples </a>""")
        links.append(f"""<a href="{bid}/ex_v_model_acc.html"> data </a>""")
        links.append(f"""<a href="{bid}/raw_index.html"> raw </a>""")
        return "|".join(links)
    summary_count["details"] = summary_count["benchmark_id"].apply(link_detail)

    def normalize(counts, includes):
        percent = counts.copy(deep=True)
        for c in includes:
            percent[c] = percent[c] / percent["size"]
        return percent
    includes_cols = ["benchmark_id", "size", "models", "SE(A)", "SE_x(A)", "SE(A-B)", "SE_x(A-B)", "corr(A,B)", "no_solve", "tau-", "details"]
    if not include_var_components:
        includes_cols = [c for c in includes_cols if not c.startswith("SE_x")]
    percent_cols = ["no_solve", "tau-"]
    summary_percent = normalize(summary_count, percent_cols)

    logger.info(f"Summary statistics:\n{summary_percent}")
    template_path = r"templates/summary.html"

    with open(output_path, "w", encoding="utf-8") as output_file:
        with open(template_path) as template_file:
            j2_template = Template(template_file.read())
            output_file.write(j2_template.render({
                "count_table": summary_count[includes_cols].to_html(escape=False, index=False),
                "percent_table": summary_percent[includes_cols].to_html(
                    escape=False,
                    classes="number-table",
                    index=False,
                    formatters={
                        "SE(A)": lambda x: _format_stats_badge(x),
                        "SE_x(A)": lambda x: _format_stats_badge(x),
                        "SE(A-B)": lambda x: _format_stats_badge(x),
                        "SE_x(A-B)": lambda x: _format_stats_badge(x),
                        "corr(A,B)": lambda x: _format_stats_badge(x),
                        "no_solve": lambda x: f"{x*100:.2g}",
                        "tau-": lambda x: f"{x*100:.2g}",
                        "sig_noise": "{:.2g}".format,
                    }),
            }))

# Reports
# =============================================================================

def write_model_report(benchmark_id: str, ares: ArenaResult, OUTPUT_PATH):
    sections = get_sections(ares, benchmark_id)
    write_figures(sections, OUTPUT_PATH)
    template_path=r"templates/template_model.html"
    output_path = Path(OUTPUT_PATH) / "model.html"
    with open(output_path, "w", encoding="utf-8") as output_file:
        with open(template_path) as template_file:
            j2_template = Template(template_file.read())
            output_file.write(j2_template.render({"benchmark_id": benchmark_id, "sections": sections}))


def write_example_report(benchmark_id: str, ares: ArenaResult, OUTPUT_PATH):
    outputs = get_example_level_results(benchmark_id, ares)
    template_path = r"templates/template_example.html"
    output_path = f"{OUTPUT_PATH}/ex.html"
    with open(output_path, "w", encoding="utf-8") as output_file:
        with open(template_path) as template_file:
            j2_template = Template(template_file.read())
            output_file.write(j2_template.render({"benchmark_id": benchmark_id, "outputs": outputs}))

    plotly_configs = dict(full_html=False, include_plotlyjs="cdn")
    with open(f"{OUTPUT_PATH}/ex_v_model.html", "wt") as f:
        f.write(outputs["fig_example_vs_model"].to_html(**plotly_configs))

    with open(f"{OUTPUT_PATH}/ex_v_model_acc.html", "wt") as f:
        f.write(outputs["fig_example_vs_model_acc"].to_html(**plotly_configs))
