import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats

import plotly.express as px
import plotly.graph_objects as go
from jinja2 import Template

from arena import ArenaResult

def fig_diff_vs_sum(bmname: str, summary: pd.DataFrame):
    data_sz = summary.iloc[0]["total"]

    figs = px.scatter(summary, x=summary["sum(A-B)"].abs(), y="sum(A!=B)",
                      custom_data=["model_a", "model_b", "sum(A!=B)", "sum(A-B)", "pvalue", "std(A-B)", "accA", "accB", "std(E(A-B))", "corr(A,B)"])
    figs.update_traces(hovertemplate=
        "<br>".join([
        "Model A: %{customdata[0]} (acc: %{customdata[6]:.1%})",
        "Model B: %{customdata[1]} (acc: %{customdata[7]:.1%})", 
        "total A≠B: %{customdata[2]}",
        "total A-B: %{customdata[3]}", 
        "std(A-B): %{customdata[5]:.4%}", 
        "std(E[A-B]): %{customdata[8]:.4%}", 
        "p-value: %{customdata[4]:.3g}", 
        "corr(A,B): %{customdata[9]:.3g}", 
        ])  + "<extra></extra>")
    figs.update_traces(
        marker=dict(
            size=3,
            opacity=0.5, 
        )
    )

    maxy = summary["sum(A!=B)"].max()
    refs = []
    x = np.linspace(0, data_sz / 2, 100)
    refs.append(pd.DataFrame({"x": x, "y": x, "type": "x=y"}))
    for alpha in [0.05, 0.1, 0.2]:
        thres = stats.chi2.ppf(1-alpha, 1)
        y = np.linspace(1, maxy, 200)
        refs.append(pd.DataFrame({"x": 1 + np.sqrt(y * thres), "y": y, "type": f"pvalue={alpha}"}))
    
    df_ref = pd.concat(refs, axis=0)
    figl = px.line(df_ref, x="x", y="y", color="type")
    figl.update_layout(hovermode=False)

    fig = go.Figure(data=figl.data + figs.data)
    fig.update_layout(
        width=800, height=600, title=bmname,
        xaxis_title="|#A_win - #B_win|",
        yaxis_title="#A_win + #B_win"
    )
    return fig


def fig_accs_and_pvalues(bmname, diffvsum):
    figs = px.scatter(diffvsum, x="accA", y="accB",
            color="pvalue", range_color=[0, 0.2],
            custom_data=["model_a", "model_b", "accA", "accB", "pvalue", "std(A-B)"])
    figs.update_traces(hovertemplate=
        "<br>".join([
        "Model A: %{customdata[0]}",
        "Model B: %{customdata[1]}", 
        "acc(A): %{customdata[2]:.1%}", 
        "acc(B): %{customdata[3]:.1%}", 
        "p-value: %{customdata[4]:.3g}", 
        "std(A-B)%: %{customdata[5]:.2%}",
        ])  + "<extra></extra>")
    
    figs.update_layout(
        width=800, height=800,
        title=bmname,
        xaxis_title="acc(Model A)",
        yaxis_title="acc(Model B)",
        legend_title="p_value",
    )
    return figs

def trend_df(selected: pd.DataFrame):
    marginals = selected.groupby(["example_id"]).agg({'pass1': 'mean'}).reset_index().sort_values(by="pass1")
    m1 = marginals["pass1"].to_numpy().copy()
    def independent_var(p: np.ndarray, alpha=1) -> float:
        """
        calculate the variance of two independent draws from p, X_i, Y_i ~ Bernoulli(p_i), I want E[(X_i - Y_i)**2]
        """
        N = len(p)
        assert np.all(p >= 0) and np.all(p <= 1)
        return np.sqrt(1 / N * np.mean(p * (1 - p)))

    df = pd.DataFrame({"alpha": np.logspace(-5, 5, 1000)})
    # df = pd.DataFrame({"alpha": np.linspace(0, 1, 200)})

    df["p_mean"] = df["alpha"].map(lambda alpha: np.mean(np.clip(m1 * alpha * 5, 0, 1)))
    df["vars"] = df["alpha"].map(lambda alpha: independent_var(np.clip(m1 * alpha * 5, 0, 1)))
    m2 = np.where((0 < m1) & (m1 < 1), 0.5, m1)
    df["p_mean_const"] = df["alpha"].map(lambda alpha: np.power(m2, alpha).mean())
    df["vars_const"] = df["alpha"].map(lambda alpha: independent_var(np.power(m2, alpha)))
    return df


def fig_cov_baseline(bmname: str, df_summary: pd.DataFrame, input_table: pd.DataFrame | None, sigma_thres=5.0):
    if input_table is not None:
        dfmodel = trend_df(input_table)
    
    df = df_summary
    # df["is_close"] = np.where(df["sum(A-B)"].abs() < df["total"] / 20, "close", "not_close")
    # df = df[df["accA"] >= df["accB"]]
    close_text = f"close: ≤{sigma_thres}σ"
    not_close_text = f"not: >{sigma_thres}σ"
    df["is_close"] = np.where(np.abs(df["accA"] - df["accB"]) / df["std(A-B)"] <= sigma_thres, close_text, not_close_text)
    color_map = {
        close_text: "blue",      # Bright red
        not_close_text: "#999999"     # Light gray
    } 
    figs = px.scatter(df,
                    x=df["accA"], y="std(A-B)",
                    color="is_close",
                    color_discrete_map=color_map,
                    custom_data=["model_a", "model_b", "sum(A!=B)", "sum(A-B)", "pvalue", "std(A-B)", "accA", "accB", "corr(A,B)"])
    figs.for_each_trace(lambda trace: trace.update(opacity=0.75) 
                   if trace.name == not_close_text else None)
    
    figs.update_traces(hovertemplate=
        "<br>".join([
        "Model A: %{customdata[0]} (acc: %{customdata[6]:.1%})",
        "Model B: %{customdata[1]} (acc: %{customdata[7]:.1%})", 
        "total A≠B: %{customdata[2]:.1f}",
        "total A-B: %{customdata[3]:.1f}", 
        "std(A-B): %{customdata[5]:.2%}", 
        "p-value: %{customdata[4]:.3g}",
        "corr(A,B): %{customdata[8]:.3g}",
        ])  + "<extra></extra>")

    figs.update_traces(
        marker=dict(
            size=3,
            opacity=0.5, 
        )
    )

    data_sz = df_summary.iloc[0]["total"]
    x = np.linspace(0, 1, 100)
    y = np.sqrt(x*(1-x) / data_sz)

    figl = go.Figure()

    figl.add_trace(go.Scatter(
        x=x, y=y, name="corr=0.5",
        # hoverinfo="skip",
        line=dict(color="lightgreen")
    ))

    figl.add_trace(go.Scatter(
        x=x, y=np.sqrt(2)*y, name="corr=0",
        # hoverinfo="skip",
        line=dict(color="darkgreen")
    ))

    figl.add_trace(go.Scatter(
        x=dfmodel["p_mean"], y=dfmodel["vars"],
        # hoverinfo="skip",
        line=dict(color="red"),
        name="scale clip corr=0"
    ))
    figl.add_trace(go.Scatter(
        x=dfmodel["p_mean_const"], y=dfmodel["vars_const"],
        # hoverinfo="skip",
        line=dict(color="pink"),
        name="const corr=0.5"
    ))

    fig = go.Figure(data=figl.data + figs.data)
    fig.update_layout(
        width=800, height=600, title=bmname,
        xaxis_title="acc(A)",
        yaxis_title="std(A-B)"
    )
    return fig

def get_sections(bres: ArenaResult, benchmark_id):
    summary = bres.summary 
    plotly_configs = dict(full_html=False, include_plotlyjs="cdn")
    sections = {
        "fig_accs_and_pvalues": fig_accs_and_pvalues(benchmark_id, summary).to_html(**plotly_configs),
        "fig_diff_vs_sum": fig_diff_vs_sum(benchmark_id, summary).to_html(**plotly_configs),
        "fig_cov_baseline": fig_cov_baseline(benchmark_id, summary, bres.input_table).to_html(**plotly_configs),
        "model_table": bres.model_table.to_html(
            index=False,
            classes="number-table",
            formatters={
                "pass1": lambda x: f"{100*x:.3g}",
                "std(A)": lambda x: f"{100*x:.2g}",
                "std(E(A))": lambda x: f"{100*x:.2g}",
                "E(std(A))": lambda x: f"{100*x:.2g}",
                "N": lambda x: f"{x:.2g}",
                "win_rate": lambda x: f"{100*x:.3g}",
                "elo": "{:.3g}".format
        }),
    }
    return sections


def summary_stats(s, f=2, percent=True):
    return f"""{s["mean"]:.2g}±{s["std"]:.2g} | [{s["min"]:.2g}--{s["max"]:.2g}] | n={int(s["count"])}"""

def format_stats_badge(s):
    s_percent = dict(s)
    print(s)
    for st in ["mean", "std", "min", "max"]:
        if s["count"] == 0:
            s[st] = float("nan")
        else:
            s_percent[st] = 100 * s[st]
    summary = summary_stats(s_percent)
    mean = 100*s["mean"]
    return f"""<span class="tooltip" data-tooltip="{summary}">{mean:.2g}</span>"""

def write_summary_table(summary_count: pd.DataFrame, output_path: Path):
    summary_count = summary_count.sort_values(by="benchmark_id")

    def link_detail(bid):
        links = []
        links.append(f"""<a href="model_{bid}.html">models </a> """)
        links.append(f"""<a href="ex_{bid}.html"> examples </a>""")
        links.append(f"""<a href="ex_v_model_{bid}.html"> data </a>""")
        links.append(f"""<a href="data_{bid}.html"> raw </a>""")
        return "|".join(links)
    summary_count["details"] = summary_count["benchmark_id"].apply(link_detail)

    def normalize(counts, includes):
        percent = counts.copy(deep=True)
        for c in includes:
            percent[c] = percent[c] / percent["size"]
        return percent

    includes_cols = ["benchmark_id", "size", "models", "std(A)", "std(E(A))", "std(A-B)", "corr(A,B)", "no_solve", "tau-", "sig_noise", "details"]
    percent_cols = ["p5_min", "p5_max", "no_solve", "tau-"]
    summary_percent = normalize(summary_count, percent_cols)

    print(summary_percent)
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
                        "std(A)": lambda x: format_stats_badge(x),
                        "std(E(A))": lambda x: format_stats_badge(x),
                        "std(A-B)": lambda x: format_stats_badge(x),
                        "std(E(A-B))": lambda x: format_stats_badge(x),
                        "corr(A,B)": lambda x: format_stats_badge(x),
                        "no_solve": lambda x: f"{x*100:.2g}",
                        "tau-": lambda x: f"{x*100:.2g}",
                        "sig_noise": "{:.2g}".format,
                    }),
            }))
            

def gen_model_report(benchmark_id: str, ares: ArenaResult, OUTPUT_PATH):
    sections = get_sections(ares, benchmark_id)
    template_path=r"templates/template_model.html"
    output_path = f"{OUTPUT_PATH}/model_{benchmark_id}.html"
    with open(output_path, "w", encoding="utf-8") as output_file:
        with open(template_path) as template_file:
            j2_template = Template(template_file.read())
            output_file.write(j2_template.render({"benchmark_id": benchmark_id, "sections": sections}))


def write_data_tables(benchmark_id: str, ares: ArenaResult, OUTPUT_PATH):
    template_path=r"templates/template_data.html"
    output_path = f"{OUTPUT_PATH}/data_{benchmark_id}.html"
    with open(output_path, "w", encoding="utf-8") as output_file:
        with open(template_path) as template_file:
            j2_template = Template(template_file.read())
            output_file.write(j2_template.render({"benchmark_id": benchmark_id}))

    data_path = Path(f"{OUTPUT_PATH}/data/{benchmark_id}/")
    os.makedirs(data_path, exist_ok=True)
    ares.input_table.to_csv(data_path / "input_table.csv")
    ares.model_table.to_csv(data_path / "model_table.csv")
    ares.example_table.to_csv(data_path / "example_table.csv")
    ares.summary.to_csv(data_path / "summary.csv")
