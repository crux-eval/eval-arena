import logging
import os
from pathlib import Path

from jinja2 import Template
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats

from arena import ArenaResult

logger = logging.getLogger(__name__)

PLOTLY_CONFIGS = dict(full_html=False, include_plotlyjs="cdn")

def fig_diff_vs_sum(bmname: str, df_summary: pd.DataFrame, perf_thres: float = 0.05):
    df = df_summary.copy()
    data_sz = df.iloc[0]["total"]

    has_ok_perf = (df["accA"] > perf_thres) & (df["accB"] > perf_thres) 
    df = df[has_ok_perf]


    figs = px.scatter(df, x=df["sum(A-B)"].abs(), y="sum(A!=B)", 
                      custom_data=["model_a", "model_b", "sum(A!=B)", "sum(A-B)", "pvalue", "SE(A-B)", "accA", "accB", "SE_x(A-B)", "corr(A,B)"])
    figs.update_traces(hovertemplate=
        "<br>".join([
        "Model A: %{customdata[0]} (acc: %{customdata[6]:.1%})",
        "Model B: %{customdata[1]} (acc: %{customdata[7]:.1%})",
        "total A≠B: %{customdata[2]}",
        "total A-B: %{customdata[3]}",
        "SE(A-B): %{customdata[5]:.4%}",
        "SE_x[A-B]: %{customdata[8]:.4%}",
        "p-value: %{customdata[4]:.3g}",
        "corr(A,B): %{customdata[9]:.3g}",
        ])  + "<extra></extra>")
    figs.update_traces(
        marker=dict(
            size=3,
            opacity=0.5, 
        )
    )

    maxy = df["sum(A!=B)"].max()
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
            custom_data=["model_a", "model_b", "accA", "accB", "pvalue", "SE(A-B)"])
    figs.update_traces(hovertemplate=
        "<br>".join([
        "Model A: %{customdata[0]}",
        "Model B: %{customdata[1]}", 
        "acc(A): %{customdata[2]:.1%}", 
        "acc(B): %{customdata[3]:.1%}", 
        "p-value: %{customdata[4]:.3g}", 
        "SE(A-B)%: %{customdata[5]:.2%}",
        ])  + "<extra></extra>")
    
    figs.update_layout(
        width=800, height=800,
        title=bmname,
        xaxis_title="acc(Model A)",
        yaxis_title="acc(Model B)",
        legend_title="p_value",
    )
    return figs


def fig_cov_baseline(bmname: str, df_summary: pd.DataFrame, input_table: pd.DataFrame | None, sigma_thres=5.0):
    df = df_summary.copy()
    CLOSE = f"SE(A-B) close model: ≤{sigma_thres}σ"
    NOT_CLOSE = f"SE(A-B) not close: >{sigma_thres}σ"
    SAME_MODEL = f"same model"
    def label_fun(r):
        if r["model_a"] == r["model_b"]:
            return SAME_MODEL
        elif np.abs(r["accA"] - r["accB"]) / (r["SE(A-B)"] + 1e-10) <= sigma_thres:
            return CLOSE
        else:
            return NOT_CLOSE

    df["type"] = df.apply(label_fun, axis=1)

    df = df[df["type"] == CLOSE]
    hover_data = df[["model_a", "model_b", "accA", "accB", "SE(A-B)", "SE_x(A-B)", "SE_pred(A-B)"]].values
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["accA"],
        y=df["SE(A-B)"],
        mode="markers",
        name="SE(A-B)",
        customdata=hover_data,
        marker=dict(size=3, symbol="x", color="blue", opacity=0.8),
    ))

    fig.add_trace(go.Scatter(
        x=df["accA"],
        y=df["SE_x(A-B)"],
        mode="markers",
        name="SE_x(A-B)",
        customdata=hover_data,
        marker=dict(size=3, symbol="circle", color="red", opacity=0.8),
        visible='legendonly', # hide series by default
    ))

    fig.add_trace(go.Scatter(
        x=df["accA"],
        y=df["SE_pred(A-B)"],
        mode="markers",
        name="SE_pred(A-B)",
        customdata=hover_data,
        marker=dict(size=3, symbol="square", color="green", opacity=0.8),
        visible='legendonly', # hide series by default
    ))

    # fig.for_each_trace(lambda trace: trace.update(opacity=0.75) 
    #                if trace.name == NOT_CLOSE else None)
    
    fig.update_traces(hovertemplate=
        "<br>".join([
        "Model A: %{customdata[0]} (acc: %{customdata[2]:.1%})",
        "Model B: %{customdata[1]} (acc: %{customdata[3]:.1%})", 
        "SE(A-B): %{customdata[4]:.2%}", 
        "SE_x(A-B): %{customdata[5]:.2%}", 
        "SE_pred(A-B): %{customdata[6]:.2%}", 
        ])  + "<extra></extra>")

    fig.update_traces(
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
        x=x, y=np.sqrt(2)*y, name="indep. theory",
        # hoverinfo="skip",
        line=dict(color="black", dash="dash")
    ))

    marginals = input_table.groupby(["example_id"]).agg({'pass1': 'mean'}).reset_index()
    m1 = marginals["pass1"].to_numpy().copy()
    max_acc = np.mean(m1 > 0) # excludes questions not solved by model
    x = np.linspace(0, max_acc, 100)
    y = np.sqrt(x/max_acc*(1-x/max_acc) * (max_acc) / (data_sz))

    figl.add_trace(go.Scatter(
        x=x, y=y,
        # hoverinfo="skip",
        line=dict(color="blue"),
        name="beta theory"
    ))

    fig = go.Figure(data=figl.data + fig.data)
    fig.update_layout(
        width=800, height=600, title=bmname,
        xaxis_title="E(A)",
        yaxis_title="SE(A-B)"
    )
    return fig

def beta_est(mean, var):
    nu = (mean * (1 - mean) / var) - 1
    if nu <= 0:
        logger.warning(f"Invalid parameter estimates. Check if data follows beta distribution. {nu=}")
        nu = 1e-2

    alpha_hat = mean * nu
    beta_hat = (1 - mean) * nu
    # Sanity check
    if alpha_hat <= 0 or beta_hat <= 0:
        raise ValueError("Estimated parameters must be positive")
    return float(alpha_hat), float(beta_hat)

def fig_marginals(bmname: str, df_input, df_model, df_example, xkey="pass1_of_ex",
                  exclude_distill=True, exclude_paired=True, interval_size=0.125):
    df = df_input[["model", "example_id", "pass1", "count"]].merge(df_example[["example_id", "pass1_of_ex"]], on="example_id")
    if exclude_distill:
        df_model = df_model[~df_model["model"].str.contains(r"_distill_", na=False)]
    model_table = df_model[["model", "pass1"]].rename(columns={"pass1": "pass1_of_model"})

    df = df.merge(model_table, on="model")
    fig = go.Figure()
    nzs = np.sum(df_example["pass1_of_ex"] == 0)
    for i, start in enumerate(np.linspace(0, 1, 9)):
        models = model_table[(model_table["pass1_of_model"] >= start) & (model_table["pass1_of_model"] < start + interval_size)]
        # display(models)
        data_inside = df[df['model'].isin(models["model"])]
        if len(data_inside) == 0:
            continue
        data_means = data_inside.groupby("example_id").agg({"pass1": "mean", "pass1_of_ex": "mean", "count": "mean"}).reset_index()
        # Merge with original marginals to ensure same sorting
        data_means = data_means.sort_values(by="pass1_of_ex")
        data_means["rank_of_ex"] = np.arange(len(data_means))

        data_means = data_means.sort_values(by="pass1")
        data_means["rank"] = np.arange(len(data_means))
        smoothed = data_means[("pass1")].rolling(window=1, min_periods=1, center=True).mean()
        mu = data_means["pass1"].mean()
        n = len(models)
        legend = f"{start:.2f}-{start+interval_size:.2f} ({n=}, {mu=:.2f})"
        colors = px.colors.qualitative.Plotly
        color_idx = i % len(colors)  # cycle through colors
        color = colors[color_idx]

        if not exclude_paired:
            fig.add_scatter(y=data_means["rank_of_ex"], x=data_means["pass1"],
                mode='markers',
                # showlegend=False,
                legendgroup=legend,
                name="rank_" + legend,
                marker=dict(
                    size=1,
                    opacity=0.9,
                    color=color,
                )
            )

        fig.add_scatter(y=data_means[xkey], x=data_means["pass1"],
            mode='lines',
            # showlegend=False,
            legendgroup=legend,
            name="CDF " + legend,
            line=dict(
                dash='solid', 
                width=2,
                color=color
            ),
            opacity=0.5,
        )

        def add_beta(fig):
            from scipy.stats import beta as betaf
            x = np.linspace(0, 1, 100)
            data_nzs = data_means[data_means["pass1_of_ex"] != 0]
            # data_nzs = data_means
            mu = data_nzs["pass1"].mean()
            var = data_nzs["pass1"].var(ddof=1)
            alpha, beta = beta_est(mu, var)
            cdf_values = betaf.cdf(x, alpha, beta)
            beta_mean = (1 - nzs/len(smoothed)) * alpha / (alpha + beta)
            logger.debug(f"nzs={nzs}, len(smoothed)={len(smoothed)}")
            y = nzs/len(smoothed) + cdf_values * (1 - nzs/len(smoothed))
            # y = cdf_values 
            y = y * len(smoothed)
            fig.add_scatter(
                x=x, 
                y=y, 
                mode='lines',
                name=f'Beta({alpha:.2f}, {beta:.2f}) mu={beta_mean:.2f}',
                # legendgroup=legend,
                line=dict(
                    dash='dot', 
                    width=2,
                    color=color
                )
            )
        add_beta(fig)

    fig.update_layout(
        width=800, height=600,
        title=f"cdf on {bmname}",
    ) 
    return fig


def experimental(benchmark_id: str, ares: ArenaResult, OUTPUT_PATH):
    fig_pref = "fig_marginal"
    os.makedirs(Path(OUTPUT_PATH) / "experimental" / fig_pref, exist_ok=True)
    with open(Path(OUTPUT_PATH) / "experimental" / fig_pref / f"{fig_pref}_{benchmark_id}.html", "w", encoding="utf-8") as output_file:
        fig = fig_marginals(ares.input_table, ares.model_table, ares.example_table, xkey="rank")
        html = fig.to_html(**PLOTLY_CONFIGS)
        output_file.write(html)

    fig_pref = "fig_ex_hist"
    os.makedirs(Path(OUTPUT_PATH) / "experimental" / fig_pref, exist_ok=True)
    with open(Path(OUTPUT_PATH) / "experimental" / fig_pref / f"{fig_pref}_{benchmark_id}.html", "w", encoding="utf-8") as output_file:
        fig =  px.histogram(ares.example_table, x="pass1_of_ex")
        html = fig.to_html(**PLOTLY_CONFIGS)
        output_file.write(html)


def get_sections(res: ArenaResult, benchmark_id):
    summary = res.summary 
    
    sections = {
        "fig_accs_and_pvalues": fig_accs_and_pvalues(benchmark_id, summary).to_html(**PLOTLY_CONFIGS),
        "fig_diff_vs_sum": fig_diff_vs_sum(benchmark_id, summary).to_html(**PLOTLY_CONFIGS),
        "fig_cov_baseline": fig_cov_baseline(benchmark_id, summary, res.input_table).to_html(**PLOTLY_CONFIGS),
        "fig_marginals": fig_marginals(benchmark_id, res.input_table, res.model_table, res.example_table, xkey="rank").to_html(**PLOTLY_CONFIGS),
        "model_table": res.model_table.to_html(
            index=False,
            classes="number-table",
            formatters={
                "pass1": lambda x: f"{100*x:.3g}",
                "win_rate": lambda x: f"{100*x:.3g}",
                "SE(A)": lambda x: f"{100*x:.2g}",
                "SE_x(A)": lambda x: f"{100*x:.2g}",
                "SE_pred(A)": lambda x: f"{100*x:.2g}",
                "count": lambda x: f"{x:.2g}",
        }),
    }
    return sections


def summary_stats(s, f=2, percent=True):
    if s["count"] == 0:
        return "n=0"
    return f"""{s["mean"]:.2g}±{s["std"]:.2g} | [{s["min"]:.2g}--{s["max"]:.2g}] | n={int(s["count"])}"""

def format_stats_badge(s):
    s_percent = dict(s)
    for st in ["mean", "std", "min", "max"]:
        if s["count"] != 0:
            s_percent[st] = 100 * s[st]
    summary = summary_stats(s)
    mean = s["mean"]
    mean_str = "N/A" if mean is None else f"{100*mean:.2g}"
    return f"""<span class="tooltip" data-tooltip="{summary}">{mean_str}</span>"""

def write_summary_table(summary_count: pd.DataFrame, output_path: Path, include_var_components: bool = False):
    summary_count = summary_count.sort_values(by="benchmark_id")

    def link_detail(bid):
        links = []
        links.append(f"""<a href="model_{bid}.html">models </a> """)
        links.append(f"""<a href="ex_{bid}.html"> examples </a>""")
        links.append(f"""<a href="ex_v_model_acc_{bid}.html"> data </a>""")
        links.append(f"""<a href="data_{bid}.html"> raw </a>""")
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
                        "SE(A)": lambda x: format_stats_badge(x),
                        "SE_x(A)": lambda x: format_stats_badge(x),
                        "SE(A-B)": lambda x: format_stats_badge(x),
                        "SE_x(A-B)": lambda x: format_stats_badge(x),
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

    data_path = Path(f"{OUTPUT_PATH}/assets/tables/")
    os.makedirs(data_path, exist_ok=True)
    ares.input_table.to_csv(data_path / f"input_table_{benchmark_id}.csv")
    ares.model_table.to_csv(data_path / f"model_table_{benchmark_id}.csv")
    ares.example_table.to_csv(data_path / f"example_table_{benchmark_id}.csv")
    ares.summary.to_csv(data_path / f"summary_{benchmark_id}.csv")
