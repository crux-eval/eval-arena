import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats

import plotly.express as px
import plotly.graph_objects as go
from jinja2 import Template

from arena import ArenaResult

PLOTLY_CONFIGS = dict(full_html=False, include_plotlyjs="cdn")

def fig_diff_vs_sum(bmname: str, summary: pd.DataFrame):
    data_sz = summary.iloc[0]["total"]

    figs = px.scatter(summary, x=summary["sum(A-B)"].abs(), y="sum(A!=B)",
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
    close_text = f"close model: ≤{sigma_thres}σ"
    not_close_text = f"not close: >{sigma_thres}σ"
    same_text = f"same model"
    def label_fun(r):
        if r["model_a"] == r["model_b"]:
            return same_text
        elif np.abs(r["accA"] - r["accB"]) / r["SE(A-B)"] <= sigma_thres:
            return close_text
        else:
            return not_close_text

    df["type"] = df.apply(label_fun, axis=1)
    color_map = {
        close_text: "blue",      # Bright red
        not_close_text: "#999999",     # Light gray
        same_text: "green",
    } 
    figs = px.scatter(df,
                    x=df["accA"], y="SE(A-B)",
                    color="type",
                    color_discrete_map=color_map,
                    custom_data=["model_a", "model_b", "sum(A!=B)", "sum(A-B)", "pvalue", "SE(A-B)", "accA", "accB", "corr(A,B)"])
    # add an extra scatter showing SE_x(A-B)
    figs.add_trace(go.Scatter(
        x=df["accA"],
        y=df["SE_x(A-B)"],
        mode="markers",
        name="SE_x(A-B)",
        customdata=df[["model_a", "model_b", "sum(A!=B)", "sum(A-B)", "pvalue", "SE(A-B)", "accA", "accB", "corr(A,B)"]].values,
        marker=dict(size=3, symbol="x", color="purple", opacity=0.8),
    ))

    figs.add_trace(go.Scatter(
        x=df["accA"],
        y=df["SE_pred(A-B)"],
        mode="markers",
        name="SE_pred(A-B)",
        customdata=df[["model_a", "model_b", "sum(A!=B)", "sum(A-B)", "pvalue", "SE(A-B)", "accA", "accB", "corr(A,B)"]].values,
        marker=dict(size=3, symbol="x", color="yellow", opacity=0.8),
    ))

    figs.for_each_trace(lambda trace: trace.update(opacity=0.75) 
                   if trace.name == not_close_text else None)
    
    figs.update_traces(hovertemplate=
        "<br>".join([
        "Model A: %{customdata[0]} (acc: %{customdata[6]:.1%})",
        "Model B: %{customdata[1]} (acc: %{customdata[7]:.1%})", 
        "total A≠B: %{customdata[2]:.1f}",
        "total A-B: %{customdata[3]:.1f}", 
        "SE(A-B): %{customdata[5]:.2%}", 
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

def beta_est(mean, var):
    nu = (mean * (1 - mean) / var) - 1
    if nu <= 0:
        print(f"Invalid parameter estimates. Check if data follows beta distribution. {nu=}")
        nu = 1e-2
    
    alpha_hat = mean * nu
    beta_hat = (1 - mean) * nu
    # Sanity check
    if alpha_hat <= 0 or beta_hat <= 0:
        raise ValueError("Estimated parameters must be positive")
    return float(alpha_hat), float(beta_hat)

def fig_marginals(df_input, df_model, df_example, xkey="pass1_of_ex", exclude_distill=True):
    df = df_input[["model", "example_id", "pass1", "count"]].merge(df_example[["example_id", "pass1_of_ex"]], on="example_id")
    if exclude_distill:
        df_model = df_model[~df_model["model"].str.contains(r"_distill_", na=False)]
    model_table = df_model[["model", "pass1"]].rename(columns={"pass1": "pass1_of_model"})

    df = df.merge(model_table, on="model")
    fig = go.Figure()
    interval_size = 0.125
    nzs = np.sum(df_example["pass1_of_ex"] == 0)
    max_example = df_example["pass1_of_ex"].max()
    min_example = df_example["pass1_of_ex"].min()
    for i, start in enumerate(np.linspace(0, 1, 9)):
        models = model_table[(model_table["pass1_of_model"] >= start) & (model_table["pass1_of_model"] < start + interval_size)]
        # display(models)
        data_inside = df[df['model'].isin(models["model"])]
        if len(data_inside) == 0:
            continue
        data_means = data_inside.groupby("example_id").agg({"pass1": "mean", "pass1_of_ex": "mean", "count": "mean"}).reset_index()
        # Merge with original marginals to ensure same sorting
        # wsz = 1
        # smoothed = data_means[("pass1")].rolling(window=5, center=True).mean()
        data_means = data_means.sort_values(by="pass1_of_ex")
        data_means["rank_of_ex"] = np.arange(len(data_means))

        data_means = data_means.sort_values(by="pass1")
        data_means["rank"] = np.arange(len(data_means))
        # display(data_means)
        # display(data_means)
        smoothed = data_means[("pass1")].rolling(window=1, min_periods=1, center=True).mean()
        mu = data_means["pass1"].mean()
        n = len(models)
        legend = f"{start:.2f}-{start+interval_size:.2f} ({n=}, {mu=:.3f})"
        colors = px.colors.qualitative.Plotly
        color_idx = i % len(colors)  # cycle through colors
        color = colors[color_idx]

        fig.add_scatter(y=data_means["rank_of_ex"], x=data_means["pass1"],
            mode='markers',
            # showlegend=False,
            legendgroup=legend,
            name="rank" + legend,
            marker=dict(
                size=1,
                opacity=0.9,
                color=color,
            )
        )
        # fig.add_scatter(x=data_means[xkey], y=smoothed,
        #     mode='lines',
        #     # showlegend=False,
        #     legendgroup=legend,
        #     name="smoothed" + legend,
        #     opacity=0.5,
        #     line=dict(
        #         dash='dash',
        #         width=2,
        #         color=color,
        #     )
        # )

        fig.add_scatter(y=data_means[xkey], x=data_means["pass1"],
            mode='lines',
            # showlegend=False,
            legendgroup=legend,
            name="sorted" + legend,
            line=dict(
                dash='solid', 
                width=2,
                color=color
            ),
            opacity=0.5,
        )

        x = np.linspace(0, 1, 100)
        # nzs = 0
        data_nzs = data_means[data_means["pass1_of_ex"] != 0]
        # data_nzs = data_means
        mu = data_nzs["pass1"].mean()
        var = data_nzs["pass1"].var(ddof=1)
        alpha, beta = beta_est(mu, var)
        from scipy.stats import beta as betaf
        s = 1
        cdf_values = betaf.cdf(x, s*alpha, s*beta)
        beta_mean = (1 - nzs/len(smoothed)) * alpha / (alpha + beta) 
        print(f"{nzs=}\t{len(smoothed)=}")

        y = nzs/len(smoothed) + cdf_values * (1 - nzs/len(smoothed))
        # y = cdf_values 
        y = y * len(smoothed)
        fig.add_scatter(
            x=x, 
            y=y, 
            mode='lines',
            name=f'Beta({alpha:.2f}, {beta:.2f}) {beta_mean=:.3f}',
            # legendgroup=legend,
            line=dict(
                dash='dot', 
                width=2,
                color=color
            )
        )
    fig.update_layout(
        width=800, height=600,
        title="cdf plots",
    ) 
    return fig

def show_betas(df_input, df_model, df_example):
    df = df_input[["model", "example_id", "pass1", "count"]].merge(df_example[["example_id", "pass1_of_ex"]], on="example_id")
    model_table = df_model[["model", "pass1"]].rename(columns={"pass1": "pass1_of_model"})
    df = df.merge(model_table, on="model")
    fig = go.Figure()
    interval_size = 0.2
    nzs = np.sum(df_example["pass1_of_ex"] == 0)
    max_example = df_example["pass1_of_ex"].max()
    min_example = df_example["pass1_of_ex"].min()
    results = []
    for i, start in enumerate(np.linspace(0, 1, 60)):
        models = model_table[(model_table["pass1_of_model"] >= start) & (model_table["pass1_of_model"] < start + interval_size)]
        # display(models)
        data_inside = df[df['model'].isin(models["model"])]
        if len(data_inside) == 0:
            continue
        data_means = data_inside.groupby("example_id").agg({"pass1": "mean", "pass1_of_ex": "mean", "count": "mean"}).reset_index()
        # Merge with original marginals to ensure same sorting
        # wsz = 1
        # smoothed = data_means[("pass1")].rolling(window=5, center=True).mean()
        data_means = data_means.sort_values(by="pass1_of_ex")
        data_means["rank_of_ex"] = np.arange(len(data_means))

        data_means = data_means.sort_values(by="pass1")
        data_means["rank"] = np.arange(len(data_means))
        # display(data_means)
        # display(data_means)
        smoothed = data_means[("pass1")].rolling(window=1, min_periods=1, center=True).mean()
        mu = data_means["pass1"].mean()
        n = len(models)
        legend = f"{start:.2f}-{start+interval_size:.2f} ({n=}, {mu=:.3f})"
        colors = px.colors.qualitative.Plotly
        color_idx = i % len(colors)  # cycle through colors
        color = colors[color_idx]
        x = np.linspace(0, 1, 100)
        # nzs = 0
        data_nzs = data_means[data_means["pass1_of_ex"] != 0]
        data_nzs = data_means
        mu = data_nzs["pass1"].mean()
        var = data_nzs["pass1"].var(ddof=1)
        alpha, beta = beta_est(mu, var)
        evar =  2 * alpha * beta/ ((alpha + beta) * (alpha + beta +1))
        results.append({"mu": mu, "alpha+beta": alpha + beta, "alpha": alpha, "beta": beta, "expected_var": evar, "expected_std": np.sqrt(evar)})
        
    return results


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


def get_sections(bres: ArenaResult, benchmark_id):
    summary = bres.summary 
    
    sections = {
        "fig_accs_and_pvalues": fig_accs_and_pvalues(benchmark_id, summary).to_html(**PLOTLY_CONFIGS),
        "fig_diff_vs_sum": fig_diff_vs_sum(benchmark_id, summary).to_html(**PLOTLY_CONFIGS),
        "fig_cov_baseline": fig_cov_baseline(benchmark_id, summary, bres.input_table).to_html(**PLOTLY_CONFIGS),
        "model_table": bres.model_table.to_html(
            index=False,
            classes="number-table",
            formatters={
                "pass1": lambda x: f"{100*x:.3g}",
                "SE(A)": lambda x: f"{100*x:.2g}",
                "SE_x(A)": lambda x: f"{100*x:.2g}",
                "SE_pred(A)": lambda x: f"{100*x:.2g}",
                "count": lambda x: f"{x:.2g}",
                "win_rate": lambda x: f"{100*x:.3g}",
                "elo": "{:.3g}".format
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

    includes_cols = ["benchmark_id", "size", "models", "SE(A)", "SE_x(A)", "SE(A-B)", "SE_x(A-B)", "corr(A,B)", "no_solve", "tau-", "sig_noise", "details"]
    percent_cols = ["no_solve", "tau-"]
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

    data_path = Path(f"{OUTPUT_PATH}/data/{benchmark_id}/")
    os.makedirs(data_path, exist_ok=True)
    ares.input_table.to_csv(data_path / "input_table.csv")
    ares.model_table.to_csv(data_path / "model_table.csv")
    ares.example_table.to_csv(data_path / "example_table.csv")
    ares.summary.to_csv(data_path / "summary.csv")
