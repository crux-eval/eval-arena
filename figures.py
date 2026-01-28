import logging

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats

from utils import pass_at_k

logger = logging.getLogger(__name__)


def _beta_est(mean, var):
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
            alpha, beta = _beta_est(mu, var)
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


def fig_pass_at_k(bmname: str, df_input: pd.DataFrame) -> go.Figure:
    """
    for each model, find the maximum k that can be used, then compute pass_at_k
    """
    fig = go.Figure()
    def get_log_k_values(max_k: int):
        k_values = []
        k = 1
        step = 1
        while k < max_k:
            k_values.append(k)
            k += step
            if k >= step * 10:
                step *= 10
        k_values.append(max_k)
        return k_values

    def pass_at_ks(g: pd.Series):
        kA = g["count"].to_numpy()
        if len(set(kA)) == 1:
            kA = kA[0]
        else:
            kA = np.min(kA)
        pass_ks = []
        N = len(g)
        for k in get_log_k_values(kA):
            pass_at_ks = [pass_at_k(n, c, k) for n, c in zip(g["count"], g["correct"])]
            pass_ks.append({
                "k": k,
                "pass_at_k": np.mean(pass_at_ks),
                "pass_at_k_stderr": 1/np.sqrt(N) * np.std(pass_at_ks),
            })
        return pd.DataFrame(pass_ks)
    model_stats = df_input[["model", "correct", "count"]].groupby("model").apply(pass_at_ks).reset_index()
    fig = px.line(
        model_stats,
        x="k",
        y="pass_at_k",
        # error_y="pass_at_k_stderr",
        color="model",
        line_dash="model",  # use different dash styles per model
        log_x=True
    )
    fig.update_traces(line=dict(width=2))
    fig.update_layout(
        width=800, height=800, title=bmname
    )
    return fig


def fig_example_vs_model(result, all_stats, ex_table, use_acc_as_position=False, zero_special=False):
    df = result[["model", "example_id", "pass1", "count"]].merge(ex_table[["example_id", "pass1_of_ex"]], on="example_id")
    model_table = all_stats[["model", "pass1"]].rename(columns={"pass1": "pass1_of_model"})
    df = df.merge(model_table, on="model")
    df.sort_values(by=["pass1_of_ex", "example_id", "pass1_of_model", "model"], inplace=True)
    if not use_acc_as_position:
        yid, xid = "example_id", "model"
    else:
        yid, xid = "example_id", "pass1_of_model"

    if zero_special:
        emp_zero_scale = [
            [0.0, "black"],
            [1e-9, "red"],
            [0.25, "yellow"],
            [1, "green"],
        ]
    else:
        emp_zero_scale = [
            [0, "red"],
            [0.25, "yellow"],
            [1, "green"],
        ]

    # df[yid] = df[yid].astype(str).str[:20]
    fig = px.scatter(df, y=yid, x=xid, color="pass1",
                     opacity=0.75,
                     color_continuous_scale=emp_zero_scale,
                     hover_data=["pass1", "pass1_of_ex", "pass1_of_model", "model", "example_id", "count"])

    fig.update_xaxes(autorange="reversed")
    show_yaxis = all(len(str(label)) <= 20 for label in df[yid].unique())
    if not show_yaxis:
        fig.update_yaxes(showticklabels=False)


    fig.update_traces(marker={"symbol": "square"})

    bid = set(result["benchmark_id"]).pop()
    fig.update_layout(
            width=900, height=1200,
            xaxis = dict(side ="top"),
            title = bid,
        )
    return fig
