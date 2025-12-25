import logging

from jinja2 import Template
import numpy as np
import pandas as pd
import plotly.express as px

from arena import ArenaResult

logger = logging.getLogger(__name__)

def get_anchor(benchmark_id: str, example_id: str):
    def get_link():
        if benchmark_id in ["humaneval", "humaneval+", "mbpp", "mbpp+"]:
            if "/" in example_id:
                dir, id = example_id.split("/") # expecting HumanEval/93 and Mbpp/622 etc
            else:
                return ""
            return f"https://crux-eval.github.io/eval-arena/evalplus/{dir}/{id}.html"
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

    df[yid] = df[yid].astype(str).str[:20]
    fig = px.scatter(df, y=yid, x=xid, color="pass1",
                     opacity=0.75,
                     color_continuous_scale=emp_zero_scale,
                     hover_data=["pass1", "pass1_of_ex", "pass1_of_model", "model", "example_id", "count"])
            
    fig.update_xaxes(autorange="reversed")

    
    fig.update_traces(marker={"symbol": "square"})

    bid = set(result["benchmark_id"]).pop()
    fig.update_layout(
            width=900, height=1200,
            xaxis = dict(side ="top"),
            title = bid,
        )
    return fig

def get_example_level_results(benchmark_id, ares: ArenaResult):
    all_stats = ares.model_table 
    ex_table = ares.example_table
    ex_table["example_link"] = ex_table["example_id"].apply(lambda x: get_anchor(benchmark_id, x))

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

def gen_example_report(benchmark_id: str, ares: ArenaResult, OUTPUT_PATH):
    outputs = get_example_level_results(benchmark_id, ares)
    template_path = r"templates/template_example.html"
    output_path = rf"{OUTPUT_PATH}/ex_{benchmark_id}.html"
    with open(output_path, "w", encoding="utf-8") as output_file:
        with open(template_path) as template_file:
            j2_template = Template(template_file.read())
            output_file.write(j2_template.render({"benchmark_id": benchmark_id, "outputs": outputs}))

    plotly_configs = dict(full_html=False, include_plotlyjs="cdn")
    with open(f"{OUTPUT_PATH}/ex_v_model_{benchmark_id}.html", "wt") as f:
        f.write(outputs["fig_example_vs_model"].to_html(**plotly_configs))

    with open(f"{OUTPUT_PATH}/ex_v_model_acc_{benchmark_id}.html", "wt") as f:
        f.write(outputs["fig_example_vs_model_acc"].to_html(**plotly_configs)) 
