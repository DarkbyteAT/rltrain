import json
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import seaborn as sns
import typer
from sklearn.linear_model import LinearRegression


def load_trials_data(policy_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    "Loads a ``DataFrame`` over all trials, for the given path."

    def _add_id_columns(df: pd.DataFrame, id_cols: tuple) -> pd.DataFrame:
        df["ID"] = id_cols[0]
        df["Environment"] = id_cols[1]
        df["Policy"] = id_cols[2]
        df["Algorithm"] = id_cols[3]

        return df

    def _add_cum_columns(df: pd.DataFrame) -> pd.DataFrame:
        df.set_index(["episode"], inplace=True)
        df["Timestep"] = df["length"].cumsum()
        df.reset_index(inplace=True)

        return df

    def _resample_columns(df: pd.DataFrame, interval: int = 500, max_steps: int = 100000) -> pd.DataFrame:
        id = str(df["ID"].unique().squeeze())
        env_name = str(df["Environment"].unique().squeeze())
        policy_name = str(df["Policy"].unique().squeeze())
        algo_type = str(df["Algorithm"].unique().squeeze())

        print(f"resampling {id=}, {env_name=}, {policy_name=}, {algo_type=}")

        timesteps = np.arange(max_steps + interval, step=interval)
        episodes = np.zeros(timesteps.shape, dtype=int)
        returns = np.zeros(timesteps.shape)
        run_returns = np.zeros_like(returns)

        for i, step in enumerate(timesteps[1:], start=1):
            sub_df = df.loc[((step - interval) < df["Timestep"]) & (df["Timestep"] <= step)]

            if sub_df.shape[0] > 0:
                episodes[i] = sub_df["episode"].max()
                returns[i] = sub_df["return"].mean()
                run_returns[i] = sub_df["running_return"].mean()

        new_df = pd.DataFrame(
            {
                "ID": [],
                "Environment": [],
                "Agent": [],
                "Policy": [],
                "Algorithm": [],
                "Timestep": [],
                "episode": [],
                "return": [],
            }
        )

        new_df["Timestep"] = timesteps
        new_df["episode"] = episodes
        new_df["return"] = returns
        new_df["running_return"] = run_returns

        new_df["ID"] = id
        new_df["Environment"] = env_name
        new_df["Policy"] = policy_name
        new_df["Algorithm"] = algo_type

        # Drop undersampled entries in resampled DataFrame
        new_df = new_df.loc[new_df["episode"] > 0]
        new_df = new_df.loc[new_df["return"] != 0]

        return new_df

    trial_paths = list(policy_path.iterdir())
    policy_name = policy_path.parts[-1]
    algo_type = "SAM" if policy_name.endswith("-SAM") else "LAMP" if policy_name.endswith("-LAMP") else "Vanilla"

    # Remove suffix from name if present
    if algo_type != "Vanilla":
        policy_name = policy_name[: policy_name.index(f"-{algo_type}")]

    env_json = json.load(open(trial_paths[0] / "config" / "env.json"))
    env_name = env_json["id"]

    # Load data, add columns, then combine
    results_dfs = [pd.read_csv(p / "metrics.csv") for p in trial_paths if p.is_dir()]
    id_dfs = [_add_id_columns(df, (i, env_name, policy_name, algo_type)) for i, df in enumerate(results_dfs)]
    cum_dfs = [_add_cum_columns(df) for df in id_dfs]
    resampled_dfs = [_resample_columns(df) for df in cum_dfs]

    return pd.concat(cum_dfs), pd.concat(resampled_dfs)


def calculate_metrics(trials_df: pd.DataFrame) -> dict[str, any]:
    cum_returns = [df["return"].sum() for _, df in trials_df.groupby(["ID"])]
    avg_cum_return = np.mean(cum_returns)
    std_cum_return = np.std(cum_returns)

    max_returns = [df["return"].max() for _, df in trials_df.groupby(["ID"])]
    avg_max_return = np.max(max_returns)

    # Linear-regression to find best-fit for cum-return vs. time curve
    X = trials_df["Timestep"].values.reshape(-1, 1)
    Y = trials_df["return"].values.reshape(-1, 1)

    lin_regr = LinearRegression(n_jobs=-1)
    lin_regr.fit(X, Y)
    R_sq = lin_regr.score(X, Y)
    M = float(lin_regr.coef_.squeeze())
    C = float(lin_regr.intercept_.squeeze())

    return {
        "Environment": str(trials_df["Environment"].unique().squeeze()),
        "Policy": str(trials_df["Policy"].unique().squeeze()),
        "Algorithm": str(trials_df["Algorithm"].unique().squeeze()),
        "Cumulative Return": rf"{avg_cum_return} $\pm$ {std_cum_return}",
        "Max Return": avg_max_return,
        "R^2": R_sq,
        "M": M,
        "C": C,
        "Slope": f"$y = {M}x {'+' if C >= 0 else '-'} {str(C).replace('-', '')}$",
    }


def format_table(latex_str: str) -> str:
    return (
        latex_str.replace("Advantage Actor-Critic", r"\gls{a2c}")
        .replace("Proximal Policy Optimisation", r"\gls{ppo}")
        .replace("SAM", r"\gls{sam}")
        .replace("LAMP", r"\gls{lamp}")
    )


app = typer.Typer(add_completion=False)


@app.command()
def main(
    in_path: Annotated[
        Path,
        typer.Argument(
            help="Path to results directory containing trial data.",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
        ),
    ],
    out_path: Annotated[Path, typer.Argument(help="Path to output directory for metrics and tables.")],
) -> None:
    """Aggregate trial results and produce metrics tables."""
    sns.set_style("darkgrid")
    sns.set_context("paper")

    # Load all CSV files into DataFrame
    env_paths = [p for p in in_path.iterdir() if p.is_dir()]
    policy_paths = [p for env_path in env_paths for p in env_path.iterdir() if p.is_dir()]
    loaded_dfs = [load_trials_data(p) for p in policy_paths if p.is_dir()]
    if not loaded_dfs:
        typer.echo(f"No trial data found in {in_path}", err=True)
        raise typer.Exit(1)
    full_df = pd.concat([dfs[0] for dfs in loaded_dfs])

    # Compute metrics for each environment-policy-algorithm triple
    metrics_df = pd.DataFrame(
        [
            calculate_metrics(algo_df)
            for _, env_df in full_df.groupby("Environment")
            for _, policy_df in env_df.groupby("Policy")
            for _, algo_df in policy_df.groupby("Algorithm")
        ]
    )

    metrics_df.set_index(["Environment", "Policy", "Algorithm"], inplace=True)
    full_df.set_index(["Environment", "Policy", "Algorithm"], inplace=True)

    out_path.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(out_path / "metrics.csv")
    full_df.to_csv(out_path / "full_results.csv")

    print(metrics_df["Cumulative Return"])
    print(metrics_df["Max Return"])
    print(metrics_df[["Slope", "R^2"]])

    cum_table = metrics_df["Cumulative Return"].to_latex(caption="Cumulative Return", longtable=True)
    max_table = metrics_df["Max Return"].to_latex(caption="Maximum Episodic Return", longtable=True)
    slope_table = metrics_df["M"].to_latex(caption="Slope", longtable=True)
    intercept_table = metrics_df["C"].to_latex(caption="Intercept", longtable=True)
    rsq_table = metrics_df["R^2"].to_latex(caption="$R^2$", longtable=True)

    (out_path / "cum_table.tex").write_text(format_table(cum_table))
    (out_path / "max_table.tex").write_text(format_table(max_table))
    (out_path / "slope_table.tex").write_text(format_table(slope_table))
    (out_path / "intercept_table.tex").write_text(format_table(intercept_table))
    (out_path / "rsq_table.tex").write_text(format_table(rsq_table))


if __name__ == "__main__":
    app()
