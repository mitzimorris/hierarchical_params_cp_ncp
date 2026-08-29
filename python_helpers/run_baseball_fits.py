"""Fit the centered and non-centered baseball models 100 times each."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from cmdstanpy import CmdStanModel, disable_logging

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "baseball_full_season.json"
RESULTS_FILE = PROJECT_ROOT / "results" / "cp_ncp_baseball_fits.csv"
ESTIMATE_RUNS_FILE = PROJECT_ROOT / "results" / "cp_ncp_baseball_estimate_runs.csv"
ESTIMATE_SUMMARY_FILE = PROJECT_ROOT / "results" / "cp_ncp_baseball_estimates.csv"

N_RUNS = 100
N_CHAINS = 4
N_WARMUP = 1_000
N_SAMPLING = 1_000
SEEDS = range(12345, 12345 + N_RUNS)
MEDIAN_R_HAT_LIMIT = 1.01
MAXIMUM_R_HAT_LIMIT = 1.03

VARIABLES = ("mu", "sigma", "alpha[1]")
N_PLAYERS = 18
ESTIMATE_VARIABLES = (
    "mu",
    "sigma",
    *(f"alpha[{index}]" for index in range(1, N_PLAYERS + 1)),
)
ESTIMATE_BASE_VARIABLES = ("mu", "sigma", "alpha")
MODEL_SPECS = (
    (
        "Centered",
        PROJECT_ROOT / "stan" / "baseball_hierarchical_logit_cp.stan",
    ),
    (
        "Non-centered",
        PROJECT_ROOT / "stan" / "baseball_hierarchical_logit_ncp.stan",
    ),
)


def summarize_runs(fits: pd.DataFrame) -> pd.DataFrame:
    """Aggregate run- and parameter-level diagnostics for the output table."""
    return (
        fits.groupby(["Parameterization", "Variable"], sort=False)
        .agg(
            runs=("Run", "size"),
            runs_with_divergence=(
                "Total divergences",
                lambda values: int((values > 0).sum()),
            ),
            average_divergences=("Total divergences", "mean"),
            average_affected_chains=("Chains with a divergence", "mean"),
            average_stepsize=("Stepsize", "mean"),
            average_n_leapfrog=("Mean leapfrog steps", "mean"),
            average_warmup_seconds=("Warmup seconds", "mean"),
            average_sampling_seconds=("Sampling seconds", "mean"),
            median_ess_per_second=("ESS_bulk/s", "median"),
            minimum_ess_per_second=("ESS_bulk/s", "min"),
            median_r_hat=("R_hat", "median"),
            maximum_r_hat=("R_hat", "max"),
            runs_with_high_r_hat=(
                "R_hat",
                lambda values: int((values > 1.01).sum()),
            ),
        )
        .reset_index()
        .rename(
            columns={
                "runs": "Runs",
                "runs_with_divergence": "Runs with a divergence",
                "average_divergences": "Average total divergences",
                "average_affected_chains": "Average affected chains",
                "average_stepsize": "Stepsize",
                "average_n_leapfrog": "Average leapfrog steps",
                "average_warmup_seconds": "Average warmup seconds",
                "average_sampling_seconds": "Average sampling seconds",
                "median_ess_per_second": "Median ESS_bulk/s",
                "minimum_ess_per_second": "Minimum ESS_bulk/s",
                "median_r_hat": "Median R_hat",
                "maximum_r_hat": "Maximum R_hat",
                "runs_with_high_r_hat": "Runs with R_hat > 1.01",
            }
        )
    )


def summarize_estimates(estimates: pd.DataFrame) -> pd.DataFrame:
    """Aggregate run-level posterior summaries for plotting."""
    return (
        estimates.groupby(["Parameterization", "Variable"], sort=False)
        .agg(
            runs=("Run", "nunique"),
            runs_passing_diagnostics=("Diagnostics OK", "sum"),
            mean=("Mean", "mean"),
            q05=("q05", "mean"),
            q50=("q50", "mean"),
            q95=("q95", "mean"),
            median_r_hat=("Median R_hat", "median"),
            maximum_r_hat=("Maximum R_hat", "max"),
        )
        .reset_index()
        .rename(
            columns={
                "runs": "Runs",
                "runs_passing_diagnostics": "Runs passing diagnostics",
                "mean": "Mean",
                "median_r_hat": "Median R_hat",
                "maximum_r_hat": "Maximum R_hat",
            }
        )
    )


def main() -> None:
    """Compile both models, run the benchmark, and save its summary."""
    models = tuple(
        (parameterization, CmdStanModel(stan_file=str(stan_file)))
        for parameterization, stan_file in MODEL_SPECS
    )
    rows: list[dict[str, object]] = []
    estimate_rows: list[dict[str, object]] = []

    with disable_logging():
        for parameterization, model in models:
            for run, seed in enumerate(SEEDS, start=1):
                with TemporaryDirectory() as output_dir:
                    fit = model.sample(
                        data=str(DATA_FILE),
                        seed=seed,
                        chains=N_CHAINS,
                        parallel_chains=N_CHAINS,
                        iter_warmup=N_WARMUP,
                        iter_sampling=N_SAMPLING,
                        show_progress=False,
                        output_dir=output_dir,
                    )

                    method_variables = fit.method_variables()
                    divergent = method_variables["divergent__"]
                    n_leapfrog = method_variables["n_leapfrog__"]
                    stepsize = method_variables["stepsize__"]
                    fit_summary = fit.summary()
                    warmup_seconds = sum(chain["warmup"] for chain in fit.time)
                    sampling_seconds = sum(chain["sampling"] for chain in fit.time)
                    run_r_hat = fit_summary.loc[list(ESTIMATE_VARIABLES), "R_hat"]
                    median_r_hat = float(run_r_hat.median())
                    maximum_r_hat = float(run_r_hat.max())
                    diagnostics_ok = bool(
                        median_r_hat <= MEDIAN_R_HAT_LIMIT and maximum_r_hat < MAXIMUM_R_HAT_LIMIT
                    )

                    for variable in VARIABLES:
                        rows.append(
                            {
                                "Parameterization": parameterization,
                                "Run": run,
                                "Variable": variable,
                                "Total divergences": int(divergent.sum()),
                                "Chains with a divergence": int(divergent.any(axis=0).sum()),
                                "Stepsize": float(stepsize.mean()),
                                "Mean leapfrog steps": float(n_leapfrog.mean()),
                                "Warmup seconds": warmup_seconds,
                                "Sampling seconds": sampling_seconds,
                                "ESS_bulk/s": fit_summary.loc[variable, "ESS_bulk/s"],
                                "R_hat": fit_summary.loc[variable, "R_hat"],
                            }
                        )

                    estimate_draws = fit.draws_pd(vars=list(ESTIMATE_BASE_VARIABLES))
                    for variable in ESTIMATE_VARIABLES:
                        variable_draws = estimate_draws[variable]
                        estimate_rows.append(
                            {
                                "Parameterization": parameterization,
                                "Run": run,
                                "Variable": variable,
                                "Median R_hat": median_r_hat,
                                "Maximum R_hat": maximum_r_hat,
                                "Diagnostics OK": diagnostics_ok,
                                "Mean": variable_draws.mean(),
                                "q05": variable_draws.quantile(0.05),
                                "q50": variable_draws.quantile(0.50),
                                "q95": variable_draws.quantile(0.95),
                            }
                        )

                if run % 10 == 0:
                    print(f"Finished {parameterization} run {run}/{N_RUNS}")

    summary = summarize_runs(pd.DataFrame(rows))
    estimate_runs = pd.DataFrame(estimate_rows)
    estimate_summary = summarize_estimates(estimate_runs)
    RESULTS_FILE.parent.mkdir(exist_ok=True)
    summary.to_csv(RESULTS_FILE, index=False)
    estimate_runs.to_csv(ESTIMATE_RUNS_FILE, index=False)
    estimate_summary.to_csv(ESTIMATE_SUMMARY_FILE, index=False)
    print(summary.round(2).to_string(index=False))
    print(estimate_summary.round(3).to_string(index=False))
    print(f"Saved results to {RESULTS_FILE}")
    print(f"Saved run-level estimates to {ESTIMATE_RUNS_FILE}")
    print(f"Saved estimate summary to {ESTIMATE_SUMMARY_FILE}")


if __name__ == "__main__":
    main()
