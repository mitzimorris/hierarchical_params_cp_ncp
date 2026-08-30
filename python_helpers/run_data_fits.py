"""Fit both parameterizations 100 times to each of eight nested datasets."""

import os
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from cmdstanpy import CmdStanModel, disable_logging

N_GROUPS = 9
N_OBS = (2, 4, 16, 64, 128, 256, 2048, 32768)
N_RUNS = 100
N_CHAINS = 4
N_WARMUP = 1_000
N_SAMPLING = 1_000
DATA_SEED = 424242
SEEDS = range(12345, 12345 + N_RUNS)
ESTIMATE_VARIABLES = ("log_sigma_sq", "theta[1]")
ESTIMATE_BASE_VARIABLES = ("log_sigma_sq", "theta")
MEDIAN_R_HAT_LIMIT = 1.01
MAXIMUM_R_HAT_LIMIT = 1.03


def make_nested_datasets(
    trials: np.ndarray, n_obs_values: Sequence[int]
) -> dict[int, dict[str, object]]:
    """Create Stan datasets from cumulative success counts."""
    cumulative_successes = np.cumsum(trials, axis=0)
    return {
        n_obs: {
            "N": n_obs,
            "y": cumulative_successes[n_obs - 1].tolist(),
        }
        for n_obs in n_obs_values
    }


datagen = CmdStanModel(stan_file=os.path.join("stan", "datagen_repeated_binary_trials.stan"))
with TemporaryDirectory() as output_dir:
    simulation = datagen.sample(
        data={"N_groups": N_GROUPS, "N_obs": max(N_OBS)},
        chains=1,
        iter_warmup=0,
        iter_sampling=1,
        adapt_engaged=False,
        show_progress=False,
        seed=DATA_SEED,
        output_dir=output_dir,
    )
    trials = simulation.stan_variable("y")[0].astype(int)

datasets = make_nested_datasets(trials, N_OBS)

MODELS = (
    (
        "Centered",
        CmdStanModel(
            stan_file=os.path.join("stan", "funnel_data_cp.stan"),
            force_compile=True,
        ),
        ("log_sigma_sq", "theta[1]"),
        ("log_sigma_sq", *(f"theta[{index}]" for index in range(1, N_GROUPS + 1))),
    ),
    (
        "Non-centered",
        CmdStanModel(
            stan_file=os.path.join("stan", "funnel_data_ncp.stan"),
            force_compile=True,
        ),
        ("log_sigma_sq_std", "theta_std[1]"),
        (
            "log_sigma_sq_std",
            *(f"theta_std[{index}]" for index in range(1, N_GROUPS + 1)),
        ),
    ),
)

rows = []
estimate_rows = []

with disable_logging():
    for parameterization, model, variables, convergence_variables in MODELS:
        for n_obs in N_OBS:
            for run, seed in enumerate(SEEDS, start=1):
                with TemporaryDirectory() as output_dir:
                    fit = model.sample(
                        data=datasets[n_obs],
                        seed=seed,
                        chains=N_CHAINS,
                        parallel_chains=N_CHAINS,
                        iter_warmup=N_WARMUP,
                        iter_sampling=N_SAMPLING,
                        show_progress=False,
                        output_dir=output_dir,
                    )

                    divergent = fit.method_variables()["divergent__"]
                    n_leapfrog = fit.method_variables()["n_leapfrog__"]
                    stepsize = fit.method_variables()["stepsize__"]
                    fit_summary = fit.summary()
                    warmup_seconds = sum(chain["warmup"] for chain in fit.time)
                    sampling_seconds = sum(chain["sampling"] for chain in fit.time)
                    total_sampler_seconds = warmup_seconds + sampling_seconds
                    run_r_hat = fit_summary.loc[list(convergence_variables), "R_hat"]
                    median_r_hat = float(run_r_hat.median())
                    maximum_r_hat = float(run_r_hat.max())
                    diagnostics_ok = bool(
                        median_r_hat <= MEDIAN_R_HAT_LIMIT and maximum_r_hat < MAXIMUM_R_HAT_LIMIT
                    )

                    for variable in variables:
                        rows.append(
                            {
                                "Parameterization": parameterization,
                                "N": n_obs,
                                "Run": run,
                                "Variable": variable,
                                "Total divergences": int(divergent.sum()),
                                "Chains with a divergence": int(divergent.any(axis=0).sum()),
                                "Stepsize": float(stepsize.mean()),
                                "Mean leapfrog steps": float(n_leapfrog.mean()),
                                "Warmup seconds": warmup_seconds,
                                "Sampling seconds": sampling_seconds,
                                "Total sampler seconds": total_sampler_seconds,
                                "ESS_bulk/s": fit_summary.loc[variable, "ESS_bulk/s"],
                                "R_hat": fit_summary.loc[variable, "R_hat"],
                            }
                        )

                    # CmdStanPy expands the base vector name ``theta`` into
                    # theta[1], ..., theta[9]; indexed elements are not valid
                    # values for its ``vars`` selector.
                    estimate_draws = fit.draws_pd(vars=list(ESTIMATE_BASE_VARIABLES))
                    for variable in ESTIMATE_VARIABLES:
                        estimates = estimate_draws[variable]
                        estimate_rows.append(
                            {
                                "Parameterization": parameterization,
                                "N": n_obs,
                                "Run": run,
                                "Variable": variable,
                                "Median R_hat": median_r_hat,
                                "Maximum R_hat": maximum_r_hat,
                                "Diagnostics OK": diagnostics_ok,
                                "ESS_bulk/s": fit_summary.loc[variable, "ESS_bulk/s"],
                                "Mean": estimates.mean(),
                                "q05": estimates.quantile(0.05),
                                "q50": estimates.quantile(0.50),
                                "q95": estimates.quantile(0.95),
                            }
                        )

                if run % 10 == 0:
                    print(f"Finished {parameterization}, N={n_obs}, run {run}/{N_RUNS}")

fits = pd.DataFrame(rows)
estimate_runs = pd.DataFrame(estimate_rows)

summary = (
    fits.groupby(["Parameterization", "N", "Variable"], sort=False)
    .agg(
        runs=("Run", "size"),
        runs_with_divergence=("Total divergences", lambda values: int((values > 0).sum())),
        average_divergences=("Total divergences", "mean"),
        average_affected_chains=("Chains with a divergence", "mean"),
        average_stepsize=("Stepsize", "mean"),
        average_n_leapfrog=("Mean leapfrog steps", "mean"),
        average_warmup_seconds=("Warmup seconds", "mean"),
        average_sampling_seconds=("Sampling seconds", "mean"),
        median_total_sampler_seconds=("Total sampler seconds", "median"),
        median_ess_per_second=("ESS_bulk/s", "median"),
        minimum_ess_per_second=("ESS_bulk/s", "min"),
        median_r_hat=("R_hat", "median"),
        maximum_r_hat=("R_hat", "max"),
        runs_with_high_r_hat=("R_hat", lambda values: int((values > 1.01).sum())),
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
            "median_total_sampler_seconds": "Median total sampler seconds",
            "median_ess_per_second": "Median ESS_bulk/s",
            "minimum_ess_per_second": "Minimum ESS_bulk/s",
            "median_r_hat": "Median R_hat",
            "maximum_r_hat": "Maximum R_hat",
            "runs_with_high_r_hat": "Runs with R_hat > 1.01",
        }
    )
)


results_dir = Path("results")
results_dir.mkdir(exist_ok=True)
summary.to_csv(results_dir / "cp_ncp_data_fits.csv", index=False)
estimate_runs.to_csv(results_dir / "cp_ncp_data_estimate_runs.csv", index=False)
print(summary.round(2).to_string(index=False))
