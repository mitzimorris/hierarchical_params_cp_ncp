"""Fit the centered and non-centered prior models 100 times each."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from cmdstanpy import CmdStanModel, disable_logging

N_RUNS = 100
N_CHAINS = 4
N_WARMUP = 1_000
N_SAMPLING = 1_000
SEEDS = range(42, 42 + N_RUNS)

MODELS = (
    (
        "Centered",
        CmdStanModel(stan_file=os.path.join("stan", "funnel_priors_cp.stan")),
        ("y", "x[1]"),
    ),
    (
        "Non-centered",
        CmdStanModel(stan_file=os.path.join("stan", "funnel_priors_ncp.stan")),
        ("y", "y_std", "x_std[1]"),
    ),
)

rows = []

with disable_logging():
    for parameterization, model, variables in MODELS:
        for run, seed in enumerate(SEEDS, start=1):
            with TemporaryDirectory() as output_dir:
                fit = model.sample(
                    seed=seed,
                    chains=N_CHAINS,
                    parallel_chains=N_CHAINS,
                    iter_warmup=N_WARMUP,
                    iter_sampling=N_SAMPLING,
                    show_progress=False,
                    output_dir=output_dir,
                )

                divergent = fit.method_variables()["divergent__"]
                fit_summary = fit.summary()
                warmup_seconds = sum(chain["warmup"] for chain in fit.time)
                sampling_seconds = sum(chain["sampling"] for chain in fit.time)

                for variable in variables:
                    rows.append(
                        {
                            "Parameterization": parameterization,
                            "Run": run,
                            "Variable": variable,
                            "Total divergences": int(divergent.sum()),
                            "Chains with a divergence": int(divergent.any(axis=0).sum()),
                            "Warmup seconds": warmup_seconds,
                            "Sampling seconds": sampling_seconds,
                            "ESS_bulk/s": fit_summary.loc[variable, "ESS_bulk/s"],
                            "R_hat": fit_summary.loc[variable, "R_hat"],
                        }
                    )

            if run % 10 == 0:
                print(f"Finished {parameterization} run {run}/{N_RUNS}")

fits = pd.DataFrame(rows)

summary = (
    fits.groupby(["Parameterization", "Variable"], sort=False)
    .agg(
        runs=("Run", "size"),
        runs_with_divergence=("Total divergences", lambda values: int((values > 0).sum())),
        average_divergences=("Total divergences", "mean"),
        average_affected_chains=("Chains with a divergence", "mean"),
        average_warmup_seconds=("Warmup seconds", "mean"),
        average_sampling_seconds=("Sampling seconds", "mean"),
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

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)
summary.to_csv(results_dir / "cp_ncp_prior_fits.csv", index=False)
print(summary.round(2).to_string(index=False))
