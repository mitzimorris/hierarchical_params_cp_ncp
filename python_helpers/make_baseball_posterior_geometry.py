"""Fit run 1 of both baseball models and save their posterior geometry plot."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from cmdstanpy import CmdStanModel, disable_logging

from .baseball_fits import plot_baseball_posterior_geometry
from .run_baseball_fits_informed_init import (
    DATA_FILE,
    MODEL_SPECS,
    N_CHAINS,
    N_SAMPLING,
    N_WARMUP,
    PROJECT_ROOT,
    make_data_informed_inits,
)

RUN_1_SEED = 12345
PLAYER_IDS = (18, 14, 7)
PLAYER_FILE = PROJECT_ROOT / "data" / "baseball_players.csv"
OUTPUT_FILE = PROJECT_ROOT / "plots" / "baseball_posterior_geom.png"


def make_baseball_posterior_geometry_plot():
    """Reproduce run 1 and return its centered/non-centered geometry plot."""
    with DATA_FILE.open() as stream:
        data = json.load(stream)
    players = pd.read_csv(PLAYER_FILE)
    initial_values = make_data_informed_inits(data, RUN_1_SEED)
    posterior_draws: dict[str, pd.DataFrame] = {}

    with disable_logging():
        for parameterization, stan_file in MODEL_SPECS:
            model = CmdStanModel(stan_file=str(stan_file))
            with TemporaryDirectory() as output_dir:
                fit = model.sample(
                    data=data,
                    inits=initial_values,
                    seed=RUN_1_SEED,
                    chains=N_CHAINS,
                    parallel_chains=N_CHAINS,
                    iter_warmup=N_WARMUP,
                    iter_sampling=N_SAMPLING,
                    show_progress=False,
                    output_dir=output_dir,
                )
                posterior_draws[parameterization] = fit.draws_pd(
                    vars=["mu", "sigma", "alpha"]
                )

    return plot_baseball_posterior_geometry(
        posterior_draws,
        players,
        player_ids=PLAYER_IDS,
        subtitle=(
            "Run 1 with data-informed initialization; "
            "players ordered from fewest to most at-bats"
        ),
    )


def main() -> None:
    """Reproduce run 1, plot the sampled coordinates, and save the image."""
    plot = make_baseball_posterior_geometry_plot()
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    plot.save(OUTPUT_FILE, dpi=180, verbose=False)
    print(f"Saved {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
