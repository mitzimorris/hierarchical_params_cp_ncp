"""Plotting helpers for repeated fits of the baseball models."""

from collections.abc import Mapping, Sequence

import pandas as pd
import plotnine as p9


def summarize_baseball_estimate_runs(
    estimates: pd.DataFrame,
    median_r_hat_limit: float = 1.01,
    maximum_r_hat_limit: float = 1.03,
) -> pd.DataFrame:
    """Summarize posterior uncertainty and variability across repeated fits."""
    required_columns = {
        "Parameterization",
        "Run",
        "Variable",
        "Mean",
        "q05",
        "q50",
        "q95",
        "Median R_hat",
        "Maximum R_hat",
    }
    missing_columns = required_columns.difference(estimates.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Baseball estimate data are missing columns: {missing}")

    estimates = estimates.copy()
    estimates["diagnostics_ok"] = estimates["Median R_hat"].le(median_r_hat_limit) & estimates[
        "Maximum R_hat"
    ].lt(maximum_r_hat_limit)
    summary = (
        estimates.groupby(["Parameterization", "Variable"], sort=False)
        .agg(
            runs=("Run", "nunique"),
            runs_passing_diagnostics=("diagnostics_ok", "sum"),
            within_q05=("q05", "mean"),
            within_q95=("q95", "mean"),
            estimate=("q50", "median"),
            between_q05=("q50", lambda values: values.quantile(0.05)),
            between_q95=("q50", lambda values: values.quantile(0.95)),
        )
        .reset_index()
    )
    summary["diagnostic_pass_rate"] = summary["runs_passing_diagnostics"] / summary["runs"]
    summary["diagnostic_opacity"] = 0.05 + 0.95 * summary["diagnostic_pass_rate"].pow(2)
    return summary


def baseball_population_parameter_table(
    estimates: pd.DataFrame,
    run: int = 1,
) -> pd.DataFrame:
    """Format ``mu`` and ``sigma`` summaries from one fit of each model."""
    required_columns = {
        "Parameterization",
        "Run",
        "Variable",
        "q05",
        "q50",
        "q95",
    }
    missing_columns = required_columns.difference(estimates.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Baseball estimate data are missing columns: {missing}")

    table = estimates[
        estimates["Run"].eq(run) & estimates["Variable"].isin(["mu", "sigma"])
    ].copy()
    if table.empty:
        raise ValueError(f"No population parameter estimates found for run {run}.")
    if table.duplicated(["Parameterization", "Variable"]).any():
        raise ValueError(
            "Each run must contain one estimate per parameterization and parameter."
        )

    table["Posterior q05–q95"] = table.apply(
        lambda row: f"[{row['q05']:.2f}, {row['q95']:.2f}]",
        axis=1,
    )
    return table.rename(
        columns={
            "Variable": "Parameter",
            "Mean": "Mean estimate",
            "q50": "Median estimate",
        }
    )[
        [
            "Parameterization",
            "Parameter",
            "Mean estimate",
            "Median estimate",
            "Posterior q05–q95",
        ]
    ]


def plot_baseball_player_estimates(
    estimates: pd.DataFrame,
    players: pd.DataFrame,
    run: int = 1,
    title: str = "Player ability estimates from one fit",
    subtitle: str | None = None,
    figure_size: tuple[float, float] = (12.0, 7.0),
) -> p9.ggplot:
    """Compare player posterior medians and intervals from one fit of each model."""
    required_estimate_columns = {
        "Parameterization",
        "Run",
        "Variable",
        "q05",
        "q50",
        "q95",
    }
    missing_estimate_columns = required_estimate_columns.difference(estimates.columns)
    if missing_estimate_columns:
        missing = ", ".join(sorted(missing_estimate_columns))
        raise ValueError(f"Baseball estimate data are missing columns: {missing}")

    player_estimates = estimates[
        estimates["Run"].eq(run) & estimates["Variable"].str.startswith("alpha[")
    ].copy()
    if player_estimates.empty:
        raise ValueError(f"No player estimates found for run {run}.")

    player_estimates["player_id"] = (
        player_estimates["Variable"].str.extract(r"alpha\[(\d+)\]")[0].astype(int)
    )


def baseball_posterior_geometry_frame(
    draws: Mapping[str, pd.DataFrame],
    players: pd.DataFrame,
    player_ids: Sequence[int] = (18, 5, 7),
) -> pd.DataFrame:
    """Prepare sampled local coordinates for selected baseball players."""
    required_parameterizations = {"Centered", "Non-centered"}
    missing_parameterizations = required_parameterizations.difference(draws)
    if missing_parameterizations:
        missing = ", ".join(sorted(missing_parameterizations))
        raise ValueError(f"Posterior draws are missing parameterizations: {missing}")

    required_player_columns = {"player_id", "player", "season_at_bats"}
    missing_player_columns = required_player_columns.difference(players.columns)
    if missing_player_columns:
        missing = ", ".join(sorted(missing_player_columns))
        raise ValueError(f"Player metadata are missing columns: {missing}")

    player_metadata = players.set_index("player_id")
    missing_player_ids = set(player_ids).difference(player_metadata.index)
    if missing_player_ids:
        missing = ", ".join(str(value) for value in sorted(missing_player_ids))
        raise ValueError(f"Player metadata are missing player IDs: {missing}")

    frames: list[pd.DataFrame] = []
    facet_order: list[str] = []
    for parameterization in ("Centered", "Non-centered"):
        parameter_draws = draws[parameterization]
        required_draw_columns = {
            "mu",
            "sigma",
            *(f"alpha[{player_id}]" for player_id in player_ids),
        }
        missing_draw_columns = required_draw_columns.difference(parameter_draws.columns)
        if missing_draw_columns:
            missing = ", ".join(sorted(missing_draw_columns))
            raise ValueError(
                f"{parameterization} posterior draws are missing columns: {missing}"
            )

        for player_id in player_ids:
            metadata = player_metadata.loc[player_id]
            coordinate_name = (
                f"alpha[{player_id}]"
                if parameterization == "Centered"
                else f"alpha_std[{player_id}]"
            )
            facet = (
                f"{parameterization}: {coordinate_name}\n"
                f"{metadata['player']} ({metadata['season_at_bats']} at-bats)"
            )
            facet_order.append(facet)
            local_coordinate = parameter_draws[f"alpha[{player_id}]"].copy()
            if parameterization == "Non-centered":
                local_coordinate = (
                    local_coordinate - parameter_draws["mu"]
                ) / parameter_draws["sigma"]

            frames.append(
                pd.DataFrame(
                    {
                        "Parameterization": parameterization,
                        "player_id": player_id,
                        "local_coordinate": local_coordinate,
                        "sigma": parameter_draws["sigma"],
                        "facet": facet,
                    }
                )
            )

    geometry = pd.concat(frames, ignore_index=True)
    geometry["facet"] = pd.Categorical(
        geometry["facet"],
        categories=facet_order,
        ordered=True,
    )
    return geometry


def plot_baseball_posterior_geometry(
    draws: Mapping[str, pd.DataFrame],
    players: pd.DataFrame,
    player_ids: Sequence[int] = (18, 5, 7),
    title: str = "Posterior geometry across players",
    subtitle: str = (
        "Centered: sigma versus alpha[j]; non-centered: sigma versus alpha_std[j]"
    ),
    figure_size: tuple[float, float] = (12.0, 7.0),
) -> p9.ggplot:
    """Plot population scale against selected local sampling coordinates."""
    geometry = baseball_posterior_geometry_frame(
        draws,
        players,
        player_ids=player_ids,
    )
    return (
        p9.ggplot(
            geometry,
            p9.aes(x="local_coordinate", y="sigma", color="Parameterization"),
        )
        + p9.geom_point(size=0.5, alpha=0.18)
        + p9.facet_wrap("facet", ncol=3, scales="free_x")
        + p9.scale_color_manual(
            values={"Centered": "#0072B2", "Non-centered": "#D55E00"}
        )
        + p9.labs(
            x="Local sampling coordinate",
            y="Population standard deviation, sigma",
            title=title,
            subtitle=subtitle,
        )
        + p9.theme_minimal()
        + p9.theme(
            figure_size=figure_size,
            legend_position="none",
            panel_grid_minor=p9.element_blank(),
            strip_text=p9.element_text(size=9),
        )
    )

    required_player_columns = {"player_id", "player", "season_at_bats"}
    missing_player_columns = required_player_columns.difference(players.columns)
    if missing_player_columns:
        missing = ", ".join(sorted(missing_player_columns))
        raise ValueError(f"Player metadata are missing columns: {missing}")

    ordered_players = players.sort_values("season_at_bats").reset_index(drop=True).copy()
    ordered_players["player_position"] = range(1, len(ordered_players) + 1)
    ordered_players["player_label"] = ordered_players.apply(
        lambda row: f"{row['player_id']}  {row['player']}",
        axis=1,
    )
    player_estimates = player_estimates.merge(
        ordered_players,
        on="player_id",
        how="left",
        validate="many_to_one",
    )
    if player_estimates[["player", "season_at_bats"]].isna().any().any():
        raise ValueError("Player metadata do not cover all player estimates.")
    if player_estimates.duplicated(["Parameterization", "player_id"]).any():
        raise ValueError(
            "Each run must contain one estimate per parameterization and player."
        )

    parameterization_offset = {"Centered": -0.18, "Non-centered": 0.18}
    offsets = player_estimates["Parameterization"].map(parameterization_offset)
    if offsets.isna().any():
        unknown = ", ".join(
            sorted(player_estimates.loc[offsets.isna(), "Parameterization"].unique())
        )
        raise ValueError(f"Unknown parameterization labels: {unknown}")
    player_estimates["plot_x"] = player_estimates["player_position"] + offsets
    if subtitle is None:
        subtitle = f"Run {run}; players ordered by season at-bats"

    return (
        p9.ggplot(
            player_estimates,
            p9.aes(x="plot_x", y="q50", color="Parameterization"),
        )
        + p9.geom_linerange(p9.aes(ymin="q05", ymax="q95"), size=0.8)
        + p9.geom_point(size=2.4)
        + p9.scale_x_continuous(
            breaks=ordered_players["player_position"].tolist(),
            labels=ordered_players["player_label"].tolist(),
        )
        + p9.scale_color_manual(
            values={"Centered": "#0072B2", "Non-centered": "#D55E00"}
        )
        + p9.labs(
            x="",
            y="Player ability, alpha (log odds)",
            color="Parameterization",
            title=title,
            subtitle=subtitle,
        )
        + p9.theme_minimal()
        + p9.theme(
            figure_size=figure_size,
            legend_position="top",
            panel_grid_minor=p9.element_blank(),
            axis_text_x=p9.element_text(
                rotation=50,
                ha="right",
                va="top",
                size=8,
            )
        )
    )
