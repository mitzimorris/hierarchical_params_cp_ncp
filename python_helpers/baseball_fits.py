"""Plotting helpers for repeated fits of the baseball models."""

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
    median_r_hat_limit: float = 1.01,
    maximum_r_hat_limit: float = 1.03,
) -> pd.DataFrame:
    """Format repeated-fit summaries for ``mu`` and ``sigma`` as a table."""
    summary = summarize_baseball_estimate_runs(
        estimates,
        median_r_hat_limit=median_r_hat_limit,
        maximum_r_hat_limit=maximum_r_hat_limit,
    )
    table = summary[summary["Variable"].isin(["mu", "sigma"])].copy()
    table["Posterior q05–q95"] = table.apply(
        lambda row: f"[{row['within_q05']:.3f}, {row['within_q95']:.3f}]",
        axis=1,
    )
    table["Run-median q05–q95"] = table.apply(
        lambda row: f"[{row['between_q05']:.3f}, {row['between_q95']:.3f}]",
        axis=1,
    )
    return table.rename(
        columns={
            "Variable": "Parameter",
            "estimate": "Median estimate",
            "diagnostic_pass_rate": "Diagnostic pass rate",
        }
    )[
        [
            "Parameterization",
            "Parameter",
            "Median estimate",
            "Posterior q05–q95",
            "Run-median q05–q95",
            "Diagnostic pass rate",
        ]
    ]


def plot_baseball_player_estimates(
    estimates: pd.DataFrame,
    players: pd.DataFrame,
    median_r_hat_limit: float = 1.01,
    maximum_r_hat_limit: float = 1.03,
    title: str = "Player ability estimates",
    subtitle: str = (
        "Players ordered by season at-bats; blue: centered; orange: non-centered\n"
        "Thin: posterior q05–q95; thick: between-run median spread; "
        "opacity: diagnostic pass rate"
    ),
    figure_size: tuple[float, float] = (12.0, 7.0),
) -> p9.ggplot:
    """Plot player log-odds ordered by the number of season at-bats."""
    summary = summarize_baseball_estimate_runs(
        estimates,
        median_r_hat_limit=median_r_hat_limit,
        maximum_r_hat_limit=maximum_r_hat_limit,
    )
    player_estimates = summary[summary["Variable"].str.startswith("alpha[")].copy()
    player_estimates["player_id"] = (
        player_estimates["Variable"].str.extract(r"alpha\[(\d+)\]")[0].astype(int)
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
    parameterization_offset = {"Centered": -0.18, "Non-centered": 0.18}
    player_estimates["plot_x"] = player_estimates["player_position"] + player_estimates[
        "Parameterization"
    ].map(parameterization_offset)

    intervals = _interval_frame(player_estimates, x_column="plot_x", interval_offset=0.045)
    return (
        _estimate_plot(
            intervals=intervals,
            points=player_estimates,
            point_x="plot_x",
            x_label="",
            y_label="Player ability, alpha (log odds)",
            title=title,
            subtitle=subtitle,
            figure_size=figure_size,
            legend_position="none",
            legend_box_spacing=0,
        )
        + p9.scale_x_continuous(
            breaks=ordered_players["player_position"].tolist(),
            labels=ordered_players["player_label"].tolist(),
        )
        + p9.theme(
            axis_text_x=p9.element_text(
                rotation=50,
                ha="right",
                va="top",
                size=8,
            )
        )
    )


def _interval_frame(
    summary: pd.DataFrame,
    x_column: str,
    interval_offset: float,
) -> pd.DataFrame:
    """Stack within- and between-run intervals with small x offsets."""
    within = summary.rename(columns={"within_q05": "lower", "within_q95": "upper"}).assign(
        interval="Within-run posterior interval",
        interval_x=lambda frame: frame[x_column] - interval_offset,
    )
    between = summary.rename(columns={"between_q05": "lower", "between_q95": "upper"}).assign(
        interval="Between-run median spread",
        interval_x=lambda frame: frame[x_column] + interval_offset,
    )
    intervals = pd.concat([within, between], ignore_index=True)
    intervals["interval"] = pd.Categorical(
        intervals["interval"],
        categories=[
            "Within-run posterior interval",
            "Between-run median spread",
        ],
        ordered=True,
    )
    return intervals


def _estimate_plot(
    intervals: pd.DataFrame,
    points: pd.DataFrame,
    point_x: str,
    x_label: str,
    y_label: str,
    title: str,
    subtitle: str,
    figure_size: tuple[float, float],
    legend_position: str,
    legend_box_spacing: float,
) -> p9.ggplot:
    """Build the shared repeated-fit interval plot."""
    diagnostic_rates = (0, 0.25, 0.5, 0.75, 1)
    diagnostic_opacities = tuple(0.05 + 0.95 * value**2 for value in diagnostic_rates)
    guide_layout = (
        p9.guides(
            color=p9.guide_legend(ncol=1),
            size=p9.guide_legend(ncol=1),
            alpha=p9.guide_legend(ncol=1),
        )
        if legend_position == "right"
        else p9.guides(
            color=p9.guide_legend(nrow=1),
            size=p9.guide_legend(nrow=1),
            alpha=p9.guide_legend(nrow=1),
        )
    )
    return (
        p9.ggplot(intervals, p9.aes(x="interval_x", color="Parameterization"))
        + p9.geom_linerange(
            p9.aes(
                ymin="lower",
                ymax="upper",
                size="interval",
                alpha="diagnostic_opacity",
            )
        )
        + p9.geom_point(
            data=points,
            mapping=p9.aes(
                x=point_x,
                y="estimate",
                alpha="diagnostic_opacity",
            ),
            size=1.8,
        )
        + p9.scale_color_manual(values={"Centered": "#0072B2", "Non-centered": "#D55E00"})
        + p9.scale_size_manual(
            values={
                "Within-run posterior interval": 0.7,
                "Between-run median spread": 2.4,
            }
        )
        + p9.scale_alpha_continuous(
            limits=(0.05, 1),
            breaks=diagnostic_opacities,
            labels=("0%", "25%", "50%", "75%", "100%"),
            range=(0.05, 1.0),
        )
        + guide_layout
        + p9.labs(
            x=x_label,
            y=y_label,
            color="Parameterization",
            size="Interval",
            alpha="Diagnostic pass rate",
            title=title,
            subtitle=subtitle,
        )
        + p9.theme_minimal()
        + p9.theme(
            figure_size=figure_size,
            legend_position=legend_position,
            legend_box="vertical",
            legend_box_just="left",
            legend_box_spacing=legend_box_spacing,
            legend_title=p9.element_text(size=9),
            legend_text=p9.element_text(size=8),
        )
    )
