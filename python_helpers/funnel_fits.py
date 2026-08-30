"""Fitting and plotting helpers for centered/non-centered funnel models."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd
import plotnine as p9
from cmdstanpy import CmdStanMCMC, CmdStanModel
from plotnine.composition import Compose


@dataclass(frozen=True)
class Parameterization:
    """A Stan model and the two coordinates sampled for a funnel plot."""

    label: str
    model: CmdStanModel
    sampled_x: str
    sampled_y: str


@dataclass(frozen=True)
class ModelFit:
    """A fitted parameterization for one dataset size."""

    parameterization: Parameterization
    n_obs: int
    fit: CmdStanMCMC


def plot_divergences(
    fit: CmdStanMCMC,
    x_var: str,
    y_var: str,
    title: str | None = None,
    subtitle: str | None = None,
) -> p9.ggplot:
    """Scatterplot of two parameters with divergent transitions in red."""
    plot_data = _draws_xy(fit, x_var, y_var)
    nondivergent = plot_data[~plot_data["divergent"]]
    divergent = plot_data[plot_data["divergent"]]

    return (
        p9.ggplot(plot_data, p9.aes(x="x", y="y"))
        + p9.geom_point(
            data=nondivergent,
            color="#333333",
            alpha=0.5,
            size=1.0,
        )
        + p9.geom_point(
            data=divergent,
            color="red",
            alpha=0.8,
            size=1.0,
        )
        + p9.labs(
            x=x_var,
            y=y_var,
            title=title if title is not None else f"{y_var} vs {x_var}",
            subtitle=subtitle,
        )
        + p9.theme_minimal()
    )


def fit_parameterizations(
    parameterizations: Sequence[Parameterization],
    datasets: Mapping[int, dict[str, object]],
    n_obs_values: Sequence[int],
    seed_offset: int,
) -> list[ModelFit]:
    """Fit every parameterization to every requested dataset size."""
    return [
        ModelFit(
            parameterization=parameterization,
            n_obs=n_obs,
            fit=parameterization.model.sample(
                data=datasets[n_obs],
                seed=seed_offset + n_obs,
                show_progress=False,
            ),
        )
        for parameterization in parameterizations
        for n_obs in n_obs_values
    ]


def _draws_xy(
    fit: CmdStanMCMC,
    x_variable: str,
    y_variable: str,
) -> pd.DataFrame:
    """Extract two variables and the divergence indicator from a fit."""
    draws = fit.draws_pd()
    return pd.DataFrame(
        {
            "x": draws[x_variable].to_numpy(),
            "y": draws[y_variable].to_numpy(),
            "divergent": draws["divergent__"].to_numpy() > 0,
        }
    )


def facet_frame(
    fits: Sequence[ModelFit],
    n_obs_values: Sequence[int],
) -> pd.DataFrame:
    """Stack selected fits into a tidy frame for faceted plotting."""
    requested_sizes = set(n_obs_values)
    selected_fits = [result for result in fits if result.n_obs in requested_sizes]

    frames = []
    for result in selected_fits:
        parameterization = result.parameterization
        facet_label = _facet_label(parameterization)
        frames.append(
            _draws_xy(
                result.fit,
                x_variable=parameterization.sampled_x,
                y_variable=parameterization.sampled_y,
            ).assign(
                parameterization=facet_label,
                N=f"N = {result.n_obs}",
            )
        )

    frame = pd.concat(frames, ignore_index=True)
    parameterization_order = list(
        dict.fromkeys(_facet_label(result.parameterization) for result in fits)
    )
    frame["parameterization"] = pd.Categorical(
        frame["parameterization"],
        categories=parameterization_order,
        ordered=True,
    )
    frame["N"] = pd.Categorical(
        frame["N"],
        categories=[f"N = {n_obs}" for n_obs in n_obs_values],
        ordered=True,
    )
    return frame


def plot_divergence_grid(
    data: pd.DataFrame,
    parameterizations: Sequence[Parameterization],
    title: str,
    subtitle: str = "Divergent transitions in red",
    figure_size: tuple[float, float] = (8.0, 6.0),
    coords: p9.coord_cartesian | None = None,
) -> Compose:
    """Plot a vertically stacked sampler-coordinate grid."""
    if coords is None:
        coords = p9.coord_cartesian(xlim=(-10, 10), ylim=(-10, 10))

    plots = []
    for index, parameterization in enumerate(parameterizations):
        facet_label = _facet_label(parameterization)
        plot_data = data[data["parameterization"] == facet_label].copy()
        nondivergent = plot_data[~plot_data["divergent"]]
        divergent = plot_data[plot_data["divergent"]]

        plots.append(
            p9.ggplot(plot_data, p9.aes(x="x", y="y"))
            + p9.geom_point(
                data=nondivergent,
                color="#333333",
                alpha=0.4,
                size=0.7,
            )
            + p9.geom_point(
                data=divergent,
                color="red",
                alpha=0.8,
                size=0.7,
            )
            + p9.facet_grid(
                cols="N",
                scales="fixed",
            )
            + coords
            + p9.labs(
                x=parameterization.sampled_x,
                y=parameterization.sampled_y,
                title=title if index == 0 else None,
                subtitle=subtitle if index == 0 else None,
            )
            + p9.theme_minimal()
            + p9.theme(figure_size=figure_size)
        )

    upper_plot, lower_plot = plots
    return upper_plot / lower_plot


def rhat_reliability_frame(
    estimate_runs: pd.DataFrame,
    maximum_r_hat_limit: float = 1.01,
) -> pd.DataFrame:
    """Calculate the R-hat pass rate for each dataset and parameterization."""
    required_columns = {
        "Parameterization",
        "N",
        "Run",
        "Maximum R_hat",
    }
    missing_columns = required_columns.difference(estimate_runs.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Run-level estimates are missing columns: {missing}")

    # Maximum R-hat is repeated for each reported variable, so reduce the
    # data to one diagnostic result for each fitted model.
    run_diagnostics = (
        estimate_runs.groupby(
            ["Parameterization", "N", "Run"],
            as_index=False,
        )
        .agg(maximum_r_hat=("Maximum R_hat", "max"))
    )
    run_diagnostics["passed"] = run_diagnostics["maximum_r_hat"].le(
        maximum_r_hat_limit
    )

    reliability = (
        run_diagnostics.groupby(
            ["Parameterization", "N"],
            as_index=False,
            sort=False,
        )
        .agg(
            runs=("Run", "nunique"),
            runs_passing=("passed", "sum"),
        )
    )
    reliability["pass_rate"] = reliability["runs_passing"] / reliability["runs"]
    reliability["Parameterization"] = pd.Categorical(
        reliability["Parameterization"],
        categories=["Centered", "Non-centered"],
        ordered=True,
    )
    return reliability.sort_values(["Parameterization", "N"])


def plot_rhat_reliability(
    estimate_runs: pd.DataFrame,
    maximum_r_hat_limit: float = 1.01,
    n_obs_breaks: Sequence[int] | None = None,
    title: str = "Reliability across repeated fits",
    figure_size: tuple[float, float] = (8.0, 4.5),
) -> p9.ggplot:
    """Plot the percentage of runs passing the R-hat criterion."""
    reliability = rhat_reliability_frame(
        estimate_runs,
        maximum_r_hat_limit=maximum_r_hat_limit,
    )
    if n_obs_breaks is None:
        n_obs_breaks = sorted(reliability["N"].unique())

    return (
        p9.ggplot(
            reliability,
            p9.aes(
                x="N",
                y="pass_rate",
                color="Parameterization",
                group="Parameterization",
            ),
        )
        + p9.geom_line(size=1.0)
        + p9.geom_point(size=2.5)
        + p9.scale_x_log10(
            breaks=list(n_obs_breaks),
            labels=[f"{value:,}" for value in n_obs_breaks],
        )
        + p9.scale_y_continuous(
            limits=(0, 1),
            breaks=(0, 0.25, 0.50, 0.75, 1),
            labels=("0%", "25%", "50%", "75%", "100%"),
            expand=(0.02, 0.02),
        )
        + p9.scale_color_manual(
            values={
                "Centered": "#0072B2",
                "Non-centered": "#D55E00",
            }
        )
        + p9.labs(
            x="Observations per group, N",
            y="Runs passing the R-hat criterion",
            color="Parameterization",
            title=title,
            subtitle=(
                f"Pass: maximum R-hat ≤ {maximum_r_hat_limit:.2f} "
                "across all monitored parameters"
            ),
        )
        + p9.theme_minimal()
        + p9.theme(
            figure_size=figure_size,
            legend_position="top",
            panel_grid_minor=p9.element_blank(),
        )
    )


def ess_efficiency_frame(
    estimate_runs: pd.DataFrame,
    maximum_r_hat_limit: float = 1.01,
) -> pd.DataFrame:
    """Prepare run-level ESS/s values with their R-hat result."""
    required_columns = {
        "Parameterization",
        "N",
        "Run",
        "Variable",
        "Maximum R_hat",
        "ESS_bulk/s",
    }
    missing_columns = required_columns.difference(estimate_runs.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Run-level estimates are missing columns: {missing}")

    run_efficiency = estimate_runs.dropna(subset=["ESS_bulk/s"]).copy()
    run_efficiency["diagnostic_status"] = run_efficiency["Maximum R_hat"].le(
        maximum_r_hat_limit
    ).map({True: "Passed", False: "Did not pass"})

    parameter_labels = {
        "log_sigma_sq": "Population scale: log_sigma_sq",
        "theta[1]": "Group effect: theta[1]",
    }
    run_efficiency["parameter"] = run_efficiency["Variable"].replace(parameter_labels)
    run_efficiency["parameter"] = pd.Categorical(
        run_efficiency["parameter"],
        categories=list(parameter_labels.values()),
        ordered=True,
    )
    run_efficiency["Parameterization"] = pd.Categorical(
        run_efficiency["Parameterization"],
        categories=["Centered", "Non-centered"],
        ordered=True,
    )
    run_efficiency["diagnostic_status"] = pd.Categorical(
        run_efficiency["diagnostic_status"],
        categories=["Passed", "Did not pass"],
        ordered=True,
    )

    # Multiplicative jitter is symmetric on the logarithmic x-axis. A small
    # parameterization offset keeps the two clouds visible at the same N.
    parameterization_offset = run_efficiency["Parameterization"].map(
        {"Centered": 1 / 1.055, "Non-centered": 1.055}
    ).astype(float)
    jitter_position = (((run_efficiency["Run"] * 37) % 101) - 50) / 50
    run_efficiency["plot_n"] = (
        run_efficiency["N"] * parameterization_offset * 1.025**jitter_position
    )
    return run_efficiency.sort_values(
        ["parameter", "Parameterization", "N", "Run"]
    )


def plot_ess_efficiency(
    estimate_runs: pd.DataFrame,
    maximum_r_hat_limit: float = 1.01,
    n_obs_breaks: Sequence[int] | None = None,
    title: str = "ESS/s by R-hat result",
    figure_size: tuple[float, float] = (8.0, 6.0),
) -> p9.ggplot:
    """Plot run-level ESS_bulk/s and an all-run median for each model."""
    run_efficiency = ess_efficiency_frame(
        estimate_runs,
        maximum_r_hat_limit=maximum_r_hat_limit,
    )
    median_efficiency = (
        run_efficiency.groupby(
            ["Parameterization", "N", "Variable", "parameter"],
            as_index=False,
            sort=False,
            observed=True,
        )
        .agg(median_ess_per_second=("ESS_bulk/s", "median"))
    )
    if n_obs_breaks is None:
        n_obs_breaks = sorted(estimate_runs["N"].unique())

    return (
        p9.ggplot(
            run_efficiency,
            p9.aes(
                x="plot_n",
                y="ESS_bulk/s",
                color="Parameterization",
                alpha="diagnostic_status",
            ),
        )
        + p9.geom_point(size=0.9)
        + p9.geom_line(
            data=median_efficiency,
            mapping=p9.aes(
                x="N",
                y="median_ess_per_second",
                color="Parameterization",
                group="Parameterization",
            ),
            inherit_aes=False,
            size=1.0,
        )
        + p9.facet_wrap(
            "parameter",
            ncol=1,
            scales="fixed",
        )
        + p9.scale_x_log10(
            breaks=list(n_obs_breaks),
            labels=[f"{value:,}" for value in n_obs_breaks],
        )
        + p9.scale_y_log10(
            labels=lambda values: [f"{value:,.0f}" for value in values]
        )
        + p9.scale_color_manual(
            values={
                "Centered": "#0072B2",
                "Non-centered": "#D55E00",
            }
        )
        + p9.scale_alpha_manual(
            values={
                "Passed": 1.0,
                "Did not pass": 0.5,
            }
        )
        + p9.labs(
            x="Observations per group, N",
            y="Median ESS_bulk/s",
            color="Parameterization",
            alpha="R-hat criterion",
            title=title,
            subtitle=(
                "Points: individual runs; lines: all-run medians; "
                f"faded: maximum R-hat > {maximum_r_hat_limit:.2f}"
            ),
        )
        + p9.theme_minimal()
        + p9.theme(
            figure_size=figure_size,
            legend_position="top",
            panel_grid_minor=p9.element_blank(),
        )
    )


def sampler_time_frame(fit_summary: pd.DataFrame) -> pd.DataFrame:
    """Reduce parameter-level summaries to one sampler time per fitted model."""
    required_columns = {
        "Parameterization",
        "N",
        "Median total sampler seconds",
    }
    missing_columns = required_columns.difference(fit_summary.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Fit summaries are missing columns: {missing}")

    time_counts = fit_summary.groupby(["Parameterization", "N"])[
        "Median total sampler seconds"
    ].nunique(dropna=False)
    if time_counts.gt(1).any():
        raise ValueError(
            "Median sampler time is inconsistent across variables from the same fits."
        )

    sampler_times = fit_summary[
        ["Parameterization", "N", "Median total sampler seconds"]
    ].drop_duplicates(["Parameterization", "N"])
    sampler_times = sampler_times.rename(
        columns={"Median total sampler seconds": "median_total_sampler_seconds"}
    )
    sampler_times["Parameterization"] = pd.Categorical(
        sampler_times["Parameterization"],
        categories=["Centered", "Non-centered"],
        ordered=True,
    )
    return sampler_times.sort_values(["Parameterization", "N"])


def plot_sampler_time(
    fit_summary: pd.DataFrame,
    n_obs_breaks: Sequence[int] | None = None,
    title: str = "Sampler time by parameterization",
    figure_size: tuple[float, float] = (8.0, 4.5),
) -> p9.ggplot:
    """Plot median warmup plus sampling time across repeated fits."""
    sampler_times = sampler_time_frame(fit_summary)
    if n_obs_breaks is None:
        n_obs_breaks = sorted(sampler_times["N"].unique())

    return (
        p9.ggplot(
            sampler_times,
            p9.aes(
                x="N",
                y="median_total_sampler_seconds",
                color="Parameterization",
                group="Parameterization",
            ),
        )
        + p9.geom_line(size=1.0)
        + p9.geom_point(size=2.5)
        + p9.scale_x_log10(
            breaks=list(n_obs_breaks),
            labels=[f"{value:,}" for value in n_obs_breaks],
        )
        + p9.scale_y_log10(
            labels=lambda values: [f"{value:.2f}" for value in values]
        )
        + p9.scale_color_manual(
            values={
                "Centered": "#0072B2",
                "Non-centered": "#D55E00",
            }
        )
        + p9.labs(
            x="Observations per group, N",
            y="Median warmup + sampling time (seconds)",
            color="Parameterization",
            title=title,
            subtitle="Sampler-reported time across four chains; median across 100 runs",
        )
        + p9.theme_minimal()
        + p9.theme(
            figure_size=figure_size,
            legend_position="top",
            panel_grid_minor=p9.element_blank(),
        )
    )


def sampler_tuning_frame(fit_summary: pd.DataFrame) -> pd.DataFrame:
    """Prepare one step-size and leapfrog summary per fitted model."""
    tuning_columns = ["Stepsize", "Average leapfrog steps"]
    required_columns = {"Parameterization", "N", *tuning_columns}
    missing_columns = required_columns.difference(fit_summary.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Fit summaries are missing columns: {missing}")

    tuning_counts = fit_summary.groupby(["Parameterization", "N"])[
        tuning_columns
    ].nunique(dropna=False)
    if tuning_counts.gt(1).any().any():
        raise ValueError(
            "Sampler tuning summaries are inconsistent across variables from the same fits."
        )

    sampler_tuning = fit_summary[
        ["Parameterization", "N", *tuning_columns]
    ].drop_duplicates(["Parameterization", "N"])
    sampler_tuning = sampler_tuning.melt(
        id_vars=["Parameterization", "N"],
        value_vars=tuning_columns,
        var_name="metric",
        value_name="value",
    )
    metric_labels = {
        "Stepsize": "Integrator step size",
        "Average leapfrog steps": "Leapfrog steps per iteration",
    }
    sampler_tuning["metric"] = sampler_tuning["metric"].replace(metric_labels)
    sampler_tuning["metric"] = pd.Categorical(
        sampler_tuning["metric"],
        categories=list(metric_labels.values()),
        ordered=True,
    )
    sampler_tuning["Parameterization"] = pd.Categorical(
        sampler_tuning["Parameterization"],
        categories=["Centered", "Non-centered"],
        ordered=True,
    )
    return sampler_tuning.sort_values(["metric", "Parameterization", "N"])


def plot_sampler_tuning(
    fit_summary: pd.DataFrame,
    n_obs_breaks: Sequence[int] | None = None,
    title: str = "Sampler adaptation by parameterization",
    figure_size: tuple[float, float] = (8.0, 6.0),
) -> p9.ggplot:
    """Plot average integrator step size and leapfrog steps."""
    sampler_tuning = sampler_tuning_frame(fit_summary)
    if n_obs_breaks is None:
        n_obs_breaks = sorted(sampler_tuning["N"].unique())

    return (
        p9.ggplot(
            sampler_tuning,
            p9.aes(
                x="N",
                y="value",
                color="Parameterization",
                group="Parameterization",
            ),
        )
        + p9.geom_line(size=1.0)
        + p9.geom_point(size=2.5)
        + p9.facet_wrap(
            "metric",
            ncol=1,
            scales="free_y",
        )
        + p9.scale_x_log10(
            breaks=list(n_obs_breaks),
            labels=[f"{value:,}" for value in n_obs_breaks],
        )
        + p9.scale_y_log10(
            labels=lambda values: [
                f"{value:.2f}" if value < 1 else f"{value:,.0f}"
                for value in values
            ]
        )
        + p9.scale_color_manual(
            values={
                "Centered": "#0072B2",
                "Non-centered": "#D55E00",
            }
        )
        + p9.labs(
            x="Observations per group, N",
            y="Sampler tuning value (log scale)",
            color="Parameterization",
            title=title,
            subtitle="Each point averages 100 runs; both axes use logarithmic scales",
        )
        + p9.theme_minimal()
        + p9.theme(
            figure_size=figure_size,
            legend_position="top",
            panel_grid_minor=p9.element_blank(),
        )
    )


def single_fit_parameter_frame(
    fits: Sequence[ModelFit],
    variables: Sequence[str] = ("log_sigma_sq", "theta[1]"),
) -> pd.DataFrame:
    """Summarize common parameters from one fit of each model and dataset."""
    if not fits:
        raise ValueError("At least one fitted model is required.")

    rows = []
    for result in fits:
        draws = result.fit.draws_pd()
        missing_variables = set(variables).difference(draws.columns)
        if missing_variables:
            missing = ", ".join(sorted(missing_variables))
            raise ValueError(f"Fit draws are missing variables: {missing}")

        for variable in variables:
            variable_draws = draws[variable]
            parameterization_label = {
                "centered": "Centered",
                "non-centered": "Non-centered",
            }.get(
                result.parameterization.label.casefold(),
                result.parameterization.label,
            )
            rows.append(
                {
                    "Parameterization": parameterization_label,
                    "N": result.n_obs,
                    "Variable": variable,
                    "q05": variable_draws.quantile(0.05),
                    "q50": variable_draws.quantile(0.50),
                    "q95": variable_draws.quantile(0.95),
                }
            )

    estimates = pd.DataFrame(rows)
    parameterization_order = list(
        dict.fromkeys(estimates["Parameterization"])
    )
    estimates["Parameterization"] = pd.Categorical(
        estimates["Parameterization"],
        categories=parameterization_order,
        ordered=True,
    )
    estimates["Variable"] = pd.Categorical(
        estimates["Variable"],
        categories=list(variables),
        ordered=True,
    )
    parameterization_offset = {
        "Centered": 1 / 1.06,
        "Non-centered": 1.06,
    }
    estimates["plot_n"] = estimates["N"] * estimates["Parameterization"].map(
        parameterization_offset
    ).astype(float)
    return estimates.sort_values(["Variable", "Parameterization", "N"])


def plot_single_fit_estimates(
    fits: Sequence[ModelFit],
    true_values: Mapping[str, float],
    n_obs_breaks: Sequence[int] | None = None,
    title: str = "Posterior estimates from a single set of fits",
    subtitle: str = (
        "Points: posterior medians; intervals: posterior q05–q95; "
        "dashed lines: data-generating values"
    ),
    figure_size: tuple[float, float] = (8.0, 6.0),
) -> p9.ggplot:
    """Plot posterior summaries from the fits used in the geometry plots."""
    estimates = single_fit_parameter_frame(
        fits,
        variables=tuple(true_values),
    )
    references = pd.DataFrame(
        [
            {"Variable": variable, "true_value": value}
            for variable, value in true_values.items()
        ]
    )
    references["Variable"] = pd.Categorical(
        references["Variable"],
        categories=list(true_values),
        ordered=True,
    )
    if n_obs_breaks is None:
        n_obs_breaks = sorted(estimates["N"].unique())

    return (
        p9.ggplot(
            estimates,
            p9.aes(
                x="plot_n",
                color="Parameterization",
                group="Parameterization",
            ),
        )
        + p9.geom_hline(
            data=references,
            mapping=p9.aes(yintercept="true_value"),
            inherit_aes=False,
            color="#555555",
            linetype="dashed",
        )
        + p9.geom_line(p9.aes(y="q50"), size=0.7)
        + p9.geom_linerange(
            p9.aes(ymin="q05", ymax="q95"),
            size=0.8,
        )
        + p9.geom_point(p9.aes(y="q50"), size=2.2)
        + p9.facet_wrap("Variable", ncol=1, scales="free_y")
        + p9.scale_x_log10(
            breaks=list(n_obs_breaks),
            labels=[f"{value:,}" for value in n_obs_breaks],
        )
        + p9.scale_color_manual(
            values={
                "Centered": "#0072B2",
                "Non-centered": "#D55E00",
            }
        )
        + p9.labs(
            x="Observations per group, N",
            y="Posterior estimate",
            color="Parameterization",
            title=title,
            subtitle=subtitle,
        )
        + p9.theme_minimal()
        + p9.theme(
            figure_size=figure_size,
            legend_position="top",
            panel_grid_minor=p9.element_blank(),
        )
    )




def _facet_label(parameterization: Parameterization) -> str:
    """Return the row label used by the faceted sampler plots."""
    return f"{parameterization.label}: {parameterization.sampled_x}"
