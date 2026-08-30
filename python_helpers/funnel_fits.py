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


def parameter_estimate_frame(
    estimates: pd.DataFrame,
    n_obs_by_parameterization: Mapping[str, Sequence[int]],
    variables: Sequence[str] = ("log_sigma_sq", "theta[1]"),
) -> pd.DataFrame:
    """Select common posterior summaries for specified data regimes."""
    selected_frames = []
    for parameterization, n_obs_values in n_obs_by_parameterization.items():
        selected_frames.append(
            estimates[
                estimates["Parameterization"].str.casefold().eq(parameterization.casefold())
                & estimates["N"].isin(n_obs_values)
                & estimates["Variable"].isin(variables)
            ]
        )

    frame = pd.concat(selected_frames, ignore_index=True)
    if frame.empty:
        raise ValueError("No fits matched the selected data regimes.")

    frame["Variable"] = pd.Categorical(
        frame["Variable"],
        categories=list(variables),
        ordered=True,
    )
    return frame.sort_values(["Variable", "Parameterization", "N"])


def plot_parameter_estimates(
    estimates: pd.DataFrame,
    n_obs_by_parameterization: Mapping[str, Sequence[int]],
    true_values: Mapping[str, float],
    diagnostics: pd.DataFrame | None = None,
    n_obs_breaks: Sequence[int] | None = None,
    median_r_hat_limit: float = 1.01,
    maximum_r_hat_limit: float = 1.03,
    title: str = "Posterior estimates by parameterization",
    subtitle: str | None = None,
    figure_size: tuple[float, float] = (8.0, 6.0),
) -> p9.ggplot:
    """Plot posterior summaries for common parameters of selected fits."""
    estimates = parameter_estimate_frame(
        estimates,
        n_obs_by_parameterization=n_obs_by_parameterization,
    )
    required_diagnostics = {"Median R_hat", "Maximum R_hat"}
    missing_diagnostics = required_diagnostics.difference(estimates.columns)
    if missing_diagnostics and diagnostics is not None and "Run" not in estimates:
        diagnostic_columns = [
            "Parameterization",
            "N",
            "Variable",
            *sorted(missing_diagnostics),
        ]
        diagnostic_frame = diagnostics[diagnostic_columns].copy()
        non_centered = diagnostic_frame["Parameterization"].eq("Non-centered")
        diagnostic_frame.loc[non_centered, "Variable"] = diagnostic_frame.loc[
            non_centered, "Variable"
        ].replace(
            {
                "log_sigma_sq_std": "log_sigma_sq",
                "theta_std[1]": "theta[1]",
            }
        )
        diagnostic_frame = diagnostic_frame[
            diagnostic_frame["Variable"].isin(estimates["Variable"].cat.categories)
        ]
        estimates = estimates.merge(
            diagnostic_frame,
            on=["Parameterization", "N", "Variable"],
            how="left",
            validate="one_to_one",
        )
        estimates["Variable"] = pd.Categorical(
            estimates["Variable"],
            categories=list(true_values),
            ordered=True,
        )
        missing_diagnostics = required_diagnostics.difference(estimates.columns)

    if missing_diagnostics:
        missing = ", ".join(sorted(missing_diagnostics))
        detail = (
            "Regenerate cp_ncp_data_estimate_runs.csv with run_data_fits.py."
            if "Run" in estimates
            else "Supply a usable diagnostic summary."
        )
        raise ValueError(f"Estimate data are missing diagnostic columns ({missing}). {detail}")

    diagnostics_ok = estimates["Median R_hat"].le(median_r_hat_limit) & estimates[
        "Maximum R_hat"
    ].lt(maximum_r_hat_limit)
    estimates["Diagnostics"] = diagnostics_ok.map({True: "OK", False: "R-hat warning"})

    pass_rate_columns = {"Runs", "Runs passing diagnostics"}
    use_pass_rate = pass_rate_columns.issubset(estimates.columns)
    if use_pass_rate:
        estimates["diagnostic_pass_rate"] = (
            estimates["Runs passing diagnostics"] / estimates["Runs"]
        )
        # Squaring the pass rate makes differences below 100% visually clear;
        # the floor keeps even the least reliable estimates visible.
        estimates["diagnostic_opacity"] = 0.05 + 0.95 * estimates["diagnostic_pass_rate"].pow(2)
        alpha_variable = "diagnostic_opacity"
    else:
        alpha_variable = "Diagnostics"

    if subtitle is None:
        if use_pass_rate:
            subtitle = (
                "Intervals average run-level q05–q95; opacity is the proportion "
                "of runs passing the R-hat criteria"
            )
        else:
            summary_description = (
                "Each interval is one fit; "
                if "Run" in estimates
                else "Run-level posterior summaries are averaged; "
            )
            subtitle = summary_description + (
                f"OK: median R-hat ≤ {median_r_hat_limit:.2f} and "
                f"maximum R-hat < {maximum_r_hat_limit:.2f}"
            )

    # Separate parameterizations only where they share an N, then add a small,
    # deterministic within-group jitter for repeated fits. Multiplicative
    # offsets remain visually symmetric on the logarithmic x-axis.
    parameterizations_per_n = estimates.groupby("N")["Parameterization"].transform("nunique")
    shared_n = parameterizations_per_n.gt(1)
    x_multiplier = pd.Series(1.0, index=estimates.index)
    x_multiplier.loc[shared_n & estimates["Parameterization"].eq("Centered")] = 1 / 1.08
    x_multiplier.loc[shared_n & estimates["Parameterization"].eq("Non-centered")] = 1.08
    if "Run" in estimates:
        jitter_position = (((estimates["Run"] * 37) % 101) - 50) / 50
        x_multiplier *= 1.025**jitter_position
    estimates["Plot N"] = estimates["N"] * x_multiplier

    id_vars = [
        "Parameterization",
        "N",
        "Plot N",
        "Variable",
        alpha_variable,
    ]
    if "Run" in estimates:
        id_vars.append("Run")
    points = estimates.melt(
        id_vars=id_vars,
        value_vars=["q50", "Mean"],
        var_name="Statistic",
        value_name="Estimate",
    )
    points["Statistic"] = points["Statistic"].replace({"q50": "Median"})

    references = pd.DataFrame(
        [{"Variable": variable, "True value": value} for variable, value in true_values.items()]
    )
    references["Variable"] = pd.Categorical(
        references["Variable"],
        categories=list(estimates["Variable"].cat.categories),
        ordered=True,
    )

    if n_obs_breaks is None:
        n_obs_breaks = sorted(estimates["N"].unique())

    run_level = "Run" in estimates
    interval_size = 0.3 if run_level else 0.8
    point_size = 1.0 if run_level else 2.4
    alpha_scale = (
        p9.scale_alpha_continuous(
            limits=(0.05, 1),
            breaks=tuple(0.05 + 0.95 * value**2 for value in (0, 0.25, 0.5, 0.75, 1)),
            labels=("0%", "25%", "50%", "75%", "100%"),
            range=(0.05, 1.0),
        )
        if use_pass_rate
        else p9.scale_alpha_manual(
            values=(
                {"OK": 0.65, "R-hat warning": 0.20}
                if run_level
                else {"OK": 1.0, "R-hat warning": 0.25}
            )
        )
    )

    return (
        p9.ggplot(estimates, p9.aes(x="Plot N", color="Parameterization"))
        + p9.geom_hline(
            data=references,
            mapping=p9.aes(yintercept="True value"),
            inherit_aes=False,
            color="#555555",
            linetype="dashed",
        )
        + p9.geom_linerange(
            p9.aes(ymin="q05", ymax="q95", alpha=alpha_variable),
            size=interval_size,
        )
        + p9.geom_point(
            data=points,
            mapping=p9.aes(
                y="Estimate",
                shape="Statistic",
                alpha=alpha_variable,
            ),
            size=point_size,
        )
        + p9.facet_wrap("Variable", ncol=1, scales="free_y")
        + p9.scale_x_log10(
            breaks=list(n_obs_breaks),
            labels=[str(value) for value in n_obs_breaks],
        )
        + p9.scale_color_manual(values={"Centered": "#0072B2", "Non-centered": "#D55E00"})
        + p9.scale_shape_manual(values={"Median": "o", "Mean": "x"})
        + alpha_scale
        + p9.guides(
            color=p9.guide_legend(nrow=1),
            shape=p9.guide_legend(nrow=1),
            alpha=p9.guide_legend(nrow=1),
        )
        + p9.labs(
            x="Observations per group",
            y="Posterior estimate",
            color="Parameterization",
            shape="Summary",
            alpha="Diagnostic pass rate" if use_pass_rate else "Sampler diagnostics",
            title=title,
            subtitle=subtitle,
        )
        + p9.theme_minimal()
        + p9.theme(
            figure_size=figure_size,
            legend_position="bottom",
            legend_box="vertical",
            legend_box_just="left",
            legend_title=p9.element_text(size=9),
            legend_text=p9.element_text(size=8),
        )
    )


def summarize_parameter_estimate_runs(
    estimates: pd.DataFrame,
    n_obs_by_parameterization: Mapping[str, Sequence[int]],
    median_r_hat_limit: float = 1.01,
    maximum_r_hat_limit: float = 1.03,
) -> pd.DataFrame:
    """Separate within-run uncertainty from variability across repeated fits."""
    estimates = parameter_estimate_frame(
        estimates,
        n_obs_by_parameterization=n_obs_by_parameterization,
    )
    required_columns = {
        "Run",
        "q05",
        "q50",
        "q95",
        "Median R_hat",
        "Maximum R_hat",
    }
    missing_columns = required_columns.difference(estimates.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Run-level estimate data are missing columns: {missing}")

    estimates["diagnostics_ok"] = estimates["Median R_hat"].le(median_r_hat_limit) & estimates[
        "Maximum R_hat"
    ].lt(maximum_r_hat_limit)
    summary = (
        estimates.groupby(
            ["Parameterization", "N", "Variable"],
            sort=False,
            observed=True,
        )
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
    summary["Variable"] = pd.Categorical(
        summary["Variable"],
        categories=list(estimates["Variable"].cat.categories),
        ordered=True,
    )

    parameterizations_per_n = summary.groupby("N")["Parameterization"].transform("nunique")
    shared_n = parameterizations_per_n.gt(1)
    x_multiplier = pd.Series(1.0, index=summary.index)
    x_multiplier.loc[shared_n & summary["Parameterization"].eq("Centered")] = 1 / 1.08
    x_multiplier.loc[shared_n & summary["Parameterization"].eq("Non-centered")] = 1.08
    summary["plot_n"] = summary["N"] * x_multiplier
    return summary


def plot_parameter_estimate_variability(
    estimates: pd.DataFrame,
    n_obs_by_parameterization: Mapping[str, Sequence[int]],
    true_values: Mapping[str, float],
    n_obs_breaks: Sequence[int] | None = None,
    median_r_hat_limit: float = 1.01,
    maximum_r_hat_limit: float = 1.03,
    title: str = "Posterior estimates across repeated fits",
    subtitle: str = (
        "Thin: mean posterior q05–q95; thick: q05–q95 of run medians\n"
        "Point: median run estimate; intervals offset for visibility"
    ),
    figure_size: tuple[float, float] = (8.0, 6.0),
) -> p9.ggplot:
    """Plot within-run uncertainty and between-run estimate variability."""
    summary = summarize_parameter_estimate_runs(
        estimates,
        n_obs_by_parameterization=n_obs_by_parameterization,
        median_r_hat_limit=median_r_hat_limit,
        maximum_r_hat_limit=maximum_r_hat_limit,
    )
    within_intervals = summary.rename(
        columns={"within_q05": "lower", "within_q95": "upper"}
    ).assign(
        interval="Within-run posterior interval",
        interval_plot_n=lambda frame: frame["plot_n"] / 1.025,
    )
    between_intervals = summary.rename(
        columns={"between_q05": "lower", "between_q95": "upper"}
    ).assign(
        interval="Between-run median spread",
        interval_plot_n=lambda frame: frame["plot_n"] * 1.025,
    )
    intervals = pd.concat(
        [within_intervals, between_intervals],
        ignore_index=True,
    )
    intervals["interval"] = pd.Categorical(
        intervals["interval"],
        categories=[
            "Within-run posterior interval",
            "Between-run median spread",
        ],
        ordered=True,
    )

    references = pd.DataFrame(
        [{"Variable": variable, "True value": value} for variable, value in true_values.items()]
    )
    references["Variable"] = pd.Categorical(
        references["Variable"],
        categories=list(summary["Variable"].cat.categories),
        ordered=True,
    )
    if n_obs_breaks is None:
        n_obs_breaks = sorted(summary["N"].unique())

    diagnostic_rates = (0, 0.25, 0.5, 0.75, 1)
    diagnostic_opacities = tuple(0.05 + 0.95 * value**2 for value in diagnostic_rates)

    return (
        p9.ggplot(intervals, p9.aes(x="interval_plot_n", color="Parameterization"))
        + p9.geom_hline(
            data=references,
            mapping=p9.aes(yintercept="True value"),
            inherit_aes=False,
            color="#555555",
            linetype="dashed",
        )
        + p9.geom_linerange(
            p9.aes(
                ymin="lower",
                ymax="upper",
                size="interval",
                alpha="diagnostic_opacity",
            )
        )
        + p9.geom_point(
            data=summary,
            mapping=p9.aes(
                x="plot_n",
                y="estimate",
                alpha="diagnostic_opacity",
            ),
            size=1.8,
        )
        + p9.facet_wrap("Variable", ncol=1, scales="free_y")
        + p9.scale_x_log10(
            breaks=list(n_obs_breaks),
            labels=[str(value) for value in n_obs_breaks],
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
        + p9.guides(
            color=p9.guide_legend(nrow=1),
            size=p9.guide_legend(nrow=1),
            alpha=p9.guide_legend(nrow=1),
        )
        + p9.labs(
            x="Observations per group",
            y="Posterior estimate",
            color="Parameterization",
            size="Interval",
            alpha="Diagnostic pass rate",
            title=title,
            subtitle=subtitle,
        )
        + p9.theme_minimal()
        + p9.theme(
            figure_size=figure_size,
            legend_position="bottom",
            legend_box="vertical",
            legend_box_just="left",
            legend_title=p9.element_text(size=9),
            legend_text=p9.element_text(size=8),
        )
    )


def _facet_label(parameterization: Parameterization) -> str:
    """Return the row label used by the faceted sampler plots."""
    return f"{parameterization.label}: {parameterization.sampled_x}"
