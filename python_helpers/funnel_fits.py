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


def _facet_label(parameterization: Parameterization) -> str:
    """Return the row label used by the faceted sampler plots."""
    return f"{parameterization.label}: {parameterization.sampled_x}"
