# -*- coding: utf-8 -*-
"""
Plots of the profiles tsib initializes and of its thermal zone.

Every function takes an optional `ax` and returns the axes it drew on, which
is matplotlib's own convention: one function draws one thing, and figures are
composed by the caller.

    fig, (a1, a2) = plt.subplots(2, 1, sharex=True)
    tsib.plotting.plot_profiles(bdg, ax=a1, window=("2010-01-01", "2010-01-05"))
    tsib.plotting.plot_zone(results, bdg.cfg, ax=a2, window=…)

Nothing here calls `plt.show()` or touches global state outside `use_style()`
- when a plot ends up in a notebook, a script or a PNG is the caller's
decision.

A year is 8760 points and does not read as a line plot. Four ways out, and
they answer different questions: `window=` to zoom in on a few days,
`resample=` for a whole-year overview, `plot_heatmap` to see every value at
once as a day x hour carpet, and `plot_load_duration` to ask how many hours
sit near the peak - the question a grid connection cares about.

Plots of a solved energy system - dispatch, storage content, grid exchange -
live with the model that produces them, in `esmkit.plotting`.
"""

import contextlib
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .envelope import comfort_bounds


#: same colour across figures. Taken from examples/energysystem/.
COLORS = {
    "heat": "#c1440e",
    "pv": "#e0a800",
    "grid": "#4a4a4a",
    "battery": "#2e6f9e",
    "secondary": "#8c8c8c",
    "cool": "#4a90a4",
}

#: component name -> colour. Names are the ones the building template
#: (`system.py`) produces.
ROLE_COLORS = {
    "grid": COLORS["grid"],
    "pv": COLORS["pv"],
    "battery": COLORS["battery"],
    "heat_pump": COLORS["heat"],
    "hp": COLORS["heat"],
    "buffer": "#e08a4a",
    "heat_supply": COLORS["heat"],
    "cool_supply": COLORS["cool"],
    "thermalzone": COLORS["secondary"],
    "dhw_load": "#7fa8c4",
    "household_load": COLORS["secondary"],
}

#: fallback for components the palette does not know
_FALLBACK_COLORS = ["#6a51a3", "#31a354", "#d95f0e", "#756bb1", "#636363"]

RC_PARAMS = {
    "figure.figsize": (11, 3.2),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.fontsize": 8,
    "legend.frameon": False,
}

@contextlib.contextmanager
def use_style(**overrides):
    """
    Applies tsib's figure defaults for the duration of the block.

    The plot functions colour their own artists, so this only affects the
    figure furniture - size, grid, spines.

        with tsib.plotting.use_style():
            fig, ax = plt.subplots()
            tsib.plotting.plot_profiles(bdg, ax=ax)
    """
    params = dict(RC_PARAMS)
    params.update(overrides)
    with plt.rc_context(params):
        yield

def _color(name, cycle=None):
    if name in ROLE_COLORS:
        return ROLE_COLORS[name]
    return next(cycle) if cycle is not None else COLORS["secondary"]

def _legend(ax, ncol=1):
    """Adds the legend with enough headroom that it clears the data."""
    ax.margins(y=0.20)
    ax.legend(loc="upper right", ncol=ncol)
    return ax

def _ax(ax, **subplot_kwargs):
    if ax is not None:
        return ax
    _, ax = plt.subplots(**subplot_kwargs)
    return ax

def _window(obj, window):
    """
    Narrows a Series/DataFrame to `window`.

    Accepts an int (the first n steps), a slice of positions, a
    `(start, end)` pair of dates, or a single partial date string such as
    "2010-01".
    """
    if window is None:
        return obj
    if isinstance(window, int):
        return obj.iloc[:window]
    if isinstance(window, slice):
        if isinstance(window.start, (int, type(None))) and isinstance(
            window.stop, (int, type(None))
        ):
            return obj.iloc[window]
        return obj.loc[window]
    if isinstance(window, tuple) and len(window) == 2:
        return obj.loc[window[0]:window[1]]
    return obj.loc[window]

def _resample(obj, rule, how="mean"):
    if rule is None:
        return obj
    return getattr(obj.resample(rule), how)()

# ---------------------------------------------------------------------
# the thermal zone
# ---------------------------------------------------------------------


def plot_zone(results, cfg, ax=None, window=None, name="thermalzone",
              band=True, title=None):
    """
    Thermal zone temperatures against the comfort band.

    The band is the *effective* one: `comfortT_lb`/`comfortT_ub` are only
    nominal, and `capControl`, `nightReduction` and `occControl` move the
    bounds per time step (`tsib.envelope.comfort_bounds`). With the
    default `nightReduction=True` the floor drops while the occupants sleep,
    which is exactly the room the optimizer uses.

    The zone's heating and cooling loads are not drawn here - they are flows
    on the heat bus, so `esmkit.plotting.plot_dispatch(results, "heat")`
    shows them together with whatever supplies them.

    Parameters
    ----------
    results: dict, required
    cfg: dict or ThermalZoneConfig, required
        The configuration the zone was built from.
    """
    ax = _ax(ax)
    frame = results[name]["timeseries"]

    if band:
        lower, upper = comfort_bounds(cfg, len(frame))
        band_frame = _window(
            pd.DataFrame({"lo": lower, "hi": upper}, index=frame.index), window
        )
        ax.fill_between(
            band_frame.index, band_frame["lo"], band_frame["hi"], step="post",
            color=COLORS["secondary"], alpha=0.15, linewidth=0,
            label="comfort band",
        )

    frame = _window(frame, window)
    for column, color, width in (
        ("T_air", COLORS["heat"], 1.4),
        ("T_m", COLORS["battery"], 1.0),
        ("T_e", COLORS["secondary"], 0.9),
    ):
        if column in frame:
            ax.plot(frame.index, frame[column].values, color=color, lw=width,
                    label=column)

    ax.set_ylabel("temperature [degC]")
    ax.set_title(title if title is not None else "Thermal zone")
    return _legend(ax, ncol=4)

# ---------------------------------------------------------------------
# profiles
# ---------------------------------------------------------------------


def plot_load_duration(series, ax=None, normalize=False, title=None):
    """
    Load duration curve: values sorted descending.

    The grid-relevant view - it answers how many hours a connection sits
    near its peak, which is the question a tariff scenario has to be judged
    on alongside total energy. Pass a dict of `{label: series}` to overlay
    scenarios.

    Parameters
    ----------
    series: pandas.Series, array-like, or dict of them
    normalize: bool, optional (default: False)
        x axis as a share of the period instead of hours.
    """
    ax = _ax(ax)
    scenarios = series if isinstance(series, dict) else {None: series}
    cycle = itertools.cycle(_FALLBACK_COLORS)

    for label, values in scenarios.items():
        values = np.sort(np.asarray(
            values.values if hasattr(values, "values") else values, dtype=float
        ))[::-1]
        x = np.arange(1, len(values) + 1)
        if normalize:
            x = x / len(values)
        color = _color(label, cycle) if label is not None else COLORS["grid"]
        ax.plot(x, values, lw=1.3, color=color, label=label)

    ax.set_xlabel("share of the period" if normalize else "hours")
    ax.set_ylabel("kW")
    ax.set_title(title if title is not None else "Load duration")
    if any(label is not None for label in scenarios):
        return _legend(ax)
    return ax

def plot_heatmap(series, ax=None, cmap="magma", colorbar=True, label=None,
                 title=None):
    """
    Day x hour carpet of an annual series - all 8760 values in one image.

    The seasonal shape runs along the x axis and the daily pattern up the
    y axis, so a tariff that moves consumption to other hours of the day is
    visible as a change in the vertical banding.

    Parameters
    ----------
    series: pandas.Series with a DatetimeIndex, required
    """
    ax = _ax(ax)
    if not isinstance(series, pd.Series) or not isinstance(
        series.index, pd.DatetimeIndex
    ):
        raise TypeError("plot_heatmap needs a Series with a DatetimeIndex")

    frame = pd.DataFrame({
        "day": series.index.dayofyear,
        "hour": series.index.hour,
        "value": series.values,
    })
    grid = frame.pivot_table(index="hour", columns="day", values="value",
                             aggfunc="mean")

    image = ax.imshow(grid.values, aspect="auto", origin="lower", cmap=cmap,
                      extent=[grid.columns.min(), grid.columns.max(), 0, 24])
    ax.set_xlabel("day of year")
    ax.set_ylabel("hour of day")
    ax.set_yticks([0, 6, 12, 18, 24])
    ax.grid(False)
    ax.set_title(title if title is not None else (series.name or "Heatmap"))
    if colorbar:
        ax.figure.colorbar(image, ax=ax, pad=0.02,
                           label=label if label is not None else "kW")
    return ax

def plot_profiles(source, columns=None, ax=None, window=None, resample=None,
                  units=None, title=None):
    """
    Profiles tsib initialized for a building.

    The y label comes from the building's own `units` mapping
    (`Building.units`), so a heating load is labelled kW_th and a PV yield
    kW/kWp without the caller repeating it.

    Parameters
    ----------
    source: Building or pandas.DataFrame, required
    columns: str or list of str, optional
        Which profiles to draw. Defaults to everything in the frame.
    units: dict, optional
        Column -> unit, when `source` is a plain DataFrame.
    resample: str, optional
        Pandas offset alias for a whole-year overview, e.g. "D".
    """
    ax = _ax(ax)
    frame = getattr(source, "timeseries", source)
    units = units if units is not None else getattr(source, "units", {}) or {}

    if columns is not None:
        columns = [columns] if isinstance(columns, str) else list(columns)
        missing = [c for c in columns if c not in frame]
        if missing:
            raise KeyError(
                "No profile(s) {} - available: {}".format(
                    missing, list(frame.columns)
                )
            )
        frame = frame[columns]

    frame = _resample(_window(frame, window), resample)
    cycle = itertools.cycle(_FALLBACK_COLORS)
    for name in frame.columns:
        ax.plot(frame.index, frame[name].values, lw=1.1,
                color=_color(_profile_role(name), cycle), label=name)

    drawn_units = {units[c] for c in frame.columns if c in units}
    ax.set_ylabel("$\\mathrm{{{}}}$".format(drawn_units.pop())
                  if len(drawn_units) == 1 else "")
    ax.set_title(title if title is not None else "Initialized profiles")
    if len(frame.columns) > 1:
        return _legend(ax, ncol=2)
    return ax

def _profile_role(column):
    """Maps a timeseries column name onto a palette role."""
    lowered = column.lower()
    if "photovoltaic" in lowered or "solar" in lowered:
        return "pv"
    if "heat pump" in lowered:
        return "heat_pump"
    if "heating" in lowered or "hot water" in lowered:
        return "heat"
    if "cooling" in lowered:
        return "cool_supply"
    if "electricity" in lowered:
        return "household_load"
    return column
