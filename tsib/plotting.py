# -*- coding: utf-8 -*-
"""
Plots of tsib results and of the profiles tsib initializes.

Every function takes an optional `ax` and returns the axes it drew on, which
is matplotlib's own convention: one function draws one thing, and figures are
composed by the caller.

    fig, (a1, a2) = plt.subplots(2, 1, sharex=True)
    tsib.plotting.plot_dispatch(results, "elec", ax=a1, window=("2010-01-01",
                                                               "2010-01-05"))
    tsib.plotting.plot_zone(results, bdg.cfg, ax=a2, window=…)

Nothing here calls `plt.show()` or touches global state outside `use_style()`
- when a plot ends up in a notebook, a script or a PNG is the caller's
decision.

A year is 8760 points and does not read as a line plot. Four ways out, and
they answer different questions: `window=` to zoom in on a few days,
`resample=` for a whole-year overview, `plot_heatmap` to see every value at
once as a day x hour carpet, and `plot_load_duration` to ask how many hours
sit near the peak - the question a grid connection cares about.
"""

import contextlib
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .optimization.zone5r1c import comfort_bounds


#: colours by the role a component plays, so the same technology keeps the
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
#: (`optimization/parameterization.py`) produces.
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
            tsib.plotting.plot_grid_exchange(results, ax=ax)
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


def _time_index(results):
    """Any datetime index found in a results dict, for series that lack one."""
    for res in results.values():
        if not isinstance(res, dict):
            continue
        frame = res.get("timeseries")
        if frame is not None and isinstance(frame.index, pd.DatetimeIndex):
            return frame.index
        for value in res.values():
            if isinstance(value, pd.Series) and isinstance(
                value.index, pd.DatetimeIndex
            ):
                return value.index
    return None


# ---------------------------------------------------------------------
# derived series - useful without plotting anything
# ---------------------------------------------------------------------


def bus_flows(results, bus, spec=None):
    """
    Every flow on one bus, as a signed frame.

    Positive is energy *into* the bus (generation, import, storage
    discharge), negative is energy *out* of it (demand, export, charging).
    A component doing both nets to one column, so a battery reads positive
    while discharging and negative while charging.

    Components are discovered from the `in_<bus>` / `out_<bus>` keys
    `optimization.results.node_results` produces - nothing here hardcodes a
    component name. The thermal zone is the exception: it reports a
    "timeseries" frame rather than flows, so its heating and cooling loads
    are added from there.

    Parameters
    ----------
    results: dict, required
        As returned by `node_results`.
    bus: str, required
        Bus name, e.g. "elec" or "heat".
    spec: SystemSpec, optional
        Used to find out which buses the thermal zone is attached to.
        Without it the template defaults ("heat"/"cool") are assumed.

    Returns
    -------
    pandas.DataFrame - one column per component, in kW.
    """
    columns = {}
    for name, res in results.items():
        if not isinstance(res, dict):
            continue

        if "timeseries" in res:
            columns.update(_zone_flows(name, res, bus, spec))
            continue

        into = res.get("out_" + bus)
        out_of = res.get("in_" + bus)
        if into is None and out_of is None:
            continue
        series = 0.0
        if into is not None:
            series = series + into
        if out_of is not None:
            series = series - out_of
        columns[name] = series

    if not columns:
        return pd.DataFrame()
    return pd.DataFrame(columns)


def _zone_flows(name, res, bus, spec):
    """The thermal zone's withdrawals, which are not reported as flows."""
    heat_bus, cool_bus = "heat", "cool"
    if spec is not None and name in spec.components:
        params = spec.components[name]
        heat_bus = params.get("heat_bus", heat_bus)
        cool_bus = params.get("cool_bus", cool_bus)

    frame = res["timeseries"]
    if bus == heat_bus and "Heating Load" in frame:
        return {name: -frame["Heating Load"]}
    if bus == cool_bus and "Cooling Load" in frame:
        return {name: -frame["Cooling Load"]}
    return {}


def net_grid_exchange(results, name="grid", bus="elec"):
    """
    Net exchange at the grid connection [kW], import positive.

    This is the one series a grid powerflow computation needs per building.

    Parameters
    ----------
    results: dict, required
    name: str, optional (default: "grid")
        Name of the grid component in the spec.
    bus: str, optional (default: "elec")

    Returns
    -------
    pandas.Series [kW], positive while drawing from the grid.
    """
    if name not in results:
        raise KeyError(
            "No component '{}' in the results - available: {}".format(
                name, sorted(results)
            )
        )
    grid = results[name]
    # the grid connection is a Source plus a Sink merged under one name:
    # "out_<bus>" is the import, "in_<bus>" the export
    imported = grid.get("out_" + bus)
    exported = grid.get("in_" + bus)
    if imported is None and exported is None:
        raise KeyError(
            "Component '{}' has no flows on bus '{}'".format(name, bus)
        )
    if imported is None:
        return -exported
    if exported is None:
        return imported
    return imported - exported


# ---------------------------------------------------------------------
# results plots
# ---------------------------------------------------------------------


def plot_dispatch(results, bus, ax=None, window=None, spec=None, net=True,
                  title=None):
    """
    Every flow on one bus, stacked: supply above zero, demand below.

    Parameters
    ----------
    results: dict, required
    bus: str, required
    ax: matplotlib.axes.Axes, optional
    window: optional - see the module docstring.
    spec: SystemSpec, optional - resolves the thermal zone's buses.
    net: bool, optional (default: True)
        Also draw the net balance, which should sit on zero.
    """
    ax = _ax(ax)
    flows = _window(bus_flows(results, bus, spec=spec), window)
    if flows.empty:
        raise ValueError("No component in the results touches bus '{}'".format(bus))

    index = flows.index
    cycle = itertools.cycle(_FALLBACK_COLORS)

    # a component is labelled once, on whichever side it first appears -
    # otherwise a pure consumer like the heat pump never reaches the legend
    labelled = set()
    colors = {name: _color(name, cycle) for name in flows.columns}

    for sign in (1, -1):
        base = np.zeros(len(index))
        for name in flows.columns:
            values = flows[name].values
            part = np.clip(values, 0, None) if sign > 0 else np.clip(values, None, 0)
            if not np.any(np.abs(part) > 1e-9):
                continue
            ax.fill_between(
                index, base, base + part, step="post", alpha=0.85,
                color=colors[name],
                label=None if name in labelled else name,
                linewidth=0,
            )
            labelled.add(name)
            base = base + part

    if net:
        ax.step(index, flows.sum(axis=1).values, where="post", color="black",
                lw=0.8, ls=":", label="net")

    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("kW")
    ax.set_title(title if title is not None else "Dispatch on '{}'".format(bus))
    return _legend(ax, ncol=3)


def plot_zone(results, cfg, ax=None, window=None, name="thermalzone",
              band=True, title=None):
    """
    Thermal zone temperatures against the comfort band.

    The band is the *effective* one: `comfortT_lb`/`comfortT_ub` are only
    nominal, and `capControl`, `nightReduction` and `occControl` move the
    bounds per time step (`optimization.zone5r1c.comfort_bounds`). With the
    default `nightReduction=True` the floor drops while the occupants sleep,
    which is exactly the room the optimizer uses.

    The zone's heating and cooling loads are not drawn here - they are flows
    on the heat bus, so `plot_dispatch(results, "heat")` shows them together
    with whatever supplies them.

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


def plot_storage(results, ax=None, window=None, index=None, title=None):
    """
    Storage content of every storage in the system.

    solph reports the content on time *points*, one more than there are time
    steps, and without a time index - so the last value is dropped and the
    model's index reattached.
    """
    ax = _ax(ax)
    index = index if index is not None else _time_index(results)
    cycle = itertools.cycle(_FALLBACK_COLORS)
    drawn = 0

    for name, res in results.items():
        if not isinstance(res, dict) or "storage_content" not in res:
            continue
        content = res["storage_content"]
        if index is not None:
            content = pd.Series(content.values[:len(index)], index=index)
        content = _window(content, window)
        ax.plot(content.index, content.values, lw=1.3, color=_color(name, cycle),
                label=name)
        drawn += 1

    if not drawn:
        raise ValueError("No component in the results has a storage content")

    ax.set_ylabel("storage content [kWh]")
    ax.set_title(title if title is not None else "Storage")
    return _legend(ax, ncol=2)


def plot_grid_exchange(results, price=None, ax=None, window=None, name="grid",
                       bus="elec", price_quantile=0.75, show_price=True,
                       title=None):
    """
    Net exchange at the grid connection, with the tariff behind it.

    Import is positive, export negative. Hours at or above
    `price_quantile` of the tariff are shaded, which is what makes a
    load shift visible at a glance.

    Parameters
    ----------
    price: array-like or pandas.Series, optional
        The tariff the building was optimized against.
    price_quantile: float, optional (default: 0.75)
        Threshold for shading the expensive hours.
    show_price: bool, optional (default: True)
        Draw the tariff itself on a secondary axis.
    """
    ax = _ax(ax)
    full = net_grid_exchange(results, name=name, bus=bus)

    # the tariff is aligned to the full horizon before either is narrowed, so
    # a window cannot silently shift the price against the exchange
    if price is not None:
        price = pd.Series(np.asarray(price, dtype=float)[:len(full)],
                          index=full.index)
        threshold = price.quantile(price_quantile)
        price = _window(price, window)
    net = _window(full, window)

    if price is not None:
        expensive = (price >= threshold).values
        span = max(abs(net.min()), abs(net.max())) or 1.0
        ax.fill_between(net.index, -span, span, where=expensive, step="post",
                        color=COLORS["heat"], alpha=0.10, linewidth=0,
                        label="price >= {:.2f}".format(threshold))

    ax.step(net.index, net.values, where="post", color=COLORS["grid"], lw=1.2,
            label="net exchange")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("kW (import > 0)")
    ax.set_title(title if title is not None else "Grid exchange")
    _legend(ax, ncol=2)

    if price is not None and show_price:
        twin = ax.twinx()
        twin.step(net.index, price.values, where="post", color=COLORS["pv"],
                  lw=0.9, alpha=0.8)
        # keep the tariff in the lower band of the axes so it reads as
        # context rather than competing with the exchange itself
        span = (price.max() - price.min()) or 1.0
        twin.set_ylim(price.min() - 0.1 * span, price.max() + 1.5 * span)
        twin.set_ylabel("price [EUR/kWh]")
        twin.grid(False)
    return ax


def plot_price_response(results, price, ax=None, bins=5, name="grid",
                        bus="elec", title=None):
    """
    Mean import power per price bin - did the tariff move the load?

    Mean power rather than energy, because bins hold different numbers of
    hours: a flat bar chart means the tariff changed nothing, a downward
    slope means the building buys when it is cheap. The hour count per bin
    is annotated so a bin resting on few hours is not over-read.
    """
    ax = _ax(ax)
    imported = np.clip(net_grid_exchange(results, name=name, bus=bus).values, 0, None)
    price = np.asarray(price, dtype=float)[:len(imported)]

    edges = np.linspace(price.min(), price.max(), bins + 1)
    # the top edge is inclusive, otherwise the most expensive hour drops out
    which = np.clip(np.digitize(price, edges) - 1, 0, bins - 1)

    means, labels, counts = [], [], []
    for b in range(bins):
        mask = which == b
        means.append(imported[mask].mean() if mask.any() else 0.0)
        counts.append(int(mask.sum()))
        labels.append("{:.2f}\n{:.2f}".format(edges[b], edges[b + 1]))

    positions = np.arange(bins)
    ax.bar(positions, means, color=COLORS["grid"], alpha=0.85, width=0.7)
    for pos, mean, count in zip(positions, means, counts):
        ax.annotate("{} h".format(count), (pos, mean), ha="center",
                    va="bottom", fontsize=7, color=COLORS["secondary"])

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_xlabel("price bin [EUR/kWh]")
    ax.set_ylabel("mean import [kW]")
    ax.set_title(title if title is not None else "Price response")
    return ax


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


# ---------------------------------------------------------------------
# the initialization side
# ---------------------------------------------------------------------


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
