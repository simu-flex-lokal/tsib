# -*- coding: utf-8 -*-
"""
Contract of the building block kit: the spec, the component registry and
the builder.

These tests are the executable specification of how a system is described
and assembled - deliberately without a thermal zone, so they stay fast and
isolate the kit from the building physics.
"""

import numpy as np
import pandas as pd
import pyomo.environ as po
import pytest
from oemof import solph

from tsib.optimization import (
    COMPONENT_FACTORIES,
    SystemSpec,
    annuity_factor,
    build_system,
    flow_series,
    node_results,
    objective_value,
    presets,
    resolve_profile,
    solve,
)


def simple_cfg(n=24, **extra):
    """Minimal configuration: only a weather index and referenced profiles."""
    index = pd.date_range("2010-01-01", periods=n, freq="h")
    cfg = {"weather": pd.DataFrame({"T": np.zeros(n)}, index=index)}
    cfg.update(extra)
    return cfg


# --- spec -------------------------------------------------------------


def test_spec_round_trip():
    """A spec is plain data: it survives a JSON round trip unchanged."""
    spec = SystemSpec()
    spec.add_bus("elec", carrier="electricity")
    spec.add_component("grid", "grid", bus="elec", import_price=0.3)
    spec.add_component("demand", "demand", bus="elec", profile="@elecLoad")

    restored = SystemSpec.from_json(spec.to_json())
    assert restored.to_dict() == spec.to_dict()


def test_spec_rejects_duplicates():
    spec = SystemSpec()
    spec.add_bus("elec")
    with pytest.raises(ValueError, match="Duplicate bus"):
        spec.add_bus("elec")
    spec.add_component("grid", "grid", bus="elec")
    with pytest.raises(ValueError, match="Duplicate component"):
        spec.add_component("grid", "grid", bus="elec")


def test_spec_copy_is_deep():
    spec = SystemSpec()
    spec.add_bus("elec")
    spec.add_component("grid", "grid", bus="elec", import_price=0.3)
    clone = spec.copy()
    clone.components["grid"]["import_price"] = 9.9
    assert spec.components["grid"]["import_price"] == 0.3


def test_presets_return_specs():
    """Presets are named specs, not built systems - that is what makes
    them serializable and modifiable before building."""
    assert isinstance(presets.heat_load_only(), SystemSpec)
    assert isinstance(presets.hp_pv_battery(), SystemSpec)


# --- profile resolution ------------------------------------------------


def test_profile_reference_resolution():
    cfg = simple_cfg(n=5, load=pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]))
    values = resolve_profile("@load", cfg, 5, "test")
    assert isinstance(values, np.ndarray)
    assert values.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_scalars_pass_through():
    """Scalars stay scalars so solph broadcasts them itself."""
    assert resolve_profile(0.35, simple_cfg(), 24, "test") == 0.35


def test_profile_errors_are_specific():
    cfg = simple_cfg(n=5, load=pd.Series([1.0, 2.0]))
    with pytest.raises(KeyError, match="missing"):
        resolve_profile("@missing", cfg, 5, "test")
    with pytest.raises(ValueError, match="has 2 values, expected 5"):
        resolve_profile("@load", cfg, 5, "test")
    with pytest.raises(ValueError, match="must be a configuration reference"):
        resolve_profile("elecLoad", cfg, 5, "test")


# --- registry ----------------------------------------------------------


def test_registry_covers_the_kit():
    expected = {
        "demand", "source", "grid", "meter", "pv",
        "heat_pump", "battery", "thermal_storage", "zone5r1c",
    }
    assert expected <= set(COMPONENT_FACTORIES)


def test_unknown_component_type_lists_alternatives():
    spec = SystemSpec()
    spec.add_bus("elec")
    spec.add_component("thing", "wind_turbine", bus="elec")
    with pytest.raises(ValueError, match="Unknown component type 'wind_turbine'"):
        build_system(spec, simple_cfg())


def test_unknown_bus_is_rejected():
    spec = SystemSpec()
    spec.add_bus("elec")
    spec.add_component("grid", "grid", bus="heat", import_price=0.3)
    with pytest.raises(ValueError, match="unknown bus 'heat'"):
        build_system(spec, simple_cfg())


def test_missing_bus_parameter_is_rejected():
    spec = SystemSpec()
    spec.add_bus("elec")
    spec.add_component("grid", "grid", import_price=0.3)
    with pytest.raises(ValueError, match="needs a 'bus'"):
        build_system(spec, simple_cfg())


def test_grid_expands_into_two_nodes():
    """A grid connection with feed-in becomes a Source and a Sink, because
    solph prices directed edges."""
    spec = SystemSpec()
    spec.add_bus("elec")
    spec.add_component("grid", "grid", bus="elec", import_price=0.3, export_price=0.08)
    spec.add_component("demand", "demand", bus="elec", profile=1.0)

    _, nodes = build_system(spec, simple_cfg())
    assert isinstance(nodes["grid"], list) and len(nodes["grid"]) == 2
    assert {str(n.label) for n in nodes["grid"]} == {"grid_import", "grid_export"}


def test_grid_without_export_is_one_node():
    spec = SystemSpec()
    spec.add_bus("elec")
    spec.add_component("grid", "grid", bus="elec", import_price=0.3)
    spec.add_component("demand", "demand", bus="elec", profile=1.0)

    _, nodes = build_system(spec, simple_cfg())
    assert not isinstance(nodes["grid"], list)


# --- builder / time index ----------------------------------------------


def test_timeindex_covers_every_step():
    """solph 0.6 defaults to infer_last_interval=False, which would turn
    N time stamps into N-1 intervals and silently drop the last step."""
    spec = SystemSpec()
    spec.add_bus("elec")
    spec.add_component("grid", "grid", bus="elec", import_price=0.3)
    spec.add_component("demand", "demand", bus="elec", profile=1.0)

    es, _ = build_system(spec, simple_cfg(n=8760))
    assert len(es.timeincrement) == 8760


def test_annuity_factor():
    assert annuity_factor(20, 0.0) == pytest.approx(1.0 / 20)
    assert annuity_factor(20, 0.05) == pytest.approx(0.080242, abs=1e-6)


# --- solving -----------------------------------------------------------


def test_bus_balance_and_cost():
    """A fixed demand supplied by a priced grid: the balance forces the
    import and the objective is the priced energy."""
    spec = SystemSpec()
    spec.add_bus("elec", carrier="electricity")
    spec.add_component("grid", "grid", bus="elec", import_price=0.30)
    spec.add_component("demand", "demand", bus="elec", profile=2.0)

    es, nodes = build_system(spec, simple_cfg(n=24))
    model, _ = solve(es)

    results = node_results(model, nodes)
    assert results["grid"]["out_elec"].sum() == pytest.approx(48.0)
    assert results["demand"]["in_elec"].sum() == pytest.approx(48.0)
    assert objective_value(model) == pytest.approx(48.0 * 0.30)


def test_meter_is_a_lossless_transfer():
    """The 14a sub-metering primitive: one flow, two buses, priced once."""
    spec = SystemSpec()
    spec.add_bus("grid_bus")
    spec.add_bus("steuerbar")
    spec.add_component("supply", "source", bus="grid_bus", price=0.0)
    spec.add_component(
        "meter", "meter", bus_up="grid_bus", bus_down="steuerbar", price=0.10
    )
    spec.add_component("demand", "demand", bus="steuerbar", profile=3.0)

    es, nodes = build_system(spec, simple_cfg(n=24))
    model, _ = solve(es)

    results = node_results(model, nodes)
    assert results["meter"]["in_grid_bus"].sum() == pytest.approx(72.0)
    assert results["meter"]["out_steuerbar"].sum() == pytest.approx(72.0)
    # metered throughput is the base the reduced grid fee is charged on
    assert objective_value(model) == pytest.approx(72.0 * 0.10)


def test_node_results_shape():
    spec = SystemSpec()
    spec.add_bus("elec")
    spec.add_component("grid", "grid", bus="elec", import_price=0.3)
    spec.add_component("demand", "demand", bus="elec", profile=1.5)

    es, nodes = build_system(spec, simple_cfg(n=12))
    model, _ = solve(es)
    results = node_results(model, nodes)

    assert set(results) == {"grid", "demand"}
    assert results["demand"]["in_elec"].sum() == pytest.approx(18.0)
