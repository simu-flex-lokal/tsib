# The building's energy system

tsib describes buildings and simulates their time series. It does **not** dispatch or size
equipment: the energy system model was split out into
[`esmkit`](https://github.com/simu-flex-lokal/esmkit), and tsib depends on it in no way — no
import, no solver, no `oemof-solph`. What tsib produces instead is a complete, serializable
*description* of a building's energy system, which esmkit or any other engine can consume.

This document is about that handover. Runnable:
[`examples/buildingsystem/system_export_demo.py`](../examples/buildingsystem/system_export_demo.py).

## Three descriptions of one building

|  | answers | example |
|---|---|---|
| `BuildingConfiguration` → `cfg` | what the building **is** | archetype, U-values, floor area, weather, occupants |
| `BuildingSystemParameters` | what the building **has** | `{"heat_pump": {"capacity_kw": 8.0}, "pv": {"kwp": 8.0}}` |
| the spec (`build_spec`) | how it is **wired** | buses, components, the zone's physics |

The configuration determines the building's *demand* and is what its cache identity is derived
from. The sheet determines how that demand is *served* — it changes no profile, which is why
scenarios can vary it freely while the expensive simulated profiles stay valid. The spec is
derived from both and is the thing that leaves.

```python
sheet  = tsib.BuildingSystemParameters(equipment={"heat_pump": {"capacity_kw": 8.0}})
spec   = bdg.system_spec(sheet)            # a plain dict
inputs, index = bdg.system_inputs(spec)    # the arrays it refers to
```

## The spec

Plain JSON: topology plus scalars, never a time series. Every time-varying parameter is a
`"@key"` reference resolved against the inputs mapping by whoever builds the model.

```json
{"version": "1",
 "buses": {"elec": {"carrier": "electricity"}, "heat": {"carrier": "heat"},
           "cool": {"carrier": "cool"}},
 "components": {
   "grid":           {"type": "grid", "bus": "elec", "import_price": "@elecPrice",
                      "export_price": 0.08},
   "household_load": {"type": "demand", "bus": "elec", "profile": "@elecLoad"},
   "dhw_load":       {"type": "demand", "bus": "heat", "profile": "@hotWaterLoad"},
   "heat_pump":      {"type": "heat_pump", "bus_in": "elec", "bus_out": "heat",
                      "cop": "@cop", "capacity": 8.0},
   "thermalzone":    {"type": "zone5r1c", "heat_bus": "heat", "cool_bus": "cool",
                      "C_m": 8.42, "H_ms": 1.97, "T_e": "@T_e", "...": "..."}}}
```

`required_inputs(spec)` lists every `"@key"` in it — the data contract, inspectable before
anything is built. `Building.system_inputs(spec)` answers exactly that list, simulating
occupancy and the renewable potentials only if the spec asks for them.

Every building shares one template, so a few hundred buildings of a synthetic grid are
described by parameters instead of hand-authored topologies:

```
grid ─── elec ──┬── household_load
                ├── pv, battery          (if the sheet has them)
                └── heat_pump            (if the sheet has one)
                        │
            heat ───────┼── buffer       (if the sheet has one)
                        ├── dhw_load
                        └── thermalzone ─── cool
```

Space heating and hot water share **one** `heat` bus: a deliberate simplification, so the two
are not distinguishable by temperature level. A building whose sheet has no heat-supplying
equipment gets a priced non-electric `heat_supply` source instead of an infeasible model — a
gas-heated building still belongs in a grid study, it just contributes household load, PV and
battery rather than heat pump electricity.

Adding equipment is one registered builder; see
[`parameters.md` §9](parameters.md#9-the-equipment-sheet).

## The zone contract

The one piece of real physics in the handover. `tsib.envelope.zone_parameters(cfg)` condenses
the whole building into **ten scalars and five time series**, computed without solving
anything:

| scalars | |
|---|---|
| `H_ms`, `H_is`, `H_door` | mass↔surface, surface↔air and door conductances [kW/K] |
| `C_m` | heat capacity of the thermal mass [kWh/K] |
| `H` | per-element conductance for Walls, Roof, Floor, Windows, Ventilation [kW/K] |
| `max_load` | ceiling of the heating/cooling system [kW], default the design heat load |
| `design_capacity` | design heat load of the envelope [kW], reported in the results |

| series | |
|---|---|
| `T_e` | ambient temperature [°C] |
| `gain_mass`, `gain_surface` | solar and internal gains on the two nodes [kW] |
| `comfort_lb`, `comfort_ub` | the *effective* comfort band [°C] |

Everything building-shaped happens on the tsib side of this line: the envelope heat transfer
coefficients (`envelope/elements.py`), the plane-of-array irradiance and the derived 5R1C
parameters (`envelope/config.py`), the distribution of gains over the nodes
(`envelope/gains.py`), and the comfort band implied by the installed controls
(`envelope/comfort.py`). Downstream there is no building left — ten numbers and five arrays.

The band is *effective*, not nominal: `comfortT_lb`/`comfortT_ub` are only the starting point,
and `capControl`, `nightReduction` and `occControl` move the bounds per time step. Plot it with
`tsib.plotting.plot_zone`, which calls the same function the model does, so the picture cannot
drift from the constraints.

This contract is pinned by `test/test_envelope_contract.py` against a capture taken while the
MILP zone still lived in tsib — if a term of the gain equations drifts, that test fails without
any model being built.

## Where the model went

| what | where it lives now |
|---|---|
| spec schema, factories, `build_system`, `solve`, `node_results` | `esmkit.core` |
| stock components (grid, PV, storage, heat pump, demand, meter) | `esmkit.components.stock` |
| the 5R1C zone as a solph component | `esmkit.components.zone5r1c` |
| dispatch / storage / grid-exchange plots | `esmkit.plotting` |
| tsib ↔ esmkit, and the check that they still agree | `orchestrator/`, `check_contract.py` |

The heat load itself is affected: `Building.getHeatLoad()` was a MILP solve and currently
raises `NotImplementedError`. The forward, non-optimizing 5R1C that replaces it is the next
piece of work — see `backlog/20260820_esmkit-split.md`. Everything it needs is already here;
`zone_parameters()` is its input, and `test/data/golden/zone_year.csv.gz` its acceptance target.
