# Energy system optimization in tsib

`tsib.optimization` builds [oemof-solph](https://oemof-solph.readthedocs.io/) energy systems out
of parameterized buildings. tsib contributes exactly two things to solph:

1. **the 5R1C thermal zone** — the one piece of building physics solph has no equivalent for;
2. **a building block kit** — specs, presets and component factories that turn a resolved building
   configuration into a ready-to-solve system.

Everything else (buses, grid, PV, storage, heat pumps, metering, investment, the solver interface)
comes from solph.

The design rule that drives the whole thing: **the 5R1C zone is a component, not a precomputed
load profile.** Its temperature states are free decision variables inside a comfort band, so the
building's thermal mass is a dispatchable flexibility resource in the same solve as storage, PV
and prices. Reducing the zone to a fixed heat demand series and optimizing dispatch afterwards
would throw that flexibility away.

- [1. Minimal example](#1-minimal-example)
- [2. The building block kit](#2-the-building-block-kit)
- [3. The 5R1C component](#3-the-5r1c-component)
- [4. Tutorial: adding a component to the kit](#4-tutorial-adding-a-component-to-the-kit)
- [5. Gotchas](#5-gotchas)

---

## 1. Minimal example

```python
import tsib
from tsib.optimization import presets, build_system, solve, node_results

bdg = tsib.Building(configurator=tsib.BuildingConfiguration({...}))

results = bdg.optimize(presets.hp_pv_battery(pv_kwp=8.0, battery_kwh=10.0))

results["thermalzone"]["timeseries"]["Heating Load"]   # pandas Series [kW]
results["pv"]["out_elec"]                              # generation [kW]
results["battery"]["storage_content"]                  # SOC [kWh]
```

`Building.optimize()` is the orchestration entry point: it pairs a spec with the building's
resolved configuration, so the *same spec* applies to any parameterized building.

Without a `Building`, the three steps are explicit:

```python
es, nodes = build_system(presets.heat_load_only(), cfg)   # cfg: resolved building config
model, _  = solve(es)
results   = node_results(model, nodes)
```

Units throughout: **kW**, **kWh**, **EUR**, **°C**, **hours**.

---

## 2. The building block kit

### Specs and presets are the same thing

A **`SystemSpec`** describes topology plus scalar parameters. A **preset** is a function returning
a spec. There is exactly one build path, so a preset can be inspected, serialized and modified
before it is built:

```python
spec = presets.hp_pv_battery(pv_kwp=8.0)
spec.add_component("wallbox", "demand", bus="elec", profile="@evLoad")
es, nodes = build_system(spec, cfg)
```

A spec is plain data:

```python
{
  "buses": {"elec": {"carrier": "electricity"}, "heat": {"carrier": "heat"}},
  "components": {
    "grid":   {"type": "grid",      "bus": "elec", "import_price": 0.35, "export_price": 0.08},
    "hp":     {"type": "heat_pump", "bus_in": "elec", "bus_out": "heat", "cop": "@cop"},
    "demand": {"type": "demand",    "bus": "elec", "profile": "@elecLoad"},
    "zone":   {"type": "zone5r1c",  "heat_bus": "heat", "cool_bus": "cool"}
  }
}
```

**Time series never live in the spec.** A `"@key"` string is resolved against the building
configuration at build time. That boundary is deliberate: it keeps a spec small enough to store,
diff and vary across thousands of buildings, and it is what makes `SystemSpec.to_json()` round-trip.
Scalars pass through untouched, so `profile=1.0` is a valid constant 1 kW demand.

Three of the keys the presets reference by default are not produced by `BuildingConfiguration`
itself, so `Building.optimize` supplies them: `"@cop"` and `"@pv_yield"` from the renewable
simulation it already runs (the heat pump COP and the specific PV yield in kW/kWp), and
`"@elecPrice"` from `presets.DEFAULT_ELEC_PRICE`. Set any of them on `bdg.cfg` — a dynamic tariff,
a measured yield — and yours is used instead. Calling `build_system` directly gives you no such
fallbacks: put the profiles in the cfg dict yourself.

### The kit

Every factory but the last returns stock solph objects:

| spec `type` | solph | notes |
|---|---|---|
| `demand` | `Sink(Flow(fix=…, nominal_capacity=1))` | inflexible load |
| `source` | `Source(Flow(variable_costs=…))` | generic priced supply |
| `grid` | `Source` **+** `Sink` | two nodes: solph prices directed edges |
| `meter` | `Converter` between two buses | the §14a sub-metering primitive |
| `pv` | `Source(Flow(maximum=yield, …))` | `curtailable=False` uses `fix=` instead |
| `heat_pump` | `Converter(conversion_factors={bus_in: 1/COP})` | couples two carriers |
| `battery` | `GenericStorage` | `balanced=True` ≙ periodic SOC wrap |
| `thermal_storage` | `GenericStorage(fixed_losses_absolute=…)` | standby loss |
| `zone5r1c` | **`ThermalZone5R1C`** | the one custom component |

Capacities are either a number (fixed) or investment parameters:

```python
spec.add_component("pv", "pv", bus="elec", specific_yield="@pv_yield",
                   capex_per_unit=1200.0, lifetime=25.0, max_capacity=20.0, wacc=0.06)
```

The annuity and the horizon scaling stay on the tsib side, because solph expects an
already-annualized `ep_costs`:

$$\text{ep\_costs} = \text{capex}\cdot\big(a(n,i) + \text{opex}_{\text{fix}}\big)\cdot
\underbrace{\tfrac{H}{8760}}_{\text{year fraction}}, \qquad
a(n,i) = \frac{i(1+i)^n}{(1+i)^n - 1}$$

The year fraction is why a 72-hour design study still trades capex against opex correctly.

### Results

`node_results(model, nodes)` returns one dict per *kit component*, not per solph node — a grid
connection's import Source and export Sink are merged under the name `"grid"`:

- `in_<bus>` / `out_<bus>` — flows [kW]
- `capacity` — invested capacity, where an investment was declared
- `storage_content` — SOC [kWh] for storages
- the thermal zone instead reports `timeseries` / `static` / `refurbishment`, the shape the
  pre-migration model produced, so `Building`'s CSV export and TinyDB cache are unaffected.

---

## 3. The 5R1C component

Three temperature nodes — mass $T_m$ (the only one with capacity $C_m$), surface $T_s$, air
$T_{air}$ — per DIN EN ISO 13790 / Schuetz et al. 2017.

**The heat supply is not a variable of the component.** It is `m.flow[heat_bus, zone, t]`, the
edge solph creates for the zone's input flow. That edge *is* the coupling mechanism, and the bus
balance ties it to whatever supplies it.

**Mass node** (the difference equation, hence the thermal storage behaviour):

$$H_{ms}(T_{m,t} - T_{s,t}) + \!\!\sum_{e \in \{W,R,F\}}\!\! Q_{e,t} + H_{door}(T_{m,t} - T_{e,t})
= \Phi^m_t - C_m \frac{T_{m,t+1} - T_{m,t}}{\Delta t}$$

**Surface node:**

$$H_{ms}(T_{s,t} - T_{m,t}) + H_{is}(T_{s,t} - T_{air,t}) + Q_{\text{Win},t} = \Phi^s_t$$

**Air node**, where heating and cooling enter:

$$Q_{\text{Vent},t} + H_{is}(T_{air,t} - T_{s,t}) = \Phi^s_t - Q_{\text{cool},t} + Q_{\text{heat},t}$$

with gains split over the nodes (eq. 15/16):

$$\Phi^m_t = \tfrac{1}{2}\tfrac{A_m}{A_{tot}} Q_{ig,t} + \tfrac{A_f}{A_{tot}} Q_{sol,t}$$

$$\Phi^s_t = \Big(1 - \tfrac{U_{win}}{h_{ms} A_{tot}}\Big)\tfrac{Q_{ig,t}}{2}
- \tfrac{\text{cross}_t}{h_{ms} A_{tot}} + Q_{sol,t} - \Phi^m_t$$

**Envelope flows** are $H_e \cdot (T_m - T_e)$ with the heat transfer coefficient $H_e$ [kW/K] of
the installed construction, read from the refurbishment catalog in `envelope.py`. That catalog is
load-bearing even though refurbishment optimization is not available: it also carries the
*existing* construction and the option-dependent solar gain series.

**Comfort band** — this is the flexibility. $T_{air}$ is a free variable inside a band that the
installed control equipment widens:

$$T_{air,t} \le T_{lb} + (T_{ub} - T_{lb})\,x_{\text{smart}} - (T_{ub} - 30)\,\text{away}_t\,x_{\text{occ}}$$
$$T_{air,t} \ge T_{lb} - (T_{lb} - 18)\,\text{asleep}_t\,x_{\text{night}} - (T_{lb} - 14)\,\text{away}_t\,x_{\text{occ}}$$

Without a smart thermostat ($x_{\text{smart}} = 0$) the band collapses and $T_{air}$ is pinned at
$T_{lb}$ — no flexibility, which is exactly the reference case the flexibility tests compare
against.

**Max load** is a *soft* constraint (a violation variable penalized at 100 EUR/kW) so an undersized
system warns instead of going infeasible. It is the component's only objective term; energy cost
belongs to whatever supplies the bus.

> **Four inherited deviations from the source papers, all kept deliberately:** envelope flows are
> driven by $(T_m - T_e)$ even for windows and ventilation, which touch other nodes; the air-node
> balance uses the surface gain $\Phi^s$ rather than half the internal gains; and the hard comfort
> ceiling forces fictitious cooling in buildings that have no cooling device. All are load-bearing
> for the validated ~197 kWh/m²/a result. **Evidence and measured impact:
> [`model-deviations.md`](model-deviations.md).**

### Not available: refurbishment optimization

Envelope and control *investment* decisions were not migrated. `refurbishment=True` raises
`NotImplementedError` rather than being silently ignored. The Big-M machinery
(`DiscreteOptionInvestment.switched_flow`, the window×solar linearization) still lives in
`optimization/investment.py` for that follow-up.

---

## 4. Tutorial: adding a component to the kit

Most additions are **one factory function**, because solph already has the component. A
combined heat and power unit, for instance:

```python
# tsib/optimization/registry.py

@factory("chp")
def _chp(name, params, buses, cfg, n_steps, step_size_h):
    """Gas-fired CHP: one fuel input, electricity and heat output."""
    fuel = _bus(buses, params, "bus_fuel", name)
    elec = _bus(buses, params, "bus_elec", name)
    heat = _bus(buses, params, "bus_heat", name)
    wacc = params.pop("wacc", 0.0)
    capacity = _capacity(params, n_steps * step_size_h, wacc)
    return [
        solph.components.Converter(
            label=name,
            inputs={fuel: solph.Flow(nominal_capacity=capacity)},
            outputs={elec: solph.Flow(), heat: solph.Flow()},
            conversion_factors={
                elec: params.pop("electrical_efficiency", 0.35),
                heat: params.pop("thermal_efficiency", 0.50),
            },
        )
    ]
```

That is the whole change. The factory is picked up by `build_component`, so
`spec.add_component("chp", "chp", bus_fuel="gas", bus_elec="elec", bus_heat="heat")` works
immediately, including profile references and investment parameters.

### When you actually need a custom solph component

Only when the physics is not expressible as flows between buses — free state variables, coupled
algebraic nodes, bounds on something that is not a flow. The 5R1C zone is the only such case in
tsib.

A custom component is always **two classes that have to find each other**: a node that hangs in
the graph, and a block that holds the pyomo variables and constraints for *all* nodes of that
type. `constraint_group()` is the link. Derive them from the bases in
[`base.py`](../tsib/optimization/base.py) rather than from `Node`/`ScalarBlock` directly:

```python
from tsib.optimization import TsibComponent, TsibBlock


class MyComponent(TsibComponent):
    def __init__(self, label, bus, **params):
        super().__init__(label=label, inputs={bus: Flow()})
        ...
    def constraint_group(self):
        return MyComponentBlock


class MyComponentBlock(TsibBlock):
    def _create(self, group=None):
        if group is None:
            return
        m = self.parent_block()
        self.UNITS = po.Set(initialize=list(group), ordered=True)
        self.state = po.Var(self.UNITS, m.TIMESTEPS, within=po.Reals)
        self.balance = po.Constraint(self.UNITS, m.TIMESTEPS, rule=...)

    def _objective_expression(self):
        return sum(...)
```

`TsibBlock` carries `CONSTRAINT_GROUP = True`, so the attribute solph looks for cannot be
forgotten, and both bases declare their central method abstract — a node without
`constraint_group()` or a block without `_create()` fails at instantiation instead of at solve
time. If you bypass the bases anyway, `build_system` still refuses to return a system whose
custom block would be ignored (`assert_constraint_groups`).

Read the bus coupling as `m.flow[bus, node, t]` (into the node) or `m.flow[node, bus, t]` (out of
it); do not create your own flow variables.

Then export it in `tsib/optimization/__init__.py`, add a `@factory` wrapping it, and test it
alongside `test/test_optimization_core.py`.

---

## 5. Gotchas

- **`CONSTRAINT_GROUP = True` is mandatory on a custom block.** `solph.Model.__init__` collects
  custom constraint groups with `if hasattr(i, "CONSTRAINT_GROUP")`, on top of the stock blocks
  already listed in `Model.CONSTRAINT_GROUPS`. Without the attribute your block is **silently
  ignored** — the model solves, the objective looks plausible, and your component imposes no
  constraints whatsoever. This is the single nastiest failure mode here, which is why
  `TsibBlock` carries the attribute and `build_system` rejects a block that lacks it. Note the
  test is `hasattr`, not truth: `CONSTRAINT_GROUP = False` switches nothing off.
- **`infer_last_interval=True` is mandatory.** solph 0.6 defaults it to `False`, which turns 8760
  time stamps into 8759 intervals and silently drops the last hour of the year. `build_system`
  sets it; if you construct an `EnergySystem` by hand, you must too.
- **Do not use `solph.Model.solve()`.** The 5R1C model is badly conditioned: `glpk` fails on it
  outright, and HiGHS needs the interior point method with crossover switched off. That tuning
  lives in `solverutils.py` and is applied by `tsib.optimization.solve()`, which works because
  `solph.Model` is a pyomo `ConcreteModel`.
- **`Flow()` without `nominal_capacity` is unbounded above but non-negative** — solph sets
  `lb = 0` for unidirectional flows and no upper bound. That matches the old `NonNegativeReals`.
- **A bus with no supply fails silently**, exactly as before: the balance just forces every
  withdrawal to zero. If a component mysteriously does nothing, check that something injects into
  every bus it touches. `oemof.network.graph.create_nx_graph(es)` gives you the topology to check.
- **`max` is deprecated in favour of `maximum`** in solph 0.6 flows; using both raises a
  `FutureWarning` and one silently overwrites the other.
- **Storages are periodically balanced by default** (`balanced=True`), reproducing the old
  periodic SOC wrap. Passing `initial_soc` switches it off.
- **Degenerate optima are real.** With a lossless storage or a constant price, shifting energy is
  cost-neutral and the solver may return any of many optima. Tests that assert on a *trajectory*
  need a strictly convex setup (lossy storage, varying price); tests on aggregates do not.
- **Solve time is dominated by binaries.** A fixed-envelope full year is an LP and solves in ~8 s.
- **Set `$SOLVER=highs`** unless you have gurobi/cplex.
- **"Maximal heat load exceeded" warnings are the soft constraint doing its job** — the design
  system is slightly undersized in some hours, exactly as in the old model.

## Where things live

```
tsib/optimization/
├── base.py         TsibComponent/TsibBlock bases for custom components
├── zone5r1c.py     ThermalZone5R1C + block, and zone_results()   <- the only custom physics
├── config.py       ThermalZoneConfig, calc_surface_irradiance
├── envelope.py     refurbishment/existing-construction catalog from the cost Excel
├── control.py      smart thermostat / occupancy / night reduction
├── investment.py   annuity, Continuous/DiscreteOptionInvestment (Big-M kept for the port)
├── spec.py         SystemSpec, build_system, profile resolution
├── registry.py     component factories -> stock solph objects
├── presets.py      heat_load_only, hp_pv_battery
├── results.py      solve() and node_results()
└── solverutils.py  solver detection and per-solver tuning
```

Reference: `examples/energysystem/EnergySystemDemo.ipynb` and the test suite, which doubles as
executable spec — `test_optimization_core.py` for the kit contract,
`test_optimization_dispatch.py` for the flexibility proof, `test_optimization_zone.py` for the
IWU reference and the migration parity against `test/data/golden/`.
