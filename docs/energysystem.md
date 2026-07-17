# The tsib energy system framework

`tsib.energysystem` is a small component-based framework that builds one pyomo MILP out of
components plugged onto carrier buses. It replaced the monolithic `Building5R1C` model.

The design rule that drives everything else: **the 5R1C thermal zone is a component like any
other, solved in the same MILP as storage, PV and prices.** The comfort band plus the building's
thermal mass are structurally a thermal storage, so they are a dispatchable flexibility resource.
Reducing the zone to a fixed heat-load time series and optimizing dispatch afterwards would throw
that flexibility away.

- [1. Minimal example](#1-minimal-example)
- [2. How it works](#2-how-it-works)
- [3. The math](#3-the-math)
- [4. Tutorial: adding a heat pump](#4-tutorial-adding-a-heat-pump)
- [5. Gotchas](#5-gotchas)

---

## 1. Minimal example

A demand supplied by a priced grid and a battery. This is the whole API surface you need for a
dispatch model:

```python
import numpy as np
import pandas as pd
import tsib.energysystem as es

times = pd.date_range("2010-01-01", periods=48, freq="h", tz="UTC")
price = pd.Series(np.where(times.hour < 12, 0.05, 0.40), index=times)   # EUR/kWh

model = es.EnergySystemModel(times, wacc=0.06)
elec = model.add(es.Bus("elec", carrier="electricity"))

model.add(es.FixedDemand("demand", elec, 1.0))                 # kW, scalar broadcast
model.add(es.GridConnection("grid", elec, import_price=price))
model.add(es.ElectricalStorage("battery", elec, capacity=10.0))  # kWh

model.solve()

print(model.objective_value)                 # EUR over the horizon
print(model.results("battery")["soc"])       # pandas Series
print(model.results())                       # dict over all components
```

`model.add()` returns what you passed in, so `bus = model.add(es.Bus(...))` is the idiom.
`solve()` builds the model on first call; `build()` separately if you want to inspect
`model.pyomo_model` first. Solver is `$SOLVER`, else auto-detected
(gurobi → cplex → scip → cbc → highs).

Units throughout: **kW** for power/flows, **kWh** for energy/capacity, **EUR**, **°C**, **hours**.

---

## 2. How it works

Three concepts:

- **`Bus`** — a balance node for one carrier. Knows nothing about components.
- **`Component`** — owns exactly one `pyomo.Block` (its private namespace) and declares its
  couplings via **`Port`** objects. Knows nothing about other components.
- **`EnergySystemModel`** — assembles blocks, builds the bus balances, sums the objective.

Components never reference each other. Everything they exchange goes through a bus, which is why
adding a component is a purely local change.

### What `build()` does, in order

| # | Step | Your hook |
|---|------|-----------|
| 1 | Reject free investments if the strategy doesn't support them | `has_free_investment()` |
| 2 | Create a `ConcreteModel`, one `Block` per component, named after it | — |
| 3 | Ask every component for its ports and register them at their buses | `ports()` |
| 4 | Pass 1 — parameters, no pyomo objects | `build_parameters(model, block)` |
| 5 | Pass 2 — all variables | `build_variables(model, block)` |
| 6 | Pass 3 — all constraints | `build_constraints(model, block)` |
| 7 | Per bus: add `<bus>_balance` over all registered ports | — |
| 8 | Strategy state coupling | `strategy.apply_state_constraints` |
| 9 | Objective = sum of every component's cost terms | `objective_terms(model, block)` |

The three passes exist so constraints can rely on *all* variables of *all* components already
existing. Port flow callables are invoked in step 7, i.e. after every pass, so they may freely
reference anything a component built.

### The `Component` interface

Everything except `ports()` has a working default — a component can be as small as the
`FixedDemand` (30 lines) or as large as `ThermalZone5R1C` (850).

```python
class Component:
    def build_parameters(self, model, block): pass   # precompute arrays, no pyomo
    def build_variables(self, model, block): pass    # block.x = pyomo.Var(...)
    def build_constraints(self, model, block): pass  # block.c = pyomo.Constraint(...)
    def objective_terms(self, model, block): return []   # list of EUR expressions
    def ports(self): ...                             # REQUIRED: {name: Port(bus, flow, sign)}
    def extract_results(self, block): return {}      # solved values -> pandas/python

    # state hooks, only needed if the component has memory across steps
    def state_variables(self, block): return {}      # {"soc": block.soc}
    def initial_state(self): return None             # {"soc": 5.0} or None
    def final_state(self, block): ...                # default: last step of state_variables
```

A `Port(bus, flow, sign)` declares a coupling. `flow` is a callable `(block, t) -> expression`
returning a **nonnegative** flow; `sign=+1` injects into the bus, `sign=-1` withdraws. A component
with two ports (a heat pump, an electrolyzer) is what couples two carriers.

### Investments are composed, not inherited

An investment is an object you *attach* (`self.investment = ...`), not a base class you subclass:

- **`ContinuousInvestment(capex_per_unit, lifetime, opex_fix_share=0, capacity=None, min_capacity=0, max_capacity=None)`**
  — a capacity. Pass `capacity=` and it's **fixed**; omit it and it's a decision variable.
- **`DiscreteOptionInvestment(name, options, fixed_option=None)`** — choose exactly one from a
  catalog (`{name: {"capex":…, "lifetime":…, "opex_fix_share":…, …payload}}`). Pass
  `fixed_option=` and it's **fixed**.

The payoff is `is_fixed`: a fixed investment creates **zero** variables and **zero** Big-M
constraints, so a fully parameterized model stays a pure LP. This is the single mechanism behind
"no investment", manual scenarios, and freezing a perfect-foresight decision before a later
dispatch-only run. Components like `PVGenerator` and `StorageComponent` accept `capacity=` as
either a float or a `ContinuousInvestment` and wrap the float in a fixed investment internally, so
there is only one code path.

### Strategies

`PerfectForesightStrategy` (the default) does one solve over the whole horizon and closes state
variables periodically (`wraps_state = True`). `MyopicStrategy` and `RollingHorizonStrategy` are
deliberate stubs that raise `NotImplementedError` — the interface they need (`TimeIndex.slice()`,
`initial_state()`/`final_state()`) already exists, so implementing them shouldn't touch any
component. `build()` refuses to mix a free investment with a strategy that returns
`supports_investment() == False`, rather than silently producing nonsense.

---

## 3. The math

### Bus balance

For every bus $b$ and every step $t$, over all ports registered at that bus:

$$\sum_{p \in \text{ports}(b)} \text{sign}_p \cdot f_p(t) = 0$$

That's the entire coupling mechanism. No slack, no losses at the bus — losses belong inside
components.

### Objective

$$\min \sum_{c \in \text{components}} \text{objective\_terms}(c)$$

in **EUR over the horizon**. Two kinds of terms, and the scaling between them matters:

- **Operational**, e.g. the grid: $\sum_t \left( \text{import}_t \cdot p^{\text{imp}}_t - \text{export}_t \cdot p^{\text{exp}}_t \right) \Delta t$
- **Investment**, annualized then scaled by the horizon's share of a year:

$$C_{\text{inv}} = \text{capex} \cdot \big( a(n, i) + \text{opex}_{\text{fix}} \big) \cdot \text{capacity} \cdot \underbrace{\frac{H}{8760}}_{\text{year\_fraction}}$$

with the annuity factor ($n$ = lifetime, $i$ = `model.wacc`):

$$a(n, i) = \frac{i\,(1+i)^n}{(1+i)^n - 1}, \qquad a(n, 0) = \frac{1}{n}$$

`year_fraction` is why a 72-hour design study still trades capex against opex correctly: both
sides are scaled to the same horizon.

### Storage

State of charge $S$, charge $c$, discharge $d$, capacity $K$, self-discharge $\sigma$ [1/h]:

$$S_{t+1} = S_t + \Big( c_t \eta_c - \frac{d_t}{\eta_d} + \delta_t \Big) \Delta t - S_t \, \sigma \, \Delta t$$

$$\text{soc}_{\min} K \le S_t \le \text{soc}_{\max} K, \qquad c_t \le \bar{c}_t, \quad d_t \le \bar{d}_t$$

Bus flow is $d_t - c_t$ at `sign=+1`. $\delta_t$ is the `_external_energy_delta` hook:
`ThermalStorage` returns $-\text{standby loss}_t$; a future `EVBattery` would return the forced
driving withdrawal, and the power limits already accept time series (0 while driving), which is
what "EV-ready" means here.

Boundary: under a wrapping strategy $S_0 = S_{\text{last}} + \Delta S_{\text{last}}$ (periodic,
no free energy). Otherwise, if `initial_soc` was given, $S_0 = \text{initial\_soc}$.

### Discrete options and `switched_flow`

For a choose-one investment with binaries $x_o$:

$$\sum_{o} x_o = 1$$

`switched_flow` is the reusable Big-M pattern for "this flow is $c_o \cdot \text{driver}(t)$ if
option $o$ is chosen, else 0" (generalized from Schuetz et al. 2017, eq. 8, 23–25):

$$x_o m_o \le f_{o,t} \le x_o M_o$$
$$(1 - x_o) m_o \le c_o \cdot \text{driver}(t) - f_{o,t} \le (1 - x_o) M_o$$

and the component uses $\sum_o f_{o,t}$. When the investment is fixed, this returns the plain
expression $c_{\text{fixed}} \cdot \text{driver}(t)$ and builds nothing — the LP collapse
mentioned above. It's generic: any option-dependent linear coefficient (envelope U-values today, a
COP catalog tomorrow) can use it.

### The 5R1C thermal zone

Three temperature nodes — mass $T_m$ (the only one with capacity $C_m$), surface $T_s$, air
$T_{air}$ — per DIN EN ISO 13790 / Schuetz et al. 2017. Envelope flows $Q_e$ come from
`switched_flow` with driver $(T_m - T_e)$ and coefficient $H_e$ [kW/K] of the chosen option.

**Mass node** (the difference equation, hence the thermal storage behavior):

$$H_{ms}(T_{m,t} - T_{s,t}) + \!\!\sum_{e \in \{W,R,F\}}\!\! Q_{e,t} + H_{door}(T_{m,t} - T_{e,t}) = \Phi^m_t - C_m \frac{T_{m,t+1} - T_{m,t}}{\Delta t}$$

**Surface node:**

$$H_{ms}(T_{s,t} - T_{m,t}) + H_{is}(T_{s,t} - T_{air,t}) + Q_{\text{Win},t} = \Phi^s_t$$

**Air node**, where heating and cooling enter:

$$Q_{\text{Vent},t} + H_{is}(T_{air,t} - T_{s,t}) = \Phi^s_t - Q_{\text{cool},t} + Q_{\text{heat},t}$$

with gains split over the nodes (eq. 15/16):

$$\Phi^m_t = \tfrac{1}{2}\tfrac{A_m}{A_{tot}} Q_{ig,t} + \tfrac{A_f}{A_{tot}} Q_{sol,t}$$

$$\Phi^s_t = \Big(1 - \tfrac{U_{win}}{h_{ms} A_{tot}}\Big)\tfrac{Q_{ig,t}}{2} - \tfrac{\text{cross}_t}{h_{ms} A_{tot}} + Q_{sol,t} - \Phi^m_t$$

The `cross` term is the window×solar-option product, linearized with auxiliary binaries $P_X$
(eq. 16–18: $P_X \le x_w$, $P_X \le x_e$, $P_X \ge x_w + x_e - 1$) — and only created for pairs
where *both* decisions are actually free.

**Comfort band** — this is the flexibility, and why the zone must stay in the shared MILP.
$T_{air}$ is a free variable inside a band that the control investments widen:

$$T_{air,t} \le T_{lb} + (T_{ub} - T_{lb})\,x_{\text{smart}} - (T_{ub} - 30)\,\text{away}_t\,x_{\text{occ}}$$
$$T_{air,t} \ge T_{lb} - (T_{lb} - 18)\,\text{asleep}_t\,x_{\text{night}} - (T_{lb} - 14)\,\text{away}_t\,x_{\text{occ}}$$

Without a smart thermostat ($x_{\text{smart}} = 0$) the band collapses and $T_{air}$ is pinned at
$T_{lb}$ — no flexibility, which is exactly the reference case the flexibility tests compare
against. Occupancy control is constrained to require the other two features.

**Max load** is a *soft* constraint (a scalar violation variable penalized at 100 EUR/kW in the
objective) so an undersized system warns instead of going infeasible:
$Q_{\text{heat},t} - v \le \bar{Q}$.

> **Four inherited deviations from the source papers, all kept deliberately for bit-parity with
> the old model:** envelope flows are driven by $(T_m - T_e)$ even for windows and ventilation,
> which touch other nodes; the air-node balance uses the surface gain $\Phi^s$ rather than half
> the internal gains; and the hard comfort ceiling forces fictitious cooling in buildings that
> have no cooling device. The first three are sanctioned by neither source paper and are probably
> bugs; the last is deliberate but limiting. All are load-bearing for the validated
> ~197 vs 195 kWh/m²/a result, so fixing them is a revalidation exercise, not a drive-by.
> **Evidence, source quotations and measured impact:
> [`model-deviations.md`](model-deviations.md).** The cooling one is written up for outside
> readers in [`open-problem-summer-overheating.md`](open-problem-summer-overheating.md).

---

## 4. Tutorial: adding a heat pump

Goal: a component that draws electricity and delivers heat at a temperature-dependent COP, with an
optional capacity investment. It is the first component that couples **two** buses, so it shows
the part the existing components don't.

The finished component lives at `tsib/energysystem/components/heatpump.py`.

### Step 1 — what does it look like from outside?

Write the call first; it forces the constructor signature.

```python
HeatPump("hp", elec_bus, heat_bus, cop=cop_series, capacity=10.0)
HeatPump("hp", elec_bus, heat_bus, cop=cop_series,
         capacity=es.ContinuousInvestment(capex_per_unit=800.0, lifetime=18.0))
```

Accepting `capacity` as either a float or an investment is the house convention (`PVGenerator`,
`StorageComponent`). Follow it.

### Step 2 — the constructor

Store inputs, normalize `capacity`, precompute nothing. Use **relative** imports (`..core`) like
every other component — an absolute `from tsib.energysystem...` inside the package is a circular
import back through `tsib/__init__.py` (see [Gotchas](#5-gotchas)).

```python
import numpy as np
import pandas as pd
import pyomo.environ as pyomo

from ..core.component import Component, Port
from ..core.investment import ContinuousInvestment
from ._profiles import profile_values


class HeatPump(Component):
    def __init__(self, name, elec_bus, heat_bus, cop, capacity):
        super().__init__(name)          # validates the name
        self.elec_bus = elec_bus
        self.heat_bus = heat_bus
        self.cop = cop
        if isinstance(capacity, ContinuousInvestment):
            self.investment = capacity
        else:
            self.investment = ContinuousInvestment(
                capex_per_unit=0.0, lifetime=18.0, capacity=float(capacity)
            )
        self._cop = None                # filled in build_parameters
```

Setting `self.investment` is all that's needed for the framework to see the investment — that's
what `has_free_investment()` and the strategy guard read.

### Step 3 — parameters

`profile_values` normalizes float/list/ndarray/Series to a float array of the right length, and
raises a clear error on a length mismatch. Use it for every profile input; it's why every
component accepts a scalar interchangeably with a series.

```python
    def build_parameters(self, model, block):
        self._cop = profile_values(self.cop, len(model.timeindex), self.name + " cop")
```

### Step 4 — variables

One variable is enough. Electricity is *derived* from heat via the COP rather than being its own
variable, which keeps the balance exact by construction — no need to constrain
`power * cop == heat`, and one fewer variable per step.

```python
    def build_variables(self, model, block):
        self.investment.build(model, block)      # no-op when fixed
        block.heat_out = pyomo.Var(model.timeindex.steps, within=pyomo.NonNegativeReals)
```

`capacity_expr(block)` then returns either the float or `block.capacity` — write the constraint
once and it works for both.

### Step 5 — constraints

```python
    def build_constraints(self, model, block):
        cap = self.investment.capacity_expr

        block.capacity_limit = pyomo.Constraint(
            model.timeindex.steps,
            rule=lambda b, t: b.heat_out[t] <= cap(b),
        )
        # tsib.simHeatpump returns cop == 0 below the cut-off temperature
        off = [t for t in model.timeindex.steps if self._cop[t] <= 0.0]
        if off:
            block.cutoff = pyomo.Constraint(off, rule=lambda b, t: b.heat_out[t] == 0.0)
```

Note `cap(b)` works unchanged whether capacity is a number or a variable — with a float it's a
plain bound, with a variable it's the investment coupling. That's the composition paying off.

### Step 6 — ports: the two-bus coupling

```python
    def _power_in(self, block, t):
        if self._cop[t] <= 0.0:
            return 0.0
        return block.heat_out[t] / self._cop[t]

    def ports(self):
        return {
            "heat": Port(self.heat_bus, lambda b, t: b.heat_out[t], sign=1),   # into heat bus
            "elec": Port(self.elec_bus, self._power_in, sign=-1),              # out of elec bus
        }
```

`self._cop` is still `None` when `ports()` is called (step 3 of `build()` precedes
`build_parameters`) — that's fine, because the callable is only *invoked* when the bus balances
are built, after all three passes. Read parameters inside the flow callable, never in the body of
`ports()`.

Dividing by `self._cop[t]` is a Python float division on a pyomo variable, so the expression stays
linear. If COP were a *decision* rather than data, this would be a bilinear term and would need
the `switched_flow` treatment instead.

### Step 7 — objective and results

```python
    def objective_terms(self, model, block):
        return self.investment.annual_cost_terms(model, block)   # [] when capex is 0

    def extract_results(self, block):
        heat = np.array([block.heat_out[t].value for t in block.heat_out])
        power = np.where(self._cop > 0.0,
                         heat / np.where(self._cop > 0.0, self._cop, 1.0), 0.0)
        return {
            "capacity": self.investment.capacity_value(block),
            "heat": pd.Series(heat),
            "power": pd.Series(power),
        }
```

Note there is no electricity cost term here. The heat pump doesn't know what electricity costs —
the `GridConnection` on the electricity bus owns that. A component pricing energy it didn't import
would double-count.

### Step 8 — run it

```python
import tsib
import tsib.energysystem as es

times = pd.date_range("2010-01-01", periods=48, freq="h", tz="UTC")
T_amb = pd.Series(np.linspace(-5, 8, 48), index=times)
cop = tsib.simHeatpump(T_amb, T_hot=45.0)                 # existing Carnot model
price = pd.Series(np.where((times.hour >= 17) & (times.hour < 21), 0.40, 0.10), index=times)

model = es.EnergySystemModel(times, wacc=0.06)
elec = model.add(es.Bus("elec", carrier="electricity"))
heat = model.add(es.Bus("heat", carrier="heat"))

model.add(es.GridConnection("grid", elec, import_price=price))
model.add(es.FixedDemand("space_heat", heat, pd.Series(np.linspace(4.0, 2.0, 48), index=times)))
model.add(HeatPump("hp", elec, heat, cop=cop,
                   capacity=es.ContinuousInvestment(capex_per_unit=800.0, lifetime=18.0,
                                                    max_capacity=20.0)))
model.add(es.ThermalStorage("buffer", heat, capacity=20.0, standby_loss_kW=0.02))
model.solve()
```

Which gives:

```
objective   : 6.014
hp capacity : 3.687 kW_th          # sized below peak demand: the buffer covers peaks
heat sum    : 147.49 kWh_th
power sum   : 45.21 kWh_el
mean SPF    : 3.262                # between the COP bounds 2.86 - 3.87, as it must be
spike share : 0.0                  # all imports shifted out of the 0.40 EUR/kWh hours
```

Both couplings are visible in that output: the capacity landing *below* peak demand is the buffer
and the investment trading off in one solve, and the zero spike share is the price signal reaching
the heat side through two buses.

### Step 9 — wiring it to a building

Swap `FixedDemand` for the real thermal zone and the heat pump now drives an actual building,
whose thermal mass adds flexibility on top of the buffer:

```python
bdg = tsib.Building(ID=..., refurbishment=False)
zone = es.ThermalZone5R1C("zone", bdg.zone_config, heat_bus=heat)   # heat_bus, not fixed profile
```

Passing `heat_bus=` is what makes the zone draw from the bus instead of pricing its own heat at
`heat_cost` internally — see `objective_terms` in `thermalzone5r1c.py`.

### If you export it

Add it to `tsib/energysystem/components/heatpump.py`, re-export in
`tsib/energysystem/__init__.py` **and** `tsib/__init__.py` (the house rule for public API), and
add a test alongside `test/test_energysystem_dispatch.py`. Good things to assert: the energy
balance closes (`heat ≈ power * cop`), a fixed capacity creates no extra variables, and the COP
cut-off is respected.

---

## 5. Gotchas

- **Circular imports, and the stale kernel they leave behind.** Inside the package always import
  relatively (`from ..core.component import Component`). An absolute `from tsib.energysystem...`
  re-enters `tsib/__init__.py`, which is already mid-import, and `import tsib` blows up. The nasty
  part is the aftermath in a *notebook*: the kernel keeps the half-built `tsib` module, so even
  after you fix the file you get `AttributeError: module 'tsib' has no attribute 'data'` — the
  import statement itself looks fine and the error surfaces later, on the attribute access.
  **Restart the kernel** after any import-time error; re-running the cell is not enough.
- **A bus with no supply fails silently.** Only a bus with *zero* connected components raises. A
  bus that has components but no source is perfectly feasible: the balance
  $\sum \text{sign} \cdot f = 0$ just forces every withdrawal to zero. Put a heat pump on an
  `elec` bus and forget the `GridConnection`, and you get no error — only a heat pump that never
  runs and an investment that sizes to 0. If a component mysteriously does nothing, check that
  every bus it touches has something injecting into it.
- **Export in two places.** A new component must be re-exported from
  `tsib/energysystem/__init__.py` *and* `tsib/__init__.py`. Doing only the first gives you a
  working `tsib.energysystem.Foo` and an `AttributeError` on `tsib.Foo`.
- **Reserved names.** Component and bus names must be valid Python identifiers and must not
  collide with pyomo `Block` attributes — `"load"` is the classic one and raises at construction.
  Use `"demand"`.
- **Ports are lazy.** `ports()` runs before `build_parameters`. Capture buses and signs there,
  read parameters inside the flow callable.
- **Flows are nonnegative, direction lives in the sign.** Don't encode direction as a negative
  flow; use `sign=-1` and a `NonNegativeReals` variable.
- **Trailing underscores.** `GridConnection` uses `block.import_` / `block.export_` because
  `import` is a Python keyword.
- **Scalars are broadcast.** Anything going through `profile_values` accepts a float, so
  `FixedDemand("d", bus, 1.0)` is a valid constant 1 kW.
- **`year_fraction` scales investments, `step_size_h` scales operation.** If you write a cost term
  by hand, multiply energy sums by `model.timeindex.step_size_h`, and any annualized cost by
  `model.timeindex.year_fraction`.
- **Solve time is dominated by binaries.** A fixed-investment full year is an LP and solves in
  ~4 s; a free envelope MILP over the same year does not finish in a useful time. The investment
  tests deliberately use 72-step horizons — decisions are stable there.
- **Set `$SOLVER=highs`** unless you have gurobi/cplex.
- **"Maximal heat load exceeded" warnings are the soft constraint doing its job** — the design
  system is slightly undersized in some hours, exactly as in the old model.

## Where things live

```
tsib/energysystem/
├── core/
│   ├── timeindex.py    TimeIndex: steps, pairs(), slice(), year_fraction
│   ├── component.py    Component ABC, Port, name validation
│   ├── bus.py          Bus + the generic balance constraint
│   ├── investment.py   annuity_factor, ContinuousInvestment, DiscreteOptionInvestment
│   ├── model.py        EnergySystemModel: build/solve/results
│   ├── strategy.py     PerfectForesight + Myopic/RollingHorizon stubs
│   └── solverutils.py  solver detection and options
└── components/
    ├── thermalzone5r1c.py   ThermalZoneConfig + ThermalZone5R1C
    ├── envelope.py          refurbishment catalog from the cost Excel
    ├── control.py           smart thermostat / occupancy / night reduction
    ├── storage.py           StorageComponent -> Electrical/ThermalStorage
    ├── pv.py, grid.py, demand.py
    └── _profiles.py         profile_values
```

Reference: `examples/energysystem/EnergySystemDemo.ipynb` (flexibility, PV+battery investment,
envelope refurbishment) and the test suite, which doubles as executable spec —
`test_energysystem_core.py` for the framework contract, `_dispatch.py` for the flexibility proof.
