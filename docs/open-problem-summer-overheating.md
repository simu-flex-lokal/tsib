# Open problem: summer overheating in an optimisation-based 5R1C building model

**Status:** open. Nothing in the code has been changed. This document is self-contained — it
assumes no knowledge of the `tsib` software, only building physics and linear optimisation.

**Question in one sentence:** in a *cost-minimising* linear building model, how do you let the
indoor temperature float freely above the comfort band when the building has no cooling device,
without simultaneously allowing the optimiser to deliberately overheat the building for economic
gain?

**Contact:** the model in question is [`tsib`](https://github.com/FZJ-IEK3-VSA/tsib) (FZJ IEK-3,
MIT licence). File: `tsib/energysystem/components/thermalzone5r1c.py`.

---

## 1. Background

### 1.1 The model

A single-zone reduced-order building model of the **5R1C** type (five thermal resistances, one
thermal capacitance), i.e. the *simple hourly method* of **DIN EN ISO 13790**. Three temperature
nodes are modelled:

| node | symbol | meaning |
|---|---|---|
| air | $\theta_{air}$ | indoor air temperature |
| surface | $\theta_s$ | mean internal surface temperature |
| mass | $\theta_m$ | thermal mass, the only node carrying capacitance $C_m$ |

The distinguishing feature versus a conventional building simulation is that this model is
embedded **inside a mathematical program**: the temperatures are *decision variables*, not the
output of a forward time-integration. This follows **Schütz et al. (2017)** [S17], who
reformulated the ISO 13790 hourly method so that it can be solved as a mixed-integer linear
program together with the building's energy system (heat pump, storage, PV, tariffs).

The energy balances ([S17] Eqs. 20–22), for each time step $t$:

$$H_{tr,ms}(\theta_m - \theta_s) + H_{tr,em}(\theta_m - \theta_e) = \phi_m - C_m \frac{\partial \theta_m}{\partial t} \tag{20}$$

$$H_{tr,ms}(\theta_s - \theta_m) + H_{tr,is}(\theta_s - \theta_{air}) + H_{tr,w}(\theta_s - \theta_e) = \phi_{st} \tag{21}$$

$$H_{ve}(\theta_{air} - \theta_e) + H_{tr,is}(\theta_{air} - \theta_s) = \phi_{ia} + \phi_{HC} \tag{22}$$

with $\theta_e$ the ambient temperature (exogenous), $H_\bullet$ heat transfer coefficients
[kW/K], $\phi_m, \phi_{st}, \phi_{ia}$ the internal and solar gains distributed over the three
nodes, and $\phi_{HC}$ the **heating/cooling power supplied to the air node** — the control
variable.

### 1.2 Why the temperature must stay a free decision variable

This is the design premise, and it constrains admissible solutions. The comfort band together
with the thermal mass $C_m$ is structurally a **thermal storage**: within the band, the building
can be pre-heated when energy is cheap and coast when it is expensive. If the building were
simulated separately and reduced to a fixed heat-demand time series, that flexibility would be
destroyed. So $\theta_{air}$, $\theta_s$, $\theta_m$ must remain free variables in the *same*
optimisation as the energy system.

### 1.3 What the two source works do

**[S17] has no upper temperature bound at all.** Its only comfort constraint is

$$\theta_{air} \ge \theta_{set}^{lb} \tag{26}$$

and it justifies this explicitly (§2.3.5):

> "Since no cooling system is investigated, an upper bound is not explicitly formulated. However,
> as heat generation leads to additional operating costs, the optimizer will not overheat the
> building and reduce heat generation if the air temperature exceeds its set point."

This free-floating behaviour is **validated** in [S17] against **ASHRAE 140 case 600FF** (the
free-float test case, heating and cooling disabled), agreeing with the reference band within
~1 K (its Fig. 8).

**Kotzur (2018)** [K18], the dissertation the software implements, *adds* an upper bound
(its Eq. 3.2), stating (p. 36):

> "The upper bound of the temperature $T_{air,ub}^{com}$ **limits the heating operation and
> determines the cooling load in the summer period** […] The general upper bound for the
> temperature is set to 26 °C."

[K18] otherwise defers all model equations to [S17] and adds only the two comfort bounds.

### 1.4 The implementation

$\phi_{HC}$ is split into two non-negative variables, and the upper bound is **hard**:

$$\phi_{HC,t} = Q_{heat,t} - Q_{cool,t}, \qquad Q_{heat,t} \ge 0, \quad Q_{cool,t} \ge 0$$
$$\theta_{set}^{lb} \le \theta_{air,t} \le \theta_{ub} \qquad \forall t$$

$Q_{cool}$ is **unbounded above** and priced at a small constant (0.02 EUR/kWh) in the objective.
The objective minimises total annual cost (energy + annualised investment).

---

## 2. The problem

### 2.1 Statement

Because $\theta_{air,t} \le \theta_{ub}$ is a **hard** constraint and $Q_{cool}$ is **always
available**, any solar or internal gain that would push the zone above $\theta_{ub}$ *must* be
removed by $Q_{cool}$. The model therefore reports a **cooling demand for buildings that have no
cooling device** — which is the overwhelming majority of the German residential stock this model
is applied to. There is no parameter anywhere in the software to declare that a building cannot
cool.

A secondary effect follows: heat that is removed by fictitious cooling is heat that is *not*
stored in $C_m$ and therefore not available later. Overheating in the afternoon should partly
carry the building through the following night; forced cooling discards it, which can create
phantom heating demand.

### 2.2 Quantitative evidence

Reference building (TABULA archetype, German SFH 1969–1978, $A_f = 173\ \mathrm{m^2}$, DWD test
reference year zone 4, comfort band 20–26 °C, 8760 h). Current formulation versus the same model
with $Q_{cool} \equiv 0$ and the upper bound removed (a valid experiment here because with a
*constant* heat price and no arbitrage, cost minimisation is equivalent to free-floating):

| | heating | cooling | $\theta_{air}$ max | hours > 26 °C |
|---|---|---|---|---|
| current (hard bound + free $Q_{cool}$) | 197.32 kWh/m²a | **8.20 kWh/m²a** | 26.00 °C | 0 |
| free-float ($Q_{cool} \equiv 0$, no bound) | 196.72 kWh/m²a | 0 | **39.59 °C** | 752 |

Two conclusions:

1. **The fabricated service is large.** 8.20 kWh/m²a ≈ 1420 kWh/a of cooling that cannot
   physically occur. $\theta_{air}$ max is *exactly* 26.00 °C with zero exceedance hours — the
   bound is binding and cooling enforces it.
2. **The knock-on error on heating is small.** 0.6 kWh/m²a, i.e. **0.3 %**. The heating and
   overheating seasons barely overlap. For scale, [K18] Fig. 3.8 reports RMSE
   **6.79–6.82 kWh/m²a** for this model against the IWU reference building stock, so the heating
   side-effect is an order of magnitude inside the existing validation noise.

So this is primarily a **fabricated-cooling** problem, not a heat-demand problem.

### 2.3 Why the obvious fix is not sufficient

**Removing the upper bound and setting $Q_{cool} \equiv 0$ — i.e. reverting exactly to [S17] —
fails for two independent reasons.**

**(a) [S17]'s justification does not survive time-varying prices.** Its argument is *"heat
generation leads to additional operating costs, [so] the optimizer will not overheat"*. That
holds only under **constant** energy prices. Modern applications of this model include
time-of-use tariffs, PV surplus and heat pumps — under which **pre-heating pays**. With no upper
bound, a cost-minimising optimiser will drive the zone to 40 °C to arbitrage a cheap night
tariff. The upper bound is therefore doing **two distinct jobs**, exactly as [K18]'s own sentence
says — it *"limits the heating operation"* **and** *"determines the cooling load"*. Only the
second is wrong; the first is load-bearing and was not needed by [S17].

The exact separation of the two jobs is a **disjunction**:

$$\theta_{air,t} \le \theta_{ub} \quad \textbf{or} \quad Q_{heat,t} = 0$$

("the building may be hotter than the band, but not *because you heated it*"). Written with a
Big-M and a binary $y_t$ per time step, this costs 8760 binaries for an annual run, which
destroys the linear program — full-year LP solve times are ~35 s, whereas the same model with
per-timestep binaries is not tractable.

**(b) Free-floating alone is physically wrong too.** The free-float run reaches **39.59 °C**.
Neither [S17] nor the implementation models **window opening**, and [S17] §2.2 explicitly notes
it does not model mechanical ventilation either, assuming the ventilation supply temperature
equals ambient. Real occupants ventilate — that is the actual physical mechanism that limits
summer temperatures in an uncooled dwelling. Removing the fictitious cooling without adding a
ventilation response simply trades one bias for the opposite one.

### 2.4 Constraints on an acceptable solution

1. **Must scale to 8760 hourly steps**, ideally without per-timestep binary variables. The model
   is solved jointly with an energy system (investment sizing included).
2. **Must preserve the thermal-mass flexibility** under time-varying prices — pre-heating *within*
   the band must remain free and unpenalised (see §1.2).
3. **Must not silently reintroduce a cooling device.**
4. **Should be traceable to a standard**, since the model's purpose is stock-level German
   residential analysis.
5. Should not shift the IWU heat-demand agreement beyond the existing ~6.8 kWh/m²a RMSE.

---

## 3. Candidate approach (unproven)

Introduce an explicit "has cooling device" property. Where absent, fix $Q_{cool} \equiv 0$ and
**soften** the upper bound with a penalised slack:

$$\theta_{air,t} \le \theta_{ub} + s_t, \qquad s_t \ge 0, \qquad \text{objective} \mathrel{+}= \lambda \sum_t s_t \,\Delta t$$

The argument that this needs no binaries: pre-heating to exactly $\theta_{ub}$ stays free
(so requirement 2 holds); exceeding it costs, so arbitrage overheating is suppressed
(requirement 2 vs. the disjunction); and in summer the penalty is a **sunk** cost the model has
no lever to avoid — $Q_{cool}=0$ and ventilation is not a decision variable — so dispatch should
be undistorted. **This last claim is an argument, not a proof, and is one of the open questions
below.**

Attractive property: $\sum_t s_t \Delta t$ has units of **Kelvin-hours per year [Kh/a]** and is
therefore *identical in form* to established over-temperature metrics — see §4.

For the ventilation gap (§2.3b), the linearity-preserving option is to make the air change rate
an **exogenous time series** (elevated in summer/at night per a standard schedule) rather than a
decision variable — as a parameter it costs nothing. Making the rate respond to $\theta_{air}$,
which is the physically honest version, is **bilinear** and needs the same binaries as the
disjunction.

---

## 4. Two independent standards use the same currency

Two unrelated lines of work measure comfort violation in **Kelvin-hours per year**, which
suggests $\sum_t s_t \Delta t$ is the right quantity:

- **Oldewurtel et al. (2012)** [O12], Swiss office Integrated Room Automation:
  > "A reasonable violation level for room temperature as it would be tolerated according to the
  > standards would be about **70 Kh/a**"

  with the footnote *"1 Kh/a (=KelvinHour/Annum) corresponds to exceeding the temperature
  constraint by 1 K for 1 h within 1 year"*, citing **EN 15251:2007**.
- **DIN 4108-2:2013-02 §8.4** ("Sommerlicher Wärmeschutz", dynamic verification), *per secondary
  sources only*: verifies summer comfort **under free-floating conditions without active
  cooling**, using *Übertemperaturgradstunden* (over-temperature degree hours, Kh/a) above a
  climate-region-dependent limit temperature, permitting **1200 Kh/a** for residential buildings.

[O12] also supplies the normative case that a **hard** bound is the wrong shape in the first
place, independent of the cooling question (§3.1.3.2):

> "the European standards specify that comfort bounds on room temperature **do not need to be
> guaranteed at all times, but may be violated for a small fraction of time during the year**,
> e.g., in extreme weather situations"

Note [O12] itself does **not** use a penalised slack — it uses *chance constraints*
$P[Ax_k \le b] \ge 1-\alpha$, which with affine disturbance feedback yield a second-order cone
program (convex, not linear, and by the authors' own description "fairly computationally
expensive at larger scales"). Its purpose is hedging weather-forecast uncertainty, not modelling
the absence of a cooling device. It is cited here for the *metric* and the *normative principle*,
not the formulation.

**Unresolved:** 70 Kh/a versus 1200 Kh/a is a factor of ~17. Presumably a scope difference —
actively conditioned office with a two-sided band, versus uncooled residential summer
overheating. The latter is this model's case. This must be settled before any penalty or budget
is chosen.

---

## 5. Open research questions

1. **Formulation.** Is there an exact or provably-tight LP formulation of *"passive gains may
   exceed the comfort band; deliberate heating may not"* that avoids one binary per time step?
   Restricting binaries to the ~1000–2000 hours where overheating is even possible, or a
   Benders/lazy-constraint scheme, may make the disjunction affordable — is that established
   anywhere?
2. **Distortion.** Does the penalised-slack surrogate (§3) provably not distort dispatch? The
   "sunk cost" argument fails as soon as the model gains *any* lever over summer temperature
   (controllable ventilation, shading, a real cooling device). Under what conditions is it safe?
3. **Calibration.** What are the correct comfort criterion and budget for **uncooled German
   residential** buildings — DIN 4108-2's Übertemperaturgradstunden (reportedly 1200 Kh/a, with
   region-dependent reference temperatures) or EN 16798-1's successor to EN 15251? What explains
   the 70 vs 1200 Kh/a gap?
4. **Occupant ventilation.** Is there a **linear or LP-representable** model of window opening /
   night ventilation for a reduced-order zone that is standard-conformant? A literature search
   found this space dominated by CFD and design-stage optimisation (SQP, response-surface and
   neural surrogates) — nothing transplantable into a linear dispatch model. Is an exogenous
   standard schedule (ISO 13790 / DIN V 18599) the state of the art here, or is there better?
5. **Upstream cause.** [K18] §3.2.1 notes that this model family's **cooling demands are
   overestimated** in the VDI 6007 comparison, and dismisses it as *"not relevant for this work
   since they are not considered in the optimization"*. Is that overestimation a separate defect
   upstream of the comfort bound, i.e. would the cooling still be wrong even for a building that
   *does* have a cooling device?

---

## 6. References and access status

| ref | work | access |
|---|---|---|
| **[S17]** | Schütz, T., Schiffer, L., Harb, H., Fuchs, M., Müller, D. (2017): *Optimal design of energy conversion units and envelopes for residential building retrofits using a comprehensive MILP model.* Applied Energy 185(1), 1–15. [doi:10.1016/j.apenergy.2016.10.049](https://doi.org/10.1016/j.apenergy.2016.10.049) | full text read |
| **[K18]** | Kotzur, L. (2018): *Future Grid Load of the Residential Building Sector.* Dissertation, RWTH Aachen / FZJ Energie & Umwelt **442**, ISBN 978-3-95806-370-9. §3.2.2 | full text read |
| **[O12]** | Oldewurtel, F. et al. (2012): *Use of model predictive control and weather forecasts for energy efficient building climate control.* Energy and Buildings **45**, 15–27. [doi:10.1016/j.enbuild.2011.09.022](https://doi.org/10.1016/j.enbuild.2011.09.022) | full text read |
| | **DIN 4108-2:2013-02** §8.4 — *Wärmeschutz und Energie-Einsparung in Gebäuden: Mindestanforderungen an den Wärmeschutz* | **not obtained** — the decisive missing document |
| | **EN 15251:2007** / **EN 16798-1:2019** — indoor environmental input parameters | **not obtained** |
| | **VDI 6007** — [S17]'s validation reference; per [K18] the source of the cooling overestimation finding | **not obtained** |
| | **DIN EN ISO 13790:2008** — the underlying standard (simple hourly method, §C.4) | not obtained |
| | ASHRAE Standard 140 — free-float test case 600FF, used by [S17] | not obtained |

Assertions attributed to DIN 4108-2 in this document rest on **secondary sources only** and must
be checked against the standard itself before use.
