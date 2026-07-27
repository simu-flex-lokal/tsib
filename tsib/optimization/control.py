# -*- coding: utf-8 -*-
"""
Comfort control features of a thermal zone (smart thermostat, occupancy
control, night reduction) as three independent single-feature
investment decisions. They share the DiscreteOptionInvestment
bookkeeping (selection/fixing/cost), but need no Big-M constraints as
the comfort bounds are linear in the selections.
"""

from .investment import DiscreteOptionInvestment
from .envelope import load_control_options


class ComfortControl(object):
    """
    Holds the three control feature decisions of a thermal zone.

    Each feature is a two-option (install / nothing) discrete
    investment. Already installed features are fixed active with zero
    cost; without refurbishment optimization missing features are fixed
    inactive.

    Parameters
    ----------
    cfg: dict, required
        Resolved building configuration; the flags cfg["capControl"],
        cfg["occControl"] and cfg["nightReduction"] mark features which
        are already installed.
    refurbishment: bool, required
        Whether missing features may be invested in by the optimizer.
    """

    # feature name -> cfg flag of an already installed feature
    FEATURES = {
        "SmartThermostat": "capControl",
        "Occupancy": "occControl",
        "NightReduction": "nightReduction",
    }

    def __init__(self, cfg, refurbishment):
        catalog = load_control_options(cfg)
        self.refurbishment = refurbishment
        self.investments = {}
        for feature, flag in self.FEATURES.items():
            installed = bool(cfg[flag])
            capex = 0.0 if installed else catalog[feature]["capex"]
            options = {
                "install": dict(catalog[feature], capex=capex),
                "nothing": {"capex": 0.0, "opex_fix_share": 0.0, "lifetime": 1.0},
            }
            if installed:
                fixed = "install"
            elif not refurbishment:
                fixed = "nothing"
            else:
                fixed = None
            self.investments[feature] = DiscreteOptionInvestment(
                "control_" + feature, options, fixed_option=fixed
            )

    def build(self, model, block):
        for investment in self.investments.values():
            investment.build(model, block)

    def selection(self, block, feature):
        """Install decision of `feature` as float or binary variable."""
        return self.investments[feature].selection(block, "install")

    def has_free_investment(self):
        return any(not inv.is_fixed for inv in self.investments.values())

    def annual_cost_terms(self, model, block):
        terms = []
        for investment in self.investments.values():
            terms.extend(investment.annual_cost_terms(model, block))
        return terms
