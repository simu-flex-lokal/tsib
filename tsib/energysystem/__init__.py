"""
Generic, component-based energy system optimization framework for tsib.

Components (thermal zone, storages, PV, grid connections, demands) are
plugged onto carrier-specific buses and optimized jointly in a single
MILP. The key architectural requirement is that the 5R1C thermal zone's
state variables (T_air, T_m, T_s) and comfort-band constraints remain
free decision variables in the same solve as storage/PV/price-driven
dispatch, so that the building's thermal mass acts as a flexibility
resource (pre-heating/-cooling within the comfort band).
"""

from .core.timeindex import TimeIndex
from .core.component import Component, Port
from .core.bus import Bus
from .core.investment import (
    InvestmentOption,
    ContinuousInvestment,
    DiscreteOptionInvestment,
    annuity_factor,
)
from .core.model import EnergySystemModel
from .core.strategy import (
    SolveStrategy,
    PerfectForesightStrategy,
    MyopicStrategy,
    RollingHorizonStrategy,
)
from .core.solverutils import detect_solver, manageSolverOpts
from .components.thermalzone5r1c import ThermalZoneConfig, ThermalZone5R1C
from .components.storage import StorageComponent, ElectricalStorage, ThermalStorage
from .components.pv import PVGenerator
from .components.grid import GridConnection
from .components.demand import FixedDemand
