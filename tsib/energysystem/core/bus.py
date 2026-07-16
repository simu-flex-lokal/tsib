# -*- coding: utf-8 -*-
"""
Bus: energy balance node per carrier. Components register their flows
via Ports; the bus builds the balance constraint generically, so no
component needs to know any other component.
"""

import pyomo.environ as pyomo


class Bus(object):
    """
    Energy balance node for a single energy carrier.

    Parameters
    ----------
    name: str, required
        Unique bus name, used as constraint name on the pyomo model.
    carrier: str, optional (default: name)
        Energy carrier, e.g. "electricity" or "heat".
    """

    def __init__(self, name, carrier=None):
        from .component import check_name

        check_name(name, "Bus")
        self.name = name
        self.carrier = carrier if carrier is not None else name
        # list of (component, Port) filled by EnergySystemModel.build()
        self._connections = []

    def _register(self, component, port):
        self._connections.append((component, port))

    def _reset(self):
        self._connections = []

    def build_balance_constraint(self, model, pyomo_model):
        """
        Adds the energy balance sum(sign * flow) == 0 for every time
        step over all registered ports.
        """
        if not self._connections:
            raise ValueError(
                "Bus '{}' has no connected components".format(self.name)
            )
        connections = list(self._connections)

        def balance(m, t):
            return (
                sum(
                    port.sign * port.flow(getattr(pyomo_model, comp.name), t)
                    for comp, port in connections
                )
                == 0
            )

        pyomo_model.add_component(
            self.name + "_balance",
            pyomo.Constraint(model.timeindex.steps, rule=balance),
        )

    def __repr__(self):
        return "Bus('{}', carrier='{}')".format(self.name, self.carrier)
