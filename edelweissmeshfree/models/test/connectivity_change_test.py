# -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
#
#  _____    _      _              _         __  __ ____  __  __
# | ____|__| | ___| |_      _____(_)___ ___|  \/  |  _ \|  \/  |
# |  _| / _` |/ _ \ \ \ /\ / / _ \ / __/ __| |\/| | |_) | |\/| |
# | |__| (_| |  __/ |\ V  V /  __/ \__ \__ \ |  | |  __/| |  | |
# |_____\__,_|\___|_| \_/\_/ \___|_|___/___/_|  |_|_|   |_|  |_|
#
#
#  Unit of Strength of Materials and Structural Analysis
#  University of Innsbruck,
#  2023 - today
#
#  Matthias Neuner matthias.neuner@uibk.ac.at
#
#  This file is part of EdelweissMeshfree.
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissMeshfree.
#  ---------------------------------------------------------------------
"""An ``MPMModel`` classifies what it changed before an increment (``MPMModel.updateConnectivity`` returns a
``Change``): a change of the locations if the nodes of its active domain are fixed, a change of the numbering if they
may change, nothing if nothing moved. The nodes are fixed without MPM managers, whose cells become active and inactive,
and with every particle manager bonding its kernel functions to the particles.

The managers and the constraint are stand-ins, so that no Marmot is needed."""

import pytest
from edelweissfe.constraints.base.constraintbase import ConstraintBase
from edelweissfe.models.change import Change

from edelweissmeshfree.models.mpmmodel import MPMModel


class _Manager:
    """A manager that reports a change of the connectivity or not, as the test says."""

    def __init__(self, changes: bool, bonded: bool = True):
        self.changes = changes
        self.kernelFunctionsAreBondedToParticles = bonded

    def updateConnectivity(self) -> bool:
        return self.changes


class _Constraint(ConstraintBase):
    """A constraint without nodes, within reach or not as the test says."""

    name = "constraint"
    scalarVariables = []
    nodes = []
    fields = []
    nDof = 0

    def __init__(self):
        super().__init__("constraint", None)
        self.inReach = True

    def updateConnectivity(self, model) -> bool:
        self.withinReach = self.inReach
        return False

    def computeKernels(self, U_np, dU, P, K, timeStep):
        pass


def _model(mpmManagers=(), particleManagers=(), constraint=None) -> MPMModel:
    model = MPMModel(2)
    model.mpmManagers = list(mpmManagers)
    model.particleManagers = list(particleManagers)
    if constraint is not None:
        model.constraints[constraint.name] = constraint
    return model


def test_without_a_change_nothing_changed():
    for model in (
        _model(),
        _model(particleManagers=[_Manager(False, bonded=False)]),
        _model(mpmManagers=[_Manager(False)]),
    ):
        assert model.updateConnectivity() == Change.NONE


def test_particles_reaching_other_nodes_with_fixed_active_nodes_is_a_change_of_the_locations():
    model = _model(particleManagers=[_Manager(True), _Manager(False)])

    assert model.updateConnectivity() == Change.LOCATIONS


@pytest.mark.parametrize(
    "model",
    [
        _model(particleManagers=[_Manager(True, bonded=False)]),
        _model(particleManagers=[_Manager(True), _Manager(False, bonded=False)]),
        _model(mpmManagers=[_Manager(True)]),
    ],
    ids=["unbonded kernel functions", "one unbonded manager among bonded ones", "material points move through cells"],
)
def test_a_change_with_active_nodes_that_may_change_is_a_change_of_the_numbering(model):
    assert model.updateConnectivity() == Change.NUMBERING


def test_a_constraint_leaving_reach_is_a_change_of_the_locations_with_fixed_active_nodes():
    constraint = _Constraint()
    model = _model(particleManagers=[_Manager(False)], constraint=constraint)

    constraint.inReach = False
    assert model.updateConnectivity() == Change.LOCATIONS
    assert model.updateConnectivity() == Change.NONE


def test_a_constraint_leaving_reach_is_a_change_of_the_numbering_with_active_nodes_that_may_change():
    constraint = _Constraint()
    model = _model(mpmManagers=[_Manager(False)], constraint=constraint)

    constraint.inReach = False
    assert model.updateConnectivity() == Change.NUMBERING
    assert model.updateConnectivity() == Change.NONE
