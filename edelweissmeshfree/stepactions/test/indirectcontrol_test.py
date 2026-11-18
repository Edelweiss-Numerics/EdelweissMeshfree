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
"""The indirect controls of material points and of particles take the indices of the controlled degrees of freedom
from the discrete system the arc length solver passes, ``system.numbering`` (EdelweissFE #303): the correction of the
load parameter makes the interpolated, weighted displacement grow by the share of the increment."""

from types import SimpleNamespace

import numpy as np
import pytest
from edelweissfe.journal.journal import Journal
from edelweissfe.timesteppers.timestep import TimeStep

from edelweissmeshfree.stepactions.indirectcontrol import (
    IndirectControl as MaterialPointIndirectControl,
)
from edelweissmeshfree.stepactions.particleindirectcontrol import (
    IndirectControl as ParticleIndirectControl,
)

# two nodes of a 2D displacement field, numbered [0, 1] and [2, 3]
NODES = [SimpleNamespace(fields={"displacement": label}) for label in ("node 1", "node 2")]
SYSTEM = SimpleNamespace(
    numbering=SimpleNamespace(idcsOfFieldVariablesInDofVector={"node 1": np.array([0, 1]), "node 2": np.array([2, 3])})
)
MODEL = SimpleNamespace(domainSize=2)
N = np.array([0.25, 0.75])
C = np.array([[1.0, 0.0]])

dU = np.array([0.1, 0.0, 0.3, 0.0])
ddU_0 = np.array([0.0, 0.0, 0.1, 0.0])
ddU_f = np.array([1.0, 0.0, 2.0, 0.0])


def _expectedDDLambda(dL):
    return (dL - N @ (dU + ddU_0)[[0, 2]]) / (N @ ddU_f[[0, 2]])


def test_material_point_indirect_control_reads_the_numbering_of_the_system():
    cell = SimpleNamespace(nodes=NODES, getInterpolationVector=lambda x: N)
    materialPoint = SimpleNamespace(assignedCells=[cell], getCenterCoordinates=lambda: np.zeros(2))
    controller = MaterialPointIndirectControl("control", MODEL, [materialPoint], 1.0, C, "displacement", Journal())

    ddLambda = controller.computeDDLambda(dU, ddU_0, ddU_f, TimeStep(1, 0.1, 0.5, 0.1, 0.5, 0.5), SYSTEM)

    assert ddLambda == pytest.approx(_expectedDDLambda(dL=0.1 * 1.0))


def test_particle_indirect_control_reads_the_numbering_of_the_system():
    particle = SimpleNamespace(
        kernelFunctions=[SimpleNamespace(node=node) for node in NODES],
        getInterpolationVector=lambda x: N,
        getCenterCoordinates=lambda: np.zeros(2),
    )
    controller = ParticleIndirectControl(
        "control", MODEL, [particle], 1.0, C, "displacement", Journal(), f_t=lambda t: t
    )

    ddLambda = controller.computeDDLambda(dU, ddU_0, ddU_f, TimeStep(1, 0.1, 0.5, 0.1, 0.5, 0.5), SYSTEM)

    # f_t(0.6) - f_t(0.5) of the target L = 1
    assert ddLambda == pytest.approx(_expectedDDLambda(dL=0.1 * 1.0))
