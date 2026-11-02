# -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
#
#  _____    _      _              _
# | ____|__| | ___| |_      _____(_)___ ___
# |  _| / _` |/ _ \ \ \ /\ / / _ \ / __/ __|
# | |__| (_| |  __/ |\ V  V /  __/ \__ \__ \
# |_____\__,_|\___|_| \_/\_/_\___|_|___/___/
# |  \/  | ___  ___| |__  / _|_ __ ___  ___
# | |\/| |/ _ \/ __| '_ \| |_| '__/ _ \/ _ \
# | |  | |  __/\__ \ | | |  _| | |  __/  __/
# |_|  |_|\___||___/_| |_|_| |_|  \___|\___|
#
#
#  Unit of Strength of Materials and Structural Analysis
#  University of Innsbruck,
#
#  Research Group for Computational Mechanics of Materials
#  Institute of Structural Engineering, BOKU University, Vienna
#
#  2023 - today
#
#  Matthias Neuner |  matthias.neuner@boku.ac.at
#  Thomas Mader    |  thomas.mader@bokut.ac.at
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
"""The mortar tie between particles and finite elements, on an interface of two quadratic element edges at x = 0 and
two particles to its right, whose stand-in shape functions are a partition of unity (see ``_residualtangent``):

- its tangent is the derivative of its residual contribution;
- it reads the increments at the nodes, from ``dU``, and the tractions as totals, from ``U_np``;
- the derivatives of the gap are the integrals they stand for: the edge mass matrix on the finite element side, and,
  as the shape functions on both sides sum to one, row sums that are the integrals of the edge shape functions;
- a translation of both sides by the same increment leaves no gap, and a common traction is transmitted with equal
  and opposite resultants;
- the integral follows the finite element nodes to their converged positions: the configuration is the current one;
- the finite element interface nodes must be in order along the interface, which is checked on their geometry.
"""

import numpy as np
import pytest
from edelweissfe.points.node import Node
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.residualtangentcheck import (
    finiteDifferenceTangent,
    residualContributionAndTangent,
)
from edelweissfe.variables.scalarvariable import ScalarVariable

from edelweissmeshfree.constraints.particlefiniteelementmortartie import (
    ParticleFiniteElementMortarTie,
)
from edelweissmeshfree.constraints.test._residualtangent import (
    ParticleStandIn,
    assertIncrementsAtNodesAndTotalMultipliersAreRead,
    stateInAnIncrement,
)
from edelweissmeshfree.fields.nodefield import MPMNodeField
from edelweissmeshfree.models.mpmmodel import MPMModel

TIMESTEP = TimeStep(3, 0.1, 0.6, 0.1, 0.6, 0.6)


def _model():
    """The finite element interface nodes, the kernel nodes, the particles, and a model holding their displacement."""

    finiteElementNodes = [Node(label, np.array([0.0, y])) for label, y in enumerate((0.0, 0.25, 0.5, 0.75, 1.0), 1)]
    kernelNodes = [
        Node(label, np.array([x, y]))
        for label, (x, y) in enumerate(((0.2, 0.2), (0.6, 0.3), (0.2, 0.8), (0.6, 0.7)), 10)
    ]
    for node in finiteElementNodes + kernelNodes:
        node.fields["displacement"] = None

    particles = [
        ParticleStandIn(kernelNodes[:2], (0.25, 0.25), [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]]),
        ParticleStandIn(kernelNodes[1:], (0.25, 0.75), [[0.0, 0.5], [0.5, 0.5], [0.5, 1.0], [0.0, 1.0]]),
    ]

    model = MPMModel(2)
    model.nodeFields = {
        "displacement": MPMNodeField("displacement", 2, NodeSet("all", finiteElementNodes + kernelNodes))
    }

    return model, finiteElementNodes, particles


def _tie(model, finiteElementNodes, particles) -> ParticleFiniteElementMortarTie:
    tie = ParticleFiniteElementMortarTie("tie", model, particles, finiteElementNodes)
    tie.assignAdditionalScalarVariables(
        [ScalarVariable() for _ in range(tie.getNumberOfAdditionalNeededScalarVariables())]
    )
    tie.updateConnectivity(model)
    return tie


@pytest.fixture
def tie():
    return _tie(*_model())


def _randomState(tie) -> tuple[np.ndarray, np.ndarray]:
    """``U_np`` and ``dU`` of random increments at the nodes and random total tractions."""

    return stateInAnIncrement(tie, 0.02 * np.random.default_rng(20261102).standard_normal(tie.nDof), 20261103)


def test_the_tangent_is_the_derivative_of_the_residual_contribution(tie):
    U_np, dU = _randomState(tie)

    P, K = residualContributionAndTangent(tie, U_np, dU, TIMESTEP)
    dP_dU = finiteDifferenceTangent(tie, U_np, dU, TIMESTEP)

    assert np.abs(K).max() > 0.0
    np.testing.assert_allclose(K, dP_dU, rtol=1e-6, atol=1e-6 * np.abs(K).max())


def test_the_tie_reads_increments_at_the_nodes_and_total_tractions(tie):
    assertIncrementsAtNodesAndTotalMultipliersAreRead(tie, *_randomState(tie), TIMESTEP)


def test_the_derivatives_of_the_gap_are_the_integrals_over_the_interface(tie):
    # the mass matrix of two quadratic edges of length 0.5, assembled
    edgeMassMatrix = 0.5 / 30 * np.array([[4.0, 2.0, -1.0], [2.0, 16.0, 2.0], [-1.0, 2.0, 4.0]])
    massMatrix = np.zeros((5, 5))
    massMatrix[:3, :3] += edgeMassMatrix
    massMatrix[2:, 2:] += edgeMassMatrix

    np.testing.assert_allclose(tie.dGap_dFiniteElementIncrement, massMatrix, atol=1e-14)

    # both partitions of unity integrate the edge shape functions: corner 1/6, midside 2/3 of the edge length
    integralsOfEdgeShapeFunctions = 0.5 * np.array([1 / 6, 2 / 3, 1 / 3, 2 / 3, 1 / 6])
    np.testing.assert_allclose(tie.dGap_dFiniteElementIncrement.sum(axis=1), integralsOfEdgeShapeFunctions)
    np.testing.assert_allclose(tie.dGap_dKernelIncrement.sum(axis=1), -integralsOfEdgeShapeFunctions)


def test_a_common_translation_leaves_no_gap_and_a_common_traction_is_balanced(tie):
    nNodes = len(tie.nodes)
    translation = np.array([0.03, -0.01])
    traction = np.array([2.0, -5.0])

    state = np.zeros(tie.nDof)
    state[: 2 * nNodes] = np.tile(translation, nNodes)
    state[2 * nNodes :] = np.tile(traction, 5)
    U_np, dU = stateInAnIncrement(tie, state, 1)

    P, _ = residualContributionAndTangent(tie, U_np, dU, TIMESTEP)
    forcesOnNodes = P[: 2 * nNodes].reshape(-1, 2)

    np.testing.assert_allclose(P[2 * nNodes :], 0.0, atol=1e-15)
    # the finite element side receives the traction times the interface length, the particles the opposite
    np.testing.assert_allclose(forcesOnNodes[:5].sum(axis=0), traction * 1.0)
    np.testing.assert_allclose(forcesOnNodes[5:].sum(axis=0), -traction * 1.0)


def test_the_interface_is_integrated_on_the_converged_configuration():
    model, finiteElementNodes, particles = _model()
    displacement = model.nodeFields["displacement"]
    # the converged displacement stretches the interface to twice its length
    for node in finiteElementNodes:
        displacement["U"][displacement.indexOfNode(node)] = [0.0, node.coordinates[1]]

    tie = _tie(model, finiteElementNodes, particles)

    np.testing.assert_allclose(tie.dGap_dFiniteElementIncrement.sum(), 2.0)


def test_the_finite_element_side_must_consist_of_quadratic_edges():
    model, finiteElementNodes, particles = _model()

    with pytest.raises(ValueError, match="quadratic edges"):
        ParticleFiniteElementMortarTie("tie", model, particles, finiteElementNodes[:4])


@pytest.mark.parametrize(
    "order",
    [
        # the corner nodes first, then the midside nodes, as in a node set in the order of its labels
        (0, 2, 4, 1, 3),
        # the same with the midside nodes backwards: each triple could be an edge, but the second one turns back
        (0, 2, 4, 3, 1),
    ],
)
def test_the_finite_element_interface_nodes_must_be_in_order_along_the_interface(order):
    model, finiteElementNodes, particles = _model()

    with pytest.raises(ValueError, match="in order along the interface"):
        ParticleFiniteElementMortarTie("tie", model, particles, [finiteElementNodes[i] for i in order])
