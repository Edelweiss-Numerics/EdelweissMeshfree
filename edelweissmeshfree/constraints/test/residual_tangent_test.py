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
"""The constraints of EdelweissMeshfree follow EdelweissFE's rule: each adds a residual contribution ``P`` and its
tangent ``K = dP/dU``.

- Every constraint that adds a tangent is checked by central differences of its own residual contribution, at a state
  where every switch (contact or not, a multiplier in tension or not) stays on one side under the perturbation. A
  ``P`` of the wrong sign fails this check as surely as a wrong ``K``.
- A constraint for explicit analyses only has no tangent; its residual contribution is compared with that of its
  implicit counterpart in the same state, which the check above covers.
- Every constraint is restartable: it declares the state it carries, here none.

The particles, cells and material points are stand-ins with fixed shape functions, so that no Marmot is needed.
"""

import numpy as np
import pytest
from edelweissfe.points.node import Node
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.variables.scalarvariable import ScalarVariable

from edelweissmeshfree.constraints.explicit.particlepenaltycartesianboundaryexplicit import (
    ParticlePenaltyCartesianBoundaryConstraintExplicit,
)
from edelweissmeshfree.constraints.explicit.particlepenaltyrigidbodycontactexplicit import (
    ParticlePenaltyContactImplicitSurfaceConstraintExplicit,
)
from edelweissmeshfree.constraints.particlelagrangiancartesianboundary import (
    ParticleLagrangianContactCartesianBoundaryConstraint,
)
from edelweissmeshfree.constraints.particlelagrangianequalvalue import (
    ParticleLagrangianEqualValueConstraint,
)
from edelweissmeshfree.constraints.particlelagrangianweakdirichlet import (
    ParticleLagrangianWeakDirichlet,
)
from edelweissmeshfree.constraints.particlemortarweakdirichlet import (
    ParticleMortarWeakDirichlet,
)
from edelweissmeshfree.constraints.particlepenaltycartesianboundary import (
    ParticlePenaltyContactCartesianBoundaryConstraint,
)
from edelweissmeshfree.constraints.particlepenaltyequalvalue import (
    ParticlePenaltyEqualValue,
)
from edelweissmeshfree.constraints.particlepenaltyrigidbodycontact import (
    ParticlePenaltyContactImplicitSurfaceConstraint,
)
from edelweissmeshfree.constraints.particlepenaltyweakdirichtlet import (
    ParticlePenaltyWeakDirichlet,
)
from edelweissmeshfree.constraints.penaltyconstrainmp2node import (
    PenaltyConstrainMP2Node,
)
from edelweissmeshfree.constraints.penaltyequalvalue import PenaltyEqualValue
from edelweissmeshfree.constraints.penaltyweakdirichlet import PenaltyWeakDirichlet
from edelweissmeshfree.constraints.test._residualtangent import (
    CellStandIn,
    MaterialPointStandIn,
    ParticleStandIn,
    finiteDifferenceTangent,
    residualContributionAndTangent,
)
from edelweissmeshfree.models.mpmmodel import MPMModel

FIELD = "displacement"
PENALTY = 1e3
TIMESTEP = TimeStep(3, 0.1, 0.6, 0.1, 0.6, 0.6)


def _nodes(*coordinates) -> list[Node]:
    return [Node(label, np.array(x, dtype=float)) for label, x in enumerate(coordinates, start=1)]


def _square(center, halfWidth: float) -> np.ndarray:
    cx, cy = center
    return np.array(
        [
            [cx - halfWidth, cy - halfWidth],
            [cx + halfWidth, cy - halfWidth],
            [cx + halfWidth, cy + halfWidth],
            [cx - halfWidth, cy + halfWidth],
        ]
    )


def _particle(nodes, center) -> ParticleStandIn:
    return ParticleStandIn(nodes, center, _square(center, 0.1))


def _withMultipliers(constraint):
    constraint.assignAdditionalScalarVariables(
        [ScalarVariable() for i in range(constraint.getNumberOfAdditionalNeededScalarVariables())]
    )
    return constraint


def _connected(constraint):
    constraint.updateConnectivity(None)
    return constraint


@pytest.fixture
def model():
    return MPMModel(2)


#: A grid of nine nodes, and particles whose kernel functions reach overlapping subsets of them.
GRID = _nodes(*[(x, y) for y in (0.0, 0.5, 1.0) for x in (0.0, 0.5, 1.0)])
PARTICLES = [
    _particle([GRID[0], GRID[1], GRID[3], GRID[4]], (0.2, 0.3)),
    _particle([GRID[1], GRID[2], GRID[4], GRID[5]], (0.7, 0.2)),
    _particle([GRID[3], GRID[4], GRID[6], GRID[7]], (0.35, 0.8)),
    _particle([GRID[4], GRID[5], GRID[7], GRID[8]], (0.8, 0.65)),
]
CELLS = [CellStandIn([GRID[0], GRID[1], GRID[4], GRID[3]]), CellStandIn([GRID[1], GRID[2], GRID[5], GRID[4]])]
MATERIAL_POINTS = [
    MaterialPointStandIn((0.2, 0.25), [CELLS[0]]),
    MaterialPointStandIn((0.5, 0.3), [CELLS[0], CELLS[1]]),
    MaterialPointStandIn((0.85, 0.1), [CELLS[1]]),
]


def _circle(center, radius):
    """The signed distance to a circle, negative inside, and its gradient."""

    center = np.asarray(center, dtype=float)

    def distance(x):
        return np.linalg.norm(x - center) - radius

    def gradient(x):
        return (x - center) / np.linalg.norm(x - center)

    return distance, gradient


#: name -> (a function building the connected constraint from the model, a function giving the state dU)
CONSTRAINTS = {
    "PenaltyWeakDirichlet": (
        lambda model: PenaltyWeakDirichlet("c", model, MATERIAL_POINTS, FIELD, {0: 0.1, 1: -0.05}, PENALTY),
        None,
    ),
    "PenaltyEqualValue": (
        lambda model: PenaltyEqualValue("c", model, MATERIAL_POINTS, FIELD, 1, PENALTY),
        None,
    ),
    "PenaltyConstrainMP2Node": (
        lambda model: PenaltyConstrainMP2Node("c", model, MATERIAL_POINTS[1], GRID[8], FIELD, [0, 1], PENALTY),
        None,
    ),
    "ParticlePenaltyWeakDirichlet": (
        lambda model: ParticlePenaltyWeakDirichlet("c", model, PARTICLES[:3], FIELD, {0: 0.1, 1: -0.05}, PENALTY),
        None,
    ),
    "ParticlePenaltyWeakDirichlet at vertices": (
        lambda model: ParticlePenaltyWeakDirichlet(
            "c", model, PARTICLES[1:], FIELD, {1: 0.02}, PENALTY, constrain=[0, 2]
        ),
        None,
    ),
    "ParticlePenaltyEqualValue": (
        lambda model: ParticlePenaltyEqualValue("c", model, PARTICLES, FIELD, [0, 1], PENALTY),
        None,
    ),
    "ParticleLagrangianWeakDirichlet": (
        lambda model: ParticleLagrangianWeakDirichlet("c", PARTICLES[3], FIELD, {0: 0.1, 1: -0.05}, model),
        None,
    ),
    "ParticleLagrangianWeakDirichlet on a face": (
        lambda model: ParticleLagrangianWeakDirichlet("c", PARTICLES[0], FIELD, {1: 0.03}, model, "face", faceID=2),
        None,
    ),
    "ParticleLagrangianEqualValueConstraint": (
        lambda model: ParticleLagrangianEqualValueConstraint("c", PARTICLES[0], [0, 1], PARTICLES[3], FIELD, model),
        None,
    ),
    "ParticleMortarWeakDirichlet": (
        lambda model: ParticleMortarWeakDirichlet("c", PARTICLES[:3], FIELD, {0: 0.1, 1: 0.02}, model, 1),
        None,
    ),
    # the boundary at x = 0.1 left of the particle at x = 0.2; penetrating by 0.05, with a multiplier in compression
    "ParticleLagrangianContactCartesianBoundaryConstraint in contact": (
        lambda model: ParticleLagrangianContactCartesianBoundaryConstraint("c", PARTICLES[0], 0, 0.1, model),
        lambda constraint: _gapState(constraint, 0, -0.15, 2.0),
    ),
    # the same, separating: the multiplier is held at zero
    "ParticleLagrangianContactCartesianBoundaryConstraint separated": (
        lambda model: ParticleLagrangianContactCartesianBoundaryConstraint("c", PARTICLES[0], 0, 0.1, model),
        lambda constraint: _gapState(constraint, 0, 0.05, 2.0),
    ),
    # the boundary at y = 0.75 above the particle at y = 0.65, penetrating by 0.05
    "ParticlePenaltyContactCartesianBoundaryConstraint": (
        lambda model: ParticlePenaltyContactCartesianBoundaryConstraint(
            "c", PARTICLES[3], 1, 0.75, model, penaltyParameter=PENALTY
        ),
        lambda constraint: _gapState(constraint, 1, 0.15, 0.0),
    ),
    # a circle of radius 0.12 around the particle's first vertex, which lies inside it
    "ParticlePenaltyContactImplicitSurfaceConstraint": (
        lambda model: ParticlePenaltyContactImplicitSurfaceConstraint(
            "c",
            PARTICLES[2],
            *_circle(PARTICLES[2].vertices[0] + (0.05, 0.02), 0.12),
            model,
            location="vertex",
            vertexIDs=[0],
            penaltyParameter=PENALTY,
        ),
        None,
    ),
}


def _gapState(constraint, component: int, displacement: float, multiplier: float) -> np.ndarray:
    """A state in which every node moves by ``displacement`` in the constrained ``component``, with the multiplier
    (if any) at ``multiplier``."""

    nNodes = len(constraint.nodes)
    dU = np.zeros(constraint.nDof)
    dU[component : 2 * nNodes : 2] = displacement
    if constraint.nDof > 2 * nNodes:
        dU[2 * nNodes :] = multiplier
    return dU


def _build(name, model):
    build, state = CONSTRAINTS[name]
    constraint = _connected(_withMultipliers(build(model)))
    dU = state(constraint) if state else 0.02 * np.random.default_rng(len(name)).standard_normal(constraint.nDof)
    return constraint, dU


@pytest.mark.parametrize("name", CONSTRAINTS)
def test_the_tangent_is_the_derivative_of_the_residual_contribution(name, model):
    constraint, dU = _build(name, model)

    P, K = residualContributionAndTangent(constraint, dU, dU, TIMESTEP)
    dP_dU = finiteDifferenceTangent(constraint, dU, TIMESTEP)

    assert np.abs(K).max() > 0.0, "the check is vacuous without a tangent"
    np.testing.assert_allclose(K, dP_dU, rtol=1e-6, atol=1e-6 * np.abs(K).max())


@pytest.mark.parametrize("name", CONSTRAINTS)
def test_a_constraint_carries_no_state_between_increments(name, model):
    constraint, _ = _build(name, model)

    assert constraint.getRestartData() == {}
    constraint.setRestartData({})


def _movedParticle(center) -> ParticleStandIn:
    """A particle that started at (0.8, 0.65), on one side of the boundaries below, and has since moved to
    ``center``."""

    particle = _particle([GRID[4], GRID[5], GRID[7], GRID[8]], (0.8, 0.65))
    particle.center[:] = center
    particle.vertices[:] = _square(center, 0.1)
    return particle


def test_explicit_cartesian_boundary_acts_like_the_implicit_one(model):
    """At the start of an increment (no displacement yet), the explicit penalty boundary adds the residual
    contribution its implicit counterpart adds, so it follows the same sign rule."""

    explicitParticle = _movedParticle((0.8, 0.65))
    explicit = ParticlePenaltyCartesianBoundaryConstraintExplicit(
        "explicit", explicitParticle, 1, 0.75, model, penaltyParameter=PENALTY
    )
    implicitParticle = _movedParticle((0.8, 0.65))
    implicit = ParticlePenaltyContactCartesianBoundaryConstraint(
        "implicit", implicitParticle, 1, 0.75, model, penaltyParameter=PENALTY
    )
    for particle, constraint in ((explicitParticle, explicit), (implicitParticle, implicit)):
        particle.center[:] = (0.8, 0.8)
        _connected(constraint)

    zero = np.zeros(implicit.nDof)
    PImplicit, _ = residualContributionAndTangent(implicit, zero, zero, TIMESTEP)
    PExplicit = np.zeros(explicit.nDof)
    explicit.computeKernelsExplicit(zero, zero, PExplicit, TIMESTEP)

    assert np.abs(PImplicit).max() > 0.0, "the particle must penetrate the boundary"
    np.testing.assert_allclose(PExplicit, PImplicit, rtol=1e-14)


def test_explicit_surface_contact_acts_like_the_implicit_one(model):
    """The same for the penalty contact with a surface given by its signed distance."""

    distance, gradient = _circle((0.95, 0.8), 0.2)
    constraints = []
    for constraintType in (
        ParticlePenaltyContactImplicitSurfaceConstraintExplicit,
        ParticlePenaltyContactImplicitSurfaceConstraint,
    ):
        particle = _movedParticle((0.8, 0.65))
        constraints.append(
            _connected(
                constraintType(
                    "c",
                    particle,
                    distance,
                    gradient,
                    model,
                    location="vertex",
                    vertexIDs=[2],
                    penaltyParameter=PENALTY,
                    doProximityCheck=False,
                )
            )
        )
    explicit, implicit = constraints

    zero = np.zeros(implicit.nDof)
    PImplicit, _ = residualContributionAndTangent(implicit, zero, zero, TIMESTEP)
    PExplicit = np.zeros(explicit.nDof)
    explicit.computeKernelsExplicit(zero, zero, PExplicit, TIMESTEP)

    assert np.abs(PImplicit).max() > 0.0, "the particle must penetrate the surface"
    np.testing.assert_allclose(PExplicit, PImplicit, rtol=1e-14)


@pytest.mark.parametrize(
    "constraintType",
    [ParticlePenaltyCartesianBoundaryConstraintExplicit, ParticlePenaltyContactImplicitSurfaceConstraintExplicit],
)
def test_an_explicit_constraint_refuses_an_implicit_analysis(constraintType, model):
    if constraintType is ParticlePenaltyCartesianBoundaryConstraintExplicit:
        constraint = constraintType("c", _movedParticle((0.8, 0.65)), 1, 0.75, model)
    else:
        constraint = constraintType("c", _movedParticle((0.8, 0.65)), *_circle((0.95, 0.8), 0.2), model)
    _connected(constraint)

    zero = np.zeros(constraint.nDof)
    with pytest.raises(TypeError, match="explicit"):
        constraint.computeKernels(zero, zero, zero, np.zeros((constraint.nDof, constraint.nDof)), TIMESTEP)
