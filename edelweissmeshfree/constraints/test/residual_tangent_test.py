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
  where every switch (contact or not, a multiplier in tension or not) stays on one side under the perturbation. This
  catches a wrong ``K``, and a ``P`` whose sign alone is wrong; a ``P`` and a ``K`` negated together pass it.
- Every constraint reads the nodal values as increments, from ``dU``, and its multipliers, if any, as totals, from
  ``U_np``. The states of the checks come from an increment that starts from a converged state that is not zero, so
  that reading the wrong one shows.
- A constraint for explicit analyses only has no tangent; its residual contribution is compared with that of its
  implicit counterpart in the same state, which the check above covers.
- The check above holds at whatever point of a particle a constraint acts. A constraint that acts at a point the user
  names -- the center of the particle, or one of its vertices -- is also checked to measure its condition there.
- Every constraint is restartable: it declares the state it carries, here none.

The particles, cells and material points are stand-ins with fixed shape functions, so that no Marmot is needed.
"""

import numpy as np
import pytest
from edelweissfe.points.node import Node
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.utils.residualtangentcheck import (
    finiteDifferenceTangent,
    residualContributionAndTangent,
)
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
    assertIncrementsAtNodesAndTotalMultipliersAreRead,
    stateInAnIncrement,
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
        [ScalarVariable("lambda_{:}".format(i)) for i in range(constraint.getNumberOfAdditionalNeededScalarVariables())]
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


#: name -> (a function building the connected constraint from the model, a function giving the state: the increments
#: at the nodes, followed by the totals of the multipliers)
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
    "ParticleLagrangianWeakDirichlet at a vertex": (
        lambda model: ParticleLagrangianWeakDirichlet("c", PARTICLES[1], FIELD, {1: 0.02}, model, "vertex", vertexID=2),
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
    # the boundary at x = 0.05 left of the particle's first vertex at x = 0.1; penetrating by 0.03
    "ParticleLagrangianContactCartesianBoundaryConstraint at a vertex in contact": (
        lambda model: ParticleLagrangianContactCartesianBoundaryConstraint(
            "c", PARTICLES[0], 0, 0.05, model, location="vertex", vertexID=0
        ),
        lambda constraint: _gapState(constraint, 0, -0.08, 2.0),
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
    # a circle of radius 0.08 around a point next to the particle's center: the center lies inside it by 0.026, at the
    # start of the increment
    "ParticlePenaltyContactImplicitSurfaceConstraint at the center": (
        lambda model: ParticlePenaltyContactImplicitSurfaceConstraint(
            "c",
            PARTICLES[2],
            *_circle(PARTICLES[2].center + (0.05, 0.02), 0.08),
            model,
            location="center",
            penaltyParameter=PENALTY,
        ),
        lambda constraint: np.zeros(constraint.nDof),
    ),
}


def _gapState(constraint, component: int, displacement: float, multiplier: float) -> np.ndarray:
    """A state in which every node moves by the increment ``displacement`` in the constrained ``component``, with the
    multiplier (if any) at the total ``multiplier``."""

    nNodes = len(constraint.nodes)
    state = np.zeros(constraint.nDof)
    state[component : 2 * nNodes : 2] = displacement
    if constraint.nDof > 2 * nNodes:
        state[2 * nNodes :] = multiplier
    return state


def _build(name, model):
    """The connected constraint, and ``U_np`` and ``dU`` of its state."""

    build, state = CONSTRAINTS[name]
    constraint = _connected(_withMultipliers(build(model)))
    seed = len(name)
    incrementsAndMultipliers = (
        state(constraint) if state else 0.02 * np.random.default_rng(seed).standard_normal(constraint.nDof)
    )
    return constraint, *stateInAnIncrement(constraint, incrementsAndMultipliers, seed + 1)


@pytest.mark.parametrize("name", CONSTRAINTS)
def test_the_tangent_is_the_derivative_of_the_residual_contribution(name, model):
    constraint, U_np, dU = _build(name, model)

    P, K = residualContributionAndTangent(constraint, U_np, dU, TIMESTEP)
    dP_dU = finiteDifferenceTangent(constraint, U_np, dU, TIMESTEP)

    assert np.abs(K).max() > 0.0, "the check is vacuous without a tangent"
    np.testing.assert_allclose(K, dP_dU, rtol=1e-6, atol=1e-6 * np.abs(K).max())


@pytest.mark.parametrize("name", CONSTRAINTS)
def test_a_constraint_reads_increments_at_the_nodes_and_total_multipliers(name, model):
    constraint, U_np, dU = _build(name, model)

    assertIncrementsAtNodesAndTotalMultipliersAreRead(constraint, U_np, dU, TIMESTEP)


@pytest.mark.parametrize("name", CONSTRAINTS)
def test_a_constraint_carries_no_state_between_increments(name, model):
    constraint, *_ = _build(name, model)

    assert constraint.getRestartData() == {}
    constraint.setRestartData({})


def test_lagrangian_weak_dirichlet_at_a_vertex_is_met_by_a_translation_by_the_prescribed_increment(model):
    """Moving every node by the share of the prescribed increment that falls into the increment moves every point of
    the particle by it, as the shape functions are a partition of unity: the constraint is met at the vertex. The
    tangent is symmetric, as the constraint stems from the potential ``sum(lambda g)``. The reaction force holds the
    multipliers -- the totals -- per component, and balances the force the constraint exerts on the nodes."""

    prescribedStepDelta = {0: 0.1, 1: -0.05}
    constraint = _connected(
        _withMultipliers(
            ParticleLagrangianWeakDirichlet("c", PARTICLES[1], FIELD, prescribedStepDelta, model, "vertex", vertexID=2)
        )
    )

    nU = 2 * len(constraint.nodes)
    multipliers = np.array([1.0, 2.0])
    state = np.zeros(constraint.nDof)
    for i, prescribedComponent in prescribedStepDelta.items():
        state[i:nU:2] = prescribedComponent * TIMESTEP.stepProgressIncrement
    state[nU:] = multipliers
    U_np, dU = stateInAnIncrement(constraint, state, 1)

    P, K = residualContributionAndTangent(constraint, U_np, dU, TIMESTEP)

    np.testing.assert_allclose(P[nU:], 0.0, atol=1e-15)
    np.testing.assert_array_equal(K, K.T)
    # the multipliers are ordered as the components in prescribedStepDelta
    np.testing.assert_array_equal(constraint.reactionForce, multipliers)
    np.testing.assert_allclose(P[:nU].reshape(-1, 2).sum(axis=0), constraint.reactionForce, rtol=1e-13)


def test_lagrangian_weak_dirichlet_at_a_vertex_measures_the_field_at_the_vertex(model):
    """With the nodes moved unevenly, the field differs from point to point of the particle: the constraint adds to the
    row of its multiplier the field interpolated at the vertex, less the prescribed increment, not the field at the
    center."""

    particle = PARTICLES[1]
    constraint = _connected(
        _withMultipliers(ParticleLagrangianWeakDirichlet("c", particle, FIELD, {0: 0.1}, model, "vertex", vertexID=2))
    )

    nodes = list(constraint.nodes)
    state = np.zeros(constraint.nDof)
    state[0 : 2 * len(nodes) : 2] = [0.1 * node.coordinates[0] ** 2 + 0.05 * node.coordinates[1] for node in nodes]
    U_np, dU = stateInAnIncrement(constraint, state, 1)

    P, _ = residualContributionAndTangent(constraint, U_np, dU, TIMESTEP)

    nodalValues = {node: dU[2 * k] for k, node in enumerate(nodes)}
    atVertex = particle.getInterpolationVector(particle.vertices[2]) @ [
        nodalValues[kf.node] for kf in particle.kernelFunctions
    ]
    atCenter = particle.getInterpolationVector(particle.center) @ [
        nodalValues[kf.node] for kf in particle.kernelFunctions
    ]
    assert atVertex != pytest.approx(atCenter, rel=1e-3), "the check is vacuous if the field is even"
    assert P[-1] == pytest.approx(atVertex - 0.1 * TIMESTEP.stepProgressIncrement, rel=1e-12)


def test_lagrangian_cartesian_boundary_at_a_vertex_measures_the_gap_at_the_vertex(model):
    """The boundary at x = 0.05 lies 0.05 left of the particle's first vertex, and 0.15 left of its center. Moved by
    0.08 to the left, the vertex has penetrated the boundary by 0.03, which the constraint adds to the row of its
    multiplier; the center would not have reached it."""

    constraint = _connected(
        _withMultipliers(
            ParticleLagrangianContactCartesianBoundaryConstraint(
                "c", PARTICLES[0], 0, 0.05, model, location="vertex", vertexID=0
            )
        )
    )
    U_np, dU = stateInAnIncrement(constraint, _gapState(constraint, 0, -0.08, 2.0), 1)

    P, _ = residualContributionAndTangent(constraint, U_np, dU, TIMESTEP)

    assert P[-1] == pytest.approx(0.03, rel=1e-12)


def test_penalty_surface_contact_at_the_center_measures_the_penetration_at_the_center(model):
    """A circle of radius 0.08 around a point next to the particle's center contains the center, but none of the
    vertices. At the start of an increment the center lies inside it by the radius less its distance from the center
    of the circle, and the constraint's residual contribution sums to the penalty force, penalty x depth, against the
    outward normal at the center."""

    particle = PARTICLES[2]
    circleCenter, radius = particle.center + (0.05, 0.02), 0.08
    constraint = _connected(
        ParticlePenaltyContactImplicitSurfaceConstraint(
            "c", particle, *_circle(circleCenter, radius), model, location="center", penaltyParameter=PENALTY
        )
    )

    zero = np.zeros(constraint.nDof)
    P, _ = residualContributionAndTangent(constraint, zero, zero, TIMESTEP)

    distance = np.linalg.norm(particle.center - circleCenter)
    outwardNormal = (particle.center - circleCenter) / distance
    np.testing.assert_allclose(P.reshape(-1, 2).sum(axis=0), -PENALTY * (radius - distance) * outwardNormal, rtol=1e-12)


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
