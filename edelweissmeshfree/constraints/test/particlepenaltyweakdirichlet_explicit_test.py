"""Tests of the penalty weak Dirichlet condition of particles in an explicit analysis, where it is a support: it holds
displacement components at zero, measured where the constrained points are now, on both sides."""

import numpy as np
import pytest
from edelweissfe.points.node import Node

from edelweissmeshfree.constraints.particlepenaltyweakdirichtlet import (
    ParticlePenaltyWeakDirichlet,
)
from edelweissmeshfree.constraints.test._residualtangent import ParticleStandIn
from edelweissmeshfree.models.mpmmodel import MPMModel

PENALTY = 1e3
SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


def _particle() -> ParticleStandIn:
    nodes = [Node(label, x) for label, x in enumerate(SQUARE + np.array([0.1, -0.2]))]
    return ParticleStandIn(nodes, SQUARE.mean(axis=0), SQUARE.copy())


def _support(particle, constrain="center") -> ParticlePenaltyWeakDirichlet:
    support = ParticlePenaltyWeakDirichlet(
        "support", MPMModel(2), [particle], "displacement", {0: 0.0}, PENALTY, constrain=constrain
    )
    support.updateConnectivity(None)
    return support


def _evaluate(support) -> np.ndarray:
    P = np.zeros(support.nDof)
    support.computeKernelsExplicit(np.zeros(support.nDof), np.zeros(support.nDof), P, None)
    return P.reshape(-1, 2)


@pytest.mark.parametrize("displacement", [0.1, -0.05], ids=["moved away", "pushed in"])
def test_the_support_pulls_a_displaced_point_back_on_both_sides(displacement):
    particle = _particle()
    support = _support(particle)

    particle.center = particle.center + np.array([displacement, 0.3])
    P = _evaluate(support)

    # the residual contribution is the negative of the force, which pulls the point back: its components sum to k u
    # in the constrained direction, and nothing in the free one
    assert P[:, 0].sum() == pytest.approx(PENALTY * displacement)
    np.testing.assert_array_equal(P[:, 1], 0.0)
    np.testing.assert_allclose(P[:, 0], PENALTY * displacement * particle.getInterpolationVector(particle.center))


def test_the_support_at_vertices_holds_each_vertex_where_it_was():
    particle = _particle()
    support = _support(particle, constrain=[0, 3])

    particle.vertices = SQUARE + np.array([[0.02, 0.0], [0.5, 0.0], [0.5, 0.0], [-0.01, 0.0]])
    P = _evaluate(support)

    assert P[:, 0].sum() == pytest.approx(PENALTY * (0.02 - 0.01))


def test_a_support_acts_in_an_explicit_analysis_and_a_prescribed_change_does_not():
    model = MPMModel(2)

    assert ParticlePenaltyWeakDirichlet(
        "c", model, [_particle()], "displacement", {0: 0.0}, PENALTY
    ).actsInExplicitAnalysis
    assert not ParticlePenaltyWeakDirichlet(
        "c", model, [_particle()], "displacement", {0: 0.0, 1: 0.1}, PENALTY
    ).actsInExplicitAnalysis
    assert not ParticlePenaltyWeakDirichlet(
        "c", model, [_particle()], "rotation", {0: 0.0}, PENALTY
    ).actsInExplicitAnalysis
