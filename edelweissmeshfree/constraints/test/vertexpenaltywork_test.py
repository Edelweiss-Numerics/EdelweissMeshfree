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
"""A penalty on the vertices of particles is conservative in an explicit analysis: driven around a closed cycle of
nodal motion, a support and a facet contact on the vertices of a real particle do no net work.

A vertex moves with its particle, by the particle's interpolation vector of an attached point times the nodal increment,
and the constraints distribute their force with the same vector: the penalty stiffness is symmetric, and the work over
a closed cycle vanishes up to the second-order path dependence of the particle's deformation gradient. Distributing the
force with the shape functions at the vertex instead (which do not move it) makes the stiffness unsymmetric: the cycle
then does work of the order of the penalty energy, which is what pumped energy into the crushed concrete beside the
supports of the FE-particle impact. Skipped where Marmot is not built.
"""

import numpy as np
import pytest
from edelweissfe.elements.contactsurfaceelement import Line2ContactFacet
from edelweissfe.points.node import Node
from edelweissfe.variables.fieldvariable import FieldVariable

from edelweissmeshfree.constraints.explicit.particlepenaltyfacetcontactexplicit import (
    ContactState,
    ParticlePenaltyFacetContactExplicit,
)
from edelweissmeshfree.constraints.particlepenaltyweakdirichtlet import (
    ParticlePenaltyWeakDirichlet,
)
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.models.test.stabletimeincrementofparticles_test import H, _model

PENALTY = 1e6
#: The largest nodal displacement of the cycle; small against the particle size, so that the path dependence of the
#: particle's deformation gradient (second order in it) stays far below the work an unsymmetric stiffness does.
AMPLITUDE = 1e-3 * H
PARTICLE_TYPES = ["DisplacementSQCNIxNSNI/PlaneStrain/Quad", "GradientEnhancedFiniteStrainSQCNIxNSNI/PlaneStrain/Quad"]


class _JournalStandIn:
    def message(self, text, senderIdentification, level=1):
        pass


def _particleOnTheTop(model):
    """A particle in the top row, away from the sides, so that it has a full set of kernel functions."""
    top = max(p.getCenterCoordinates()[1] for p in model.particles.values())
    candidates = [p for p in model.particles.values() if abs(p.getCenterCoordinates()[1] - top) < 1e-6 * H]
    return sorted(candidates, key=lambda p: p.getCenterCoordinates()[0])[len(candidates) // 2]


def _move(particle, displacementOfNodes: np.ndarray):
    """Move a particle by a nodal increment (one row of displacement components per kernel function), as an explicit
    analysis does: take the increment and accept it."""
    nNodes = len(particle.kernelFunctions)
    dU = np.zeros((nNodes, particle.nDof // nNodes))
    dU[:, :2] = displacementOfNodes
    particle.updatePhysicsExplicit(np.ascontiguousarray(dU.ravel()), 0.0, 1e-6)
    particle.acceptStateAndPosition()


def _workAndEnergyOverACycle(constraint, particle, nSteps: int = 200) -> tuple[float, float]:
    """Drive the nodes of a particle around the closed cycle u(t) = a (cos t - 1) e1 + a sin t e2 of two random nodal
    fields, and integrate the work of the constraint's force on the nodes (trapezoidal, exact for a constant
    stiffness). Returns the work and the largest penalty energy of the cycle, k/2 times the squared motion of the
    vertices."""

    rng = np.random.default_rng(42)
    nodesOfParticle = [kf.node for kf in particle.kernelFunctions]
    e1, e2 = (rng.uniform(-AMPLITUDE, AMPLITUDE, (len(nodesOfParticle), 2)) for _ in range(2))
    indexInConstraint = [list(constraint.nodes).index(node) for node in nodesOfParticle]

    def forceOnTheNodes():
        P = np.zeros(constraint.nDof)
        constraint.computeKernelsExplicit(np.zeros(constraint.nDof), np.zeros(constraint.nDof), P, None)
        # the residual contribution is the negative of the force
        return -P.reshape(-1, 2)[indexInConstraint]

    verticesAtStart = particle.getVertexCoordinates().copy()
    t = np.linspace(0.0, 2.0 * np.pi, nSteps + 1)
    u = np.cos(t)[:, None, None] * e1 - e1 + np.sin(t)[:, None, None] * e2
    work, energy = 0.0, 0.0
    force = forceOnTheNodes()
    for n in range(nSteps):
        dU = u[n + 1] - u[n]
        _move(particle, dU)
        nextForce = forceOnTheNodes()
        work += 0.5 * np.sum((force + nextForce) * dU)
        force = nextForce
        energy = max(energy, 0.5 * PENALTY * np.sum((particle.getVertexCoordinates() - verticesAtStart) ** 2))
    return work, energy


@pytest.mark.parametrize("particleType", PARTICLE_TYPES)
def test_a_support_on_the_vertices_does_no_work_over_a_closed_cycle(particleType):
    model = _model(particleType, 2.4, 3)
    particle = _particleOnTheTop(model)
    support = ParticlePenaltyWeakDirichlet(
        "support", model, [particle], "displacement", {0: 0.0, 1: 0.0}, PENALTY, constrain=[0, 1, 2, 3]
    )
    support.updateConnectivity(model)

    work, energy = _workAndEnergyOverACycle(support, particle)

    assert energy > 0.0
    assert abs(work) < 1e-3 * energy, f"the support does net work {work:.3e} over a cycle (penalty energy {energy:.3e})"


@pytest.mark.parametrize("particleType", PARTICLE_TYPES)
def test_a_facet_contact_on_the_vertices_does_no_work_over_a_closed_cycle(particleType):
    """The top vertices of a particle are pressed 3 a into a flat, fixed master facet (the bottom of a body above it)
    and driven around the cycle while staying in contact."""

    model = _model(particleType, 2.4, 3)
    particle = _particleOnTheTop(model)
    vertices = particle.getVertexCoordinates()
    top = vertices[:, 1].max()
    left, right = vertices[:, 0].min() - H, vertices[:, 0].max() + H
    standoff = 0.1 * H
    # the bottom face of the master body runs from left to right, so that its normal points down, out of the body
    nodes = [Node(1000 + i, np.array([x, top + standoff])) for i, x in enumerate((left, right))]
    for node in nodes:
        node.fields["displacement"] = FieldVariable(node, "displacement")
    facet = Line2ContactFacet("Line2ContactFacet", 1)
    facet.setNodes(nodes)
    facet.initializeElement()
    contact = ParticlePenaltyFacetContactExplicit(
        "contact", MPMModel(2), [particle], [facet], PENALTY, 0.5 * H, _JournalStandIn(), twoWayCoupling=False
    )

    contact.updateConnectivity(model)
    _move(particle, np.tile([0.0, standoff + 3 * AMPLITUDE], (len(particle.kernelFunctions), 1)))
    contact.updateConnectivity(model)
    inContact = contact.stateOfPoints == ContactState.IN_CONTACT
    assert np.count_nonzero(inContact) == 2

    work, energy = _workAndEnergyOverACycle(contact, particle)

    assert contact.numberOfPointsInContact == 2, "the cycle must keep both top vertices in contact"
    assert abs(work) < 1e-3 * energy, f"the contact does net work {work:.3e} over a cycle (penalty energy {energy:.3e})"
