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
"""The contact of particle faces with a discrete rigid body: EdelweissFE's rigid-body contact
(:mod:`edelweissfe.constraints.surfacetodiscreterigidbodypenalty`) with
:class:`~edelweissmeshfree.constraints.base.contactpointsonparticlefaces.ContactPointsOnParticleFaces`, the same contact
points as against a deformable master. The particles are stand-ins with the interpolation of a real particle
linearized about its center (affine in the point); the rigid body is a box whose top face is the support."""

import os

import numpy as np
import pytest
import pyvista as pv
from edelweissfe.constraints.surfacetodiscreterigidbodypenalty import (
    Constraint as RigidSurfaceContact,
)
from edelweissfe.constraints.surfacetodiscreterigidbodypenalty import (
    SurfaceToDiscreteRigidBodyPenaltySchema,
)
from edelweissfe.fields.nodefield import NodeField
from edelweissfe.generators.discreterigidbodygenerator import (
    generateDiscreteRigidBodyFromMeshFile,
)
from edelweissfe.points.node import Node
from edelweissfe.sets.nodeset import NodeSet
from edelweissfe.utils.residualtangentcheck import (
    finiteDifferenceTangent,
    residualContributionAndTangent,
)

from edelweissmeshfree.constraints.base.contactpointsonparticlefaces import (
    ContactPointsOnParticleFaces,
)
from edelweissmeshfree.constraints.test._residualtangent import ParticleStandIn
from edelweissmeshfree.models.mpmmodel import MPMModel

PENALTY = 1e3
SEARCH = 0.2
SIZE = 0.25
#: The bottom face of a hexahedral particle, in the Abaqus numbering of its faces.
BOTTOM = 1


class JournalStandIn:
    """Collects what the constraint reports."""

    def __init__(self):
        self.messages = []

    def message(self, text, senderIdentification, level=1):
        self.messages.append(text)


def _rigidSupport(directory) -> tuple:
    """A model with a rigid box below z = 0, its top face the support, and its reference point at (0.5, 0.5, 0) with a
    displacement and a rotation field.

    Returns
    -------
    tuple
        The model and the rigid body.
    """

    stlFile = os.path.join(directory, "support.stl")
    pv.Box(bounds=(-1.0, 2.0, -1.0, 2.0, -1.0, 0.0)).triangulate().save(stlFile)

    model = MPMModel(3)
    journal = JournalStandIn()
    with model.topology.changes():
        rigidBody = generateDiscreteRigidBodyFromMeshFile(
            model, journal, "support", stlFile, rpCoordinate=np.array([0.5, 0.5, 0.0])
        )
    rp = rigidBody.rpNode
    for fieldName in ("displacement", "rotation"):
        rp.fields[fieldName] = 3
        model.nodeFields[fieldName] = NodeField(fieldName, 3, NodeSet("rp", [rp]))
    return model, rigidBody


def _particle(x0: float, y0: float, z0: float, number: int = 1, angle: float = 0.0) -> ParticleStandIn:
    """A cubic particle stand-in of the side SIZE with its lowest vertex at (x0, y0, z0), tilted by ``angle`` about the
    y axis through that vertex, on eight kernel nodes around it."""

    rotation = np.array([[np.cos(angle), 0.0, np.sin(angle)], [0.0, 1.0, 0.0], [-np.sin(angle), 0.0, np.cos(angle)]])
    corners = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float
    )
    vertices = np.array([x0, y0, z0]) + (SIZE * corners) @ rotation.T
    center = vertices.mean(axis=0)
    offsets = (corners - 0.5) * np.array([1.4, 1.5, 1.6]) * SIZE
    nodes = [Node(100 * number + i, center + offset) for i, offset in enumerate(offsets)]
    particle = ParticleStandIn(nodes, center, vertices)
    particle.number = number
    return particle


def _move(particle: ParticleStandIn, offset):
    particle.vertices = particle.vertices + np.asarray(offset, dtype=float)
    particle.center = particle.center + np.asarray(offset, dtype=float)


def _contact(model, rigidBody, particles, nQuadraturePoints=1, contactType="linear"):
    """The contact of the bottom faces of the particles with the rigid body, searched once."""

    contact = RigidSurfaceContact(
        "contact",
        model,
        ContactPointsOnParticleFaces([(particle, BOTTOM) for particle in particles], model, nQuadraturePoints),
        rigidBody,
        JournalStandIn(),
        configuration=SurfaceToDiscreteRigidBodyPenaltySchema(
            penalty=PENALTY, contactType=contactType, searchDistance=SEARCH
        ),
    )
    contact.updateConnectivity(model)
    return contact


def _residual(contact, U=None) -> np.ndarray:
    U = np.zeros(contact.nDof) if U is None else U
    P = np.zeros(contact.nDof)
    contact.computeKernelsExplicit(U, U, P, None)
    return P


@pytest.mark.parametrize("nQuadraturePoints", [1, 4])
def test_a_face_pressed_into_a_rigid_body_transmits_the_penalty_times_its_area_times_the_overlap(
    nQuadraturePoints, tmp_path
):
    """A face overlapping the rigid support uniformly by d: the pressure is the penalty times d at every point, the
    force the pressure times the face's area, whatever the rule; it pushes the particle up and the rigid body down,
    equal and opposite, with the moment of the force about the reference point."""

    model, rigidBody = _rigidSupport(tmp_path)
    particle = _particle(0.3, 0.6, -0.02)
    contact = _contact(model, rigidBody, [particle], nQuadraturePoints)

    assert contact.nDof == 8 * 3 + 6, "the eight kernel nodes of the particle, then the reference point"
    P = _residual(contact)
    forceOnParticle = -P[:24].reshape(-1, 3).sum(axis=0)
    forceOnRigidBody = -P[24:27]
    momentOnRigidBody = -P[27:]

    np.testing.assert_allclose(contact.getNormalPressures(), PENALTY * 0.02, rtol=1e-6)
    np.testing.assert_allclose(forceOnParticle, [0.0, 0.0, PENALTY * 0.02 * SIZE**2], rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(forceOnParticle + forceOnRigidBody, 0.0, atol=1e-12)
    assert contact.totalNormalForce == pytest.approx(-PENALTY * 0.02 * SIZE**2, rel=1e-6)

    # the force acts at the center of the face (uniform pressure), its moment about the reference point is r x F
    faceCenter = particle.vertices[[0, 1, 2, 3]].mean(axis=0)
    np.testing.assert_allclose(
        momentOnRigidBody, np.cross(faceCenter - [0.5, 0.5, 0.0], forceOnRigidBody), rtol=1e-6, atol=1e-12
    )


@pytest.mark.parametrize("contactType", ["linear", "quadratic"])
def test_the_tangent_is_the_derivative_of_the_residual_contribution(contactType, tmp_path):
    """``K == dP/dU`` by central differences, for the tilted faces of two particles against a rigid body that has moved
    and rotated, whose points stay in contact; the particle points move with the increment, the rigid body with its
    total displacement and rotation. The rotation-rotation block of the reference point is omitted by design."""

    model, rigidBody = _rigidSupport(tmp_path)
    particles = [_particle(0.12, 0.3, -0.03, 1, angle=0.05), _particle(0.45, 0.4, -0.02, 2, angle=-0.03)]
    contact = _contact(model, rigidBody, particles, nQuadraturePoints=4, contactType=contactType)

    rng = np.random.default_rng(1)
    dU = 1e-3 * rng.standard_normal(contact.nDof)
    dU[-3:] = [0.01, -0.005, 0.008]
    # a total that is not the increment: far from it at the nodes of the kernel functions, which the contact does not
    # read there, slightly off at the reference point, which it reads
    nSlaveDof = contact.nDof - 6
    U = dU.copy()
    U[:nSlaveDof] += 0.3 * rng.standard_normal(nSlaveDof)
    U[nSlaveDof:] += [0.002, -0.001, 0.001, 0.002, 0.001, -0.002]

    P, K = residualContributionAndTangent(contact, U, dU)
    dP_dU = finiteDifferenceTangent(contact, U, dU, h=1e-8)
    assert np.all(contact.getGaps() < 0.0), "every contact point must stay closed"
    rotation = slice(contact.nDof - 3, contact.nDof)
    K[rotation, rotation] = 0.0
    dP_dU[rotation, rotation] = 0.0
    np.testing.assert_allclose(K, dP_dU, atol=1e-6 * np.abs(K).max())


def test_a_particle_out_of_reach_takes_no_part_until_it_approaches(tmp_path):
    """A particle farther than the search distance from the rigid body is not in the block of the contact; once it has
    approached, the search locates its points and the block takes the nodes of its kernel functions."""

    model, rigidBody = _rigidSupport(tmp_path)
    particle = _particle(0.3, 0.6, 0.5)
    contact = _contact(model, rigidBody, [particle])
    assert contact.nDof == 6
    np.testing.assert_array_equal(_residual(contact), 0.0)

    _move(particle, [0.0, 0.0, -0.45])
    assert contact.updateConnectivity(model), "the nodes of the contact changed"
    assert contact.nDof == 8 * 3 + 6
    np.testing.assert_array_equal(_residual(contact), 0.0)
    np.testing.assert_allclose(contact.getGaps(), 0.05, rtol=1e-6)


def test_the_restart_locates_the_points_once_the_particles_are_connected(tmp_path):
    """A contact restored from a checkpoint before the particles are connected to their kernel functions (a resumed run
    connects them at its first update of the connectivity) keeps the projection and locates its points then, with the
    same forces."""

    model, rigidBody = _rigidSupport(tmp_path)
    particle = _particle(0.3, 0.6, -0.02)
    contact = _contact(model, rigidBody, [particle])
    forces = _residual(contact)

    kernelFunctions, particle.kernelFunctions = particle.kernelFunctions, None
    restored = RigidSurfaceContact(
        "contact",
        model,
        ContactPointsOnParticleFaces([(particle, BOTTOM)], model, 1),
        rigidBody,
        JournalStandIn(),
        configuration=SurfaceToDiscreteRigidBodyPenaltySchema(penalty=PENALTY, searchDistance=SEARCH),
    )
    restored.setRestartData(contact.getRestartData())
    assert restored.nDof == 6, "not located before the particles are connected"

    particle.kernelFunctions = kernelFunctions
    assert restored.updateConnectivity(model)
    np.testing.assert_array_equal(_residual(restored), forces)
