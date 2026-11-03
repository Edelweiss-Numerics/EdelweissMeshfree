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
"""The explicit meshfree solver carries the velocity of the kernel functions' nodes in the model, by node: a body
translates over a grid of kernel functions that are not bonded to its particles, so that the active nodes change
while it moves. A node active before and after a change keeps its velocity, a node that has just become active starts
at rest, and the model holds the velocity of the active nodes, zero solution, and zero results outside the active
domain."""

import numpy as np
from edelweissfe.journal.journal import Journal
from edelweissfe.steps.adaptivestep import AdaptiveStep
from edelweissfe.steps.stepmanager import StepActionCollection

from edelweissmeshfree.fieldoutput.fieldoutput import MPMFieldOutputController
from edelweissmeshfree.generators.rectangularkernelfunctiongridgenerator import (
    generateRectangularKernelFunctionGrid,
)
from edelweissmeshfree.generators.rectangularquadparticlegridgenerator import (
    generateRectangularQuadParticleGrid,
)
from edelweissmeshfree.meshfree.approximations.marmot.marmotmeshfreeapproximation import (
    MarmotMeshfreeApproximationWrapper,
)
from edelweissmeshfree.meshfree.kernelfunctions.marmot.marmotmeshfreekernelfunction import (
    MarmotMeshfreeKernelFunctionWrapper,
)
from edelweissmeshfree.meshfree.particlekerneldomain import ParticleKernelDomain
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.particlemanagers.kdbinorganizedparticlemanager import (
    KDBinOrganizedParticleManager,
)
from edelweissmeshfree.particles.marmot.marmotparticlewrapper import (
    MarmotParticleWrapper,
)
from edelweissmeshfree.solvers.meshfreeexplicitdynamic import MeshfreeExplicitDynamic

PARTICLE_SIZE = 20.0
KERNEL_SPACING = 28.0
SUPPORT_RADIUS = 1.5 * KERNEL_SPACING
VELOCITY = 1.0e6
TIME_INCREMENT = 1.0e-6
N_INCREMENTS = 40

# soft, so that the body translates nearly rigidly, far within the stable time increment
YOUNGS_MODULUS, POISSONS_RATIO = 1.0, 0.3
BULK_MODULUS = YOUNGS_MODULUS / (3 * (1 - 2 * POISSONS_RATIO))
SHEAR_MODULUS = YOUNGS_MODULUS / (2 * (1 + POISSONS_RATIO))
# K, G, yield stress (never reached), ..., mass density
MATERIAL = {
    "material": "FiniteStrainJ2Plasticity",
    "properties": np.array([BULK_MODULUS, SHEAR_MODULUS, 1e9, 1e9, 1, 0, 1, 1.0e-8]),
}


class _SolverRecordingTheVelocityAtReconnection(MeshfreeExplicitDynamic):
    """The explicit solver, recording the velocity of the active nodes before and after every reconnection that
    changes the active nodes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.activeNodeChanges = []

    def _reconnect(self, model):
        before = self.velocityOfTheActiveNodes()
        super()._reconnect(model)
        after = self.velocityOfTheActiveNodes()
        if set(before) != set(after):
            self.activeNodeChanges.append((before, after))

    def velocityOfTheActiveNodes(self) -> dict:
        dofManager = self._layout.dofManager
        return {
            node: np.array(self._V[dofManager.idcsOfFieldVariablesInDofVector[node.fields["displacement"]]])
            for node in self._layout.activeNodes
        }


def _translatingBody(journal: Journal) -> MPMModel:
    """A block of particles moving to the right, over a grid of kernel functions that reaches beyond it."""

    dimension = 2
    model = MPMModel(dimension)

    def kernelFunctionFactory(node):
        return MarmotMeshfreeKernelFunctionWrapper(
            node, "BSplineBoxed", supportRadius=SUPPORT_RADIUS, continuityOrder=3
        )

    generateRectangularKernelFunctionGrid(
        model,
        journal,
        kernelFunctionFactory,
        x0=-3 * KERNEL_SPACING,
        y0=-3 * KERNEL_SPACING,
        l=12 * KERNEL_SPACING,
        h=10 * KERNEL_SPACING,
        nX=12,
        nY=10,
        name="grid",
    )

    approximation = MarmotMeshfreeApproximationWrapper("ReproducingKernel", dimension, completenessOrder=1)

    def particleFactory(number, vertexCoordinates, volume):
        return MarmotParticleWrapper(
            "Displacement/SQCNIxNSNI/PlaneStrain/Quad", number, vertexCoordinates, 0.0, approximation, MATERIAL
        )

    generateRectangularQuadParticleGrid(
        model,
        journal,
        particleFactory,
        x0=0.0,
        y0=0.0,
        l=5 * PARTICLE_SIZE,
        h=5 * PARTICLE_SIZE,
        nX=5,
        nY=5,
        name="body",
    )

    particleKernelDomain = ParticleKernelDomain(
        list(model.particles.values()), list(model.meshfreeKernelFunctions.values())
    )
    model.particleKernelDomains["body"] = particleKernelDomain
    model.particleManagers.append(
        KDBinOrganizedParticleManager(particleKernelDomain, dimension, journal, bondParticlesToKernelFunctions=False)
    )

    model.prepareYourself(journal)

    for particle in model.particles.values():
        particle.getResultArray("velocity")[0] = VELOCITY
        particle.acceptStateAndPosition()

    return model


def test_the_velocity_is_carried_by_node_across_a_change_of_the_active_nodes():
    journal = Journal()
    model = _translatingBody(journal)
    solver = _SolverRecordingTheVelocityAtReconnection({}, journal)

    fieldOutputController = MPMFieldOutputController(model, journal)
    fieldOutputController.initializeJob()
    stepLength = N_INCREMENTS * TIME_INCREMENT
    step = AdaptiveStep(
        0,
        model,
        fieldOutputController,
        journal,
        {},
        solver,
        [],
        StepActionCollection(),
        stepLength=stepLength,
        startInc=1.0 / N_INCREMENTS,
        maxInc=1.0 / N_INCREMENTS,
        minInc=1.0 / N_INCREMENTS,
        maxNumInc=2 * N_INCREMENTS,
    )
    try:
        step.solve()
    finally:
        fieldOutputController.finalizeJob()

    newlyActiveNodes = [set(after) - set(before) for before, after in solver.activeNodeChanges]
    assert any(newlyActiveNodes), "the body did not move far enough for a node to become active"
    for before, after in solver.activeNodeChanges:
        for node, velocity in after.items():
            if node in before:
                np.testing.assert_array_equal(velocity, before[node])
            else:
                np.testing.assert_array_equal(velocity, 0.0)

    # the model holds the velocity of the active nodes, the solver's; zero elsewhere, and no solution anywhere
    displacement = model.nodeFields["displacement"]
    velocityOfTheActiveNodes = solver.velocityOfTheActiveNodes()
    for node in displacement.nodes:
        row = displacement.indexOfNode(node)
        if node in velocityOfTheActiveNodes:
            np.testing.assert_array_equal(displacement["V"][row], velocityOfTheActiveNodes[node])
        else:
            for entry in ("V", "P"):
                np.testing.assert_array_equal(displacement[entry][row], 0.0)
    np.testing.assert_array_equal(displacement["U"], 0.0)
    assert np.any(displacement["dU"][:, 0] > 0.0)
