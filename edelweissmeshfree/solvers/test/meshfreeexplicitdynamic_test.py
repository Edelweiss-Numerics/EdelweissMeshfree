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
from edelweissmeshfree.models.mpmequationsystemlayout import MPMEquationSystemLayout
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


def _material(youngsModulus: float) -> dict:
    """An elastic material (its yield stress is never reached) of the given Young's modulus."""

    bulkModulus = youngsModulus / (3 * (1 - 2 * POISSONS_RATIO))
    shearModulus = youngsModulus / (2 * (1 + POISSONS_RATIO))
    # K, G, yield stress (never reached), ..., mass density
    return {
        "material": "FiniteStrainJ2Plasticity",
        "properties": np.array([bulkModulus, shearModulus, 1e9, 1e9, 1, 0, 1, 1.0e-8]),
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


class _SolverRecordingTheNodesWithoutInertia(_SolverRecordingTheVelocityAtReconnection):
    """The explicit solver, recording at every acceptance the velocity of the active nodes without inertia, and the
    nodes among them that moved before."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.velocityOfNodesWithoutInertia = []
        self.nodesThatLostTheirInertiaWhileMoving = set()

    def acceptIncrement(self, step, model, timeStep):
        velocityBefore = self.velocityOfTheActiveNodes()
        super().acceptIncrement(step, model, timeStep)
        dofManager = self._layout.dofManager
        for node, velocity in self.velocityOfTheActiveNodes().items():
            indices = dofManager.idcsOfFieldVariablesInDofVector[node.fields["displacement"]]
            if np.all(np.asarray(self._Minv)[indices] == 0.0):
                self.velocityOfNodesWithoutInertia.append(velocity)
                if np.any(velocityBefore.get(node, 0.0)):
                    self.nodesThatLostTheirInertiaWhileMoving.add(node)


def _translatingBody(journal: Journal, youngsModulus: float = YOUNGS_MODULUS, stretching: float = 0.0) -> MPMModel:
    """A block of particles moving to the right, over a grid of kernel functions that reaches beyond it; with a
    stretching, its velocity grows from its left to its right end by that rate times its length."""

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
            "Displacement/SQCNIxNSNI/PlaneStrain/Quad",
            number,
            vertexCoordinates,
            0.0,
            approximation,
            _material(youngsModulus),
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
        center = particle.getCenterCoordinates()
        particle.getResultArray("velocity")[0] = VELOCITY + stretching * (center[0] - 2.5 * PARTICLE_SIZE)
        particle.acceptStateAndPosition()

    return model


def _solve(model: MPMModel, journal: Journal, solversOfTheSteps: list, nIncrementsOfTheSteps: list):
    """Solve consecutive steps of the given solvers, each of the given number of increments of TIME_INCREMENT."""

    fieldOutputController = MPMFieldOutputController(model, journal)
    fieldOutputController.initializeJob()
    try:
        for number, (solver, nIncrements) in enumerate(zip(solversOfTheSteps, nIncrementsOfTheSteps)):
            AdaptiveStep(
                number,
                model,
                fieldOutputController,
                journal,
                {},
                solver,
                [],
                StepActionCollection(),
                stepLength=nIncrements * TIME_INCREMENT,
                startInc=1.0 / nIncrements,
                maxInc=1.0 / nIncrements,
                minInc=1.0 / nIncrements,
                maxNumInc=2 * nIncrements,
            ).solve()
    finally:
        fieldOutputController.finalizeJob()


def test_the_velocity_is_carried_by_node_across_a_change_of_the_active_nodes():
    journal = Journal()
    model = _translatingBody(journal)
    solver = _SolverRecordingTheVelocityAtReconnection({}, journal)

    _solve(model, journal, [solver], [N_INCREMENTS])

    newlyActiveNodes = [set(after) - set(before) for before, after in solver.activeNodeChanges]
    assert any(newlyActiveNodes), "the body did not move far enough for a node to become active"
    for before, after in solver.activeNodeChanges:
        for node, velocity in after.items():
            if node in before:
                np.testing.assert_array_equal(velocity, before[node])
            else:
                np.testing.assert_array_equal(velocity, 0.0)

    # The model holds the velocity of the active nodes at the end of the step: the solver's, half an increment behind,
    # advanced by that half increment with the acceleration. Zero elsewhere, and no solution anywhere.
    displacement = model.nodeFields["displacement"]
    velocityOfTheActiveNodes = solver.velocityOfTheActiveNodes()
    halfIncrement = 0.5 * solver.prevTimeStep.timeIncrement
    for node in displacement.nodes:
        row = displacement.indexOfNode(node)
        if node in velocityOfTheActiveNodes:
            np.testing.assert_array_equal(
                displacement["V"][row], velocityOfTheActiveNodes[node] + halfIncrement * displacement["A"][row]
            )
        else:
            for entry in ("V", "A", "P"):
                np.testing.assert_array_equal(displacement[entry][row], 0.0)
    np.testing.assert_array_equal(displacement["U"], 0.0)
    assert np.any(displacement["dU"][:, 0] > 0.0)


def test_a_node_without_inertia_has_no_velocity(monkeypatch):
    """A node whose mass is not positive has no equation of motion, so it has no velocity: also a node that moved while
    it had a mass, and lost it as the body moved on, keeps none. The test takes away the mass of the node with the
    largest one, from the tenth evaluation on, as a body leaving the node would."""

    assembleLumpedMassOfParticles = MPMEquationSystemLayout.assembleLumpedMassOfParticles
    evaluations = []
    nodeLosingItsMass = []

    def assembleWithoutTheMassOfOneNode(layout, forEachChunk):
        M = assembleLumpedMassOfParticles(layout, forEachChunk)
        evaluations.append(len(evaluations))
        if len(evaluations) >= 10:
            dofs = layout.dofManager.idcsOfFieldVariablesInDofVector
            if not nodeLosingItsMass:
                nodeLosingItsMass.append(
                    max(layout.activeNodes, key=lambda node: float(np.sum(M[dofs[node.fields["displacement"]]])))
                )
            if nodeLosingItsMass[0] in layout.activeNodes:
                M[dofs[nodeLosingItsMass[0].fields["displacement"]]] = 0.0
        return M

    monkeypatch.setattr(MPMEquationSystemLayout, "assembleLumpedMassOfParticles", assembleWithoutTheMassOfOneNode)

    journal = Journal()
    model = _translatingBody(journal)
    solver = _SolverRecordingTheNodesWithoutInertia({}, journal)

    _solve(model, journal, [solver], [N_INCREMENTS])

    assert nodeLosingItsMass[0] in solver.nodesThatLostTheirInertiaWhileMoving
    for velocity in solver.velocityOfNodesWithoutInertia:
        np.testing.assert_array_equal(velocity, 0.0)


def test_two_explicit_steps_are_one():
    """A step continues from the velocity and the acceleration the step before ends with: a body that stretches as it
    translates, stiff enough to accelerate visibly within an increment and still far within the stable time increment,
    solved by one step and by two steps of half its length, each by its own solver, ends in the same state. The second
    step takes the velocity of the nodes the first one leaves, rather than projecting it again from the particles, and
    its first increment of motion continues the leapfrog."""

    def stretchingBody(journal):
        return _translatingBody(journal, youngsModulus=1.0e2, stretching=0.1 * VELOCITY / (5 * PARTICLE_SIZE))

    journal = Journal()
    oneStep = stretchingBody(journal)
    _solve(oneStep, journal, [MeshfreeExplicitDynamic({}, journal)], [N_INCREMENTS])
    twoSteps = stretchingBody(journal)
    _solve(
        twoSteps,
        journal,
        [MeshfreeExplicitDynamic({}, journal), MeshfreeExplicitDynamic({}, journal)],
        [N_INCREMENTS // 2, N_INCREMENTS // 2],
    )

    expected, actual = oneStep.nodeFields["displacement"], twoSteps.nodeFields["displacement"]
    # the body accelerates: the acceleration changes the velocity within an increment by more than round-off
    assert TIME_INCREMENT * np.max(np.abs(expected["A"])) > 1e-6 * VELOCITY
    for entry in ("V", "A"):
        np.testing.assert_allclose(
            actual[entry], expected[entry], rtol=0.0, atol=1e-9 * np.max(np.abs(expected[entry])), err_msg=entry
        )
    for one, two in zip(oneStep.particles.values(), twoSteps.particles.values()):
        np.testing.assert_allclose(two.getCenterCoordinates(), one.getCenterCoordinates(), rtol=1e-12)
