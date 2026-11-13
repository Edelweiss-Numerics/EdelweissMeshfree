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
domain.

The solver builds its discrete system on the one path of EdelweissFE's solvers: a change of the active nodes gives a new
numbering, particles reaching other kernel functions of the same nodes give new locations on the same numbering, and
following them through new locations is what building anew gives. The lumped mass is assembled in every increment, not
only for new locations. A restart checkpoint holds the numbering of the net force and the inverse lumped mass, and a
resume on another numbering is refused."""

import h5py
import numpy as np
import pytest
from edelweissfe.journal.journal import Journal
from edelweissfe.models.change import Change
from edelweissfe.outputmanagers.restart import OutputManager as RestartOutputManager
from edelweissfe.outputmanagers.restart import RestartOutputManagerSchema
from edelweissfe.steps.adaptivestep import AdaptiveStep
from edelweissfe.steps.stepmanager import StepActionCollection
from edelweissfe.utils.checkpoint import ResumeCheckpoint
from edelweissfe.utils.exceptions import RestartError

from edelweissmeshfree.fieldoutput.fieldoutput import MPMFieldOutputController
from edelweissmeshfree.generators.kernelmatchingtoparticlegenerator import (
    generateKernelMatchingToParticle,
)
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
from edelweissmeshfree.models.mpmdiscretesystem import MPMDiscreteSystem
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


class _SolverRecordingTheBuildPath(MeshfreeExplicitDynamic):
    """The explicit solver, recording what the build path of the discrete system does: every hook it calls, the
    numbering the discrete system has then, and whether the increment and the velocity are wrapped with its
    locations."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hooks = []
        self.numberings = []
        self.vectorsAreOnTheLocations = []

    def onNewNumbering(self):
        super().onNewNumbering()
        self.hooks.append("numbering")

    def onNewLocations(self):
        super().onNewLocations()
        self.hooks.append("locations")
        self.numberings.append(self.system.numbering)
        self.vectorsAreOnTheLocations.append(
            self._V.locations is self.system.locations and self._dU.locations is self.system.locations
        )


class _SolverRecordingTheVelocityAtReconnection(_SolverRecordingTheBuildPath):
    """The explicit solver, recording the velocity of the active nodes before and after every update of the discrete
    system that changes the active nodes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.activeNodeChanges = []

    def updateDiscreteSystem(self, model, change):
        before = self.velocityOfTheActiveNodes() if self.system is not None else None
        super().updateDiscreteSystem(model, change)
        after = self.velocityOfTheActiveNodes()
        if before is not None and set(before) != set(after):
            self.activeNodeChanges.append((before, after))

    def velocityOfTheActiveNodes(self) -> dict:
        numbering = self.system.numbering
        return {
            node: np.array(self._V[numbering.idcsOfFieldVariablesInDofVector[node.fields["displacement"]]])
            for node in self.system.activeNodes
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
        numbering = self.system.numbering
        for node, velocity in self.velocityOfTheActiveNodes().items():
            indices = numbering.idcsOfFieldVariablesInDofVector[node.fields["displacement"]]
            if np.all(np.asarray(self._Minv)[indices] == 0.0):
                self.velocityOfNodesWithoutInertia.append(velocity)
                if np.any(velocityBefore.get(node, 0.0)):
                    self.nodesThatLostTheirInertiaWhileMoving.add(node)


def _translatingBody(
    journal: Journal,
    youngsModulus: float = YOUNGS_MODULUS,
    stretching: float = 0.0,
    bondedKernelFunctions: bool = False,
) -> MPMModel:
    """A block of particles moving to the right, over a grid of kernel functions that reaches beyond it; with a
    stretching, its velocity grows from its left to its right end by that rate times its length. With bonded kernel
    functions, every particle carries its own, and the active nodes are those of the particles: the grid is left out."""

    dimension = 2
    model = MPMModel(dimension)

    if not bondedKernelFunctions:

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

    if bondedKernelFunctions:

        def matchingKernelFunctionFactory(node, supportSize):
            return MarmotMeshfreeKernelFunctionWrapper(
                node, "BSplineBoxed", supportRadius=supportSize, continuityOrder=3
            )

        generateKernelMatchingToParticle(
            model, journal, matchingKernelFunctionFactory, model.particleSets["body_all"], supportScalingFactor=2.4
        )

    particleKernelDomain = ParticleKernelDomain(
        list(model.particles.values()), list(model.meshfreeKernelFunctions.values())
    )
    model.particleKernelDomains["body"] = particleKernelDomain
    model.particleManagers.append(
        KDBinOrganizedParticleManager(
            particleKernelDomain, dimension, journal, bondParticlesToKernelFunctions=bondedKernelFunctions
        )
    )

    model.prepareYourself(journal)

    for particle in model.particles.values():
        center = particle.getCenterCoordinates()
        particle.getResultArray("velocity")[0] = VELOCITY + stretching * (center[0] - 2.5 * PARTICLE_SIZE)
        particle.acceptStateAndPosition()

    return model


def _step(
    number: int,
    model: MPMModel,
    journal: Journal,
    fieldOutputController: MPMFieldOutputController,
    solver: MeshfreeExplicitDynamic,
    nIncrements: int,
    outputManagers: list = (),
    maxNumInc: int = None,
) -> AdaptiveStep:
    """A step of the given number of increments of TIME_INCREMENT, which ends on its increment cap."""

    return AdaptiveStep(
        number,
        model,
        fieldOutputController,
        journal,
        {},
        solver,
        list(outputManagers),
        StepActionCollection(),
        stepLength=nIncrements * TIME_INCREMENT,
        startInc=1.0 / nIncrements,
        maxInc=1.0 / nIncrements,
        minInc=1.0 / nIncrements,
        maxNumInc=maxNumInc if maxNumInc is not None else 2 * nIncrements,
    )


def _solve(model: MPMModel, journal: Journal, solversOfTheSteps: list, nIncrementsOfTheSteps: list):
    """Solve consecutive steps of the given solvers, each of the given number of increments of TIME_INCREMENT."""

    fieldOutputController = MPMFieldOutputController(model, journal)
    fieldOutputController.initializeJob()
    try:
        for number, (solver, nIncrements) in enumerate(zip(solversOfTheSteps, nIncrementsOfTheSteps)):
            _step(number, model, journal, fieldOutputController, solver, nIncrements).solve()
    finally:
        fieldOutputController.finalizeJob()


def test_the_velocity_is_carried_by_node_across_a_change_of_the_active_nodes():
    journal = Journal()
    model = _translatingBody(journal)
    solver = _SolverRecordingTheVelocityAtReconnection({}, journal)

    _solve(model, journal, [solver], [N_INCREMENTS])

    # a change of the active nodes is always a new numbering, and new locations follow it; never new locations alone
    assert len(solver.hooks) > 2, "the body did not move far enough for a node to become active"
    assert solver.hooks == ["numbering", "locations"] * (len(solver.hooks) // 2)
    assert all(solver.vectorsAreOnTheLocations)

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

    assembleLumpedMassOfEntitiesWithOwnInertia = MPMDiscreteSystem.assembleLumpedMassOfEntitiesWithOwnInertia
    evaluations = []
    nodeLosingItsMass = []

    def assembleWithoutTheMassOfOneNode(system, forEachChunk):
        M = assembleLumpedMassOfEntitiesWithOwnInertia(system, forEachChunk)
        evaluations.append(len(evaluations))
        if len(evaluations) >= 10:
            dofs = system.numbering.idcsOfFieldVariablesInDofVector
            if not nodeLosingItsMass:
                nodeLosingItsMass.append(
                    max(system.activeNodes, key=lambda node: float(np.sum(M[dofs[node.fields["displacement"]]])))
                )
            if nodeLosingItsMass[0] in system.activeNodes:
                M[dofs[nodeLosingItsMass[0].fields["displacement"]]] = 0.0
        return M

    monkeypatch.setattr(
        MPMDiscreteSystem, "assembleLumpedMassOfEntitiesWithOwnInertia", assembleWithoutTheMassOfOneNode
    )

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


class _SolverWithANewNumberingForEveryChange(_SolverRecordingTheBuildPath):
    """The explicit solver, building the discrete system anew whenever the connectivity changed."""

    def updateDiscreteSystem(self, model, change):
        super().updateDiscreteSystem(model, max(change, Change.NUMBERING) if change > Change.NONE else change)


def _stretchingBody(journal: Journal) -> MPMModel:
    """A body with bonded kernel functions that stretches enough for particles to leave the support of kernel functions
    and to enter that of others, while the active nodes stay the same."""

    return _translatingBody(journal, stretching=1.5 * VELOCITY / (5 * PARTICLE_SIZE), bondedKernelFunctions=True)


def test_particles_reaching_other_nodes_give_new_locations_on_the_same_numbering():
    journal = Journal()
    model = _stretchingBody(journal)
    solver = _SolverRecordingTheBuildPath({}, journal)

    _solve(model, journal, [solver], [N_INCREMENTS])

    assert solver.hooks[:2] == ["numbering", "locations"]
    assert solver.hooks[2:].count("locations") > 0, "no particle reached other kernel functions"
    assert solver.hooks.count("numbering") == 1
    assert all(numbering is solver.numberings[0] for numbering in solver.numberings)
    assert all(solver.vectorsAreOnTheLocations)


def test_new_locations_give_the_state_of_a_discrete_system_built_anew():
    """Following the particles through new locations on the same numbering is what building the discrete system anew
    at every change gives: the same particles, at the same positions, with the same velocity at the nodes."""

    journal = Journal()
    relocated, built = _stretchingBody(journal), _stretchingBody(journal)
    _solve(relocated, journal, [_SolverRecordingTheBuildPath({}, journal)], [N_INCREMENTS])
    solverBuildingAnew = _SolverWithANewNumberingForEveryChange({}, journal)
    _solve(built, journal, [solverBuildingAnew], [N_INCREMENTS])

    assert solverBuildingAnew.hooks.count("numbering") > 1
    for particle, expected in zip(relocated.particles.values(), built.particles.values()):
        np.testing.assert_array_equal(particle.getCenterCoordinates(), expected.getCenterCoordinates())
    for entry in ("dU", "V", "A", "P"):
        np.testing.assert_array_equal(
            relocated.nodeFields["displacement"][entry], built.nodeFields["displacement"][entry], err_msg=entry
        )


def test_the_lumped_mass_is_assembled_anew_in_every_increment_not_only_for_new_locations(monkeypatch):
    """The mass is that of the particles at the kernel functions as they are at the particles' current positions, so it
    changes from one increment to the next, also when no hook is called: it cannot be derived in onNewLocations."""

    solver = _SolverRecordingTheBuildPath({}, Journal())
    assembleLumpedMassOfEntitiesWithOwnInertia = MPMDiscreteSystem.assembleLumpedMassOfEntitiesWithOwnInertia
    massesWithTheNumberOfHooksSoFar = []

    def recordingTheMass(system, forEachChunk):
        M = assembleLumpedMassOfEntitiesWithOwnInertia(system, forEachChunk)
        massesWithTheNumberOfHooksSoFar.append((len(solver.hooks), np.array(M)))
        return M

    monkeypatch.setattr(MPMDiscreteSystem, "assembleLumpedMassOfEntitiesWithOwnInertia", recordingTheMass)

    journal = Journal()
    _solve(_translatingBody(journal), journal, [solver], [N_INCREMENTS])

    # the first increment is the zero increment: nothing moves in it
    increments = massesWithTheNumberOfHooksSoFar[1:]
    consecutiveWithoutAHook = [
        (before, after)
        for (hooksBefore, before), (hooksAfter, after) in zip(increments, increments[1:])
        if hooksBefore == hooksAfter
    ]
    assert consecutiveWithoutAHook, "every increment called a hook, so this test proves nothing"
    for before, after in consecutiveWithoutAHook:
        assert not np.array_equal(before, after)


def _writeCheckpointAndResume(tmp_path, tamper=None):
    """Run a few increments of the translating body writing a checkpoint, let ``tamper`` change what the checkpoint holds
    of the solver, and resume from it for the rest of the step."""

    journal = Journal()
    model = _translatingBody(journal)
    baseName = str(tmp_path / "restart")
    fieldOutputController = MPMFieldOutputController(model, journal)
    fieldOutputController.initializeJob()
    restartOutputManager = RestartOutputManager(
        "restart",
        model,
        fieldOutputController,
        journal,
        None,
        configuration=RestartOutputManagerSchema(writeInterval=1, baseName=baseName),
    )
    try:
        _step(
            0,
            model,
            journal,
            fieldOutputController,
            MeshfreeExplicitDynamic({}, journal),
            N_INCREMENTS,
            [restartOutputManager],
            maxNumInc=N_INCREMENTS // 2,
        ).solve()
    finally:
        fieldOutputController.finalizeJob()
    checkpoint = max(tmp_path.glob("restart_*.h5"), key=lambda path: int(path.stem.split("_")[-1]))

    if tamper is not None:
        with h5py.File(checkpoint, "r+") as f:
            tamper(f["solver"])

    resumedModel = _translatingBody(journal)
    resumeCheckpoint = ResumeCheckpoint(str(checkpoint))
    resumeCheckpoint.restoreModel(resumedModel, journal)
    fieldOutputController = MPMFieldOutputController(resumedModel, journal)
    fieldOutputController.initializeJob()
    try:
        _step(
            0, resumedModel, journal, fieldOutputController, MeshfreeExplicitDynamic({}, journal), N_INCREMENTS
        ).solve(resumeCheckpoint)
    finally:
        fieldOutputController.finalizeJob()

    return checkpoint


def test_a_checkpoint_holds_the_numbering_of_the_net_force_and_a_resume_on_it_goes_on(tmp_path):
    checkpoint = _writeCheckpointAndResume(tmp_path)

    with h5py.File(checkpoint, "r") as f:
        assert len(f["solver/_numberingOfNetForce"][()]) > 0


def test_a_resume_with_another_numbering_than_the_vectors_were_written_for_is_refused(tmp_path):
    def writtenForAnotherNumbering(solverGroup):
        del solverGroup["_numberingOfNetForce"]
        solverGroup.create_dataset("_numberingOfNetForce", data=np.array(b"0123456789abcdef"))

    with pytest.raises(RestartError, match="another numbering"):
        _writeCheckpointAndResume(tmp_path, writtenForAnotherNumbering)


def test_a_checkpoint_without_the_numbering_of_the_vectors_is_refused(tmp_path):
    with pytest.raises(RestartError, match="does not hold the numbering"):
        _writeCheckpointAndResume(tmp_path, lambda solverGroup: solverGroup.__delitem__("_numberingOfNetForce"))
