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
"""EdelweissFE's explicit dynamic solver ``NED`` integrates the particles of an RKPM model. It carries the velocity of the kernel functions' nodes in the model, by node: a body
translates over a grid of kernel functions that are not bonded to its particles, so that the active nodes change
while it moves. A node active before and after a change keeps its velocity, a node that has just become active starts
at rest, and the model holds the velocity of the active nodes, zero solution, and zero results outside the active
domain.

The solver builds its discrete system on the one path of EdelweissFE's solvers: a change of the active nodes gives a new
numbering, particles reaching other kernel functions of the same nodes give new locations on the same numbering, and
following them through new locations is what building anew gives. The lumped mass is assembled in every increment, not
only for new locations. A restart checkpoint holds the numbering of the net force and the inverse lumped mass, and a
resume on another numbering is refused.

With ``carry-critical-time-step-weights``, the weights of the estimate of the particles' critical time step are carried
across new locations -- particles reaching other kernel functions -- and the estimate stays below the exact critical
time step; they start anew where an estimate with them falls below the one they started from, with a new numbering, and
after a resume."""

import h5py
import numpy as np
import pytest
from edelweissfe.journal.journal import Journal
from edelweissfe.models.change import Change
from edelweissfe.outputmanagers.restart import OutputManager as RestartOutputManager
from edelweissfe.outputmanagers.restart import RestartOutputManagerSchema
from edelweissfe.solvers.base.criticaltimestep import (
    CriticalTimeStepEstimateFromTheBoundOfEachDof,
)
from edelweissfe.solvers.nonlinearexplicitdynamic import NED
from edelweissfe.steps.adaptivestepforexplicitsimulations import (
    AdaptiveStepForExplicitSimulations,
)
from edelweissfe.steps.stepmanager import StepActionCollection
from edelweissfe.utils.checkpoint import ResumeCheckpoint
from edelweissfe.utils.exceptions import RestartError, StepFailed

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
from edelweissmeshfree.models.mpmdiscretesystem import (
    STABLE_FRACTION_OF_THE_TANGENT_LIMIT,
    MPMDiscreteSystem,
)
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.models.test._criticaltimestep import (
    exactCriticalTimeStepOfTheSystem,
)
from edelweissmeshfree.particlemanagers.kdbinorganizedparticlemanager import (
    KDBinOrganizedParticleManager,
)
from edelweissmeshfree.particles.marmot.marmotparticlewrapper import (
    MarmotParticleWrapper,
)

PARTICLE_SIZE = 20.0
KERNEL_SPACING = 28.0
SUPPORT_RADIUS = 1.5 * KERNEL_SPACING
VELOCITY = 1.0e6
TIME_INCREMENT = 1.0e-6
N_INCREMENTS = 40

# soft, so that the body translates nearly rigidly, far within the critical time step
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


#: The options of the solver: it reports (and writes output) after every increment.
NED_OPTIONS = {"output-frequency": 1}

#: The critical time step of the particles, as MPMDiscreteSystem estimates it.
ESTIMATE_OF_THE_PARTICLES = MPMDiscreteSystem.criticalTimeStepOfEntitiesWithOwnInertia


def _estimateOf(value: float):
    """A stand-in for the particles' estimate that gives the critical time step ``value``, whatever their state."""

    return lambda *args: CriticalTimeStepEstimateFromTheBoundOfEachDof(value, np.zeros(0, dtype=int), np.zeros(0))


@pytest.fixture(autouse=True)
def _theIncrementsOfTheStepAreTakenAsTheyAre(monkeypatch):
    """The particles' critical time step is replaced by an infinite one, so that the increments are the step's: these
    tests follow the discrete system and the state the increments carry at a given increment, not the stability of the
    body, whose critical time step falls as it stretches. The tests of the critical time step set their own."""

    monkeypatch.setattr(MPMDiscreteSystem, "criticalTimeStepOfEntitiesWithOwnInertia", _estimateOf(np.inf))


def _ned(journal: Journal, **options) -> NED:
    """The explicit dynamic solver with :data:`NED_OPTIONS`, and the given options besides."""

    return NED({}, journal, **{**NED_OPTIONS, **options})


class _SolverRecordingTheBuildPath(NED):
    """The explicit solver of :func:`_ned`, recording what the build path of the discrete system does: every hook it
    calls, the numbering the discrete system has then, and whether the increment and the velocity are wrapped with its
    locations."""

    def __init__(self, journal: Journal, **options):
        super().__init__({}, journal, **{**NED_OPTIONS, **options})
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

    def __init__(self, journal: Journal):
        super().__init__(journal)
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

    def __init__(self, journal: Journal):
        super().__init__(journal)
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
    particleType: str = "Displacement/SQCNIxNSNI/PlaneStrain/Quad",
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
            particleType,
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
    solver: NED,
    nIncrements: int,
    outputManagers: list = (),
    maxNumInc: int = None,
) -> AdaptiveStepForExplicitSimulations:
    """A step of the given number of increments of TIME_INCREMENT, which ends on its increment cap."""

    return AdaptiveStepForExplicitSimulations(
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
    solver = _SolverRecordingTheVelocityAtReconnection(journal)

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


def test_a_particle_without_the_explicit_path_stops_the_run_with_the_name_of_the_method():
    """A particle with a mass but without the explicit path (an SQCNI particle of the displacement family) would be
    integrated with its mass and without its force. Marmot refuses it instead, in the first call of the explicit path,
    and the refusal reaches Python as an exception that names the method, instead of ending the process."""

    journal = Journal(verbose=False)
    model = _translatingBody(journal, particleType="DisplacementSQCNI/PlaneStrain/Quad")

    with pytest.raises(StepFailed, match="Particle::updatePhysicsExplicit: not implemented for this particle"):
        _solve(model, journal, [_ned(journal)], [2])


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
    solver = _SolverRecordingTheNodesWithoutInertia(journal)

    _solve(model, journal, [solver], [N_INCREMENTS])

    assert nodeLosingItsMass[0] in solver.nodesThatLostTheirInertiaWhileMoving
    for velocity in solver.velocityOfNodesWithoutInertia:
        np.testing.assert_array_equal(velocity, 0.0)


def test_two_explicit_steps_are_one():
    """A step continues from the velocity and the acceleration the step before ends with: a body that stretches as it
    translates, stiff enough to accelerate visibly within an increment and still far within the critical time step,
    solved by one step and by two steps of half its length, each by its own solver, ends in the same state. The second
    step takes the velocity of the nodes the first one leaves, rather than projecting it again from the particles, and
    its first increment of motion continues the leapfrog."""

    def stretchingBody(journal):
        return _translatingBody(journal, youngsModulus=1.0e2, stretching=0.1 * VELOCITY / (5 * PARTICLE_SIZE))

    journal = Journal()
    oneStep = stretchingBody(journal)
    _solve(oneStep, journal, [_ned(journal)], [N_INCREMENTS])
    twoSteps = stretchingBody(journal)
    _solve(
        twoSteps,
        journal,
        [_ned(journal), _ned(journal)],
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


def _stretchingBody(journal: Journal, youngsModulus: float = YOUNGS_MODULUS) -> MPMModel:
    """A body with bonded kernel functions that stretches enough for particles to leave the support of kernel functions
    and to enter that of others, while the active nodes stay the same."""

    return _translatingBody(
        journal, youngsModulus, stretching=1.5 * VELOCITY / (5 * PARTICLE_SIZE), bondedKernelFunctions=True
    )


def test_particles_reaching_other_nodes_give_new_locations_on_the_same_numbering():
    journal = Journal()
    model = _stretchingBody(journal)
    solver = _SolverRecordingTheBuildPath(journal)

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
    _solve(relocated, journal, [_SolverRecordingTheBuildPath(journal)], [N_INCREMENTS])
    solverBuildingAnew = _SolverWithANewNumberingForEveryChange(journal)
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

    solver = _SolverRecordingTheBuildPath(Journal())
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


class _SolverRecordingTheEstimates(NED):
    """The explicit solver recording, for every accepted increment, its number, the numbering, the lumped inertia of the
    particles, the time increment the step takes next, and whether the particles' critical time step was estimated."""

    def __init__(self, journal: Journal, **options):
        super().__init__({}, journal, **{**NED_OPTIONS, **options})
        self.increments = []

    def acceptIncrement(self, step, model, timeStep):
        estimatesBefore = self._criticalTimeStepOfTheEntities.numberOfEstimates
        super().acceptIncrement(step, model, timeStep)
        self.increments.append(
            {
                "number": timeStep.number,
                "numbering": self.system.numbering,
                "inertia": np.array(self._inertiaOfEntities),
                "timeIncrementInUse": self._timeIncrementInUse(step),
                "estimated": self._criticalTimeStepOfTheEntities.numberOfEstimates > estimatesBefore,
            }
        )


def test_with_any_margin_the_particles_are_estimated_in_the_zero_increment_after_a_new_numbering_and_when_a_node_gains_a_mass(
    monkeypatch,
):
    """With a critical time step of the particles that leaves any margin (infinite), they are estimated exactly in the
    zero increment, after a new numbering, and when a node has a mass it did not have at the last estimate: a node
    becomes active when a kernel function reaches an evaluation point of a particle, and gets its mass when the
    particle's centre enters its support, from zero (review of #344, finding 2). In this body a node gets its mass
    together with a new numbering; a gain on the same numbering is tested on the policy itself (EdelweissFE,
    ``tests/test_critical_time_step.py``)."""

    monkeypatch.setattr(MPMDiscreteSystem, "criticalTimeStepOfEntitiesWithOwnInertia", _estimateOf(np.inf))
    journal = Journal()
    solver = _SolverRecordingTheEstimates(journal)
    _solve(_translatingBody(journal), journal, [solver], [N_INCREMENTS])

    expected, last = [], None
    for increment in solver.increments:
        isDue = (
            last is None
            or increment["numbering"] is not last["numbering"]
            or bool(np.any((increment["inertia"] > 0) & ~(last["inertia"] > 0)))
        )
        expected.append(isDue)
        if isDue:
            last = increment
    assert [increment["estimated"] for increment in solver.increments] == expected
    assert 1 < sum(expected) < len(expected)


def test_the_particles_are_estimated_in_every_increment_where_they_set_the_time_increment(monkeypatch):
    """Where the particles' critical time step sets the time increment, there is no margin left after an increment: they
    are estimated in every one."""

    courantNumber = 0.8
    monkeypatch.setattr(
        MPMDiscreteSystem,
        "criticalTimeStepOfEntitiesWithOwnInertia",
        _estimateOf(0.5 * TIME_INCREMENT / courantNumber),
    )
    journal = Journal()
    solver = _SolverRecordingTheEstimates(journal, **{"courant-number": courantNumber})
    _solve(_stretchingBody(journal), journal, [solver], [N_INCREMENTS // 2])

    assert all(increment["estimated"] for increment in solver.increments)
    assert solver.increments[-1]["timeIncrementInUse"] == pytest.approx(0.5 * TIME_INCREMENT)


def test_where_the_particles_have_a_margin_they_are_estimated_when_their_masses_and_the_assumed_decrease_may_use_it_up(
    monkeypatch,
):
    """Where the finite elements, or the step, set the time increment, the particles are estimated anew only when
    ``courant-number`` times v (M/M0)^(1/2) (1 - delta)^n, the lower bound of their critical time step n increments
    after an estimate v, could fall below the increment in use."""

    courantNumber, delta, margin = 0.8, 0.1, 1.6
    value = margin * TIME_INCREMENT / courantNumber
    monkeypatch.setattr(MPMDiscreteSystem, "criticalTimeStepOfEntitiesWithOwnInertia", _estimateOf(value))
    journal = Journal()
    solver = _SolverRecordingTheEstimates(
        journal, **{"courant-number": courantNumber, "critical-time-step-decrease-per-increment": delta}
    )
    _solve(_stretchingBody(journal), journal, [solver], [N_INCREMENTS])

    expected, last = [], None
    for increment in solver.increments:
        if last is None:
            isDue = True
        else:
            withInertia = (increment["inertia"] > 0) & (last["inertia"] > 0)
            alpha = min(1.0, np.min(increment["inertia"][withInertia] / last["inertia"][withInertia]))
            criticalTimeStepAtWorst = value * np.sqrt(alpha) * (1 - delta) ** (increment["number"] - last["number"])
            isDue = courantNumber * criticalTimeStepAtWorst < increment["timeIncrementInUse"]
        expected.append(isDue)
        if isDue:
            last = increment
    assert [increment["estimated"] for increment in solver.increments] == expected
    assert 2 < sum(expected) < len(expected), "the margin should be used up a few times, not never or always"


def test_the_time_increment_never_exceeds_the_courant_fraction_of_a_fresh_estimate(monkeypatch):
    """With the particles' own estimate, in a stiff body that stretches fast, the margin policy takes the critical time
    step anew often enough: after every increment the time increment the step takes next is at most ``courant-number``
    times the critical time step of the configuration it starts from, estimated afresh."""

    monkeypatch.setattr(MPMDiscreteSystem, "criticalTimeStepOfEntitiesWithOwnInertia", ESTIMATE_OF_THE_PARTICLES)
    youngsModulus = 2.0e5
    journal = Journal()

    # the courant number that makes the particles' critical time step of the start twice the step's increment
    probe = _SolverRecordingTheEstimates(journal)
    _solve(_stretchingBody(journal, youngsModulus), journal, [probe], [1])
    courantNumber = 2.0 * TIME_INCREMENT / probe._criticalTimeStepOfTheEntities.value

    class _SolverCheckingTheTimeIncrement(_SolverRecordingTheEstimates):
        def acceptIncrement(self, step, model, timeStep):
            super().acceptIncrement(step, model, timeStep)
            fresh = ESTIMATE_OF_THE_PARTICLES(
                self.system, self._inertiaOfEntities, self._dampingOfEntities, self.forEachChunk
            ).value
            self.increments[-1]["fresh"] = fresh

    solver = _SolverCheckingTheTimeIncrement(journal, **{"courant-number": courantNumber})
    _solve(_stretchingBody(journal, youngsModulus), journal, [solver], [N_INCREMENTS])

    assert 2 < sum(increment["estimated"] for increment in solver.increments) < len(solver.increments)
    for increment in solver.increments:
        assert increment["timeIncrementInUse"] <= courantNumber * increment["fresh"], increment["number"]


def _writeCheckpointAndResume(tmp_path, tamper=None, **options):
    """Run a few increments of the translating body writing a checkpoint, let ``tamper`` change what the checkpoint holds
    of the solver, and resume from it for the rest of the step, with the given solver options besides
    :data:`NED_OPTIONS`."""

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
            _ned(journal, **options),
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
        _step(0, resumedModel, journal, fieldOutputController, _ned(journal, **options), N_INCREMENTS).solve(
            resumeCheckpoint
        )
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


#: The options of a solver that carries the weights of the estimate of the particles' critical time step and estimates
#: them in every increment.
CARRIED_WEIGHTS = {"carry-critical-time-step-weights": True, "critical-time-step-decrease-per-increment": 1.0}


class _SolverRecordingTheCarriedWeights(_SolverRecordingTheBuildPath):
    """The explicit solver with :data:`CARRIED_WEIGHTS`, recording, in the order they happen, the hooks of the build
    path and every estimate: the weights it was handed, the critical time step with them, the one of the same system
    without weights, and the exact critical time step of the system's tangent."""

    def __init__(self, journal: Journal):
        super().__init__(journal, **CARRIED_WEIGHTS)
        self.events = []

    def onNewNumbering(self):
        super().onNewNumbering()
        self.events.append("numbering")

    def onNewLocations(self):
        super().onNewLocations()
        self.events.append("locations")


def _stretchingBodyWithinItsCriticalTimeStep(journal: Journal) -> MPMModel:
    """The stretching body, stretching half as fast again: with its critical time step estimated, the step's increments
    are lowered to it, and the body still stretches enough for particles to reach other kernel functions."""

    return _translatingBody(journal, stretching=2.25 * VELOCITY / (5 * PARTICLE_SIZE), bondedKernelFunctions=True)


def _recordTheEstimates(monkeypatch, solver: _SolverRecordingTheCarriedWeights):
    """Let every estimate of the particles' critical time step be recorded in the solver's events."""

    def recordingTheEstimate(system, inertia, damping, forEachChunk, weightsOfTheEstimate=None):
        handed = np.array(weightsOfTheEstimate)
        estimate = ESTIMATE_OF_THE_PARTICLES(system, inertia, damping, forEachChunk, weightsOfTheEstimate)
        withoutWeights = ESTIMATE_OF_THE_PARTICLES(system, inertia, damping, forEachChunk).value
        solver.events.append((handed, estimate.value, withoutWeights, exactCriticalTimeStepOfTheSystem(system)))
        return estimate

    monkeypatch.setattr(MPMDiscreteSystem, "criticalTimeStepOfEntitiesWithOwnInertia", recordingTheEstimate)


def _estimates(events: list) -> list:
    return [event for event in events if isinstance(event, tuple)]


def test_carried_weights_survive_particles_reaching_other_kernel_functions_and_bound_the_critical_time_step(
    monkeypatch,
):
    """In the stretching body, particles reach other kernel functions of the same nodes: new locations on the same
    numbering. The weights are carried across them, indexed by the degrees of freedom of the numbering, i.e. by node
    and field; every estimate stays below the exact critical time step of the tangent, scaled by
    STABLE_FRACTION_OF_THE_TANGENT_LIMIT. The estimate the weights start from (weights constant per field) is never below
    the estimate without weights, and an estimate with carried weights below it starts them anew: the next estimate is
    handed zero weights."""

    journal = Journal()
    solver = _SolverRecordingTheCarriedWeights(journal)
    _recordTheEstimates(monkeypatch, solver)

    _solve(_stretchingBodyWithinItsCriticalTimeStep(journal), journal, [solver], [N_INCREMENTS])

    assert solver.events.count("numbering") == 1
    afterNewLocations = [
        later
        for earlier, later in zip(solver.events, solver.events[1:])
        if earlier == "locations" and isinstance(later, tuple)
    ]
    assert len(afterNewLocations) > 1, "no particle reached other kernel functions"
    for handed, *_ in afterNewLocations[1:]:
        assert np.any(handed > 0.0), "the weights were not carried across new locations"
    estimates = _estimates(solver.events)
    for (handed, estimate, withoutWeights, exact), following in zip(estimates, estimates[1:] + [None]):
        assert estimate <= STABLE_FRACTION_OF_THE_TANGENT_LIMIT * exact
        if not np.any(handed > 0.0):
            assert estimate >= withoutWeights
            start = estimate
        elif estimate < start:
            assert following is not None and np.all(following[0] == 0.0), "the weights were not started anew"


def test_carried_weights_start_anew_with_a_new_numbering(monkeypatch):
    """Building the discrete system anew whenever the connectivity changed gives a new numbering each time: the first
    estimate on every numbering is handed no weights, the estimates after it the weights the last one left, unless the
    last one fell below the estimate its weights started from."""

    journal = Journal()

    class _SolverWithANewNumberingForEveryChangeCarryingTheWeights(_SolverRecordingTheCarriedWeights):
        def updateDiscreteSystem(self, model, change):
            super().updateDiscreteSystem(model, max(change, Change.NUMBERING) if change > Change.NONE else change)

    solver = _SolverWithANewNumberingForEveryChangeCarryingTheWeights(journal)
    _recordTheEstimates(monkeypatch, solver)

    _solve(_stretchingBodyWithinItsCriticalTimeStep(journal), journal, [solver], [N_INCREMENTS])

    assert solver.events.count("numbering") > 1
    lastWasNumbering = lastFellBelowTheStart = False
    for event in solver.events:
        if event == "numbering":
            lastWasNumbering = True
        elif isinstance(event, tuple):
            handed, estimate = event[:2]
            assert np.all(handed == 0.0) if lastWasNumbering or lastFellBelowTheStart else np.any(handed > 0.0)
            if np.all(handed == 0.0):
                start = estimate
            lastWasNumbering, lastFellBelowTheStart = False, estimate < start


def test_a_resume_with_carried_weights_goes_on_and_starts_the_weights_anew(tmp_path, monkeypatch):
    """The carried weights are not part of the checkpoint: a resumed step starts them anew, as every step does."""

    monkeypatch.setattr(MPMDiscreteSystem, "criticalTimeStepOfEntitiesWithOwnInertia", ESTIMATE_OF_THE_PARTICLES)
    checkpoint = _writeCheckpointAndResume(tmp_path, **CARRIED_WEIGHTS)

    with h5py.File(checkpoint, "r") as f:
        assert not [name for name in f["solver"] if "weights" in name.lower()]
