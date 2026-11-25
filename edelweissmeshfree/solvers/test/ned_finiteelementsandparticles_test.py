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
"""EdelweissFE's explicit dynamic solver ``NED`` integrates a model of finite elements and RKPM particles in one
leapfrog: the elements carry their total solution and their mass at their nodes, the particles carry their state and
their inertia themselves, and their kernel functions move with them (see the documentation of ``NED``, "Entities that
carry their own inertia").

A free block of finite elements (Marmot's ``CPE8UL``) moves with the initial velocity of its nodes, next to a free block
of particles, which moves with the initial velocity of its particles; the two never touch. Then

* the block of finite elements moves exactly as it does alone, in a model without particles: the increment of the
  coupled model, which evaluates the net force on the connectivity the particles reach, changes nothing for it, and the
  projection of the particles' momentum does not touch the velocity of its nodes;
* the block of particles moves exactly as it does alone, with the same time increment;
* both translate with their initial velocities, and the momentum of the model holds, to round-off, in every increment;
* the Ensight output of the block of elements holds the solution of its nodes, node by node, in the order of the nodes
  of its part.

The material of the elements is initialized explicitly, as the input file does with ``*initialConditions``: an
updated-Lagrangian Marmot element is valid only after it (Marmot issue #129).
"""

import numpy as np
from edelweissfe.generators.planerectquad import Generator as PlaneRectQuadGenerator
from edelweissfe.generators.planerectquad import PlaneRectQuadSchema
from edelweissfe.journal.journal import Journal
from edelweissfe.sections.plane import PlaneSectionSchema
from edelweissfe.sections.plane import Section as PlaneSection
from edelweissfe.solvers.nonlinearexplicitdynamic import NED
from edelweissfe.steps.adaptivestepforexplicitsimulations import (
    AdaptiveStepForExplicitSimulations,
)
from edelweissfe.steps.stepmanager import StepActionCollection

from edelweissmeshfree.fieldoutput.fieldoutput import MPMFieldOutputController
from edelweissmeshfree.generators.kernelmatchingtoparticlegenerator import (
    generateKernelMatchingToParticle,
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
from edelweissmeshfree.outputmanagers.ensight import (
    OutputManager as EnsightOutputManager,
)
from edelweissmeshfree.particlemanagers.kdbinorganizedparticlemanager import (
    KDBinOrganizedParticleManager,
)
from edelweissmeshfree.particles.marmot.marmotparticlewrapper import (
    MarmotParticleWrapper,
)

YOUNGS_MODULUS, POISSONS_RATIO, DENSITY = 1.0e3, 0.3, 1.0e-8
BLOCK_OF_ELEMENTS_SIZE, BLOCK_OF_PARTICLES_SIZE, PARTICLE_SIZE = 40.0, 100.0, 20.0
#: far enough from the block of elements that the two never touch
BLOCK_OF_PARTICLES_X0 = 200.0
ELEMENTS_VELOCITY = np.array([1.0e3, -2.0e2])
PARTICLES_VELOCITY = np.array([-5.0e2, 3.0e2])
N_INCREMENTS = 20


#: The material of the elements and the particles alike: finite strain J2 plasticity, whose yield stress is never
#: reached.
MATERIAL = "FiniteStrainJ2Plasticity"


def _materialProperties() -> np.ndarray:
    """K, G, the yield stress (never reached), ..., the mass density."""

    bulkModulus = YOUNGS_MODULUS / (3 * (1 - 2 * POISSONS_RATIO))
    shearModulus = YOUNGS_MODULUS / (2 * (1 + POISSONS_RATIO))
    return np.array([bulkModulus, shearModulus, 1e9, 1e9, 1, 0, 1, DENSITY])


def _addBlockOfElements(model: MPMModel, journal: Journal):
    """A square block of 2 x 2 quadratic plane strain elements, ``CPE8UL``."""

    with model.topology.changes():
        PlaneRectQuadGenerator(
            "elements",
            model,
            journal,
            configuration=PlaneRectQuadSchema(
                x0=0.0, y0=0.0, length=BLOCK_OF_ELEMENTS_SIZE, h=BLOCK_OF_ELEMENTS_SIZE, nX=2, nY=2, elType="CPE8UL"
            ),
        )

    model.sections["elements"] = PlaneSection(
        "elements",
        model,
        {"name": MATERIAL, "properties": _materialProperties()},
        [model.elementSets["elements_all"]],
        configuration=PlaneSectionSchema(thickness=1.0),
    )


def _addBlockOfParticles(model: MPMModel, journal: Journal, firstKernelFunctionNumber: int):
    """A square block of 5 x 5 particles, each with a kernel function bonded to it."""

    dimension = 2
    approximation = MarmotMeshfreeApproximationWrapper("ReproducingKernel", dimension, completenessOrder=1)

    def particleFactory(number, vertexCoordinates, volume):
        return MarmotParticleWrapper(
            "Displacement/SQCNIxNSNI/PlaneStrain/Quad",
            number,
            vertexCoordinates,
            0.0,
            approximation,
            {"material": MATERIAL, "properties": _materialProperties()},
        )

    nParticles = round(BLOCK_OF_PARTICLES_SIZE / PARTICLE_SIZE)
    generateRectangularQuadParticleGrid(
        model,
        journal,
        particleFactory,
        x0=BLOCK_OF_PARTICLES_X0,
        y0=0.0,
        l=BLOCK_OF_PARTICLES_SIZE,
        h=BLOCK_OF_PARTICLES_SIZE,
        nX=nParticles,
        nY=nParticles,
        name="particles",
    )

    def kernelFunctionFactory(node, supportSize):
        return MarmotMeshfreeKernelFunctionWrapper(node, "BSplineBoxed", supportRadius=supportSize, continuityOrder=3)

    generateKernelMatchingToParticle(
        model,
        journal,
        kernelFunctionFactory,
        model.particleSets["particles_all"],
        supportScalingFactor=2.4,
        firstKernelFunctionNumber=firstKernelFunctionNumber,
    )

    particleKernelDomain = ParticleKernelDomain(
        list(model.particles.values()), list(model.meshfreeKernelFunctions.values())
    )
    model.particleKernelDomains["particles"] = particleKernelDomain
    model.particleManagers.append(
        KDBinOrganizedParticleManager(particleKernelDomain, dimension, journal, bondParticlesToKernelFunctions=True)
    )


def _model(journal: Journal, withElements: bool, withParticles: bool) -> MPMModel:
    """The model with the block of elements, the block of particles, or both, each block with its initial velocity: the
    elements' at the nodes (the model's velocity), the particles' in the particles."""

    model = MPMModel(2)
    if withElements:
        _addBlockOfElements(model, journal)
    if withParticles:
        _addBlockOfParticles(model, journal, max(model.nodes, default=0) + 1)
    model.prepareYourself(journal)

    if withElements:
        for element in model.elements.values():
            element.setInitialCondition("initialize material", np.array([0.0]))
        displacement = model.nodeFields["displacement"]
        for node in model.nodeSets["elements_all"]:
            displacement["V"][displacement.indexOfNode(node)] = ELEMENTS_VELOCITY

    if withParticles:
        for particle in model.particles.values():
            particle.getResultArray("velocity")[: PARTICLES_VELOCITY.size] = PARTICLES_VELOCITY
            particle.acceptStateAndPosition()

    return model


#: The fraction of the critical time step of the elements the increments use. The critical time step ``CPE8UL`` reports
#: is above the true one of its lumped mass: at the default courant number, round-off grows into a deformation of the
#: free block within a few increments.
COURANT_NUMBER = 0.3


class _SolverRecordingTheMomentum(NED):
    """``NED``, reporting every increment, and recording the linear momentum of the model at every acceptance."""

    def __init__(self, journal: Journal):
        super().__init__({}, journal, **{"output-frequency": 1, "courant-number": COURANT_NUMBER})
        self.momentumOfTheIncrements = []

    def acceptIncrement(self, step, model, timeStep):
        super().acceptIncrement(step, model, timeStep)
        self.momentumOfTheIncrements.append(self.secondOrderMomentum(self._rawLumpedMass, self._V, model))


def _solve(
    model: MPMModel, journal: Journal, timeIncrement: float, withEnsightOutput: bool = False
) -> tuple[_SolverRecordingTheMomentum, EnsightOutputManager | None]:
    """Solve N_INCREMENTS increments of the given size, or of the critical time step of the elements, if smaller;
    with the Ensight output of the solution of the nodes of the block of elements, if asked for."""

    solver = _SolverRecordingTheMomentum(journal)
    fieldOutputController = MPMFieldOutputController(model, journal)
    outputManagers = []
    if withEnsightOutput:
        fieldOutputController.addPerNodeFieldOutput(
            "U", model.nodeFields["displacement"].subset(model.elementSets["elements_all"]), "U"
        )
    fieldOutputController.initializeJob()
    if withEnsightOutput:
        ensightOutput = EnsightOutputManager("ensight", model, fieldOutputController, journal, None)
        ensightOutput.createPerNodeOutput(fieldOutputController.fieldOutputs["U"])
        ensightOutput.initializeJob()
        outputManagers.append(ensightOutput)
    try:
        AdaptiveStepForExplicitSimulations(
            0,
            model,
            fieldOutputController,
            journal,
            {},
            solver,
            outputManagers,
            StepActionCollection(),
            stepLength=1.0,
            startInc=timeIncrement,
            maxInc=timeIncrement,
            minInc=timeIncrement,
            maxNumInc=N_INCREMENTS,
        ).solve()
    finally:
        fieldOutputController.finalizeJob()
        for outputManager in outputManagers:
            outputManager.finalizeJob()

    return solver, outputManagers[0] if outputManagers else None


def _solveTheThreeModels():
    """The coupled model, solved on the critical time step of its elements, the block of elements alone, which has the
    same, and the block of particles alone, given the same time increment."""

    journal = Journal(verbose=False)
    coupled = _model(journal, withElements=True, withParticles=True)
    coupledSolver, ensightOutput = _solve(coupled, journal, 1.0, withEnsightOutput=True)
    timeIncrement = coupledSolver._largestTimeIncrement

    elementsAlone = _model(journal, withElements=True, withParticles=False)
    _solve(elementsAlone, journal, 1.0)
    particlesAlone = _model(journal, withElements=False, withParticles=True)
    _solve(particlesAlone, journal, timeIncrement)

    return coupled, coupledSolver, ensightOutput, elementsAlone, particlesAlone, timeIncrement


def test_two_free_blocks_move_as_each_does_alone_and_the_momentum_holds(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    coupled, coupledSolver, ensightOutput, elementsAlone, particlesAlone, timeIncrement = _solveTheThreeModels()

    assert np.isfinite(timeIncrement), "the elements report no critical time step"

    # the block of elements: bitwise as alone, at every node, in the solution and the velocity
    expected, actual = elementsAlone.nodeFields["displacement"], coupled.nodeFields["displacement"]
    for node in coupled.nodeSets["elements_all"]:
        rowAlone = expected.indexOfNode(elementsAlone.nodes[node.label])
        for entry in ("U", "V"):
            np.testing.assert_array_equal(
                actual[entry][actual.indexOfNode(node)], expected[entry][rowAlone], err_msg=entry
            )

    # the block of particles: bitwise as alone
    for number, particle in coupled.particles.items():
        np.testing.assert_array_equal(
            particle.getCenterCoordinates(), particlesAlone.particles[number].getCenterCoordinates()
        )

    # both translate with their initial velocities, the block of elements without deforming
    endTime = N_INCREMENTS * timeIncrement
    for node in coupled.nodeSets["elements_all"]:
        np.testing.assert_allclose(actual["U"][actual.indexOfNode(node)], ELEMENTS_VELOCITY * endTime, rtol=1e-9)
    initial = _model(Journal(verbose=False), withElements=False, withParticles=True)
    for number, particle in coupled.particles.items():
        np.testing.assert_allclose(
            particle.getCenterCoordinates() - initial.particles[number].getCenterCoordinates(),
            PARTICLES_VELOCITY * endTime,
            rtol=1e-9,
        )

    # the momentum of the model, in every increment, the zero increment included
    momentum = DENSITY * (
        BLOCK_OF_ELEMENTS_SIZE**2 * ELEMENTS_VELOCITY + BLOCK_OF_PARTICLES_SIZE**2 * PARTICLES_VELOCITY
    )
    assert len(coupledSolver.momentumOfTheIncrements) == N_INCREMENTS + 1
    for momentumOfTheIncrement in coupledSolver.momentumOfTheIncrements:
        np.testing.assert_allclose(momentumOfTheIncrement, momentum, rtol=1e-12, atol=0.0)

    # the Ensight output of the block of elements, by position: the i-th value of the variable is the solution of the
    # i-th node of the part
    (job,) = ensightOutput._transientPerNodeVariableJobs["U"]
    written = job["fieldOutput"].getLastResult()
    partNodes = list(job["part"].nodes)
    assert len(partNodes) == len(written) == len(coupled.nodeSets["elements_all"])
    for position, node in enumerate(partNodes):
        np.testing.assert_array_equal(written[position], actual["U"][actual.indexOfNode(node)])
