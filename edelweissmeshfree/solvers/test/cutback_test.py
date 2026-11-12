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
"""A failure in Marmot -- a material that cannot integrate an increment -- requests a cutback, in the material point
wrapper and in the threaded Marmot kernel alike, and the quasi-static solvers cut the increment back.

One cell of a grid holds one material point of the gradient-enhanced finite-strain Drucker-Prager material, elastic
throughout (its cohesion is far above any stress here). The right edge of the grid is pushed to the left by more than
the width of the cell, so the first full increment would turn the material point inside out. The material refuses it
("non-positive elastic volume ratio"), Marmot throws, the solver cuts the increment back, and the smaller increment
converges.

The deprecated solver cuts back by its own factor. EdelweissFE's solver meets the failure where it updates the material
points (:meth:`~edelweissmeshfree.models.mpmdiscretesystem.MPMDiscreteSystem.updateMaterialPoints`); the
material point asks for half the increment, as an EdelweissFE element does, and the step cuts back by the larger of
this and its ``cutbackFactor``.

A particle translates a failure of Marmot into a cutback request in its wrapper as well, in the implicit kernel and in
the explicit update, and the request reaches the solver from whichever thread of the pool evaluated the particle. The
cells and cell elements translate it too, but cannot be made to fail: their kernels integrate the stresses their
material points have computed before.
"""

import itertools

import numpy as np
import pytest
from edelweissfe.journal.journal import Journal
from edelweissfe.numerics import parallelizationutilities
from edelweissfe.timesteppers.adaptivetimestepper import AdaptiveTimeStepper
from edelweissfe.utils.exceptions import CutbackRequest

CELL_WIDTH = 2.0
SHORTENING = 1.1 * CELL_WIDTH
CUTBACK_FACTOR = 0.25

# K, G, c0, phi, psi, H, epsF, omegaMax, l, m, rho
ELASTIC_DRUCKERPRAGER = {
    "material": "GradientEnhancedFiniteStrainDruckerPrager",
    "properties": np.array([3500.0, 1500.0, 1e9, 30.0, 10.0, 100.0, 0.02, 0.99, 2.0, 1.0, 1.0]),
}


class RecordingJournal(Journal):
    """A journal that keeps what it is told."""

    def __init__(self):
        super().__init__()
        self.messages = []

    def message(self, message: str, senderIdentification: str, level: int = 1):
        self.messages.append(str(message))
        super().message(message, senderIdentification, level)

    def errorMessage(self, errorMessage: str, senderIdentification: str):
        self.messages.append(str(errorMessage))
        super().errorMessage(errorMessage, senderIdentification)


def _compressOneCell(solve):
    """Push the right edge of a one-cell grid to the left by more than the width of the cell.

    Parameters
    ----------
    solve
        Solves the step, ``solve(model, mpmManager, dirichlets, fieldOutputController, journal)``, and returns its time
        stepper.

    Returns
    -------
    tuple
        The model, the journal, and the time stepper of the step.
    """

    from edelweissfe.stepactions.dirichlet import StepAction as Dirichlet

    from edelweissmeshfree.fieldoutput.fieldoutput import MPMFieldOutputController
    from edelweissmeshfree.generators import (
        rectangulargridgenerator,
        rectangularmpgenerator,
    )
    from edelweissmeshfree.models.mpmmodel import MPMModel
    from edelweissmeshfree.mpmmanagers.simplempmmanager import (
        SimpleMaterialPointManager,
    )

    journal = RecordingJournal()
    model = MPMModel(2)

    rectangulargridgenerator.generateModelData(
        model,
        journal,
        x0=0.0,
        l=CELL_WIDTH,
        y0=-1.0,
        h=4.0,
        nX=1,
        nY=1,
        cellProvider="LagrangianMarmotCell",
        cellType="GradientEnhancedFiniteStrain/Quad4",
    )
    rectangularmpgenerator.generateModelData(
        model,
        journal,
        x0=0.5,
        l=1.0,
        y0=0.5,
        h=1.0,
        nX=1,
        nY=1,
        mpProvider="marmot",
        mpType="GradientEnhancedFiniteStrain/PlaneStrain",
        material=ELASTIC_DRUCKERPRAGER,
    )

    model.prepareYourself(journal)

    mpmManager = SimpleMaterialPointManager(model.cellSets["all"], model.materialPointSets["all"])

    fieldOutputController = MPMFieldOutputController(model, journal)
    fieldOutputController.initializeJob()

    dirichlets = [
        Dirichlet("left", model.nodeSets["rectangular_grid_left"], "displacement", {0: 0.0}, model, journal),
        Dirichlet("fix", model.nodeSets["rectangular_grid_leftBottom"], "displacement", {1: 0.0}, model, journal),
        Dirichlet("right", model.nodeSets["rectangular_grid_right"], "displacement", {0: -SHORTENING}, model, journal),
    ]

    try:
        timeStepper = solve(model, mpmManager, dirichlets, fieldOutputController, journal)
    finally:
        fieldOutputController.finalizeJob()

    return model, journal, timeStepper


def _assertMarmotFailedAndTheIncrementWasCutBackTo(incrementSize: float, journal: RecordingJournal):
    assert any("non-positive elastic volume ratio" in message for message in journal.messages), "Marmot did not fail"
    cutbacks = [message for message in journal.messages if message.startswith("Cutback to increment size")]
    assert cutbacks[0] == "Cutback to increment size {:}".format(incrementSize)


def test_a_failing_material_point_is_a_cutback():
    """The solver integrates the material points through the material point wrapper."""

    from edelweissfe.config.linsolve import getLinSolverByName

    from edelweissmeshfree.solvers.deprecated.nqs import NonlinearQuasistaticSolver

    def solve(model, mpmManager, dirichlets, fieldOutputController, journal):
        # the zero increment, then one more: the full step, which fails, and its cutback
        timeStepper = AdaptiveTimeStepper(0.0, 1.0, 1.0, 1.0, 0.1, 1, journal)
        NonlinearQuasistaticSolver(journal).solveStep(
            timeStepper,
            getLinSolverByName("pardiso", {}),
            model,
            fieldOutputController,
            mpmManagers=[mpmManager],
            dirichlets=dirichlets,
            userIterationOptions={"max. iterations": 15, "failed increment cutback factor": CUTBACK_FACTOR},
        )
        return timeStepper

    model, journal, timeStepper = _compressOneCell(solve)

    _assertMarmotFailedAndTheIncrementWasCutBackTo(CUTBACK_FACTOR, journal)
    # the zero increment and the smaller increment converged
    assert timeStepper.numberOfIncrementsDone() == 2
    assert model.time == pytest.approx(CUTBACK_FACTOR)


@pytest.mark.parametrize("cutbackFactor, cutbackSize", [(0.25, 0.5), (0.6, 0.6)])
def test_a_failing_material_point_cuts_the_increment_back_to_the_larger_of_half_and_the_cutback_factor(
    cutbackFactor: float, cutbackSize: float
):
    """EdelweissFE's NIST updates the material points through the discrete system of the model."""

    from edelweissfe.solvers.nonlinearimplicitstatic import NIST
    from edelweissfe.steps.adaptivestep import AdaptiveStep
    from edelweissfe.steps.stepmanager import StepActionCollection

    from edelweissmeshfree.solvers.meshfreeconvergencecriterion import (
        jobInfoWithMeshfreeTolerances,
        meshfreeCriterionOptions,
    )

    def solve(model, mpmManager, dirichlets, fieldOutputController, journal):
        model.mpmManagers.append(mpmManager)
        stepActions = StepActionCollection()
        for dirichlet in dirichlets:
            stepActions["dirichlet"][dirichlet.name] = dirichlet

        jobInfo = jobInfoWithMeshfreeTolerances()
        # the zero increment, then one more: the full step, which fails, and its retry; the step ends there
        step = AdaptiveStep(
            0,
            model,
            fieldOutputController,
            journal,
            jobInfo,
            NIST(jobInfo, journal, linsolver="pardiso", **meshfreeCriterionOptions),
            [],
            stepActions,
            startInc=1.0,
            maxInc=1.0,
            minInc=0.1,
            maxNumInc=1,
            maxIter=15,
            cutbackFactor=cutbackFactor,
        )
        step.solve()
        return step.timeStepper

    model, journal, _ = _compressOneCell(solve)

    _assertMarmotFailedAndTheIncrementWasCutBackTo(cutbackSize, journal)
    assert model.time == pytest.approx(cutbackSize)


#: The finite-strain J2 plasticity of the explicit solver's tests -- K, G, fy, fyInf, eta, H, implementation, density --
#: with an implementation it does not have, so that every stress update fails.
NO_SUCH_IMPLEMENTATION = 7
WHAT_THE_MATERIAL_SAYS = "implementation type not supported"
FAILING_J2_PLASTICITY = {
    "material": "FiniteStrainJ2Plasticity",
    "properties": np.array([1.0, 0.5, 1e9, 1e9, 1.0, 0.0, NO_SUCH_IMPLEMENTATION, 1.0e-8]),
}


def _particleWhoseMaterialFails():
    """An RKPM particle on the unit square, with the nine kernel functions that cover it, of a material that fails at
    every stress update."""

    from edelweissfe.points.node import Node

    from edelweissmeshfree.meshfree.approximations.marmot.marmotmeshfreeapproximation import (
        MarmotMeshfreeApproximationWrapper,
    )
    from edelweissmeshfree.meshfree.kernelfunctions.marmot.marmotmeshfreekernelfunction import (
        MarmotMeshfreeKernelFunctionWrapper,
    )
    from edelweissmeshfree.particles.marmot.marmotparticlewrapper import (
        MarmotParticleWrapper,
    )

    particle = MarmotParticleWrapper(
        "Displacement/SQCNIxNSNI/PlaneStrain/Quad",
        1,
        np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
        0.0,
        MarmotMeshfreeApproximationWrapper("ReproducingKernel", 2, completenessOrder=1),
        FAILING_J2_PLASTICITY,
    )
    kernelCenters = itertools.product((-0.5, 0.5, 1.5), repeat=2)
    particle.assignKernelFunctions(
        [
            MarmotMeshfreeKernelFunctionWrapper(
                Node(label, np.array(center)), "BSplineBoxed", supportRadius=1.5, continuityOrder=3
            )
            for label, center in enumerate(kernelCenters)
        ]
    )
    particle.initializeYourself()
    return particle


def _evaluateImplicitly(particle):
    nDof = particle.nDof
    particle.computeKernels(np.zeros(nDof * nDof), np.zeros(nDof), np.zeros(nDof), np.zeros(nDof), 1.0, 1.0)


def test_a_particle_whose_material_fails_requests_a_cutback():
    particle = _particleWhoseMaterialFails()

    with pytest.raises(CutbackRequest, match=WHAT_THE_MATERIAL_SAYS) as request:
        _evaluateImplicitly(particle)
    assert request.value.cutbackSize == 0.5

    with pytest.raises(CutbackRequest, match=WHAT_THE_MATERIAL_SAYS):
        particle.updatePhysicsExplicit(np.zeros(particle.nDof), 1.0, 1.0)


def test_the_cutback_request_of_a_particle_reaches_the_solver_from_any_thread_of_the_pool(monkeypatch):
    """The solvers evaluate the particles in chunks on the thread pool
    (:func:`~edelweissfe.numerics.parallelizationutilities.mapChunksInParallel`), which hands the first failure on to
    the solver as it is."""

    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    monkeypatch.setattr(parallelizationutilities, "getNumberOfWorkerThreads", lambda: 4)
    particles = [_particleWhoseMaterialFails() for _ in range(16)]

    def evaluate(chunk):
        for particle in chunk:
            _evaluateImplicitly(particle)

    with pytest.raises(CutbackRequest, match=WHAT_THE_MATERIAL_SAYS):
        parallelizationutilities.mapChunksInParallel(particles, evaluate)
