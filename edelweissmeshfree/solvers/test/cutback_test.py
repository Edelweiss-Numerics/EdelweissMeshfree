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
"""

import numpy as np
import pytest
from edelweissfe.journal.journal import Journal
from edelweissfe.timesteppers.adaptivetimestepper import AdaptiveTimeStepper

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


def _compressOneCell(solverType):
    """Push the right edge of a one-cell grid to the left by more than the width of the cell, in at most two
    increments (the zero increment and one more)."""

    from edelweissfe.config.linsolve import getLinSolverByName

    from edelweissmeshfree.fieldoutput.fieldoutput import MPMFieldOutputController
    from edelweissmeshfree.generators import (
        rectangulargridgenerator,
        rectangularmpgenerator,
    )
    from edelweissmeshfree.models.mpmmodel import MPMModel
    from edelweissmeshfree.mpmmanagers.simplempmmanager import (
        SimpleMaterialPointManager,
    )
    from edelweissmeshfree.stepactions.dirichlet import Dirichlet

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
    model.nodeFields["displacement"].createFieldValueEntry("dU")
    model.nodeFields["nonlocal damage"].createFieldValueEntry("dU")

    mpmManager = SimpleMaterialPointManager(model.cellSets["all"], model.materialPointSets["all"])

    fieldOutputController = MPMFieldOutputController(model, journal)
    fieldOutputController.initializeJob()

    dirichlets = [
        Dirichlet("left", model.nodeSets["rectangular_grid_left"], "displacement", {0: 0.0}, model, journal),
        Dirichlet("fix", model.nodeSets["rectangular_grid_leftBottom"], "displacement", {1: 0.0}, model, journal),
        Dirichlet("right", model.nodeSets["rectangular_grid_right"], "displacement", {0: -SHORTENING}, model, journal),
    ]

    # the zero increment, then one more: the full step, which fails, and its cutback
    timeStepper = AdaptiveTimeStepper(0.0, 1.0, 1.0, 1.0, 0.1, 1, journal)

    solver = solverType(journal)
    try:
        solver.solveStep(
            timeStepper,
            getLinSolverByName("pardiso", {}),
            model,
            fieldOutputController,
            mpmManagers=[mpmManager],
            dirichlets=dirichlets,
            userIterationOptions={"max. iterations": 15, "failed increment cutback factor": CUTBACK_FACTOR},
        )
    finally:
        fieldOutputController.finalizeJob()

    return model, timeStepper, journal


def _checkCutback(model, timeStepper, journal):
    assert any("non-positive elastic volume ratio" in message for message in journal.messages), "Marmot did not fail"
    assert any("Cutback to increment size {:}".format(CUTBACK_FACTOR) in message for message in journal.messages)
    # the zero increment and the smaller increment converged
    assert timeStepper.numberOfIncrementsDone() == 2
    assert model.time == pytest.approx(CUTBACK_FACTOR)


def test_a_failing_material_point_is_a_cutback():
    """The serial solver integrates the material points through the material point wrapper."""

    from edelweissmeshfree.solvers.nqs import NonlinearQuasistaticSolver

    _checkCutback(*_compressOneCell(NonlinearQuasistaticSolver))


def test_a_failing_material_point_in_the_threaded_kernel_is_a_cutback():
    """The parallel solver integrates the material points in a threaded Marmot kernel."""

    from edelweissmeshfree.solvers.nqsmarmotparallel import NQSParallelForMarmot

    _checkCutback(*_compressOneCell(NQSParallelForMarmot))
