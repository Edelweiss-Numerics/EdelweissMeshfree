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
"""The deprecated quasi-static solver evaluates the finite elements of a meshfree model, and their nodes carry their
total solution from one step to the next.

A strip of plane strain quadrilaterals (EdelweissFE's linear elastic ``CPE4``), one element high, is stretched along x
in two steps, by the same amount in each: the left end is held in x, and its bottom node in y as well, the right end is
pulled. The stress is uniaxial in the plane and uniform, so after the second step the strip carries the force
:math:`E/(1-\\nu^2)\\, \\varepsilon\\, h` of the total stretch. A solver that started the nodes of the elements from zero
in the second step, as it starts the nodes of cells and particles, would end with the force and the displacement of
the second step alone, half of them.
"""

import numpy as np
from edelweissfe.config.linsolve import getLinSolverByName
from edelweissfe.journal.journal import Journal
from edelweissfe.stepactions.dirichlet import StepAction as Dirichlet
from edelweissfe.timesteppers.adaptivetimestepper import AdaptiveTimeStepper

from edelweissmeshfree.fieldoutput.fieldoutput import MPMFieldOutputController
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.models.test._finiteelementstrip import (
    POISSONS_RATIO,
    YOUNGS_MODULUS,
    addFiniteElementStrip,
)
from edelweissmeshfree.solvers.deprecated.nqs import NonlinearQuasistaticSolver

NUMBER_OF_ELEMENTS = 4
STRETCH_PER_STEP = 0.01


def _stretch(model: MPMModel, solver, fieldOutputController, journal: Journal):
    """One step that pulls the right end of the strip by ``STRETCH_PER_STEP`` further, in one increment."""

    dirichlets = [
        Dirichlet("left", model.nodeSets["left"], "displacement", {0: 0.0}, model, journal),
        Dirichlet("leftBottom", model.nodeSets["leftBottom"], "displacement", {1: 0.0}, model, journal),
        Dirichlet("right", model.nodeSets["right"], "displacement", {0: STRETCH_PER_STEP}, model, journal),
    ]

    solver.solveStep(
        AdaptiveTimeStepper(model.time, 1.0, 1.0, 1.0, 1.0, 10, journal),
        getLinSolverByName("pardiso", {}),
        model,
        fieldOutputController,
        dirichlets=dirichlets,
    )


def test_the_nodes_of_the_finite_elements_carry_their_total_solution_into_the_next_step():
    journal = Journal(verbose=False)
    model = MPMModel(2)
    addFiniteElementStrip(model, NUMBER_OF_ELEMENTS)
    model.prepareYourself(journal)

    fieldOutputController = MPMFieldOutputController(model, journal)
    fieldOutputController.initializeJob()

    solver = NonlinearQuasistaticSolver(journal)
    for _ in range(2):
        _stretch(model, solver, fieldOutputController, journal)

    fieldOutputController.finalizeJob()

    displacement = model.nodeFields["displacement"]
    left = [displacement.indexOfNode(node) for node in model.nodeSets["left"]]
    right = [displacement.indexOfNode(node) for node in model.nodeSets["right"]]

    totalStretch = 2 * STRETCH_PER_STEP
    force = YOUNGS_MODULUS / (1.0 - POISSONS_RATIO**2) * totalStretch / NUMBER_OF_ELEMENTS

    np.testing.assert_allclose(displacement["U"][right, 0], totalStretch, rtol=1e-12)
    # the internal forces of the elements at the ends, P = f_int: the reactions to the pull
    np.testing.assert_allclose(displacement["P"][right, 0].sum(), force, rtol=1e-10)
    np.testing.assert_allclose(displacement["P"][left, 0].sum(), -force, rtol=1e-10)
