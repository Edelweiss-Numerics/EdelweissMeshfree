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
#  Thomas Mader    |  thomas.mader@boku.ac.at
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
"""
2D plane-strain compression test using GradientPlasticitySmallStrainSNI particles
with the implicit gradient-enhanced von Mises plasticity material (GradientVonMises).

Domain    : 20 mm x 40 mm
Grid      : 10 x 20 quad particles, kernel nodes at particle centers (RKPM)
BCs       : Lagrange multiplier weak Dirichlet
            - bottom: ux = uy = 0 (fully fixed)
            - top   : uy = -0.2 mm (prescribed compression), ux = 0
Imperfection: 5 % yield-stress reduction at centre row triggers shear band
Integration : Variationally Consistent Integration (VCI)

Particle type  : GradientPlasticitySmallStrainSNI/PlaneStrain/Quad
Material       : GradientVonMises
Solver         : NonlinearQuasistaticSolver (implicit)
"""

import edelweissfe.utils.performancetiming as performancetiming
import numpy as np
import pytest
from edelweissfe.journal.journal import Journal
from edelweissfe.linsolve.pardiso.pardiso import pardisoSolve
from edelweissfe.timesteppers.adaptivetimestepper import AdaptiveTimeStepper
from edelweissfe.utils.exceptions import StepFailed

from edelweissmeshfree.constraints.particlelagrangianweakdirichlet import (
    ParticleLagrangianWeakDirichletOnParticleSetFactory,
)
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
from edelweissmeshfree.meshfree.vci import (
    BoundaryParticleDefinition,
    VariationallyConsistentIntegrationManager,
)
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.outputmanagers.ensight import OutputManager as EnsightOutputManager
from edelweissmeshfree.particlemanagers.kdbinorganizedparticlemanager import (
    KDBinOrganizedParticleManager,
)
from edelweissmeshfree.particles.marmot.marmotparticlewrapper import MarmotParticleWrapper
from edelweissmeshfree.solvers.nqs import NonlinearQuasistaticSolver


def run_sim():
    dimension = 2

    np.set_printoptions(linewidth=200, precision=4)

    theJournal = Journal()
    theModel = MPMModel(dimension)

    # ── geometry ─────────────────────────────────────────────────────────────
    x0 = 0.0
    y0 = 0.0
    l  = 20.0   # width  [mm]
    h  = 40.0   # height [mm]
    nX = 10     # particles in x
    nY = 20     # particles in y

    particleSize = l / nX   # 2.0 mm (assumed square particles)

    # ── kernel function grid: one node per particle, placed at particle centre ──
    # np.mgrid[a:b:n*1j] creates n points from a to b (inclusive).
    # Offset by half a particle so each node sits at a particle centre.
    supportRadius = particleSize * 2.5

    def kernelFunctionFactory(node):
        return MarmotMeshfreeKernelFunctionWrapper(
            node, "BSplineBoxed", supportRadius=supportRadius, continuityOrder=2
        )

    theModel = generateRectangularKernelFunctionGrid(
        theModel,
        theJournal,
        kernelFunctionFactory,
        x0 = x0 + particleSize / 2.0,
        y0 = y0 + particleSize / 2.0,
        l  = l - particleSize,
        h  = h - particleSize,
        nX = nX,
        nY = nY,
        name = "kernel_grid",
    )

    # ── reproducing-kernel approximation (completeness order 1) ───────────────
    theApproximation = MarmotMeshfreeApproximationWrapper(
        "ReproducingKernel", dimension, completenessOrder=1
    )

    # ── material: implicit-gradient von Mises plasticity with softening ───────
    # Properties: [E, nu, fy0, H, g, implementation, density]
    E, nu, fy0, H, g = 20000.0, 0.3, 20.0, -2000.0, 4.0

    theMaterial = {
        "material": "GradientVonMises",
        "properties": np.array([E, nu, fy0, H, g, 0.0, 0.0]),
    }

    # 5 % yield-stress reduction at centre row to trigger shear band
    theMaterialImperfect = {
        "material": "GradientVonMises",
        "properties": np.array([E, nu, fy0 * 0.95, H, g, 0.0, 0.0]),
    }

    # ── particle properties: [VCI order, Newmark-β β, Newmark-β γ] ────────────
    particleProperties = np.array([1.0, 0.25, 0.5])

    def particleFactory(number, vertexCoordinates, volume):
        yCentroid = np.mean(vertexCoordinates[:, 1])
        isImperfect = abs(yCentroid - h / 2.0) < particleSize * 0.6
        mat = theMaterialImperfect if isImperfect else theMaterial
        p = MarmotParticleWrapper(
            "GradientPlasticitySmallStrainSNI/PlaneStrain/Quad",
            number,
            vertexCoordinates,
            volume,
            theApproximation,
            mat,
        )
        p.setProperties(particleProperties)
        return p

    theModel = generateRectangularQuadParticleGrid(
        theModel,
        theJournal,
        particleFactory,
        x0 = x0,
        y0 = y0,
        l  = l,
        h  = h,
        nX = nX,
        nY = nY,
        name = "specimen",
    )

    # ── particle–kernel domain ────────────────────────────────────────────────
    theParticleKernelDomain = ParticleKernelDomain(
        list(theModel.particles.values()),
        list(theModel.meshfreeKernelFunctions.values()),
    )
    theModel.particleKernelDomains["domain"] = theParticleKernelDomain

    # ── particle manager: frozen connectivity after first step (small strain) ──
    theParticleManager = KDBinOrganizedParticleManager(
        theParticleKernelDomain,
        dimension,
        theJournal,
        bondParticlesToKernelFunctions=False,
        kinematicMode="small_strain",
    )

    # ── VCI (Variationally Consistent Integration) ─────────────────────────────
    theBoundary = [
        BoundaryParticleDefinition(theModel.particleSets["specimen_left"],   np.empty(2), 4),
        BoundaryParticleDefinition(theModel.particleSets["specimen_right"],  np.empty(2), 2),
        BoundaryParticleDefinition(theModel.particleSets["specimen_bottom"], np.empty(2), 1),
        BoundaryParticleDefinition(theModel.particleSets["specimen_top"],    np.empty(2), 3),
    ]
    vciManager = VariationallyConsistentIntegrationManager(
        list(theModel.particles.values()),
        list(theModel.meshfreeKernelFunctions.values()),
        theBoundary,
    )

    # ── boundary conditions (Lagrange multiplier weak Dirichlet) ──────────────
    totalCompression = -5 # mm (compressive)

    dirichletBottom = ParticleLagrangianWeakDirichletOnParticleSetFactory(
        "bottom", theModel.particleSets["specimen_bottom"],
        "displacement", {0: 0.0, 1: 0.0}, theModel, location="center"
    )
    dirichletTop = ParticleLagrangianWeakDirichletOnParticleSetFactory(
        "top", theModel.particleSets["specimen_top"],
        "displacement", {1: totalCompression}, theModel, location="center"
    )

    theModel.constraints.update(dirichletBottom)
    theModel.constraints.update(dirichletTop)

    theModel.prepareYourself(theJournal)
    theJournal.printPrettyTable(theModel.makePrettyTableSummary(), "summary")

    # ── field output ──────────────────────────────────────────────────────────
    fieldOutputController = MPMFieldOutputController(theModel, theJournal)

    fieldOutputController.addPerParticleFieldOutput(
        "displacement",
        theModel.particleSets["all"],
        "displacement",
    )
    fieldOutputController.addPerParticleFieldOutput(
        "vertex displacements",
        theModel.particleSets["all"],
        "vertex displacements",
        f_x=lambda x: np.pad(np.reshape(x, (-1, 2)), ((0, 0), (0, 1)), mode="constant", constant_values=0),
    )
    fieldOutputController.addPerParticleFieldOutput(
        "plastic multiplier",
        theModel.particleSets["all"],
        "plastic multiplier",
    )
    fieldOutputController.addPerParticleFieldOutput(
        "stress",
        theModel.particleSets["all"],
        "stress",
    )
    fieldOutputController.addPerParticleFieldOutput(
        "strain",
        theModel.particleSets["all"],
        "strain",
    )
    fieldOutputController.initializeJob()

    # ── Ensight output (overwrite=True → no timestamp in filename) ─────────────
    ensightOutput = EnsightOutputManager(
        "ensight",
        theModel,
        fieldOutputController,
        theJournal,
        None,
        configurations=[{"overwrite": True, "intermediateSaveInterval": 10, "transient": True, "nSet": None, "elSet": None}],
    )
    ensightOutput.updateDefinition(
        fieldOutput=fieldOutputController.fieldOutputs["displacement"], create="perElement"
    )
    ensightOutput.updateDefinition(
        fieldOutput=fieldOutputController.fieldOutputs["vertex displacements"], create="perNode"
    )
    ensightOutput.updateDefinition(
        fieldOutput=fieldOutputController.fieldOutputs["plastic multiplier"], create="perElement"
    )
    ensightOutput.updateDefinition(
        fieldOutput=fieldOutputController.fieldOutputs["stress"], create="perElement"
    )
    ensightOutput.updateDefinition(
        fieldOutput=fieldOutputController.fieldOutputs["strain"], create="perElement"
    )
    ensightOutput.initializeJob()

    # ── time stepping & solver ────────────────────────────────────────────────
    incSize = 0.1
    adaptiveTimeStepper = AdaptiveTimeStepper(
        0.0, 1.0, incSize, incSize, incSize / 1e4, 100, theJournal, increaseFactor=1.2
    )

    nonlinearSolver = NonlinearQuasistaticSolver(theJournal)

    iterationOptions = {
        "max. iterations": 20,
        "critical iterations": 5,
        "allowed residual growths": 3,
    }

    try:
        nonlinearSolver.solveStep(
            adaptiveTimeStepper,
            pardisoSolve,
            theModel,
            fieldOutputController,
            outputManagers=[ensightOutput],
            particleManagers=[theParticleManager],
            constraints=theModel.constraints.values(),
            userIterationOptions=iterationOptions,
            vciManagers=[vciManager],
        )

    except StepFailed as e:
        theJournal.message(f"Step failed: {str(e)}", "error")
        raise

    finally:
        fieldOutputController.finalizeJob()
        ensightOutput.finalizeJob()

        prettytable = performancetiming.makePrettyTable()
        prettytable.min_table_width = theJournal.linewidth
        theJournal.printPrettyTable(prettytable, "Summary")

    return theModel, fieldOutputController


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_sim(assert_gold):
    import warnings

    warnings.filterwarnings("ignore")

    theModel, fieldOutputController = run_sim()

    res = fieldOutputController.fieldOutputs["plastic multiplier"].getLastResult().flatten()
    gold = np.loadtxt("gold.csv")
    assert_gold(res, gold, atol=1e-10)


if __name__ == "__main__":
    theModel, fieldOutputController = run_sim()
