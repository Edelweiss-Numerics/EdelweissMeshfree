#-*- coding: utf-8 -*-
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

Domain    : 60 mm x 120 mm
Grid      : 15 x 30 quad particles (particleSize = 4 mm), kernel nodes at particle centers (RKPM)
BCs       : Lagrange multiplier weak Dirichlet
            - bottom edge : roller (uy = 0)
            - bottom-left : pin (ux = uy = 0)
            - top edge    : prescribed compression uy = totalCompression
Imperfection: 5 % yield-stress reduction in a 2x2-particle block at the bottom-left corner
              to seed the shear band.
Integration : Smoothed Node Integration with Natural Stabilization (NSNI),
              which removes the spurious hourglass modes of plain SNI.

Output     : Ensight fields + a load-displacement curve (load_displacement.png) built from the
             summed top-boundary Lagrange-multiplier reaction vs the prescribed top displacement.

Particle type  : GradientPlasticitySmallStrainSNI/PlaneStrain/Quad
Material       : GradientVonMises
Solver         : NonlinearQuasistaticSolver (implicit)
"""

import os

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


_EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))


class _ReactionMonitor:
    """Minimal output-manager hook that accumulates (prescribed top u_y, total F_y reaction)
    after each converged increment.  Implements the interface expected by
    NonlinearSolverBase._finalizeIncrementOutput and nqs.py (finalizeIncrement /
    finalizeFailedIncrement / finalizeStep), without touching any solver internals.

    The Lagrange multiplier IS the reaction force; each constraint's ``reactionForce[1]``
    holds the y-component increment of lambda for the last converged Newton state.
    Accumulating these over time steps yields the total vertical reaction.
    """

    def __init__(self, model, top_constraints: dict, total_compression: float):
        self._model = model
        self._top_constraints = top_constraints   # dict name -> ParticleLagrangianWeakDirichlet
        self._total_compression = total_compression
        self.u_history = []   # prescribed top u_y  [mm], negative = compression
        self.F_history = []   # accumulated total reaction F_y  [model force units]
        self._F_accum = 0.0

    def initializeJob(self):
        pass

    def finalizeIncrement(self):
        # model.time is set by model.advanceToTime() before this is called
        t = self._model.time
        u = self._total_compression * t   # prescribed top displacement at this step
        dF = sum(c.reactionForce[1] for c in self._top_constraints.values())
        self._F_accum += dF
        self.u_history.append(u)
        self.F_history.append(self._F_accum)

    def finalizeFailedIncrement(self):
        pass   # discard: no accumulation on failed increments

    def finalizeStep(self):
        pass


def run_sim():
    dimension = 2

    np.set_printoptions(linewidth=200, precision=4)

    theJournal = Journal()
    theModel = MPMModel(dimension)

    # ── geometry ─────────────────────────────────────────────────────────────
    x0 = 0.0
    y0 = 0.0
    l  = 60.#20.0   # width  [mm]
    h  = 120.#40.0   # height [mm]
    nX = 15     # particles in x
    nY = 30     # particles in y

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
    #   implementation: 0 = standard return map, 1 = Fischer-Burmeister NCP (used here)
    # Regularization length of the softening band: l_c = sqrt(g / |H|) = sqrt(3600/400) = 3 mm.
    # NOTE (mesh objectivity): l_c must span >~2-3 particle spacings to make the post-peak band
    # mesh-independent and Newton-robust. With particleSize = 4 mm, l_c = 3 mm is UNDER-resolved,
    # so a finer mesh localizes more sharply and Newton diverges earlier in the post-peak.
    # To resolve on a mesh of spacing h, raise g so that l_c ~ 2*h, i.e. g ~ |H|*(2h)^2.
    # nu = 0.49 is near-incompressible: expect volumetric locking with linear RKPM + SNI.
    #E, nu, fy0, H, g, imp = 11920, 0.3, 100, -400, 3600, 1
    E, nu, fy0, H, g, imp = 11920, 0.49, 100, -400, 3600, 1
    #E, nu, fy0, H, g = 20000.0, 0.3, 200.0, -2000.0, 4.0

    theMaterial = {
        "material": "GradientVonMises",
        "properties": np.array([E, nu, fy0, H, g, imp, 0.0]),
    }

    # 5 % yield-stress reduction at centre row to trigger shear band
    theMaterialImperfect = {
        "material": "GradientVonMises",
        "properties": np.array([E, nu, fy0 * 0.95, H, g, imp, 0.0]),
    }

    # ── particle properties: [VCI order, Newmark-β β, Newmark-β γ] ────────────
    particleProperties = np.array([1.0, 0.25, 0.5])

    def particleFactory(number, vertexCoordinates, volume):
        xCentroid = np.mean(vertexCoordinates[:, 0])
        yCentroid = np.mean(vertexCoordinates[:, 1])
        isImperfect = xCentroid < particleSize * 2 and yCentroid < particleSize * 2
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
    # Compressive prescribed top displacement. With the strong softening (H<0) the
    # shear band localizes at the centre imperfection and the load–displacement curve
    # snaps back shortly after the peak (around uy ~ -0.068 mm); since loading is
    # displacement-controlled via the Lagrange constraint (no arc-length control),
    # the prescribed value is kept just below the snap-back so the run completes and
    # shows the fully forming band. Increase it (and add arc-length control) to trace
    # the post-peak softening branch.
    totalCompression = -5.0  # mm (~2× yield displacement; captures shear band post-peak)

    dirichletBottom = ParticleLagrangianWeakDirichletOnParticleSetFactory(
        "bottom", theModel.particleSets["specimen_bottom"],
        "displacement", {1: 0.0}, theModel, location="center"
        #"displacement", {0: 0.0, 1: 0.0}, theModel, location="center"
    )
    dirichletBottomLeft = ParticleLagrangianWeakDirichletOnParticleSetFactory(
        "bottom", theModel.particleSets["specimen_leftBottom"],
        "displacement", {0: 0.0, 1: 0.0}, theModel, location="center"
        #"displacement", {0: 0.0, 1: 0.0}, theModel, location="center"
    )
    dirichletTop = ParticleLagrangianWeakDirichletOnParticleSetFactory(
        "top", theModel.particleSets["specimen_top"],
        "displacement", {1: totalCompression}, theModel, location="center"
    )

    theModel.constraints.update(dirichletBottom)
    theModel.constraints.update(dirichletBottomLeft)
    theModel.constraints.update(dirichletTop)

    # Reaction monitor: accumulates (u_y, F_y) at each converged increment.
    reactionMonitor = _ReactionMonitor(theModel, dirichletTop, totalCompression)

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
        "ensight_alex_finer",
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
    incSize = 0.01
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
            outputManagers=[ensightOutput, reactionMonitor],
            particleManagers=[theParticleManager],
            constraints=theModel.constraints.values(),
            userIterationOptions=iterationOptions,
           # vciManagers=[vciManager],
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

        # ── load–displacement curve ───────────────────────────────────────────
        if len(reactionMonitor.u_history) >= 1:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                # Compressive shortening on x (u_y <= 0 -> -u_y >= 0); RAW signed reaction on y.
                # Do NOT take abs() of the reaction: if it dips toward/below zero (softening,
                # snap-back, or numerical noise) that must show as a real dip, not be folded
                # into a spurious upward "kink".
                u_arr = -np.array(reactionMonitor.u_history)   # compressive shortening [mm], >= 0
                F_arr = np.array(reactionMonitor.F_history)     # summed top reaction [N], signed

                fig, ax = plt.subplots(figsize=(7, 5))
                ax.plot(u_arr, F_arr, "b-o", markersize=3, linewidth=1.2)
                ax.axhline(0.0, color="0.6", linewidth=0.8)
                ax.set_xlabel(r"compressive shortening  $-u_y$  (mm)")
                ax.set_ylabel(r"reaction  $F_y$  (N)   [summed top Lagrange multipliers]")
                ax.set_title("Load–Displacement Curve — Example 144")
                ax.grid(True, linestyle="--", alpha=0.5)
                fig.tight_layout()

                png_path = os.path.join(_EXAMPLE_DIR, "load_displacement.png")
                fig.savefig(png_path, dpi=150)
                plt.close(fig)
                theJournal.message(
                    f"Load–displacement curve saved to {png_path}  "
                    f"({len(u_arr)} points)",
                    "run_sim",
                )
            except ImportError:
                theJournal.message(
                    "matplotlib not available — skipping load–displacement plot.",
                    "run_sim",
                )

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
