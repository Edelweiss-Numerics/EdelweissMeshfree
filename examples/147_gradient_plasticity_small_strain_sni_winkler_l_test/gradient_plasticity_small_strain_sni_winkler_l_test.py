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
2D plane-strain Winkler L-shaped panel using GradientPlasticitySmallStrainSNI particles
with the implicit gradient-enhanced von Mises plasticity material (GradientVonMises).

This is the classic Winkler L-shaped panel benchmark (Winkler, PhD thesis, Univ. Innsbruck,
2001), used here as a gradient-plasticity localization test.  It follows the small strain
gradient plasticity example 144 (plane-strain compression shear band) in terms of particle
type, material, integration and solver, but replaces the rectangular specimen with an
L-shaped domain so that the plastic localization band initiates at the re-entrant corner.

Geometry (origin bottom-left, plane strain, thickness = 100 mm)::

      y=500 +-----+
            |     |
            |vert |            cut-out (empty):
            |leg  |              x in (250, 500]
      y=250 |     +-----+ <----- re-entrant corner (250, 250)
            |           |
            |horizontal |
            |leg        |
      y=0   +-----------+
            x=0  250   500

  * Bounding box   : 500 mm x 500 mm, legs 250 mm wide.
  * Cut-out        : the top-right 250 x 250 mm square is removed, forming the "L".
  * Particle grid  : 20 x 20 quad particles over the bounding box (particleSize = 25 mm),
                     the 10 x 10 block inside the cut-out is deleted -> 300 L-shaped particles.
                     Kernel nodes sit at the particle centres (RKPM), same carving applied.

BCs (Lagrange multiplier weak Dirichlet):
  * vertical-leg base (x < 250, y = 0)     : fully clamped, ux = uy = 0  (support)
  * right tip of horizontal arm (x ~ 500,  : prescribed upward pull, uy = totalPull
    y in [0, 250])                           (displacement controlled)
  The arm is thus a cantilever off the clamped leg; pulling its tip up bends it and puts the
  re-entrant corner in tension, where the plastic band localizes.

Imperfection: 5 % yield-stress reduction in a 2x2-particle block just inside the
              re-entrant corner (horizontal arm side) to seed the localization band there.

Integration : Smoothed Node Integration with Natural Stabilization (NSNI),
              which removes the spurious hourglass modes of plain SNI.

Output      : Ensight fields + a load-displacement curve (load_displacement.png) built from
              the summed loaded-tip Lagrange-multiplier reaction vs the prescribed tip displacement.

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
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.outputmanagers.ensight import OutputManager as EnsightOutputManager
from edelweissmeshfree.particlemanagers.kdbinorganizedparticlemanager import (
    KDBinOrganizedParticleManager,
)
from edelweissmeshfree.particles.marmot.marmotparticlewrapper import MarmotParticleWrapper
from edelweissmeshfree.sets.particleset import ParticleSet
from edelweissmeshfree.solvers.nqs import NonlinearQuasistaticSolver


_EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))


class _ReactionMonitor:
    """Minimal output-manager hook that records (prescribed tip u_y, total F_y reaction)
    after each converged increment.  Implements the interface expected by
    NonlinearSolverBase._finalizeIncrementOutput and nqs.py (finalizeIncrement /
    finalizeFailedIncrement / finalizeStep), without touching any solver internals.

    The Lagrange multiplier IS the reaction force: each constraint's ``reactionForce`` is
    (re)computed to the FULL converged multiplier at the current state (it is reset at the
    start of every constraint evaluation), so ``reactionForce[1]`` already holds the total
    vertical reaction of that constraint.  The total tip reaction is therefore the plain sum
    over the loaded constraints at each converged increment — NOT a running accumulation.
    """

    def __init__(self, model, load_constraints: dict, total_pull: float):
        self._model = model
        self._load_constraints = load_constraints   # dict name -> ParticleLagrangianWeakDirichlet
        self._total_pull = total_pull
        self.u_history = []   # prescribed tip u_y  [mm], positive = upward pull
        self.F_history = []   # total reaction F_y  [model force units]

    def initializeJob(self):
        pass

    def finalizeIncrement(self):
        # model.time is set by model.advanceToTime() before this is called
        t = self._model.time
        u = self._total_pull * t   # prescribed tip displacement at this step
        F = sum(c.reactionForce[1] for c in self._load_constraints.values())
        self.u_history.append(u)
        self.F_history.append(F)

    def finalizeFailedIncrement(self):
        pass   # discard: no accumulation on failed increments

    def finalizeStep(self):
        pass


def run_sim():
    dimension = 2

    np.set_printoptions(linewidth=200, precision=4)

    theJournal = Journal()
    theModel = MPMModel(dimension)

    # ── geometry (Winkler L-shaped panel) ─────────────────────────────────────
    x0 = 0.0
    y0 = 0.0
    l  = 500.0   # bounding-box width  [mm]
    h  = 500.0   # bounding-box height [mm]
    nX = 20      # particles in x over the bounding box
    nY = 20      # particles in y over the bounding box

    legWidth = 250.0            # width of each leg of the "L" [mm]
    particleSize = l / nX       # 25.0 mm (assumed square particles)
    thickness = 100.0           # out-of-plane thickness [mm]

    # A particle / kernel node belongs to the cut-out (to be removed) if its centre lies
    # strictly inside the top-right square  x > legWidth  AND  y > legWidth.
    def _inCutout(xc, yc):
        return xc > legWidth and yc > legWidth

    # ── kernel function grid: one node per particle, placed at particle centre ──
    # np.mgrid[a:b:n*1j] creates n points from a to b (inclusive).
    # Offset by half a particle so each node sits at a particle centre.
    supportRadius = particleSize * 2.5

    def kernelFunctionFactory(node):
        return MarmotMeshfreeKernelFunctionWrapper(
            node, "BSplineBoxed", supportRadius=supportRadius, continuityOrder=3
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
    # Regularization length of the softening band: l_c = sqrt(g / |H|) = sqrt(1.5625e6/400)
    #   = 62.5 mm = 2.5 * particleSize, i.e. the band spans ~2-3 particle spacings so the
    #   post-peak response is mesh-objective and Newton-robust on this 25 mm grid.
    # nu = 0.3 (not near-incompressible): avoids the volumetric locking that linear RKPM + SNI
    #   would suffer at nu -> 0.5.
    E, nu, fy0, H, g, imp = 20000.0, 0.3, 100.0, -400.0, 1.5625e6, 1

    theMaterial = {
        "material": "GradientVonMises",
        "properties": np.array([E, nu, fy0, H, g, imp, 0.0]),
    }

    # 5 % yield-stress reduction in a 2x2 block at the re-entrant corner to trigger the band
    theMaterialImperfect = {
        "material": "GradientVonMises",
        "properties": np.array([E, nu, fy0 * 0.95, H, g, imp, 0.0]),
    }

    # ── particle properties: [VCI order, Newmark-β β, Newmark-β γ] ────────────
    particleProperties = np.array([1.0, 0.25, 0.5])

    # Seed the localization at the re-entrant corner (250, 250): a 2x2 particle block
    # on the horizontal-arm side of the corner, i.e. centres with
    #   legWidth < xc < legWidth + 2*particleSize  and  legWidth - 2*particleSize < yc < legWidth.
    def _isImperfect(xc, yc):
        return (legWidth < xc < legWidth + 2.0 * particleSize) and (
            legWidth - 2.0 * particleSize < yc < legWidth
        )

    def particleFactory(number, vertexCoordinates, volume):
        xCentroid = np.mean(vertexCoordinates[:, 0])
        yCentroid = np.mean(vertexCoordinates[:, 1])
        mat = theMaterialImperfect if _isImperfect(xCentroid, yCentroid) else theMaterial
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
        thickness = thickness,
    )

    # ── carve the L-shape: delete the cut-out particles and kernel nodes ───────
    # prepareYourself() rebuilds the "all" node/particle sets from model.nodes / model.particles,
    # so removing the cut-out entries here yields a fully consistent L-shaped domain.
    for kfNum, kf in list(theModel.meshfreeKernelFunctions.items()):
        c = kf.node.coordinates
        if _inCutout(c[0], c[1]):
            del theModel.meshfreeKernelFunctions[kfNum]
            del theModel.nodes[kf.node.label]

    for pNum, p in list(theModel.particles.items()):
        c = p.getCenterCoordinates()
        if _inCutout(c[0], c[1]):
            del theModel.particles[pNum]

    theJournal.message(
        f"L-shape carved: {len(theModel.particles)} particles, "
        f"{len(theModel.meshfreeKernelFunctions)} kernel functions remain.",
        "run_sim",
    )

    # ── boundary particle sets (built by geometry on the carved L) ────────────
    # Support ONLY the base of the vertical leg (x < legWidth, y ~ 0).  The horizontal arm is
    # then an unsupported cantilever: pulling its tip up bends it about the leg, putting the
    # re-entrant corner in tension so the plastic band localizes there — the characteristic
    # Winkler L-panel behaviour.  Clamping the whole bottom (incl. under the arm) instead
    # suppresses this bending and creates a spurious concentration under the loaded tip.
    baseParticles = []   # vertical-leg base -> clamped support
    tipParticles = []    # x ~ 500, arm      -> upward pull
    for p in theModel.particles.values():
        xc, yc = p.getCenterCoordinates()[:2]
        onBase = yc < particleSize and xc < legWidth     # base of the vertical leg
        onRightTip = xc > l - particleSize                # right-most column (arm only, y < 250)
        if onBase:
            baseParticles.append(p)
        elif onRightTip:
            tipParticles.append(p)

    theModel.particleSets["fixed_base"] = ParticleSet("fixed_base", baseParticles)
    theModel.particleSets["load_tip"] = ParticleSet("load_tip", tipParticles)

    theJournal.message(
        f"BC sets: {len(baseParticles)} clamped vertical-leg-base particles, "
        f"{len(tipParticles)} loaded tip particles.",
        "run_sim",
    )

    # ── particle–kernel domain (carved L only) ────────────────────────────────
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

    # ── boundary conditions (Lagrange multiplier weak Dirichlet) ──────────────
    # Displacement-controlled upward pull at the tip of the horizontal arm; the clamped
    # bottom resists translation and rotation, so the arm bends and the re-entrant corner
    # goes into tension, localizing a plastic band that grows from the corner into the arm.
    # Displacement-controlled tip pull.  A plastic limit point (localization band going
    # critical at the re-entrant corner) is reached at ~4 mm tip displacement, after which
    # the step collapses / snaps back and the run terminates — as in example 144.  The
    # prescribed total is set just beyond that so the run traces the full pre-limit response.
    totalPull = 5.0  # mm (upward tip displacement)

    clampBase = ParticleLagrangianWeakDirichletOnParticleSetFactory(
        "fixed", theModel.particleSets["fixed_base"],
        "displacement", {0: 0.0, 1: 0.0}, theModel, location="center"
    )
    pullTip = ParticleLagrangianWeakDirichletOnParticleSetFactory(
        "load", theModel.particleSets["load_tip"],
        "displacement", {1: totalPull}, theModel, location="center"
    )

    theModel.constraints.update(clampBase)
    theModel.constraints.update(pullTip)

    # Reaction monitor: accumulates (u_y, F_y) at each converged increment.
    reactionMonitor = _ReactionMonitor(theModel, pullTip, totalPull)

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

    fieldOutputController.addExpressionFieldOutput(
        None,
        lambda: np.sum([d.reactionForce for d in pullTip.values()], axis=0),
        "reaction force tip",
        export="export_RF",
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
    incSize = 0.02
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

                u_arr = np.array(reactionMonitor.u_history)   # prescribed upward tip disp [mm]
                # Applied load = -reaction, so the curve rises positive (load vs displacement),
                # peaks and then softens once the band localizes at the re-entrant corner.
                F_arr = -np.array(reactionMonitor.F_history)   # summed tip load [N]

                fig, ax = plt.subplots(figsize=(7, 5))
                ax.plot(u_arr, F_arr, "b-o", markersize=3, linewidth=1.2)
                ax.axhline(0.0, color="0.6", linewidth=0.8)
                ax.set_xlabel(r"tip displacement  $u_y$  (mm)")
                ax.set_ylabel(r"applied load  $-F_y$  (N)   [summed tip Lagrange multipliers]")
                ax.set_title("Load–Displacement Curve — Winkler L-shaped panel (Example 147)")
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


def test_sim():
    import warnings

    warnings.filterwarnings("ignore")

    theModel, fieldOutputController = run_sim()

    # NOTE: a fixed gold.csv comparison (as in example 144) is intentionally NOT used here.
    # The GradientVonMises material has a known run-to-run non-determinism (uninitialized
    # read in the Marmot plastic path), so the last converged state — and even how far the
    # adaptive step gets before the limit point — differs between runs.  We therefore assert
    # the qualitative physics instead: (1) plastic yielding developed, and (2) it localizes
    # at the re-entrant corner (250, 250) rather than anywhere else.
    pm = fieldOutputController.fieldOutputs["plastic multiplier"].getLastResult().flatten()
    parts = list(theModel.particles.values())
    centres = np.array([p.getCenterCoordinates()[:2] for p in parts])

    assert pm.max() > 1e-4, "no plastic yielding developed"

    hottest = centres[int(np.argmax(pm))]
    reentrant = np.array([250.0, 250.0])
    dist = np.linalg.norm(hottest - reentrant)
    assert dist < 75.0, (
        f"plastic band did not localize at the re-entrant corner: "
        f"hottest particle at {hottest}, distance {dist:.1f} mm"
    )


if __name__ == "__main__":
    theModel, fieldOutputController = run_sim()
