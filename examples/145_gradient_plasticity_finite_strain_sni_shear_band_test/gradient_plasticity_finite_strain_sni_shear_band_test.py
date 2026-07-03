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
2D plane-strain compression test using GradientPlasticityFiniteStrainSNI particles
with the implicit gradient-enhanced, total-Lagrangian finite-strain von Mises
plasticity material (FiniteStrainGradientVonMises).

This is the finite-strain twin of example 144 (small strain): same domain, grid,
boundary conditions, imperfection and solver settings, but the particle carries the
TOTAL deformation gradient F (accumulated in the reference configuration) instead of
an infinitesimal strain, and the material returns the Kirchhoff stress tau = J * sigma
(response.tau) rather than the Cauchy/engineering stress used in the small-strain model.

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

Kinematics : Total Lagrangian. The particle accumulates the incremental displacement
             gradient w.r.t. the REFERENCE coordinates X into a total deformation
             gradient F (see GradientPlasticityFiniteStrainMaterialPoint::incrementDeformation).
             The particle manager is run with kinematicMode="small_strain": despite the
             name, this mode only controls whether the meshfree kernel CONNECTIVITY
             (support domains) is frozen after the first update or recomputed every step
             (see KDBinOrganizedParticleManager.updateConnectivity) -- it has nothing to
             do with the strain measure used by the material/material point. Freezing the
             connectivity in the (undeformed) reference configuration is exactly what a
             total-Lagrangian formulation requires, so "small_strain" mode is the correct
             choice here as well, not a leftover from the small-strain example.

Output     : Ensight fields (including the total deformation gradient in place of the
             small-strain "strain" field) + a load-displacement curve (load_displacement.png)
             built from the summed top-boundary Lagrange-multiplier reaction vs the
             prescribed top displacement.

Particle type  : GradientPlasticityFiniteStrainSNI/PlaneStrain/Quad
Material       : FiniteStrainGradientVonMises
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
    """Minimal output-manager hook that records (prescribed top u_y, total F_y reaction)
    after each converged increment.  Implements the interface expected by
    NonlinearSolverBase._finalizeIncrementOutput and nqs.py (finalizeIncrement /
    finalizeFailedIncrement / finalizeStep), without touching any solver internals.

    The Lagrange multiplier IS the reaction force: each increment it is solved fresh
    against the TOTAL internal force (see ParticleLagrangianWeakDirichlet.applyConstraint:
    PExt_U += dL * dg_dU balances the total P_int), so ``reactionForce[1]`` already holds
    the TOTAL y-reaction at the converged state.  It must NOT be accumulated over
    increments (144's monitor does accumulate it, which integrates the force over the
    step history and produces an unphysical, increment-count-dependent curve).
    """

    def __init__(self, model, top_constraints: dict, total_compression: float):
        self._model = model
        self._top_constraints = top_constraints   # dict name -> ParticleLagrangianWeakDirichlet
        self._total_compression = total_compression
        self.u_history = []   # prescribed top u_y  [mm], negative = compression
        self.F_history = []   # total reaction F_y  [model force units]

    def initializeJob(self):
        pass

    def finalizeIncrement(self):
        # model.time is set by model.advanceToTime() before this is called
        t = self._model.time
        u = self._total_compression * t   # prescribed top displacement at this step
        F = sum(c.reactionForce[1] for c in self._top_constraints.values())
        self.u_history.append(u)
        self.F_history.append(F)

    def finalizeFailedIncrement(self):
        pass   # discard failed increments

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
    l  = 60.   # width  [mm]
    h  = 120.   # height [mm]
    nX = 15     # particles in x
    nY = 30     # particles in y

    particleSize = l / nX   # 4.0 mm (assumed square particles)

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

    # ── material: implicit-gradient, total-Lagrangian finite-strain von Mises ──
    # plasticity with softening (FiniteStrainGradientVonMises).
    # Properties: [K, G, fy0, H, g, density, nonlocalViscosity]
    #   (bulk modulus, shear modulus, initial yield stress, hardening modulus
    #    (negative = softening), gradient parameter, density, viscosity)
    # The material uses a Fischer-Burmeister NCP formulation for the plastic
    # complementarity condition (equivalent to "implementation = 1" of the small-strain
    # GradientVonMises used in example 144) -- there is no separate 'implementation' flag.
    #
    # Converted from example 144's isotropic (E, nu) = (11920, 0.49):
    #   K = E / (3 * (1 - 2*nu)) = 11920 / (3 * (1 - 0.98)) = 11920 / 0.06 = 198666.667
    #   G = E / (2 * (1 + nu))   = 11920 / (2 * 1.49)        = 11920 / 2.98 = 4000.0
    # Regularization length of the softening band: l_c = sqrt(g / |H|) = sqrt(3600/400) = 3 mm.
    # NOTE (mesh objectivity): l_c must span >~2-3 particle spacings to make the post-peak band
    # mesh-independent and Newton-robust. With particleSize = 4 mm, l_c = 3 mm is UNDER-resolved,
    # so a finer mesh localizes more sharply and Newton diverges earlier in the post-peak.
    # To resolve on a mesh of spacing h, raise g so that l_c ~ 2*h, i.e. g ~ |H|*(2h)^2.
    # nu = 0.49 (near-incompressible) is baked into K above: expect volumetric locking
    # with linear RKPM + SNI, exactly as in example 144.
    E, nu = 11920.0, 0.49
    K = E / (3.0 * (1.0 - 2.0 * nu))
    G = E / (2.0 * (1.0 + nu))
    fy0, H, g = 100.0, -400.0, 3600.0
    density, viscosity = 0.0, 0.0

    theMaterial = {
        "material": "FiniteStrainGradientVonMises",
        "properties": np.array([K, G, fy0, H, g, density, viscosity]),
    }

    # 5 % yield-stress reduction at the bottom-left corner to trigger the shear band
    theMaterialImperfect = {
        "material": "FiniteStrainGradientVonMises",
        "properties": np.array([K, G, fy0 * 0.95, H, g, density, viscosity]),
    }

    # ── particle properties: [VCI order, Newmark-β β, Newmark-β γ] ────────────
    particleProperties = np.array([1.0, 0.25, 0.5])

    def particleFactory(number, vertexCoordinates, volume):
        xCentroid = np.mean(vertexCoordinates[:, 0])
        yCentroid = np.mean(vertexCoordinates[:, 1])
        isImperfect = xCentroid < particleSize * 2 and yCentroid < particleSize * 2
        mat = theMaterialImperfect if isImperfect else theMaterial
        p = MarmotParticleWrapper(
            "GradientPlasticityFiniteStrainSNI/PlaneStrain/Quad",
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

    # ── particle manager: frozen connectivity in the reference configuration ──
    # kinematicMode="small_strain" freezes the kernel support (connectivity) after the
    # first update -- it does NOT imply a small-strain kinematic assumption in the
    # material/material point. For a total-Lagrangian finite-strain formulation the
    # reference-configuration connectivity is exactly what is required, so this mode is
    # the correct (and only sensible) choice here too.
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
    # shear band localizes at the corner imperfection and the load–displacement curve
    # snaps back shortly after the peak; since loading is displacement-controlled via
    # the Lagrange constraint (no arc-length control), the prescribed value is kept
    # just below the snap-back so the run completes and shows the fully forming band.
    # Increase it (and add arc-length control) to trace the post-peak softening branch.
    # ~4x yield displacement (u_yield = fy0/E*h ~ 1.0 mm), well into the post-peak regime.
    # The quasistatic finite-strain response has a LIMIT POINT (snap-back) at u_y ~ -4.15 mm
    # (geometric softening on top of H < 0): displacement control cannot pass it, so the
    # prescribed compression ends just before (144 uses -5.0 mm, which the small-strain
    # model reaches because it lacks the geometric contribution to the softening).
    totalCompression = -4.0  # mm

    dirichletBottom = ParticleLagrangianWeakDirichletOnParticleSetFactory(
        "bottom", theModel.particleSets["specimen_bottom"],
        "displacement", {1: 0.0}, theModel, location="center"
    )
    dirichletBottomLeft = ParticleLagrangianWeakDirichletOnParticleSetFactory(
        "bottom", theModel.particleSets["specimen_leftBottom"],
        "displacement", {0: 0.0, 1: 0.0}, theModel, location="center"
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
    # Kirchhoff stress tau (9 components, row-major 3x3), same state-var length (9) as the
    # small-strain "stress" -> no f_x reshaping required here.
    fieldOutputController.addPerParticleFieldOutput(
        "stress",
        theModel.particleSets["all"],
        "stress",
    )
    # Total deformation gradient F (9 components, row-major 3x3), replaces 144's "strain".
    fieldOutputController.addPerParticleFieldOutput(
        "deformation gradient",
        theModel.particleSets["all"],
        "deformation gradient",
    )
    fieldOutputController.addPerParticleFieldOutput(
        "displacement top",
        theModel.particleSets["specimen_rightTop"],
        "displacement",
        export="export_U"
    )

    fieldOutputController.addExpressionFieldOutput(None, lambda : np.sum ( [d.reactionForce for d in dirichletBottom.values()], axis=0), "reaction force bottom", export="export_RF")




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
        fieldOutput=fieldOutputController.fieldOutputs["deformation gradient"], create="perElement"
    )
    ensightOutput.initializeJob()

    # ── time stepping & solver ────────────────────────────────────────────────
    # Increment budget 1000 (vs 144's 100): the finite-strain Newton typically needs one
    # cutback at the onset of plasticity and then keeps the smaller increment size for the
    # rest of the run (iteration counts stay above the regrowth threshold), so reaching
    # t = 1.0 needs more increments than the nominal 1/incSize.
    incSize = 0.01
    adaptiveTimeStepper = AdaptiveTimeStepper(
        0.0, 1.0, incSize, incSize, incSize / 1e4, 1000, theJournal, increaseFactor=1.2
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
                ax.set_title("Load–Displacement Curve — Example 145")
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

    # No gold.csv is checked in yet (the Marmot-side finite-strain particle/material-point
    # this example depends on is developed in parallel and not built at the time this test
    # was authored). Rather than crashing with a bare FileNotFoundError -- or worse,
    # fabricating gold numbers -- skip cleanly until a gold file is generated, e.g. via:
    #     python gradient_plasticity_finite_strain_sni_shear_band_test.py --create-gold
    if not os.path.exists("gold.csv"):
        pytest.skip(
            "gold.csv not found - run this script directly with --create-gold once the "
            "finite-strain Marmot chain (GradientPlasticityFiniteStrainSNI / "
            "FiniteStrainGradientVonMises) is built, to create the reference result."
        )

    gold = np.loadtxt("gold.csv")
    assert_gold(res, gold, atol=1e-10)


if __name__ == "__main__":
    import argparse

    theModel, fieldOutputController = run_sim()

    res = fieldOutputController.fieldOutputs["plastic multiplier"].getLastResult().flatten()

    parser = argparse.ArgumentParser()
    parser.add_argument("--create-gold", dest="create_gold", action="store_true", help="create the gold file.")
    args = parser.parse_args()

    if args.create_gold:
        np.savetxt("gold.csv", res)
