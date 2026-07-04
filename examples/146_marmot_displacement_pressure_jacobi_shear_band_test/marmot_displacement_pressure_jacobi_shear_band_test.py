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
"""Plane-strain compression shear band with the mixed u-p-J
(DisplacementPressureJacobi) particle and LOCAL finite-strain J2 plasticity
(FiniteStrainJ2Plasticity, Voce softening).

Purpose: compare the two VMS pressure stabilization modes of the particle on a
problem with strongly nonlinear, isochoric (plastically incompressible)
material response and localized deformation:

    vmsMode = 0 : pressure-only stabilization (grad(p) part of the strong-form
                  momentum residual, classical PSPG-type term)
    vmsMode = 1 : full VMS (grad(p) + div(S_dev) - rho0*a); inside a shear band
                  div(S_dev) is NOT small, so the difference between the modes
                  is expected to show here (unlike on the smooth Cook problem).

Domain      : 60 mm x 120 mm, nX x nY quad particles (default 15 x 30)
BCs         : mortar weak Dirichlet (smooth multiplier field -> no forced
              boundary pressure checkerboard, cf. example 134):
              - bottom edge : roller (uy = 0)
              - top edge    : prescribed compression uy = totalCompression
              - bottom-left : ux = 0 pin (Lagrange, single particle)
Material    : FiniteStrainJ2Plasticity [K, G, fy, fyInf, eta, H, impl, density]
              E = 11920, nu = 0.49 (K/G ~ 50, near-incompressible elasticity;
              the plastic flow is exactly isochoric). Voce softening
              beta(alpha) = fyInf + (fy - fyInf) exp(-eta alpha) + H alpha with
              fy = 100, fyInf = 80, eta = 30, H = 0.
              NOTE: the softening is LOCAL, so the band width is set by the
              particle spacing (no regularization) -- fine for the purpose of
              comparing pressure stabilizations, NOT mesh-objective.
Imperfection: 5 % yield-stress reduction in a 2x2-particle block at the
              bottom-left corner to seed the band (as in examples 144/145).
Particle    : DisplacementPressureJacobiSQCNIxNSNI/PlaneStrain/Quad
Solver      : NonlinearQuasistaticSolver (implicit), adaptive time stepper

Loading     : displacement control; the softening response has a structural
              snap-back at u_y ~ -2.1 mm which displacement control cannot
              pass, hence the default totalCompression = -2.0 mm.

Observed (nX=15, nY=30, default settings, see vms_mode_comparison.png):
- alpha = 0.1: both modes complete, identical peak (F_y = 6733) and band;
  pressure differences between modes concentrate along the band (< 10 %).
- alpha = 0.5: pressure-only (mode 0) DIVERGES at t ~ 0.43 (residual growth
  in the pressure/jacobi fields at the onset of localization -- the
  inconsistent grad(p)-penalty fights the physical pressure gradient that
  must balance div(S_dev) inside the band), while the consistent full VMS
  (mode 1) completes the run AND gives the cleanest pressure field
  (checkerboard RMS ratio 20.7 % vs 22.2 % of the alpha = 0.1 runs).
"""

import argparse
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
from edelweissmeshfree.constraints.particlemortarweakdirichlet import (
    ParticleMortarWeakDirichletOnParticleSetFactory,
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
from edelweissmeshfree.outputmanagers.ensight import (
    OutputManager as EnsightOutputManager,
)
from edelweissmeshfree.particlemanagers.kdbinorganizedparticlemanager import (
    KDBinOrganizedParticleManager,
)
from edelweissmeshfree.particles.marmot.marmotparticlewrapper import (
    MarmotParticleWrapper,
)
from edelweissmeshfree.solvers.nqs import NonlinearQuasistaticSolver

_EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))


class _ReactionMonitor:
    """Records (prescribed top u_y, total F_y top reaction) after each converged
    increment (see example 145 for the interface rationale). The mortar
    multiplier is solved fresh against the total internal force each increment,
    so reactionForce holds the TOTAL reaction -- no accumulation."""

    def __init__(self, model, top_constraints: dict, total_compression: float):
        self._model = model
        self._top_constraints = top_constraints
        self._total_compression = total_compression
        self.u_history = []
        self.F_history = []

    def initializeJob(self):
        pass

    def finalizeIncrement(self):
        t = self._model.time
        self.u_history.append(self._total_compression * t)
        self.F_history.append(sum(c.reactionForce[1] for c in self._top_constraints.values()))

    def finalizeFailedIncrement(self):
        pass

    def finalizeStep(self):
        pass


def run_sim(
    vmsAlpha=0.1,
    vmsMode=0,
    nX=15,
    nY=30,
    totalCompression=-2.0,
    incSize=0.01,
    outputName=None,
):
    dimension = 2

    np.set_printoptions(linewidth=200, precision=4)

    theJournal = Journal()
    theModel = MPMModel(dimension)

    x0, y0 = 0.0, 0.0
    l, h = 60.0, 120.0
    particleSize = l / nX
    supportRadius = 2.2 * particleSize

    if outputName is None:
        outputName = f"shearband_upj_alpha{vmsAlpha}_mode{vmsMode}"

    def kernelFunctionFactory(node):
        return MarmotMeshfreeKernelFunctionWrapper(
            node, "BSplineBoxed", supportRadius=supportRadius, continuityOrder=2
        )

    # kernel nodes at the particle centres
    theModel = generateRectangularKernelFunctionGrid(
        theModel,
        theJournal,
        kernelFunctionFactory,
        x0=x0 + particleSize / 2.0,
        y0=y0 + particleSize / 2.0,
        l=l - particleSize,
        h=h - particleSize,
        nX=nX,
        nY=nY,
        name="kernel_grid",
    )

    theApproximation = MarmotMeshfreeApproximationWrapper("ReproducingKernel", dimension, completenessOrder=1)

    # near-incompressible elasticity + isochoric Voce-softening J2 plasticity
    E, nu = 11920.0, 0.49
    K = E / (3.0 * (1.0 - 2.0 * nu))
    G = E / (2.0 * (1.0 + nu))
    fy, fyInf, eta, H = 100.0, 80.0, 30.0, 0.0
    implementationType, density = 1, 1e-7

    def makeMaterial(yieldScale=1.0):
        return {
            "material": "FiniteStrainJ2Plasticity",
            "properties": np.array([K, G, yieldScale * fy, yieldScale * fyInf, eta, H, implementationType, density]),
        }

    theMaterial = makeMaterial()
    theMaterialImperfect = makeMaterial(0.95)

    def particleFactory(number, vertexCoordinates, volume):
        xC = np.mean(vertexCoordinates[:, 0])
        yC = np.mean(vertexCoordinates[:, 1])
        isImperfect = xC < x0 + 2 * particleSize and yC < y0 + 2 * particleSize
        return MarmotParticleWrapper(
            "DisplacementPressureJacobiSQCNIxNSNI/PlaneStrain/Quad",
            number,
            vertexCoordinates,
            0.0,  # volume is computed from the vertex coordinates
            theApproximation,
            theMaterialImperfect if isImperfect else theMaterial,
        )

    theModel = generateRectangularQuadParticleGrid(
        theModel,
        theJournal,
        particleFactory,
        x0=x0,
        y0=y0,
        l=l,
        h=h,
        nX=nX,
        nY=nY,
        name="specimen",
    )

    for particle in theModel.particles.values():
        particle.setProperty("newmark-beta beta", 0.0)
        particle.setProperty("newmark-beta gamma", 0.0)
        particle.setProperty("vms alpha", vmsAlpha)
        particle.setProperty("vms mode", float(vmsMode))

    theParticleKernelDomain = ParticleKernelDomain(
        list(theModel.particles.values()), list(theModel.meshfreeKernelFunctions.values())
    )
    theModel.particleKernelDomains["domain"] = theParticleKernelDomain

    theParticleManager = KDBinOrganizedParticleManager(
        theParticleKernelDomain, dimension, theJournal, bondParticlesToKernelFunctions=True
    )

    # mortar weak Dirichlet: smooth multiplier field -> no forced boundary
    # pressure checkerboard (see example 134 docstring)
    multiplierOrder = min(6, nX - 1)
    dirichletBottom = ParticleMortarWeakDirichletOnParticleSetFactory(
        "bottom",
        theModel.particleSets["specimen_bottom"],
        "displacement",
        {1: 0.0},
        theModel,
        multiplierOrder=multiplierOrder,
    )
    dirichletTop = ParticleMortarWeakDirichletOnParticleSetFactory(
        "top",
        theModel.particleSets["specimen_top"],
        "displacement",
        {1: totalCompression},
        theModel,
        multiplierOrder=multiplierOrder,
    )
    # single-point ux pin against rigid body translation in x
    dirichletPin = ParticleLagrangianWeakDirichletOnParticleSetFactory(
        "pin",
        theModel.particleSets["specimen_leftBottom"],
        "displacement",
        {0: 0.0},
        theModel,
    )

    theModel.constraints.update(dirichletBottom)
    theModel.constraints.update(dirichletTop)
    theModel.constraints.update(dirichletPin)

    reactionMonitor = _ReactionMonitor(theModel, dirichletTop, totalCompression)

    theModel.prepareYourself(theJournal)
    theJournal.printPrettyTable(theModel.makePrettyTableSummary(), "summary")

    fieldOutputController = MPMFieldOutputController(theModel, theJournal)

    for fname in ("displacement", "pressure", "jacobi", "stress", "deformation gradient"):
        fieldOutputController.addPerParticleFieldOutput(fname, theModel.particleSets["all"], fname)
    # equivalent-plastic-strain-like hardening variable of FiniteStrainJ2Plasticity
    fieldOutputController.addPerParticleFieldOutput("alphaP", theModel.particleSets["all"], "alphaP")
    fieldOutputController.addPerParticleFieldOutput(
        "vertex displacements",
        theModel.particleSets["all"],
        "vertex displacements",
        f_x=lambda x: np.pad(np.reshape(x, (-1, 2)), ((0, 0), (0, 1)), mode="constant", constant_values=0),
    )
    fieldOutputController.addExpressionFieldOutput(
        None,
        lambda: sum(c.reactionForce for c in dirichletTop.values()),
        "reaction force top",
        saveHistory=True,
        export=f"RF_{outputName}",
    )

    fieldOutputController.initializeJob()

    ensightOutput = EnsightOutputManager(
        f"_ensight_{outputName}",
        theModel,
        fieldOutputController,
        theJournal,
        None,
        configurations=[{"overwrite": True, "intermediateSaveInterval": 10, "transient": True, "nSet": None, "elSet": None}],
    )
    for fname in ("displacement", "pressure", "jacobi", "stress", "deformation gradient", "alphaP"):
        ensightOutput.updateDefinition(fieldOutput=fieldOutputController.fieldOutputs[fname], create="perElement")
    ensightOutput.updateDefinition(
        fieldOutput=fieldOutputController.fieldOutputs["vertex displacements"], create="perNode"
    )
    ensightOutput.initializeJob()

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

        if len(reactionMonitor.u_history) >= 1:
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                u_arr = -np.array(reactionMonitor.u_history)
                F_arr = np.array(reactionMonitor.F_history)

                fig, ax = plt.subplots(figsize=(7, 5))
                ax.plot(u_arr, F_arr, "b-o", markersize=3, linewidth=1.2)
                ax.axhline(0.0, color="0.6", linewidth=0.8)
                ax.set_xlabel(r"compressive shortening  $-u_y$  (mm)")
                ax.set_ylabel(r"top reaction  $F_y$  (N)")
                ax.set_title(f"Shear band, u-p-J particle, VMS mode {vmsMode}, alpha {vmsAlpha}")
                ax.grid(True, linestyle="--", alpha=0.5)
                fig.tight_layout()
                fig.savefig(os.path.join(_EXAMPLE_DIR, f"load_displacement_{outputName}.png"), dpi=150)
                plt.close(fig)
            except ImportError:
                pass

    return theModel, fieldOutputController, reactionMonitor


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_sim():
    import warnings

    warnings.filterwarnings("ignore")

    theModel, fieldOutputController, _ = run_sim(nX=6, nY=12, totalCompression=-1.5, incSize=0.05)

    res = fieldOutputController.fieldOutputs["alphaP"].getLastResult().flatten()

    if not os.path.exists("gold.csv"):
        pytest.skip("gold.csv not found - run with --create-gold to create it.")

    gold = np.loadtxt("gold.csv")
    # LOOSE tolerance: the Marmot finite-strain plastic path is currently not
    # run-to-run deterministic (same class of issue as the confirmed
    # uninitialized-memory read behind the example-144 non-determinism, still
    # open), and the softening localization amplifies it to a few percent of
    # the band alphaP. Tighten once the Marmot issue is fixed.
    assert np.isclose(res, gold, rtol=0.15, atol=5e-3).all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--create-gold", dest="create_gold", action="store_true",
                        help="run the test_sim configuration and store its result as gold.csv")
    parser.add_argument("--vmsAlpha", "-a", dest="vmsAlpha", type=float, default=0.1)
    parser.add_argument("--vmsMode", "-m", dest="vmsMode", type=int, choices=[0, 1], default=0,
                        help="0: pressure-only VMS, 1: full VMS (grad p + div S_dev - rho0 a)")
    parser.add_argument("--nX", type=int, default=15)
    parser.add_argument("--nY", type=int, default=30)
    parser.add_argument("--compression", type=float, default=-2.0)
    parser.add_argument("--incSize", type=float, default=0.01)
    args = parser.parse_args()

    if args.create_gold:
        # must match the test_sim configuration exactly
        theModel, fieldOutputController, monitor = run_sim(nX=6, nY=12, totalCompression=-1.5, incSize=0.05)
        res = fieldOutputController.fieldOutputs["alphaP"].getLastResult().flatten()
        np.savetxt("gold.csv", res)
    else:
        theModel, fieldOutputController, monitor = run_sim(
            vmsAlpha=args.vmsAlpha,
            vmsMode=args.vmsMode,
            nX=args.nX,
            nY=args.nY,
            totalCompression=args.compression,
            incSize=args.incSize,
        )
