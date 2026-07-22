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
material response and localized deformation. Both modes share the
orthogonal-subscale (OSS) fluctuation penalty on grad(p) at the smoothing-domain
face centers with tau = alpha h^2/(2 G_elastic) -- it kills pressure
checkerboarding without diffusing the resolved (physical) pressure field:

    vmsMode = 0 : fluctuation penalty only (pressure-only stabilization)
    vmsMode = 1 : additionally weights the RESOLVED-scale strong-form momentum
                  residual grad(p) + div(S_dev) - rho0*a (full VMS). CAUTION:
                  the div(S_dev) tangent is only approximate in the plastic
                  regime (second-order material derivatives unavailable), so
                  Newton robustness degrades for large alpha.

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
              fy = 100, fyInf = 80, eta = 20, H = 0 (a 20 % drop at a MODERATE rate:
              localizes a clear diagonal band AND produces a genuine post-peak load
              drop that plain displacement control traces for every vmsAlpha in
              mode 0; fyInf/eta are CLI-swept). Two failure modes bracket this
              choice: eta = 15 is too GENTLE -- the load merely plateaus (no real
              drop) and the near-zero tangent plus plastic switch cycling stalls
              Newton right at the peak (this was the old default and the reported
              "fails before the band forms"); eta = 30 is too STEEP -- the peak
              becomes a near-singular snap-back that plain displacement control
              cannot trace at all (needs arc length / control='indirect').
              fyInf = 90 is too mild and forms no band.
              NOTE: the softening is LOCAL, so the band width is set by the
              particle spacing (no regularization) -- fine for the purpose of
              comparing pressure stabilizations, NOT mesh-objective.
Imperfection: 5 % yield-stress reduction in a 2x2-particle block at the
              bottom-left corner to seed the band (as in examples 144/145).
Particle    : DisplacementPressureJacobiSQCNIxNSNI/PlaneStrain/Quad
Solver      : NonlinearQuasistaticSolver (implicit), adaptive time stepper

Loading     : displacement control to totalCompression = -1.9 mm (default).

Peak-crossing: at the load peak (u_y ~ -0.9 mm) the diagonal shear band forms as
a near-bifurcation (nearly singular tangent) and the plastic loading/unloading
state cycles between iterations. Two ingredients make plain displacement control
traverse it robustly for EVERY vmsAlpha in mode 0:
  1. MODERATE softening (eta = 20, see Material) -- a genuine post-peak drop, not
     the flat plateau of the too-gentle eta = 15 that stalled Newton at the peak.
  2. A line search with SMALL trial step lengths ({0.15,0.4,0.7,1.0}, enabled from
     the 2nd iteration). The default alphas {0.8,1.0,1.2} are all too large to
     damp the strong-stabilization limit cycle: there the trial residual is
     monotone increasing in the step, the quadratic fit runs off to the alpha=1.5
     clamp, and the cycle never breaks (this was the vmsAlpha=0.5 stall).
Mode 1 (full VMS) still degrades for large alpha because its div(S_dev) tangent
is only approximate in the plastic regime: it crosses for alpha <~ 0.1 but stalls
at the peak for alpha = 0.5 even with the tuned line search -- use mode 0 there,
or control='indirect' (arc length). The known run-to-run non-determinism of the
Marmot plastic path (open issue, cf. example 144; -ftrivial-auto-var-init=zero
does NOT fix it) only jitters the post-peak branch by a few percent now; it no
longer decides whether the peak is crossed.

WARNING (2026-07-15, measured -- see handoff_vms.md part 5): the crossing
statements above hold for the TOTAL-field OSS stabilization (particles receiving
assignTotalNodalSolution). With the INCREMENT-form stabilization (total-nodal-
solution interface removed) the same configurations STALL at the peak (10x20,
alpha=0.02, inc 0.01: stalls at u=0.90mm; alpha=0 crosses, so it is the
alpha-scaled increment-form penalty that blocks the band's pressure
reorganization). Workarounds in increment form: alpha=0, or incSize <= 0.002
(15x30 crosses with 0.002). The peak itself is a LOCAL-softening bifurcation:
band width is always ~2 particle spacings (12mm at 10x20 -> 8mm at 15x30, mesh-
dependent) and the stall point moves earlier under refinement (u=0.90/0.88/0.82mm
at nX=10/15/20) -- only a regularized material makes this mesh-objective.
control='indirect' (band-slip arc length) does NOT cross the bifurcation even
with the line search now ported to the arc-length inner Newton.

Observed (nX=10, nY=20, eta=20/fyInf=80 defaults, bottom edge fully fixed
{ux=uy=0}, displacement control, u = -5 mm):
- clean diagonal band from the bottom-left imperfection to the top-right corner
  (alphaP localization max/mean ~ 9.6, band alphaP ~ 0.22 at u = -3 mm).
- mode 0 crosses the peak (F_y ~ 6.5 kN) and traces the ~15-29 % softening drop
  for alpha in {0, 0.01, 0.05, 0.1, 0.5, 1.0}; mode 1 for alpha in {0.1}.
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
    so reactionForce holds the TOTAL reaction -- no accumulation.

    get_reaction returns the current TOTAL top reaction vector (mortar/lagrange:
    sum of the multiplier reactions; nitsche: the consistency+penalty force),
    exactly as in example 134."""

    def __init__(self, model, get_reaction, controlTarget: float,
                 xlabel: str = r"compressive shortening  $-u_y$  (mm)"):
        self._model = model
        self._get_reaction = get_reaction
        # the control quantity driven linearly by f_t(t) = t: the prescribed top
        # shortening (displacement control) or the imposed band slip (indirect control)
        self._controlTarget = controlTarget
        self.xlabel = xlabel
        self.u_history = []
        self.F_history = []

    def initializeJob(self):
        pass

    def finalizeIncrement(self):
        t = self._model.time
        self.u_history.append(self._controlTarget * t)
        self.F_history.append(self._get_reaction()[1])

    def finalizeFailedIncrement(self):
        pass

    def finalizeStep(self):
        pass


def run_sim(
    particleType="DisplacementPressureJacobiSQCNIxNSNI/PlaneStrain/Quad",
    vmsAlpha=0.02,
    vmsMode=0,
    vmsKawareRatio=0.0,
    nX=15,
    nY=30,
    totalCompression=-1.9,
    incSize=0.01,
    fyInf=80.0,
    eta=20.0,
    control="displacement",
    constraintType="mortar",
    multiplierOrder=6,
    constraintStride=1,
    cwfCorrection="off",
    vci=False,
    vciOrder=1,
    completenessOrder=1,
    nitscheBeta=10.0,
    outputName=None,
):
    """The integration/constraint knobs are ported verbatim from example 134
    (Cook's membrane) so the same variants can be studied here on a NONLINEAR
    (finite-strain, softening J2) localizing problem. See the run_sim docstring
    of example 134 for the detailed rationale and the measured Cook's-membrane
    behaviour of each option; only the geometry-specific mapping differs:

    - the DRIVEN edge is the top (uy = totalCompression, face 3), the ROLLER is
      the bottom (uy = 0, face 1); both carry only the y component. The x
      rigid-body mode is removed by the single-particle bottom-left ux pin,
      which stays a Lagrange point constraint for every constraintType.
    - control : 'displacement' (default) prescribes the top uy directly and
      cannot cross the softening-band load peak once the pressure stabilization
      is strong (alpha >~ 0.05) -- the tangent is near-singular at the limit
      point (see the measured failures in the module docstring). 'indirect'
      switches to INDIRECT DISPLACEMENT CONTROL: the top is loaded by a UNIFORM
      reference pressure scaled by an unknown load factor lambda (no flat platen
      -- a platen + mean-uy control is algebraically identical to displacement
      control), and the arc-length controller drives a LOCAL band slip -- the
      relative uy of two particles straddling the diagonal band at the
      bottom-left imperfection -- monotonically up to abs(totalCompression). That
      slip keeps growing through the load peak while lambda (and the mean
      shortening) may snap back, so the softening branch is traceable. 'indirect'
      requires constraintType='mortar' and cwfCorrection='off' (the top has no
      displacement BC to enforce/correct); it is solved by the arc-length solver
      (no predictor -- the base updateHistory does not thread the load factor).
      The load-displacement plot x-axis is then the imposed band slip, not the
      global shortening.
    - particleType : SQCNIxNSNI (default) or SQCNIxSDI, as in 134.
    - constraintType : 'mortar' (default, smooth multiplier -> no forced
      boundary pressure checkerboard), 'lagrange' (point collocation, with
      constraintStride), or 'nitsche' (NITSCHEDIRICHLET load, beta =
      nitscheBeta * K / h). Do not combine 'nitsche' with cwfCorrection.
    - cwfCorrection : consistent-weak-form traction term on the constrained edges,
      'off' / 'top' (driven edge only) / 'bottom' (reaction roller only) /
      'both'. Both edges constrain only uy, so the mask is (0, 1). Conditionally
      stable, see 134.
    - vci / vciOrder : variationally consistent integration over all four edges.
    - completenessOrder : RKPM completeness order (must be >= vciOrder); order 2
      widens the support radius (2.2 -> 3.2 particle spacings) as in 134."""
    dimension = 2

    np.set_printoptions(linewidth=200, precision=4)

    theJournal = Journal()
    theModel = MPMModel(dimension)

    x0, y0 = 0.0, 0.0
    l, h = 60.0, 120.0
    particleSize = l / nX
    # quadratic completeness needs a larger support (more nodes than monomials everywhere)
    supportRadius = (2.2 if completenessOrder == 1 else 3.2) * particleSize

    if outputName is None:
        outputName = f"shearband_upj_{particleType.split('/')[0]}_alpha{vmsAlpha}_mode{vmsMode}" + (
            f"_cwf-{cwfCorrection}" if cwfCorrection != "off" else ""
        ) + (f"_vci{vciOrder}" if vci else "") + (
            f"_compl{completenessOrder}" if completenessOrder != 1 else ""
        ) + (f"_nitsche{nitscheBeta}" if constraintType == "nitsche" else "")

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

    theApproximation = MarmotMeshfreeApproximationWrapper("ReproducingKernel", dimension, completenessOrder=completenessOrder)

    # near-incompressible elasticity + isochoric Voce-softening J2 plasticity
    E, nu = 11920.0, 0.49
    K = E / (3.0 * (1.0 - 2.0 * nu))
    G = E / (2.0 * (1.0 + nu))
    # MODERATE Voce softening beta(alpha) = fyInf + (fy - fyInf) exp(-eta alpha) + H alpha.
    # The purpose of this example is to check that the VMS pressure stabilization also
    # behaves on a nonlinear, isochoric problem with a DEVELOPING shear band. The defaults
    # (fyInf=80 -> 20 % drop at a MODERATE eta=20 rate) localize a clear diagonal band at
    # the imperfection (measured alphaP concentration ~9.6x at u=-3 mm) AND give a genuine
    # post-peak load drop that plain displacement control traces for every vmsAlpha in
    # mode 0 (with the small-step line search set below). Both neighbours fail: eta=15 is
    # too gentle (the load only plateaus, so the near-zero tangent + plastic switch cycling
    # stalls Newton at the peak -- the reported "fails before the band forms"); eta=30 is
    # the steep snap-back that plain displacement control cannot trace (needs arc length).
    # (fyInf=90 alone is too mild -- the load rises monotonically and no band forms.) fyInf
    # and eta are run_sim/CLI arguments to sweep the softening severity.
    fy, H = 100.0, 0.0
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
            particleType,
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
        particle.setProperty("vms kaware ratio", float(vmsKawareRatio))
        if vci:
            # default order 1 matches the completeness order 1 of the RKPM approximation
            particle.setProperty("VCI order", float(vciOrder))

    theParticleKernelDomain = ParticleKernelDomain(
        list(theModel.particles.values()), list(theModel.meshfreeKernelFunctions.values())
    )
    theModel.particleKernelDomains["domain"] = theParticleKernelDomain

    theParticleManager = KDBinOrganizedParticleManager(
        theParticleKernelDomain, dimension, theJournal, bondParticlesToKernelFunctions=True
    )

    # Boundary conditions, two control modes (see run_sim docstring):
    #   'displacement' -- top uy prescribed directly (mortar/lagrange/nitsche);
    #   'indirect'     -- top loaded by a reference pressure scaled by an unknown
    #                     load factor lambda + a flat uy platen, with the mean top
    #                     shortening driven by the arc-length IndirectControl.
    # The single-point bottom-left ux pin removes the x rigid-body mode in both.
    particleDistributedLoads = []
    extraConstraints = []
    indirectController = None
    nitscheBottom = nitscheTop = None
    # x-axis of the load-displacement plot: prescribed top shortening (displacement
    # control) or imposed band slip (indirect control); overridden in the indirect branch
    monitorTarget = totalCompression
    monitorXlabel = r"compressive shortening  $-u_y$  (mm)"

    # single-point ux pin against rigid body translation in x (mode-independent)
    dirichletPin = ParticleLagrangianWeakDirichletOnParticleSetFactory(
        "pin",
        theModel.particleSets["specimen_leftBottom"],
        "displacement",
        {0: 0.0},
        theModel,
    )

    if control == "indirect":
        if constraintType != "mortar":
            raise ValueError(
                "control='indirect' drives the top by a load factor, not a displacement BC "
                "-- use constraintType='mortar' (bottom roller only)"
            )
        if cwfCorrection != "off":
            raise ValueError("control='indirect' has no top displacement BC to correct; use cwfCorrection='off'")

        from edelweissfe.surfaces.entitybasedsurface import EntityBasedSurface

        from edelweissmeshfree.stepactions.particledistributedload import (
            ParticleDistributedLoad,
        )
        from edelweissmeshfree.stepactions.particleindirectcontrol import IndirectControl

        # bottom roller uy = 0 (mortar, smooth multiplier); its reaction is the total F_y
        mOrder = min(multiplierOrder, nX - 1)
        dirichletBottom = ParticleMortarWeakDirichletOnParticleSetFactory(
            "bottom",
            theModel.particleSets["specimen_bottom"],
            "displacement",
            {1: 0.0},
            theModel,
            multiplierOrder=mOrder,
        )
        theModel.constraints.update(dirichletBottom)
        theModel.constraints.update(dirichletPin)

        # top loaded by a UNIFORM reference pressure scaled by the unknown load factor
        # lambda (positive 'pressure' acts along -normal -> downward compression). NB: no
        # flat uy platen -- a flat platen + mean-uy control is algebraically identical to
        # displacement control and stalls at the same bifurcation. Leaving the top free to
        # deform lets the shear band localize; lambda (hence the mean compression) is then
        # free to snap back while the LOCAL band slip below drives the solution.
        surfaceTop = EntityBasedSurface(
            name="surface_indirect_top",
            faceToEntities={3: list(theModel.particleSets["specimen_top"])},
        )
        particleDistributedLoads.append(
            ParticleDistributedLoad(
                name="indirect_top_pressure",
                model=theModel,
                journal=theJournal,
                particleSurface=surfaceTop,
                distributedLoadType="pressure",
                loadVector=np.array([fy]),  # reference stress scale -> lambda is O(1)
            )
        )
        # LOCAL band-opening control: the relative vertical displacement of two particles
        # straddling the diagonal shear band seeded at the bottom-left imperfection. This
        # slip grows monotonically through the load peak (and any snap-back), so the
        # arc-length solver traces the post-peak branch -- c.u = f_t(t) * slipTarget.
        def _nearestParticle(xt, yt):
            return min(
                theModel.particles.values(),
                key=lambda p: float(np.sum((np.asarray(p.getCenterCoordinates())[:2] - [xt, yt]) ** 2)),
            )

        pAbove = _nearestParticle(x0 + 0.5 * particleSize, y0 + 2.5 * particleSize)  # left of / above band
        pBelow = _nearestParticle(x0 + 2.5 * particleSize, y0 + 0.5 * particleSize)  # right wedge / below band
        controlParticles = [pAbove, pBelow]
        # c.u = uy(pAbove) - uy(pBelow): the sliding corner wedge (pBelow) drops relative
        # to pAbove, so this difference is positive and increasing
        cVector = np.array([0.0, 1.0, 0.0, -1.0])
        slipTarget = abs(totalCompression)  # drive the band slip up to |totalCompression| mm
        indirectController = IndirectControl(
            "indirect", theModel, controlParticles, slipTarget, cVector,
            "displacement", theJournal, f_t=lambda t: t,
        )
        monitorTarget = slipTarget
        monitorXlabel = r"imposed band slip  $\Delta u_y$  (mm)"
        # total vertical reaction = bottom roller reaction (equilibrates the applied
        # compressive load); positive under compression, matching the displacement curves
        getReactionTop = lambda: np.asarray(sum(c.reactionForce for c in dirichletBottom.values()))  # noqa: E731
    elif constraintType == "nitsche":
        if cwfCorrection != "off":
            raise ValueError("nitsche already contains the CWF consistency term; use cwfCorrection='off'")

        from edelweissfe.surfaces.entitybasedsurface import EntityBasedSurface

        from edelweissmeshfree.stepactions.particlenitschedirichlet import (
            ParticleNitscheDirichlet,
        )

        h = l / nX
        beta = nitscheBeta * K / h

        # bottom edge: face 1, roller uy = 0; top edge: face 3, driven uy = totalCompression
        surfaceBottom = EntityBasedSurface(
            name="surface_nitsche_bottom",
            faceToEntities={1: list(theModel.particleSets["specimen_bottom"])},
        )
        surfaceTop = EntityBasedSurface(
            name="surface_nitsche_top",
            faceToEntities={3: list(theModel.particleSets["specimen_top"])},
        )
        nitscheBottom = ParticleNitscheDirichlet(
            "nitsche_bottom", theModel, theJournal, surfaceBottom, {1: 0.0}, beta, dimension
        )
        nitscheTop = ParticleNitscheDirichlet(
            "nitsche_top", theModel, theJournal, surfaceTop, {1: totalCompression}, beta, dimension
        )
        particleDistributedLoads += [nitscheBottom, nitscheTop]
        theModel.constraints.update(dirichletPin)
        getReactionTop = lambda: nitscheTop.reactionForce  # noqa: E731
    else:
        # ---- direct displacement control on both edges (mortar / lagrange) ----
        if constraintType == "mortar":
            # mortar weak Dirichlet: smooth multiplier field -> no forced boundary
            # pressure checkerboard (see example 134 docstring)
            mOrder = min(multiplierOrder, nX - 1)
            dirichletBottom = ParticleMortarWeakDirichletOnParticleSetFactory(
                    "bottom", theModel.particleSets["specimen_bottom"], "displacement", {1: 0.0, 0: 0.0}, theModel, multiplierOrder=mOrder,
            )
            dirichletTop = ParticleMortarWeakDirichletOnParticleSetFactory(
                "top", theModel.particleSets["specimen_top"], "displacement", {1: totalCompression}, theModel, multiplierOrder=mOrder,
            )
        elif constraintType == "lagrange":
            dirichletBottom = ParticleLagrangianWeakDirichletOnParticleSetFactory(
                "bottom", list(theModel.particleSets["specimen_bottom"])[::constraintStride], "displacement", {1: 0.0}, theModel,
            )
            dirichletTop = ParticleLagrangianWeakDirichletOnParticleSetFactory(
                "top", list(theModel.particleSets["specimen_top"])[::constraintStride], "displacement", {1: totalCompression}, theModel,
            )
        else:
            raise ValueError(f"unknown constraintType {constraintType}")

        theModel.constraints.update(dirichletBottom)
        theModel.constraints.update(dirichletTop)
        theModel.constraints.update(dirichletPin)

        if cwfCorrection != "off":
            from edelweissfe.surfaces.entitybasedsurface import EntityBasedSurface

            from edelweissmeshfree.stepactions.particledistributedload import (
                ParticleDistributedLoad,
            )

            # the load vector is the per-component Dirichlet mask. Both edges are rollers
            # (only uy prescribed) -> mask (0, 1). 'top' corrects the driven edge (face 3),
            # 'bottom' the reaction roller (face 1), 'both' corrects both.
            cwfEdges = []
            if cwfCorrection in ("top", "both"):
                cwfEdges.append(("cwf_dirichlet_top", 3, "specimen_top", (0.0, 1.0)))
            if cwfCorrection in ("bottom", "both"):
                cwfEdges.append(("cwf_dirichlet_bottom", 1, "specimen_bottom", (0.0, 1.0)))
            for name, faceID, particleSet, mask in cwfEdges:
                surface = EntityBasedSurface(
                    name=f"surface_{name}",
                    faceToEntities={faceID: list(theModel.particleSets[particleSet])},
                )
                particleDistributedLoads.append(
                    ParticleDistributedLoad(
                        name=name,
                        model=theModel,
                        journal=theJournal,
                        particleSurface=surface,
                        distributedLoadType="cwfcorrection",
                        loadVector=np.array(mask),
                        f_t=lambda t: 1.0,
                    )
                )

        getReactionTop = lambda: sum(c.reactionForce for c in dirichletTop.values())  # noqa: E731

    vciManagers = []
    if vci:
        from edelweissmeshfree.meshfree.vci import (
            BoundaryParticleDefinition,
            VariationallyConsistentIntegrationManager,
        )

        theBoundary = [
            BoundaryParticleDefinition(theModel.particleSets["specimen_left"], np.empty(2), 4),
            BoundaryParticleDefinition(theModel.particleSets["specimen_right"], np.empty(2), 2),
            BoundaryParticleDefinition(theModel.particleSets["specimen_bottom"], np.empty(2), 1),
            BoundaryParticleDefinition(theModel.particleSets["specimen_top"], np.empty(2), 3),
        ]
        vciManagers.append(
            VariationallyConsistentIntegrationManager(
                list(theModel.particles.values()),
                list(theModel.meshfreeKernelFunctions.values()),
                theBoundary,
            )
        )

    reactionMonitor = _ReactionMonitor(theModel, getReactionTop, monitorTarget, xlabel=monitorXlabel)

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
        getReactionTop,
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

    iterationOptions = {
        "max. iterations": 50 if control == "displacement" else 25,
        "critical iterations": 5 if control == "displacement" else 7,
        "allowed residual growths": 10,
        # the jacobi flux r_J = T (kP - p) V0 is a difference of O(K) quantities: its
        # roundoff floor is small relative to the vanishing current flux, so the relative
        # check needs an honest absolute floor (needed by the SQCNIxSDI particle, cf. 134)
        "spec. absolute flux residual tolerances": {"jacobi": 1e-9},
    }
    # Plastic switch cycling (particles alternating between loading/unloading every
    # iteration) stalls plain Newton at the near-singular band-onset peak; damped
    # steps settle it. The DEFAULT line-search alphas {0.8,1.0,1.2} are all too large
    # to damp the strong-stabilization (alpha >~ 0.5) limit cycle -- there the trial
    # residual is monotone increasing in the step length, the quadratic fit extrapolates
    # to the alpha=1.5 clamp, and the cycle never breaks (observed at vmsAlpha=0.5).
    # Small trial steps let the fit find a genuine damping length (alpha -> 0.15..0.3).
    # The arc-length solver damps (ddU, ddLambda) jointly with the same settings, so
    # this applies to BOTH control modes.
    iterationOptions["line search"] = True
    iterationOptions["line search alphas"] = [0.15, 0.4, 0.7, 1.0]
    iterationOptions["line search after n iterations"] = 2
    iterationOptions["line search every n iterations"] = 1

    # the platen equal-value constraint (indirect mode) rides alongside the model constraints
    allConstraints = list(theModel.constraints.values()) + extraConstraints

    try:
        if control == "indirect":
            from edelweissmeshfree.solvers.nqsmparclength import (
                NonlinearQuasistaticMarmotArcLengthSolver,
            )

            # NB: no arc-length PREDICTOR -- the base solveStep's updateHistory does not
            # thread dLambda through (it would crash the arcLength predictor). The
            # IndirectControl itself supplies the arc-length constraint each iteration;
            # the predictor only guesses the initial increment, so omitting it is safe.
            nonlinearSolver = NonlinearQuasistaticMarmotArcLengthSolver(theJournal)
            nonlinearSolver.solveStep(
                adaptiveTimeStepper,
                pardisoSolve,
                theModel,
                fieldOutputController,
                indirectController,
                outputManagers=[ensightOutput, reactionMonitor],
                particleManagers=[theParticleManager],
                constraints=allConstraints,
                userIterationOptions=iterationOptions,
                particleDistributedLoads=particleDistributedLoads,
                vciManagers=vciManagers,
            )
        else:
            nonlinearSolver = NonlinearQuasistaticSolver(theJournal)
            nonlinearSolver.solveStep(
                adaptiveTimeStepper,
                pardisoSolve,
                theModel,
                fieldOutputController,
                outputManagers=[ensightOutput, reactionMonitor],
                particleManagers=[theParticleManager],
                constraints=allConstraints,
                userIterationOptions=iterationOptions,
                particleDistributedLoads=particleDistributedLoads,
                vciManagers=vciManagers,
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

                u_arr = np.abs(np.array(reactionMonitor.u_history))
                F_arr = np.array(reactionMonitor.F_history)

                fig, ax = plt.subplots(figsize=(7, 5))
                ax.plot(u_arr, F_arr, "b-o", markersize=3, linewidth=1.2)
                ax.axhline(0.0, color="0.6", linewidth=0.8)
                ax.set_xlabel(reactionMonitor.xlabel)
                ax.set_ylabel(r"top reaction  $F_y$  (N)")
                ax.set_title(f"Shear band, u-p-J particle, VMS mode {vmsMode}, alpha {vmsAlpha}, control {control}")
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
    parser.add_argument("--vmsAlpha", "-a", dest="vmsAlpha", type=float, default=0.02,
                        help="pressure-stabilization strength. With the eta=20 default + small-step "
                             "line search, mode 0 crosses the band peak for alpha in {0..1.0}; mode 1 "
                             "only up to ~0.1 (approximate plastic tangent), see docstring")
    parser.add_argument("--vmsMode", "-m", dest="vmsMode", type=int, choices=[0, 1, 2, 3], default=0,
                        help="0: pressure-only VMS, 1: full VMS (grad p + div S_dev - rho0 a)")
    parser.add_argument("--vmsKawareRatio", dest="vmsKawareRatio", type=float, default=0.0,
                        help="K-aware stabilization: cap the effective incompressibility at this "
                             "target K/G ratio R (C = alpha h^2/(2 min(G_ref, K_ref/R))); suppression "
                             "factor becomes 1+(alpha/2)R independent of the material. 0 = disabled "
                             "(pure elastic-G). This material has K/G~50, so R~500 gives Cook's-like "
                             "checkerboard control without raising alpha (see handoff_vms.md Part 7.7)")
    parser.add_argument("--nX", type=int, default=15)
    parser.add_argument("--nY", type=int, default=30)
    parser.add_argument("--compression", type=float, default=-2.0)
    parser.add_argument("--incSize", type=float, default=0.01)
    parser.add_argument("--fyInf", dest="fyInf", type=float, default=80.0,
                        help="Voce saturation yield stress (fy=100); fyInf<fy softens. "
                             "Lower = stronger softening. fyInf=90 is too mild (no band); "
                             "fyInf=80 with the steep eta=30 is the snap-back case")
    parser.add_argument("--eta", dest="eta", type=float, default=20.0,
                        help="Voce softening rate; higher = faster/sharper drop. eta=20 (default) "
                             "gives a REAL post-peak load drop the displacement-control solver traces "
                             "for every vmsAlpha in mode 0; eta=15 is too gentle (flat plateau, stalls "
                             "at the peak); eta=30 is the steep snap-back case that needs arc length")
    parser.add_argument("--particleType", "-p", dest="particleType",
                        default="DisplacementPressureJacobiSQCNIxNSNI/PlaneStrain/Quad")
    parser.add_argument("--control", choices=["displacement", "indirect"], default="displacement",
                        help="'indirect': indirect displacement control (arc-length) to cross the load peak; "
                             "requires constraintType=mortar and cwf=off")
    parser.add_argument("--constraintType", choices=["mortar", "lagrange", "nitsche"], default="mortar")
    parser.add_argument("--nitscheBeta", dest="nitscheBeta", type=float, default=10.0,
                        help="with 'nitsche': penalty factor, beta = nitscheBeta * K / h")
    parser.add_argument("--multiplierOrder", type=int, default=6, help="polynomial order of the mortar multiplier field")
    parser.add_argument("--constraintStride", type=int, default=1,
                        help="with 'lagrange': clamp every n-th top/bottom-edge particle")
    parser.add_argument("--cwf", dest="cwf", choices=["off", "top", "bottom", "both"], default="off",
                        help="consistent-weak-form traction correction on the constrained edges (see run_sim docstring)")
    parser.add_argument("--vci", dest="vci", action="store_true",
                        help="variationally consistent integration (test gradient correction)")
    parser.add_argument("--vciOrder", dest="vciOrder", type=int, default=1,
                        help="polynomial order of the VCI constraints (default 1)")
    parser.add_argument("--completenessOrder", dest="completenessOrder", type=int, default=1,
                        help="completeness order of the RKPM approximation (default 1); must be >= vciOrder")
    parser.add_argument("--outputName", dest="outputName", default=None,
                        help="basename for the RF export and the ensight directory")
    args = parser.parse_args()

    if args.create_gold:
        # must match the test_sim configuration exactly
        theModel, fieldOutputController, monitor = run_sim(nX=6, nY=12, totalCompression=-1.5, incSize=0.05)
        res = fieldOutputController.fieldOutputs["alphaP"].getLastResult().flatten()
        np.savetxt("gold.csv", res)
    else:
        theModel, fieldOutputController, monitor = run_sim(
            particleType=args.particleType,
            vmsAlpha=args.vmsAlpha,
            vmsMode=args.vmsMode,
            vmsKawareRatio=args.vmsKawareRatio,
            nX=args.nX,
            nY=args.nY,
            totalCompression=args.compression,
            incSize=args.incSize,
            fyInf=args.fyInf,
            eta=args.eta,
            control=args.control,
            constraintType=args.constraintType,
            multiplierOrder=args.multiplierOrder,
            constraintStride=args.constraintStride,
            cwfCorrection=args.cwf,
            vci=args.vci,
            vciOrder=args.vciOrder,
            completenessOrder=args.completenessOrder,
            nitscheBeta=args.nitscheBeta,
            outputName=args.outputName,
        )
