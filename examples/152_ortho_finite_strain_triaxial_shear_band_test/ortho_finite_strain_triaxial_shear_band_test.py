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
Plane-strain TRIAXIAL COMPRESSION of a bedded specimen, RKPM meshfree, with the
gradient-enhanced orthotropic finite-strain damage-plasticity model
``GRADIENTENHANCEDORTHOCDPFINITESTRAIN`` -- run to a localised inclined SHEAR BAND, and
used to show what the PLASTIC CONVECTION OF THE MATERIAL FRAME does inside that band.

WHAT THIS TEST IS FOR
---------------------
The material carries its orthotropy axes on the intermediate stress-free configuration.
Until recently they were frozen at the card orientation at every state; they are now a
closed-form function of the stored plastic deformation gradient (eq. framestate of the
paper),

    e1(Fp) = Fp^-T n0 / ||.||,   e2(Fp) = (I - e1 x e1) Fp e2_0 / ||.||,   e3 = e1 x e2

i.e. the bedding normal is transported as the normal of a material SURFACE and an in-plane
material LINE by Fp.  A shear band that cuts ACROSS the bedding is exactly where the two
choices part company: the frozen frame drifts 14.3 / 28.1 / 53.1 deg from the convected
bedding normal at an accumulated band shear of 0.25 / 0.5 / 1.0.  This test puts a real
band on the screen and measures that drift in it.

The optional 33rd material property selects the frame:  1 = convected (default, the paper),
0 = frozen (legacy).  ``--frozen`` runs the legacy variant, and ``--compare`` runs both and
overlays them.

MODEL
-----
* RKPM, completeness order 1, implicit-gradient approximation, box B-spline kernels;
  one nodally integrated POINT particle per cell (``GradientEnhancedFiniteStrain/PlaneStrain/Point``).
* Two nodal fields: ``displacement`` and ``nonlocal damage`` (the implicit-gradient
  Helmholtz field that regularises the softening).
* Material card: the calibrated Sect.-4 set of T. Mader et al., Acta Mechanica (2023),
  https://doi.org/10.1007/s00707-023-03706-z, with the paper's own 'bedded' Walpole weight
  set relabelled into the code's axis order (the code puts the bedding NORMAL first, main.tex
  puts it second).

SPECIMEN, SUPPORTS AND WHY THEY ARE WHAT THEY ARE
-------------------------------------------------
Rectangle W x H = 10 x 20 mm, load axis y.  The nonlocal length l = 1.25 mm is FIXED by the
material, so the SPECIMEN is sized against it (H/l = 16) rather than the other way round --
the first cylinder study never localised because l equalled the specimen radius and the
Helmholtz averaging spanned the whole body.

    bottom face (y = 0) : u_y = 0                 frictionless platen
    top face (y = H)    : u_y = -uMax             frictionless platen, uMax = 20 % of H
    one bottom particle : u_x = 0                 kills the x translation, nothing else

FREE LATERAL BOUNDARIES, and that is a deliberate choice with a caveat.  A triaxial cell
applies a CONSTANT CONFINING PRESSURE, and that load is not available here: a nodally
integrated point particle carries no faces, so ``ParticleDistributedLoad`` (which needs an
``EntityBasedSurface``) cannot reach it, and ``computeDistributedLoad`` is a no-op -- exactly
as in the micropolar point particle this one derives from.  The two substitutes that ARE
available were both tried and both are wrong for this purpose:

* prescribing the lateral displacement and holding it is K0 / oedometer compression, not a
  triaxial test.  Measured: it drives the mid-height axial stress to -434 MPa at 20 %
  shortening (against fcu* = 51 MPa) and leaves omega below 0.006, because a fully laterally
  restrained state is plastically COMPACTIVE and this model's damage is driven by plastic
  volumetric EXPANSION.  No band forms, by construction.
* a soft penalty as a stand-in for a pressure membrane does not work either: the penalty
  weak Dirichlet constraint acts on the INCREMENT (its force is k*(du - d*df_t)), so a small
  k gives an increment-size-dependent force rather than a constant one.

So this is a plane-strain COMPRESSION (biaxial) test with free lateral boundaries.  That is
the configuration in which this material was previously shown to localise a genuine inclined
shear band, and it is the harder case for the frame update, since nothing suppresses the
lateral expansion that the plastic flow needs.  Adding a real constant-pressure confinement
means implementing a boundary-surface-vector load for the point particle (the VCI machinery
already passes such vectors in ``vci_compute_Test_P_BoundaryIntegral``); that is a follow-up,
not a workaround.

Two ingredients are needed to get a band in the MIDDLE rather than at the platens:

* HARD CAPS -- the top and bottom two rows get all four strengths x3.  Implicit-gradient
  damage with natural (zero-flux) boundary conditions reads HIGH at a boundary, because the
  nonlocal average cannot be diluted by material outside the specimen, and with the
  over-nonlocal weight m = 1.05 that is amplified until damage nucleates at both loaded ends
  and saturates there.  A stronger seed does NOT fix it (going x0.98 -> x0.70 barely moves
  the mid/end damage contrast, which is what proves the end damage is a boundary artefact);
  hard caps do.
* A WEAK SEED -- one patch at mid-height on the left edge gets all four strengths x0.90, to
  pick out one band instead of letting damage smear over the whole core.

Usage
-----
    python ortho_finite_strain_triaxial_shear_band_test.py              # convected frame
    python ortho_finite_strain_triaxial_shear_band_test.py --frozen     # legacy frozen frame
    python ortho_finite_strain_triaxial_shear_band_test.py --compare    # both, overlaid
    python ortho_finite_strain_triaxial_shear_band_test.py --coarse     # h = 2.5 mm (quick)
Run under BASE python (/home/tom/miniforge3/bin/python).
"""

import argparse
import math
import os

import numpy as np

import edelweissfe.utils.performancetiming as performancetiming
import pytest
from edelweissfe.config.linsolve import getLinSolverByName
from edelweissfe.journal.journal import Journal
from edelweissfe.timesteppers.adaptivetimestepper import AdaptiveTimeStepper
from edelweissfe.utils.exceptions import StepFailed

from edelweissmeshfree.constraints.particlepenaltyweakdirichtlet import (
    ParticlePenaltyWeakDirichlet,
)
from edelweissmeshfree.fieldoutput.fieldoutput import MPMFieldOutputController
from edelweissmeshfree.generators.rectangularkernelfunctiongridgenerator import (
    generateRectangularKernelFunctionGrid,
)
from edelweissfe.surfaces.entitybasedsurface import EntityBasedSurface
from edelweissmeshfree.generators.rectangularparticlegridgenerator import (
    generateRectangularParticleGrid,
)
from edelweissmeshfree.generators.rectangularquadparticlegridgenerator import (
    generateRectangularQuadParticleGrid,
)
from edelweissmeshfree.stepactions.particledistributedload import (
    ParticleDistributedLoad,
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

# =============================================================================================
#  material card -- the calibrated Sect.-4 set (Mader et al., Acta Mechanica 2023)
# =============================================================================================


def saintVenantG(Ei, Ej, nuij):
    """Extended Saint Venant formula, Eq. (33) of the paper: 1/Gij = 1/Ei + 1/Ej + 2 nuij/Ej."""
    return 1.0 / (1.0 / Ei + 1.0 / Ej + 2.0 * nuij / Ej)


E1, E2, E3 = 2400.0, 2400.0, 1800.0
NU12, NU13, NU23 = 0.21, 0.24, 0.24
G12 = saintVenantG(E1, E2, NU12)
G13 = saintVenantG(E1, E3, NU13)
G23 = saintVenantG(E2, E3, NU23)

FCU = 51.03  # cast reference strength fcu*
FTU = FCU / 10.0
FCY = FCU / 3.0
FBU = 1.16 * FCU

AH, BH, CH, DH = 0.08, 0.003, 2.0, 1e-6
# Ductility divisor of the DAMAGE DRIVER: deltaAlphaLocal = dEPVol / xs, xs = 1 + As(4 sqrt(Rs)-3).
# The calibrated 15 makes xs ~ 16 in compression, so damage crawls and the run stalls at the
# plastic limit load with omega ~ 2e-4 while the implicit-gradient regularisation -- which acts on
# the damage -- is still doing nothing.  A material-point scan (test/compression_damage_scan.cpp)
# shows omega at 12 % shortening and 5 MPa confinement going 0.34 (As = 15) -> 0.78 (As = 0.5)
# with the PEAK UNCHANGED to 3 decimals, so this buys damage without touching the strength
# calibration.  As = 2 is the value that both damages and stays traversable.
AS = 2.0
DF = 0.85
# softMod is THE lever for whether the damage localises at all, and it has a NARROW usable
# window: on the earlier cylinder study 3.95e-3 gave a mid/end damage contrast of 1.43
# (diffuse), 1.0e-3 gave 12.11 (a sharp band), and 3.0e-4 snapped back at the peak with omega
# stuck at 0.17.  Here 1.0e-3 was tried and is NOT usable under plain displacement control on
# this specimen -- it snaps back within a few increments even on the coarse mesh -- so this runs
# at the calibrated 3.95e-3 and finds the band by contouring the most heterogeneous increment
# instead of the last one.  Getting 1.0e-3 to run needs arc-length or indirect control.
SOFTMOD = 3.95e-3
MAXDMG = 0.9999

# Walpole weights about the material frame, in the CODE's axis order (e1 = bedding normal).
# This is main.tex's 'bedded' set (1.0, 1.3, 1.0, 1.6, 1.0, 1.0), whose e2 is the normal,
# relabelled: the weak direction is the bedding NORMAL (alpha), and the weak shear is the one
# on planes containing the normal (zeta).  weight_i = fcu* / fcu^(i), so weight > 1 = weaker.
# Milder anisotropy than the paper's 'bedded' set: in the cylinder study alpha = 1.2 alone
# produced bands at EVERY orientation, while stronger weights made some orientations either
# unable to damage or unable to peak.
ALPHA, BETA, GAMMA = 1.20, 1.00, 1.00
ZETA, XI, ETA = 1.30, 1.00, 1.00

WEIGHT_M = 1.05  # over-nonlocal m > 1
L_NONLOCAL = 1.25  # nonlocal length l [mm]; FIXED by the material, not by the mesh

# Hardening level at which damage is allowed to start.  1.0 is the material's default and has a
# structural convergence wall: dqH/dalphaP vanishes quadratically at alphaP = 1, so the tangent
# is singular at the plastic limit load while omega is still exactly 0 -- the implicit-gradient
# regularisation is gated off precisely where it is needed.  Coarse meshes overshoot the gate;
# fine meshes resolve it and park there, which is why refinement made every run fail earlier and
# why arc-length control would not have helped.  0.95 is what the material README recommends.
DAMAGE_ONSET = 0.95

# Residual hardening slope of the yield surface over alphaP in [1,2].  This is the lever that
# actually makes the softening traversable: without it dqH/dalphaP is identically 0 beyond
# alphaP = 1, so the tangent is singular at the plastic limit load while omega is still 0 and
# the implicit-gradient regularisation is not yet doing anything.  The strength gain is bounded
# by the value itself.
H_RESIDUAL = 0.02

CAP_FACTOR = 3.00  # strengths of the platen caps
SEED_FACTOR = 0.90  # strengths of the seed patch

# =============================================================================================
#  specimen
# =============================================================================================

WIDTH = 10.0
HEIGHT = 20.0

# =============================================================================================
#  SECOND CASE: the Niandou Tournemire-shale triaxial tests, card from
#  Mader/Schreter/Hofstetter, IJNAMG 46 (2022) 933-960, Table 3.  Selected with --case niandou.
#  This is the RKPM half of the FE-vs-RKPM comparison; the FE half is
#  Marmot/modules/materials/GradientEnhancedOrthoCDPFiniteStrain/testCases/edelweissFE/
#  niandou_triaxial_fe.py, which solves the identical plane-strain problem with GCPE8RUL.
#  What is exact, what is derived and what cannot be transferred is documented there and in
#  test/niandou_triaxial_single_element.cpp.
# =============================================================================================
NIANDOU = dict(
    WIDTH=37.0, HEIGHT=75.0,          # the paper's specimen, 37 mm diameter x 75 mm high
    E1=7000.0, E2=18000.0, E3=18000.0,  # 1 = normal to the stratification planes
    NU12=0.2, NU13=0.2, NU23=0.25,
    G12=4000.0, G13=4000.0, G23=18000.0 / (2.0 * 1.25),
    FCU=42.54, FCY=22.25, FTU=9.1608, FBU=43.8059,  # ftu/fbu derived: e = 0.51, m0* = 4.487
    DF=0.85, AH=0.022, BH=0.01, CH=1.0, DH=1e-6, AS=15.0,
    SOFTMOD=4.75e-4,                  # eps_f*
    L_NONLOCAL=5.0, WEIGHT_M=1.05,
    ALPHA=1.0, BETA=1.0, GAMMA=1.0, ZETA=1.0, XI=1.0, ETA=1.0,
    BEDDING_PHI_DEG=45.0,             # beta: inclination of the stratification planes
    AXIAL_STRAIN=0.04,                # 3 mm of 75 mm
)


def applyCase(name):
    """Switch the module-level card and geometry to a named case."""
    if name != "niandou":
        return
    g = globals()
    for k, v in NIANDOU.items():
        g[k] = v
BEDDING_PHI_DEG = 45.0  # angle of the bedding NORMAL from the x axis, in the x-y plane
AXIAL_STRAIN = 0.12  # nominal shortening, compression mode
SHEAR_STRAIN = 0.30  # nominal shear angle gamma = u_x(top)/H, shear mode


OVERRIDES = {}  # set from the CLI: softMod, maxDmg, l, m


def materialProperties(strengthFactor, frameUpdate):
    """The 33-property card.  ``strengthFactor`` scales the four strengths (caps / seed)."""
    phi = math.radians(BEDDING_PHI_DEG)
    return np.array(
        [
            E1, E2, E3,
            NU12, NU13, NU23,
            G12, G13, G23,
            math.cos(phi), math.sin(phi), 0.0,          # bedding normal n0
            FCY * strengthFactor, FCU * strengthFactor,
            FBU * strengthFactor, FTU * strengthFactor,
            DF,
            AH, BH, CH, DH, OVERRIDES.get("As", AS),
            OVERRIDES.get("softMod", SOFTMOD), OVERRIDES.get("maxDmg", MAXDMG),
            *OVERRIDES.get("weights", (ALPHA, BETA, GAMMA, ZETA, XI, ETA)),
            OVERRIDES.get("l", L_NONLOCAL), OVERRIDES.get("m", WEIGHT_M),
            float(frameUpdate),                          # 1 = convected frame, 0 = frozen
            OVERRIDES.get("damageOnset", DAMAGE_ONSET),   # alphaP at which damage may start
            OVERRIDES.get("hres", H_RESIDUAL),            # residual hardening slope over [1,2]
        ]
    )


def lNonlocal():
    return OVERRIDES.get("l", L_NONLOCAL)


def capDepth():
    return 2.0 * lNonlocal()   # depth of the hard platen caps [mm]


def seedSize():
    return 1.0 * lNonlocal()   # half-height / width of the weak seed patch [mm]


SLAB_ANGLE_DEG = 45.0   # inclination of the weak slab to the load axis
SLAB_WIDTH = 2.0        # width of the weak slab in units of the nonlocal length l
SLAB_FACTOR = 0.85      # strengths inside the slab


def strengthFactorAt(x, y, seed="patch"):
    """Hard caps at both platens plus a weak seed; both sized against the NONLOCAL LENGTH.

    Two seed shapes:

    "patch" -- one small weak square at mid-height on the left edge.  This lets the band choose
        its own path, which sounds better and is worse: the band only forms once the global
        plastic limit load is reached, and at that point (in compression) omega is still ~2e-4,
        the hardening modulus has vanished and the run stops.  Every mesh with h <= l dies there.

    "slab" (default) -- a weak slab of FIXED PHYSICAL WIDTH (2 l) inclined at SLAB_ANGLE_DEG,
        crossing the specimen.  This is the recipe that produced the 3D shear band of the old
        example 147.  It pre-localises the band geometrically, so the slab yields and starts to
        dilate -- and therefore to damage -- BEFORE the specimen reaches its global limit load.
        The softening is then carried by the regularised damage rather than by unregularised
        perfect plasticity, and the band width is set by l, not by the slab, which is what makes
        the mesh study mean something.
    """
    if y < capDepth() or y > HEIGHT - capDepth():
        return CAP_FACTOR
    if seed == "slab":
        th = math.radians(SLAB_ANGLE_DEG)
        # signed distance from the slab's mid-line through the specimen centre
        d = ( ( x - 0.5 * WIDTH ) * math.cos( th ) - ( y - 0.5 * HEIGHT ) * math.sin( th ) )
        if abs( d ) <= 0.5 * SLAB_WIDTH * lNonlocal():
            return SLAB_FACTOR
        return 1.0
    if abs(y - 0.5 * HEIGHT) <= seedSize() and x <= 2.0 * seedSize():
        return SEED_FACTOR
    return 1.0


# =============================================================================================
#  the simulation
# =============================================================================================


def run_sim(frameUpdate=1, coarse=False, ensightName=None, spacing=None,
            particleType="sqcnixnsni", confiningPressure=2.0, mode="compression", seed="patch"):
    np.set_printoptions(linewidth=200, precision=3)

    dimension = 2
    journal = Journal()
    theModel = MPMModel(dimension)

    h = spacing if spacing else (2.5 if coarse else lNonlocal())
    nX = int(round(WIDTH / h))
    nY = int(round(HEIGHT / h))
    supportRadius = 2.0 * h  # UNIFORM; local scaling is what OOM-killed the hexa studies

    quad = particleType == "sqcnixnsni"
    pName = ( "GradientEnhancedFiniteStrainSQCNIxNSNI/PlaneStrain/Quad" if quad
              else "GradientEnhancedFiniteStrain/PlaneStrain/Point" )

    journal.message(
        f"specimen {WIDTH} x {HEIGHT} mm, h = {h} mm, {nX} x {nY} cells, "
        f"H/l = {HEIGHT / lNonlocal():.1f}, particle = {pName}, frameUpdate = {frameUpdate}, "
        f"softMod = {OVERRIDES.get('softMod', SOFTMOD):g}, "
        f"damageOnset = {OVERRIDES.get('damageOnset', DAMAGE_ONSET):g}, "
        f"Hres = {OVERRIDES.get('hres', H_RESIDUAL):g}, As = {OVERRIDES.get('As', AS):g}, "
        f"confining pressure = {confiningPressure} MPa",
        "setup",
    )

    # The kernel grid is point-INCLUSIVE (nX points, nX-1 gaps), the quad particle grid is
    # cell-based (nX cells).  Using the same nX for both, as the existing SQCNI examples do,
    # leaves the kernel spacing a factor nX/(nX-1) above the cell size, which the 2h support
    # covers comfortably.
    theModel = generateRectangularKernelFunctionGrid(
        theModel,
        journal,
        lambda node: MarmotMeshfreeKernelFunctionWrapper(
            node, "BSplineBoxed", supportRadius=supportRadius, continuityOrder=2
        ),
        x0=0.0, y0=0.0, h=HEIGHT, l=WIDTH, nX=nX, nY=nY,
    )

    theApproximation = MarmotMeshfreeApproximationWrapper(
        "ReproducingKernelImplicitGradient", dimension, completenessOrder=1
    )

    # one card per zone, built once; the particle factory picks by particle centre
    cards = {
        f: {
            "material": "GRADIENTENHANCEDORTHOCDPFINITESTRAIN",
            "properties": materialProperties(f, frameUpdate),
        }
        for f in (1.0, CAP_FACTOR, SEED_FACTOR, SLAB_FACTOR)
    }

    def theParticleFactory(number, coordinates, volume):
        c = np.asarray(coordinates).reshape(-1, 2).mean(axis=0)  # centre, from centre or vertices
        return MarmotParticleWrapper(
            pName, number, coordinates, volume, theApproximation,
            cards[strengthFactorAt(c[0], c[1], seed)],
        )

    generator = generateRectangularQuadParticleGrid if quad else generateRectangularParticleGrid
    theModel = generator(
        theModel, journal, theParticleFactory, x0=0.0, y0=0.0, h=HEIGHT, l=WIDTH, nX=nX, nY=nY
    )

    theParticleKernelDomain = ParticleKernelDomain(
        list(theModel.particles.values()), list(theModel.meshfreeKernelFunctions.values())
    )
    theParticleManager = KDBinOrganizedParticleManager(
        theParticleKernelDomain, dimension, journal, bondParticlesToKernelFunctions=True
    )
    theModel.particleKernelDomains["all_with_all"] = theParticleKernelDomain

    theModel.prepareYourself(journal)
    journal.printPrettyTable(theModel.makePrettyTableSummary(), "summary")

    fieldOutputController = MPMFieldOutputController(theModel, journal)
    for name in ("displacement", "stress", "omega", "alphaP", "frameRotation",
                 "materialAxis1", "materialAxis2"):
        fieldOutputController.addPerParticleFieldOutput(name, theModel.particleSets["all"], name)
    if quad:
        fieldOutputController.addPerParticleFieldOutput(
            "vertex displacements",
            theModel.particleSets["all"],
            "vertex displacements",
            f_x=lambda x: np.pad(np.reshape(x, (-1, 2)), ((0, 0), (0, 1)), mode="constant",
                                 constant_values=0),
        )
    fieldOutputController.initializeJob()

    outputManagers = []
    if ensightName:
        ensightOutput = EnsightOutputManager(ensightName, theModel, fieldOutputController, journal, None)
        # the particle CENTRE displacement is one value per particle; a quad particle's Ensight
        # part has 4 vertices, so perNode would abort with "Variable displacement result size
        # (128) does not match the number of nodes (512)".  perElement for quads, and a proper
        # per-vertex field alongside it.
        ensightOutput.updateDefinition(
            fieldOutput=fieldOutputController.fieldOutputs["displacement"],
            create="perElement" if quad else "perNode",
        )
        if quad:
            ensightOutput.updateDefinition(
                fieldOutput=fieldOutputController.fieldOutputs["vertex displacements"], create="perNode"
            )
        for name in ("omega", "alphaP", "frameRotation", "materialAxis1", "stress"):
            ensightOutput.updateDefinition(
                fieldOutput=fieldOutputController.fieldOutputs[name], create="perElement"
            )
        ensightOutput.initializeJob()
        outputManagers.append(ensightOutput)

    PEN = 1e8
    sets = theModel.particleSets

    def bc(name, particles, values, **kw):
        return ParticlePenaltyWeakDirichlet(name, theModel, particles, "displacement", values, PEN, **kw)

    # one single particle carries u_x = 0: enough to kill the x translation without turning the
    # platen into a rough (confining) one, which smears the damage instead of localising it
    def centreOf(p):
        return np.asarray(p.getVertexCoordinates()).reshape(-1, 2).mean(axis=0)

    bottomParticles = list(sets["rectangular_grid_bottom"])
    xyBottom = np.array([centreOf(p) for p in bottomParticles])
    anchor = [bottomParticles[int(np.argmin(np.abs(xyBottom[:, 0] - 0.5 * WIDTH)))]]

    # Solver leash.  The wall is Newton divergence ("residual grew 3 times, cutting back") at
    # the PLASTIC limit point -- alphaP ~ 1.2 with omega still ~2e-4, so it is not the damage
    # softening at all.  A short leash (3 allowed growths, 15 iterations, no line search) gives
    # up on a step that a longer one walks through.  The earlier finding that a line search
    # "made it worse" was confounded: that test also carried a 1e-8 minimum increment, and the
    # crawling came from the minimum increment, not from the line search.
    iterationOptions = {
        "max. iterations": 40,
        "critical iterations": 10,
        "allowed residual growths": 15,
        "line search": True,
        "line search after n iterations": 6,
        "line search every n iterations": 2,
        "line search alphas": [0.25, 0.5, 0.75, 1.0],
    }
    linearSolver = getLinSolverByName("pardiso", {})
    nonlinearSolver = NonlinearQuasistaticSolver(journal)

    history = []  # (nominal axial shortening, mean tau_yy over the mid-height slice, omega, R^p)
    xyAll = np.array([centreOf(p) for p in sets["all"]])
    # the undeformed smoothing-domain corners, captured before anything moves.  For a quad
    # particle getVertexCoordinates() returns the 4 corners of the smoothing domain; for the
    # point particle it returns the single material point, and there is no domain to draw.
    verts0 = (
        np.array([np.asarray(p.getVertexCoordinates()).reshape(-1, 2) for p in sets["all"]])
        if quad else None
    )
    midSlice = np.abs(xyAll[:, 1] - 0.5 * HEIGHT) <= 1.01 * h

    snapshots = []

    uRef = {"u0": None}

    def recordHistory():
        fo = fieldOutputController.fieldOutputs
        u = fo["displacement"].getLastResult()[:, :2]
        # the confinement step already moves the specimen, so the axial shortening is counted
        # from the state at the END of it, not from the undeformed configuration
        if uRef["u0"] is None:
            uRef["u0"] = u.copy()
        du = u - uRef["u0"]
        tau = fo["stress"].getLastResult().reshape(-1, 3, 3)
        omega = fo["omega"].getLastResult().reshape(-1).copy()
        history.append(
            (
                ( -du[:, 1].min() if mode == "compression" else du[:, 0].max() ) / HEIGHT,
                float(tau[midSlice, 1, 1].mean()),
                float(omega.max()),
                float(fo["frameRotation"].getLastResult().max()),
                float(tau[:, 0, 0].mean()),
            )
        )
        # The band is a TRANSIENT: run far enough and omega saturates over almost the whole
        # specimen, so the LAST increment shows a uniformly destroyed body and no band at all.
        # Keep every increment and let the post-processing pick the most heterogeneous one.
        snapshots.append(
            dict(
                shortening=history[-1][0],
                u=u.copy(),
                omega=omega,
                alphaP=fo["alphaP"].getLastResult().reshape(-1).copy(),
                frameRotation=fo["frameRotation"].getLastResult().reshape(-1).copy(),
                axis1=fo["materialAxis1"].getLastResult().reshape(-1, 3).copy(),
                axis2=fo["materialAxis2"].getLastResult().reshape(-1, 3).copy(),
                # the smoothing domains as they actually are at this increment
                verts=( verts0
                        + fo["vertex displacements"].getLastResult().reshape(-1, 4, 3)[:, :, :2]
                        if quad else None ),
                heterogeneity=float(omega.std()),
            )
        )

    # A GENUINE TRIAXIAL CONFINEMENT is possible with the SQCNI family and only with it: the
    # smoothing domain gives the particle faces, so a surface load can reach it.  Quad face ids
    # for the generator's CCW vertex order (x,y),(x+1,y),(x+1,y+1),(x,y+1) are
    # 1 = bottom, 2 = right, 3 = top, 4 = left.  A NEGATIVE load is compressive (the load is
    # applied along the OUTWARD surface vector), which is checked below by measuring tau_xx.
    distributedLoads = []
    if confiningPressure > 0.0:
        if not quad:
            raise ValueError(
                "a constant-pressure confinement needs a particle with faces; the Point "
                "particle has none.  Use --particle sqcnixnsni."
            )
        theModel.surfaces["confinement"] = EntityBasedSurface(
            "confinement",
            {4: list(sets["rectangular_grid_left"]), 2: list(sets["rectangular_grid_right"])},
        )
        distributedLoads.append(
            ParticleDistributedLoad(
                "confinement", theModel, journal, theModel.surfaces["confinement"],
                "pressure", np.array([-confiningPressure]),
            )
        )

    # ---- step 1: build the confinement up, with the axial faces held ------------------------
    # ParticleDistributedLoad ramps its load over a step and then goes IDLE, holding the full
    # value for every later step it is passed to.  So the SAME object in two steps gives exactly
    # ramp-then-hold, which is what a triaxial cell does: consolidate, then shear.  Applying it
    # in one step instead makes the confinement grow together with the axial load -- measured
    # -0.19 MPa of an intended -5.0 at the point the run stopped, i.e. essentially unconfined.
    if distributedLoads:
        journal.message(f"STEP 1 -- building up {confiningPressure} MPa of confinement", "step")
        try:
            nonlinearSolver.solveStep(
                AdaptiveTimeStepper(theModel.time, 1.0, 0.2, 0.5, 1e-3, 100, journal),
                linearSolver, theModel, fieldOutputController,
                outputManagers=outputManagers,
                particleManagers=[theParticleManager],
                constraints=[
                    bc("botY", bottomParticles, {1: 0.0}),
                    bc("topY", list(sets["rectangular_grid_top"]), {1: 0.0}),
                    bc("anchorX", anchor, {0: 0.0}),
                ],
                particleDistributedLoads=distributedLoads,
                userIterationOptions=iterationOptions,
            )
        except StepFailed as e:
            journal.message(f"confinement step failed: {e}", "error")
            raise
        # The meshfree NQS solver does NOT call applyAtStepEnd on distributed loads (only the
        # arc-length solver in EdelweissFE does), so the load object never latches the value it
        # ramped to and step 2 would ramp it from ZERO all over again.  Measured before this
        # call: tau_xx back to -0.000 at the first increment of step 2, then linear in the step
        # progress, reaching only -2.0 of the intended -5.0 by the end.  Latch it by hand.
        for dl in distributedLoads:
            dl.applyAtStepEnd(theModel)
        tau = fieldOutputController.fieldOutputs["stress"].getLastResult().reshape(-1, 3, 3)
        journal.message(
            f"confinement in place and latched: mean tau_xx = {tau[:, 0, 0].mean():.3f} MPa "
            f"(target {-confiningPressure:.3f})",
            "step",
        )

    # ---- step 2: compress axially, confinement held ----------------------------------------
    # ---- step 2: drive the band ------------------------------------------------------------
    # TWO LOADING MODES, and the choice matters more than any material parameter:
    #
    #  "compression" -- axial shortening of a slender specimen.  This is the configuration of
    #      the earlier sessions, and on a mesh with h <= l it does NOT work: the run terminates
    #      at the PLASTIC limit load with omega still ~2e-4, because in compression this model's
    #      damage is driven by plastic volumetric EXPANSION and grows far too slowly to take
    #      over the softening.  The tangent is singular there (dqH/dalphaP vanishes
    #      quadratically at alphaP = 1) and the implicit-gradient regularisation, which acts on
    #      the DAMAGE, is not yet doing anything.  Neither a residual hardening slope, nor
    #      moving the damage onset, nor a longer solver leash, nor dropping the confinement
    #      gets a full curve out of it -- each buys a little more strain (2.09 % -> 3.6 % at
    #      best) and then stalls.  The material README says as much: "use 3D with free lateral
    #      faces, not plane strain".
    #
    #  "shear" (the default) -- SIMPLE SHEAR of the same panel.  Two reasons it behaves:
    #      almost no axial elastic energy is stored, so there is nothing to drive a snap-back;
    #      and shearing across the bedding is dilatant from the start, so damage engages early
    #      and the regularised damage softening -- not unregularised perfect plasticity -- is
    #      what governs the localisation.  That is also the state the material-point check
    #      test/frame_update_check.cpp exercises, where the frame turns 16.9 deg at gamma = 0.6.
    uMax = ( AXIAL_STRAIN * HEIGHT if mode == "compression" else SHEAR_STRAIN * HEIGHT )
    if mode == "compression":
        journal.message(f"STEP 2 -- compressing to {AXIAL_STRAIN * 100:.0f} % shortening", "step")
        axialBCs = [
            bc("botY", bottomParticles, {1: 0.0}),
            bc("topY", list(sets["rectangular_grid_top"]), {1: -uMax}),
            bc("anchorX", anchor, {0: 0.0}),
        ]
    else:
        journal.message(f"STEP 2 -- shearing to gamma = {SHEAR_STRAIN:.2f}", "step")
        axialBCs = [
            bc("bot", bottomParticles, {0: 0.0, 1: 0.0}),
            bc("top", list(sets["rectangular_grid_top"]), {0: uMax, 1: 0.0}),
        ]
    inc = 0.005
    stepFailed = False

    class _Recorder:
        """An output manager whose only job is to sample the load-displacement history."""

        def initializeJob(self):
            pass

        def initializeStep(self, *a, **kw):
            pass

        def finalizeIncrement(self, *a, **kw):
            recordHistory()

        def finalizeFailedIncrement(self, *a, **kw):
            pass

        def finalizeStep(self, *a, **kw):
            pass

        def finalizeJob(self):
            pass

    try:
        recordHistory()  # establishes the zero of the axial shortening after the confinement
    except Exception:
        pass
    try:
        nonlinearSolver.solveStep(
            AdaptiveTimeStepper(theModel.time, 1.0, inc, 4.0 * inc, 1e-5, 600, journal),
            linearSolver, theModel, fieldOutputController,
            outputManagers=outputManagers + [_Recorder()],
            particleManagers=[theParticleManager],
            constraints=axialBCs,
            particleDistributedLoads=distributedLoads,
            userIterationOptions=iterationOptions,
        )
    except StepFailed as e:
        # a limit point or a return-map failure deep in the softening branch is a RESULT here,
        # not a crash: the band has formed by then.  Report it and keep the state.
        journal.message(f"step stopped early: {e}", "warning")
        stepFailed = True
    finally:
        recordHistory()
        fieldOutputController.finalizeJob()
        for om in outputManagers:
            om.finalizeJob()
        prettytable = performancetiming.makePrettyTable()
        prettytable.min_table_width = journal.linewidth
        journal.printPrettyTable(prettytable, "Summary")

    return dict(
        model=theModel,
        snapshots=snapshots,
        xy0=xyAll,
        verts0=verts0,
        fieldOutputController=fieldOutputController,
        history=np.array(history),
        h=h,
        mode=mode,
        seed=seed,
        particleType=pName,
        confiningPressure=confiningPressure,
        stepFailed=stepFailed,
        frameUpdate=frameUpdate,
    )


# =============================================================================================
#  post-processing
# =============================================================================================


def pickSnapshot(result, which="best"):
    """Return one recorded increment.

    ``which = "best"`` picks the increment of MAXIMUM SPATIAL HETEROGENEITY of omega, which is
    where the band is sharpest.  This is not cosmetic: the band is a transient, and by the end
    of a far-enough run omega has saturated over almost the whole specimen and there is no band
    left to see.  ``which = "last"`` returns the final state.
    """
    snaps = [sn for sn in result["snapshots"] if sn["omega"].max() > 0.0]
    if not snaps:
        snaps = result["snapshots"]
    if which == "last":
        sn = snaps[-1]
    elif isinstance(which, (int, float)):
        # matched shortening: the two frames do not reach the same strain, so any comparison
        # of band sharpness has to be made at the same point on the load path
        sn = min(snaps, key=lambda t: abs(t["shortening"] - float(which)))
    else:
        sn = max(snaps, key=lambda t: t["heterogeneity"])
    f = dict(sn)
    f["xy0"] = result["xy0"]
    f["xy"] = result["xy0"] + sn["u"]
    return f


def report(result, which="best"):
    f = pickSnapshot(result, which)
    whichLabel = which if isinstance(which, str) else f"{float(which) * 100:.2f} %"
    y = f["xy0"][:, 1]
    core = (y > capDepth()) & (y < HEIGHT - capDepth())
    caps = ~core
    hist = result["history"]
    print("\n" + "=" * 78)
    print(f" PLANE-STRAIN {'SIMPLE SHEAR' if result['mode'] == 'shear' else 'COMPRESSION'},"
          f" frameUpdate = {result['frameUpdate']}"
          f" ({'convected' if result['frameUpdate'] else 'frozen'} material frame)")
    print(f"   {result['particleType']},  h = {result['h']} mm,  "
          f"confining pressure {result['confiningPressure']} MPa")
    print("=" * 78)
    print(f"  increments recorded / reached shortening    : {len(result['snapshots']):5d}"
          f" / {hist[:, 0].max() * 100:.2f} %"
          f"{'   (stopped early)' if result['stepFailed'] else ''}")
    print(f"  contoured increment ({whichLabel:>8s})              : shortening"
          f" {f['shortening'] * 100:.2f} %, omega std {f['heterogeneity']:.4f}")
    print(f"  peak mean tau_yy at mid-height              : {hist[:, 1].min():8.3f} MPa")
    print(f"  control measure reached                     : {hist[:, 0].max() * 100:8.3f} %"
          f"   ({'shortening' if result['mode'] == 'compression' else 'shear angle gamma'})")
    print(f"  mean tau_xx over the specimen (confinement) : {hist[1:, 4].mean():8.3f} MPa"
          f"   (target {-result['confiningPressure']:.3f})")
    print(f"  max omega  (core / caps)                    : {f['omega'][core].max():8.4f}"
          f" / {f['omega'][caps].max():.4f}")
    print(f"  max alphaP (core)                           : {f['alphaP'][core].max():8.3f}")
    print(f"  MATERIAL FRAME ROTATION R^p, max over core  : {f['frameRotation'][core].max():8.3f} deg")
    dmg = f["omega"] > 0.5 * max(f["omega"].max(), 1e-12)
    if dmg.any() and f["omega"].max() > 1e-6:
        print(f"  frame rotation inside the band (omega > 50 % of max):"
              f" mean {f['frameRotation'][dmg].mean():.3f} deg,"
              f" max {f['frameRotation'][dmg].max():.3f} deg")
        print(f"  band occupies {dmg.sum()} / {len(dmg)} particles")
        pts = f["xy0"][dmg] - f["xy0"][dmg].mean(axis=0)
        if len(pts) > 2:
            _, _, vt = np.linalg.svd(pts, full_matrices=False)
            ang = math.degrees(math.atan2(abs(vt[0, 1]), abs(vt[0, 0])))
            print(f"  band inclination to the load axis           : {90.0 - ang:8.1f} deg")
    print(f"  mid/end damage contrast                     : "
          f"{f['omega'][core].max() / max(f['omega'][caps].max(), 1e-12):8.2f}"
          f"   (> 1 means the band is in the middle)")
    return f


def makePlots(results, which="best", fname="contour_plots.png"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    def drawField(ax, f, values, cmap, alpha=1.0):
        """Draw the field on the SMOOTHING DOMAINS where they exist, else on markers.

        The smoothing domain is the quantity the SQCNI gradients are integrated over, so it is
        the honest picture of the discretisation: it shows how the band is resolved and how the
        domains themselves distort, which a scatter of markers hides.
        """
        if f.get("verts") is not None:
            pc = PolyCollection(f["verts"], array=np.asarray(values), cmap=cmap,
                                edgecolors="0.35", linewidths=0.25, alpha=alpha)
            ax.add_collection(pc)
            ax.autoscale_view()
            allV = f["verts"].reshape(-1, 2)
            ax.set_xlim(allV[:, 0].min() - 0.3, allV[:, 0].max() + 0.3)
            ax.set_ylim(allV[:, 1].min() - 0.3, allV[:, 1].max() + 0.3)
            return pc
        size = max(4.0, 1400.0 / np.sqrt(len(values)))
        return ax.scatter(f["xy"][:, 0], f["xy"][:, 1], c=values, s=size, cmap=cmap,
                          marker="s", linewidths=0, alpha=alpha)

    n = len(results)
    fig, axes = plt.subplots(n, 3, figsize=(13.5, 6.4 * n), squeeze=False)

    for row, result in enumerate(results):
        f = pickSnapshot(result, which)
        label = ("convected frame" if result["frameUpdate"] else "frozen frame (legacy)") + \
            f", {f['shortening'] * 100:.1f} % shortening"

        for col, (field, title, cmap) in enumerate(
            (
                ("omega", r"damage $\omega$", "inferno"),
                ("frameRotation", r"material frame rotation $R^p$ [deg]", "viridis"),
            )
        ):
            ax = axes[row][col]
            sc = drawField(ax, f, f[field], cmap)
            plt.colorbar(sc, ax=ax, shrink=0.7)
            ax.set_title(f"{title}\n{label}", fontsize=9)
            ax.set_aspect("equal")
            ax.set_xlabel("x [mm]")
            ax.set_ylabel("y [mm]")

        # the material axes themselves, on the deformed configuration, over the damage field
        ax = axes[row][2]
        sc = drawField(ax, f, f["omega"], "inferno", alpha=0.5)
        plt.colorbar(sc, ax=ax, shrink=0.7)
        # scale the axis glyphs to the particle size, not to the nonlocal length, so they stay
        # readable when the mesh is refined
        span = 0.45 * result["h"]
        for sign in (+1.0, -1.0):
            # e2 spans the bedding PLANE in the x-y plane, so drawing it draws the bedding trace
            ax.quiver(f["xy"][:, 0], f["xy"][:, 1], sign * f["axis2"][:, 0], sign * f["axis2"][:, 1],
                      color="tab:cyan", angles="xy", scale_units="xy", scale=1.0 / span,
                      width=0.005, headwidth=0, headlength=0, headaxislength=0, pivot="tail")
        ax.quiver(f["xy"][:, 0], f["xy"][:, 1], f["axis1"][:, 0], f["axis1"][:, 1],
                  color="w", angles="xy", scale_units="xy", scale=1.7 / span,
                  width=0.006, headwidth=3.5, pivot="tail")
        ax.set_title(f"bedding trace $e^{{(2)}}$ (cyan), normal $e^{{(1)}}$ (white)\n{label}",
                     fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")

    fig.tight_layout()
    fig.savefig(fname, dpi=140)
    print(f"  wrote {fname}")

    # load-displacement and the frame rotation over the load path
    fig2, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for result in results:
        hist = result["history"]
        lab = "convected frame" if result["frameUpdate"] else "frozen frame (legacy)"
        ax.plot(hist[:, 0] * 100, -hist[:, 1], "-", lw=1.4, label=lab)
        ax2.plot(hist[:, 0] * 100, hist[:, 3], "-", lw=1.4, label=lab)
    ax.set_xlabel("nominal axial shortening [%]")
    ax.set_ylabel(r"$-\tau_{yy}$ at mid-height [MPa]")
    ax.set_title(f"bedding normal at {BEDDING_PHI_DEG:.0f}$^\\circ$ to $x$", fontsize=10)
    ax2.set_xlabel("nominal axial shortening [%]")
    ax2.set_ylabel(r"max material frame rotation $R^p$ [deg]")
    ax2.set_title("how far the bedding has turned", fontsize=10)
    for a in (ax, ax2):
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig2.tight_layout()
    ldName = fname.replace("contour_plots", "load_displacement")
    fig2.savefig(ldName, dpi=140)
    print(f"  wrote {ldName}")


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    """No matter where pytest is ran, we set the working dir to this testscript's parent."""
    monkeypatch.chdir(request.fspath.dirname)


def test_sim():
    """A physics test rather than a gold test.

    A gold file would be brittle here: the run is deliberately pushed into the softening
    branch, where the reachable shortening depends on how the adaptive stepper happens to cut
    back.  What must hold regardless is asserted directly -- the frame stays a rotation, the
    legacy switch really freezes it, the convected frame really turns, and the two frames give
    different mechanics.
    """
    import matplotlib

    matplotlib.use("Agg")
    import warnings

    warnings.filterwarnings("ignore")

    frozen = run_sim(frameUpdate=0, coarse=True)
    convected = run_sim(frameUpdate=1, coarse=True)

    fF = pickSnapshot(frozen, "last")
    fC = pickSnapshot(convected, "last")

    # the legacy switch must freeze the frame exactly
    assert np.abs(fF["frameRotation"]).max() < 1e-12

    # the convected frame must turn, and by a lot: this specimen shears across its bedding
    assert fC["frameRotation"].max() > 1.0

    # it must remain an orthonormal right-handed triad at every particle
    Q = np.stack([fC["axis1"], fC["axis2"], np.cross(fC["axis1"], fC["axis2"])], axis=2)
    gram = np.einsum("nij,nkj->nik", Q.transpose(0, 2, 1), Q.transpose(0, 2, 1))
    assert np.abs(gram - np.eye(3)).max() < 1e-10
    assert np.abs(np.linalg.det(Q) - 1.0).max() < 1e-10

    # and it must reach the mechanics, not just the output
    assert abs(convected["history"][-1, 1] - frozen["history"][-1, 1]) > 1e-6


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", action="store_true", help="legacy frozen material frame")
    parser.add_argument("--compare", action="store_true", help="run both frames and overlay")
    parser.add_argument("--coarse", action="store_true", help="h = 2.5 mm instead of 1.25 mm")
    parser.add_argument("--h", type=float, default=None, help="particle spacing [mm], overrides --coarse")
    parser.add_argument("--particle", choices=("sqcnixnsni", "point"), default="sqcnixnsni",
                        help="sqcnixnsni = stabilized nodal integration on quad smoothing "
                             "domains (default); point = plain unstabilized nodal integration")
    parser.add_argument("--confine", type=float, default=2.0,
                        help="constant confining pressure [MPa] on the lateral faces "
                             "(needs --particle sqcnixnsni); 0 = free lateral boundaries")
    parser.add_argument("--no-ensight", action="store_true")
    parser.add_argument("--softmod", type=float, default=None,
                        help="softeningModulus; LARGER = slower damage growth = more ductile = "
                             "traversable softening under displacement control")
    parser.add_argument("--maxdmg", type=float, default=None,
                        help="maxDamage cap; < 1 leaves the band residual stiffness")
    parser.add_argument("--As", dest="As", type=float, default=None,
                        help="ductility divisor As of the damage driver: xs = 1 + As(4 sqrt(Rs)-3). "
                             "The calibrated 15 makes xs ~ 16 in compression and damage crawl; "
                             "0.5-2 makes compression damage properly and leaves the peak intact")
    parser.add_argument("--lnl", type=float, default=None, help="nonlocal length l [mm]")
    parser.add_argument("--seed", choices=("slab", "patch"), default="patch",
                        help="patch = one small weak square at mid-height on the left edge "
                             "(default; with As = 2 this localises cleanly); slab = a "
                             "fixed-width inclined weak slab, which does not help here")
    parser.add_argument("--weights", default=None,
                        help="six Walpole/Kelvin weights 'alpha,beta,gamma,zeta,xi,eta'.  NOTE: at "
                             "1,1,1,1,1,1 the mapping tensor is the IDENTITY for every frame "
                             "(verified to 3.6e-16), so the yield surface cannot see the material "
                             "frame at all and the frame update only reaches the elasticity.")
    parser.add_argument("--beta", type=float, default=None,
                        help="inclination of the stratification PLANES to the horizontal [deg], "
                             "the paper's beta.  beta = 0 puts the bedding normal along the load "
                             "axis, beta = 90 perpendicular to it.  Internally the card carries "
                             "the NORMAL angle from x, so phi = 90 - beta.")
    parser.add_argument("--case", choices=("ortho", "niandou"), default="ortho",
                        help="ortho = the 3D-printed-concrete card and the 10x20 mm specimen "
                             "(default); niandou = the Tournemire-shale card of "
                             "Mader/Schreter/Hofstetter (2022) Table 3 and their 37x75 mm "
                             "specimen, for comparison against EdelweissFE and the paper")
    parser.add_argument("--mode", choices=("compression", "shear"), default="compression",
                        help="compression = axial shortening (default; gives a LOCALISED "
                             "inclined shear band and a full curve at h >= l with As = 2); "
                             "shear = simple shear (traverses at every mesh but the damage is "
                             "diffuse, since simple shear of a panel is a homogeneous state)")
    parser.add_argument("--hres", type=float, default=None,
                        help="residual hardening slope over alphaP in [1,2] (0 = the original "
                             "law, which has a singular tangent at the plastic limit load)")
    parser.add_argument("--onset", type=float, default=None,
                        help="alphaP at which damage may start (1.0 = the material default, "
                             "which has a singular tangent at the plastic limit load)")
    parser.add_argument("--m", type=float, default=None, help="over-nonlocal weighting m")
    parser.add_argument("--strain", type=float, default=None, help="nominal shortening target")
    parser.add_argument("--tag", default="", help="suffix for the output file names, so several "
                                                 "mesh sizes can be run side by side")
    args = parser.parse_args()

    applyCase(args.case)
    if args.weights is not None:
        OVERRIDES["weights"] = tuple(float(v) for v in args.weights.split(","))
    if args.beta is not None:
        # the card stores the angle of the bedding NORMAL from x; the paper's beta is the
        # inclination of the PLANES from the horizontal, and the load axis is y, so the two are
        # complementary.  Keeping the CLI in the paper's beta means the FE and RKPM drivers take
        # the same number.
        globals()["BEDDING_PHI_DEG"] = 90.0 - args.beta

    for key, val in (("softMod", args.softmod), ("maxDmg", args.maxdmg),
                     ("l", args.lnl), ("m", args.m), ("damageOnset", args.onset),
                     ("hres", args.hres), ("As", args.As)):
        if val is not None:
            OVERRIDES[key] = val
    if args.strain is not None:
        globals()["AXIAL_STRAIN"] = args.strain
        globals()["SHEAR_STRAIN"] = args.strain

    modes = [0, 1] if args.compare else [0 if args.frozen else 1]
    results = []
    for m in modes:
        name = None if args.no_ensight else f"_ensight_frame{m}"
        results.append(run_sim(frameUpdate=m, coarse=args.coarse, ensightName=name, spacing=args.h,
                               particleType=args.particle, confiningPressure=args.confine,
                               mode=args.mode, seed=args.seed))
        report(results[-1])

    tag = f"_{args.tag}" if args.tag else ""
    makePlots(results, fname=f"contour_plots{tag}.png")

    print("\n  --- final increment, for comparison ---")
    for r in results:
        report(r, which="last")
    makePlots(results, which="last", fname=f"contour_plots_last{tag}.png")

    if len(results) > 1:
        # The two frames do NOT reach the same shortening -- the convected one localises
        # earlier and therefore stalls earlier -- so the honest comparison of band sharpness
        # is at the largest shortening BOTH of them reached.
        target = min(r["history"][:, 0].max() for r in results)
        print(f"\n  --- MATCHED shortening {target * 100:.2f} %, the largest both runs reached ---")
        for r in results:
            report(r, which=target)
        makePlots(results, which=target, fname=f"contour_plots_matched{tag}.png")

    for r in results:
        np.savez_compressed(
            f"snapshots_frame{r['frameUpdate']}{tag}.npz",
            xy0=r["xy0"],
            history=r["history"],
            shortening=np.array([sn["shortening"] for sn in r["snapshots"]]),
            omega=np.array([sn["omega"] for sn in r["snapshots"]]),
            frameRotation=np.array([sn["frameRotation"] for sn in r["snapshots"]]),
            alphaP=np.array([sn["alphaP"] for sn in r["snapshots"]]),
            axis1=np.array([sn["axis1"] for sn in r["snapshots"]]),
            axis2=np.array([sn["axis2"] for sn in r["snapshots"]]),
            u=np.array([sn["u"] for sn in r["snapshots"]]),
            **( {"verts": np.array([sn["verts"] for sn in r["snapshots"]])}
                if r["snapshots"][0]["verts"] is not None else {} ),
        )
        print(f"  wrote snapshots_frame{r['frameUpdate']}{tag}.npz")
