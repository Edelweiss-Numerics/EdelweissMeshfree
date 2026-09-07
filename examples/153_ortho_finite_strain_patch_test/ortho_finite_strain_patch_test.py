#!/usr/bin/env python3
"""
Patch test of the orthotropic finite-strain RKPM discretisation, over the bedding orientation.

WHAT A PATCH TEST IS HERE.  A homogeneous linear displacement field u_i = A_ij X_j is imposed
on the boundary of a square patch of randomly perturbed particles, and the interior is required
to reproduce that field.  Nothing about it is a fit: the exact solution is a member of the
approximation space (a linear basis reproduces a linear field), and a homogeneous deformation
gradient produces a homogeneous Kirchhoff stress, so div tau = 0 is satisfied exactly and the
exact solution is also the exact solution of the discrete problem.  Any departure is therefore
a defect of the discretisation -- of the reproducing conditions, of the domain integration, or
of the way the essential conditions are imposed -- and not a discretisation error in the usual
sense.  That is what makes the test binary rather than a convergence statement.

WHY OVER THE BEDDING ORIENTATION.  The material is orthotropic at finite strain, and off the
material axes the Biot stress and the elastic stretch stop being coaxial (main.tex Sec. 3.1,
Fig. 4): the constitutive routine then exercises the full non-symmetric Mandel stress, the
spectral evaluation of d ln U / dU with distinct eigenvalues, and the Walpole mapping tensor in
a frame that is not the coordinate frame.  A patch test run only at beta = 0 misses all of it.
The sweep below therefore runs every load case at seven bedding orientations from 0 to 90 deg.

WHY THREE LOAD CASES.  A single one can pass by accident.  Uniaxial stretch keeps the stretch
coaxial with the coordinate axes; simple shear makes the deformation gradient non-symmetric;
the mixed case is coaxial with neither the coordinate axes nor the material axes at any
orientation in the sweep.  All three are homogeneous, so all three have the same exact-solution
property.

WHAT IS MEASURED
    err(u)  max over interior particles of |u_num - A x| / max|A x|
    err(F)  max over interior particles of |F_num - (I + A)| / |A|
    alphaP  the hardening variable, which must be identically zero -- the check that the run
            really was elastic, and hence that the exact-solution argument applies at all.
The strengths are scaled by 1e3 so that no orientation can yield at the amplitudes used; the
elastic card and everything else is the reference card of examples/152.

USAGE
    python ortho_finite_strain_patch_test.py                # the 3 x 7 sweep, coarse patch
    python ortho_finite_strain_patch_test.py --nx 12        # a finer patch
    python ortho_finite_strain_patch_test.py --perturb 0.0  # regular particle distribution
    python ortho_finite_strain_patch_test.py --refine       # the same sweep at 3 spacings
    python ortho_finite_strain_patch_test.py --figure       # write the paper figure
Run under BASE python (/home/tom/miniforge3/bin/python).
"""

import argparse
import math
import os

import numpy as np
import pytest
from edelweissfe.config.linsolve import getLinSolverByName
from edelweissfe.journal.journal import Journal
from edelweissfe.timesteppers.adaptivetimestepper import AdaptiveTimeStepper
from edelweissfe.utils.exceptions import StepFailed

from edelweissmeshfree.constraints.particlelagrangianweakdirichlet import (
    ParticleLagrangianWeakDirichlet,
)
from edelweissmeshfree.fieldoutput.fieldoutput import MPMFieldOutputController
from edelweissmeshfree.meshfree.approximations.marmot.marmotmeshfreeapproximation import (
    MarmotMeshfreeApproximationWrapper,
)
from edelweissmeshfree.meshfree.kernelfunctions.marmot.marmotmeshfreekernelfunction import (
    MarmotMeshfreeKernelFunctionWrapper,
)
from edelweissmeshfree.meshfree.particlekerneldomain import ParticleKernelDomain
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.particlemanagers.kdbinorganizedparticlemanager import (
    KDBinOrganizedParticleManager,
)
from edelweissmeshfree.particles.marmot.marmotparticlewrapper import (
    MarmotParticleWrapper,
)
from edelweissmeshfree.sets.particleset import ParticleSet
from edelweissmeshfree.solvers.nqs import NonlinearQuasistaticSolver
from edelweissfe.points.node import Node

HERE = os.path.dirname(os.path.abspath(__file__))

# =============================================================================================
#  material -- the reference card of examples/152, with the strengths lifted out of reach
# =============================================================================================


def saintVenantG(Ei, Ej, nuij):
    """Extended Saint Venant formula: 1/Gij = 1/Ei + 1/Ej + 2 nuij/Ej."""
    return 1.0 / (1.0 / Ei + 1.0 / Ej + 2.0 * nuij / Ej)


E1, E2, E3 = 2400.0, 2400.0, 1800.0
NU12, NU13, NU23 = 0.21, 0.24, 0.24
G12 = saintVenantG(E1, E2, NU12)
G13 = saintVenantG(E1, E3, NU13)
G23 = saintVenantG(E2, E3, NU23)

# The patch test is a statement about the DISCRETISATION, so the constitutive law must stay on
# its elastic branch at every orientation.  1e3 on the four strengths puts the yield surface
# three orders above the ~1e1 MPa the amplitudes below produce; `alphaP` is checked afterwards
# rather than assumed.
STRENGTH_SCALE = 1.0e3
FCU = 51.03 * STRENGTH_SCALE
FTU = FCU / 10.0
FCY = FCU / 3.0
FBU = 1.16 * FCU

AH, BH, CH, DH = 0.08, 0.003, 2.0, 1e-6
AS, DF = 2.0, 0.85
SOFTMOD, MAXDMG = 3.95e-3, 0.9999

# genuinely orthotropic weights, so the mapping tensor is not the identity in any frame
ALPHA, BETA_W, GAMMA_W = 1.20, 1.00, 1.00
ZETA, XI, ETA = 1.30, 1.00, 1.00

L_NONLOCAL, WEIGHT_M = 1.25, 1.05
DAMAGE_ONSET, H_RESIDUAL = 0.95, 0.02

LENGTH = 10.0  # the patch is LENGTH x LENGTH


def materialProperties(beddingDeg, frameUpdate=1):
    """The 33-property card of GradientEnhancedOrthoCDPFiniteStrain."""
    phi = math.radians(beddingDeg)
    return np.array(
        [
            E1, E2, E3,
            NU12, NU13, NU23,
            G12, G13, G23,
            math.cos(phi), math.sin(phi), 0.0,      # bedding normal n0 in the x-y plane
            FCY, FCU, FBU, FTU,
            DF,
            AH, BH, CH, DH, AS,
            SOFTMOD, MAXDMG,
            ALPHA, BETA_W, GAMMA_W, ZETA, XI, ETA,
            L_NONLOCAL, WEIGHT_M,
            float(frameUpdate),
            DAMAGE_ONSET, H_RESIDUAL,
        ]
    )


# =============================================================================================
#  the load cases: constant displacement gradients A, so that u = A X and F = I + A
# =============================================================================================

AMPLITUDE = 0.02

LOAD_CASES = {
    # uniaxial stretch with a lateral contraction: the stretch stays coaxial with x, y
    "stretch": np.array([[AMPLITUDE, 0.0], [0.0, -0.3 * AMPLITUDE]]),
    # simple shear: F is non-symmetric, so R^e is not the identity anywhere
    "shear": np.array([[0.0, AMPLITUDE], [0.0, 0.0]]),
    # coaxial with neither the coordinate axes nor the material axes at any beta in the sweep
    "mixed": np.array([[0.8 * AMPLITUDE, 0.6 * AMPLITUDE],
                       [-0.35 * AMPLITUDE, -0.5 * AMPLITUDE]]),
}


def exactDisplacement(A, xy):
    """u_i = A_ij X_j, evaluated on an (n, 2) array of reference coordinates."""
    return xy @ A.T


# =============================================================================================
#  a randomly perturbed quad particle grid
# =============================================================================================


def generatePerturbedQuadGrid(model, journal, particleFactory, kernelFactory,
                              length, nX, perturb, rng, shrink=0.0):
    """Tile [0,L]^2 with nX x nX quad particles on a randomly perturbed vertex lattice.

    The lattice is perturbed rather than the particles independently, so the smoothing domains
    still TILE the patch: they share their vertices, no area is counted twice and none is
    missed.  That matters here because the domain integration is nodal -- one point per
    particle, weighted by the particle's own area -- so an overlapping or gapped tiling would
    break the integration constraint (main.tex Eq. 72) by construction and the test would be
    measuring the generator rather than the discretisation.

    Boundary vertices slide ALONG the boundary and the four corners are pinned, so the patch
    stays exactly square: a perturbed boundary would make the imposed field and the domain
    disagree about where the boundary is.

    `perturb` is the perturbation amplitude in units of the cell size h = L/nX; 0.4 is the
    usual choice and is what the paper reports.
    """
    h = length / nX
    nV = nX + 1
    g = np.mgrid[0.0 : length : nV * 1j, 0.0 : length : nV * 1j]
    V = np.stack([g[0], g[1]], axis=-1)  # (nV, nV, 2)

    if perturb > 0.0:
        d = perturb * h * (2.0 * rng.random((nV, nV, 2)) - 1.0)
        interior = np.zeros((nV, nV), dtype=bool)
        interior[1:-1, 1:-1] = True
        V[interior] += d[interior]
        # edges: one tangential component only
        V[1:-1, 0, 0] += d[1:-1, 0, 0]
        V[1:-1, -1, 0] += d[1:-1, -1, 0]
        V[0, 1:-1, 1] += d[0, 1:-1, 1]
        V[-1, 1:-1, 1] += d[-1, 1:-1, 1]

    def quadArea(v):
        """Shoelace area of the four corners, in the generator's CCW order."""
        x, y = v[:, 0], v[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    # `shrink` deliberately BREAKS the conforming tiling: each smoothing domain is scaled about
    # its own centroid by a random factor, so the cells no longer share their vertices and
    # sum_q V_q < |Omega|.  That is what violates the integration constraint of Eq. (72), and it
    # is the configuration in which the correction of Eq. (73) has something to repair.  With a
    # conforming tiling the SQCNI gradients satisfy the constraint by construction.
    particles, kernels = [], []
    number = 1
    for i in range(nX):
        for j in range(nX):
            v = np.asarray([V[i, j], V[i + 1, j], V[i + 1, j + 1], V[i, j + 1]])
            if shrink > 0.0:
                c = v.mean(axis=0)
                v = c + (1.0 - shrink * rng.random()) * (v - c)
            p = particleFactory(number, v, quadArea(v))
            model.particles[number] = p
            particles.append(p)
            # one kernel function per particle, at the particle's own centroid: the classical
            # RKPM arrangement, and it is what makes the perturbation a perturbation of the
            # PARTICLE distribution rather than of the quadrature alone
            kf = kernelFactory(Node(number, v.mean(axis=0)))
            model.meshfreeKernelFunctions[number] = kf
            kernels.append(kf.node)
            number += 1

    model.nodes.update({n.label: n for n in kernels})
    grid = np.asarray(particles).reshape(nX, nX)
    model.particleSets["all_particles"] = ParticleSet("all_particles", grid.flatten())
    boundary = (list(grid[0, :]) + list(grid[-1, :])
                + list(grid[1:-1, 0]) + list(grid[1:-1, -1]))
    model.particleSets["boundary"] = ParticleSet("boundary", boundary)
    for k in (1, 2, 3):
        band, inner = [], []
        for i in range(nX):
            for j in range(nX):
                if i < k or j < k or i >= nX - k or j >= nX - k:
                    band.append(grid[i, j])
                else:
                    inner.append(grid[i, j])
        model.particleSets[f"band{k}"] = ParticleSet(f"band{k}", band)
        model.particleSets[f"inner{k}"] = ParticleSet(f"inner{k}", inner)
    # the four sides separately: the VCI boundary integral of Eq. (72) needs one
    # BoundaryParticleDefinition per side, with the quad face id of that side
    for tag, sel in (("left", grid[0, :]), ("right", grid[-1, :]),
                     ("bottom", grid[:, 0]), ("top", grid[:, -1])):
        model.particleSets[tag] = ParticleSet(tag, list(sel))
    inner = grid[1:-1, 1:-1].flatten()
    model.particleSets["interior"] = ParticleSet("interior", list(inner))
    journal.message(
        f"patch {length} x {length}, {nX} x {nX} particles, h = {h:.4f}, "
        f"perturbation {perturb:.2f} h, {len(boundary)} boundary / {len(inner)} interior",
        "setup",
    )
    return model


# =============================================================================================
#  one patch test
# =============================================================================================


def run_patch(beddingDeg, case="mixed", nX=8, perturb=0.4, seed=7, particle="sqcnixnsni",
              vci=True, vciOrder=1, nRings=1, support=2.5, shrink=0.0, journal=None):
    """Impose u = A X on the boundary and report how well the interior reproduces it.

    ``vci`` switches the variationally consistent integration correction of Eq. (73) on and
    off.  It is not a detail here: without it the one-point nodal rule violates the first-order
    integration constraint (72), the exact linear field is then NOT a solution of the discrete
    equations, and the patch test fails at a few per cent -- which is the whole reason the
    correction exists.

    ``vciOrder`` is the order of the monomial basis the correction is built on, and the DEFAULT
    OF THE PARTICLE IS 0.  Order 0 enforces only the constant term, which is the zeroth-order
    (partition-of-unity) constraint; the constraint of Eq. (72) is the FIRST-order one and needs
    ``vciOrder = 1``.  Running with the correction switched on but left at its default order
    changes nothing at all in this test, which is exactly the trap met while setting it up.
    """
    A = LOAD_CASES[case]
    journal = journal or Journal()
    dimension = 2
    theModel = MPMModel(dimension)

    h = LENGTH / nX
    # UNIFORM support; a locally scaled one is what OOM-killed the earlier hexa studies.
    # THE FACTOR MATTERS AND 2.0 IS THE WRONG CHOICE HERE.  The kernels sit at the cell
    # centroids, so on an unperturbed lattice the second ring of neighbours lies at a distance
    # of exactly 2h -- the support radius -- where the cubic B-spline is exactly zero.  The
    # support then collapses to the 3x3 ring, the tangent comes out singular, and the run dies
    # with ||ddU|| ~ 1e2 and "return mapping not successful" on the FIRST increment.  Measured
    # on the regular 8x8 patch: 2.0 fails, 2.5 and 3.0 pass to 1e-13, 1.5 has too few
    # neighbours for the moment matrix.  A random perturbation of the lattice hides the problem
    # by moving the neighbours off the knife edge, which is exactly the wrong reason for a test
    # to pass, so the support is set away from it instead.
    supportRadius = support * h

    PARTICLES = {
        "sqcnixnsni": "GradientEnhancedFiniteStrainSQCNIxNSNI/PlaneStrain/Quad",
        "sqcni": "GradientEnhancedFiniteStrainSQCNI/PlaneStrain/Quad",
        "snni": "GradientEnhancedFiniteStrainSNNI/PlaneStrain/Quad",
        "snnixnsni": "GradientEnhancedFiniteStrainSNNIxNSNI/PlaneStrain/Quad",
        "point": "GradientEnhancedFiniteStrain/PlaneStrain/Point",
    }
    pName = PARTICLES[particle]
    quad = particle != "point"

    theApproximation = MarmotMeshfreeApproximationWrapper(
        "ReproducingKernelImplicitGradient", dimension, completenessOrder=1
    )
    card = {
        "material": "GRADIENTENHANCEDORTHOCDPFINITESTRAIN",
        "properties": materialProperties(beddingDeg),
    }

    theModel = generatePerturbedQuadGrid(
        theModel, journal,
        lambda number, verts, volume: MarmotParticleWrapper(
            pName, number, verts, volume, theApproximation, card
        ),
        lambda node: MarmotMeshfreeKernelFunctionWrapper(
            node, "BSplineBoxed", supportRadius=supportRadius, continuityOrder=2
        ),
        LENGTH, nX, perturb, np.random.default_rng(seed), shrink=shrink,
    )

    theParticleKernelDomain = ParticleKernelDomain(
        list(theModel.particles.values()), list(theModel.meshfreeKernelFunctions.values())
    )
    theParticleManager = KDBinOrganizedParticleManager(
        theParticleKernelDomain, dimension, journal, bondParticlesToKernelFunctions=True
    )
    if vci:
        for p in theModel.particles.values():
            p.setProperty("VCI order", vciOrder)

    theModel.particleKernelDomains["all_with_all"] = theParticleKernelDomain

    # ---- the essential conditions: the exact field at every boundary particle centre --------
    # One Lagrange multiplier constraint per particle and per component, so that each boundary
    # particle carries its OWN value of u = A x.  The multiplier method is used rather than the
    # penalty variant of examples/152 on purpose: it enforces the boundary values EXACTLY, so a
    # nonzero interior error cannot be blamed on a finite penalty stiffness.
    # THE CONSTRAINT MUST SIT ON THE DOMAIN BOUNDARY, NOT AT THE BOUNDARY PARTICLE'S CENTRE.
    # A particle centre lies half a cell inside the patch, and the weak form's surface integral
    # over the outer faces is then still traction FREE -- so prescribing the field at the
    # centres poses a body with a free outer strip and a displacement condition behind it,
    # whose solution is not the linear field at all.  Measured: 2.5 % error in u, insensitive
    # to the particle type, to VCI, and almost to h, which is what an inconsistent problem
    # rather than a discretisation defect looks like.  Constraining the boundary FACE centres
    # instead makes the multiplier the reaction of that face: the true traction integral over
    # the face is one-point-integrated at its centre, which is exactly the virtual work the
    # multiplier delivers.
    faceOf = {"bottom": 1, "right": 2, "top": 3, "left": 4}
    constraints = []
    for side, faceID in faceOf.items():
        for p in theModel.particleSets[side]:
            xy = np.asarray(p.getFaceCoordinates(faceID)).reshape(-1)[:2]
            u = A @ xy
            constraints.append(
                ParticleLagrangianWeakDirichlet(
                    f"bc_{side}_{p.number}", p, "displacement",
                    {0: float(u[0]), 1: float(u[1])}, theModel,
                    location="face", faceID=faceID,
                )
            )
    for c in constraints:
        theModel.constraints[c.name] = c

    theModel.prepareYourself(journal)

    fieldOutputController = MPMFieldOutputController(theModel, journal)
    for name in ("displacement", "deformation gradient", "alphaP", "stress"):
        fieldOutputController.addPerParticleFieldOutput(
            name, theModel.particleSets["all_particles"], name
        )
    fieldOutputController.initializeJob()

    iterationOptions = {
        "max. iterations": 25,
        "critical iterations": 8,
        "allowed residual growths": 6,
    }
    linearSolver = getLinSolverByName("pardiso", {})
    nonlinearSolver = NonlinearQuasistaticSolver(journal)

    vciManagers = []
    if vci:
        from edelweissmeshfree.meshfree.vci import (
            BoundaryParticleDefinition,
            VariationallyConsistentIntegrationManager,
        )

        # quad face ids for the generator's CCW vertex order: 1 bottom, 2 right, 3 top, 4 left
        theBoundary = [
            BoundaryParticleDefinition(theModel.particleSets["left"], np.empty(2), 4),
            BoundaryParticleDefinition(theModel.particleSets["right"], np.empty(2), 2),
            BoundaryParticleDefinition(theModel.particleSets["bottom"], np.empty(2), 1),
            BoundaryParticleDefinition(theModel.particleSets["top"], np.empty(2), 3),
        ]
        vciManagers.append(
            VariationallyConsistentIntegrationManager(
                list(theModel.particles.values()),
                list(theModel.meshfreeKernelFunctions.values()),
                theBoundary,
            )
        )

    xy0 = np.array([np.asarray(p.getCenterCoordinates()).reshape(2)
                    for p in theModel.particleSets["all_particles"]])
    innerSet = set(theModel.particleSets[f"inner{nRings}"])
    isInterior = np.array([p in innerSet for p in theModel.particleSets["all_particles"]])

    failed = False
    try:
        nonlinearSolver.solveStep(
            AdaptiveTimeStepper(theModel.time, 1.0, 0.5, 1.0, 1e-4, 50, journal),
            linearSolver, theModel, fieldOutputController,
            outputManagers=[], particleManagers=[theParticleManager],
            constraints=constraints, userIterationOptions=iterationOptions,
            vciManagers=vciManagers,
        )
    except StepFailed as e:
        journal.message(f"patch test step failed: {e}", "error")
        failed = True

    fo = fieldOutputController.fieldOutputs
    nP = len(theModel.particles)
    # The per-particle displacement comes back as a 3-vector per particle even in plane strain,
    # so it is reshaped by the particle count rather than by the model dimension; getting this
    # wrong is a silent broadcast error against the interior mask.
    uNum = fo["displacement"].getLastResult().reshape(nP, -1)[:, :dimension]
    FNum = fo["deformation gradient"].getLastResult().reshape(nP, 3, 3)
    alphaP = fo["alphaP"].getLastResult().reshape(nP, -1)

    uEx = exactDisplacement(A, xy0)
    scale = np.abs(uEx).max()
    FEx = np.eye(3)
    FEx[:2, :2] += A

    inner = isInterior & np.isfinite(uNum).all(axis=1)
    errU = np.abs(uNum - uEx)[inner].max() / scale
    errF = np.abs(FNum - FEx)[inner].max() / np.abs(A).max()

    return dict(
        bedding=beddingDeg, case=case, nX=nX, h=h, perturb=perturb, vci=vci,
        vciOrder=(vciOrder if vci else None), nRings=nRings, support=support, shrink=shrink,
        errU=float(errU), errF=float(errF),
        alphaPMax=float(np.abs(alphaP).max()),
        nInterior=int(inner.sum()), failed=failed,
        errUField=np.abs(uNum - uEx).max(axis=1) / scale,
        xy0=xy0, interior=isInterior,
    )


# =============================================================================================
#  the sweep
# =============================================================================================

BEDDINGS = [0, 15, 30, 45, 60, 75, 90]


def sweep(nX=8, perturb=0.4, cases=tuple(LOAD_CASES), beddings=tuple(BEDDINGS), seed=7,
          vci=True, vciOrder=1, nRings=1, support=2.5, shrink=0.0):
    journal = Journal()
    out = []
    for case in cases:
        for b in beddings:
            r = run_patch(b, case=case, nX=nX, perturb=perturb, seed=seed, vci=vci,
                          vciOrder=vciOrder, nRings=nRings, support=support, shrink=shrink,
                          journal=journal)
            out.append(r)
            print(f"  {case:8s} beta = {b:5.1f} deg   err(u) = {r['errU']:.2e}   "
                  f"err(F) = {r['errF']:.2e}   max alphaP = {r['alphaPMax']:.1e}"
                  f"{'   STEP FAILED' if r['failed'] else ''}")
    return out


def report(results):
    print("\n" + "=" * 92)
    print(f"  patch test, {results[0]['nX']}x{results[0]['nX']} particles, "
          f"perturbation {results[0]['perturb']:.2f} h")
    print("=" * 92)
    print(f"  {'case':10s}" + "".join(f"{b:>10g}" for b in BEDDINGS) + "     worst")
    for tag, key in (("err(u)", "errU"), ("err(F)", "errF")):
        for case in LOAD_CASES:
            row = [r for r in results if r["case"] == case]
            if not row:
                continue
            print(f"  {case + ' ' + tag:16s}" + "".join(f"{r[key]:10.1e}" for r in row)
                  + f"{max(r[key] for r in row):10.1e}")
    worstU = max(r["errU"] for r in results)
    worstF = max(r["errF"] for r in results)
    worstA = max(r["alphaPMax"] for r in results)
    nFail = sum(r["failed"] for r in results)
    print("-" * 92)
    print(f"  worst over the whole sweep:  err(u) = {worstU:.2e}   err(F) = {worstF:.2e}   "
          f"max alphaP = {worstA:.1e}   failed steps: {nFail}")
    verdict = "PASS" if (worstU < 1e-8 and worstF < 1e-8 and worstA == 0.0 and nFail == 0) else "FAIL"
    print(f"  {verdict}: the interior reproduces the imposed field to machine precision at "
          f"every orientation" if verdict == "PASS" else f"  {verdict}")
    print("=" * 92)
    return verdict


# =============================================================================================
#  the figure
# =============================================================================================


def paperStyle(figWidthIn, pageFrac=1.0, legacyBase=10.0):
    """The paper's one figure style; see paper_FiniteStrainOrthoCDP/tools/paperstyle.py."""
    import sys
    cand = os.path.abspath(os.path.join(HERE, "..", "..", "..",
                                        "paper_FiniteStrainOrthoCDP", "tools"))
    if cand not in sys.path:
        sys.path.insert(0, cand)
    try:
        import paperstyle
    except ImportError:
        print(f"  paperstyle.py not found under {cand} -- using matplotlib defaults")
        return 1.0
    return paperstyle.apply(fig_width_in=figWidthIn, page_frac=pageFrac,
                            legacy_base=legacyBase, grid=False)


def latticeForDrawing(nX, perturb, seed=7, shrink=0.0):
    """Rebuild, for drawing only, the lattice a run with these settings used."""
    rng = np.random.default_rng(seed)
    h = LENGTH / nX
    nV = nX + 1
    g = np.mgrid[0.0 : LENGTH : nV * 1j, 0.0 : LENGTH : nV * 1j]
    V = np.stack([g[0], g[1]], axis=-1)
    if perturb > 0.0:
        d = perturb * h * (2.0 * rng.random((nV, nV, 2)) - 1.0)
        interior = np.zeros((nV, nV), dtype=bool)
        interior[1:-1, 1:-1] = True
        V[interior] += d[interior]
        V[1:-1, 0, 0] += d[1:-1, 0, 0]
        V[1:-1, -1, 0] += d[1:-1, -1, 0]
        V[0, 1:-1, 1] += d[0, 1:-1, 1]
        V[-1, 1:-1, 1] += d[-1, 1:-1, 1]
    cells, shrunk, isBnd = [], [], []
    for i in range(nX):
        for j in range(nX):
            v = np.asarray([V[i, j], V[i + 1, j], V[i + 1, j + 1], V[i, j + 1]])
            cells.append(v)
            c = v.mean(axis=0)
            shrunk.append(c + (1.0 - shrink * rng.random()) * (v - c) if shrink > 0 else v)
            isBnd.append(i in (0, nX - 1) or j in (0, nX - 1))
    return cells, shrunk, np.asarray(isBnd)


def makeFigure(conforming, nonconforming=None, out=None):
    """Three panels, in the style of the plane-strain compression figures of the paper."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.lines import Line2D

    figW = 12.2
    FS = paperStyle(figW)
    fig, ax = plt.subplots(1, 3, figsize=(figW, 3.9))

    cols = {"stretch": "#1b6ca8", "shear": "#e8871a", "mixed": "#2e8b57"}
    marks = {"stretch": "o", "shear": "s", "mixed": "^"}
    FLOOR = 1e-17

    # ---------------------------------------------------------------- (a) the patch
    a = ax[0]
    r0 = conforming[0]
    nX, pert = r0["nX"], r0["perturb"]
    cells, shrunk, isBnd = latticeForDrawing(nX, pert, shrink=0.25)
    a.add_collection(PolyCollection([cells[k] for k in np.where(~isBnd)[0]],
                                    facecolors="#eef3f8", edgecolors="0.55",
                                    linewidths=0.5 * FS))
    a.add_collection(PolyCollection([cells[k] for k in np.where(isBnd)[0]],
                                    facecolors="#f7e2d3", edgecolors="0.55",
                                    linewidths=0.5 * FS))
    a.add_collection(PolyCollection(shrunk, facecolors="none", edgecolors="#b1500f",
                                    linewidths=0.55 * FS, linestyles="--", alpha=0.75))
    cen = np.array([c.mean(axis=0) for c in cells])
    a.plot(cen[:, 0], cen[:, 1], ".", color="0.2", ms=2.4 * FS)
    # the imposed field acts on the boundary FACE centres, which is where the reaction lives
    fc = []
    e = LENGTH
    for c, b in zip(cells, isBnd):
        if not b:
            continue
        for k in range(4):
            m = 0.5 * (c[k] + c[(k + 1) % 4])
            if min(m[0], m[1]) < 1e-9 or max(m[0], m[1]) > e - 1e-9:
                fc.append(m)
    fc = np.asarray(fc)
    a.plot(fc[:, 0], fc[:, 1], "o", color="#b1500f", ms=2.8 * FS)
    a.set_xlim(-0.5, LENGTH + 0.5)
    a.set_ylim(-0.5, LENGTH + 0.5)
    a.set_aspect("equal")
    a.set_xlabel(r"$X_1$ [mm]")
    a.set_ylabel(r"$X_2$ [mm]")
    a.set_title(rf"(a) patch, {nX}$\times${nX} particles, ${pert:.1f}\,h_p$ perturbation",
                fontsize=9.5 * FS)
    a.legend(handles=[
        Line2D([], [], marker="o", ls="none", color="#b1500f", ms=2.8 * FS,
               label=r"$u=\mathbf{A}\mathbf{X}$ imposed"),
        Line2D([], [], marker=".", ls="none", color="0.2", ms=2.4 * FS, label="particle"),
        Line2D([], [], ls="--", color="#b1500f", lw=0.55 * FS, label="non-conforming domains"),
    ], loc="upper center", ncol=1, fontsize=7.4 * FS, frameon=True, framealpha=0.92)

    # ------------------------------------------------- (b) conforming, over the orientation
    b = ax[1]
    for case in LOAD_CASES:
        rr = sorted([r for r in conforming if r["case"] == case], key=lambda r: r["bedding"])
        if not rr:
            continue
        b.semilogy([r["bedding"] for r in rr], [max(r["errU"], FLOOR) for r in rr],
                   "-", marker=marks[case], color=cols[case], ms=3.4 * FS, lw=1.1 * FS,
                   label=rf"{case}, $u$")
        b.semilogy([r["bedding"] for r in rr], [max(r["errF"], FLOOR) for r in rr],
                   "--", marker=marks[case], color=cols[case], ms=3.0 * FS, lw=1.0 * FS,
                   mfc="none", label=rf"{case}, $F$")
    b.set_xticks(BEDDINGS)
    b.set_xlabel(r"bedding orientation $\beta$ [deg]")
    b.set_ylabel("relative reproduction error")
    b.set_ylim(FLOOR, 1.0)
    b.set_title("(b) conforming smoothing domains", fontsize=9.5 * FS)
    b.legend(loc="upper center", ncol=3, fontsize=7.0 * FS, frameon=False,
             columnspacing=0.9, handlelength=1.5)
    b.grid(True, which="major", color="#DDDDDD", lw=0.4 * FS)

    # ------------------------------------------------- (c) non-conforming, with / without VCI
    c = ax[2]
    if nonconforming:
        for case in LOAD_CASES:
            for vciOn, style, kw in ((True, "-", dict()), (False, ":", dict(mfc="none"))):
                rr = sorted([r for r in nonconforming
                             if r["case"] == case and r["vci"] is vciOn],
                            key=lambda r: r["bedding"])
                if not rr:
                    continue
                c.semilogy([r["bedding"] for r in rr], [max(r["errU"], FLOOR) for r in rr],
                           style, marker=marks[case], color=cols[case], ms=3.4 * FS,
                           lw=1.1 * FS,
                           label=rf"{case}, {'with' if vciOn else 'without'} VCI", **kw)
        c.set_xticks(BEDDINGS)
        c.set_xlabel(r"bedding orientation $\beta$ [deg]")
        c.set_ylabel(r"relative error in $u$")
        c.set_ylim(FLOOR, 1.0)
        # no equation number in the title: the paper renumbers, the figure does not
        c.set_title("(c) non-conforming domains, VCI on and off", fontsize=9.5 * FS)
        c.legend(loc="center right", ncol=1, fontsize=7.0 * FS, frameon=False,
                 handlelength=1.5)
        c.grid(True, which="major", color="#DDDDDD", lw=0.4 * FS)
    else:
        c.axis("off")

    fig.tight_layout()
    out = out or os.path.join(HERE, "fig_patch_test.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=145)
    print(f"  wrote {out}")


# =============================================================================================
#  CLI and the regression test
# =============================================================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=8)
    ap.add_argument("--perturb", type=float, default=0.4)
    ap.add_argument("--support", type=float, default=2.5,
                    help="kernel support radius in units of the particle spacing")
    ap.add_argument("--no-vci", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--case", default=None, choices=list(LOAD_CASES))
    ap.add_argument("--bedding", type=float, default=None)
    ap.add_argument("--nonconforming", action="store_true",
                    help="also run the sweep on non-conforming smoothing domains")
    ap.add_argument("--refine", action="store_true",
                    help="repeat the sweep at three particle spacings")
    ap.add_argument("--figure", action="store_true", help="write fig_patch_test.pdf")
    args = ap.parse_args()

    cases = (args.case,) if args.case else tuple(LOAD_CASES)
    beddings = (args.bedding,) if args.bedding is not None else tuple(BEDDINGS)

    results = sweep(nX=args.nx, perturb=args.perturb, cases=cases, beddings=beddings,
                    seed=args.seed, vci=not args.no_vci, support=args.support)
    report(results)

    nonconforming = None
    if args.nonconforming or args.figure:
        print("\n  the same sweep on NON-CONFORMING smoothing domains, Eq. (73) on and off")
        nonconforming = []
        for vciOn in (True, False):
            nonconforming += sweep(nX=args.nx, perturb=args.perturb, cases=cases,
                                   beddings=beddings, seed=args.seed, vci=vciOn,
                                   support=args.support, shrink=0.25)
        report([r for r in nonconforming if r["vci"] is False])

    if args.refine:
        worst = max(results, key=lambda r: r["errU"])["bedding"]
        print(f"\n  refinement at beta = {worst} deg, conforming and non-conforming")
        for nX in (6, 8, 12, 16):
            for shrink, vciOn in ((0.0, True), (0.25, False)):
                r = run_patch(worst, case="mixed", nX=nX, perturb=args.perturb,
                              seed=args.seed, vci=vciOn, support=args.support, shrink=shrink)
                tag = "conforming+VCI" if shrink == 0.0 else "non-conforming, no VCI"
                print(f"    {tag:24s} nX = {nX:3d}  h = {r['h']:.3f}  "
                      f"err(u) = {r['errU']:.2e}  err(F) = {r['errF']:.2e}")

    if args.figure:
        makeFigure(results, nonconforming)


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_patch():
    """The patch test at three orientations, one load case -- a fast regression guard."""
    results = sweep(nX=6, perturb=0.4, cases=("mixed",), beddings=(0, 45, 90))
    assert all(not r["failed"] for r in results)
    assert max(r["errU"] for r in results) < 1e-8
    assert max(r["errF"] for r in results) < 1e-8
    assert max(r["alphaPMax"] for r in results) == 0.0


if __name__ == "__main__":
    main()
