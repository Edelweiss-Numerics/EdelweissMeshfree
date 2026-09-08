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

A FOURTH CASE, AND WHY.  The three above are pure gradients, u = A X, which leave the origin
fixed and test only the first-order part of the reproducing conditions.  `affine` carries a
translation as well, u = c + A X with c = (0.10, 0.05) mm and A = [[0.10, 0.20], [0.15, 0.10]],
so it tests the zeroth-order part too -- that a rigid translation is reproduced exactly -- and
it is the largest deformation in this file, det F = 1.18.  It is not in the Fig. 14 sweep; it
has its own study and its own figure, `--affine`, which adds the ENERGY error to the two below
and shows both as contours over the patch.

WHAT IS MEASURED
    err(u)  max over interior particles of |u_num - u_ex| / max|u_ex|
    err(F)  max over interior particles of |F_num - (I + A)| / |A|
    err(E)  max over interior particles of |Psi(F_num) - Psi(F_ex)| / Psi(F_ex), with Psi the
            STORED energy density Psi^e(U) - Psi^e(I).  Marmot does not export it, so it is
            evaluated here with the paper's own reference implementation of the potential.
    alphaP  the hardening variable, which must be identically zero -- the check that the run
            really was elastic, and hence that the exact-solution argument applies at all.
The strengths are scaled by 1e3 so that no orientation can yield at the amplitudes used; the
elastic card is the one of examples/152 with its AXES RELABELLED so that the soft modulus is
the one across the bedding, e^(1) -- see the note above E1 -- and the Walpole weights are made
transversely isotropic about the same axis, so that the elastic anisotropy and the plastic one
are the same anisotropy.  Everything else is the card of examples/152.  NOTE that 152 itself
has not been changed; whether to propagate this is open.

THE THREE THINGS THE TEST SETTLED, each a switch of its own
    --bc {face,center,cwf}   where the essential condition acts.  `face` is the one that
        works; `center` is the instructive failure (2 % error, insensitive to everything);
        `cwf` adds the consistent-weak-form correction, which does not converge on a fully
        constrained boundary -- see sweepBoundary().
    --perturb X              how random the distribution is, in units of h_p.  Limited by the
        VALIDITY of the quad cells, not by the approximation: see sweepPerturbation() and
        cellQuality(), which counts concave cells.
    --particle {sqcni,sqcni_r,sqcni_ru,snni,...}   how the smoothing domain is carried into
        the deformed configuration.  Only the full deformation gradient passes for every load
        case; see sweepUpdateType().

USAGE
    python ortho_finite_strain_patch_test.py --all --figure   # everything + the paper figure
    python ortho_finite_strain_patch_test.py --affine         # u = c + A X, energy + contours
    python ortho_finite_strain_patch_test.py --nx 12          # a finer patch
    python ortho_finite_strain_patch_test.py --refine         # the spacing study
    python ortho_finite_strain_patch_test.py --no-vci         # the VCI correction off
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
from edelweissmeshfree.stepactions.particledistributedload import (
    ParticleDistributedLoad,
)
from edelweissfe.surfaces.entitybasedsurface import EntityBasedSurface
from edelweissfe.points.node import Node

HERE = os.path.dirname(os.path.abspath(__file__))

# =============================================================================================
#  material -- examples/152's card, axes aligned on the bedding normal, strengths out of reach
# =============================================================================================


def saintVenantG(Ei, Ej, nuij):
    """Extended Saint Venant formula: 1/Gij = 1/Ei + 1/Ej + 2 nuij/Ej."""
    return 1.0 / (1.0 / Ei + 1.0 / Ej + 2.0 * nuij / Ej)


# THE CARD, AND THE ONE THING THAT WAS WRONG WITH IT.  e^(1) is the bedding NORMAL, by the
# convention of the paper and of the material alike, so a bedded rock has the soft modulus on
# axis 1 and its isotropy plane spanned by e^(2), e^(3).  Examples 152 and 153 used to carry
# E1 = E2 = 2400, E3 = 1800 -- the same three moduli, but with the distinguished axis on 3.
# That makes the material transversely isotropic about the OUT-OF-PLANE axis, i.e. the plane
# of a plane-strain model IS its isotropy plane, and then no in-plane deformation can feel the
# bedding orientation at all: measured before the fix, the stored energy and every stress
# component were identical to the last digit at beta = 0, 30, 45 and 90.  The numbers below are
# the same material with the axes relabelled the way the convention requires.
E1, E2, E3 = 1800.0, 2400.0, 2400.0        # 1 = across the bedding, 2-3 = the bedding plane
NU12, NU13, NU23 = 0.24, 0.24, 0.21        # 1-2 and 1-3 across, 2-3 within
G12 = saintVenantG(E1, E2, NU12)
G13 = saintVenantG(E1, E3, NU13)
G23 = saintVenantG(E2, E3, NU23)           # = E2/2(1+nu23): the isotropy plane, exactly

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

# The Walpole weights of the yield surface, in Marmot's Voigt order (11, 22, 33, 12, 13, 23),
# so (ALPHA, BETA_W, GAMMA_W) scale the normal components and (ZETA, XI, ETA) the shears.  They
# are TRANSVERSELY ISOTROPIC ABOUT e^(1) as well, which is what makes the plastic anisotropy
# and the elastic one the same anisotropy rather than two unrelated ones: o22 = o33 and
# o12 = o13, with o23 = o22 as isotropy in the 2-3 plane requires.  ZETA alone used to be
# raised, which left the map orthotropic while the elasticity was transversely isotropic about
# a different axis again.
ALPHA, BETA_W, GAMMA_W = 1.20, 1.00, 1.00      # o11 across the bedding, o22 = o33 within it
ZETA, XI, ETA = 1.30, 1.30, 1.00               # o12 = o13 across, o23 within

L_NONLOCAL, WEIGHT_M = 1.25, 1.05
DAMAGE_ONSET, H_RESIDUAL = 0.95, 0.02

LENGTH = 10.0  # the patch is LENGTH x LENGTH


def materialProperties(beddingDeg, frameUpdate=1, strengthScale=None):
    """The 33-property card of GradientEnhancedOrthoCDPFiniteStrain.

    `strengthScale` overrides STRENGTH_SCALE, which lifts the yield surface out of reach.  Set
    it to 1 and the same homogeneous field yields: the exact-solution argument survives that
    (a homogeneous F still gives a homogeneous stress, whatever the constitutive path), so the
    patch test remains binary and now tests the return map and the gradient-damage coupling
    rather than the elastic branch alone.
    """
    phi = math.radians(beddingDeg)
    k = 1.0 if strengthScale is None else strengthScale / STRENGTH_SCALE
    return np.array(
        [
            E1, E2, E3,
            NU12, NU13, NU23,
            G12, G13, G23,
            math.cos(phi), math.sin(phi), 0.0,      # bedding normal n0 in the x-y plane
            k * FCY, k * FCU, k * FBU, k * FTU,
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

# NOTE ON THE NAMES.  These are prescribed DEFORMATIONS, not stress states.  The whole boundary
# is constrained, so both components of every case are imposed: the -0.3 in "stretch" is a
# chosen lateral contraction, NOT a Poisson response to an axial pull, and the deformed patch
# contracts laterally because it was told to.  Nothing here is a material response; the test is
# about reproduction in the interior.
LOAD_CASES = {
    # axial stretch with a prescribed lateral contraction; coaxial with x, y
    "stretch": np.array([[AMPLITUDE, 0.0], [0.0, -0.3 * AMPLITUDE]]),
    # simple shear: F is non-symmetric, so R^e is not the identity anywhere
    "shear": np.array([[0.0, AMPLITUDE], [0.0, 0.0]]),
    # coaxial with neither the coordinate axes nor the material axes at any beta in the sweep
    "mixed": np.array([[0.8 * AMPLITUDE, 0.6 * AMPLITUDE],
                       [-0.35 * AMPLITUDE, -0.5 * AMPLITUDE]]),
}


def exactDisplacement(A, xy, c=None):
    """u_i = c_i + A_ij X_j, evaluated on an (n, 2) array of reference coordinates."""
    u = xy @ A.T
    return u if c is None else u + np.asarray(c).reshape(1, 2)


# A FOURTH CASE, WITH A TRANSLATION IN IT.  The three cases above are pure gradients, u = A X,
# which leaves the origin fixed and tests only the FIRST-order part of the reproducing
# conditions.  This one carries a constant term as well,
#
#     u = c + A X,   c = (0.10, 0.05) mm,   A = [[0.10, 0.20], [0.15, 0.10]],
#
# so it also tests the zeroth-order part -- that the approximation reproduces a rigid
# translation exactly, which is the partition-of-unity property of the kernels, and that the
# smoothing-domain update carries it without drift.  It is far from small: F = I + A has
# det F = 1.18 and 20 % off-diagonal terms, non-symmetric, coaxial with nothing, and it is the
# largest deformation anywhere in this file.  Everything the three cases above are measured
# with applies unchanged, because the field is still affine and the stress it produces is
# still homogeneous.
EXTRA_CASES = {
    # THE FIELD, and every choice in it is forced by something.
    #
    #   u = c + A X,   c = (0.10, 0.05) mm,   A = [[-0.010, -0.020], [-0.015, -0.010]]
    #
    # * AFFINE, so it is in the approximation space and produces a homogeneous stress: the
    #   exact field is then the exact solution of the discrete problem and the test is binary.
    # * with a TRANSLATION, because u = A X leaves the origin fixed and tests only the
    #   first-order part of the reproducing conditions.  The zeroth-order part -- a rigid
    #   translation reproduced exactly, which is the partition of unity of the kernels, and
    #   carried without drift by the smoothing-domain update -- needs a constant term.
    # * NON-SYMMETRIC and coaxial with neither the coordinate nor the material axes, so the
    #   Biot stress and the elastic stretch are not coaxial and the Mandel stress is genuinely
    #   non-symmetric.
    # * COMPRESSIVE and this small.  The test is run with the strengths at their REAL values,
    #   so the patch yields, and the amplitude is then set by the material and not by us: in
    #   tension this card reaches f_tu = 5.1 MPa and softens to nothing, and past the peak the
    #   homogeneous state stops being the only solution of the discrete problem, so the test
    #   would stop being binary.  At this amplitude the stress is 24 to 34 MPa over the sweep,
    #   between f_cy = 17 and f_cu = 51, and alphaP is 0.57 to 0.95: hardening throughout.
    #
    # Running it plastically is what makes the elastic run redundant rather than the other way
    # round: everything an elastic patch test exercises is exercised here as well, and the
    # return map and the gradient-damage coupling on top of it.
    "affine": (np.array([[-0.010, -0.020], [-0.015, -0.010]]), np.array([0.10, 0.05])),
}


def caseField(case):
    """(A, c) of a load case, from either table.  The three homogeneous ones have c = 0."""
    if case in LOAD_CASES:
        return LOAD_CASES[case], np.zeros(2)
    return EXTRA_CASES[case]


# =============================================================================================
#  the strain energy, for the energy error
# =============================================================================================
#
# The material does not export the elastic energy density (`response.elasticEnergyDensity` is
# left at zero in the Marmot module), so it is evaluated here, from the deformation gradient
# the run reports, with the paper's OWN reference implementation of the potential --
# `paper_FiniteStrainOrthoCDP/tools/orthotropic_hyperelasticity.py`, the same file the
# stress-measure and dissipation verifications use.  Psi^e is a function of the elastic right
# stretch in the MATERIAL frame, so U = sqrt(F^T F) is rotated by the bedding frame before it
# is passed; e^(1) is the bedding normal, which is the convention of the card and of the paper.
# The run is elastic (alphaP = 0 is checked), so F^e = F and there is no damage factor.

_POTENTIAL = {}


def potential():
    """The paper's Psi^e on the card of this study, or None if the paper tree is not there."""
    if "model" not in _POTENTIAL:
        import sys
        cand = os.path.abspath(os.path.join(HERE, "..", "..", "..",
                                            "paper_FiniteStrainOrthoCDP", "tools"))
        if cand not in sys.path:
            sys.path.insert(0, cand)
        try:
            from orthotropic_hyperelasticity import OrthotropicCard, OrthotropicNeoHooke
            _POTENTIAL["model"] = OrthotropicNeoHooke(
                OrthotropicCard(E1, E2, E3, NU12, NU13, NU23, G12, G13, G23))
        except ImportError:
            print(f"  orthotropic_hyperelasticity.py not found under {cand} -- "
                  f"the energy error is not evaluated")
            _POTENTIAL["model"] = None
    return _POTENTIAL["model"]


def strainEnergyDensity(F, beddingDeg):
    """The STORED energy density of a 3x3 deformation gradient: Psi^e(U) - Psi^e(I).

    The potential is not normalised to zero at the reference state -- Psi^e(I) = 1530 MPa on
    this card, against the 89 MPa the affine case stores -- so the reference value is
    subtracted.  Without that the relative energy error is flattered by a factor of twenty.
    U is taken to the bedding frame first; e^(1) is the bedding normal.
    """
    model = potential()
    if model is None:
        return float("nan")
    C = np.asarray(F).T @ np.asarray(F)
    w, V = np.linalg.eigh(0.5 * (C + C.T))
    U = V @ np.diag(np.sqrt(np.maximum(w, 1e-30))) @ V.T
    phi = math.radians(beddingDeg)
    ct, st = math.cos(phi), math.sin(phi)
    Q = np.array([[ct, -st, 0.0], [st, ct, 0.0], [0.0, 0.0, 1.0]])   # columns e1, e2, e3
    return model.energy(Q.T @ U @ Q) - model.energy(np.eye(3))


# The illustration case, and the reason it exists: a patch test CANNOT show what the
# smoothing-domain update does.  The update is per particle,
#     x = c0 + u(c0) + F(c0) (X - c0),
# so two domains sharing a reference vertex map it with THEIR OWN gradients -- but a
# homogeneous field gives every centre the same F, every per-centre map is the same affine map,
# and the deformed domains tile to 1e-15 no matter how large the stretch or how coarse the
# patch.  To see the update do anything the deformation has to be INHOMOGENEOUS.  This is a
# bending-type field, whose gradient varies linearly over the patch:
#     u_1 = -kappa X_1 (X_2 - L/2),      u_2 = kappa X_1^2 / 2
# It is prescribed on the whole boundary, so it is a well-posed elastic Dirichlet problem, but
# it is NOT a patch test: the exact field is not in the approximation space and div tau does
# not vanish, so the interior is whatever equilibrium gives.  It is drawn, not measured against.
BENDING_KAPPA = 0.060  # 1/mm; chosen so the per-centre gap reads at figure scale (~8 % of h_p)


def bendingDisplacement(xy, kappa=BENDING_KAPPA):
    x1, x2 = xy[:, 0], xy[:, 1]
    return np.stack([-kappa * x1 * (x2 - 0.5 * LENGTH), 0.5 * kappa * x1 ** 2], axis=1)


# =============================================================================================
#  a randomly perturbed quad particle grid
# =============================================================================================


def generatePerturbedQuadGrid(model, journal, particleFactory, kernelFactory,
                              length, nX, perturb, rng):
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

    particles, kernels = [], []
    number = 1
    for i in range(nX):
        for j in range(nX):
            v = np.asarray([V[i, j], V[i + 1, j], V[i + 1, j + 1], V[i, j + 1]])
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
              vci=True, vciOrder=1, nRings=1, support=2.5, bc="face",
              cwfRamp=lambda t: 1.0, amplitude=None, tolerance=None, fieldFun=None,
              strengthScale=None, journal=None):
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
    A, cOff = caseField(case)
    if amplitude is not None:
        A, cOff = A * (amplitude / AMPLITUDE), cOff * (amplitude / AMPLITUDE)
    # `fieldFun` replaces the homogeneous field by an arbitrary one; the error measures below
    # are then meaningless and the caller must not use them (see bendingDisplacement)
    uOf = fieldFun if fieldFun is not None else (lambda q: exactDisplacement(A, q, cOff))
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
        # The smoothing-domain update variants.  SQCNI deforms the domain by the deformation
        # gradient evaluated at its centre, SQCNI_R by the rotation only, SQCNI_RU by the
        # rotation and the principal stretches, SNNI not at all.  This is the physical source
        # of non-conformity in this framework: a domain that is not carried by F stops tiling
        # the DEFORMED body, and the integration constraint is a statement about the deformed
        # configuration.
        "sqcni_r": "GradientEnhancedFiniteStrainSQCNI_R/PlaneStrain/Quad",
        "sqcni_ru": "GradientEnhancedFiniteStrainSQCNI_RU/PlaneStrain/Quad",
        "sqcni_rxnsni": "GradientEnhancedFiniteStrainSQCNI_RxNSNI/PlaneStrain/Quad",
        "sqcni_ruxnsni": "GradientEnhancedFiniteStrainSQCNI_RUxNSNI/PlaneStrain/Quad",
        "point": "GradientEnhancedFiniteStrain/PlaneStrain/Point",
    }
    pName = PARTICLES[particle]
    quad = particle != "point"

    theApproximation = MarmotMeshfreeApproximationWrapper(
        "ReproducingKernelImplicitGradient", dimension, completenessOrder=1
    )
    card = {
        "material": "GRADIENTENHANCEDORTHOCDPFINITESTRAIN",
        "properties": materialProperties(beddingDeg, strengthScale=strengthScale),
    }

    theModel = generatePerturbedQuadGrid(
        theModel, journal,
        lambda number, verts, volume: MarmotParticleWrapper(
            pName, number, verts, volume, theApproximation, card
        ),
        lambda node: MarmotMeshfreeKernelFunctionWrapper(
            node, "BSplineBoxed", supportRadius=supportRadius, continuityOrder=2
        ),
        LENGTH, nX, perturb, np.random.default_rng(seed),
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
    # THREE WAYS TO IMPOSE THE ESSENTIAL CONDITION, and the choice decides whether the problem
    # posed is the one the patch test assumes.
    #
    # "center" -- the field at the boundary particle CENTRES.  A centre lies half a cell inside
    #     the patch, so the surface integral of the weak form over the outer faces is still
    #     traction FREE: the body has a free outer strip behind a displacement condition, and
    #     its solution is not the linear field.  Measured 2.5 % error in u, insensitive to the
    #     particle type, to VCI, and almost to the spacing -- an inconsistent problem, not a
    #     discretisation defect.  Kept as a switch because it is the instructive failure.
    #
    # "cwf" -- the same centre constraints PLUS the consistent-weak-form correction on the
    #     Dirichlet boundary.  This is the framework's own answer: `cwfcorrection` subtracts
    #     the particle's own traction S.n dA, with n from Nanson's formula on the CURRENT
    #     smoothing-domain face, from the external force of every kernel function reaching that
    #     face -- i.e. it supplies exactly the surface term that integration by parts left
    #     behind, evaluated consistently with the internal stress instead of being dropped.
    #
    # "face" -- the field at the boundary FACE centres.  The multiplier then IS the reaction of
    #     that face, which is what a one-point integration of the true traction over the face
    #     delivers, so no correction is needed.
    faceOf = {"bottom": 1, "right": 2, "top": 3, "left": 4}
    constraints = []
    if bc == "face":
        for side, faceID in faceOf.items():
            for p in theModel.particleSets[side]:
                xy = np.asarray(p.getFaceCoordinates(faceID)).reshape(-1)[:2]
                u = uOf(xy.reshape(1, 2))[0]
                constraints.append(
                    ParticleLagrangianWeakDirichlet(
                        f"bc_{side}_{p.number}", p, "displacement",
                        {0: float(u[0]), 1: float(u[1])}, theModel,
                        location="face", faceID=faceID,
                    )
                )
    else:
        for p in theModel.particleSets[f"band{nRings}"]:
            xy = np.asarray(p.getCenterCoordinates()).reshape(2)
            u = uOf(xy.reshape(1, 2))[0]
            constraints.append(
                ParticleLagrangianWeakDirichlet(
                    f"bc_{p.number}", p, "displacement",
                    {0: float(u[0]), 1: float(u[1])}, theModel, location="center",
                )
            )

    distributedLoads = []
    if bc == "cwf":
        theModel.surfaces["dirichlet"] = EntityBasedSurface(
            "dirichlet",
            {faceID: list(theModel.particleSets[side]) for side, faceID in faceOf.items()},
        )
        distributedLoads.append(
            ParticleDistributedLoad(
                name="cwf_dirichlet", model=theModel, journal=journal,
                particleSurface=theModel.surfaces["dirichlet"],
                distributedLoadType="cwfcorrection",
                loadVector=np.array([0.0]), f_t=cwfRamp,
            )
        )
    for c in constraints:
        theModel.constraints[c.name] = c

    theModel.prepareYourself(journal)

    fieldOutputController = MPMFieldOutputController(theModel, journal)
    for name in ("displacement", "deformation gradient", "alphaP", "stress", "Fp"):
        fieldOutputController.addPerParticleFieldOutput(
            name, theModel.particleSets["all_particles"], name
        )
    if quad:
        fieldOutputController.addPerParticleFieldOutput(
            "vertex displacements", theModel.particleSets["all_particles"],
            "vertex displacements",
            f_x=lambda x: np.pad(np.reshape(x, (-1, 2)), ((0, 0), (0, 1)), mode="constant",
                                 constant_values=0),
        )
    fieldOutputController.initializeJob()

    iterationOptions = {
        "max. iterations": 25,
        "critical iterations": 8,
        "allowed residual growths": 6,
    }
    if tolerance is not None:
        # The measured error of a patch test is not a discretisation error -- the exact field
        # solves the discrete equations -- so it is set by how far the last Newton step
        # happened to go past the convergence tolerance.  Tightening the tolerance is how that
        # is demonstrated rather than asserted.
        iterationOptions.update({
            "default relative flux residual tolerance": tolerance,
            "default relative field correction tolerance": tolerance,
            "default absolute flux residual tolerance": 1e-16,
            "default absolute field correction tolerance": 1e-16,
        })
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
    verts0 = (np.array([np.asarray(p.getVertexCoordinates()).reshape(-1, 2)
                        for p in theModel.particleSets["all_particles"]])
              if quad else None)
    innerSet = set(theModel.particleSets[f"inner{nRings}"])
    isInterior = np.array([p in innerSet for p in theModel.particleSets["all_particles"]])

    failed = False
    try:
        nonlinearSolver.solveStep(
            AdaptiveTimeStepper(theModel.time, 1.0, 0.5, 1.0, 1e-4, 50, journal),
            linearSolver, theModel, fieldOutputController,
            outputManagers=[], particleManagers=[theParticleManager],
            constraints=constraints, userIterationOptions=iterationOptions,
            vciManagers=vciManagers, particleDistributedLoads=distributedLoads,
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
    FpNum = fo["Fp"].getLastResult().reshape(nP, 3, 3)

    vertsDef = (verts0 + fo["vertex displacements"].getLastResult().reshape(-1, 4, 3)[:, :, :2]
                if quad else None)
    # HOW NON-CONFORMING ARE THE DEFORMED DOMAINS?  Each domain is carried by the deformation
    # gradient at its OWN centre, so in general the images of a shared reference vertex do not
    # coincide and gaps open.  In a patch test they must coincide: the field is homogeneous, so
    # every centre sees the same F and the per-centre maps are the same affine map.  Measured
    # rather than asserted -- the spread of the images of each shared reference vertex.
    domainGap = None
    if quad:
        shared = {}
        for pp in range(verts0.shape[0]):
            for kk in range(4):
                k_ = (round(float(verts0[pp, kk, 0]), 9), round(float(verts0[pp, kk, 1]), 9))
                shared.setdefault(k_, []).append(vertsDef[pp, kk])
        spreads = [np.linalg.norm(np.asarray(im) - np.mean(im, axis=0), axis=1).max()
                   for im in shared.values() if len(im) > 1]
        domainGap = float(max(spreads)) if spreads else 0.0

    uEx = uOf(xy0)
    scale = np.abs(uEx).max()
    FEx = np.eye(3)
    FEx[:2, :2] += A

    inner = isInterior & np.isfinite(uNum).all(axis=1)
    errU = np.abs(uNum - uEx)[inner].max() / scale
    errF = np.abs(FNum - FEx)[inner].max() / np.abs(A).max()

    # THE ENERGY ERROR.  Psi^e is a function of the ELASTIC stretch, so once the run yields it
    # has to be evaluated on F^e = F Fp^-1 and not on F -- the material exports Fp, and using F
    # instead would measure the energy of a deformation the material never stored.
    # The exact field is homogeneous, so the exact solution has one Psi^e for the whole patch;
    # but with a return map in the loop that number is not known in closed form, so the measure
    # is the departure from the patch's own mean.  That is the same statement -- a homogeneous
    # problem must give a homogeneous answer -- and it is the only computable form of it.  It
    # is the quantity the weak form is stationary with respect to, and it weights the
    # components of the gradient error the way the material does rather than by the largest.
    Fe = np.array([FNum[i] @ np.linalg.inv(FpNum[i]) for i in range(nP)])
    psiNum = np.array([strainEnergyDensity(Fe[i], beddingDeg) for i in range(nP)])
    psiEx = float(psiNum[inner].mean()) if np.isfinite(psiNum).all() else float("nan")
    errEAbsField = np.abs(psiNum - psiEx)
    errEField = errEAbsField / abs(psiEx)
    errE = float(errEField[inner].max()) if np.isfinite(psiEx) else float("nan")

    return dict(
        bedding=beddingDeg, case=case, nX=nX, h=h, perturb=perturb, vci=vci,
        vciOrder=(vciOrder if vci else None), nRings=nRings, support=support,
        bc=bc, particle=particle,
        errU=float(errU), errF=float(errF), errE=errE,
        psiEx=float(psiEx), psiNum=psiNum, errEField=errEField,
        alphaPMax=float(np.abs(alphaP).max()),
        # the hardening variable per particle: for a homogeneous field it must be one number,
        # and how nearly it is one number is the plastic half of the patch test
        alphaPField=alphaP.reshape(-1),
        nInterior=int(inner.sum()), failed=failed,
        errUField=np.abs(uNum - uEx).max(axis=1) / scale,
        # the same two fields unnormalised, for the contours: mm and MPa
        errUAbsField=np.linalg.norm(uNum - uEx, axis=1),
        errEAbsField=errEAbsField,
        errFComp=np.abs(FNum - FEx)[inner].max(axis=0),
        stressMax=float(np.abs(fo["stress"].getLastResult()).max()),
        # The smoothing domains AS THE COMPUTATION LEFT THEM.  "vertex displacements" on the
        # SQCNI particle is `_vertexDisplacements_SmoothingDomain`, the state variable
        # updateSmoothingDomain() writes, so this is the domain the integration actually used
        # -- not the approximated displacement field sampled at the vertices, which would be
        # single valued and conforming by construction and would prove nothing.
        verts0=verts0,
        verts=vertsDef,
        domainGap=domainGap,
        xy0=xy0, uNum=uNum, uEx=uEx, interior=isInterior,
    )


# =============================================================================================
#  the sweeps
# =============================================================================================

BEDDINGS = [0, 15, 30, 45, 60, 75, 90]

# the smoothing-domain update variants, in the order they are reported
UPDATES = [
    ("sqcni", r"$\mathbf{F}$ (SQCNI)"),
    ("sqcni_ru", r"$\mathbf{R}^{\rm e}\mathbf{U}$ (SQCNI\_RU)"),
    ("sqcni_r", r"$\mathbf{R}^{\rm e}$ (SQCNI\_R)"),
    ("snni", r"frozen (SNNI)"),
]

PERTURBATIONS = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8]


def sweep(nX=8, perturb=0.4, cases=tuple(LOAD_CASES), beddings=tuple(BEDDINGS), seed=7,
          vci=True, vciOrder=1, nRings=1, support=2.5, bc="face",
          particle="sqcnixnsni", quiet=False):
    journal = Journal()
    out = []
    for case in cases:
        for b in beddings:
            r = run_patch(b, case=case, nX=nX, perturb=perturb, seed=seed, vci=vci,
                          vciOrder=vciOrder, nRings=nRings, support=support, bc=bc,
                          particle=particle, journal=journal)
            out.append(r)
            if not quiet:
                print(f"  {case:8s} beta = {b:5.1f} deg   err(u) = {r['errU']:.2e}   "
                      f"err(F) = {r['errF']:.2e}   domain gap = "
                      f"{r['domainGap']:.1e} mm   max alphaP = {r['alphaPMax']:.1e}"
                      f"{'   STEP FAILED' if r['failed'] else ''}")
    return out


def cellQuality(nX, perturb, seed=7):
    """Validity of the conforming quad tiling: the bilinear map's corner Jacobians.

    A quad smoothing domain is a valid integration cell only while the bilinear map is
    injective, i.e. while its Jacobian is positive at all four corners.  A positive shoelace
    AREA is not enough -- a quad can be concave and still have positive area, and that is what
    limits how far the lattice may be perturbed.
    """
    cells, _, _ = latticeForDrawing(nX, perturb, seed=seed)
    h2 = (LENGTH / nX) ** 2
    xi = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    jac, area = [], []
    for v in cells:
        js = []
        for a, b in xi:
            dN = 0.25 * np.array([[-(1 - b), -(1 - a)], [(1 - b), -(1 + a)],
                                  [(1 + b), (1 + a)], [-(1 + b), (1 - a)]])
            js.append(np.linalg.det(v.T @ dN))
        jac.append(js)
        x, y = v[:, 0], v[:, 1]
        area.append(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / h2)
    jac = np.asarray(jac)
    area = np.asarray(area)
    return dict(minJac=float(jac.min()), nConcave=int((jac.min(axis=1) <= 0).sum()),
                nCells=len(cells), areaSpread=float(area.max() / max(area.min(), 1e-12)),
                minArea=float(area.min()))


def sweepPerturbation(nX=8, case="affine", bedding=30.0, seed=7, support=2.5):
    """How random may the particle distribution be?  Sweep the lattice perturbation."""
    print("\n  HOW RANDOM MAY THE DISTRIBUTION BE -- lattice perturbation sweep")
    print(f"  {'perturb':>9s}{'err(u)':>11s}{'err(F)':>11s}{'min corner J':>14s}"
          f"{'concave':>9s}{'area max/min':>14s}")
    out = []
    for pert in PERTURBATIONS:
        q = cellQuality(nX, pert, seed=seed)
        r = run_patch(bedding, case=case, nX=nX, perturb=pert, seed=seed, support=support)
        r.update(q)
        out.append(r)
        print(f"  {pert:9.2f}{r['errU']:11.2e}{r['errF']:11.2e}{q['minJac']:14.4f}"
              f"{q['nConcave']:6d}/{q['nCells']:<3d}{q['areaSpread']:14.1f}")
    return out


def sweepUpdateType(nX=8, perturb=0.4, bedding=30.0, seed=7, support=2.5):
    """The smoothing-domain update, with and without the VCI correction, per load case.

    This is the physical source of non-conformity in this framework, and it is what the
    integration constraint is about: the constraint is a statement about the DEFORMED
    configuration, so a smoothing domain that is not carried there by the deformation gradient
    stops tiling the body it is supposed to integrate over.
    """
    print("\n  THE SMOOTHING-DOMAIN UPDATE -- err(u), VCI on / off")
    print(f"  {'update':>22s}" + "".join(f"{c:>24s}" for c in LOAD_CASES))
    out = []
    for pName, _ in UPDATES:
        row = []
        for case in LOAD_CASES:
            for vci in (True, False):
                r = run_patch(bedding, case=case, nX=nX, perturb=perturb, seed=seed,
                              support=support, particle=pName, vci=vci)
                r["update"] = pName
                out.append(r)
                row.append(r["errU"])
        print(f"  {pName:>22s}" + "".join(f"{row[2 * k]:11.1e} /{row[2 * k + 1]:10.1e}"
                                          for k in range(len(LOAD_CASES))))
    return out


def sweepBoundary(nX=8, perturb=0.4, bedding=30.0, seed=7, support=2.5, case="affine"):
    """The three ways of imposing the essential condition, including the CWF correction."""
    print("\n  THE ESSENTIAL CONDITION -- three boundary treatments")
    out = []
    for bc, what in (("face", "multiplier at the boundary FACE centres"),
                     ("center", "multiplier at the boundary PARTICLE centres"),
                     ("cwf", "particle centres + consistent-weak-form correction")):
        r = run_patch(bedding, case=case, nX=nX, perturb=perturb, seed=seed,
                      support=support, bc=bc, strengthScale=1.0)
        r["bcWhat"] = what
        out.append(r)
        print(f"  {bc:>8s}  err(u) = {r['errU']:.2e}  err(F) = {r['errF']:.2e}   {what}")
    return out


def report(results, tag=""):
    worstU = max(r["errU"] for r in results)
    worstF = max(r["errF"] for r in results)
    worstA = max(r["alphaPMax"] for r in results)
    nFail = sum(r["failed"] for r in results)
    print("-" * 92)
    print(f"  {tag}worst over the sweep:  err(u) = {worstU:.2e}   err(F) = {worstF:.2e}   "
          f"max alphaP = {worstA:.1e}   failed steps: {nFail}")
    verdict = "PASS" if (worstU < 1e-8 and worstF < 1e-8 and worstA == 0.0 and nFail == 0) else "FAIL"
    print(f"  {verdict}")
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


def latticeForDrawing(nX, perturb, seed=7):
    """Rebuild, for drawing and for the cell-quality check, the lattice a run used."""
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
    cells, isBnd = [], []
    for i in range(nX):
        for j in range(nX):
            cells.append(np.asarray([V[i, j], V[i + 1, j], V[i + 1, j + 1], V[i, j + 1]]))
            isBnd.append(i in (0, nX - 1) or j in (0, nX - 1))
    return cells, V, np.asarray(isBnd)


DEFORMED_AMPLITUDE = 0.10  # the amplitude panel (b) is computed at, so that it is visible
DEFORMED_CASE = "mixed"    # and the load case it draws: the one coaxial with neither the
                           # coordinate axes nor the material axes, so it cannot be misread


ARROW_OFFSET = 0.35   # mm; how far outside the boundary the boundary-condition arrows start


def fieldHeader(A, cOff, sep=r",\;\;"):
    """The imposed field as one typeset line: u = A X, or u = c + A X when there is a c."""
    mat = (r"\mathbf{A}=\begin{bmatrix}%+.3f & %+.3f\\ %+.3f & %+.3f\end{bmatrix}"
           % (A[0, 0], A[0, 1], A[1, 0], A[1, 1]))
    if np.abs(cOff).max() > 0.0:
        return (r"$u=\mathbf{c}+\mathbf{A}\mathbf{X}" + sep
                + r"\mathbf{c}=(%.2f,\,%.2f)^{\mathsf{T}}\,\mathrm{mm}" % (cOff[0], cOff[1])
                + sep + mat + "$")
    return r"$u=\mathbf{A}\mathbf{X},\quad" + mat + "$"


def drawImposedDeformation(ax, r, A, cOff, fc, FS, matrixAt=0.99, header=True,
                           atDeformed=False, gain=1.0):
    """The deformed patch as computed, the reference outline, and the imposed field on it.

    `r` is a run_patch result, `A` and `cOff` the field it was run with, u = c + A X, and `fc`
    the boundary face centres of the same lattice (what panel (a) marks).  Three things go on
    the axes and each answers a question the drawing would otherwise leave open:

      * the smoothing domains AS THE COMPUTATION LEFT THEM -- the vertex displacements the
        particles carry, not a re-drawn affine map.  That they still tile is the result: each
        is carried by the deformation gradient at its OWN centre, and for a homogeneous field
        those gradients coincide;
      * the reference outline, dashed, so the deformation is readable;
      * an arrow at every constrained face centre, from that centre to its imposed image, AT
        TRUE SCALE.  Both components are prescribed on every face, tangential as well as
        normal, so no part of the deformed shape is a material response.  The tails are set
        `ARROW_OFFSET` outside the boundary PLUS the outward part of the displacement itself,
        which is what keeps them off a face that moves outwards.
      * the matrix, which is the complete statement of what was imposed.
    """
    from matplotlib.collections import PolyCollection

    # `gain` amplifies the displacement for the drawing.  The patch test is run at the
    # amplitude the material allows -- 2 % here, because the strengths are real and the run has
    # to stay in the hardening branch -- and at 2 % the deformed patch is the reference patch
    # to the eye.  Everything drawn is the computed field, scaled uniformly; the panel says so.
    vDraw = r["verts0"] + gain * (r["verts"] - r["verts0"])
    ax.add_collection(PolyCollection(list(vDraw), facecolors="#eef3f8",
                                     edgecolors="#1b6ca8", linewidths=0.6 * FS))
    ax.plot([0, LENGTH, LENGTH, 0, 0], [0, 0, LENGTH, LENGTH, 0], "--",
            color="0.45", lw=0.7 * FS, zorder=4)
    for q0 in fc:
        du = gain * (A @ q0 + cOff)
        if np.hypot(*du) < 0.02:      # the field can vanish; a 20 um arrow is a blob
            continue
        if atDeformed:
            # the arrow ENDS on the constrained point where it now is: tail at the reference
            # face centre, head at its image, and a dot on the head.  That is the multiplier's
            # own position in the deformed configuration, and the same point SQCNI evaluates
            # the shape functions at (panel (c)).
            ax.annotate("", xy=tuple(q0 + du), xytext=tuple(q0), zorder=5,
                        arrowprops=dict(arrowstyle="-|>", color="#b1500f", lw=0.6 * FS,
                                        shrinkA=0.0, shrinkB=0.0, mutation_scale=5.0 * FS))
            ax.plot([q0[0] + du[0]], [q0[1] + du[1]], "o", color="#b1500f", ms=2.4 * FS,
                    zorder=6)
            continue
        n = np.zeros(2)
        n[0] = -1.0 if q0[0] < 1e-9 else (1.0 if q0[0] > LENGTH - 1e-9 else 0.0)
        n[1] = -1.0 if q0[1] < 1e-9 else (1.0 if q0[1] > LENGTH - 1e-9 else 0.0)
        n /= max(np.linalg.norm(n), 1.0)
        t = q0 + n * (ARROW_OFFSET + max(0.0, float(n @ du)))
        ax.annotate("", xy=tuple(t + du), xytext=tuple(t), zorder=5,
                    arrowprops=dict(arrowstyle="-|>", color="#b1500f", lw=0.6 * FS,
                                    shrinkA=0.0, shrinkB=0.0, mutation_scale=5.0 * FS))
    if header:
        ax.text(0.5, matrixAt, fieldHeader(A, cOff), transform=ax.transAxes,
                color="#b1500f", ha="center", va="top", fontsize=7.2 * FS)
    # the frame follows whichever configuration reaches furthest, plus the arrows, plus the
    # strip the header needs -- so the same call works for a patch that grows and one that does
    # not
    xs = np.concatenate([vDraw[:, :, 0].reshape(-1), [LENGTH]])
    ys = np.concatenate([vDraw[:, :, 1].reshape(-1), [LENGTH]])
    ax.set_xlim(-0.6, max(LENGTH, xs.max()) + 2.0 * ARROW_OFFSET + 0.6)
    ax.set_ylim(-0.6, max(LENGTH, ys.max()) + 2.0 * ARROW_OFFSET + (2.6 if header else 0.6))
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$ in mm")
    ax.set_ylabel(r"$x_2$ in mm")


def drawReferencePatch(ax, nX, perturb, FS, legendLabel=r"$u=\mathbf{A}\mathbf{X}$ imposed"):
    """The undeformed patch: smoothing domains, particles, and the constrained face centres.

    The boundary ring is shaded because the error measures exclude it, and the orange points
    are the face centres of the boundary smoothing domains -- where the Lagrange multipliers of
    Sec. 6.1 act, and, not by coincidence, the points at which SQCNI evaluates the shape
    functions for its boundary integration (see drawSqcniStencil).  Returns
    (cells, isBnd, faceCentres) so a caller can reuse the same lattice.
    """
    from matplotlib.collections import PolyCollection
    from matplotlib.lines import Line2D

    cells, _, isBnd = latticeForDrawing(nX, perturb)
    ax.add_collection(PolyCollection([cells[k] for k in np.where(~isBnd)[0]],
                                     facecolors="#eef3f8", edgecolors="0.55",
                                     linewidths=0.5 * FS))
    ax.add_collection(PolyCollection([cells[k] for k in np.where(isBnd)[0]],
                                     facecolors="#f7e2d3", edgecolors="0.55",
                                     linewidths=0.5 * FS))
    cen = np.array([c.mean(axis=0) for c in cells])
    ax.plot(cen[:, 0], cen[:, 1], ".", color="0.2", ms=2.4 * FS)
    fc = []
    for c, b in zip(cells, isBnd):
        if not b:
            continue
        for k in range(4):
            mid = 0.5 * (c[k] + c[(k + 1) % 4])
            if min(mid[0], mid[1]) < 1e-9 or max(mid[0], mid[1]) > LENGTH - 1e-9:
                fc.append(mid)
    fc = np.asarray(fc)
    ax.plot(fc[:, 0], fc[:, 1], "o", color="#b1500f", ms=2.8 * FS)
    ax.set_xlim(-0.6, LENGTH + 0.6)
    ax.set_ylim(-0.6, LENGTH + 0.6)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$X_1$ in mm")
    ax.set_ylabel(r"$X_2$ in mm")
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="none", color="#b1500f", ms=2.8 * FS, label=legendLabel),
        Line2D([], [], marker=".", ls="none", color="0.2", ms=2.4 * FS, label="particle"),
    ], loc="upper center", ncol=2, fontsize=7.0 * FS, frameon=True, framealpha=0.92)
    return cells, isBnd, fc


def drawSqcniStencil(ax, nX, perturb, FS, support=2.5, which=None):
    """What one particle IS, in SQCNI -- the panel the patch test needs and never had.

    Stabilized quasi-conforming nodal integration puts ONE integration point per particle, at
    the centre of its smoothing domain, and builds the smoothed gradient there by integrating
    over the domain BOUNDARY: the shape functions are evaluated at the four face centres and
    contracted with the face's n dA (Marmot, GradientEnhancedFiniteStrainParticleSQCNI, the
    one-point rule per face).  NSNI adds the second derivatives from the same boundary
    integration, contracted with the second moments of the domain about that centre.

    So the picture is: the domain, its four face centres with their outward n dA, the particle
    at the centre, and the kernel support that decides which particles enter the sum -- the
    normalised support s_hat = 2.5, which is where the support-size discussion of Sec. 6.1
    lands.  The same face centres carry the Dirichlet multipliers on the boundary, which is why
    imposing the field there is what makes the patch test pass.
    """
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Circle
    from matplotlib.lines import Line2D

    cells, _, isBnd = latticeForDrawing(nX, perturb)
    cen = np.array([c.mean(axis=0) for c in cells])
    if which is None:                      # the interior particle closest to the centre
        which = int(np.argmin(np.linalg.norm(cen - 0.5 * LENGTH, axis=1)))
    h = LENGTH / nX
    R = support * h
    c0 = cen[which]

    inSupport = np.linalg.norm(cen - c0, axis=1) <= R + 1e-12
    ax.add_collection(PolyCollection(cells, facecolors="none", edgecolors="0.72",
                                     linewidths=0.45 * FS))
    ax.add_collection(PolyCollection([cells[k] for k in np.where(inSupport)[0]],
                                     facecolors="#eef3f8", edgecolors="0.72",
                                     linewidths=0.45 * FS))
    ax.add_collection(PolyCollection([cells[which]], facecolors="#dce8f2",
                                     edgecolors="#1b6ca8", linewidths=1.0 * FS, zorder=3))
    ax.add_patch(Circle(c0, R, fill=False, ls=":", ec="#2e8b57", lw=1.0 * FS, zorder=4))
    ax.plot(cen[inSupport, 0], cen[inSupport, 1], ".", color="0.2", ms=2.4 * FS, zorder=5)
    ax.plot(cen[~inSupport, 0], cen[~inSupport, 1], ".", color="0.72", ms=2.0 * FS)
    # the four face centres of THIS domain, with the outward n dA that multiplies them
    quad = cells[which]
    for k in range(4):
        p1, p2 = quad[k], quad[(k + 1) % 4]
        mid = 0.5 * (p1 + p2)
        e = p2 - p1
        n = np.array([e[1], -e[0]])                       # outward for a ccw quad
        if float(n @ (mid - c0)) < 0.0:
            n = -n
        ax.annotate("", xy=tuple(mid + 0.75 * n), xytext=tuple(mid), zorder=6,
                    arrowprops=dict(arrowstyle="-|>", color="#b1500f", lw=0.8 * FS,
                                    shrinkA=0.0, shrinkB=0.0, mutation_scale=6.0 * FS))
        ax.plot([mid[0]], [mid[1]], "o", color="#b1500f", ms=2.8 * FS, zorder=7)
    ax.plot([c0[0]], [c0[1]], "s", color="#1b6ca8", ms=3.4 * FS, zorder=7)
    ax.set_aspect("equal")
    ax.set_xlim(c0[0] - R - 0.7, c0[0] + R + 0.7)
    ax.set_ylim(c0[1] - R - 0.7, c0[1] + R + 3.4)
    ax.set_xlabel(r"$X_1$ in mm")
    ax.set_ylabel(r"$X_2$ in mm")
    ax.legend(handles=[
        Line2D([], [], marker="s", ls="none", color="#1b6ca8", ms=3.0 * FS,
               label="integration point"),
        Line2D([], [], marker="o", ls="none", color="#b1500f", ms=2.6 * FS,
               label=r"face centre, $\mathbf{n}\,\mathrm{d}A$"),
        Line2D([], [], ls=":", color="#2e8b57", lw=1.0 * FS,
               label=rf"support $\hat s={support:g}\,h_p$"),
    ], loc="upper center", ncol=1, fontsize=6.6 * FS, frameon=True, framealpha=0.94)
    return which, R


# =============================================================================================
#  the affine study: u = c + A X, the displacement error and the energy error
# =============================================================================================

AFFINE_CONTOUR_BEDDING = 30.0


def sweepAffine(nX=8, perturb=0.4, seed=7, support=2.5, beddings=tuple(BEDDINGS)):
    """The patch test over the bedding sweep, with the strengths at their real values."""
    print("\n  THE PATCH TEST, u = c + A X, OVER THE BEDDING ORIENTATION")
    A, cOff = caseField("affine")
    print(f"    c = ({cOff[0]:.2f}, {cOff[1]:.2f}) mm,  A = [[{A[0,0]:.3f}, {A[0,1]:.3f}], "
          f"[{A[1,0]:.3f}, {A[1,1]:.3f}]],  det F = {np.linalg.det(np.eye(2) + A):.4f},  "
          f"strengths at 1x (f_cy = {FCY/STRENGTH_SCALE:.1f}, f_cu = {FCU/STRENGTH_SCALE:.1f} MPa)")
    journal = Journal()
    out = []
    for b in beddings:
        r = run_patch(float(b), case="affine", nX=nX, perturb=perturb, seed=seed,
                      support=support, strengthScale=1.0, journal=journal)
        ap = r["alphaPField"][r["interior"]]
        r["alphaPMean"] = float(ap.mean())
        r["alphaPSpread"] = float((ap.max() - ap.min()) / max(ap.mean(), 1e-30))
        out.append(r)
        print(f"    beta = {b:5.1f} deg   err(u) = {r['errU']:.2e}   err(F) = {r['errF']:.2e}"
              f"   err(E) = {r['errE']:.2e}   alphaP = {r['alphaPMean']:.4f}"
              f" (spread {r['alphaPSpread']:.1e})   max|stress| = {r['stressMax']:.2f} MPa")
    aP = [r["alphaPMean"] for r in out]
    print(f"    worst: err(u) = {max(r['errU'] for r in out):.2e}   "
          f"err(F) = {max(r['errF'] for r in out):.2e}   "
          f"err(E) = {max(r['errE'] for r in out):.2e}   "
          f"alphaP spread = {max(r['alphaPSpread'] for r in out):.1e}")
    print(f"    alphaP over the sweep: {min(aP):.3f} to {max(aP):.3f}, a factor "
          f"{max(aP)/max(min(aP), 1e-30):.1f}; stress "
          f"{min(r['stressMax'] for r in out):.1f} to "
          f"{max(r['stressMax'] for r in out):.1f} MPa")
    return out


def makeAffineFigure(results, out=None, nX=8, perturb=0.4):
    """Six panels: what is set up, what is imposed, what a particle is -- then the errors.

    Row 1 is the test itself: (a) the undeformed patch, as Fig. 14(a); (b) the same patch
    deformed, with the imposed displacement drawn AT the points it is applied to, which in the
    deformed configuration is where those face centres have moved to; (c) one particle's SQCNI
    stencil, which is what makes those points the right ones.
    Row 2 is the result: (d), (e) the two errors resolved over the body, (f) the sweep.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.collections import PolyCollection

    figW = 13.4
    FS = paperStyle(figW)
    fig, axes = plt.subplots(2, 3, figsize=(figW, 8.2))
    (a, b, cc), (d, e, f) = axes

    A, cOff = caseField("affine")
    rC = [r for r in results if abs(r["bedding"] - AFFINE_CONTOUR_BEDDING) < 1e-9]
    rC = rC[0] if rC else results[0]

    # ------------------------------------------------------------------- (a) the setup
    cells, isBnd, fc = drawReferencePatch(a, nX, perturb, FS,
                                          legendLabel=r"$u=\mathbf{c}+\mathbf{A}\mathbf{X}$")
    a.set_title(rf"(a) reference, {nX}$\times${nX} particles", fontsize=9.5 * FS)

    # ------------------------------------------------------- (b) the field where it is applied
    GAIN = 10.0
    drawImposedDeformation(b, rC, A, cOff, fc, FS, header=False, atDeformed=True, gain=GAIN)
    b.set_title(rf"(b) imposed at the face centres, $\times{GAIN:.0f}$", fontsize=9.5 * FS)

    # ------------------------------------------------------------------ (c) what a particle is
    which, R = drawSqcniStencil(cc, nX, perturb, FS, support=rC["support"])
    cc.set_title("(c) the SQCNI stencil of one particle", fontsize=9.5 * FS)

    # ------------------------------------------------------------------- (d), (e) the contours
    xy0, interior = rC["xy0"], rC["interior"]
    xyD = xy0 + rC["uNum"]
    vD, v0 = rC["verts"], rC["verts0"]
    # The field is carried over the whole body, not just to the outermost particle centres: the
    # particles carry one value each, so the smoothing-domain corners are added as nodes, with
    # the mean of the particles sharing them.  That is the usual nodal averaging of a cell-wise
    # field and the only interpolation in this figure.
    corners = {}
    for pp in range(v0.shape[0]):
        for kk in range(4):
            k_ = (round(float(v0[pp, kk, 0]), 9), round(float(v0[pp, kk, 1]), 9))
            corners.setdefault(k_, []).append((vD[pp, kk], pp))
    cornerXY = np.array([np.mean([q for q, _ in im], axis=0) for im in corners.values()])
    cornerOf = [[i for _, i in im] for im in corners.values()]
    # the SAME two measures panel (f) plots, so the largest value of each field over the
    # interior particles IS the point (f) shows at this orientation
    for ax, field, title, cmap in (
            (d, rC["errUField"],
             r"(d) $\max_i|u_i-u_{{\rm ex},i}|/\max|u_{\rm ex}|$", "Blues"),
            (e, rC["errEField"],
             r"(e) $|\Psi^{\rm e}-\Psi^{\rm e}_{\rm ex}|/\Psi^{\rm e}_{\rm ex}$",
             "Greens")):
        field = np.asarray(field)
        vals = np.concatenate([field, [field[ii].mean() for ii in cornerOf]])
        pts = np.vstack([xyD, cornerXY])
        tri = mtri.Triangulation(pts[:, 0], pts[:, 1])
        expo = int(math.floor(math.log10(max(vals.max(), 1e-300))))
        cf = ax.tricontourf(tri, vals / 10.0 ** expo, levels=12, cmap=cmap)
        ax.add_collection(PolyCollection(list(vD), facecolors="none", edgecolors="0.45",
                                         linewidths=0.35 * FS, alpha=0.75, zorder=3))
        ax.plot([0, LENGTH, LENGTH, 0, 0], [0, 0, LENGTH, LENGTH, 0], "--",
                color="0.55", lw=0.5 * FS, zorder=2)
        ax.plot(xyD[interior, 0], xyD[interior, 1], ".", color="0.15", ms=2.2 * FS, zorder=4)
        ax.plot(xyD[~interior, 0], xyD[~interior, 1], "o", mfc="none", mec="0.15",
                ms=2.6 * FS, mew=0.5 * FS, zorder=4)
        cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=7.0 * FS)
        cb.set_label(rf"$\times 10^{{{expo}}}$", fontsize=7.4 * FS)
        ax.set_aspect("equal")
        pad = 0.3
        ax.set_xlim(min(0.0, vD[:, :, 0].min()) - pad, max(LENGTH, vD[:, :, 0].max()) + pad)
        ax.set_ylim(min(0.0, vD[:, :, 1].min()) - pad, max(LENGTH, vD[:, :, 1].max()) + pad)
        ax.set_xlabel(r"$x_1$ in mm")
        ax.set_ylabel(r"$x_2$ in mm")
        ax.set_title(title + rf" at $\beta={rC['bedding']:.0f}^\circ$", fontsize=8.6 * FS)

    # -------------------------------------------------------------------- (f) over the sweep
    # The three measures and, on the right axis, the hardening variable: the errors say the
    # field is reproduced, alphaP says the run was plastic and that the sweep varies the
    # physics -- both of which an elastic patch test leaves open.
    from matplotlib.lines import Line2D
    MEASURES = (("errU", r"$\mathrm{err}(u)$", "#1b6ca8", "o"),
                ("errF", r"$\mathrm{err}(F)$", "#e8871a", "s"),
                ("errE", r"$\mathrm{err}(\Psi^{\rm e})$", "#2e8b57", "^"))
    rr = sorted(results, key=lambda q: q["bedding"])
    bs = [q["bedding"] for q in rr]
    for key, lab, col, mk in MEASURES:
        f.semilogy(bs, [max(q[key], 1e-17) for q in rr], "-", marker=mk, color=col,
                   ms=3.0 * FS, lw=1.1 * FS)
    f.set_xticks(BEDDINGS)
    f.set_ylim(1e-16, 1e-9)      # headroom for the legend, which the curves leave empty
    f.set_xlabel(r"bedding orientation $\beta$ in deg")
    f.set_ylabel("relative error, interior particles")
    f.set_title("(f) the errors and the plastic strain", fontsize=9.5 * FS)
    f.grid(True, which="major", color="#DDDDDD", lw=0.4 * FS)
    handles = [Line2D([], [], color=col, marker=mk, ms=3.0 * FS, lw=1.1 * FS, label=lab)
               for _, lab, col, mk in MEASURES]
    g = f.twinx()
    g.plot(bs, [q.get("alphaPMean", q["alphaPMax"]) for q in rr], ":", color="0.35",
           lw=1.2 * FS, marker="v", ms=3.0 * FS)
    g.set_ylabel(r"$\alpha_{\rm p}$", color="0.35")
    g.tick_params(axis="y", labelcolor="0.35")
    g.set_ylim(0.0, 1.35 * max(q.get("alphaPMean", q["alphaPMax"]) for q in rr))
    handles.append(Line2D([], [], color="0.35", ls=":", marker="v", ms=3.0 * FS,
                          lw=1.2 * FS, label=r"$\alpha_{\rm p}$"))
    f.legend(handles=handles, loc="upper center", ncol=4, fontsize=7.0 * FS, frameon=False,
             columnspacing=0.9, handlelength=1.6)

    print(f"  contours at beta = {rC['bedding']:.0f} deg, relative, and their interior maxima "
          f"against panel (f): err(u) {rC['errUField'][interior].max():.2e} vs "
          f"{rC['errU']:.2e}, err(E) {rC['errEField'][interior].max():.2e} vs "
          f"{rC['errE']:.2e}")
    print(f"             the same two absolutely: |u - u_ex| max "
          f"{rC['errUAbsField'].max():.2e} mm, |Psi - Psi_ex| max "
          f"{rC['errEAbsField'].max():.2e} MPa on Psi_ex = {rC['psiEx']:.4f} MPa")
    print(f"             panel (c): particle {which}, support radius {R:.3f} mm")
    fig.tight_layout(pad=0.5, w_pad=0.9, h_pad=1.0)
    out = out or os.path.join(HERE, "fig_patch_affine.pdf")
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
    ap.add_argument("--particle", default="sqcnixnsni")
    ap.add_argument("--bc", default="face", choices=("face", "center", "cwf"))
    ap.add_argument("--case", default=None, choices=list(LOAD_CASES) + list(EXTRA_CASES))
    ap.add_argument("--bedding", type=float, default=None)
    ap.add_argument("--all", action="store_true",
                    help="the orientation sweep plus the boundary, perturbation and "
                         "smoothing-domain studies")
    ap.add_argument("--refine", action="store_true")
    ap.add_argument("--figure", action="store_true",
                    help="write fig_patch_affine.pdf, the paper figure")
    ap.add_argument("--affine", action="store_true",
                    help="the affine study u = c + A X, with the energy error, and "
                         "fig_patch_affine.pdf")
    args = ap.parse_args()

    cases = (args.case,) if args.case else tuple(LOAD_CASES)
    beddings = (args.bedding,) if args.bedding is not None else tuple(BEDDINGS)

    print("\n  THE PATCH TEST OVER THE BEDDING ORIENTATION")
    orientation = sweep(nX=args.nx, perturb=args.perturb, cases=cases, beddings=beddings,
                        seed=7, vci=not args.no_vci, support=args.support, bc=args.bc,
                        particle=args.particle)
    report(orientation)

    perturbation = updates = None
    if args.all or args.figure:
        sweepBoundary(nX=args.nx, perturb=args.perturb, support=args.support)
        perturbation = sweepPerturbation(nX=args.nx, support=args.support)
        updates = sweepUpdateType(nX=args.nx, perturb=args.perturb, support=args.support)

    if args.refine:
        print("\n  REFINEMENT at beta = 30 deg, mixed")
        for nX in (6, 8, 12, 16):
            for pName in ("sqcni", "snni"):
                r = run_patch(30.0, case="mixed", nX=nX, perturb=args.perturb,
                              support=args.support, particle=pName)
                print(f"    {pName:>8s}  nX = {nX:3d}  h = {r['h']:.3f}  "
                      f"err(u) = {r['errU']:.2e}  err(F) = {r['errF']:.2e}")

    if args.affine or args.figure:
        aff = sweepAffine(nX=args.nx, perturb=args.perturb, support=args.support)
        makeAffineFigure(aff, nX=args.nx, perturb=args.perturb)


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


def test_affine_patch():
    """u = c + A X with the strengths at their real values: the patch test the paper runs.

    The run must YIELD -- otherwise it is the elastic test under another name -- and the
    hardening variable must come out the same at every particle, which is the plastic half of
    the reproduction.  The energy error is only asserted if the paper's potential is
    importable; without the paper tree next to this repo `strainEnergyDensity` returns NaN.
    """
    r = run_patch(45.0, case="affine", nX=6, perturb=0.4, strengthScale=1.0)
    assert not r["failed"]
    assert r["errU"] < 1e-8
    assert r["errF"] < 1e-8
    assert math.isnan(r["errE"]) or r["errE"] < 1e-6
    ap = r["alphaPField"][r["interior"]]
    assert ap.min() > 0.0
    assert (ap.max() - ap.min()) / ap.mean() < 1e-8


if __name__ == "__main__":
    main()
