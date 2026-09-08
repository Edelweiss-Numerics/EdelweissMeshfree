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
elastic card and everything else is the reference card of examples/152.

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
    "affine": (np.array([[0.10, 0.20], [0.15, 0.10]]), np.array([0.10, 0.05])),
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
              journal=None):
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
    for name in ("displacement", "deformation gradient", "alphaP", "stress"):
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

    # THE ENERGY ERROR.  The exact field is homogeneous, so the exact stored energy density is
    # one number, and the departure of the per-particle energy from it is the error in the
    # quantity the weak form is stationary with respect to -- a scalar that weights the
    # components of the gradient error the way the material does, rather than by the largest
    # of them.  Relative to the stored energy itself, and reported the same way as the other
    # two, as the largest value over the interior.
    psiEx = strainEnergyDensity(FEx, beddingDeg)
    psiNum = np.array([strainEnergyDensity(FNum[i], beddingDeg) for i in range(nP)])
    errEField = np.abs(psiNum - psiEx) / abs(psiEx)
    errE = float(errEField[inner].max()) if np.isfinite(psiEx) else float("nan")

    return dict(
        bedding=beddingDeg, case=case, nX=nX, h=h, perturb=perturb, vci=vci,
        vciOrder=(vciOrder if vci else None), nRings=nRings, support=support,
        bc=bc, particle=particle,
        errU=float(errU), errF=float(errF), errE=errE,
        psiEx=float(psiEx), psiNum=psiNum, errEField=errEField,
        alphaPMax=float(np.abs(alphaP).max()),
        nInterior=int(inner.sum()), failed=failed,
        errUField=np.abs(uNum - uEx).max(axis=1) / scale,
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
        xy0=xy0, interior=isInterior,
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


def sweepPerturbation(nX=8, case="mixed", bedding=30.0, seed=7, support=2.5):
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


def sweepBoundary(nX=8, perturb=0.4, bedding=30.0, seed=7, support=2.5, case="mixed"):
    """The three ways of imposing the essential condition, including the CWF correction."""
    print("\n  THE ESSENTIAL CONDITION -- three boundary treatments")
    out = []
    for bc, what in (("face", "multiplier at the boundary FACE centres"),
                     ("center", "multiplier at the boundary PARTICLE centres"),
                     ("cwf", "particle centres + consistent-weak-form correction")):
        r = run_patch(bedding, case=case, nX=nX, perturb=perturb, seed=seed,
                      support=support, bc=bc)
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


def drawImposedDeformation(ax, r, A, cOff, fc, FS, matrixAt=0.99, header=True):
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

    ax.add_collection(PolyCollection(list(r["verts"]), facecolors="#eef3f8",
                                     edgecolors="#1b6ca8", linewidths=0.6 * FS))
    ax.plot([0, LENGTH, LENGTH, 0, 0], [0, 0, LENGTH, LENGTH, 0], "--",
            color="0.45", lw=0.7 * FS, zorder=4)
    for q0 in fc:
        du = A @ q0 + cOff
        if np.hypot(*du) < 0.02:      # the field can vanish; a 20 um arrow is a blob
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
    xs = np.concatenate([r["verts"][:, :, 0].reshape(-1), [LENGTH]])
    ys = np.concatenate([r["verts"][:, :, 1].reshape(-1), [LENGTH]])
    ax.set_xlim(-0.9, max(LENGTH, xs.max()) + 2.0 * ARROW_OFFSET + 0.9)
    ax.set_ylim(-0.9, max(LENGTH, ys.max()) + 2.0 * ARROW_OFFSET + (2.6 if header else 0.9))
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$ [mm]")
    ax.set_ylabel(r"$x_2$ [mm]")


def makeFigure(orientation, out=None, nX=8, perturb=0.4):
    """Three panels: the patch, the deformed patch as computed, and the error."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.lines import Line2D

    figW = 12.2
    FS = paperStyle(figW)
    fig, ax = plt.subplots(1, 3, figsize=(figW, 3.7))

    cols = {"stretch": "#1b6ca8", "shear": "#e8871a", "mixed": "#2e8b57"}
    marks = {"stretch": "o", "shear": "s", "mixed": "^"}
    FLOOR = 1e-17

    # ---------------------------------------------------------------- (a) the patch
    a = ax[0]
    cells, _, isBnd = latticeForDrawing(nX, perturb)
    a.add_collection(PolyCollection([cells[k] for k in np.where(~isBnd)[0]],
                                    facecolors="#eef3f8", edgecolors="0.55",
                                    linewidths=0.5 * FS))
    a.add_collection(PolyCollection([cells[k] for k in np.where(isBnd)[0]],
                                    facecolors="#f7e2d3", edgecolors="0.55",
                                    linewidths=0.5 * FS))
    cen = np.array([c.mean(axis=0) for c in cells])
    a.plot(cen[:, 0], cen[:, 1], ".", color="0.2", ms=2.4 * FS)
    fc = []
    for c, b in zip(cells, isBnd):
        if not b:
            continue
        for k in range(4):
            mid = 0.5 * (c[k] + c[(k + 1) % 4])
            if min(mid[0], mid[1]) < 1e-9 or max(mid[0], mid[1]) > LENGTH - 1e-9:
                fc.append(mid)
    fc = np.asarray(fc)
    a.plot(fc[:, 0], fc[:, 1], "o", color="#b1500f", ms=2.8 * FS)
    a.set_xlim(-0.6, LENGTH + 0.6)
    a.set_ylim(-0.6, LENGTH + 0.6)
    a.set_aspect("equal")
    a.set_xlabel(r"$X_1$ [mm]")
    a.set_ylabel(r"$X_2$ [mm]")
    a.set_title(rf"(a) reference, {nX}$\times${nX} particles", fontsize=9.5 * FS)
    a.legend(handles=[
        Line2D([], [], marker="o", ls="none", color="#b1500f", ms=2.8 * FS,
               label=r"$u=\mathbf{A}\mathbf{X}$ imposed"),
        Line2D([], [], marker=".", ls="none", color="0.2", ms=2.4 * FS, label="particle"),
    ], loc="upper center", ncol=2, fontsize=7.0 * FS, frameon=True, framealpha=0.92)

    # ---------------------------------------------------------------- (b) as computed
    # The smoothing domains AS THE COMPUTATION LEFT THEM -- the vertex displacements the
    # particles carry, not a re-drawn affine map -- at an amplitude large enough to see.
    # Two things are drawn on top of it, and both answer the reading that a deformed patch
    # with a contracted top face is a tension test whose boundary condition came loose:
    #   * the arrows are the BOUNDARY CONDITION at true scale, from each boundary face centre
    #     to its imposed image u = A X.  Both components are prescribed on all four sides,
    #     tangential as well as normal, so nothing here is a material response, and the fan
    #     out of the fixed origin is what a homogeneous gradient looks like;
    #   * the matrix is A itself, at the amplitude drawn, and the corner label is what it
    #     does to the far corner of the patch.
    # The domains still tile, which is the other point: each is carried by the deformation
    # gradient at its OWN centre, and for a homogeneous field those gradients coincide.
    b = ax[1]
    AB, cB = caseField(DEFORMED_CASE)
    AB, cB = AB * (DEFORMED_AMPLITUDE / AMPLITUDE), cB * (DEFORMED_AMPLITUDE / AMPLITUDE)
    r = run_patch(30.0, case=DEFORMED_CASE, nX=nX, perturb=perturb,
                  amplitude=DEFORMED_AMPLITUDE, journal=Journal())
    uL = AB @ np.array([LENGTH, LENGTH]) + cB
    drawImposedDeformation(b, r, AB, cB, fc, FS)
    b.set_title(rf"(b) as computed, {DEFORMED_CASE} at ${DEFORMED_AMPLITUDE*100:.0f}\,\%$",
                fontsize=9.5 * FS)
    print(f"  panel (b): {DEFORMED_CASE} at {DEFORMED_AMPLITUDE*100:.0f} %, imposed corner "
          f"displacement ({uL[0]:+.2f}, {uL[1]:+.2f}) mm, deformed smoothing domains "
          f"non-conforming by {r['domainGap']:.2e} mm over a {LENGTH:g} mm patch, "
          f"alphaP = {r['alphaPMax']:.1e}")
    # And what the same measurement gives when the field is NOT homogeneous, which no patch
    # test can show: the bending field of the text, same discretisation, per-centre gradients
    # that differ.  Printed rather than drawn -- it is the number Sec. 6.1 quotes.
    rB = run_patch(30.0, case="stretch", nX=nX, perturb=perturb,
                   fieldFun=lambda q: bendingDisplacement(q, kappa=BENDING_KAPPA),
                   journal=Journal())
    print(f"             companion, inhomogeneous (bending, kappa = {BENDING_KAPPA} 1/mm): "
          f"gap {rB['domainGap']*1e3:.0f} um = "
          f"{100 * rB['domainGap'] / rB['h']:.1f} % of h_p, "
          f"max|u| = {np.abs(rB['verts'] - rB['verts0']).max():.2f} mm, "
          f"alphaP = {rB['alphaPMax']:.1e}")

    # ---------------------------------------------------------------- (c) the error
    c = ax[2]
    for case in LOAD_CASES:
        rr = sorted([q for q in orientation if q["case"] == case], key=lambda q: q["bedding"])
        if not rr:
            continue
        c.semilogy([q["bedding"] for q in rr], [max(q["errF"], FLOOR) for q in rr],
                   "-", marker=marks[case], color=cols[case], ms=3.4 * FS, lw=1.1 * FS,
                   label=case)
    c.set_xticks(BEDDINGS)
    c.set_xlabel(r"bedding orientation $\beta$ [deg]")
    c.set_ylabel(r"error in $F_{iI}$, relative to $\|\mathbf{A}\|$")
    c.set_ylim(1e-15, 1e-10)
    c.set_title(r"(c) interior error, $\|\mathbf{A}\|=2\,\%$", fontsize=9.5 * FS)
    c.legend(loc="upper center", ncol=3, fontsize=7.6 * FS, frameon=False,
             columnspacing=1.1, handlelength=1.5)
    c.grid(True, which="major", color="#DDDDDD", lw=0.4 * FS)

    fig.tight_layout()
    out = out or os.path.join(HERE, "fig_patch_test.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=145)
    print(f"  wrote {out}")


# =============================================================================================
#  the affine study: u = c + A X, the displacement error and the energy error
# =============================================================================================

AFFINE_CONTOUR_BEDDING = 30.0


def sweepAffine(nX=8, perturb=0.4, seed=7, support=2.5, beddings=tuple(BEDDINGS)):
    """The affine case of EXTRA_CASES over the bedding sweep, with the energy error."""
    print("\n  THE AFFINE FIELD u = c + A X, OVER THE BEDDING ORIENTATION")
    A, cOff = caseField("affine")
    print(f"    c = ({cOff[0]:.2f}, {cOff[1]:.2f}) mm,  A = [[{A[0,0]:.2f}, {A[0,1]:.2f}], "
          f"[{A[1,0]:.2f}, {A[1,1]:.2f}]],  det F = {np.linalg.det(np.eye(2) + A):.4f}")
    journal = Journal()
    out = []
    for b in beddings:
        r = run_patch(float(b), case="affine", nX=nX, perturb=perturb, seed=seed,
                      support=support, journal=journal)
        out.append(r)
        print(f"    beta = {b:5.1f} deg   err(u) = {r['errU']:.2e}   err(F) = {r['errF']:.2e}"
              f"   err(E) = {r['errE']:.2e}   Psi = {r['psiEx']:.4f} MPa"
              f"   max alphaP = {r['alphaPMax']:.1e}")
    print(f"    worst: err(u) = {max(r['errU'] for r in out):.2e}   "
          f"err(F) = {max(r['errF'] for r in out):.2e}   "
          f"err(E) = {max(r['errE'] for r in out):.2e}")
    # The elastic card of this study is transversely isotropic about the OUT-OF-PLANE axis --
    # E1 = E2, nu13 = nu23, and the Saint Venant formula makes G12 = E1/2(1+nu12) exactly --
    # so an in-plane deformation stores the same energy at every bedding orientation, and the
    # Biot stress stays coaxial with U.  Psi above is printed at every beta to show it: the
    # sweep varies the material bookkeeping, not the physical problem, as long as the run is
    # elastic.  What breaks the orientation symmetry is the Walpole map of the yield surface,
    # which a patch test never reaches (alphaP = 0).
    return out


def makeAffineFigure(results, out=None, nX=8, perturb=0.4):
    """Four panels: the field, the two errors over the bedding sweep, and both as contours."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    figW = 9.2
    FS = paperStyle(figW)
    fig, axes = plt.subplots(2, 2, figsize=(figW, 7.2))
    (a, b), (c, d) = axes

    A, cOff = caseField("affine")
    cells, _, isBnd = latticeForDrawing(nX, perturb)
    fc = []
    for cell, onB in zip(cells, isBnd):
        if not onB:
            continue
        for k in range(4):
            mid = 0.5 * (cell[k] + cell[(k + 1) % 4])
            if min(mid[0], mid[1]) < 1e-9 or max(mid[0], mid[1]) > LENGTH - 1e-9:
                fc.append(mid)
    fc = np.asarray(fc)

    # ------------------------------------------------------------- (a) the field as imposed
    rC = [r for r in results if abs(r["bedding"] - AFFINE_CONTOUR_BEDDING) < 1e-9]
    rC = rC[0] if rC else results[0]
    drawImposedDeformation(a, rC, A, cOff, fc, FS, header=False)
    a.set_title("(a) the imposed field, as computed", fontsize=9.5 * FS)
    # the field itself over the whole figure, once: it is what all four panels are about, and
    # a 4.4 in panel is too narrow for the line
    fig.suptitle(fieldHeader(A, cOff), fontsize=9.0 * FS, color="#b1500f", y=0.998)

    # ------------------------------------------------------- (b) both errors over the sweep
    rr = sorted(results, key=lambda q: q["bedding"])
    bs = [q["bedding"] for q in rr]
    for key, lab, col, mk in (("errU", r"$\mathrm{err}(u)$", "#1b6ca8", "o"),
                              ("errF", r"$\mathrm{err}(F)$", "#e8871a", "s"),
                              ("errE", r"$\mathrm{err}(\Psi^{\rm e})$", "#2e8b57", "^")):
        b.semilogy(bs, [max(q[key], 1e-17) for q in rr], "-", marker=mk, color=col,
                   ms=3.4 * FS, lw=1.1 * FS, label=lab)
    b.set_xticks(BEDDINGS)
    b.set_ylim(1e-16, 1e-12)
    b.set_xlabel(r"bedding orientation $\beta$ [deg]")
    b.set_ylabel("relative error, interior particles")
    b.set_title(r"(b) displacement, gradient and energy error", fontsize=9.5 * FS)
    b.legend(loc="upper center", ncol=3, fontsize=7.6 * FS, frameon=False,
             columnspacing=1.1, handlelength=1.5)
    b.grid(True, which="major", color="#DDDDDD", lw=0.4 * FS)

    # ------------------------------------------------------------------- (c), (d) contours
    xy0, interior = rC["xy0"], rC["interior"]
    tri = mtri.Triangulation(xy0[:, 0], xy0[:, 1])
    for ax, field, title, cmap in (
            (c, rC["errUField"], r"(c) $|u-u_{\rm ex}|/\max|u_{\rm ex}|$", "Blues"),
            (d, rC["errEField"], r"(d) $|\Psi^{\rm e}-\Psi^{\rm e}_{\rm ex}|"
                                 r"/\Psi^{\rm e}_{\rm ex}$", "Greens")):
        f15 = np.asarray(field) * 1e15
        cf = ax.tricontourf(tri, f15, levels=12, cmap=cmap)
        ax.plot(xy0[interior, 0], xy0[interior, 1], ".", color="0.25", ms=2.2 * FS)
        ax.plot(xy0[~interior, 0], xy0[~interior, 1], "o", mfc="none", mec="0.25",
                ms=2.6 * FS, mew=0.5 * FS)
        ax.plot([0, LENGTH, LENGTH, 0, 0], [0, 0, LENGTH, LENGTH, 0], "-",
                color="0.35", lw=0.6 * FS)
        cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=7.0 * FS)
        cb.set_label(r"$\times 10^{-15}$", fontsize=7.4 * FS)
        ax.set_aspect("equal")
        ax.set_xlim(-0.4, LENGTH + 0.4)
        ax.set_ylim(-0.4, LENGTH + 0.4)
        ax.set_xlabel(r"$X_1$ [mm]")
        ax.set_ylabel(r"$X_2$ [mm]")
        ax.set_title(title + rf" at $\beta={rC['bedding']:.0f}^\circ$", fontsize=9.5 * FS)
    # no legend on the contours: the filled dots are the interior particles the maxima are
    # taken over, the open ones the constrained ring, and the caption says so rather than a
    # box over the field
    print(f"  contours at beta = {rC['bedding']:.0f} deg: "
          f"err(u) max {rC['errUField'].max():.2e} (interior "
          f"{rC['errUField'][interior].max():.2e}), "
          f"err(E) max {rC['errEField'].max():.2e} (interior "
          f"{rC['errEField'][interior].max():.2e}), Psi = {rC['psiEx']:.4f} MPa")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
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
    ap.add_argument("--figure", action="store_true", help="write fig_patch_test.pdf")
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

    if args.figure:
        makeFigure(orientation, nX=args.nx, perturb=args.perturb)

    if args.affine:
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
    """u = c + A X: the same guard on the case that carries a translation and 20 % strain.

    The energy error is only asserted if the paper's potential is importable -- without the
    paper tree next to this repo `strainEnergyDensity` returns NaN by design.
    """
    r = run_patch(45.0, case="affine", nX=6, perturb=0.4)
    assert not r["failed"]
    assert r["errU"] < 1e-8
    assert r["errF"] < 1e-8
    assert r["alphaPMax"] == 0.0
    assert math.isnan(r["errE"]) or r["errE"] < 1e-8


if __name__ == "__main__":
    main()
