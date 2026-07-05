"""Finite-difference verification of the DisplacementPressureJacobiSQCNIxNSNI
particle tangent dFInt/ddQ, for the pressure-only (vms mode 0) and the full
(vms mode 1) VMS stabilization.

Builds a small Cook's membrane particle grid (no solve), picks interior
particles, and compares computePhysicsKernels' K against central finite
differences of fInt. K is stored column-major (Eigen default).
"""

import sys

import numpy as np

sys.path.insert(0, "/home/tom/CMP_MEC_MAT/EdelweissMeshfree/examples/134_marmot_displacement_pressure_jacobi_cooks_membrane_test")

from edelweissfe.journal.journal import Journal

from edelweissmeshfree.generators.cooksmembranekernelfunctiongridgenerator import (
    generateCooksMembraneKernelFunctionGrid,
)
from edelweissmeshfree.generators.cooksmembranequadparticlegridgenerator import (
    generateCooksMembraneParticleGrid,
)
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

NX = NY = 5
VMS_ALPHA = 0.1


def build_model(vmsMode):
    journal = Journal()
    model = MPMModel(2)
    length, h0, h1 = 48.0, 44.0, 16.0
    supportRadius = 2.2 * length / NX

    def kernelFactory(node):
        return MarmotMeshfreeKernelFunctionWrapper(
            node, "BSplineBoxed", supportRadius=supportRadius, continuityOrder=2
        )

    model = generateCooksMembraneKernelFunctionGrid(
        model, journal, kernelFactory, x0=0, y0=0, l=length, h0=h0, h1=h1, nX=NX, nY=NY
    )

    approximation = MarmotMeshfreeApproximationWrapper("ReproducingKernel", 2, completenessOrder=1)
    material = {
        "material": "FiniteStrainJ2Plasticity",
        "properties": np.array([40000.0, 10.0, 1e8, 1e8, 1e0, 1e0, 1, 1e-7]),
    }

    def particleFactory(number, vertexCoordinates, volume):
        return MarmotParticleWrapper(
            "DisplacementPressureJacobiSQCNIxNSNI/PlaneStrain/Quad",
            number,
            vertexCoordinates,
            0.0,
            approximation,
            material,
        )

    model = generateCooksMembraneParticleGrid(
        model, journal, particleFactory, x0=0, y0=0, l=length, h0=h0, h1=h1, nX=NX, nY=NY
    )

    for p in model.particles.values():
        p.setProperty("newmark-beta beta", 0.0)
        p.setProperty("newmark-beta gamma", 0.0)
        p.setProperty("vms alpha", VMS_ALPHA)
        p.setProperty("vms mode", float(vmsMode))

    domain = ParticleKernelDomain(list(model.particles.values()), list(model.meshfreeKernelFunctions.values()))
    manager = KDBinOrganizedParticleManager(domain, 2, journal, bondParticlesToKernelFunctions=True)
    model.particleKernelDomains["all"] = domain
    model.prepareYourself(journal)
    manager.updateConnectivity()
    return model


def make_dQ(nDof, rng):
    """Random increment: per-node layout (ux, uy, p, j)."""
    dQ = np.zeros(nDof)
    dQ[0::4] = rng.uniform(-1, 1, nDof // 4) * 1e-2  # ux
    dQ[1::4] = rng.uniform(-1, 1, nDof // 4) * 1e-2  # uy
    dQ[2::4] = rng.uniform(-1, 1, nDof // 4) * 1e0   # p
    dQ[3::4] = rng.uniform(-1, 1, nDof // 4) * 1e-3  # j
    return dQ


def check_particle(particle, dQ0, qTotal=None, label=""):
    nDof = particle.nDof

    def fint(dQ):
        if qTotal is not None:
            particle.assignTotalNodalSolution(np.ascontiguousarray(qTotal + dQ))
        P = np.zeros(nDof)
        K = np.zeros(nDof * nDof)
        particle.computePhysicsKernels(np.ascontiguousarray(dQ), P, K, 1.0, 1.0)
        return P, K

    _, Kflat = fint(dQ0)
    K = Kflat.reshape((nDof, nDof), order="F")

    K_fd = np.zeros_like(K)
    hs = {0: 1e-6, 1: 1e-6, 2: 1e-4, 3: 1e-8}  # per dof type: u, u, p, j
    for jdof in range(nDof):
        h = hs[jdof % 4]
        e = np.zeros(nDof)
        e[jdof] = h
        Pp, _ = fint(dQ0 + e)
        Pm, _ = fint(dQ0 - e)
        K_fd[:, jdof] = (Pp - Pm) / (2 * h)

    # blockwise comparison: rows/cols by field (u, p, j)
    rows = {"u": np.sort(np.r_[0:nDof:4, 1:nDof:4]), "p": np.arange(2, nDof, 4), "j": np.arange(3, nDof, 4)}
    print(f"\n=== {label}: particle {particle.number}, nDof={nDof} ===")
    ok = True
    for rn, ri in rows.items():
        rowscale = max(np.abs(K[ri, :]).max(), np.abs(K_fd[ri, :]).max())
        for cn, ci in rows.items():
            Kb, Fb = K[np.ix_(ri, ci)], K_fd[np.ix_(ri, ci)]
            scale = max(np.abs(Fb).max(), np.abs(Kb).max())
            if scale < 1e-7 * rowscale:  # numerically zero block vs FD noise
                print(f"  K_{rn}{cn}: (zero block, |K|<{1e-7 * rowscale:.2g})")
                continue
            err = np.abs(Kb - Fb).max() / scale
            # K_uu/K_uj carry the documented pre-existing NSNI geometric omissions;
            # K_pu/K_pj carry the omitted dC_vms/dq (second-order material tangent,
            # not exposed by the material) -- ~0.5% at the G-scaled tau
            tol = 5e-4 if rn == "u" else (6e-3 if cn == "j" else 1e-3)
            flag = "" if err < tol else "   <-- MISMATCH"
            if err >= tol:
                ok = False
            print(f"  K_{rn}{cn}: max|K|={np.abs(Kb).max():10.4g}  rel.err={err:9.3g}{flag}")
    return ok


def check_mode_difference(p0, p1, dQ0, qTotal=None, label=""):
    """FD check of the PURE full-VMS contribution: (mode1 - mode0) of fInt and K.

    The (large) base residual/tangent cancels exactly, so the new terms are
    compared at their own scale instead of hiding under the base FD noise.
    """
    nDof = p0.nDof

    def dfint(dQ):
        Ps = []
        for p in (p0, p1):
            if qTotal is not None:
                p.assignTotalNodalSolution(np.ascontiguousarray(qTotal + dQ))
            P = np.zeros(nDof)
            K = np.zeros(nDof * nDof)
            p.computePhysicsKernels(np.ascontiguousarray(dQ), P, K, 1.0, 1.0)
            Ps.append((P, K))
        return Ps[1][0] - Ps[0][0], Ps[1][1] - Ps[0][1]

    dP0, dKflat = dfint(dQ0)
    dK = dKflat.reshape((nDof, nDof), order="F")

    dK_fd = np.zeros_like(dK)
    hs = {0: 1e-6, 1: 1e-6, 2: 1e-4, 3: 1e-8}
    for jdof in range(nDof):
        h = hs[jdof % 4]
        e = np.zeros(nDof)
        e[jdof] = h
        Pp, _ = dfint(dQ0 + e)
        Pm, _ = dfint(dQ0 - e)
        dK_fd[:, jdof] = (Pp - Pm) / (2 * h)

    rows = {"u": np.sort(np.r_[0:nDof:4, 1:nDof:4]), "p": np.arange(2, nDof, 4), "j": np.arange(3, nDof, 4)}
    print(f"\n=== DELTA(mode1-mode0) {label}: particle {p0.number}, nDof={nDof} ===")
    print(f"  |dP| max: {np.abs(dP0).max():.4g}  (nonzero => full-VMS terms active)")
    ok = True
    for rn, ri in rows.items():
        for cn, ci in rows.items():
            Kb, Fb = dK[np.ix_(ri, ci)], dK_fd[np.ix_(ri, ci)]
            scale = max(np.abs(Fb).max(), np.abs(Kb).max())
            if scale < 1e-10:  # analytically zero block; FD noise from base-term cancellation
                print(f"  dK_{rn}{cn}: (zero block)")
                continue
            err = np.abs(Kb - Fb).max() / scale
            # the full-VMS tangent deliberately omits O(increment) geometric and
            # second-order material couplings (d2S/dFbar2, d2S/dTheta2, dY_dx/J
            # variations) -> 1% tolerance on the delta blocks; the kept terms
            # dominate and match far better at small increments
            tol = 1e-2
            flag = "" if err < tol else "   <-- MISMATCH"
            if err >= tol:
                ok = False
            print(f"  dK_{rn}{cn}: max|dK|={np.abs(Kb).max():10.4g}  rel.err={err:9.3g}{flag}")
    return ok


def main():
    rng = np.random.default_rng(42)
    all_ok = True

    # NOTE: assignTotalNodalSolution is sticky -- once assigned, a particle stays on the
    # total-field VMS path. Hence separate model instances for the two paths.
    for path in ("increment", "total-solution"):
        model0 = build_model(0)
        model1 = build_model(1)
        parts0 = list(model0.particles.values())
        parts1 = list(model1.particles.values())

        for idx in (len(parts0) // 2, 0):
            p0, p1 = parts0[idx], parts1[idx]
            nDof = p0.nDof
            dQ0 = make_dQ(nDof, rng)
            qTot = make_dQ(nDof, rng) if path == "total-solution" else None
            all_ok &= check_particle(p1, dQ0, qTotal=qTot, label=f"mode 1, {path} path")
            all_ok &= check_mode_difference(p0, p1, dQ0, qTotal=qTot, label=f"{path} path")
    print("\nALL OK" if all_ok else "\nMISMATCHES FOUND")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
