"""Finite-difference verification of the FULLY stabilized VMS ('vms mode' = 2):
momentum- and jacobi-equation stabilization terms of the u-p-J derivation,

    r_U(A,j) -= C int Psi_{A,i,l} R_k dSdev_ij/d(u_{k,l}) dV0
    r_J(A)   -= C/3 int Psi_{A,l} R_k dtr(tau)/d(u_{k,l}) dV0

added on top of mode 1 in DisplacementPressureJacobiParticleSQCNIxNSNI.h.

Reuses the model builder and checkers of fd_tangent_check_upj_vms.py (same
directory). The delta check (mode2 - mode1) isolates exactly the new terms:
the mode-1 residual/tangent cancels, so the new blocks are compared at their
own scale.

Tolerances on the delta blocks: the new-term stiffness deliberately omits the
variation of the stabilization weights W_A / wJ_A themselves (second-order
material tangent d2S/dF2, not exposed by the material, plus the geometry of
the second-derivative push-forward) and the dY_dx variation inside div(S_dev)
-- the same category as the documented mode-1 omissions. The kept couplings
(residual variation through the material tangent, inertia, resolved grad(p),
jacobi field) dominate at small increments.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fd_tangent_check_upj_vms import build_model, check_particle, make_dQ


def check_mode2_difference(p1, p2, dQ0, qTotal=None, label="", skipRows=()):
    """FD check of the PURE mode-2 contribution: (mode2 - mode1) of fInt and K.

    Like check_mode_difference of the base harness, but with a RELATIVE zero-block
    threshold: the jacobi-row blocks are analytically near-zero in the Fbar-based
    u-p-J (the Fbar projector annihilates the volumetric direction of dtr(tau)/dF),
    so they must be classified as zero against the overall delta scale, not
    compared against pure FD cancellation noise.
    """
    nDof = p1.nDof

    def dfint(dQ):
        Ps = []
        for p in (p1, p2):
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

    globalscale = max(np.abs(dK).max(), np.abs(dK_fd).max())
    rows = {"u": np.sort(np.r_[0:nDof:4, 1:nDof:4]), "p": np.arange(2, nDof, 4), "j": np.arange(3, nDof, 4)}
    print(f"\n=== DELTA(mode2-mode1) {label}: particle {p1.number}, nDof={nDof} ===")
    print(f"  |dP| max: {np.abs(dP0).max():.4g}  (nonzero => mode-2 terms active)")
    ok = True
    for rn, ri in rows.items():
        if rn in skipRows:
            # mode 3 adds d2S/dF2 terms to this row's TANGENT while its residual is
            # identical to the mode-1 baseline -- the delta methodology sees additions
            # with a zero FD counterpart. Validated by the full-particle check instead.
            print(f"  dK_{rn}*: (skipped -- tangent-only improvements, see full check)")
            continue
        for cn, ci in rows.items():
            Kb, Fb = dK[np.ix_(ri, ci)], dK_fd[np.ix_(ri, ci)]
            scale = max(np.abs(Fb).max(), np.abs(Kb).max())
            # analytically (near-)zero block: the ANALYTIC part vanishes (e.g. the whole
            # jacobi row -- the Fbar projector annihilates dtr(tau)/dF, so wJ_A ~ 1e-11)
            # while the FD side only carries the cancellation noise of the two full-size
            # kernels (~1e-4 of the global scale for the 1e-8 jacobi probe)
            if np.abs(Kb).max() < 1e-8 * globalscale and np.abs(Fb).max() < 1e-3 * globalscale:
                print(f"  dK_{rn}{cn}: (zero block, |dK|={np.abs(Kb).max():.2g}, FD noise {np.abs(Fb).max():.2g})")
                continue
            err = np.abs(Kb - Fb).max() / scale
            # kept exactly: W_A itself, its contraction with d(R_res)/dq (material tangent,
            # inertia, resolved grad(p), jacobi field), the geometric variation of W_A's
            # push-forward and the gamma (Fbar-factor) variations. Omitted (documented in
            # the particle header): the second-order material tangent d2S/dF2 inside W_A
            # and R_res, and the dY_dx variation inside div(S_dev) -- same category as the
            # mode-1 omissions. Measured (elastic, random state): interior particles
            # 0.2-0.7 %, corner particles at extreme random states up to ~6.5 %.
            tol = 8e-2
            flag = "" if err < tol else "   <-- MISMATCH"
            if err >= tol:
                ok = False
            print(f"  dK_{rn}{cn}: max|dK|={np.abs(Kb).max():10.4g}  rel.err={err:9.3g}{flag}")
    return ok


def main():
    rng = np.random.default_rng(42)
    all_ok = True

    # NOTE: assignTotalNodalSolution is sticky (see base harness) -- separate model
    # instances per path.
    # mode 3 = mode 2 + FD second-order material tangent: identical residual, so the
    # delta(mode3 - mode1) check probes the SAME stabilization forces as
    # delta(mode2 - mode1) but against the completed tangent -- the previously omitted
    # d2S/dF2 blocks (up to ~6.5 % on corner particles in mode 2) should collapse.
    for highMode in (2, 3):
        for path in ("increment", "total-solution"):
            model1 = build_model(1)
            model2 = build_model(highMode)
            parts1 = list(model1.particles.values())
            parts2 = list(model2.particles.values())

            for idx in (len(parts1) // 2, 0):
                p1, p2 = parts1[idx], parts2[idx]
                nDof = p1.nDof
                dQ0 = make_dQ(nDof, rng)
                qTot = make_dQ(nDof, rng) if path == "total-solution" else None
                skip = ("p",) if highMode == 3 else ()
                all_ok &= check_mode2_difference(
                    p1, p2, dQ0, qTotal=qTot, label=f"mode{highMode}-mode1, {path} path", skipRows=skip
                )
                if highMode == 3:
                    # full-particle check: the mode-3 pressure/momentum rows must match the
                    # FD of the FULL residual at least as well as the mode-1 baseline does
                    all_ok &= check_particle(p2, dQ0, qTotal=qTot, label=f"mode 3 full, {path} path")
    print("\nALL OK" if all_ok else "\nMISMATCHES FOUND")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
