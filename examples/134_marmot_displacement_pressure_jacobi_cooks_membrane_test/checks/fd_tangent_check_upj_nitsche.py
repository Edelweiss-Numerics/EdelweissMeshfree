"""FD verification of the NITSCHEDIRICHLET boundary term of the u-p-J particle
(consistency traction + penalty on the boundary gap): d(fExt)/ddQ vs the analytic
dFExt_ddQ, chained through computePhysicsKernels (which sets the material-point
state the boundary term reads); the penalty gap u(Y_N) - g reads the TRACKED
particle geometry plus the current increment (kernel-drift trap: never the total
nodal coefficients)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fd_tangent_check_upj_vms import build_model, make_dQ

BETA = 1e4  # ~ nitscheBeta * K / h at the FD-check scale
MASK = np.array([1.0, 1.0])
G_PRESCRIBED = np.array([0.03, -0.02])  # nonzero g so the gap has both signs


def main():
    rng = np.random.default_rng(7)
    model = build_model(0)
    parts = list(model.particles.values())
    ok = True
    for p in (parts[0], parts[2]):  # boundary particles (left edge region)
        nDof = p.nDof
        dQ0 = make_dQ(nDof, rng)

        loadVec = np.concatenate([MASK, G_PRESCRIBED, [BETA]])

        def nitsche(dQ, faceID=4):
            Pphys = np.zeros(nDof)
            Kphys = np.zeros(nDof * nDof)
            p.computePhysicsKernels(np.ascontiguousarray(dQ), Pphys, Kphys, 1.0, 1.0)
            Pc = np.zeros(nDof)
            Kc = np.zeros(nDof * nDof)
            p.computeDistributedLoad("NITSCHEDIRICHLET", faceID, loadVec, Pc, Kc, 1.0, 1.0)
            return Pc, Kc

        _, Kflat = nitsche(dQ0)
        K = Kflat.reshape((nDof, nDof), order="F")
        K_fd = np.zeros_like(K)
        hs = {0: 1e-6, 1: 1e-6, 2: 1e-4, 3: 1e-8}
        for j in range(nDof):
            h = hs[j % 4]
            e = np.zeros(nDof)
            e[j] = h
            Pp, _ = nitsche(dQ0 + e)
            Pm, _ = nitsche(dQ0 - e)
            K_fd[:, j] = (Pp - Pm) / (2 * h)

        cols = {"u": np.sort(np.r_[0:nDof:4, 1:nDof:4]), "p": np.arange(2, nDof, 4), "j": np.arange(3, nDof, 4)}
        urows = cols["u"]
        print(f"\nparticle {p.number}:")
        for cn, ci in cols.items():
            Kb, Fb = K[np.ix_(urows, ci)], K_fd[np.ix_(urows, ci)]
            scale = max(np.abs(Kb).max(), np.abs(Fb).max(), 1e-14)
            err = np.abs(Kb - Fb).max() / scale
            flag = "" if err < 1e-4 else "   <-- MISMATCH"
            ok &= err < 1e-4
            print(f"  dP_u/dq_{cn}: max|K|={np.abs(Kb).max():10.4g}  rel.err={err:9.3g}{flag}")
    print("\nALL OK" if ok else "\nMISMATCHES FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
