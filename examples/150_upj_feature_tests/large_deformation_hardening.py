"""Large finite-strain plane-strain compression on a NON-SOFTENING (linearly
hardening) u-p-J material.

Companion to the softening shear-band study (_upj_compression.run_sim, the
"current example"): here fyInf = fy (the Voce softening term (fy-fyInf)exp(-eta a)
vanishes) and the linear hardening modulus H > 0, so beta(alpha) = fy + H*alpha is
monotone. The material does NOT localize, so displacement control traces to large
finite strain (~10 % nominal) instead of stalling at a softening bifurcation. Used
to exercise the referential-gradient total pressure gradient (vms refgrad) and to
visualise the deformed smoothing domains at large deformation.

This is a NEW, self-driving input file; it does not modify the softening example.
Run:  python large_deformation_hardening.py
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _upj_compression as C  # noqa: E402

# the C++ particles keep a bare reference to the approximation wrapper -> keep alive
_keep = []
_orig = C.MarmotMeshfreeApproximationWrapper
C.MarmotMeshfreeApproximationWrapper = lambda *a, **k: (_keep.append(_orig(*a, **k)) or _keep[-1])

HEIGHT = 120.0


def _order(V):
    """cyclic-order the 4 quad corners for polygon fill"""
    c = V.mean(0)
    ang = np.arctan2(V[:, 1] - c[1], V[:, 0] - c[0])
    return V[np.argsort(ang)]


def main(nX=10, nY=20, totalCompression=-12.0, H=500.0, vmsRefgrad=1.0, cwf="bottom"):
    model, foc, rm = C.run_sim(
        particleType="DisplacementPressureJacobiSQCNIxNSNI/PlaneStrain/Quad",
        vmsMode=1,
        vmsRefgrad=vmsRefgrad,
        vmsAlpha=0.02,
        nX=nX,
        nY=nY,
        totalCompression=totalCompression,
        incSize=0.01,
        fyInf=100.0,  # == fy  -> NO softening
        H=H,          # linear hardening modulus
        cwfCorrection=cwf,
        outputName="large_def_hardening",
    )
    u = np.abs(np.array(rm.u_history))
    F = np.array(rm.F_history)
    V = np.array([np.asarray(p.getVertexCoordinates()) for p in model.particleSets["all"]])
    p = np.asarray(foc.fieldOutputs["pressure"].getLastResult()).ravel()
    ap = np.asarray(foc.fieldOutputs["alphaP"].getLastResult()).ravel()
    print(
        f"reached u={u[-1]:.3f} mm ({100 * u[-1] / HEIGHT:.1f}% nominal), incs={u.size}, "
        f"F_end={F[-1]:.0f}, p[{p.min():.1f},{p.max():.1f}], apMax={ap.max():.3f}"
    )

    polys = [_order(v) for v in V]
    allv = np.vstack(polys)
    fig = plt.figure(figsize=(12.5, 6.6))
    ax = fig.add_subplot(1, 3, 1)
    ax.plot(u, F, "-o", c="#d62728", ms=3, lw=1.5)
    ax.set_xlabel("top shortening |u$_y$| (mm)")
    ax.set_ylabel(r"top reaction F$_y$ (N)")
    ax.set_title(f"Load-displacement\nhardening H={H:.0f}, refgrad ON", fontsize=9)
    ax.grid(alpha=0.3)
    for i, (fld, cmap, lab) in enumerate([("ap", "viridis", r"plastic strain $\alpha_P$"),
                                          ("p", "RdBu_r", "pressure")]):
        ax = fig.add_subplot(1, 3, 2 + i)
        vals = ap if fld == "ap" else p
        if fld == "p":
            vmax = np.abs(p).max()
            pc = PolyCollection(polys, array=vals, cmap=cmap, edgecolors="k", linewidths=0.25)
            pc.set_clim(-vmax, vmax)
        else:
            pc = PolyCollection(polys, array=vals, cmap=cmap, edgecolors="k", linewidths=0.25)
        ax.add_collection(pc)
        ax.set_xlim(allv[:, 0].min() - 2, allv[:, 0].max() + 2)
        ax.set_ylim(allv[:, 1].min() - 2, allv[:, 1].max() + 2)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{lab}\ndeformed @ u={u[-1]:.2f} mm", fontsize=9)
        plt.colorbar(pc, ax=ax, fraction=0.09, pad=0.02)
    fig.suptitle(
        f"Non-softening (hardening) compression, {nX}x{nY}, mode 1 + refgrad -- traces to large strain",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "large_deformation_hardening.png")
    fig.savefig(out, dpi=145)
    print("wrote", out)


if __name__ == "__main__":
    main()
