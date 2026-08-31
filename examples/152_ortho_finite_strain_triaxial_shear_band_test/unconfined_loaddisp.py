#!/usr/bin/env python3
"""
Load-displacement curves of the UNCONFINED plane-strain compression tests.

Plane strain with a lateral confining pressure is an awkward idealisation -- the pressure acts on
the two side surfaces only, so it is neither a triaxial state nor a free one -- and it also
suppresses the brittleness the model is meant to show.  The unconfined test is the clean
plane-strain case, and it is where the paper's own card produces what a quasi-brittle rock should:
an almost complete loss of stiffness and a stress falling to a few per cent of the peak.  The
confining pressure belongs to the three-dimensional triaxial tests.

Usage:  python unconfined_loaddisp.py [tagPrefix]     default: NC_b
"""
import glob
import os
import re
import sys

import numpy as np

from band_evolution import fastCmap  # noqa: F401  (keeps one colour convention in the study)

HERE = os.path.dirname(os.path.abspath(__file__))
HEIGHT = 75.0
BETAS = [0, 45, 90]


def main():
    pre = sys.argv[1] if len(sys.argv) > 1 else "NC_b"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    cols = {0: "#1b6ca8", 45: "#e8871a", 90: "#2e8b57"}
    fig, ax = plt.subplots(1, 3, figsize=(12.2, 3.9))
    rows = []
    for b in BETAS:
        f = glob.glob(os.path.join(HERE, f"snapshots_frame1_{pre}{b}*.npz"))
        if not f:
            continue
        d = np.load(f[0])
        s = d["shortening"] * 100.0
        sig = -d["history"][:, 1]          # unconfined: no offset to subtract
        om = d["omega"]
        omMax = np.array([om[k].max() for k in range(len(om))])
        iL = int(np.argmax(s))
        iPk = int(np.argmax(sig))
        # split the loading branch from any unloading tail
        ax[0].plot(s[: iL + 1], sig[: iL + 1], color=cols[b], lw=1.9)
        ax[1].plot(s[: iL + 1], omMax[: iL + 1], color=cols[b], lw=1.9)
        if iL < len(s) - 2:
            ax[0].plot(s[iL:], sig[iL:], color=cols[b], lw=1.4, ls=":")
            ax[1].plot(s[iL:], omMax[iL:], color=cols[b], lw=1.4, ls=":")
        # damage against stress: the degradation path
        ax[2].plot(omMax[: iL + 1], sig[: iL + 1] / max(sig[iPk], 1e-30), color=cols[b], lw=1.9)
        rows.append((b, sig[iPk], s[iPk], sig[iL], s[iL], omMax[iL],
                     100 * (1 - sig[iL] / sig[iPk]), iL < len(s) - 2))
    ax[0].set_xlabel("axial shortening [%]")
    ax[0].set_ylabel(r"$\sigma_{yy}$ [MPa]")
    ax[1].set_xlabel("axial shortening [%]")
    ax[1].set_ylabel(r"max damage $\omega$")
    ax[1].set_ylim(-0.03, 1.03)
    ax[2].set_xlabel(r"max damage $\omega$")
    ax[2].set_ylabel(r"$\sigma_{yy}/\sigma_{yy}^{\rm peak}$")
    ax[2].set_xlim(-0.03, 1.03)
    for a in ax:
        a.grid(alpha=0.3)
    h = [Line2D([], [], color=cols[b], lw=1.9, label=rf"$\beta = {b}^\circ$") for b in BETAS]
    h += [Line2D([], [], color="0.35", lw=1.4, ls=":", label="unloading")]
    fig.legend(handles=h, loc="upper center", ncol=4, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Unconfined plane-strain compression, paper card, "
                 r"$h_p = l_d = 5$ mm", fontsize=9.5, y=0.90)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    out = os.path.join(HERE, "fig_unconfined_loaddisp.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=145)
    print(f"  wrote {out}\n")
    print("  beta | peak [MPa] @eps%  | end [MPa] @eps% | omegaMax | softening | unloading")
    print("  " + "-" * 78)
    for b, sp, ep, se, ee, o, sf, ul in rows:
        print(f"  {b:4d} | {sp:10.2f} {ep:6.2f}  | {se:9.2f} {ee:6.2f} | {o:8.3f} | "
              f"{sf:8.0f} % | {'yes' if ul else 'no'}")


if __name__ == "__main__":
    main()
