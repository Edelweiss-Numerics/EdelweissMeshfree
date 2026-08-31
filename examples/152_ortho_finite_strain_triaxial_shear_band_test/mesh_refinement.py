#!/usr/bin/env python3
"""
Mesh refinement of the unconfined plane-strain test at FIXED internal length.

Three particle spacings, one card, one internal length: the test of whether the gradient
enhancement regularises the problem.  What must be mesh independent is the PEAK; the band width
must be set by the internal length rather than by the spacing.

BAND WIDTHS ARE COMPARED AT MATCHED MAXIMUM DAMAGE, NEVER AT THE END OF EACH RUN.  The runs end
at wildly different stages -- omega = 0.996, 0.992 and 0.126 for the three spacings here, because
the finest one stalls just past its peak -- so comparing end states measures how far each run got,
not how wide its band is.  Doing that once produced an apparent loss of mesh objectivity
(25.4 -> 16.7 -> 8.9 mm) that does not exist: at matched damage the widths are 2.1 to 3.1 times
the internal length at every stage, and the peak is mesh independent to 2.8 % over a 17x range in
particle number.

The residual difference between h_p = l and h_p = l/2 is discretisation error at the coarse end,
where h_p = l leaves about two particles across the band; the usual requirement is
h_p <= l/2 to l/3.

Usage:  python mesh_refinement.py [omegaTarget ...]      default: 0.3 0.5 0.6 0.8
"""
import os
import sys

import numpy as np

from band_evolution import fastCmap  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = [(5.00, "MR5.0_b"), (2.50, "MR2.5_b"), (1.25, "MR1.25_b")]
L_NL = 5.0


def bandWidth(xy, om):
    """bin-free width of the localised zone: for a strip of width w the across-variance is w^2/12.

    A profile-FWHM measure was tried first and rejected: scanning all directions and taking the
    narrowest profile is strongly bin-width dependent (the same field gave 6.25 mm at one bin
    width and 3.00 mm at another), because at some angles the projection aligns with the particle
    rows and produces a spuriously narrow profile.
    """
    m = om > 0.5 * om.max()
    if m.sum() < 6:
        return np.nan, np.nan
    pts, wt = xy[m], om[m]
    mu = np.average(pts, axis=0, weights=wt)
    C = np.cov((pts - mu).T, aweights=wt)
    ev, evec = np.linalg.eigh(C)
    return float(np.sqrt(12 * ev[0])), float(np.degrees(np.arctan2(evec[1, 1], evec[0, 1])) % 180)


def main():
    targets = [float(a) for a in sys.argv[1:]] or [0.3, 0.5, 0.6, 0.8]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    cols = {5.00: "#1b6ca8", 2.50: "#e8871a", 1.25: "#2e8b57"}
    fig, ax = plt.subplots(1, 3, figsize=(12.4, 4.0))
    rows = []
    for h, tag in CASES:
        d = np.load(os.path.join(HERE, f"snapshots_frame1_{tag}.npz"))
        s = d["shortening"] * 100
        sig = -d["history"][:, 1]
        om = d["omega"]
        omM = np.array([om[k].max() for k in range(len(om))])
        iL, iP = int(np.argmax(s)), int(np.argmax(sig))
        n = len(d["xy0"])
        ax[0].plot(s[: iL + 1], sig[: iL + 1], color=cols[h], lw=1.9)
        ax[1].plot(s[: iL + 1], omM[: iL + 1], color=cols[h], lw=1.9)
        ax[2].plot([n], [sig[iP]], "o", color=cols[h], ms=7)
        wm = {}
        for t in targets:
            if omM.max() >= t - 0.02:
                k = int(np.argmin(np.abs(omM[: iL + 1] - t)))
                wm[t] = bandWidth(d["xy0"] + d["u"][k], om[k])[0]
        rows.append((h, n, s[iL], sig[iP], s[iP], sig[iL], omM[iL], wm))
    ax[0].set_xlabel("axial shortening [%]"); ax[0].set_ylabel(r"$\sigma_{yy}$ [MPa]")
    ax[1].set_xlabel("axial shortening [%]"); ax[1].set_ylabel(r"max damage $\omega$")
    ax[1].set_ylim(-0.03, 1.03)
    ax[2].set_xscale("log")
    ax[2].set_xlabel("particles")
    ax[2].set_ylabel(r"peak $\sigma_{yy}$ [MPa]")
    pk = [r[3] for r in rows]
    ax[2].set_ylim(min(pk) - 3, max(pk) + 3)
    ax[2].axhline(np.mean(pk), color="0.5", ls="--", lw=1.0)
    for a_ in ax:
        a_.grid(alpha=0.3)
    h_ = [Line2D([], [], color=cols[h], lw=1.9, label=rf"$h_p$ = {h:.2f} mm ({n} particles)")
          for h, n, *_ in rows]
    fig.legend(handles=h_, loc="upper center", ncol=3, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(r"Unconfined plane strain, $\beta=45^\circ$, fixed $l_d = 5$ mm: "
                 "the peak is mesh independent, the reachable post-peak is not",
                 fontsize=9.5, y=0.90)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    out = os.path.join(HERE, "fig_mesh_refinement.pdf")
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=145)
    print(f"  wrote {out}\n")
    print("   h_p  part.  reach%   peak @eps%    end   omegaEnd")
    for h, n, sr, pkv, pe, ev, o, _ in rows:
        print(f"  {h:5.2f} {n:5d} {sr:7.2f} {pkv:7.2f} {pe:6.2f} {ev:7.2f} {o:9.3f}")
    print(f"\n  peak spread over a {max(r[1] for r in rows)//min(r[1] for r in rows)}x range in "
          f"particle number: {100*(max(pk)-min(pk))/np.mean(pk):.1f} %")
    print("\n  band width at MATCHED maximum damage [mm]  (in brackets: width / l)")
    hdr = "   omega |" + "".join(f"  h_p = {h:.2f}      " for h, *_ in rows)
    print(hdr)
    for t in targets:
        line = f"   {t:5.2f} |"
        for h, n, *_, wm in rows:
            line += (f"  {wm[t]:6.2f} ({wm[t]/L_NL:4.2f} l)" if t in wm
                     else f"  {'not reached':>16s}")
        print(line)


if __name__ == "__main__":
    main()
