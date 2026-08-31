#!/usr/bin/env python3
"""
Band width at CONSTANT fracture energy: halving the internal length halves the band.

For exponential softening smeared over a band whose width the gradient enhancement fixes,
    G_f ~ w ftu eps_f*,      w proportional to l,
so the width can be reduced at fixed toughness by halving l and doubling eps_f*.  This figure
shows that the model does exactly that: two computations of the same unconfined plane-strain test,
same G_f = 43.5 J/m^2, with l = 5 mm and l = 2.5 mm and the particle spacing refined with l to keep
the band resolved.

The two are compared at MATCHED MAXIMUM DAMAGE, not at matched strain: the finer computation
localises earlier and cannot be driven as far, so comparing end states would confound the width
with the stage of the failure.

Usage:  python bandwidth_gf.py [omegaTarget]      default 0.60
"""
import os
import sys

import numpy as np

from band_evolution import fastCmap

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = [("GFa_b", 5.00, 4.7485e-4, 2.50), ("GFb_b", 2.50, 9.4970e-4, 1.25)]
FTU = 9.1608


def bandWidth(xy, om):
    """Bin-free width of the localised zone, from the damage-weighted covariance.

    For a strip of width w the variance across it is w^2/12, hence w = sqrt(12 lambda_min).

    A profile FWHM was used first and is REJECTED: scanning all directions and keeping the
    narrowest profile depends strongly on the bin width -- the same field gave 6.25 mm at one bin
    and 3.00 mm at another -- because at some angles the projection aligns with the particle rows
    and produces a spuriously narrow profile.  The numbers it produced were reported once and were
    wrong.
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
    omTarget = float(sys.argv[1]) if len(sys.argv) > 1 else 0.60
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection
    from matplotlib.patheffects import withStroke

    PE = [withStroke(linewidth=2.4, foreground="0.12")]
    fig, ax = plt.subplots(1, len(CASES), figsize=(3.9 * len(CASES) + 1.1, 6.8))
    ax = np.atleast_1d(ax)
    cm = fastCmap()
    rows = []
    for a, (tag, l, ef, h) in zip(ax, CASES):
        d = np.load(os.path.join(HERE, f"snapshots_frame1_{tag}.npz"))
        om, u, xy0, verts, a2, ap = (d["omega"], d["u"], d["xy0"], d["verts"],
                                     d["axis2"], d["alphaP"])
        s = d["shortening"] * 100
        iL = int(np.argmax(s))
        omMax = np.array([om[k].max() for k in range(iL + 1)])
        k = int(np.argmin(np.abs(omMax - omTarget)))          # matched damage
        xy = xy0 + u[k]
        w, ang = bandWidth(xy, om[k])
        Gf = 2 * l * FTU * ef * 1000
        pc = PolyCollection(verts[k], array=om[k], cmap=cm, edgecolors="0.5",
                            linewidths=0.10 if h > 2 else 0.05)
        pc.set_clim(0.0, max(om[k].max(), 1e-6))
        a.add_collection(pc)
        k0 = next((j for j in range(len(a2))
                   if abs(np.linalg.norm(a2[j][0]) - 1.0) < 1e-6 and ap[j].max() < 1e-12), 0)
        # subsample so both panels carry a similar trace DENSITY: the fine mesh has 4x the
        # particles and a trace on each buries the damage field entirely
        step = max(1, int(round(len(xy0) / 160)))
        v = a2[k][:, :2]
        span = 1.6
        # the stored frame, and the REFERENCE orientation behind it: without a reference the
        # few degrees of rotation at this damage level are invisible, which reads as "the frame
        # does not rotate at all"
        v0 = a2[k0][:, :2]
        a.add_collection(LineCollection(
            np.stack([xy - 1.15 * span * v0, xy + 1.15 * span * v0], axis=1)[::step],
            colors="#bbbbbb", linewidths=2.2, path_effects=PE))
        a.add_collection(LineCollection(
            np.stack([xy - span * v, xy + span * v], axis=1)[::step],
            colors="#ff2ec4", linewidths=1.2, path_effects=PE))
        allV = verts[k].reshape(-1, 2)
        a.set_xlim(allV[:, 0].min() - 2, allV[:, 0].max() + 2)
        a.set_ylim(-2, verts[0].reshape(-1, 2)[:, 1].max() + 2)
        a.set_aspect("equal")
        a.set_axis_off()
        a.set_title(f"$l_d$ = {l:.2f} mm,  $h_p$ = {h:.2f} mm  ({len(xy0)} particles)\n"
                    f"$\\varepsilon_f^*$ = {ef:.2e},  $G_f$ = {Gf:.1f} J/m$^2$\n"
                    f"band width = {w:.2f} mm = {w/l:.2f} $l_d$,  dir {ang:.0f}$^\\circ$",
                    fontsize=9.5)
        rows.append((l, ef, h, len(xy0), Gf, w, om[k].max(), s[k], ang))
        last = pc
    cb = fig.colorbar(last, ax=list(ax), shrink=0.6, pad=0.015)
    cb.set_label(r"damage $\omega$", fontsize=10)
    fig.suptitle(r"Unconfined plane strain, $\beta = 45^\circ$, compared at matched damage "
                 r"$\omega_{\max}\approx$ " + f"{omTarget:.2f}" + "\n"
                 "same fracture energy, half the internal length\n"
                 "grey: reference bedding direction, magenta: stored (convected) one",
                 fontsize=10.5, y=1.05)
    out = os.path.join(HERE, "fig_bandwidth_gf.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=145, bbox_inches="tight")
    print(f"  wrote {out}\n")
    print("     l   eps_f*     h_p   part.    Gf     width   w/l   omega   eps%   dir")
    for l, ef, h, n, Gf, w, o, e, a in rows:
        print(f"  {l:5.2f}  {ef:.2e}  {h:5.2f}  {n:5d}  {Gf:5.1f}  {w:6.2f}  {w/l:5.2f}  "
              f"{o:6.2f}  {e:5.2f}  {a:5.1f}")
    if len({round(r[8]/10) for r in rows}) > 1:
        print("\n  NOTE: the two computations selected DIFFERENT conjugate bands "
              "(see the dir column),")
        print("  so the panels are not visually comparable even though the widths are.")


if __name__ == "__main__":
    main()
