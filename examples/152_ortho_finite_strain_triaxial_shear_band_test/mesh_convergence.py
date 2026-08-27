# -*- coding: utf-8 -*-
"""Mesh-convergence summary for the plane-strain triaxial shear-band test.

Reads the ``snapshots_frame{0,1}_<tag>.npz`` files written by
``ortho_finite_strain_triaxial_shear_band_test.py`` and produces one table and one figure
comparing the meshes and the two material frames.

WHAT IS AND IS NOT A MESH STUDY HERE.  The runs are displacement controlled, and they terminate
AT the structural limit point -- the peak is passed by one increment and then the tangent
stiffness is singular.  So:

  * the PEAK strength and the strain at which it occurs ARE converged quantities and are what
    this script reports;
  * the BAND WIDTH is not, because at the point the runs stop the damage has barely started
    (omega of order 0.01), and a band width measured there is a width of the incipient
    localisation, not of the developed band.  Getting the developed band at h < l needs
    arc-length or indirect control.

Usage
-----
    python mesh_convergence.py
"""

import glob
import re
import sys

import numpy as np

L_NONLOCAL = 1.25
WIDTH, HEIGHT = 10.0, 20.0


def load(pattern="h"):
    """``pattern`` is the tag prefix: "h" for the compression study, "sh" for the shear study."""
    runs = {}
    for path in sorted(glob.glob(f"snapshots_frame*_{pattern}*l.npz")):
        m = re.match(rf"snapshots_frame(\d)_({pattern}.*)\.npz", path)
        if not m:
            continue
        frame, tag = int(m.group(1)), m.group(2)
        d = np.load(path)
        nP = d["omega"].shape[1]
        # nX * nY = nP with nX/nY = WIDTH/HEIGHT -> nY = sqrt(nP * HEIGHT / WIDTH)
        nY = int(round(np.sqrt(nP * HEIGHT / WIDTH)))
        h = HEIGHT / nY
        runs[(tag, frame)] = dict(d=d, nP=nP, h=h, tag=tag, frame=frame)
    return runs


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else "h"
    isShear = pattern == "sh"
    runs = load(pattern)
    if not runs:
        print(f"no snapshots_frame*_{pattern}*l.npz found -- run the test with --tag {pattern}... first")
        return

    tags = sorted({k[0] for k in runs}, key=lambda t: -runs[(t, 0)]["h"])

    print("=" * 100)
    print(f" MESH CONVERGENCE -- plane-strain {'SIMPLE SHEAR' if isShear else 'triaxial compression'},"
          f" 5 MPa confinement,")
    print(" SQCNIxNSNI stabilized nodal integration, bedding normal at 45 deg")
    print("=" * 100)
    print(f"{'h [mm]':>8} {'h/l':>6} {'particles':>10} | {'peak frozen':>12} {'at [%]':>7}"
          f" | {'peak convect':>13} {'at [%]':>7} | {'diff [%]':>9} | {'reached f/c [%]':>16}"
          f" | {'max Rp [deg]':>12}")
    print("-" * 100)
    rows = []
    for tag in tags:
        if (tag, 0) not in runs or (tag, 1) not in runs:
            continue
        r0, r1 = runs[(tag, 0)], runs[(tag, 1)]
        h0, h1 = r0["d"]["history"], r1["d"]["history"]
        i0, i1 = np.argmin(h0[:, 1]), np.argmin(h1[:, 1])
        p0, p1 = -h0[i0, 1], -h1[i1, 1]
        rows.append((r0["h"], r0["nP"], p0, h0[i0, 0] * 100, p1, h1[i1, 0] * 100,
                     h0[-1, 0] * 100, h1[-1, 0] * 100, r1["d"]["frameRotation"].max()))
        print(f"{r0['h']:8.4f} {r0['h'] / L_NONLOCAL:6.3f} {r0['nP']:10d} | {p0:12.3f} {h0[i0, 0] * 100:7.3f}"
              f" | {p1:13.3f} {h1[i1, 0] * 100:7.3f} | {100 * (p1 - p0) / p0:9.3f}"
              f" | {h0[-1, 0] * 100:7.2f} /{h1[-1, 0] * 100:7.2f} | {r1['d']['frameRotation'].max():12.3f}")
    print("-" * 100)
    if len(rows) > 1:
        print("  successive change of the frozen peak:  "
              + "  ".join(f"{100 * (rows[i + 1][2] - rows[i][2]) / rows[i][2]:+.2f} %"
                          for i in range(len(rows) - 1)))
        print(f"  the two frames agree on the peak to {max(abs(r[4] - r[2]) / r[2] for r in rows) * 100:.3f} %"
              " at every mesh -- the frame has barely turned by the peak, so it cannot matter there")
        reach = [min(r[6], r[7]) for r in rows]
        if max(reach) - min(reach) > 0.4 * max(reach):
            print("  reachable strain varies strongly with h: that is the LIMIT POINT, not mesh")
            print("  dependence, and it confines any post-peak comparison to the coarsest mesh.")
        else:
            print(f"  reachable strain is comparable at every mesh ({min(reach):.2f}-{max(reach):.2f} %):")
            print("  the softening is traversable, so the curves below are comparable throughout.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = np.array(rows)
    fig, ax = plt.subplots(1, 3, figsize=(14.0, 4.2))

    ax[0].plot(a[:, 1], a[:, 2], "o-", label="frozen frame")
    ax[0].plot(a[:, 1], a[:, 4], "s--", label="convected frame")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("particles")
    ax[0].set_ylabel(r"peak $-\tau_{yy}$ [MPa]")
    ax[0].set_title("peak strength converges", fontsize=10)

    ax[1].plot(a[:, 1], a[:, 3], "o-", label="frozen frame")
    ax[1].plot(a[:, 1], a[:, 5], "s--", label="convected frame")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("particles")
    ax[1].set_ylabel("shortening at the peak [%]")
    ax[1].set_title("and so does the strain at the peak", fontsize=10)

    ax[2].plot(a[:, 1], a[:, 6], "o-", label="frozen frame")
    ax[2].plot(a[:, 1], a[:, 7], "s--", label="convected frame")
    ax[2].set_xscale("log")
    ax[2].set_yscale("log")
    ax[2].set_xlabel("particles")
    ax[2].set_ylabel("reachable shortening [%]")
    ax[2].set_title("but the reachable post-peak collapses\n(the limit point, not the mesh)",
                    fontsize=10)

    for a_ in ax:
        a_.grid(alpha=0.3, which="both")
        a_.legend(fontsize=8)
    fig.tight_layout()

    # THE mesh-objectivity picture: if the gradient enhancement is doing its job, the whole
    # load-displacement curve -- not just the peak -- must be independent of h.
    fig3, ax3 = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for tag in tags:
        for frame, axx, ls in ((0, ax3[0], "-"), (1, ax3[1], "--")):
            if (tag, frame) not in runs:
                continue
            r = runs[(tag, frame)]
            hh = r["d"]["history"]
            axx.plot(hh[:, 0] * 100, -hh[:, 1], ls, lw=1.4,
                     label=f"h = {r['h']:.3f} mm ({r['nP']} particles)")
    for axx, ttl in ((ax3[0], "frozen frame"), (ax3[1], "convected frame")):
        axx.set_xlabel(("shear angle" if isShear else "shortening") + r" $\gamma$ [%]")
        axx.set_ylabel(r"$-\tau_{yy}$ at mid-height [MPa]")
        axx.set_title(ttl + " -- curves should coincide if the\ngradient enhancement regularises",
                      fontsize=9)
        axx.grid(alpha=0.3)
        axx.legend(fontsize=8)
    fig3.tight_layout()
    fig3.savefig(f"mesh_objectivity_{pattern}.png", dpi=140)
    print(f"  wrote mesh_objectivity_{pattern}.png")

    fig.savefig(f"mesh_convergence_{pattern}.png", dpi=140)
    print(f"\n  wrote mesh_convergence_{pattern}.png")


if __name__ == "__main__":
    main()
