#!/usr/bin/env python3
"""
Evolution of the plane-strain compression test at one bedding inclination, over five stages.

Shows how the specimen gets from a homogeneous elastic state to a fully formed shear band, with
the CONVECTED material orientation drawn on every particle so the frame reorientation can be
followed alongside the damage.

WHAT THE DRAWN ORIENTATION IS, AND WHAT IT IS NOT.  The cyan traces are the stored material frame
`e^(2)(Fp)`, which convects with the PLASTIC deformation only, by eq. (framestate).  The elastic
part of the rotation is deliberately NOT folded into it -- it is carried by the image
`F^e e^(i)` on the current configuration, which is what the potential charges energy for, and
applying it twice would break objectivity.  The consequence is directly visible and was verified
on this data set: at the final increment 13 of the 105 particles have never yielded
(`alphaP == 0`), and their frame rotation is EXACTLY 0.000 deg, while the plastic particles reach
20.1 deg.  So a purely elastically deforming region shows no trace rotation here even though it
has deformed and rotated elastically.  To see the total bedding orientation as it appears in the
deformed body one would plot the push-forward of the reference direction, a different quantity.

WHY THE MATERIAL OUTSIDE THE BAND LOOKS UNDEFORMED AT THE END.  Two effects, both real.  It
unloads elastically once the band softens, so its elastic strain partly reverses; and, more
visibly, it moves almost RIGIDLY -- the mean displacement of the never-yielded particles grows
from 0.85 mm at the peak to 5.87 mm at the end.  Those regions are displacing a great deal, they
are simply not straining: the deformation is concentrated in the band and the blocks either side
translate and rotate as bodies.

Usage:  python band_evolution.py [beta] [meshkey]      default: 45 m1
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NSTAGE = 5


def main():
    beta = sys.argv[1] if len(sys.argv) > 1 else "45"
    mk = sys.argv[2] if len(sys.argv) > 2 else "m1"
    tag = f"snapshots_frame1_A_b{beta}_{mk}"
    d = np.load(os.path.join(HERE, tag + ".npz"))
    verts, u, om, a2 = d["verts"], d["u"], d["omega"], d["axis2"]
    ap, fr, xy0 = d["alphaP"], d["frameRotation"], d["xy0"]
    hist, short = d["history"], d["shortening"]
    sig = -hist[:, 1] - 30.0
    n = len(short)
    iPk = int(np.argmax(sig))

    # five stages: two on the way up (one clearly elastic), the peak, and two post-peak
    stages = sorted({max(1, iPk // 3), max(2, (2 * iPk) // 3), iPk,
                     iPk + (n - 1 - iPk) // 3, n - 1})
    while len(stages) > NSTAGE:
        stages.pop(1)
    print(f"  {tag}: {n} increments, peak at {iPk} ({short[iPk]*100:.2f} %); stages {stages}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection

    omMax = float(om[stages[-1]].max())
    fig, ax = plt.subplots(1, len(stages), figsize=(3.05 * len(stages), 6.6))
    ax = np.atleast_1d(ax)
    for a, k in zip(ax, stages):
        pc = PolyCollection(verts[k], array=om[k], cmap="inferno", edgecolors="0.45",
                            linewidths=0.2)
        pc.set_clim(0.0, max(omMax, 1e-6))
        a.add_collection(pc)
        # the convected bedding trace on every particle, at the deformed position
        xy = xy0 + u[k]
        v = a2[k][:, :2]
        span = 2.1
        segs = np.stack([xy - span * v, xy + span * v], axis=1)
        a.add_collection(LineCollection(segs, colors="#00f0ff", linewidths=0.8))
        allV = verts[k].reshape(-1, 2)
        a.set_xlim(allV[:, 0].min() - 2, allV[:, 0].max() + 2)
        a.set_ylim(-2, verts[0].reshape(-1, 2)[:, 1].max() + 2)
        a.set_aspect("equal")
        el = ap[k] < 1e-12
        a.set_title(f"{short[k]*100:.2f} % shortening\n"
                    f"sig_dev = {sig[k]:.1f} MPa{'  (peak)' if k == iPk else ''}\n"
                    f"omega max {om[k].max():.3f},  Rp max {np.abs(fr[k]).max():.1f} deg\n"
                    f"{el.sum()} of {len(el)} particles still elastic", fontsize=7.5)
        a.set_xlabel("x [mm]", fontsize=8)
        a.tick_params(labelsize=7)
    ax[0].set_ylabel("y [mm]", fontsize=8)
    fig.colorbar(pc, ax=list(ax), shrink=0.55, pad=0.015).set_label(
        "damage omega", fontsize=9)
    fig.suptitle(f"Plane-strain compression, bedding at beta = {beta} deg, RKPM, "
                 f"sigma_0 = 30 MPa, deformed at true scale\n"
                 "cyan: the stored material orientation e2(Fp), which convects with the PLASTIC "
                 "deformation only", fontsize=9.5, y=0.99)
    out = os.path.join(HERE, f"band_evolution_b{beta}_{mk}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
