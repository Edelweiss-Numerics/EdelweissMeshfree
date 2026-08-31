#!/usr/bin/env python3
"""
Evolution of the plane-strain compression test at one bedding inclination, over five stages.

Shows how the specimen gets from a homogeneous elastic state to a fully formed shear band, with
the CONVECTED material orientation drawn on every particle so the frame reorientation can be
followed alongside the damage.

TWO ORIENTATIONS ARE DRAWN.  Cyan is the stored frame, yellow the push-forward of the reference
bedding direction by the TOTAL deformation gradient, F e_0 / |F e_0| -- the orientation as it
appears in the deformed body.  The difference between them is exactly the ELASTIC part of the
reorientation, which the stored frame does not carry (see below).  F is fitted per particle by
least squares from the reference to the deformed smoothing-domain vertices; the reference is
increment 0, which carries only the confinement (max |u| = 0.16 mm over a 75 mm specimen), so the
bias is negligible for an orientation.

WHAT THE STORED ORIENTATION IS, AND WHAT IT IS NOT.  The cyan traces are the stored material frame
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

Usage:  python band_evolution.py [beta] [tag] [--which=both|total|plastic]
                                 [--defscale=S] [--cmap=fast|<matplotlib name>]
        default: 45 m1 --which=both --defscale=1 --cmap=fast
        --defscale exaggerates the displacements; 1 is true scale.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

NSTAGE = 5      # loading stages; an unloaded panel is appended when the record has one

# ParaView's "Fast" preset, RGBPoints read straight out of the installed ParaView via
# `pvpython -c "... vtkSMTransferFunctionPresets ... GetPresetAsString"`, not from memory: an
# earlier hand-written table was actually a cool-warm map and looked nothing like Fast, which runs
# dark blue -> cyan -> pale yellow-green -> orange -> dark red.  ParaView declares
# "ColorSpace": "Lab" for it, so the control points are interpolated in CIELAB below rather than
# in RGB, which is what makes the reproduction faithful rather than merely similar.
PARAVIEW_FAST = [
    (0.000000, (0.0564, 0.0564, 0.4700)),
    (0.171592, (0.2430, 0.4604, 0.8100)),
    (0.298491, (0.3568, 0.7450, 0.9544)),
    (0.432129, (0.6882, 0.9300, 0.9179)),
    (0.500000, (0.8995, 0.9446, 0.7687)),
    (0.588226, (0.9571, 0.8338, 0.5089)),
    (0.706141, (0.9275, 0.6214, 0.3154)),
    (0.847640, (0.8000, 0.3520, 0.1600)),
    (1.000000, (0.5900, 0.0767, 0.1195)),
]

_WP = np.array([0.95047, 1.0, 1.08883])          # D65
_M = np.array([[0.4124, 0.3576, 0.1805],
               [0.2126, 0.7152, 0.0722],
               [0.0193, 0.1192, 0.9505]])


def _srgbToLab(c):
    c = np.asarray(c, float)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = lin @ _M.T / _WP
    d = (6.0 / 29.0) ** 3
    f = np.where(xyz > d, np.cbrt(xyz), xyz / (3 * (6.0 / 29.0) ** 2) + 4.0 / 29.0)
    return np.stack([116 * f[..., 1] - 16,
                     500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], axis=-1)


def _labToSrgb(lab):
    lab = np.asarray(lab, float)
    fy = (lab[..., 0] + 16) / 116
    fx = fy + lab[..., 1] / 500
    fz = fy - lab[..., 2] / 200
    f = np.stack([fx, fy, fz], axis=-1)
    d = 6.0 / 29.0
    xyz = np.where(f > d, f ** 3, 3 * d ** 2 * (f - 4.0 / 29.0)) * _WP
    lin = xyz @ np.linalg.inv(_M).T
    c = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.abs(lin) ** (1 / 2.4) - 0.055)
    return np.clip(c, 0.0, 1.0)


def fastCmap(n=256):
    """ParaView's Fast preset, interpolated in CIELAB as ParaView does."""
    from matplotlib.colors import ListedColormap
    pos = np.array([p for p, _ in PARAVIEW_FAST])
    lab = _srgbToLab(np.array([c for _, c in PARAVIEW_FAST]))
    t = np.linspace(0.0, 1.0, n)
    out = np.empty((n, 3))
    for k in range(3):
        out[:, k] = np.interp(t, pos, lab[:, k])
    return ListedColormap(_labToSrgb(out), name="pvFast")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = {a.split("=")[0]: (a.split("=", 1)[1] if "=" in a else "")
            for a in sys.argv[1:] if a.startswith("--")}
    beta = args[0] if args else "45"
    mk = args[1] if len(args) > 1 else "m1"
    which = opts.get("--which", "both")
    defScale = float(opts.get("--defscale", "1") or 1)
    cmapName = opts.get("--cmap", "fast")
    # `mk` may be a mesh key of the standard sweep ("m1") or a full run tag ("UL2_b")
    cand = [f"snapshots_frame1_A_b{beta}_{mk}", f"snapshots_frame1_{mk}"]
    tag = next((c for c in cand if os.path.exists(os.path.join(HERE, c + ".npz"))), cand[0])
    d = np.load(os.path.join(HERE, tag + ".npz"))
    verts, u, om, a2 = d["verts"], d["u"], d["omega"], d["axis2"]
    ap, fr, xy0 = d["alphaP"], d["frameRotation"], d["xy0"]
    short = d["shortening"]
    sig = -d["history"][:, 1] - 30.0
    n = len(short)
    iPk = int(np.argmax(sig))

    # If the record contains an UNLOADING tail (the shortening decreases at the end), the
    # loading stages must be chosen from the loading part only, and the unloaded state is added
    # as a final panel.  The state to show is the increment where sigma_dev crosses ZERO, not the
    # last one: a prescribed reversal easily overshoots into REVERSE loading (measured -26 MPa),
    # where the elastic reorientation has changed sign rather than relaxed.
    iEndLoad = int(np.argmax(short))
    iUnload = None
    if iEndLoad < n - 2:
        post = np.arange(iEndLoad, n)
        iUnload = int(post[int(np.argmin(np.abs(sig[post])))])
        print(f"  unloading tail found: load to {short[iEndLoad]*100:.2f} %, "
              f"sigma_dev = 0 at increment {iUnload} ({short[iUnload]*100:.2f} %)")
    iLast = iEndLoad
    # Choose the stages by STRAIN, not by increment index: the increments are far from uniform in
    # strain (the stepper cuts back around the peak and in the band), so index interpolation
    # clusters two panels at almost the same state.
    sPk, sMax = short[iPk], short[iLast]
    targets = [0.25 * sPk, 0.7 * sPk, sPk,
               sPk + 0.40 * (sMax - sPk), sMax][:NSTAGE]
    stages = sorted({int(np.argmin(np.abs(short[: iLast + 1] - t))) for t in targets})
    if iUnload is not None:
        stages.append(iUnload)
    print(f"  {tag}: {n} increments, peak at {iPk} ({short[iPk]*100:.2f} %); stages {stages}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection
    from matplotlib.lines import Line2D
    from matplotlib.patheffects import withStroke

    # The Fast colour map runs dark blue -> near white -> dark red, so no single flat colour is
    # legible over all of it.  Both trace families get a dark outline instead.
    PE = [withStroke(linewidth=2.6, foreground="0.12")]
    C_PLASTIC, C_TOTAL = "#ff2ec4", "#00ff7f"

    def fitF(ref, cur):
        """least-squares 2x2 deformation gradient per particle, from quad vertices"""
        R = ref - ref.mean(axis=1, keepdims=True)
        D = cur - cur.mean(axis=1, keepdims=True)
        return np.array([np.linalg.lstsq(R[i], D[i], rcond=None)[0].T for i in range(len(R))])

    # The REFERENCE in-plane bedding axis, taken from the data rather than from beta.  This
    # example sets the card's direction from BEDDING_PHI_DEG = 90 - beta, so an analytic
    # (-sin beta, cos beta) is the NORMAL, not the in-plane axis, at every beta except 45 where the
    # two conventions happen to agree -- which is exactly how the error hid: it gave a degenerate
    # 90 deg at beta = 0 and 90 and looked correct at 45.  axis2 at the first increment is the
    # reference axis by construction (Fp = I there); verified unit-norm with alphaP = 0.
    k0 = next((k for k in range(len(a2))
               if abs(np.linalg.norm(a2[k][0]) - 1.0) < 1e-6 and ap[k].max() < 1e-12), 0)
    e0all = a2[k0][:, :2]
    if abs(np.linalg.norm(e0all[0]) - 1.0) > 1e-6:
        raise SystemExit("reference material axis not found in the record")
    print(f"  reference in-plane axis from increment {k0}: {np.round(e0all[0], 4)}")

    omMax = float(om[iLast].max())
    # a common frame for every panel, so the specimens share a baseline and one length scale
    allV = np.concatenate([(verts[0] + defScale * (verts[k] - verts[0])).reshape(-1, 2)
                           for k in stages])
    x0, x1 = allV[:, 0].min() - 2, allV[:, 0].max() + 2
    y0, y1 = -2.0, allV[:, 1].max() + 2

    # a dedicated narrow column for the colour bar, so it cannot overlap the last specimen
    aspect = (y1 - y0) / (x1 - x0)
    fig = plt.figure(figsize=(2.35 * len(stages) + 1.0, 2.35 * aspect + 1.5))
    gs = fig.add_gridspec(1, len(stages) + 1,
                          width_ratios=[1] * len(stages) + [0.06], wspace=0.06)
    ax = np.array([fig.add_subplot(gs[0, j]) for j in range(len(stages))])
    cax = fig.add_subplot(gs[0, len(stages)])
    for a, k in zip(ax, stages):
        cm = fastCmap() if cmapName == "fast" else cmapName
        vk = verts[0] + defScale * (verts[k] - verts[0])
        pc = PolyCollection(vk, array=om[k], cmap=cm, edgecolors="0.45", linewidths=0.15)
        pc.set_clim(0.0, max(omMax, 1e-6))
        a.add_collection(pc)
        xy = xy0 + defScale * u[k]
        span = 2.1
        F = fitF(verts[0], verts[k])
        tot = np.einsum("nij,nj->ni", F, e0all)
        tot /= np.maximum(np.linalg.norm(tot, axis=1)[:, None], 1e-30)
        if which in ("both", "plastic"):
            v = a2[k][:, :2]
            # drawn LONGER and thicker than the total one, which is laid on top: where the two
            # coincide only a magenta shadow shows at the ends, and where they diverge the pair
            # opens into a visible V
            sp = span * (1.35 if which == "both" else 1.0)
            a.add_collection(LineCollection(
                np.stack([xy - sp * v, xy + sp * v], axis=1),
                colors=C_PLASTIC, linewidths=2.3 if which == "both" else 1.4,
                path_effects=PE))
        if which in ("both", "total"):
            a.add_collection(LineCollection(
                np.stack([xy - span * tot, xy + span * tot], axis=1),
                colors=C_TOTAL, linewidths=1.5 if which == "both" else 1.3,
                path_effects=PE))
        if which == "both":
            v = a2[k][:, :2]
            dev = np.degrees(np.arccos(np.clip(np.abs(np.einsum("ni,ni->n", v, tot)), 0, 1)))
            print(f"    {short[k]*100:6.2f} %: plastic vs total differ by max {dev.max():5.2f}, "
                  f"mean {dev.mean():5.2f} deg")
        a.set_xlim(x0, x1)
        a.set_ylim(y0, y1)
        a.set_aspect("equal")
        a.set_axis_off()                      # no box, no ticks, no annotations
        if iUnload is not None and k == iUnload:
            a.set_title(f"unloaded, $\\sigma_{{\\rm dev}}\\!\\approx\\!0$\n"
                        f"$\\varepsilon_{{yy}}$ = {short[k] * 100:.2f} %", fontsize=10)
        else:
            # t/t_end from the PRESCRIBED shortening, not from the increment index: the axial
            # displacement is imposed at a constant rate so time is proportional to it, whereas
            # the increments cluster heavily where the stepper cuts back (in the band), which made
            # index fractions read 0.07 ... 0.24 for states spanning most of the loading.
            a.set_title(f"$t/t_{{\\rm end}}$ = {short[k] / max(short[iLast], 1e-30):.2f}\n"
                        f"$\\varepsilon_{{yy}}$ = {short[k] * 100:.2f} %", fontsize=10)
    hs = []
    if which in ("both", "plastic"):
        hs.append(Line2D([], [], color=C_PLASTIC, lw=2.2, path_effects=PE,
                         label=r"stored (plastic) bedding direction  $e(F^{\rm p})$"))
    if which in ("both", "total"):
        hs.append(Line2D([], [], color=C_TOTAL, lw=2.2, path_effects=PE,
                         label=r"total bedding direction  $F e_0/\|F e_0\|$"))
    fig.legend(handles=hs, loc="lower center", ncol=2, fontsize=9.5, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    cb = fig.colorbar(pc, cax=cax)
    cb.set_label(r"damage $\omega$", fontsize=10)
    cb.ax.tick_params(labelsize=8)
    fig.subplots_adjust(left=0.01, right=0.93, top=0.90, bottom=0.11)
    suffix = "" if which == "both" else f"_{which}"
    out = os.path.join(HERE, f"band_evolution_b{beta}_{mk}{suffix}.png")
    fig.savefig(out, dpi=145, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
