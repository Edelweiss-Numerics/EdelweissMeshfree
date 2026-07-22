"""K-aware stabilization sweep for the u-p-J plane-strain compression (example 146).

Background (handoff_vms.md Part 7): C = alpha h^2/(2 G_ref) gives a
checkerboard-suppression factor 1+(alpha/2)(K/G). This material is only mildly
incompressible (K/G~50), so at the robust alpha=0.02 the coherent checkerboard is
barely suppressed, while cranking alpha fights localization-peak crossing. The
K-aware option (Marmot 6fb6021, --vmsKawareRatio R) caps the effective
incompressibility: G_eff = min(G_ref, K_ref/R), so the suppression factor becomes
1+(alpha/2)R INDEPENDENT of the material -- checkerboard control without raising alpha.

This study shows (a) K-aware C removes the coherent CHECKERBOARD (cb_alt) as predicted,
(b) it is COMPLEMENTARY to CWF, which removes the loading-edge BOUNDARY layer (edge_osc)
-- neither alone gives a clean field, together they do, and (c) in the plastic band
regime K-aware does not break the CWF peak crossing.

Run:  python kaware_sweep_study.py
Outputs (workspace study dir): kaware_sweep.png, printed summary.
"""
import gc
import os
import sys
import traceback

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import marmot_displacement_pressure_jacobi_shear_band_test as ex

_instances = []
_orig_wrap = ex.MarmotMeshfreeApproximationWrapper
ex.MarmotMeshfreeApproximationWrapper = lambda *a, **k: (_instances.append(_orig_wrap(*a, **k)) or _instances[-1])

NX, NY = 10, 20
ALPHA = 0.02
RS = [0.0, 100.0, 300.0, 1000.0, 3000.0]
STUDY_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "..", "..", "..", "compression_stabilization_study"))


def metrics(P):
    """cb_alt: coherent (+/-) checkerboard (what K-aware targets), normalized.
    edge_osc: absolute RMS oscillation of the top/bottom 2 rows (what CWF targets).
    int_std: absolute std of the interior (rows 2..-2) pressure -- overall interior noise."""
    rms_p = np.sqrt(np.mean((P - P.mean()) ** 2))
    sgn = (-1.0) ** (np.add.outer(np.arange(P.shape[0]), np.arange(P.shape[1])))
    cb_alt = abs(np.mean(P * sgn)) / rms_p
    edge = np.concatenate([P[:, :2].ravel() - P[:, :2].mean(), P[:, -2:].ravel() - P[:, -2:].mean()])
    edge_osc = np.sqrt(np.mean(edge ** 2))
    Pi = P[:, 2:-2]
    # interior checkerboard amplitude (absolute): coherent alternating part
    sgni = (-1.0) ** (np.add.outer(np.arange(Pi.shape[0]), np.arange(Pi.shape[1])))
    int_cb_abs = abs(np.mean(Pi * sgni))
    return cb_alt, edge_osc, int_cb_abs


def run(cwf, R, comp=-0.15, inc=0.05):
    m, foc, mon = ex.run_sim(nX=NX, nY=NY, vmsAlpha=ALPHA, vmsMode=0, vmsKawareRatio=R,
                             cwfCorrection=cwf, totalCompression=comp, incSize=inc,
                             outputName=f"kaware_cwf-{cwf}_R{R:.0f}_c{abs(comp)}")
    P = np.array(foc.fieldOutputs["pressure"].getLastResult()).reshape(NX, NY)
    aP = np.array(foc.fieldOutputs["alphaP"].getLastResult()).reshape(NX, NY)
    u = np.abs(np.array(mon.u_history))
    return P, aP, u.max()


def main():
    os.makedirs(STUDY_DIR, exist_ok=True)
    # ---------- elastic R sweep, cwf off vs both ----------
    elastic = {"off": {}, "both": {}}
    for cwf in ("off", "both"):
        for R in RS:
            try:
                P, _, _ = run(cwf, R)
                cb_alt, edge_osc, int_cb = metrics(P)
                elastic[cwf][R] = dict(cb_alt=cb_alt, edge_osc=edge_osc, int_cb=int_cb,
                                       pmin=float(P.min()), pmax=float(P.max()), P=P)
                print(f"[elastic] cwf={cwf:4} R={R:6.0f}  cb_alt {cb_alt*100:6.3f}%  "
                      f"edge_osc {edge_osc:5.2f}  int_cb {int_cb:5.3f}  p[{P.min():6.1f},{P.max():5.1f}]", flush=True)
            except Exception:
                traceback.print_exc()
            gc.collect()

    # ---------- plastic band regime: does K-aware break CWF peak crossing? ----------
    plastic = {}
    for R in (0.0, 500.0):
        try:
            P, aP, reached = run("both", R, comp=-1.2, inc=0.01)
            cb_alt, edge_osc, int_cb = metrics(P)
            plastic[R] = dict(reached=reached, cb_alt=cb_alt, edge_osc=edge_osc,
                              aPmax=float(aP.max()), P=P, aP=aP)
            print(f"[plastic] cwf=both R={R:6.0f}  reached {reached:.3f}/1.20  "
                  f"cb_alt {cb_alt*100:.3f}%  edge_osc {edge_osc:.2f}  aPmax {aP.max():.3f}", flush=True)
        except Exception:
            traceback.print_exc()
        gc.collect()

    # ---------- figure ----------
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axs[0]
    for cwf, mk in (("off", "o-"), ("both", "s-")):
        Rv = [R for R in RS if R in elastic[cwf]]
        ax.plot([max(r, 30) for r in Rv], [elastic[cwf][r]["cb_alt"] * 100 for r in Rv], mk,
                label=f"checkerboard cb_alt, cwf={cwf}")
    ax.set_xscale("log")
    ax.set_xlabel("K-aware ratio R  (suppression factor 1+α/2·R)")
    ax.set_ylabel("coherent checkerboard cb_alt [%]")
    ax.set_title(f"K-aware C removes the checkerboard\n(ex146 elastic, {NX}x{NY}, α={ALPHA})")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8.5)

    ax = axs[1]
    Rv = [R for R in RS if R in elastic["off"]]
    ax.plot([max(r, 30) for r in Rv], [elastic["off"][r]["edge_osc"] for r in Rv], "o-", label="edge_osc, cwf=off")
    ax.plot([max(r, 30) for r in Rv], [elastic["both"][r]["edge_osc"] for r in Rv], "s-", label="edge_osc, cwf=both")
    ax.set_xscale("log")
    ax.set_xlabel("K-aware ratio R")
    ax.set_ylabel("loading-edge oscillation edge_osc")
    ax.set_title("boundary layer needs CWF, not R\n(K-aware & CWF are complementary)")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(STUDY_DIR, "kaware_sweep.png"), dpi=130)
    plt.close(fig)

    print("\n================= K-aware summary (ex146 elastic + plastic, mode 0, alpha 0.02) =================")
    print("ELASTIC  cb_alt[%] (checkerboard) / edge_osc (boundary):")
    for cwf in ("off", "both"):
        row = "  cwf=%-4s " % cwf + "  ".join(
            f"R{int(R)}:{elastic[cwf][R]['cb_alt']*100:.3f}/{elastic[cwf][R]['edge_osc']:.1f}"
            for R in RS if R in elastic[cwf])
        print(row)
    print("PLASTIC band (cwf=both, comp -1.2):")
    for R in (0.0, 500.0):
        if R in plastic:
            p = plastic[R]
            print(f"  R={int(R):5d}  reached {p['reached']:.3f}  cb_alt {p['cb_alt']*100:.3f}%  "
                  f"edge_osc {p['edge_osc']:.2f}  aPmax {p['aPmax']:.3f}")
    print(f"\nfigure -> {STUDY_DIR}/kaware_sweep.png")


if __name__ == "__main__":
    main()
