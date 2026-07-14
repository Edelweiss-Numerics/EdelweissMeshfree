"""VMS mode study: pressure-field quality of 'vms mode' 0 / 1 / 2 on Cook's membrane.

Runs example 134 (12x12, uYTip=30 -- the strong checkerboard / boundary-layer
regime) for the three VMS stabilization levels at the same alpha and reports the
pressure-field metrics of cooks_nitsche_study.py:

- mode 0: pressure-equation fluctuation (local OSS) penalty only
- mode 1: + resolved-scale strong-form momentum residual in the pressure equation
- mode 2: + momentum- and jacobi-equation stabilization terms (fully stabilized
          form of the corrected u-p-J VMS derivation; jacobi term inert for Fbar)

NOTE mode 2 bounds the usable alpha (the momentum consistency term subtracts a
C |L u|^2-like contribution): alpha = 0.02 is robust, alpha = 0.1 fails on coarse
grids. Default here: 0.02 for a fair three-way comparison.

Writes cooks_vms_mode_study.npz (P fields, metrics) and a pressure-map figure.
"""

import os
import sys
import traceback

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import marmot_displacement_pressure_jacobi_cooks_membrane_test as ex
from cooks_nitsche_study import metrics

# the C++ particles keep bare references to the approximation wrapper -> keep alive
_instances = []
_orig = ex.MarmotMeshfreeApproximationWrapper


def _keepAlive(*a, **k):
    inst = _orig(*a, **k)
    _instances.append(inst)
    return inst


ex.MarmotMeshfreeApproximationWrapper = _keepAlive

NX = NY = 12
UYTIP = 30.0
ALPHAS = (0.02, 0.05, 0.1)


def main():
    results = {}
    for alpha in ALPHAS:
      for mode in (0, 1, 2):
        label = f"a{alpha} mode {mode}"
        print(f"\n########## running {label} (uYTip={UYTIP}) ##########", flush=True)
        try:
            model, foc = ex.run_sim(
                nX=NX, nY=NY, uYTip=UYTIP, vmsAlpha=alpha, vmsMode=mode, outputName=f"vmsstudy_a{alpha}_mode{mode}"
            )
            P = foc.fieldOutputs["pressure"].getLastResult().reshape(NX, NY)
            U = foc.fieldOutputs["U_Right"].getLastResult()
            cb_lap, cb_alt, col_osc = metrics(P)
            results[label] = dict(
                P=P, U=float(np.atleast_1d(U).ravel()[-1]), cb_lap=cb_lap, cb_alt=cb_alt, col_osc=col_osc, failed=False
            )
        except Exception:
            traceback.print_exc()
            results[label] = dict(failed=True)
        import gc

        gc.collect()

    print(f"\n\n========== SUMMARY ({NX}x{NY}, uYTip={UYTIP}) ==========")
    print(f"{'variant':16s} {'u_tip':>7s} {'cb_lap':>8s} {'cb_alt':>8s}  osc(col 0..3)")
    for label, r in results.items():
        if r.get("failed"):
            print(f"{label:16s}   FAILED")
            continue
        osc = " ".join(f"{v:5.2f}" for v in r["col_osc"][:4])
        print(f"{label:16s} {r['U']:7.2f} {r['cb_lap'] * 100:7.1f}% {r['cb_alt'] * 100:7.2f}%  {osc}")

    np.savez(
        "cooks_vms_mode_study.npz",
        **{
            f"P_a{a}_mode{m}": results[f"a{a} mode {m}"]["P"]
            for a in ALPHAS
            for m in (0, 1, 2)
            if not results[f"a{a} mode {m}"].get("failed")
        },
        summary=str({k: {kk: vv for kk, vv in v.items() if kk != "P"} for k, v in results.items()}),
    )

    fig, axes = plt.subplots(len(ALPHAS), 3, figsize=(13, 4 * len(ALPHAS)), constrained_layout=True)
    for row, a in enumerate(ALPHAS):
        ok = [m for m in (0, 1, 2) if not results[f"a{a} mode {m}"].get("failed")]
        vmax = max(np.abs(results[f"a{a} mode {m}"]["P"]).max() for m in ok) if ok else 1.0
        for m in (0, 1, 2):
            ax = axes[row, m]
            r = results[f"a{a} mode {m}"]
            if r.get("failed"):
                ax.text(0.5, 0.5, "FAILED", ha="center", va="center", fontsize=16, color="crimson")
                ax.set_title(f"alpha {a}, vms mode {m}", fontsize=10)
                ax.set_axis_off()
                continue
            im = ax.imshow(r["P"].T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_title(
                f"alpha {a}, vms mode {m}\ncb_lap {r['cb_lap'] * 100:.1f}%, osc[0] {r['col_osc'][0]:.2f}", fontsize=10
            )
            fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle(f"Cook's membrane {NX}x{NY}, uYTip={UYTIP}: Kirchhoff pressure at particles")
    fig.savefig("cooks_vms_mode_study_pressure.png", dpi=160)
    print("\nwrote cooks_vms_mode_study.npz / cooks_vms_mode_study_pressure.png")


if __name__ == "__main__":
    main()
