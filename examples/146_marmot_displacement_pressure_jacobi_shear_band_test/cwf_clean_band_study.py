"""CWF boundary-correction clean-band study for the plane-strain compression
shear band (example 146), u-p-J SQCNIxNSNI particle, LOCAL finite-strain J2 Voce
softening.

Motivation (handoff_vms.md Part 7): the compression pressure field oscillates far
more than the elastic Cook's membrane. Measured cause is NOT a wrong stabilization
parameter C (= alpha h^2 / 2 G_ref, frozen elastic G, correct sign) but
  (1) the material is only mildly incompressible here (K/G ~ 50 vs ~4000 in Cook's),
      so at equal alpha the OSS checkerboard suppression 1+(alpha/2)(K/G) is ~80x
      weaker -- raising alpha barely helps; and
  (2) the DOMINANT oscillation is a weak-Dirichlet CONSISTENCY boundary layer at the
      two constrained loading edges (top driven, bottom roller), which alpha cannot
      touch. The consistent-weak-form (CWF) traction term on those edges removes it
      (elastically: cb_lap 52% -> 10%, edge_osc 6.5 -> 2.0; Nitsche -> exact uniform).

This script measures whether CWF also helps the NONLINEAR localizing problem: how far
each variant gets on the softening branch, and the pressure-field / band quality
there. mode 0, alpha = 0.02 (the robust default), displacement control.

Reaction caveat: with CWF active the exported mortar multiplier reaction holds only
the multiplier part (the CWF traction integral is missing). For a FAIR cross-variant
load-displacement curve we therefore record a BC-INDEPENDENT section reaction:
F_sec(t) = (l/nX) * sum_i sigma_yy over the mid-height particle row. Validated against
the true multiplier reaction of the cwf=off run at the peak (proxy ~ +6 %).

Run:  python cwf_clean_band_study.py
Outputs (workspace study dir): cwf_loaddisp.png, cwf_fields.png, printed summary.
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

# C++ particles keep bare references to the approximation wrapper -> keep alive
_instances = []
_orig_wrap = ex.MarmotMeshfreeApproximationWrapper
ex.MarmotMeshfreeApproximationWrapper = lambda *a, **k: (_instances.append(_orig_wrap(*a, **k)) or _instances[-1])

L = 60.0  # domain width (mm)


class _SectionMonitor(ex._ReactionMonitor):
    """Adds a BC-independent mid-height section reaction F_sec to the base monitor,
    sampled every converged increment from the particle Cauchy stresses."""

    def _section_reaction(self):
        ps = list(self._model.particles.values())
        ys = np.array([p.getCenterCoordinates()[1] for p in ps])
        ymid = 0.5 * (ys.min() + ys.max())
        rowy = ys[np.argmin(np.abs(ys - ymid))]
        sel = [p for p, y in zip(ps, ys) if abs(y - rowy) < 1e-6]
        syy = sum(p.getResultArray("stress")[4] for p in sel)  # index 4 = sigma_yy of 3x3
        return (L / len(sel)) * syy

    def finalizeIncrement(self):
        super().finalizeIncrement()
        if not hasattr(self, "F_section"):
            self.F_section = []
        try:
            self.F_section.append(float(self._section_reaction()))
        except Exception:
            self.F_section.append(np.nan)


ex._ReactionMonitor = _SectionMonitor  # run_sim instantiates our subclass


def metrics(P):
    """P: pressure at particle centers, shape (nX, nY)."""
    rms_p = np.sqrt(np.mean((P - P.mean()) ** 2))
    lap = P[1:-1, 1:-1] - 0.25 * (P[2:, 1:-1] + P[:-2, 1:-1] + P[1:-1, 2:] + P[1:-1, :-2])
    cb_lap = np.sqrt(np.mean(lap ** 2)) / rms_p
    edge = np.concatenate([P[:, :2].ravel() - P[:, :2].mean(), P[:, -2:].ravel() - P[:, -2:].mean()])
    edge_osc = np.sqrt(np.mean(edge ** 2))
    return cb_lap, edge_osc


NX, NY = 10, 20
ALPHA, MODE, INC, COMP = 0.02, 0, 0.01, -1.5
VARIANTS = ["off", "bottom", "both"]
STUDY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                         "compression_stabilization_study")
STUDY_DIR = os.path.abspath(STUDY_DIR)


def main():
    os.makedirs(STUDY_DIR, exist_ok=True)
    res = {}
    for cwf in VARIANTS:
        print(f"\n########## cwf = {cwf} ##########", flush=True)
        try:
            m, foc, mon = ex.run_sim(nX=NX, nY=NY, vmsAlpha=ALPHA, vmsMode=MODE, incSize=INC,
                                     totalCompression=COMP, cwfCorrection=cwf,
                                     outputName=f"cleanband_cwf-{cwf}")
            u = np.abs(np.array(mon.u_history))
            Fmult = np.array(mon.F_history)
            Fsec = np.array(getattr(mon, "F_section", [np.nan] * len(u)))
            P = np.array(foc.fieldOutputs["pressure"].getLastResult()).reshape(NX, NY)
            aP = np.array(foc.fieldOutputs["alphaP"].getLastResult()).reshape(NX, NY)
            cb, eo = metrics(P)
            ipk = int(np.nanargmax(np.abs(Fsec)))
            res[cwf] = dict(u=u, Fmult=Fmult, Fsec=Fsec, P=P, aP=aP, cb=cb, eo=eo,
                            reached=u.max(), upeak=u[ipk], Fpeak=Fsec[ipk],
                            Fmult_pk=Fmult[ipk], Ffinal=Fsec[-1], failed=False)
            print(f"  reached u={u.max():.3f}/{abs(COMP):.2f}  peak |Fsec|={abs(Fsec[ipk]):.0f} "
                  f"(mult {abs(Fmult[ipk]):.0f}) at u={u[ipk]:.3f}  final Fsec={Fsec[-1]:.0f}  "
                  f"cb_lap {cb*100:.1f}%  edge_osc {eo:.2f}  aPmax {aP.max():.3f}", flush=True)
        except Exception:
            traceback.print_exc()
            res[cwf] = dict(failed=True)
        gc.collect()

    # ---------------- load-displacement ----------------
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    col = {"off": "tab:red", "bottom": "tab:orange", "both": "tab:green"}
    for cwf in VARIANTS:
        r = res.get(cwf, {})
        if r.get("failed"):
            continue
        ax.plot(r["u"], -r["Fsec"], "-", color=col[cwf], lw=1.8,
                label=f"cwf={cwf}: reached {r['reached']:.2f} mm")
        ax.plot(r["reached"], -r["Fsec"][-1], "o", color=col[cwf], ms=6)
    ax.set_xlabel(r"compressive shortening $-u_y$ (mm)")
    ax.set_ylabel(r"mid-section reaction $-F_y$ (N/mm)  [BC-independent proxy]")
    ax.set_title(f"CWF lets the increment-form tree cross the localization peak\n"
                 f"(ex146, {NX}x{NY}, mode {MODE}, alpha {ALPHA}, incSize {INC})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(STUDY_DIR, "cwf_loaddisp.png"), dpi=130)
    plt.close(fig)

    # ---------------- pressure + band fields at final state ----------------
    ok = [c for c in VARIANTS if not res.get(c, {}).get("failed")]
    fig, axs = plt.subplots(2, len(ok), figsize=(3.1 * len(ok), 6.4), squeeze=False)
    for k, cwf in enumerate(ok):
        r = res[cwf]
        Pd = r["P"] - r["P"].mean()
        v = max(np.abs(Pd).max(), 1e-6)
        im0 = axs[0, k].imshow(Pd.T, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v)
        axs[0, k].set_title(f"cwf={cwf}  (u={r['reached']:.2f})\npressure p-p̄  [cb {r['cb']*100:.0f}%, "
                            f"edge {r['eo']:.1f}]", fontsize=8.5)
        plt.colorbar(im0, ax=axs[0, k], fraction=0.046, pad=0.04)
        im1 = axs[1, k].imshow(r["aP"].T, origin="lower", aspect="auto", cmap="inferno")
        axs[1, k].set_title(f"alphaP (band)  max {r['aP'].max():.2f}", fontsize=8.5)
        plt.colorbar(im1, ax=axs[1, k], fraction=0.046, pad=0.04)
        for ax in (axs[0, k], axs[1, k]):
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Pressure fluctuation (top) and plastic band (bottom) at the furthest converged state", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(STUDY_DIR, "cwf_fields.png"), dpi=130)
    plt.close(fig)

    # ---------------- summary ----------------
    print("\n\n================= CWF clean-band summary "
          f"(ex146 {NX}x{NY}, mode {MODE}, alpha {ALPHA}, incSize {INC}, comp {COMP}) =================")
    print(f"{'cwf':8} {'reached':>8} {'u_peak':>7} {'|Fpeak|':>8} {'Ffinal':>8} {'cb_lap%':>8} {'edge_osc':>9} {'aPmax':>7}")
    for cwf in VARIANTS:
        r = res.get(cwf, {})
        if r.get("failed"):
            print(f"{cwf:8} FAILED"); continue
        print(f"{cwf:8} {r['reached']:8.3f} {r['upeak']:7.3f} {abs(r['Fpeak']):8.0f} "
              f"{-r['Ffinal']:8.0f} {r['cb']*100:8.1f} {r['eo']:9.2f} {r['aP'].max():7.3f}")
    print(f"\nfigures -> {STUDY_DIR}/cwf_loaddisp.png , cwf_fields.png")
    np.savez(os.path.join(STUDY_DIR, "cwf_clean_band_study.npz"),
             **{f"{c}_{k}": v for c in VARIANTS if not res.get(c, {}).get("failed")
                for k, v in res[c].items() if k in ("u", "Fsec", "Fmult", "P", "aP")})


if __name__ == "__main__":
    main()
