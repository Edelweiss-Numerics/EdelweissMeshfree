"""Feature (experimental, host-free): referential-gradient accumulation of the
total pressure gradient. With ``vmsRefgrad=1`` the full-VMS resolved-scale
residual R_res uses the TOTAL pressure gradient reconstructed from the material
point's accumulated referential gradient G = grad_X p (advected, drift-free,
pushed forward by the total F), instead of the step-increment gradient. Only the
mode >= 1 resolved term is affected; the mode-0 face fluctuation penalty is
untouched. See handoff_vms Part 11.6. Default (vmsRefgrad=0) is unchanged.

This checks the resolved-scale modes (1/2/3) still produce a finite, compressive
pressure field with the option on -- i.e. the accumulated referential gradient
does NOT self-excite here (contrast the pre-0bb728a particle-side accumulation,
handoff Part 6.1, which pre-dated the corrected stabilization sign).
"""
import pytest

from _upj_base import assert_sane_compression, run


@pytest.mark.parametrize("vmsMode", [1, 2, 3])
def test_upj_refgrad(vmsMode):
    _, foc, _ = run(vmsMode=vmsMode, vmsAlpha=0.02, vmsRefgrad=1.0)
    assert_sane_compression(foc)


if __name__ == "__main__":
    for m in (1, 2, 3):
        test_upj_refgrad(m)
        print(f"OK: vms refgrad (referential-gradient total), mode {m}")
