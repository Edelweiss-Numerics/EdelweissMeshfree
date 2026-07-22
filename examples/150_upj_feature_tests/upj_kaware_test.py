"""Feature: K-aware / bounded stabilization coefficient C. Caps the effective
incompressibility at a target ratio R = vmsKawareRatio so the checkerboard
suppression factor is 1+(alpha/2)R independent of the material's K/G. Verifies the
property is accepted (R>0) and the solve stays sane; R=0 is the plain elastic-G form."""
from _upj_base import assert_sane_compression, run


def test_upj_kaware():
    # R = 0 (disabled) and R = 500 (active) must both run and stay sane
    _, foc0, _ = run(vmsKawareRatio=0.0, vmsAlpha=0.02)
    assert_sane_compression(foc0)
    _, focR, _ = run(vmsKawareRatio=500.0, vmsAlpha=0.02)
    assert_sane_compression(focR)


if __name__ == "__main__":
    test_upj_kaware()
    print("OK: K-aware stabilization (R=0 and R=500)")
