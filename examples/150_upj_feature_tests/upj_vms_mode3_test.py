"""Feature: u-p-J 'vms mode' 3 -- mode 2 plus the finite-difference SECOND-ORDER
material tangent d2(tau)/dF2 for the VMS terms (exact-tangent reference; identical
residual to mode 2)."""
from _upj_base import assert_sane_compression, run


def test_upj_vms_mode3():
    _, foc, _ = run(vmsMode=3, vmsAlpha=0.02)
    assert_sane_compression(foc)


if __name__ == "__main__":
    test_upj_vms_mode3()
    print("OK: vms mode 3 (mode 2 + FD second-order material tangent)")
