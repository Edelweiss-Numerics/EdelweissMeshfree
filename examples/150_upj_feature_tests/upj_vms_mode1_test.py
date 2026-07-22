"""Feature: u-p-J 'vms mode' 1 -- full VMS. Adds the resolved-scale strong-form
momentum residual grad(p) + div(S_dev) - rho0*a on top of the mode-0 penalty."""
from _upj_base import assert_sane_compression, run


def test_upj_vms_mode1():
    _, foc, _ = run(vmsMode=1, vmsAlpha=0.02)
    assert_sane_compression(foc)


if __name__ == "__main__":
    test_upj_vms_mode1()
    print("OK: vms mode 1 (full VMS momentum residual)")
