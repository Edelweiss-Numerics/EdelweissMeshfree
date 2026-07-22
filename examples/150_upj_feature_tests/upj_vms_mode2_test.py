"""Feature: u-p-J 'vms mode' 2 -- fully stabilized VMS. Adds the momentum- and
jacobi-equation consistency terms of the derivation on top of mode 1 (use small
alpha; the momentum term bounds the stability envelope)."""
from _upj_base import assert_sane_compression, run


def test_upj_vms_mode2():
    _, foc, _ = run(vmsMode=2, vmsAlpha=0.02)
    assert_sane_compression(foc)


if __name__ == "__main__":
    test_upj_vms_mode2()
    print("OK: vms mode 2 (fully stabilized: momentum + jacobi terms)")
