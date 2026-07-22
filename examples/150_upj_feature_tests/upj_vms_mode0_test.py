"""Feature: u-p-J particle with 'vms mode' 0 -- pressure-only OSS stabilization
(the classical PSPG-type grad(p) fluctuation penalty). Default u-p-J variant."""
from _upj_base import assert_sane_compression, run


def test_upj_vms_mode0():
    _, foc, _ = run(vmsMode=0, vmsAlpha=0.02)
    assert_sane_compression(foc)


if __name__ == "__main__":
    test_upj_vms_mode0()
    print("OK: vms mode 0 (pressure-only OSS)")
