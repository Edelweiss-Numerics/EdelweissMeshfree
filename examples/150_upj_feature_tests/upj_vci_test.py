"""Feature: variationally consistent integration (VCI) -- correction of the nodal
integration so the discrete divergence theorem is satisfied (order 1 here), applied
over the specimen boundary edges of the u-p-J particle."""
from _upj_base import assert_sane_compression, run


def test_upj_vci():
    _, foc, _ = run(vci=True, vciOrder=1, completenessOrder=1)
    assert_sane_compression(foc)


if __name__ == "__main__":
    test_upj_vci()
    print("OK: variationally consistent integration (order 1)")
