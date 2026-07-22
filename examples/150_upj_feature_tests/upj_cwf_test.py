"""Feature: consistent-weak-form (CWF) boundary correction -- a consistency
traction integral on the constrained loading edges that removes the weak-Dirichlet
boundary-layer pressure oscillation (here on both the top and bottom edges)."""
from _upj_base import assert_sane_compression, run


def test_upj_cwf():
    _, foc, _ = run(cwfCorrection="both")
    assert_sane_compression(foc)


if __name__ == "__main__":
    test_upj_cwf()
    print("OK: CWF boundary correction (both edges)")
