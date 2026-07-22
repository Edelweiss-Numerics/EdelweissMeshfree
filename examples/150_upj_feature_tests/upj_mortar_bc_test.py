"""Feature: mortar weak Dirichlet BC (ParticleMortarWeakDirichlet) -- a smooth
multiplier field enforcing the boundary displacement at the particle centers,
avoiding a forced boundary-pressure checkerboard."""
from _upj_base import assert_sane_compression, run


def test_upj_mortar_bc():
    _, foc, _ = run(constraintType="mortar")
    assert_sane_compression(foc)


if __name__ == "__main__":
    test_upj_mortar_bc()
    print("OK: mortar weak Dirichlet BC")
