"""Feature: Lagrange-multiplier weak Dirichlet BC
(ParticleLagrangianWeakDirichlet) -- point-collocation enforcement of the boundary
displacement via Lagrange multipliers."""
from _upj_base import assert_sane_compression, run


def test_upj_lagrange_bc():
    _, foc, _ = run(constraintType="lagrange")
    assert_sane_compression(foc)


if __name__ == "__main__":
    test_upj_lagrange_bc()
    print("OK: Lagrange-multiplier weak Dirichlet BC")
