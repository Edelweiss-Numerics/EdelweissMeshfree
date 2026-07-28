"""Feature: Nitsche Dirichlet BC (NITSCHEDIRICHLET). Enforces the boundary
displacement at the particle FACES via a consistency (CWF) traction + a penalty
beta*(u_face - g); the gap is built from tracked particle geometry, NOT the drifted
total nodal coefficients (the kernel-drift trap).

This feature lives on the ``tom/feat/nitsche-dirichlet`` branch (Marmot particle +
materialpoint + MarmotMeshfreeCore + this step action); it is not on the VMS branch.
"""
from _upj_base import assert_sane_compression, run


def test_upj_nitsche_bc():
    _, foc, _ = run(constraintType="nitsche", nitscheBeta=10.0)
    assert_sane_compression(foc)


if __name__ == "__main__":
    test_upj_nitsche_bc()
    print("OK: Nitsche Dirichlet BC")
