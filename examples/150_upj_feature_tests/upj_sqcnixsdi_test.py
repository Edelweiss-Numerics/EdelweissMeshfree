"""Feature: SQCNIxSDI u-p-J particle -- the mixed 3-field (u, p, J) particle using
SUB-DOMAIN INTEGRATION (SDI, with an OSS split on the subdomain-gradient
fluctuations) instead of nodal integration (NSNI)."""
from _upj_base import assert_sane_compression, run

_SDI = "DisplacementPressureJacobiSQCNIxSDI/PlaneStrain/Quad"


def test_upj_sqcnixsdi():
    _, foc, _ = run(particleType=_SDI, vmsAlpha=0.02)
    assert_sane_compression(foc)


if __name__ == "__main__":
    test_upj_sqcnixsdi()
    print("OK: SQCNIxSDI u-p-J particle")
