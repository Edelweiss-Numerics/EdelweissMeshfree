"""Shared tiny u-p-J plane-strain compression problem for the per-feature
verification tests in this directory.

Each ``upj_<feature>_test.py`` calls :func:`run` with ONE feature toggled and
asserts a sane result. Target runtime: < 3 s per test (typically ~0.2-0.5 s).

Rationale: the large Cook's-membrane (example 134) and plane-strain-compression
shear-band (example 146) scripts are numerical STUDIES -- they exercise every
feature at once and take many seconds. These files are the opposite: one tiny,
fast, isolated check per feature, so a regression in any single feature is caught
immediately and adding a test for a NEW feature is a copy-paste of one short file.

They reuse example 146's ``run_sim`` (the reference u-p-J implementation) at a
4x8, nearly-elastic, 2-increment scale. The exact solution of this homogeneous
compression is a near-uniform (negative) pressure field, so a finite field with a
negative mean is the basic sanity signal.
"""
import os
import sys

import numpy as np

_EX146 = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "146_marmot_displacement_pressure_jacobi_shear_band_test")
)
sys.path.insert(0, _EX146)
import marmot_displacement_pressure_jacobi_shear_band_test as _compression  # noqa: E402

# the C++ particles keep a bare reference to the approximation wrapper -> keep it alive
_instances = []
_orig_wrap = _compression.MarmotMeshfreeApproximationWrapper
_compression.MarmotMeshfreeApproximationWrapper = lambda *a, **k: (_instances.append(_orig_wrap(*a, **k)) or _instances[-1])

# tiny + nearly elastic (gentle softening effectively off) + 2 increments -> fast
_TINY = dict(nX=4, nY=8, totalCompression=-0.05, incSize=0.5, multiplierOrder=2, eta=1.0, fyInf=99.0)


def run(**overrides):
    """Run the tiny u-p-J compression with the given feature flags; returns
    (model, fieldOutputController, reactionMonitor)."""
    kw = dict(_TINY)
    kw.update(overrides)
    tag = "_".join(f"{k}{v}" for k, v in sorted(overrides.items()) if k != "outputName")
    kw.setdefault("outputName", ("upjtest_" + tag).replace("/", "-")[:60])
    return _compression.run_sim(**kw)


def pressure(foc):
    return np.asarray(foc.fieldOutputs["pressure"].getLastResult()).ravel()


def assert_sane_compression(foc, model=None):
    """Basic verification shared by all feature tests: the pressure field is finite
    and its mean is compressive (negative, tension-positive convention)."""
    p = pressure(foc)
    assert np.isfinite(p).all(), "non-finite pressure -> the feature broke the solve"
    assert p.mean() < 0.0, f"expected compressive (negative) mean pressure, got {p.mean():.4f}"
    return p
