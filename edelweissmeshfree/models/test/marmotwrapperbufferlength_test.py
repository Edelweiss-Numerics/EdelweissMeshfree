# -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
#
#  _____    _      _              _         __  __ ____  __  __
# | ____|__| | ___| |_      _____(_)___ ___|  \/  |  _ \|  \/  |
# |  _| / _` |/ _ \ \ \ /\ / / _ \ / __/ __| |\/| | |_) | |\/| |
# | |__| (_| |  __/ |\ V  V /  __/ \__ \__ \ |  | |  __/| |  | |
# |_____\__,_|\___|_| \_/\_/ \___|_|___/___/_|  |_|_|   |_|  |_|
#
#
#  Unit of Strength of Materials and Structural Analysis
#  University of Innsbruck,
#  2023 - today
#
#  Matthias Neuner matthias.neuner@uibk.ac.at
#
#  This file is part of EdelweissMeshfree.
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissMeshfree.
#  ---------------------------------------------------------------------
"""The Marmot wrappers of particles and cells refuse a buffer of the wrong length before Marmot reads or writes behind
its end: Marmot receives a bare pointer and writes as many entries as the entity has dofs.

Skipped where Marmot is not built. A Marmot cell element is not covered: Marmot registers none.
"""

import itertools

import numpy as np
import pytest
from edelweissfe.points.node import Node

from edelweissmeshfree.models.test.marmotentityconformance_test import _cell, _particle


def _shortenedByOneEntry(buffer: np.ndarray) -> np.ndarray:
    """The buffer with one entry (one row, for an array) less, in the same memory order."""
    return np.zeros((buffer.shape[0] - 1,) + buffer.shape[1:], order="F" if buffer.flags.f_contiguous else "C")


def _particleCalls(particle):
    """Every buffer-taking method of the particle: its buffers, and a call taking the buffers by name."""
    n, nDim = particle.nDof, particle.dimension
    nNodes, nC = len(particle.nodes), particle.vci_getNumberOfConstraints()

    return {
        "computeKernels": (
            dict(Kc=np.zeros(n * n), Pc=np.zeros(n), dUc=np.zeros(n)),
            lambda b: particle.computeKernels(b["Kc"], b["Pc"], np.zeros(n), b["dUc"], 1.0, 1.0),
        ),
        "updatePhysicsExplicit": (dict(dUc=np.zeros(n)), lambda b: particle.updatePhysicsExplicit(b["dUc"], 1.0, 1.0)),
        "computePhysicsKernelsExplicit": (
            dict(Pc=np.zeros(n)),
            lambda b: particle.computePhysicsKernelsExplicit(b["Pc"]),
        ),
        "computeLumpedInertia": (dict(mLumped=np.zeros(n)), lambda b: particle.computeLumpedInertia(b["mLumped"])),
        "computeLumpedMomentum": (dict(pLumped=np.zeros(n)), lambda b: particle.computeLumpedMomentum(b["pLumped"])),
        "computeLumpedDamping": (dict(cLumped=np.zeros(n)), lambda b: particle.computeLumpedDamping(b["cLumped"])),
        "assembleStiffness": (dict(K=np.zeros(n * n)), lambda b: particle.assembleStiffness(b["K"])),
        "addBoundOfAbsoluteStiffnessTimes": (
            {"W rows": np.zeros((n, 2), order="F"), "bound rows": np.zeros((n, 2), order="F")},
            lambda b: particle.addBoundOfAbsoluteStiffnessTimes(b["W rows"], b["bound rows"]),
        ),
        "computeBodyLoad": (
            dict(load=np.zeros(nDim), Pc=np.zeros(n), Kc=np.zeros(n * n)),
            lambda b: particle.computeBodyLoad("BODYFORCE", b["load"], b["Pc"], b["Kc"], 1.0, 1.0),
        ),
        "computeDistributedLoad": (
            dict(Pc=np.zeros(n), Kc=np.zeros(n * n)),
            lambda b: particle.computeDistributedLoad("PRESSURE", 1, np.ones(1), b["Pc"], b["Kc"], 1.0, 1.0),
        ),
        "getInterpolationVector": (
            dict(coordinates=np.full(nDim, 0.5)),
            lambda b: particle.getInterpolationVector(b["coordinates"]),
        ),
        "getInterpolationVectorLinearizedAboutCenter": (
            dict(coordinates=np.full(nDim, 0.5)),
            lambda b: particle.getInterpolationVectorLinearizedAboutCenter(b["coordinates"]),
        ),
        "vci_compute_TestGradient_P_Integral": (
            {"r_AiC nodes": np.zeros((nNodes, nDim, nC))},
            lambda b: particle.vci_compute_TestGradient_P_Integral(b["r_AiC nodes"]),
        ),
        "vci_compute_MMatrix": (
            {"m_ACD nodes": np.zeros((nNodes, nC, nC))},
            lambda b: particle.vci_compute_MMatrix(b["m_ACD nodes"]),
        ),
    }


def _cellCalls(cell):
    """Every buffer-taking method of the cell: its buffers, and a call taking the buffers by name."""
    n = cell.nDof

    return {
        "computeKernels": (
            dict(Kc=np.zeros(n * n), Pc=np.zeros(n), dUc=np.zeros(n)),
            lambda b: cell.computeKernels(b["Kc"], b["Pc"], np.zeros(n), b["dUc"], 1.0, 1.0),
        ),
        "interpolateFieldsToMaterialPoints": (
            dict(dUc=np.zeros(n)),
            lambda b: cell.interpolateFieldsToMaterialPoints(b["dUc"]),
        ),
        "computeLumpedInertia": (dict(M=np.zeros(n)), lambda b: cell.computeLumpedInertia(b["M"])),
        "computeConsistentInertia": (dict(M=np.zeros(n * n)), lambda b: cell.computeConsistentInertia(b["M"])),
        "computeBodyLoad": (
            dict(load=np.zeros(2), Pc=np.zeros(n), Kc=np.zeros(n * n)),
            lambda b: cell.computeBodyLoad("BODYFORCE", b["load"], b["Pc"], b["Kc"], 1.0, 1.0),
        ),
        "isCoordinateInCell": (
            dict(coordinate=np.full(2, 0.5)),
            lambda b: cell.isCoordinateInCell(b["coordinate"]),
        ),
        "getInterpolationVector": (
            dict(coordinate=np.full(2, 0.5)),
            lambda b: cell.getInterpolationVector(b["coordinate"]),
        ),
    }


particleMethods = [
    "computeKernels",
    "updatePhysicsExplicit",
    "computePhysicsKernelsExplicit",
    "computeLumpedInertia",
    "computeLumpedMomentum",
    "computeLumpedDamping",
    "assembleStiffness",
    "addBoundOfAbsoluteStiffnessTimes",
    "computeBodyLoad",
    "computeDistributedLoad",
    "getInterpolationVector",
    "getInterpolationVectorLinearizedAboutCenter",
    "vci_compute_TestGradient_P_Integral",
    "vci_compute_MMatrix",
]

cellMethods = [
    "computeKernels",
    "interpolateFieldsToMaterialPoints",
    "computeLumpedInertia",
    "computeConsistentInertia",
    "computeBodyLoad",
    "isCoordinateInCell",
    "getInterpolationVector",
]


@pytest.mark.parametrize(
    "makeEntity, calls, method",
    [pytest.param(_particle, _particleCalls, m, id=f"particle-{m}") for m in particleMethods]
    + [pytest.param(_cell, _cellCalls, m, id=f"cell-{m}") for m in cellMethods],
)
def test_every_buffer_one_entry_too_short_is_refused_and_not_written(makeEntity, calls, method):
    entity = makeEntity()
    buffers, call = calls(entity)[method]

    for name in buffers:
        shortened = dict(buffers)
        shortened[name] = _shortenedByOneEntry(buffers[name])

        with pytest.raises(ValueError, match=f"{method}: {name}"):
            call(shortened)

        assert not shortened[name].any()


@pytest.mark.parametrize(
    "makeEntity, calls, methods", [(_particle, _particleCalls, particleMethods), (_cell, _cellCalls, cellMethods)]
)
def test_the_parametrization_covers_every_call(makeEntity, calls, methods):
    assert sorted(calls(makeEntity())) == sorted(methods)


def test_buffers_of_the_required_length_give_the_particle_operators():
    """The lumped mass sums to rho * V per direction, the stiffness is symmetric and leaves a rigid translation free,
    and the shape functions are a partition of unity."""
    particle = _particle()
    n = particle.nDof
    density, volume = 1.0e-8, 1.0

    mLumped = np.zeros(n)
    particle.computeLumpedInertia(mLumped)
    assert mLumped.sum() == pytest.approx(2 * density * volume)

    K = np.zeros(n * n)
    particle.assembleStiffness(K)
    K = K.reshape(n, n)
    assert np.allclose(K, K.T, atol=1e-12 * np.abs(K).max())
    assert np.allclose(K @ np.tile([1.0, 0.0], n // 2), 0.0, atol=1e-9 * np.abs(K).max())

    assert particle.getInterpolationVector(np.full(2, 0.5)).sum() == pytest.approx(1.0)


def test_buffers_of_the_required_length_give_the_cell_operators():
    cell = _cell()

    assert cell.getInterpolationVector(np.full(2, 0.5)) == pytest.approx(np.full(4, 0.25))
    assert cell.isCoordinateInCell(np.full(2, 0.5))


def test_a_particle_refuses_a_buffer_of_its_nDof_before_its_kernel_functions_were_reassigned():
    """The nDof of a particle is nBaseDof times the number of its kernel functions, which changes as it moves."""
    from edelweissmeshfree.meshfree.kernelfunctions.marmot.marmotmeshfreekernelfunction import (
        MarmotMeshfreeKernelFunctionWrapper,
    )

    particle = _particle()
    nDofBefore = particle.nDof

    particle.assignKernelFunctions(
        [
            MarmotMeshfreeKernelFunctionWrapper(
                Node(label, np.array(center, dtype=float)), "BSplineBoxed", supportRadius=1.5, continuityOrder=3
            )
            for label, center in enumerate(itertools.product((-0.25, 1.25), repeat=2))
        ]
    )
    assert particle.nDof < nDofBefore

    with pytest.raises(
        ValueError, match=f"computeLumpedInertia: mLumped has {nDofBefore} entries, but {particle.nDof}"
    ):
        particle.computeLumpedInertia(np.zeros(nDofBefore))
