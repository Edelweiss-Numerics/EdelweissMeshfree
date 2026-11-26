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
"""The F-bar patch over the aggregates of the aggregated MPM (CellPatchProjection, AggregatedMaterialPointMethod with
``projectionMode="patch"``), assembled as the solver assembles it: the residual and the exact tangent across the cells
of an aggregate (central differences), its symmetry at a zero increment from the undeformed state, and that every
material point of a patch receives the volume change of its patch."""

import numpy as np
import pytest
from edelweissfe.journal.journal import Journal
from edelweissfe.numerics.systemmatrixassembly import VIJSystemMatrixAssembly
from edelweissfe.timesteppers.timestep import TimeStep

from edelweissmeshfree.config.mplibrary import getMaterialPointClass
from edelweissmeshfree.constraints.aggregatedmpm import AggregatedMaterialPointMethod
from edelweissmeshfree.generators import rectangulargridgenerator
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.mpmmanagers.smartmpmmanager import SmartMaterialPointManager
from edelweissmeshfree.numerics.aggregation import aggregationOfModel
from edelweissmeshfree.sets.materialpointset import MaterialPointSet

pytest.importorskip("edelweissmeshfree.cells.marmotcell.lagrangianmarmotcell")

NEOHOOKE = {"material": "COMPRESSIBLENEOHOOKE", "properties": np.array([833.0, 385.0, 1.0])}
J2 = {"material": "FINITESTRAINJ2PLASTICITY", "properties": np.array([833.0, 385.0, 6.0, 6.0, 0.0, 0.0, 1, 1.0])}


def _model(material):
    """A 5 x 4 grid of unit FBar cells; material points, off the cell centres, fill a staircase domain, so that
    there are interior cells, aggregated boundary cells and patches of several cells."""

    journal = Journal()
    model = MPMModel(2)
    rectangulargridgenerator.generateModelData(
        model, journal, x0=0.0, l=5.0, y0=0.0, h=4.0, nX=5, nY=4, cellProvider="LagrangianMarmotCell",
        cellType="DisplacementFBar/Quad4",
    )  # fmt: skip
    factory = getMaterialPointClass("marmot")
    rng = np.random.default_rng(3)
    number = 1
    for i in range(4):
        for j in range(4 if i < 2 else 3):
            for a, b in ((0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)):
                if i == 3 and b > 0.5 and j == 2:
                    continue  # a cell with two material points
                xy = np.array([[i + a + 0.15 * rng.uniform(-1, 1), j + b + 0.15 * rng.uniform(-1, 1)]])
                model.materialPoints[number] = factory("Displacement/PlaneStrain", number, xy, 0.25, material)
                number += 1
    model.materialPointSets["all"] = MaterialPointSet("all", list(model.materialPoints.values()))
    model.prepareYourself(journal)
    model.mpmManagers.append(
        SmartMaterialPointManager(model.cellSets["all"], model.materialPointSets["all"], 2, options={"KDTreeLevels": 2})
    )
    agmpm = AggregatedMaterialPointMethod("agmpm", model, journal, projectionMode="patch")
    model.multiPointConstraints["agmpm"] = agmpm
    model.updateConnectivity()
    return model, model.buildDiscreteSystem()


def _assemble(system, dU):
    """Residual (internal forces) and tangent as NIST assembles them: material points, cells, then the patches."""

    locations = system.locations
    timeStep = TimeStep(1, 1.0, 1.0, 1.0, 1.0, 1.0)
    U = locations.constructDofVector()
    U[:] = dU
    system.updateMaterialPoints(dU, timeStep, lambda entities, work: work(entities))
    P = locations.constructDofVector()
    P[:] = 0.0
    K = VIJSystemMatrixAssembly(locations, 1)
    K.beginAssembly()
    for cell in system.activeCells:
        Pe = np.zeros(cell.nDof)
        cell.computeKernels(K[cell], Pe, U[cell], dU[cell], 1.0, 1.0)
        P[cell] += Pe
    for patch in system.cellPatchProjections:
        patch.computeKernels(U[patch], dU[patch], np.zeros(patch.nDof), K[patch], timeStep)
    K.flush()
    return np.array(P), K.assembledMatrix().toarray()


def _increment(system, scale):
    dU = system.locations.constructDofVector()
    dU[:] = scale * np.sin(1.7 * np.arange(1, len(dU) + 1))
    return dU


def test_patches_are_the_aggregates():
    model, system = _model(NEOHOOKE)
    patches = system.cellPatchProjections
    aggregation = aggregationOfModel(model, system.activeCells)
    assert len(patches) == len(aggregation.activeCells) - len(aggregation.connectedInteriorCell)
    assert sum(len(p.cells) for p in patches) == len(aggregation.activeCells)
    assert any(len(p.cells) > 2 for p in patches)


@pytest.mark.parametrize("material, scale", [(NEOHOOKE, 0.03), (J2, 0.01)], ids=["neohooke", "j2"])
def test_exact_tangent_across_the_cells_of_a_patch(material, scale):
    model, system = _model(material)
    dU = _increment(system, scale)
    P, K = _assemble(system, dU)

    if material is J2:
        assert any(mp.getResultArray("alphaP")[0] > 0 for mp in model.materialPoints.values())

    # every material point of a patch has the volume change of the patch
    for patch in system.cellPatchProjections:
        for cell in patch.cells:
            for mp in cell.assignedMaterialPoints:
                dF = np.asarray(mp.getResultArray("delta deformation gradient")).reshape(3, 3)
                assert np.linalg.det(dF) == pytest.approx(patch.value, abs=1e-13)
    assert max(abs(p.value - 1) for p in system.cellPatchProjections) > 1e-3

    h = 1e-7
    KFD = np.zeros_like(K)
    for j in range(len(dU)):
        d = dU.copy()
        d[j] += h
        Pp, _ = _assemble(system, d)
        d[j] -= 2 * h
        Pm, _ = _assemble(system, d)
        KFD[:, j] = (Pp - Pm) / (2 * h)
    assert np.linalg.norm(K - KFD) <= 1e-6 * np.linalg.norm(KFD)
    # without the coupling of the patches, the tangent would be wrong
    patchesOff = np.zeros_like(K)
    for patch in system.cellPatchProjections:
        Kp = np.zeros((patch.nDof, patch.nDof))
        patch.computeKernels(None, None, None, Kp, None)
        idx = system.locations[patch]
        patchesOff[np.ix_(idx, idx)] += Kp
    assert np.linalg.norm(patchesOff) > 1e-2 * np.linalg.norm(KFD)


def test_tangent_is_symmetric_positive_semidefinite_at_a_zero_increment():
    model, system = _model(NEOHOOKE)
    dU = _increment(system, 0.0)
    _, K = _assemble(system, dU)
    assert np.linalg.norm(K - K.T) <= 1e-12 * np.linalg.norm(K)
    assert np.linalg.eigvalsh(0.5 * (K + K.T)).min() >= -1e-10 * np.abs(K).max()
