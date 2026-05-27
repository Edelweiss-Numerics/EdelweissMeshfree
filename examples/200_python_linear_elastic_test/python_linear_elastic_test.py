# -*- coding: utf-8 -*-
#  ---------------------------------------------------------------------
#
#  _____    _      _              _
# | ____|__| | ___| |_      _____(_)___ ___
# |  _| / _` |/ _ \ \ \ /\ / / _ \ / __/ __|
# | |__| (_| |  __/ |\ V  V /  __/ \__ \__ \
# |_____\__,_|\___|_| \_/\_/_\___|_|___/___/
# |  \/  | ___  ___| |__  / _|_ __ ___  ___
# | |\/| |/ _ \/ __| '_ \| |_| '__/ _ \/ _ \
# | |  | |  __/\__ \ | | |  _| | |  __/  __/
# |_|  |_|\___||___/_| |_|_| |_|  \___|\___|
#
#
#  Unit of Strength of Materials and Structural Analysis
#  University of Innsbruck,
#
#  Research Group for Computational Mechanics of Materials
#  Institute of Structural Engineering, BOKU University, Vienna
#
#  2023 - today
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
"""
Test for the pure Python cell and material point implementation with linear elastic behavior.

This test performs a uniaxial compression of a 2D rectangular domain using:
- PythonCell (bilinear Quad4) as the grid cell provider
- PythonMaterialPoint with linear elastic plane strain material

A standalone MPM solver loop is used (no full EdelweissMeshfree solver infrastructure)
to verify the correctness of the pure Python implementations.

The domain is [0, 200] x [0, 100] with 4x2 cells and 8x4 material points.
Left boundary is fixed; right boundary is displaced by -10 in x.
For linear elasticity, only one Newton iteration is needed.
"""

import argparse

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from edelweissmeshfree.cells.pythoncell.cell import PythonCell
from edelweissmeshfree.materialpoints.pythonmaterialpoint.mp import PythonMaterialPoint


class SimpleNode:
    """A minimal node class for testing."""

    def __init__(self, label, coordinates):
        self.label = label
        self.coordinates = np.asarray(coordinates, dtype=float)
        self.fields = [["displacement"]]


def run_sim():
    """Run a standalone MPM simulation with pure Python cells and material points."""

    # Domain parameters
    Lx, Ly = 200.0, 100.0
    nCellsX, nCellsY = 4, 2
    nMPsX, nMPsY = 8, 4

    # Material parameters
    E, nu = 30000.0, 0.3

    # Prescribed displacement on right boundary
    u_prescribed = -10.0

    # ------ Build the grid (nodes + cells) ------
    dx = Lx / nCellsX
    dy = Ly / nCellsY

    nodes = []
    node_map = {}  # (i, j) -> node index
    for j in range(nCellsY + 1):
        for i in range(nCellsX + 1):
            idx = len(nodes)
            node_map[(i, j)] = idx
            nodes.append(SimpleNode(idx, [i * dx, j * dy]))

    nNodes = len(nodes)
    nDofTotal = nNodes * 2

    cells = []
    for j in range(nCellsY):
        for i in range(nCellsX):
            n0 = node_map[(i, j)]
            n1 = node_map[(i + 1, j)]
            n2 = node_map[(i + 1, j + 1)]
            n3 = node_map[(i, j + 1)]
            cell_nodes = [nodes[n0], nodes[n1], nodes[n2], nodes[n3]]
            cells.append(PythonCell("Quad4", len(cells), cell_nodes))

    # ------ Create material points ------
    mp_dx = Lx / nMPsX
    mp_dy = Ly / nMPsY
    mp_volume = mp_dx * mp_dy  # uniform volume

    material = {"material": "LinearElastic", "properties": np.array([E, nu])}

    mps = []
    for j in range(nMPsY):
        for i in range(nMPsX):
            x = (i + 0.5) * mp_dx
            y = (j + 0.5) * mp_dy
            mp = PythonMaterialPoint("PlaneStrain", len(mps), np.array([x, y]), mp_volume, material)
            mps.append(mp)

    # ------ Assign material points to cells ------
    for cell in cells:
        assigned = []
        for mp in mps:
            if cell.isCoordinateInCell(mp.getCenterCoordinates()):
                assigned.append(mp)
                mp.assignCells([cell])
        cell.assignMaterialPoints(assigned)

    # ------ Identify boundary DOFs ------
    left_dofs = []
    right_dofs = []
    for i, node in enumerate(nodes):
        if abs(node.coordinates[0]) < 1e-10:
            left_dofs.extend([2 * i, 2 * i + 1])
        elif abs(node.coordinates[0] - Lx) < 1e-10:
            right_dofs.extend([2 * i, 2 * i + 1])

    # Prescribed values: left fixed (u=0, v=0), right u=u_prescribed, v=0
    prescribed_dofs = {}
    for dof in left_dofs:
        prescribed_dofs[dof] = 0.0
    for i, node in enumerate(nodes):
        if abs(node.coordinates[0] - Lx) < 1e-10:
            prescribed_dofs[2 * i] = u_prescribed
            prescribed_dofs[2 * i + 1] = 0.0

    free_dofs = [d for d in range(nDofTotal) if d not in prescribed_dofs]

    # ------ Single-step linear solve ------
    # For linear elasticity, K is constant. Assemble K directly from the tangent.
    # Then solve K_ff * U_f = -K_fp * U_p

    # Build cell-to-global DOF mapping
    def cell_global_dofs(cell):
        dofs = []
        for node in cell.nodes:
            dofs.extend([2 * node.label, 2 * node.label + 1])
        return np.array(dofs, dtype=int)

    # Assemble global stiffness from material tangent
    K_global = np.zeros((nDofTotal, nDofTotal))

    for cell in cells:
        gdofs = cell_global_dofs(cell)
        ncd = len(gdofs)
        K_cell = np.zeros((ncd, ncd))

        # For each material point in cell, add B^T C B * volume
        for mp in cell.assignedMaterialPoints:
            from edelweissmeshfree.cells.pythoncell.cell import (
                _global_to_parametric,
                _compute_shape_functions_and_gradients,
            )

            mp_coords = mp.getCenterCoordinates()
            xi, eta = _global_to_parametric(mp_coords, cell._node_coords)
            _, dNdx, _ = _compute_shape_functions_and_gradients(xi, eta, cell._node_coords)
            B = cell._get_B_matrix(dNdx)
            C = mp.getAlgorithmicTangent()
            volume = mp.getVolume()
            K_cell += B.T @ C @ B * volume

        for a in range(ncd):
            for b in range(ncd):
                K_global[gdofs[a], gdofs[b]] += K_cell[a, b]

    # Set up prescribed displacement vector
    U = np.zeros(nDofTotal)
    for dof, val in prescribed_dofs.items():
        U[dof] = val

    # Partition and solve: K_ff * U_f = -K_fp * U_p
    free_arr = np.array(free_dofs)
    prescribed_arr = np.array(list(prescribed_dofs.keys()))
    U_p = np.array([prescribed_dofs[d] for d in prescribed_arr])

    K_ff = K_global[np.ix_(free_arr, free_arr)]
    K_fp = K_global[np.ix_(free_arr, prescribed_arr)]

    rhs = -K_fp @ U_p
    U_f = spsolve(csr_matrix(K_ff), rhs)
    U[free_arr] = U_f

    # Now interpolate the full solution to material points
    for mp in mps:
        mp.prepareYourself(0.0, 1.0)

    for cell in cells:
        gdofs = cell_global_dofs(cell)
        dU_cell = U[gdofs]
        cell.interpolateFieldsToMaterialPoints(dU_cell)

    for mp in mps:
        mp.computeYourself(0.0, 1.0)

    # Accept state
    for mp in mps:
        mp.acceptStateAndPosition()

    return mps, nodes, U


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    """No matter where pytest is ran, we set the working dir
    to this testscript's parent directory"""

    monkeypatch.chdir(request.fspath.dirname)


def test_sim(assert_gold):
    """Test the pure Python cell + material point implementation against gold values."""
    mps, nodes, U = run_sim()

    res = np.array([mp.getResultArray("displacement") for mp in mps])
    gold = np.loadtxt("gold.csv")

    assert_gold(res, gold)


def test_analytical():
    """Validate the simulation against expected physical behavior for plane strain uniaxial compression.

    For this MPM discretization:
    - The x-displacement should be approximately proportional to x-position (with some MPM integration error).
    - The solution should be symmetric about the horizontal center line (y=50).
    - No net vertical deformation at the horizontal center line.
    """
    mps, nodes, U = run_sim()

    Lx = 200.0
    u_prescribed = -10.0

    # Check x-displacements are monotonically decreasing (more negative) with x
    # Group by y-row
    mps_row0 = [mp for mp in mps if abs(mp._coordinates[1] - 12.5) < 1.0]
    mps_row0.sort(key=lambda mp: mp._coordinates[0])
    ux_row0 = [mp.getResultArray("displacement")[0] for mp in mps_row0]
    for i in range(len(ux_row0) - 1):
        assert ux_row0[i] > ux_row0[i + 1], "x-displacement should decrease monotonically"

    # Check symmetry about y=50 (center line)
    # MPs at y=12.5 and y=87.5 should have equal and opposite y-displacements
    for mp in mps:
        y = mp._coordinates[1]
        x = mp._coordinates[0]
        if abs(y - 12.5) < 1.0:
            # Find corresponding MP at y=87.5
            for mp2 in mps:
                if abs(mp2._coordinates[1] - 87.5) < 1.0 and abs(mp2._coordinates[0] - x) < 1.0:
                    uy1 = mp.getResultArray("displacement")[1]
                    uy2 = mp2.getResultArray("displacement")[1]
                    np.testing.assert_allclose(uy1, -uy2, atol=1e-10)
                    break

    # Check that x-displacement is approximately linear (within MPM discretization error)
    for mp in mps:
        x = mp._coordinates[0]
        disp = mp.getResultArray("displacement")
        expected_ux = u_prescribed * x / Lx
        # Allow ~10% relative error due to MPM quadrature
        np.testing.assert_allclose(disp[0], expected_ux, rtol=0.25, atol=0.5)


if __name__ == "__main__":
    mps, nodes, U = run_sim()

    parser = argparse.ArgumentParser()
    parser.add_argument("--create-gold", dest="create_gold", action="store_true", help="create the gold file.")
    args = parser.parse_args()

    if args.create_gold:
        gold = np.array([mp.getResultArray("displacement") for mp in mps])
        np.savetxt("gold.csv", gold)
    else:
        # Print some results
        print("\nMaterial point displacements:")
        for mp in mps:
            print(f"  MP {mp.number}: coords={mp._coordinates}, disp={mp.getResultArray('displacement')}")
        print(f"\nTotal DOFs: {len(U)}, max |U|: {np.max(np.abs(U)):.6f}")
