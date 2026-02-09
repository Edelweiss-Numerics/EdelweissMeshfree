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
#  This file is part of EdelweissMPM.
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#
#  The full text of the license can be found in the file LICENSE.md at
#  the top level directory of EdelweissMPM.
#  ---------------------------------------------------------------------

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from edelweissfe.fields.nodefield import NodeField
from edelweissfe.numerics.dofmanager import DofManager
from edelweissfe.numerics.parallelizationutilities import (
    getNumberOfThreads,
    isFreeThreadingSupported,
)
from edelweissfe.variables.scalarvariable import ScalarVariable


class MPMDofManager(DofManager):
    """A DofManager class for the Material Point Method.
    Derived from DofManager, it is used to manage the degrees of freedom of the model
    and to provide the necessary information for the assembly of the global system matrix.

    Parameters
    ----------
    nodeFields : list
        A list of node fields.
    scalarVariables : list
        A list of scalar variables.
    elements : list
        A list of elements.
    constraints : list
        A list of constraints.
    nodeSets : list
        A list of node sets.
    cells : list
        A list of cells.
    cellElements : list
        A list of cell elements.
    particles : list
        The list of particles (RKPM).
    initializeVIJPattern : bool
        A flag indicating whether the VIJ pattern should be initialized.
    """

    def __init__(
        self,
        nodeFields: list[NodeField],
        scalarVariables: list[ScalarVariable] = [],
        elements: list = [],
        constraints: list = [],
        nodeSets: list = [],
        cells: list = [],
        cellElements: list = [],
        particles: list = [],
        initializeVIJPattern: bool = True,
    ):

        super().__init__(nodeFields, scalarVariables, elements, constraints, nodeSets, initializeVIJPattern=False)

        (
            self.accumulatedCellNDof,
            self._accumulatedCellVIJSize,
            self._nAccumulatedNodalFluxesFieldwiseFromCells,
            self.largestNumberOfCellNDof,
        ) = self._gatherCellsInformation(cells)

        (
            self.accumulatedCellElementNDof,
            self._accumulatedCellElementVIJSize,
            self._nAccumulatedNodalFluxesFieldwiseFromCellElements,
            self.largestNumberOfCellElementNDof,
        ) = self._gatherCellsInformation(cellElements)

        (
            self.accumulatedParticleNDof,
            self._accumulatedParticleVIJSize,
            self._nAccumulatedNodalFluxesFieldwiseFromParticles,
            self.largestNumberOfParticleNDof,
        ) = self._gatherCellsInformation(particles)

        self.idcsOfCellsInDofVector = self._locateCellsInDofVector(cells)
        self.idcsOfCellElementsInDofVector = self._locateCellsInDofVector(cellElements)
        self.idcsOfParticlesInDofVector = self._locateParticlesInDofVector(particles)

        for field in self.nAccumulatedNodalFluxesFieldwise.keys():
            self.nAccumulatedNodalFluxesFieldwise[field] += self._nAccumulatedNodalFluxesFieldwiseFromCells[field]
            self.nAccumulatedNodalFluxesFieldwise[field] += self._nAccumulatedNodalFluxesFieldwiseFromCellElements[
                field
            ]
            self.nAccumulatedNodalFluxesFieldwise[field] += self._nAccumulatedNodalFluxesFieldwiseFromParticles[field]

        self.idcsOfHigherOrderEntitiesInDofVector |= self.idcsOfCellsInDofVector
        self.idcsOfHigherOrderEntitiesInDofVector |= self.idcsOfCellElementsInDofVector
        self.idcsOfHigherOrderEntitiesInDofVector |= self.idcsOfParticlesInDofVector

        self._sizeVIJ = (
            self._accumulatedElementVIJSize
            + self._accumulatedConstraintVIJSize
            + self._accumulatedCellVIJSize
            + self._accumulatedCellElementVIJSize
            + self._accumulatedParticleVIJSize
        )
        if initializeVIJPattern:
            (self.I, self.J, self.idcsOfHigherOrderEntitiesInVIJ) = self._initializeVIJPattern()

    def _gatherCellsInformation(self, entities: list) -> tuple[int, int, int, int]:
        """Generates some auxiliary information,
        which may be required by some modules of EdelweissFE.

        Parameters
        ----------
        entities
           The list of entities, for which the information is gathered.

        Returns
        -------
        tuple[int,int]
            The tuple of
                - number of accumulated elemental degrees of freedom.
                - number of accumulated system matrix sizes.
                - the number of  acummulated fluxes Σ_entities Σ_nodes ( nDof (field) ) for Abaqus-like convergence tests.
                - largest occuring number of dofs on any element.
        """

        return self._gatherElementsInformation(entities)

    def _locateCellsInDofVector(self, cells: list) -> dict:
        """Creates a dictionary containing the location (indices) of each cell
        within the DofVector structure.

        Returns
        -------
        dict
            A dictionary containing the location mapping.
        """

        idcsOfCellsInDofVector = {}

        for cl in cells:
            destList = np.hstack(
                [
                    self.idcsOfFieldVariablesInDofVector[node.fields[nodeField]]
                    for iNode, node in enumerate(cl.nodes)  # for each node of the cell ..
                    for nodeField in cl.fields[iNode]  # for each field of this node
                ]
            )  # the index in the global system

            if cl.dofIndicesPermutation is not None:
                idcsOfCellsInDofVector[cl] = destList[cl.dofIndicesPermutation]
            else:
                idcsOfCellsInDofVector[cl] = destList

        return idcsOfCellsInDofVector

    def _locateParticlesInDofVector(self, particles: list) -> dict:
        """Creates a dictionary containing the location (indices) of each particle
        within the DofVector structure.

        In contrast to elements, cells and cell elements, particles have an identical set of fields
        on each attached node.
        Furthermore, due to the varying number attached nodes, no permutation is allowed.
        For particles generally a node-wise layout is assumed.

        For instance: [node_1_displacement, node_1_temperature, node_2_displacement, node_2_temperature, ...]

        Returns
        -------
        dict
            A dictionary containing the location mapping.
        """

        return self._locateNodeCouplingEntitiesInDofVector(particles)

    def updateParticles(self, particles: list):
        """
        Updates the connectivity mapping for particles without rebuilding
        the entire DofManager structure.

        Parameters
        ----------
        particles : list
            The list of particles to update.
        """

        if not particles:
            return

        elements = list(particles)  # Ensure we have a list for slicing
        num_elements = len(particles)

        # 1. Access the pre-cached field variable map
        field_var_map = self.idcsOfFieldVariablesInDofVector

        numThreads = getNumberOfThreads() if isFreeThreadingSupported() else 1
        chunk_size = max(1, num_elements // numThreads)

        def process_element_chunk(chunk_elements):
            local_results = {}
            for ent in chunk_elements:
                # Optimized extraction:
                # Avoid nested generators; use list comprehension for speed
                indices = []
                for iNode, node in enumerate(ent.nodes):
                    for nodeField in ent.fields[iNode]:
                        # Direct access to the pre-calculated global indices
                        indices.extend(field_var_map[node.fields[nodeField]])

                destArr = np.array(indices, dtype=int)

                if ent.dofIndicesPermutation is not None:
                    local_results[ent] = destArr[ent.dofIndicesPermutation]
                else:
                    local_results[ent] = destArr
            return local_results

        # 3. Execute update
        chunks = [elements[i : i + chunk_size] for i in range(0, num_elements, chunk_size)]

        with ThreadPoolExecutor(max_workers=numThreads) as executor:
            results = executor.map(process_element_chunk, chunks)

        # 4. Merge results back into the manager
        for partial_map in results:
            self.idcsOfElementsInDofVector.update(partial_map)

        # 5. Crucial: Synchronize the "Higher Order" map used by System Matrices
        self.idcsOfHigherOrderEntitiesInDofVector = self.idcsOfElementsInDofVector | self.idcsOfConstraintsInDofVector

    def updateConstraints(self, constraints: list):
        """
        Updates the connectivity mapping for constraints in a serial loop.
        Reuses global index maps to avoid re-instancing the manager.

        Parameters
        ----------
        constraints : list
            The list of constraints to be considered.
        """
        if not constraints:
            return

        constraints = list(constraints)  # Ensure we have a list for iteration

        # Cache lookups for local speed
        field_var_map = self.idcsOfFieldVariablesInDofVector
        scalar_var_map = self.idcsOfScalarVariablesInDofVector

        # Persistent dictionary for the manager
        updated_idcs = {}

        for constraint in constraints:
            # 1. Collect indices from nodes
            # constraint.fieldsOnNodes matches the structure of constraint.nodes
            node_indices = []
            for iNode, node in enumerate(constraint.nodes):
                for nodeField in constraint.fieldsOnNodes[iNode]:
                    node_indices.extend(field_var_map[node.fields[nodeField]])

            # 2. Collect indices from scalar variables
            scalar_indices = [scalar_var_map[v] for v in constraint.scalarVariables]

            # 3. Combine and store as a flat NumPy array
            # We use concatenation to maintain the expected order (nodes then scalars)
            updated_idcs[constraint] = np.array(node_indices + scalar_indices, dtype=int)

        # Update the DofManager's internal maps
        self.idcsOfConstraintsInDofVector.update(updated_idcs)

        # Re-sync the higher order map for assembly
        self.idcsOfHigherOrderEntitiesInDofVector = self.idcsOfElementsInDofVector | self.idcsOfConstraintsInDofVector
