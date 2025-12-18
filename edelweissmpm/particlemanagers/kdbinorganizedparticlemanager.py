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


import numpy as np
from edelweissfe.journal.journal import Journal

from edelweissmpm.meshfree.particlekerneldomain import ParticleKernelDomain
from edelweissmpm.particlemanagers.base.baseparticlemanager import BaseParticleManager


class _FastKDBinOrganizer:
    """
    Optimized Bin Organizer using pure NumPy vectorization and integer indexing.
    """

    def __init__(self, kernelFunctions, dimension):
        self._dimension = dimension
        self._kernelFunctions = kernelFunctions

        # --- 1. Vectorized Bounding Box Extraction ---
        bboxes = [sf.getBoundingBox() for sf in kernelFunctions]

        if not bboxes:
            self._mins = np.empty((0, dimension))
            self._maxs = np.empty((0, dimension))
            self._bins = []
            # specific fix: ensure these attributes exist even if empty
            self._boundingBoxMin = np.zeros(dimension)
            self._boundingBoxMax = np.zeros(dimension)
            self._nBins = np.zeros(dimension, dtype=int)
            return

        bboxes = np.array(bboxes)
        self._mins = bboxes[:, 0, :]
        self._maxs = bboxes[:, 1, :]

        # --- 2. Grid Setup ---
        # RESTORED: Naming compatibility with Manager
        self._boundingBoxMin = np.min(self._mins, axis=0) - 1e-12
        self._boundingBoxMax = np.max(self._maxs, axis=0) + 1e-12

        avg_size = np.mean(self._maxs - self._mins, axis=0)
        self._binSize = avg_size / 2.0

        # Calculate grid dimensions
        # RESTORED: _nBins naming
        self._nBins = np.ceil((self._boundingBoxMax - self._boundingBoxMin) / self._binSize).astype(int)

        # Calculate strides for 1D flattening
        self._strides = np.ones(3, dtype=int)
        if dimension >= 2:
            self._strides[1] = self._nBins[0]
        if dimension == 3:
            self._strides[2] = self._nBins[0] * self._nBins[1]

        total_bins = np.prod(self._nBins)

        self._bins = [[] for _ in range(total_bins)]

        # --- 3. Vectorized Bin Index Calculation ---
        min_indices = ((self._mins - self._boundingBoxMin) / self._binSize).astype(int)
        max_indices = ((self._maxs - self._boundingBoxMin) / self._binSize).astype(int)

        # --- 4. Fill Bins ---
        _, sy, sz = self._strides[0], self._strides[1], self._strides[2]
        bins = self._bins

        for k_idx, (l, u) in enumerate(zip(min_indices, max_indices)):
            if dimension == 3:
                for z in range(l[2], u[2] + 1):
                    z_offset = z * sz
                    for y in range(l[1], u[1] + 1):
                        y_offset = z_offset + y * sy
                        start = y_offset + l[0]
                        end = y_offset + u[0] + 1
                        for bin_idx in range(start, end):
                            bins[bin_idx].append(k_idx)
            elif dimension == 2:
                for y in range(l[1], u[1] + 1):
                    y_offset = y * sy
                    start = y_offset + l[0]
                    end = y_offset + u[0] + 1
                    for bin_idx in range(start, end):
                        bins[bin_idx].append(k_idx)
            else:
                for bin_idx in range(l[0], u[0] + 1):
                    bins[bin_idx].append(k_idx)

    def getCandidateIndices(self, query_min, query_max):
        """Returns a set of Kernel Indices that overlap the query box."""
        if not self._bins:
            return set()

        _l = ((query_min - self._boundingBoxMin) / self._binSize).astype(int)
        _u = ((query_max - self._boundingBoxMin) / self._binSize).astype(int)

        # Clamp to grid bounds
        np.maximum(_l, 0, out=_l)
        np.minimum(_u, self._nBins - 1, out=_u)

        candidates = set()
        bins = self._bins
        _, sy, sz = self._strides[0], self._strides[1], self._strides[2]

        if self._dimension == 3:
            for z in range(_l[2], _u[2] + 1):
                z_offset = z * sz
                for y in range(_l[1], _u[1] + 1):
                    y_offset = z_offset + y * sy
                    start = y_offset + _l[0]
                    end = y_offset + _u[0] + 1
                    for bin_idx in range(start, end):
                        if bins[bin_idx]:
                            candidates.update(bins[bin_idx])

        elif self._dimension == 2:
            for y in range(_l[1], _u[1] + 1):
                y_offset = y * sy
                start = y_offset + _l[0]
                end = y_offset + _u[0] + 1
                for bin_idx in range(start, end):
                    if bins[bin_idx]:
                        candidates.update(bins[bin_idx])

        elif self._dimension == 1:
            for bin_idx in range(_l[0], _u[0] + 1):
                if bins[bin_idx]:
                    candidates.update(bins[bin_idx])

        return candidates


class _KDBinOrganizer:
    """A class to manage the kernel functions in cartesian bins in multiple dimension for fast access.

    We store the bounding box of each kerne lfunction (support) in the respective (partially-)covereing bins.
    This way, we can quickly find for given coordinates (of particles) the shape function candidates of which the support might cover the coordinates.

    Parameters
    ----------
    kernelFunctions
        The list of shape functions.
    dimension
        The dimension of the problem.
    """

    def __init__(self, kernelFunctions, dimension, randomlyShiftPartliceShapeFunctions: bool | float = False):
        self._dimension = dimension
        self._kernelFunctions = kernelFunctions

        # we make the bin size the average of the bounding box of the shape functions

        # boundingBoxes = np.array([sf.getBoundingBox() for sf in kernelFunctions])
        boundingBoxes = [sf.getBoundingBox() for sf in kernelFunctions]
        boundingBoxesMins = np.array([bb[0] for bb in boundingBoxes])
        boundingBoxesMaxs = np.array([bb[1] for bb in boundingBoxes])

        self._binSize = np.mean(boundingBoxesMaxs - boundingBoxesMins, axis=0) / 2
        self._boundingBoxMin = np.min(boundingBoxesMins, axis=0) - 1e-12
        self._boundingBoxMax = np.max(boundingBoxesMaxs, axis=0) + 1e-12

        self._nBins = np.ceil((self._boundingBoxMax - self._boundingBoxMin) / self._binSize).astype(int)

        self._thebins = np.frompyfunc(list, 0, 1)(np.empty(self._nBins, dtype=object))
        # print(self._thebins.shape)

        # for every shape function, we need to get the bin indices for the min. and max. bounding box:
        for i, kf in enumerate(kernelFunctions):
            minIndices = self._getBinIndices(boundingBoxes[i][0])
            maxIndices = self._getBinIndices(boundingBoxes[i][1])

            # now we need to loop over all bins in the bounding box of the shape function and add the shape function to the bin
            # 3d case:
            if self._dimension == 3:
                for i in range(minIndices[0], maxIndices[0] + 1):
                    for j in range(minIndices[1], maxIndices[1] + 1):
                        for k in range(minIndices[2], maxIndices[2] + 1):
                            self._thebins[i, j, k].append(kf)
            # 2d case:
            elif self._dimension == 2:
                for i in range(minIndices[0], maxIndices[0] + 1):
                    for j in range(minIndices[1], maxIndices[1] + 1):
                        self._thebins[i, j].append(kf)
            # 1d case:
            elif self._dimension == 1:
                for i in range(minIndices[0], maxIndices[0] + 1):
                    self._thebins[i].append

            else:
                raise ValueError("Dimension not supported")

    def _getBinIndices(self, coordinate: np.ndarray) -> list:
        """Get the cartesian bin indices for a given coordinate.

        Parameters
        ----------
        coordinate
            The coordinate for which the indices should be computed.

        Returns
        -------
        list
            The list of indices.
        """

        return [int((coordinate[i] - self._boundingBoxMin[i]) / self._binSize[i]) for i in range(self._dimension)]

    def getKernelFunctionCandidates(self, coordinate: np.ndarray) -> list:
        """Get the shape function candidates for a given coordinate.

        Candidates are shape functions that might cover the given coordinate.
        It is guaranteed that all candidates are returned, but not that all returned shape functions might cover the coordinate,
        since this is only a fast pre-selection based on the bounding box of the shape functions.

        Parameters
        ----------
        coordinate
            The coordinate for which the shape function candidates should be determined.

        Returns
        -------
        list
            The list of shape function candidates.
        """

        indices = self._getBinIndices(coordinate)
        thebin = self._thebins[tuple(indices)]
        return thebin

    def __str__(self):
        return f"KDBin with {len(self._kernelFunctions)} shape functions in {self._nBins} bins of size {self._binSize} in a bounding box from {self._boundingBoxMin} to {self._boundingBoxMax}."


class KDBinOrganizedParticleManager(BaseParticleManager):
    """A k-dimensional bin organized manager for particles and meshfree shape functions  for locating points in supports.

    Parameters
    ----------
    meshfreeKernelFunctions
        The list of shape functions.
    particles
        The list of particles.
    dimension
        The dimension of the problem.
    journal
        The journal for logging messages.
    bondParticlesToKernelFunctions
        Whether to bond the particles to the kernel functions (one particle per kernel function).
        If True, the kernel functions are moved to the particle center coordinates at each update.
    randomlyShiftPartliceShapeFunctions
        Whether to randomly shift the shape functions a bit to avoid alignment artifacts.
        If a float value is given, this value is used as the maximum shift factor in each direction, which is scaled with the approximate particle size:
        randdisp = (np.random.rand(self._dimension) - 0.5) * np.sqrt(particle.getVolumeUndeformed()) * randomlyShiftPartliceShapeFunctions * particleSize
    """

    def __init__(
        self,
        particleKernelDomain: ParticleKernelDomain,
        dimension: int,
        journal: Journal,
        bondParticlesToKernelFunctions: bool = False,
        randomlyShiftPartliceShapeFunctions: bool | float = False,
    ):

        self._meshfreeKernelFunctions = particleKernelDomain.meshfreeKernelFunctions
        self._particles = particleKernelDomain.particles
        self._dimension = dimension
        self._bondParticlesToKernelFunctions = bondParticlesToKernelFunctions
        self._journal = journal

        if not isinstance(randomlyShiftPartliceShapeFunctions, (bool, float)):
            raise ValueError("randomlyShiftPartliceShapeFunctions must be a boolean or a float.")
        self._randomlyShiftPartliceShapeFunctions = randomlyShiftPartliceShapeFunctions

        if self._bondParticlesToKernelFunctions:
            if len(self._particles) != len(self._meshfreeKernelFunctions):
                raise ValueError("The number of particles and kernel functions must be equal.")

            for particle, kernelFunction in zip(self._particles, self._meshfreeKernelFunctions):
                particleCoordinates = particle.getCenterCoordinates()
                kernelFunction.moveTo(particleCoordinates)

        self.signalizeKernelFunctionUpdate()

    def signalizeKernelFunctionUpdate(self):
        # Use the new fast organizer
        self._theBins = _FastKDBinOrganizer(self._meshfreeKernelFunctions, self._dimension)

    def updateConnectivity(self):
        hasChanged = False

        # --- Bonding Logic (Existing code) ---
        if self._bondParticlesToKernelFunctions:
            self._journal.message("Updating kernel function positions...", "ParticleManager")
            for particle, kernelFunction in zip(self._particles, self._meshfreeKernelFunctions):
                particleCoordinates = particle.getCenterCoordinates()
                if self._randomlyShiftPartliceShapeFunctions:
                    if isinstance(self._randomlyShiftPartliceShapeFunctions, float):
                        particleVol = particle.getVolumeUndeformed()
                        # particleSize = np.pow(particleVol, 1.0 / self._dimension)
                        particleSize = particleVol ** (1.0 / self._dimension)
                        randdisp = (
                            (np.random.rand(self._dimension) - 0.5)
                            * np.sqrt(particle.getVolumeUndeformed())
                            * self._randomlyShiftPartliceShapeFunctions
                            * particleSize
                        )
                    particleCoordinates += randdisp

                kernelFunction.moveTo(particleCoordinates)

            # Rebuild grid after moving kernels
            self.signalizeKernelFunctionUpdate()

        # --- Fast Search Logic ---

        # Local references for speed
        all_kernels = self._meshfreeKernelFunctions
        kernel_mins = self._theBins._mins  # (N, D) array
        kernel_maxs = self._theBins._maxs  # (N, D) array
        dim = self._dimension

        for p in self._particles:
            evaluationCoordinates = p.getEvaluationCoordinates()

            # 1. Broad Phase: Particle Bounding Box
            # Instead of querying the grid 8 times (for 8 vertices),
            # we query it once for the particle's bounding box.
            p_min = np.min(evaluationCoordinates, axis=0)
            p_max = np.max(evaluationCoordinates, axis=0)

            # Get INDICES of potential kernels
            candidate_indices = self._theBins.getCandidateIndices(p_min, p_max)

            valid_kernels = []

            # 2. Narrow Phase: AABB Intersection + Precise Check
            for k_idx in candidate_indices:

                # A. FAST AABB CHECK (Box vs Box)
                # If the particle box doesn't touch the kernel box, skip expensive math.
                # This filters out ~90% of candidates returned by the bins.
                k_min = kernel_mins[k_idx]
                k_max = kernel_maxs[k_idx]

                # Check for separation (no overlap)
                if p_max[0] < k_min[0] or p_min[0] > k_max[0]:
                    continue
                if dim > 1:
                    if p_max[1] < k_min[1] or p_min[1] > k_max[1]:
                        continue
                if dim > 2:
                    if p_max[2] < k_min[2] or p_min[2] > k_max[2]:
                        continue

                # B. PRECISE CHECK (Vertices vs Support)
                sf = all_kernels[k_idx]

                # Check actual vertices. Stop at the first one found.
                is_covered = False
                for coord in evaluationCoordinates:
                    if sf.isCoordinateInCurrentSupport(coord):
                        is_covered = True
                        break  # Optimization: Found one, no need to check others

                if is_covered:
                    valid_kernels.append(sf)

            # 3. Assignment
            # Sort by node label for determinism
            valid_kernels.sort(key=lambda x: x.node.label)

            if not hasChanged and valid_kernels != p.kernelFunctions:
                hasChanged = True

            p.assignKernelFunctions(valid_kernels)

        return hasChanged

    # def signalizeKernelFunctionUpdate(
    #     self,
    # ):
    #     self._theBins = _KDBinOrganizer(self._meshfreeKernelFunctions, self._dimension)

    # def updateConnectivity(
    #     self,
    # ):
    #     hasChanged = False

    #     if self._bondParticlesToKernelFunctions:
    #         self._journal.message(
    #             f"Updating kernel function positions for the bonding definition with {len(self._particles)} particles.",
    #             "ParticleManager",
    #         )
    #         for particle, kernelFunction in zip(self._particles, self._meshfreeKernelFunctions):
    #             particleCoordinates = particle.getCenterCoordinates()
    #             if self._randomlyShiftPartliceShapeFunctions:
    #                 if isinstance(self._randomlyShiftPartliceShapeFunctions, float):
    #                     particleVol = particle.getVolumeUndeformed()
    #                     # particleSize = np.pow(particleVol, 1.0 / self._dimension)
    #                     particleSize = particleVol ** (1.0 / self._dimension)
    #                     randdisp = (
    #                         (np.random.rand(self._dimension) - 0.5)
    #                         * np.sqrt(particle.getVolumeUndeformed())
    #                         * self._randomlyShiftPartliceShapeFunctions
    #                         * particleSize
    #                     )
    #                 particleCoordinates += randdisp

    #             kernelFunction.moveTo(particleCoordinates)

    #         self.signalizeKernelFunctionUpdate()

    #     for p in self._particles:
    #         evaluationCoordinates = p.getEvaluationCoordinates()

    #         # rough search based on the bounding box of the kernel functions
    #         kernelFunctionCandidates = {
    #             candidate
    #             for vertex in evaluationCoordinates
    #             for candidate in self._theBins.getKernelFunctionCandidates(vertex)
    #         }
    #         # fine search based on the actual support of the kernel functions
    #         kernelFunctions = {
    #             sf
    #             for coordinate in evaluationCoordinates
    #             for sf in kernelFunctionCandidates
    #             if sf.isCoordinateInCurrentSupport(coordinate)
    #         }

    #         # we need to sort the kernel functions using their node number to ensure that the order is always the same
    #         kernelFunctions = list(sorted(kernelFunctions, key=lambda x: x.node.label))

    #         if hasChanged or kernelFunctions != p.kernelFunctions:
    #             hasChanged = True

    #         p.assignKernelFunctions(kernelFunctions)

    #     return hasChanged

    def getCoveredDomain(
        self,
    ):
        return self._theBins._boundingBoxMin, self._theBins._boundingBoxMax

    def __str__(self):
        return f"KDBinOrganizedParticleManager with {len(self._particles)} particles and {len(self._meshfreeKernelFunctions)} shape functions in {self._dimension} dimensions. Covered domain: {self.getCoveredDomain()}."

    def visualize(self):
        """For 2D only: Visualize the number of kernel functions in the bins."""

        if self._dimension != 2:
            raise ValueError("Visualization only supported for 2D.")

        import matplotlib.pyplot as plt

        nBins = self._theBins._nBins
        nKernelFunctions = np.zeros(nBins)
        for i in range(nBins[0]):
            for j in range(nBins[1]):
                nKernelFunctions[i, j] = len(self._theBins._thebins[i, j])

        plt.figure()
        plt.imshow(
            nKernelFunctions.T,
        )
        plt.title("Number of kernel functions in the bins of the KDBinOrganizer")

        nBins = self._theBins._nBins
        # plot the lines:
        for i in range(nBins[0] + 1):
            plt.plot([i - 0.5, i - 0.5], [0 - 0.5, nBins[1] - 0.5], "k")
        for j in range(nBins[1] + 1):
            plt.plot([0 - 0.5, nBins[0] - 0.5], [j - 0.5, j - 0.5], "k")

        plt.colorbar()
        plt.show()
