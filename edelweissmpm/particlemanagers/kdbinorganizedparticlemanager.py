# -*- coding: utf-8 -*-
#   ---------------------------------------------------------------------
#   EdelweissMPM - High-Performance Particle Search
#   (Vectorized AABB + Cython Fast-Path + Thread Safety)
#   ---------------------------------------------------------------------

import concurrent.futures
import itertools
from typing import Any, List, Set, Tuple, Union

import numpy as np
from edelweissfe.journal.journal import Journal
from edelweissfe.numerics.parallelizationutilities import (
    getNumberOfThreads,
    isFreeThreadingSupported,
)
from numpy.typing import NDArray

from edelweissmpm.meshfree.particlekerneldomain import ParticleKernelDomain
from edelweissmpm.particlemanagers.base.baseparticlemanager import BaseParticleManager


class _FastKDBinOrganizer:
    """
    Optimized Bin Organizer using pure NumPy vectorization and integer indexing.
    """

    def __init__(self, kernelFunctions: List[Any], dimension: int) -> None:
        self._dimension = dimension

        # --- 1. Vectorized Bounding Box Extraction ---
        bboxes = [sf.getBoundingBox() for sf in kernelFunctions]

        if not bboxes:
            self._mins = np.empty((0, dimension))
            self._maxs = np.empty((0, dimension))
            self._bins = []
            self._boundingBoxMin = np.zeros(dimension)
            self._boundingBoxMax = np.zeros(dimension)
            self._nBins = np.zeros(dimension, dtype=int)
            self._binSize = np.ones(dimension)
            self._strides = np.ones(3, dtype=int)
            return

        bboxes_arr = np.array(bboxes)
        self._mins = bboxes_arr[:, 0, :]
        self._maxs = bboxes_arr[:, 1, :]

        # --- 2. Grid Setup ---
        self._boundingBoxMin = np.min(self._mins, axis=0) - 1e-12
        self._boundingBoxMax = np.max(self._maxs, axis=0) + 1e-12

        avg_size = np.mean(self._maxs - self._mins, axis=0)
        self._binSize = avg_size / 2.0

        self._nBins = np.ceil((self._boundingBoxMax - self._boundingBoxMin) / self._binSize).astype(int)

        self._strides = np.ones(3, dtype=int)
        if dimension >= 2:
            self._strides[1] = self._nBins[0]
        if dimension == 3:
            self._strides[2] = self._nBins[0] * self._nBins[1]

        total_bins = int(np.prod(self._nBins))
        self._bins = [[] for _ in range(total_bins)]

        # --- 3. Vectorized Bin Index Calculation ---
        min_indices = ((self._mins - self._boundingBoxMin) / self._binSize).astype(int)
        max_indices = ((self._maxs - self._boundingBoxMin) / self._binSize).astype(int)

        # --- 4. Fill Bins ---
        _, stride_y, stride_z = self._strides[0], self._strides[1], self._strides[2]
        bins = self._bins

        for k_idx, (min_idx, max_idx) in enumerate(zip(min_indices, max_indices)):
            if dimension == 3:
                for z in range(min_idx[2], max_idx[2] + 1):
                    z_offset = z * stride_z
                    for y in range(min_idx[1], max_idx[1] + 1):
                        y_offset = z_offset + y * stride_y
                        start = y_offset + min_idx[0]
                        end = y_offset + max_idx[0] + 1
                        for bin_idx in range(start, end):
                            bins[bin_idx].append(k_idx)
            elif dimension == 2:
                for y in range(min_idx[1], max_idx[1] + 1):
                    y_offset = y * stride_y
                    start = y_offset + min_idx[0]
                    end = y_offset + max_idx[0] + 1
                    for bin_idx in range(start, end):
                        bins[bin_idx].append(k_idx)
            else:
                for bin_idx in range(min_idx[0], max_idx[0] + 1):
                    bins[bin_idx].append(k_idx)

    def getCandidateIndices(self, query_min: NDArray[np.float64], query_max: NDArray[np.float64]) -> Set[int]:
        if not self._bins:
            return set()

        idx_lower = ((query_min - self._boundingBoxMin) / self._binSize).astype(int)
        idx_upper = ((query_max - self._boundingBoxMin) / self._binSize).astype(int)

        np.maximum(idx_lower, 0, out=idx_lower)
        np.minimum(idx_upper, self._nBins - 1, out=idx_upper)

        bins = self._bins
        _, stride_y, stride_z = self._strides[0], self._strides[1], self._strides[2]

        lists_to_chain = []

        if self._dimension == 3:
            for z in range(idx_lower[2], idx_upper[2] + 1):
                z_offset = z * stride_z
                for y in range(idx_lower[1], idx_upper[1] + 1):
                    y_offset = z_offset + y * stride_y
                    start = y_offset + idx_lower[0]
                    end = y_offset + idx_upper[0] + 1
                    for bin_idx in range(start, end):
                        if bins[bin_idx]:
                            lists_to_chain.append(bins[bin_idx])

        elif self._dimension == 2:
            for y in range(idx_lower[1], idx_upper[1] + 1):
                y_offset = y * stride_y
                start = y_offset + idx_lower[0]
                end = y_offset + idx_upper[0] + 1
                for bin_idx in range(start, end):
                    if bins[bin_idx]:
                        lists_to_chain.append(bins[bin_idx])

        elif self._dimension == 1:
            for bin_idx in range(idx_lower[0], idx_upper[0] + 1):
                if bins[bin_idx]:
                    lists_to_chain.append(bins[bin_idx])

        if not lists_to_chain:
            return set()

        return set(itertools.chain.from_iterable(lists_to_chain))


class KDBinOrganizedParticleManager(BaseParticleManager):
    def __init__(
        self,
        particleKernelDomain: ParticleKernelDomain,
        dimension: int,
        journal: Journal,
        bondParticlesToKernelFunctions: bool = False,
        randomlyShiftPartliceShapeFunctions: Union[bool, float] = False,
    ):

        self._meshfreeKernelFunctions = particleKernelDomain.meshfreeKernelFunctions
        self._particles = particleKernelDomain.particles
        self._dimension = dimension
        self._bondParticlesToKernelFunctions = bondParticlesToKernelFunctions
        self._journal = journal

        if isFreeThreadingSupported():
            self._numThreads = getNumberOfThreads()
        else:
            self._numThreads = 1

        # Pre-fetch labels for integer sorting
        self._kernelLabels = np.array([k.node.label for k in self._meshfreeKernelFunctions], dtype=int)

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

    def signalizeKernelFunctionUpdate(self) -> None:
        self._theBins = _FastKDBinOrganizer(list(self._meshfreeKernelFunctions), self._dimension)

    def updateConnectivity(self) -> bool:
        hasChanged = False

        if self._bondParticlesToKernelFunctions:
            self._journal.message("Updating kernel function positions...", "ParticleManager")
            for particle, kernelFunction in zip(self._particles, self._meshfreeKernelFunctions):
                particleCoordinates = particle.getCenterCoordinates()

                if self._randomlyShiftPartliceShapeFunctions:
                    if isinstance(self._randomlyShiftPartliceShapeFunctions, float):
                        particleVol = particle.getVolumeUndeformed()
                        particleSize = particleVol ** (1.0 / self._dimension)

                        randdisp = (
                            (np.random.rand(self._dimension) - 0.5)
                            * np.sqrt(particle.getVolumeUndeformed())
                            * self._randomlyShiftPartliceShapeFunctions
                            * particleSize
                        )
                        particleCoordinates += randdisp

                kernelFunction.moveTo(particleCoordinates)

            self.signalizeKernelFunctionUpdate()

        # Capture variables for closure
        all_kernels = self._meshfreeKernelFunctions
        kernel_mins = self._theBins._mins
        kernel_maxs = self._theBins._maxs
        bin_organizer = self._theBins
        kernel_labels = self._kernelLabels
        dim = self._dimension

        def processParticleChunk(particleChunk: List[Any]) -> bool:
            particlesInChunkHaveChanged = False

            for p in particleChunk:
                evaluationCoordinates = p.getEvaluationCoordinates()

                # Broad Phase Min/Max Calculation
                if len(evaluationCoordinates) == 1:
                    p_min = evaluationCoordinates[0]
                    p_max = evaluationCoordinates[0]
                else:
                    p_min = np.min(evaluationCoordinates, axis=0)
                    p_max = np.max(evaluationCoordinates, axis=0)

                # 1. Grid Search
                candidate_indices = bin_organizer.getCandidateIndices(p_min, p_max)

                # 2. Vectorized AABB Filter
                cand_idx_arr = np.array(list(candidate_indices), dtype=int)

                c_mins = kernel_mins[cand_idx_arr, :dim]
                c_maxs = kernel_maxs[cand_idx_arr, :dim]
                p_max_s = p_max[:dim]
                p_min_s = p_min[:dim]

                overlap_mask = np.all((p_max_s >= c_mins) & (p_min_s <= c_maxs), axis=1)
                surviving_indices = cand_idx_arr[overlap_mask]

                # 3. Precise Check (Geometric)
                valid_indices = []

                # Ensure coordinates are 2D for the Cython signature
                eval_coords_view = evaluationCoordinates
                if eval_coords_view.ndim == 1:
                    # Reshape (dim,) -> (1, dim)
                    eval_coords_view = eval_coords_view.reshape(1, -1)

                for k_idx in surviving_indices:
                    sf = all_kernels[k_idx]

                    if sf.isAnyCoordinateInSupport(eval_coords_view):
                        valid_indices.append(k_idx)

                valid_indices.sort(key=lambda idx: kernel_labels[idx])
                validKernels = [all_kernels[i] for i in valid_indices]

                if not validKernels:
                    raise ValueError(
                        f"Particle at {p.getCenterCoordinates()} has no associated kernel functions after connectivity update."
                    )

                if validKernels != p.kernelFunctions:
                    particlesInChunkHaveChanged = True

                p.assignKernelFunctions(
                    validKernels
                )  # assign the kernel functions. This happens even if they are the same, because the overlap with the particle usually changes due to movement.

            return particlesInChunkHaveChanged

        if self._numThreads <= 1:
            if processParticleChunk(self._particles):
                hasChanged = True
        else:
            chunkSize = len(self._particles) // self._numThreads + 1
            chunks = [self._particles[i : i + chunkSize] for i in range(0, len(self._particles), chunkSize)]

            with concurrent.futures.ThreadPoolExecutor(max_workers=self._numThreads) as executor:
                results = executor.map(processParticleChunk, chunks)

            if any(results):
                hasChanged = True

        return hasChanged

    def getCoveredDomain(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        return self._theBins._boundingBoxMin, self._theBins._boundingBoxMax

    def __str__(self) -> str:
        return (
            f"KDBinOrganizedParticleManager with {len(self._particles)} particles "
            f"and {len(self._meshfreeKernelFunctions)} shape functions "
            f"in {self._dimension} dimensions. Covered domain: {self.getCoveredDomain()}."
        )

    def visualize(self) -> None:
        if self._dimension != 2:
            raise ValueError("Visualization only supported for 2D.")

        import matplotlib.pyplot as plt

        nBins = self._theBins._nBins
        nKernelFunctions = np.zeros(nBins)

        for i in range(nBins[0]):
            for j in range(nBins[1]):
                flat_idx = j * self._theBins._strides[1] + i
                nKernelFunctions[i, j] = len(self._theBins._bins[flat_idx])

        plt.figure()
        plt.imshow(nKernelFunctions.T, origin="lower")
        plt.title("Number of kernel functions in the bins")

        for i in range(nBins[0] + 1):
            plt.plot([i - 0.5, i - 0.5], [0 - 0.5, nBins[1] - 0.5], "k")
        for j in range(nBins[1] + 1):
            plt.plot([0 - 0.5, nBins[0] - 0.5], [j - 0.5, j - 0.5], "k")

        plt.colorbar()
        plt.show()
