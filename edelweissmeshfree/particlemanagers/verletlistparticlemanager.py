import concurrent.futures
from typing import Any, List, Union

import numpy as np
from edelweissfe.journal.journal import Journal

from edelweissmeshfree.meshfree.particlekerneldomain import ParticleKernelDomain
from edelweissmeshfree.particlemanagers.kdbinorganizedparticlemanager import (
    KDBinOrganizedParticleManager,
)


class VerletListParticleManager(KDBinOrganizedParticleManager):
    """
    A particle manager that extends the KDBinOrganizedParticleManager to use a
    Verlet list (skin margin) algorithm. This optimizes the connectivity update
    by skipping the global spatial hashing and geometric AABB filter for most
    time steps, only rebuilding the list when particles move beyond the skin margin.
    """

    def __init__(
        self,
        particleKernelDomain: ParticleKernelDomain,
        dimension: int,
        journal: Journal,
        bondParticlesToKernelFunctions: bool = False,
        randomlyShiftPartliceShapeFunctions: Union[bool, float] = False,
        skinMargin: float = 0.5,
    ):
        super().__init__(
            particleKernelDomain,
            dimension,
            journal,
            bondParticlesToKernelFunctions,
            randomlyShiftPartliceShapeFunctions,
        )
        self._skinMargin = skinMargin
        self._needsRebuild = True
        self._lastRebuildParticleCoords = None
        self._lastRebuildKernelCoords = None
        self._verlet_candidates_map = {}

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

        # Check if rebuild is needed based on displacements
        currentParticleCoords = np.array([p.getCenterCoordinates() for p in self._particles])
        currentKernelCoords = np.array([k.node.coordinates for k in self._meshfreeKernelFunctions])

        if self._lastRebuildParticleCoords is None:
            self._needsRebuild = True
        else:
            # Max displacement of any particle
            max_p_disp = np.max(np.linalg.norm(currentParticleCoords - self._lastRebuildParticleCoords, axis=1))
            # Max displacement of any kernel
            max_k_disp = np.max(np.linalg.norm(currentKernelCoords - self._lastRebuildKernelCoords, axis=1))

            if max_p_disp + max_k_disp >= self._skinMargin:
                self._needsRebuild = True

        if self._needsRebuild:
            self._lastRebuildParticleCoords = currentParticleCoords.copy()
            self._lastRebuildKernelCoords = currentKernelCoords.copy()

        # Capture variables for closure
        all_kernels = self._meshfreeKernelFunctions
        kernel_mins = self._theBins._mins
        kernel_maxs = self._theBins._maxs
        bin_organizer = self._theBins
        kernel_labels = self._kernelLabels
        dim = self._dimension
        skin_margin = self._skinMargin

        # Rebuild candidates list sequentially if needed (rebuild is infrequent)
        if self._needsRebuild:
            self._verlet_candidates_map = {}
            for p in self._particles:
                evaluationCoordinates = p.getEvaluationCoordinates()
                if len(evaluationCoordinates) == 1:
                    p_min = evaluationCoordinates[0]
                    p_max = evaluationCoordinates[0]
                else:
                    p_min = np.min(evaluationCoordinates, axis=0)
                    p_max = np.max(evaluationCoordinates, axis=0)

                # Inflate search box by skin margin
                p_min_inflated = p_min.copy()
                p_max_inflated = p_max.copy()
                p_min_inflated[:dim] -= skin_margin
                p_max_inflated[:dim] += skin_margin

                candidate_indices = bin_organizer.getCandidateIndices(p_min_inflated, p_max_inflated)
                cand_idx_arr = np.array(list(candidate_indices), dtype=int)
                if len(cand_idx_arr) > 0:
                    c_mins = kernel_mins[cand_idx_arr, :dim]
                    c_maxs = kernel_maxs[cand_idx_arr, :dim]

                    p_max_s_inf = p_max_inflated[:dim]
                    p_min_s_inf = p_min_inflated[:dim]

                    overlap_mask = np.all((p_max_s_inf >= c_mins) & (p_min_s_inf <= c_maxs), axis=1)
                    self._verlet_candidates_map[id(p)] = cand_idx_arr[overlap_mask]
                else:
                    self._verlet_candidates_map[id(p)] = np.array([], dtype=int)

        verlet_map = self._verlet_candidates_map

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

                p_min_s = p_min[:dim]
                p_max_s = p_max[:dim]

                # 3. Precise Check (Geometric) using exact non-inflated coordinates
                valid_indices = []

                # Ensure coordinates are 2D for the Cython signature
                eval_coords_view = evaluationCoordinates
                if eval_coords_view.ndim == 1:
                    # Reshape (dim,) -> (1, dim)
                    eval_coords_view = eval_coords_view.reshape(1, -1)

                candidates = verlet_map.get(id(p))
                if candidates is not None and len(candidates) > 0:
                    c_mins = kernel_mins[candidates, :dim]
                    c_maxs = kernel_maxs[candidates, :dim]
                    overlap_mask = np.all((p_max_s >= c_mins) & (p_min_s <= c_maxs), axis=1)
                    precise_candidates = candidates[overlap_mask]

                    for k_idx in precise_candidates:
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

                p.assignKernelFunctions(validKernels)

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

        self._needsRebuild = False
        return hasChanged
