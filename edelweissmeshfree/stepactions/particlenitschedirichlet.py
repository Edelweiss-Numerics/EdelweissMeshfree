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
#  Matthias Neuner |  matthias.neuner@boku.ac.at
#  Thomas Mader    |  thomas.mader@boku.ac.at
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
import numpy as np
from edelweissfe.journal.journal import Journal
from edelweissfe.surfaces.entitybasedsurface import EntityBasedSurface
from edelweissfe.timesteppers.timestep import TimeStep

from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.particles.base.baseparticle import BaseParticle


class ParticleNitscheDirichlet:
    """Weak Dirichlet boundary condition via (non-symmetric/incomplete) Nitsche's method.

    Enforces prescribed displacement components on a particle boundary surface through the
    particle's 'nitschedirichlet' distributed-load type, which assembles

      - the CONSISTENCY term  -int_Gamma T . mask o ((S_dev + p I) DeltaF^-T (N dA)_Y)
        (the boundary traction the weak form drops under weak Dirichlet enforcement), and
      - the PENALTY term      +int_Gamma beta T mask o (u - g) dA

    directly into the momentum residual -- no Lagrange multipliers or mortar fields are
    needed, and the boundary traction is smooth by construction (no point-collocation
    checkerboard). The adjoint (symmetry) term is omitted (theta = 0).

    Unlike ParticleDistributedLoad, only the prescribed value g is ramped by the step
    amplitude; the component mask and the penalty parameter beta are constant. The load
    vector handed to the particle is [ mask (nDim) | g(t) (nDim) | beta ].

    Parameters
    ----------
    name
        Name of the boundary condition.
    model
        The MPM model tree.
    journal
        The journal to write messages to.
    particleSurface
        The surface (faceID -> particles) on which the condition acts.
    prescribedComponents
        Mapping component index -> prescribed TOTAL value at step end, e.g. {0: 0.0, 1: 5.0}.
        Components not listed remain traction-free (natural condition).
    beta
        Nitsche penalty parameter (force / area / displacement). A good scale in the
        nearly incompressible regime is O(10) * K / h.
    dimension
        Spatial dimension (default 2).
    f_t
        Amplitude function of the prescribed value over step progress (default linear).
    """

    def __init__(
        self,
        name: str,
        model: MPMModel,
        journal: Journal,
        particleSurface: EntityBasedSurface,
        prescribedComponents: dict[int, float],
        beta: float,
        dimension: int = 2,
        f_t=None,
    ):
        self.name = name
        self._dimension = dimension
        self._particleSurface = particleSurface
        self._beta = beta

        self._mask = np.zeros(dimension)
        gFinal = np.zeros(dimension)
        for component, value in prescribedComponents.items():
            self._mask[component] = 1.0
            gFinal[component] = value

        self._gAtStepStart = np.zeros(dimension)
        self._delta = gFinal
        self._amplitude = f_t if f_t is not None else (lambda t: t)
        self._idle = False

        # cached at every getCurrentParticleLoads call; lets reactionForce evaluate the
        # boundary terms at the converged state without knowing the time step
        self._lastLoads: list[tuple[BaseParticle, int, np.ndarray]] = []

    @property
    def loadType(self) -> str:
        return "nitschedirichlet"

    def applyAtStepEnd(self, model, stepMagnitude=None):
        if not self._idle:
            if stepMagnitude is None:
                self._gAtStepStart = self._gAtStepStart + self._delta * self._amplitude(1.0)
            else:
                self._gAtStepStart = self._gAtStepStart + self._delta * stepMagnitude

            self._delta = 0
            self._idle = True

    def getCurrentParticleLoads(self, timeStep: TimeStep) -> list[tuple[BaseParticle, int, np.ndarray]]:
        """Assemble the per-particle load vectors [ mask | g(t) | beta ]."""

        if self._idle:
            t = 1.0
        else:
            t = timeStep.stepProgress

        g = self._gAtStepStart + self._delta * self._amplitude(t)
        loadVec = np.concatenate([self._mask, g, [self._beta]])

        particleLoads = list()
        for surfaceID, pSet in self._particleSurface.items():
            particleLoads += [(p, surfaceID, loadVec) for p in pSet]

        self._lastLoads = particleLoads
        return particleLoads

    @property
    def reactionForce(self) -> np.ndarray:
        """Total force transmitted through the constrained surface: the sum of the
        assembled Nitsche boundary forces (consistency + penalty) over all nodes,
        evaluated at the current particle state (call after the increment converged)."""

        nDim = self._dimension
        R = np.zeros(nDim)
        for p, surfaceID, loadVec in self._lastLoads:
            Pc = np.zeros(p.nDof)
            Kc = np.zeros(p.nDof * p.nDof)
            p.computeDistributedLoad(self.loadType, surfaceID, loadVec, Pc, Kc, 0.0, 0.0)
            # Pc stores the NEGATIVE applied force; per-node layout (u..., p, j)
            R -= Pc.reshape(-1, p.nDof // len(p.nodes))[:, :nDim].sum(axis=0)
        return R
