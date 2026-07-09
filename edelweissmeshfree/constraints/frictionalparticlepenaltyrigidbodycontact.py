from collections.abc import Callable, Iterable

import numpy as np
from edelweissfe.surfaces.entitybasedsurface import EntityBasedSurface

from edelweissmeshfree.constraints.particlepenaltyrigidbodycontact import (
    ParticlePenaltyContactImplicitSurfaceConstraint,
)
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.particles.base.baseparticle import BaseParticle


class FrictionalParticlePenaltyContactImplicitSurfaceConstraint(
    ParticlePenaltyContactImplicitSurfaceConstraint
):
    """
    Frictional penalty contact constraint for arbitrary implicit surfaces.
    Uses a simple Coulomb stick-slip friction model with a tangential penalty.
    """

    def __init__(
        self,
        name: str,
        particle: BaseParticle,
        implicit_function: Callable[[np.ndarray], float],
        gradient_function: Callable[[np.ndarray], np.ndarray],
        model: MPMModel,
        location: str = "center",
        faceIDs: list[int] = None,
        vertexIDs: list[int] = None,
        penaltyParameter: float = 1e5,
        frictionCoefficient: float = 0.0,
        tangentialPenaltyParameter: float = None,
        doProximityCheck: bool = True,
        proximityFactor: float = 2.0,
    ):
        super().__init__(
            name=name,
            particle=particle,
            implicit_function=implicit_function,
            gradient_function=gradient_function,
            model=model,
            location=location,
            faceIDs=faceIDs,
            vertexIDs=vertexIDs,
            penaltyParameter=penaltyParameter,
            doProximityCheck=doProximityCheck,
            proximityFactor=proximityFactor,
        )

        self._frictionCoefficient = frictionCoefficient
        self._tangentialPenaltyParameter = (
            tangentialPenaltyParameter if tangentialPenaltyParameter is not None else penaltyParameter
        )
        
        # Track the accumulated friction force for stick-slip history
        self._frictional_forces = [
            np.zeros(self._fieldSize) for _ in self._constrained_points
        ]

    def updateConnectivity(self, model):
        hasChanged = super().updateConnectivity(model)
        
        # Reset friction history if the particle leaves contact
        if not self.isActive:
            for i in range(len(self._frictional_forces)):
                self._frictional_forces[i].fill(0.0)
                
        return hasChanged

    def _applyAdditionalForces(
        self,
        pt_idx: int,
        constrained_point: np.ndarray,
        normal: np.ndarray,
        N_vec: np.ndarray,
        deltaU_point: np.ndarray,
        g: float,
        PExt: np.ndarray,
        K_UU: np.ndarray,
    ):
        if self._frictionCoefficient <= 0.0:
            return

        # Tangential displacement increment
        deltaU_n = np.dot(deltaU_point, normal) * normal
        deltaU_t = deltaU_point - deltaU_n

        # Trial friction force (assuming f_t points against sliding)
        f_t_trial = self._frictional_forces[pt_idx] - self._tangentialPenaltyParameter * deltaU_t

        f_t_norm = np.linalg.norm(f_t_trial)
        f_n_mag = self._penaltyParameter * g
        f_t_max = self._frictionCoefficient * f_n_mag

        if f_t_norm > f_t_max:
            # Slip
            f_t = f_t_trial * (f_t_max / f_t_norm) if f_t_norm > 1e-14 else np.zeros_like(f_t_trial)
        else:
            # Stick
            f_t = f_t_trial

        self._frictional_forces[pt_idx] = f_t

        # Apply friction force to nodal external forces
        for i in range(len(self._nodes)):
            start_idx = i * self._fieldSize
            PExt[start_idx : start_idx + self._fieldSize] += N_vec[i] * f_t


def FrictionalParticlePenaltyContactImplicitSurfaceConstraintFactory(
    baseName: str,
    implicit_function: Callable[[np.ndarray], float],
    gradient_function: Callable[[np.ndarray], np.ndarray],
    particleCollection: Iterable[BaseParticle] | EntityBasedSurface,
    model: MPMModel,
    location: str = "center",
    faceIDs: list[int] | int = None,
    vertexIDs: list[int] | int = None,
    penaltyParameter: float = 1e5,
    frictionCoefficient: float = 0.0,
    tangentialPenaltyParameter: float = None,
    doProximityCheck: bool = True,
    proximityFactor: float = 2.0,
):
    constraints = []

    if isinstance(particleCollection, EntityBasedSurface):
        elements = particleCollection.getEntities()
    else:
        elements = particleCollection

    if location == "center":
        for i, particle in enumerate(elements):
            c = FrictionalParticlePenaltyContactImplicitSurfaceConstraint(
                name=f"{baseName}_{i}",
                particle=particle,
                implicit_function=implicit_function,
                gradient_function=gradient_function,
                model=model,
                location=location,
                penaltyParameter=penaltyParameter,
                frictionCoefficient=frictionCoefficient,
                tangentialPenaltyParameter=tangentialPenaltyParameter,
                doProximityCheck=doProximityCheck,
                proximityFactor=proximityFactor,
            )
            constraints.append(c)

    elif location == "face":
        for i, particle in enumerate(elements):
            faces = particle.geometry.getFaces() if faceIDs is None else faceIDs
            for j in faces:
                c = FrictionalParticlePenaltyContactImplicitSurfaceConstraint(
                    name=f"{baseName}_{i}_f{j}",
                    particle=particle,
                    implicit_function=implicit_function,
                    gradient_function=gradient_function,
                    model=model,
                    location=location,
                    faceIDs=j,
                    penaltyParameter=penaltyParameter,
                    frictionCoefficient=frictionCoefficient,
                    tangentialPenaltyParameter=tangentialPenaltyParameter,
                    doProximityCheck=doProximityCheck,
                    proximityFactor=proximityFactor,
                )
                constraints.append(c)

    elif location == "vertex":
        for i, particle in enumerate(elements):
            vertices = particle.geometry.getVertices() if vertexIDs is None else vertexIDs
            for j in vertices:
                c = FrictionalParticlePenaltyContactImplicitSurfaceConstraint(
                    name=f"{baseName}_{i}_v{j}",
                    particle=particle,
                    implicit_function=implicit_function,
                    gradient_function=gradient_function,
                    model=model,
                    location=location,
                    vertexIDs=j,
                    penaltyParameter=penaltyParameter,
                    frictionCoefficient=frictionCoefficient,
                    tangentialPenaltyParameter=tangentialPenaltyParameter,
                    doProximityCheck=doProximityCheck,
                    proximityFactor=proximityFactor,
                )
                constraints.append(c)

    return constraints
