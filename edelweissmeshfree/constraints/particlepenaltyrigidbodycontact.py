from collections.abc import Callable, Iterable

import numpy as np
from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.surfaces.entitybasedsurface import EntityBasedSurface
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.variables.scalarvariable import ScalarVariable

from edelweissmeshfree.constraints.base.mpmconstraintbase import MPMConstraintBase
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.particles.base.baseparticle import BaseParticle


class ParticlePenaltyContactImplicitSurfaceConstraint(MPMConstraintBase):
    """
    Penalty contact constraint for arbitrary implicit surfaces.

    Parameters
    ----------
    implicit_function : function
        Returns signed distance. Convention: Negative = Inside (Penetration).
    gradient_function : function
        Returns gradient vector. Convention: Points OUTWARD (Normal).
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
        doProximityCheck: bool = True,
        proximityFactor: float = 2.0,
    ):
        self._name = name
        self._field = "displacement"
        self._fieldSize = getFieldSize(self._field, model.domainSize)
        self._nodes = dict()

        self._implicit_function = implicit_function
        self._gradient_function = gradient_function

        self.reactionForce = 0.0

        def _getConstraintLocations():
            if location == "center":
                [constraintLocation] = particle.getCenterCoordinates()
            elif location == "face":
                if faceIDs is None:
                    raise ValueError("faceID must be specified when location is 'face'.")
                ids = [faceIDs] if isinstance(faceIDs, int) else faceIDs
                constraintLocation = [particle.getFaceCoordinates(idx) for idx in ids]
            elif location == "vertex":
                if vertexIDs is None:
                    raise ValueError("vertexID must be specified when location is 'vertex'.")
                ids = [vertexIDs] if isinstance(vertexIDs, int) else vertexIDs
                constraintLocation = [particle.getVertexCoordinates()[idx] for idx in ids]
            return constraintLocation

        self._getConstraintLocations = _getConstraintLocations
        self._domainSize = model.domainSize
        self._constrained_points = self._getConstraintLocations()
        self._particle = particle

        self._penaltyParameter = penaltyParameter
        self._doProximityCheck = doProximityCheck

        if self._doProximityCheck:
            self._particleCharacteristicLength_x_proximityFactor = (
                particle.getVolumeUndeformed() ** (1.0 / self._domainSize) * proximityFactor
            )

        self.isActive = True

    @property
    def name(self) -> str:
        return self._name

    @property
    def nodes(self) -> list:
        return self._nodes

    @property
    def fieldsOnNodes(self) -> list:
        return [[self._field] for _ in self._nodes]

    @property
    def nDof(self) -> int:
        return len(self._nodes) * self._fieldSize

    @property
    def scalarVariables(self) -> list:
        return list()

    @property
    def active(self) -> bool:
        return self.isActive

    def getNumberOfAdditionalNeededScalarVariables(self) -> int:
        return 0

    def assignAdditionalScalarVariables(self, scalarVariables: list[ScalarVariable]):
        pass

    def updateConnectivity(self, model):
        # nodes = {n: i for i, n in enumerate(set(kf.node for kf in self._particle.kernelFunctions))}
        nodes = [kf.node for kf in self._particle.kernelFunctions]
        hasChanged = nodes != self._nodes
        self._nodes = nodes
        self._constrained_points = self._getConstraintLocations()

        wasActive = self.isActive
        self.isActive = True

        if self._doProximityCheck:
            self.isActive = False
            for p_coord in self._constrained_points:
                dist = self._implicit_function(p_coord)
                if dist < self._particleCharacteristicLength_x_proximityFactor:
                    self.isActive = True
                    break

            if self.isActive != wasActive:
                hasChanged = True

        if not self.isActive:
            self.reactionForce = 0.0

        return hasChanged

    def applyConstraint(self, dU_: np.ndarray, PExt: np.ndarray, V: np.ndarray, timeStep: TimeStep):
        if not self.isActive:
            return

        K_UU = V.reshape((self.nDof, self.nDof))

        dU = dU_

        for constrained_point in self._constrained_points:

            # 1. Geometry at t_n
            current_dist = self._implicit_function(constrained_point)
            grad = self._gradient_function(constrained_point)

            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-14:
                # Fallback for singularity
                normal = np.zeros(self._fieldSize)
                if self._fieldSize > 0:
                    normal[0] = 1.0
            else:
                normal = grad / grad_norm

            # 2. Kinematics (Displacement Increment)
            N_vec = self._particle.getInterpolationVector(constrained_point).flatten()

            deltaU_point = N_vec @ dU.reshape((-1, self._fieldSize))

            constraint_vec = -normal
            g = -current_dist + np.dot(constraint_vec, deltaU_point)

            if g < 0:
                continue

            # dg_dU = np.einsum('i,j->ij', constraint_vec, N_vec).flatten()

            dg_dU = np.zeros(len(dU.flatten()))
            for i in range(len(self._nodes)):

                start_idx = i * self._fieldSize
                dg_dU[start_idx : start_idx + self._fieldSize] = N_vec[i] * constraint_vec

            # Update Force (Using += just like original)
            PExt += self._penaltyParameter * dg_dU * g

            # Update Stiffness (Using += just like original)
            K_update = self._penaltyParameter * np.outer(dg_dU, dg_dU)

            K_UU += K_update

            self.reactionForce += self._penaltyParameter * g


def ParticlePenaltyContactImplicitSurfaceConstraintFactory(
    baseName: str,
    implicit_function: Callable[[np.ndarray], float],
    gradient_function: Callable[[np.ndarray], np.ndarray],
    particleCollection: Iterable[BaseParticle] | EntityBasedSurface,
    model: MPMModel,
    location: str = "center",
    faceIDs: list[int] | int = None,
    vertexIDs: list[int] | int = None,
    penaltyParameter: float = 1e5,
    doProximityCheck: bool = True,
    proximityFactor: float = 2.0,
):
    constraints = dict()

    if isinstance(particleCollection, EntityBasedSurface):
        for faceID, particles in particleCollection.items():
            for i, p in enumerate(particles):
                name = f"{baseName}_face{faceID}_{i}"
                constraints[name] = ParticlePenaltyContactImplicitSurfaceConstraint(
                    name,
                    p,
                    implicit_function,
                    gradient_function,
                    model,
                    location="face",
                    faceIDs=faceID,
                    penaltyParameter=penaltyParameter,
                    doProximityCheck=doProximityCheck,
                    proximityFactor=proximityFactor,
                )
        return constraints

    elif isinstance(particleCollection, Iterable):
        for i, p in enumerate(particleCollection):
            name = f"{baseName}_{i}"
            constraints[name] = ParticlePenaltyContactImplicitSurfaceConstraint(
                name,
                p,
                implicit_function,
                gradient_function,
                model,
                location,
                faceIDs,
                vertexIDs,
                penaltyParameter,
                doProximityCheck,
                proximityFactor,
            )
        return constraints

    else:
        raise TypeError("particleCollection must be a list of particles or an EntityBasedSurface.")
