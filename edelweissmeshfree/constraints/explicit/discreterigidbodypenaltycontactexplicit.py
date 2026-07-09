from collections.abc import Iterable

import numpy as np
from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.timesteppers.timestep import TimeStep

from edelweissmeshfree.constraints.base.mpmconstraintbase import MPMConstraintBase
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.particles.base.baseparticle import BaseParticle
from edelweissmeshfree.utils.discretesurfacequery import DiscreteSurfaceQuery


class DiscreteRigidBodyPenaltyContactExplicit(MPMConstraintBase):
    """
    Vectorized penalty contact constraint for discrete rigid bodies in EXPLICIT simulations.

    Parameters
    ----------
    name : str
        Name of the constraint.
    particles : Iterable[BaseParticle]
        The collection of particles to which the constraint is applied.
    model : MPMModel
        The MPM model instance.
    penaltyParameter : float
        The penalty stiffness parameter. Default is 1e5.
    doProximityCheck : bool
        If True, only particles within a threshold are considered active. (Current vectorization inherently checks all).
    proximityFactor : float
        Multiplier for the proximity distance threshold. Default is 2.0.
    """

    def __init__(
        self,
        name: str,
        particles: Iterable[BaseParticle],
        model: MPMModel,
        rigidBody,
        penaltyParameter: float = 1e5,
        doProximityCheck: bool = True,
        proximityFactor: float = 2.0,
    ):
        self._name = name
        self._field = "displacement"
        self._fieldSize = getFieldSize(self._field, model.domainSize)
        self._nodes = []
        self._node_to_idx = {}
        self._model = model

        self.particles = list(particles)
        self.rigidBody = rigidBody
        self.rigidBodyRPNode = rigidBody.rpNode

        self.reactionForce = 0.0

        self._domainSize = model.domainSize
        self._penaltyParameter = penaltyParameter
        self._doProximityCheck = doProximityCheck
        self.proximityFactor = proximityFactor

        self._aabb_min = None
        self._aabb_max = None

        self.isActive = True

    @property
    def name(self) -> str:
        return self._name

    @property
    def nodes(self) -> list:
        return self._nodes

    @property
    def fieldsOnNodes(self) -> list:
        fields = []
        for n in self._nodes:
            if n is self.rigidBodyRPNode:
                fields.append(["displacement", "rotation"] if self._domainSize == 3 else ["displacement"])
            else:
                fields.append([self._field])
        return fields

    @property
    def nDof(self) -> int:
        dofs = 0
        for f in self.fieldsOnNodes:
            for field_name in f:
                dofs += (
                    self._fieldSize
                )  # displacement is domainSize, rotation is 3 or 1. Let's assume fieldSize applies.
                # Actually, displacement is _domainSize. rotation in 3D is 3.
                if field_name == "rotation" and self._domainSize == 3:
                    pass  # We already added self._fieldSize assuming it's 3.
        return dofs

    @property
    def scalarVariables(self) -> list:
        return []

    @property
    def active(self) -> bool:
        return self.isActive

    def getNumberOfAdditionalNeededScalarVariables(self) -> int:
        return 0

    def assignAdditionalScalarVariables(self, scalarVariables: list):
        pass

    def updateConnectivity(self, model: MPMModel) -> bool:
        """
        Updates the constraint connectivity by gathering nodes from all particles in the collection,
        plus the rigid body RP node.
        """
        all_nodes = set()
        for p in self.particles:
            for kf in p.kernelFunctions:
                all_nodes.add(kf.node)

        # Add the RP node to receive reaction forces
        all_nodes.add(self.rigidBodyRPNode)

        nodes = list(all_nodes)
        nodes.sort(key=lambda n: n.label)
        hasChanged = nodes != self._nodes

        self._nodes = nodes
        self._node_to_idx = {node: i for i, node in enumerate(self._nodes)}

        # Build DOF offsets
        self._node_to_offset = {}
        current_offset = 0
        for n in self._nodes:
            self._node_to_offset[n] = current_offset
            current_offset += self._fieldSize
            if n is self.rigidBodyRPNode and self._domainSize == 3:
                current_offset += 3  # For rotation

        # The constraint is always "active" because we evaluate the entire collection in applyConstraint
        # and only apply forces to those actually penetrating.
        self.isActive = True

        return hasChanged

    def applyConstraint(self, PExt: np.ndarray, timeStep: TimeStep):
        """
        Applies penalty forces to the local constraint force vector.
        """
        if not self.isActive or not self.particles:
            return

        # 1. Gather all particle coordinates (Vectorized)
        coords = np.array([p.getCenterCoordinates() for p in self.particles])
        # Sometimes particles return a list wrapping the array
        if len(coords.shape) > 2 or (len(coords) > 0 and isinstance(coords[0], list)):
            coords = np.array([c[0] if isinstance(c, list) else c for c in coords])

        # 2. Broadphase Proximity Check (AABB)
        active_indices = np.arange(len(self.particles))
        if self._doProximityCheck and hasattr(self.rigidBody, "getAABB"):
            curr_min, curr_max = self.rigidBody.getAABB()
            aabb_min = curr_min - self.proximityFactor
            aabb_max = curr_max + self.proximityFactor

            in_aabb = np.all((coords >= aabb_min) & (coords <= aabb_max), axis=1)
            active_indices = np.where(in_aabb)[0]
            if len(active_indices) == 0:
                self.reactionForce = 0.0
                return
            coords_to_query = coords[active_indices]
        else:
            coords_to_query = coords

        # 3. Narrowphase Query (VTK)
        dists, normals = self.rigidBody.querySurface(coords_to_query)

        # 4. Filter penetrating
        penetrating_mask = dists < 0
        if not np.any(penetrating_mask):
            self.reactionForce = 0.0
            return

        pen_indices = active_indices[penetrating_mask]
        pen_dists = dists[penetrating_mask]
        pen_normals = normals[penetrating_mask]
        pen_coords = coords_to_query[penetrating_mask]

        # 5. Calculate force vectors
        g = np.abs(pen_dists)
        grad_norms = np.linalg.norm(pen_normals, axis=1, keepdims=True)
        valid_mask = grad_norms[:, 0] > 1e-14
        pen_normals[valid_mask] = pen_normals[valid_mask] / grad_norms[valid_mask]
        pen_normals[~valid_mask] = 0.0

        forces = self._penaltyParameter * g[:, np.newaxis] * pen_normals

        # 6. Apply to RP
        rp_offset = self._node_to_offset[self.rigidBodyRPNode]
        if self._domainSize == 3:
            if hasattr(self.rigidBody, "getCurrentKinematics"):
                u_rp, _, rp_initial = self.rigidBody.getCurrentKinematics()
                rp_pos = rp_initial + u_rp
            else:
                rp_pos = self.rigidBodyRPNode.coordinates
                
            r = pen_coords - rp_pos
            moments = np.cross(r, -forces)
            for d in range(3):
                PExt[rp_offset + d] -= np.sum(forces[:, d])
                PExt[rp_offset + 3 + d] += np.sum(moments[:, d])
        else:
            for d in range(self._domainSize):
                PExt[rp_offset + d] -= np.sum(forces[:, d])

        # 7. Scatter forces to particles (Nodes)
        # Because each particle has different nodes, this requires a loop over active particles
        for idx, (p_idx, force_vector, coord) in enumerate(zip(pen_indices, forces, pen_coords)):
            p = self.particles[p_idx]
            N = p.getInterpolationVector(coord).flatten()
            for j, kf in enumerate(p.kernelFunctions):
                offset = self._node_to_offset[kf.node]
                for d in range(self._domainSize):
                    PExt[offset + d] += N[j] * force_vector[d]

        self.reactionForce = np.sum(self._penaltyParameter * g)


def DiscreteRigidBodyPenaltyContactExplicitFactory(
    name: str,
    particleCollection: Iterable[BaseParticle],
    model: MPMModel,
    rigidBody,
    penaltyParameter: float = 1e5,
    doProximityCheck: bool = True,
    proximityFactor: float = 2.0,
    filename: str = None,
    initial_offset: np.ndarray = None,
):
    """
    Factory function to create a vectorized Explicit Penalty Contact constraint for a discrete rigid body.
    Automatically registers the constraint in the model.

    Parameters
    ----------
    name : str
        Name of the constraint.
    particleCollection : Iterable[BaseParticle]
        Particles to check for contact.
    model : MPMModel
        The MPM model instance.
    rigidBody : DiscreteRigidBody
        The discrete rigid body entity.
    """
    constraint = DiscreteRigidBodyPenaltyContactExplicit(
        name=name,
        particles=particleCollection,
        model=model,
        rigidBody=rigidBody,
        penaltyParameter=penaltyParameter,
        doProximityCheck=doProximityCheck,
        proximityFactor=proximityFactor,
    )
    
    # Automatically register in the model
    model.constraints[name] = constraint
    model.constraintSets[name] = constraint
    
    return constraint
