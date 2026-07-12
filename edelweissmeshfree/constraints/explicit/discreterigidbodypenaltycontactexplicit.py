from collections.abc import Iterable

import numpy as np
from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.timesteppers.timestep import TimeStep

from edelweissmeshfree.constraints.base.mpmconstraintbase import MPMConstraintBase
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.particles.base.baseparticle import BaseParticle


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
    rigidBody : DiscreteRigidBody
        The rigid body to interact with.
    penaltyParameter : float, optional
        The penalty stiffness parameter. Default is 1e5.
    doProximityCheck : bool, optional
        If True, a broadphase AABB check is performed to skip distant points.
    proximityFactor : float, optional
        Multiplier/padding for the proximity distance threshold. Default is 2.0.
    """

    def __init__(
        self,
        name: str,
        particles: Iterable[BaseParticle],
        model: MPMModel,
        rigidBody,
        location: str = "center",
        faceIDs: list[int] | int = None,
        vertexIDs: list[int] | int = None,
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
        self.location = location
        if location == "face" and faceIDs is None:
            raise ValueError("faceIDs must be specified when location is 'face'.")
        if location == "vertex" and vertexIDs is None:
            raise ValueError("vertexIDs must be specified when location is 'vertex'.")
        self.faceIDs = faceIDs
        self.vertexIDs = vertexIDs
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

        Parameters
        ----------
        model : MPMModel
            The current simulation model.

        Returns
        -------
        hasChanged : bool
            True if the connectivity has changed since the last update, False otherwise.
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
        Applies penalty forces to the global external force vector for penetrating particles.

        Parameters
        ----------
        PExt : numpy.ndarray
            The global external force vector. Modified in place.
        timeStep : TimeStep
            The current time step information.
        """
        if not self.isActive or not self.particles:
            return

        # 1. Gather all query coordinates and map them back to particles
        coords_list = []
        particle_mapping = []
        for i, p in enumerate(self.particles):
            if self.location == "center":
                pts = [p.getCenterCoordinates()]
            elif self.location == "face":
                ids = [self.faceIDs] if isinstance(self.faceIDs, int) else self.faceIDs
                pts = [p.getFaceCoordinates(idx) for idx in ids]
            elif self.location == "vertex":
                ids = [self.vertexIDs] if isinstance(self.vertexIDs, int) else self.vertexIDs
                vertices = p.getVertexCoordinates()
                pts = [vertices[idx] for idx in ids]
            else:
                pts = []

            for pt in pts:
                # Sometimes particles return a list wrapping the array
                pt_val = pt[0] if isinstance(pt, list) else pt
                coords_list.append(pt_val)
                particle_mapping.append(i)

        coords = np.array(coords_list)
        particle_mapping = np.array(particle_mapping)

        # 2. Query Rigid Body (Handles Broadphase AABB internally)
        proximity = self.proximityFactor if self._doProximityCheck else None
        dists, normals = self.rigidBody.querySurface(coords, proximity_factor=proximity)

        # 3. Filter penetrating (dists < 0 evaluates to False for np.inf)
        penetrating_mask = dists < 0
        if not np.any(penetrating_mask):
            self.reactionForce = 0.0
            return

        pen_indices = np.where(penetrating_mask)[0]
        pen_dists = dists[penetrating_mask]
        pen_normals = normals[penetrating_mask]
        pen_coords = coords[penetrating_mask]

        # 5. Calculate force vectors
        g = np.abs(pen_dists)
        grad_norms = np.linalg.norm(pen_normals, axis=1, keepdims=True)
        valid_mask = grad_norms[:, 0] > 1e-14
        pen_normals[valid_mask] = pen_normals[valid_mask] / grad_norms[valid_mask]
        pen_normals[~valid_mask] = 0.0

        forces = self._penaltyParameter * g[:, np.newaxis] * pen_normals

        forces = self._addFrictionForces(forces, pen_coords, pen_normals, pen_indices, g, particle_mapping)

        # 6. Apply to RP
        rp_offset = self._node_to_offset[self.rigidBodyRPNode]
        if self._domainSize == 3:
            u_rp, _, rp_initial = self.rigidBody.getCurrentKinematics()
            rp_pos = rp_initial + u_rp

            r = pen_coords - rp_pos
            moments = np.cross(r, -forces)
            for d in range(3):
                PExt[rp_offset + d] -= np.sum(forces[:, d])
                PExt[rp_offset + 3 + d] += np.sum(moments[:, d])
        else:
            for d in range(self._domainSize):
                PExt[rp_offset + d] -= np.sum(forces[:, d])

        # 6. Accumulate forces to the respective particles
        for i, pen_idx in enumerate(pen_indices):
            p = self.particles[particle_mapping[pen_idx]]
            force_vector = forces[i]
            coord = pen_coords[i]

            # Use EdelweissFE shape functions to distribute the force to the particle's nodes
            N = p.getInterpolationVector(coord).flatten()
            for j, kf in enumerate(p.kernelFunctions):
                offset = self._node_to_offset[kf.node]
                for d in range(self._domainSize):
                    PExt[offset + d] += N[j] * force_vector[d]

        self.reactionForce = np.sum(self._penaltyParameter * g)

    def _addFrictionForces(self, forces, pen_coords, pen_normals, pen_indices, g, particle_mapping):
        return forces


def DiscreteRigidBodyPenaltyContactExplicitFactory(
    name: str,
    particleCollection: Iterable[BaseParticle],
    model: MPMModel,
    rigidBody,
    location: str = "center",
    faceIDs: list[int] | int = None,
    vertexIDs: list[int] | int = None,
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
        The discrete rigid body entity to interact with.
    penaltyParameter : float, optional
        The penalty stiffness parameter. Default is 1e5.
    doProximityCheck : bool, optional
        If True, a broadphase AABB check is performed. Default is True.
    proximityFactor : float, optional
        Padding for the proximity AABB. Default is 2.0.
    filename : str, optional
        Deprecated file name parameter.
    initial_offset : numpy.ndarray, optional
        Deprecated offset parameter.

    Returns
    -------
    constraint : DiscreteRigidBodyPenaltyContactExplicit
        The created constraint instance.
    """
    constraint = DiscreteRigidBodyPenaltyContactExplicit(
        name=name,
        particles=particleCollection,
        model=model,
        rigidBody=rigidBody,
        location=location,
        faceIDs=faceIDs,
        vertexIDs=vertexIDs,
        penaltyParameter=penaltyParameter,
        doProximityCheck=doProximityCheck,
        proximityFactor=proximityFactor,
    )

    # Automatically register in the model
    model.constraints[name] = constraint
    model.constraintSets[name] = constraint

    return constraint
