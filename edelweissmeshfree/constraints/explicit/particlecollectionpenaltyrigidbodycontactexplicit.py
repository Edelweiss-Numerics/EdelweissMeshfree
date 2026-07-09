from collections.abc import Iterable
import numpy as np
from edelweissfe.config.phenomena import getFieldSize
from edelweissfe.timesteppers.timestep import TimeStep
from edelweissfe.variables.scalarvariable import ScalarVariable
from edelweissmeshfree.constraints.base.mpmconstraintbase import MPMConstraintBase
from edelweissmeshfree.models.mpmmodel import MPMModel
from edelweissmeshfree.particles.base.baseparticle import BaseParticle
from edelweissmeshfree.utils.discretesurfacequery import DiscreteSurfaceQuery

class ParticleCollectionPenaltyContactDiscreteRigidBodyConstraintExplicit(MPMConstraintBase):
    """
    Vectorized penalty contact constraint for discrete rigid bodies in EXPLICIT simulations.

    Parameters
    ----------
    name : str
        Name of the constraint.
    particles : Iterable[BaseParticle]
        The collection of particles to which the constraint is applied.
    query_engine : DiscreteSurfaceQuery
        The vectorized query engine for the rigid body mesh.
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
        query_engine: DiscreteSurfaceQuery,
        model: MPMModel,
        rigidBodyRPNode,
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
        self.query_engine = query_engine
        self.rigidBodyRPNode = rigidBodyRPNode
        
        self.reactionForce = 0.0

        self._domainSize = model.domainSize
        self._penaltyParameter = penaltyParameter
        self._doProximityCheck = doProximityCheck

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
                dofs += self._fieldSize # displacement is domainSize, rotation is 3 or 1. Let's assume fieldSize applies.
                # Actually, displacement is _domainSize. rotation in 3D is 3. 
                if field_name == "rotation" and self._domainSize == 3:
                    pass # We already added self._fieldSize assuming it's 3.
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
                current_offset += 3 # For rotation
                
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

        total_reaction = 0.0

        # Vectorized gathering of coordinates
        coords = np.empty((len(self.particles), self._domainSize))
        for i, p in enumerate(self.particles):
            # p.getCenterCoordinates() returns a list with one item (the coord array)
            # wait, getCenterCoordinates() usually returns a list of coordinates. We assume center.
            c = p.getCenterCoordinates()
            if isinstance(c, list):
                coords[i] = c[0]
            else:
                coords[i] = c

        # Determine the rigid body's current translation from the RP node's accumulated
        # displacement in the model's NodeField. This mirrors the _getFieldU pattern used
        # by the kinematic tie and is the authoritative source of the RP displacement.
        disp_field = self._model.nodeFields.get("displacement")
        if disp_field is not None and "U" in disp_field:
            translation = disp_field.subset(self.rigidBodyRPNode)["U"][0].copy()
        else:
            translation = np.zeros(self._domainSize)

        # Query in the rigid body's local (reference) frame by shifting coords
        dists, normals = self.query_engine.query(coords, rigid_body_translation=translation)

        # Apply forces for penetrating particles
        for i, p in enumerate(self.particles):
            dist = dists[i]

            if dist < 0:
                g = abs(dist)
                normal = normals[i]
                
                # Check normal validity
                grad_norm = np.linalg.norm(normal)
                if grad_norm > 1e-14:
                    normal = normal / grad_norm
                else:
                    normal = np.zeros(self._domainSize)

                force_vector = self._penaltyParameter * g * normal

                # Get interpolation vector N mapping particle to grid nodes
                # Note: getInterpolationVector expects a coordinate.
                N = p.getInterpolationVector(coords[i]).flatten()
                
                p_nodes = [kf.node for kf in p.kernelFunctions]

                # Scatter to local Pc vector
                for j, node in enumerate(p_nodes):
                    offset = self._node_to_offset[node]
                    for d in range(self._domainSize):
                        PExt[offset + d] += N[j] * force_vector[d]

                # Apply equal and opposite reaction to the Rigid Body Reference Point
                rp_offset = self._node_to_offset[self.rigidBodyRPNode]
                for d in range(self._domainSize):
                    PExt[rp_offset + d] -= force_vector[d]
                
                # Calculate moment if 3D
                if self._domainSize == 3:
                    # Current RP position
                    rp_pos = self.rigidBodyRPNode.coordinates + translation
                    # Moment arm from RP to contact point (particle center)
                    r = coords[i] - rp_pos
                    # Moment = r x (-F)
                    moment = np.cross(r, -force_vector)
                    for d in range(3):
                        PExt[rp_offset + 3 + d] += moment[d]

                total_reaction += self._penaltyParameter * g

        self.reactionForce = total_reaction


def ParticleCollectionPenaltyContactDiscreteRigidBodyConstraintExplicitFactory(
    baseName: str,
    filename: str,
    particleCollection: Iterable[BaseParticle],
    model: MPMModel,
    rigidBodyRPNode,
    penaltyParameter: float = 1e5,
    doProximityCheck: bool = True,
    proximityFactor: float = 2.0,
    initial_offset: np.ndarray = None,
):
    """
    Factory function to create a vectorized Explicit Penalty Contact constraint for a discrete rigid body.

    Parameters
    ----------
    rigidBodyRPNode :
        The reference point node of the rigid body. Its displacement field is used as the
        rigid body's current translation when querying the (fixed-frame) surface locators.
    initial_offset : np.ndarray, optional
        A translation vector applied to the rigid body mesh before building
        the VTK locators.  Use this to position the contact surface at its
        physical initial location (matching the visualization geometry).
    """
    constraints = dict()
    
    # Initialize the high-performance distance query engine
    query_engine = DiscreteSurfaceQuery(filename, initial_offset=initial_offset)
    
    # We create a single constraint for the entire collection to maximize vectorization
    name = f"{baseName}_collection"
    constraints[name] = ParticleCollectionPenaltyContactDiscreteRigidBodyConstraintExplicit(
        name=name,
        particles=particleCollection,
        query_engine=query_engine,
        model=model,
        rigidBodyRPNode=rigidBodyRPNode,
        penaltyParameter=penaltyParameter,
        doProximityCheck=doProximityCheck,
        proximityFactor=proximityFactor,
    )
    
    return constraints
