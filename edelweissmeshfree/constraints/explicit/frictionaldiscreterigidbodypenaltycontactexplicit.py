import numpy as np

from edelweissmeshfree.constraints.explicit.discreterigidbodypenaltycontactexplicit import (
    DiscreteRigidBodyPenaltyContactExplicit,
)


class FrictionalDiscreteRigidBodyPenaltyContactExplicit(DiscreteRigidBodyPenaltyContactExplicit):
    """
    Frictional penalty contact constraint for discrete rigid bodies in EXPLICIT simulations.
    Uses a velocity-based regularized Coulomb friction model.
    """

    def __init__(
        self,
        *args,
        frictionCoefficient: float = 0.0,
        viscousRegularization: float = 1e4,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.frictionCoefficient = frictionCoefficient
        self.viscousRegularization = viscousRegularization

    def _addFrictionForces(self, forces, pen_coords, pen_normals, pen_indices, g, particle_mapping):
        if self.frictionCoefficient <= 0.0:
            return forces

        # Get rigid body velocities
        v_rp = np.zeros(self._domainSize)
        if hasattr(self.rigidBodyRPNode, "current_velocity"):
            v_rp = self.rigidBodyRPNode.current_velocity

        omega = np.zeros(3)
        if self._domainSize == 3 and hasattr(self.rigidBodyRPNode, "current_angular_velocity"):
            omega = self.rigidBodyRPNode.current_angular_velocity

        if hasattr(self.rigidBody, "getCurrentKinematics"):
            u_rp, _, rp_initial = self.rigidBody.getCurrentKinematics()
            rp_pos = rp_initial + u_rp
        else:
            rp_pos = self.rigidBodyRPNode.coordinates

        frictional_forces = np.zeros_like(forces)

        for i, pen_idx in enumerate(pen_indices):
            p = self.particles[particle_mapping[pen_idx]]
            coord = pen_coords[i]
            normal = pen_normals[i]

            # 1. Particle Velocity
            v_p = np.zeros(self._domainSize)
            N = p.getInterpolationVector(coord).flatten()
            for j, kf in enumerate(p.kernelFunctions):
                if getattr(kf.node, "_velocity_initialized", False):
                    v_p += N[j] * kf.node.current_velocity

            # 2. Rigid Body Velocity at contact point
            if self._domainSize == 3:
                r = coord - rp_pos
                v_rb = v_rp + np.cross(omega, r)
            else:
                v_rb = v_rp

            # 3. Relative Velocity
            v_rel = v_p - v_rb

            # 4. Tangential Velocity
            v_t = v_rel - np.dot(v_rel, normal) * normal
            v_t_norm = np.linalg.norm(v_t)

            if v_t_norm > 1e-12:
                # 5. Friction Force Magnitude
                f_n_mag = self._penaltyParameter * g[i]
                f_t_max = self.frictionCoefficient * f_n_mag

                # Regularized stick-slip (viscous limit)
                f_t_mag = min(self.viscousRegularization * v_t_norm, f_t_max)

                # Friction force opposes sliding
                f_t = -f_t_mag * (v_t / v_t_norm)
                frictional_forces[i] = f_t

        # Return total forces
        return forces + frictional_forces


def FrictionalDiscreteRigidBodyPenaltyContactExplicitFactory(
    name: str,
    particleCollection,
    model,
    rigidBody,
    location: str = "center",
    faceIDs=None,
    vertexIDs=None,
    penaltyParameter: float = 1e5,
    frictionCoefficient: float = 0.0,
    viscousRegularization: float = 1e4,
    doProximityCheck: bool = True,
    proximityFactor: float = 2.0,
):
    constraints = []

    # Get particles from collection
    if hasattr(particleCollection, "getEntities"):
        elements = particleCollection.getEntities()
    else:
        elements = list(particleCollection)

    # Note: Explicit solver applies everything in one constraint instance!
    c = FrictionalDiscreteRigidBodyPenaltyContactExplicit(
        name=name,
        particles=elements,
        model=model,
        rigidBody=rigidBody,
        location=location,
        faceIDs=faceIDs,
        vertexIDs=vertexIDs,
        penaltyParameter=penaltyParameter,
        frictionCoefficient=frictionCoefficient,
        viscousRegularization=viscousRegularization,
        doProximityCheck=doProximityCheck,
        proximityFactor=proximityFactor,
    )
    constraints.append(c)

    return constraints
