# -*- coding: utf-8 -*-
import numpy as np
from edelweissfe.timesteppers.timestep import TimeStep

from edelweissmeshfree.constraints.discreterigidbodypenaltycontact import (
    DiscreteRigidBodyPenaltyContact,
)


class FrictionalDiscreteRigidBodyPenaltyContact(DiscreteRigidBodyPenaltyContact):
    """
    Frictional penalty contact constraint for discrete rigid bodies in implicit analyses.

    Mathematical Formulation
    ------------------------
    1. Tangential Kinematics:
       The relative tangential slip increment vector Δu_t is:
         Δu_t = P_t · (Δu_p - Δu_rp - Δθ_rp × r_s)
       where:
         - P_t = I - n_s ⊗ n_s is the projection tensor onto the tangent plane.

    2. Frictional Forces:
       The friction force is modeled using a regularized Coulomb friction law:
         f_t = -min(k_t · ||Δu_t||, μ · f_n) · t_s
       where:
         - k_t is the regularized tangential contact stiffness.
         - t_s = Δu_t / ||Δu_t|| is the slip direction unit vector.
         - f_n is the normal penalty force from the base class.

    3. Tangent Stiffness Contribution:
       - Sticking state (k_t · ||Δu_t|| < μ · f_n):
         K_stick = k_t · (G_t^T · G_t)
         where G_t = P_t · [ N_i · I,  -I,  r_s_hat^T ] is the slip gradient matrix.

       - Slipping state (k_t · ||Δu_t|| >= μ · f_n):
         K_slip = μ · k_n · (w_t ⊗ w) + (μ · f_n / ||Δu_t||) · G_t^T · (I - t_s ⊗ t_s) · G_t
         where:
           - w_t is the gradient of slip direction: w_t = [ N_i · t_s,  -t_s,  r_s × t_s ]
           - w is the normal contact gradient from the base class.
    """

    def __init__(self, *args, frictionCoefficient: float = 0.3, viscousRegularization: float = 1e6, **kwargs):
        super().__init__(*args, **kwargs)
        self.frictionCoefficient = frictionCoefficient
        self.tangentStiffness = viscousRegularization

        self._totalNormalForce = np.zeros(self._domainSize)
        self._totalFrictionForce = np.zeros(self._domainSize)

    @property
    def totalNormalForce(self) -> np.ndarray:
        return self._totalNormalForce

    @property
    def totalFrictionForce(self) -> np.ndarray:
        return self._totalFrictionForce

    def applyConstraint(self, dU: np.ndarray, PExt: np.ndarray, K, timeStep: TimeStep):
        if not self.isActive:
            self._totalNormalForce = np.zeros(self._domainSize)
            self._totalFrictionForce = np.zeros(self._domainSize)
            return

        K = self.shapeVIJContribution(K)

        self.reactionForce = 0.0
        nDim = self._domainSize
        nRot = 3

        self._totalNormalForce = np.zeros(nDim)
        self._totalFrictionForce = np.zeros(nDim)

        node_local_dofs = self._getLocalDofMapping()
        rp_dofs = node_local_dofs[self.rigidBody.rpNode]

        # RP start-of-step configuration
        _, _, rp_initial = self.rigidBody.getCurrentKinematics()

        # RP trial increments at current iteration
        delta_u_rp = dU[rp_dofs[0:nDim]]
        delta_theta_rp = dU[rp_dofs[nDim : nDim + nRot]]

        # Vectorized surface query
        coords = np.array([p.getCenterCoordinates() for p in self._candidates])
        dists, normals = self.rigidBody.querySurface(coords, proximity_factor=self.proximityFactor)

        for idx, p in enumerate(self._candidates):
            d0 = dists[idx]
            if d0 >= self.proximityFactor:
                continue

            n_s = normals[idx]
            x_s0 = coords[idx] - d0 * n_s
            r_s = x_s0 - rp_initial

            # Particle shape functions and background DOFs
            N_vec = p.getInterpolationVector(coords[idx]).flatten()
            p_nodes = [kf.node for kf in p.kernelFunctions]
            p_dofs = []
            for node in p_nodes:
                p_dofs.extend(node_local_dofs[node])

            delta_u_p = np.zeros(nDim)
            for i, node in enumerate(p_nodes):
                local_node_dofs = node_local_dofs[node]
                delta_u_p += N_vec[i] * dU[local_node_dofs]

            # 1. Normal Contact
            c = -n_s
            delta_u_rel = delta_u_p - delta_u_rp - np.cross(delta_theta_rp, r_s)
            g = -d0 + np.dot(c, delta_u_rel)

            if g < 0:
                continue

            # Normal gradient
            dg_dU = np.zeros(self.nDof)
            for i in range(len(p_nodes)):
                dg_dU[p_dofs[i * nDim : (i + 1) * nDim]] = N_vec[i] * c
            dg_dU[rp_dofs[0:nDim]] = -c
            dg_dU[rp_dofs[nDim : nDim + nRot]] = np.cross(r_s, c)

            f_n_mag = self._penaltyParameter * g
            self.reactionForce += f_n_mag
            self._totalNormalForce += f_n_mag * n_s

            # 2. Tangential Frictional Contact
            P_t = np.eye(nDim) - np.outer(n_s, n_s)
            delta_u_t = P_t @ delta_u_rel

            slip_norm = np.linalg.norm(delta_u_t)

            f_t_mag_stick = self.tangentStiffness * slip_norm
            f_t_mag_max = self.frictionCoefficient * f_n_mag

            is_slipping = (f_t_mag_stick >= f_t_mag_max) and (f_t_mag_max > 0.0)

            if slip_norm > 1e-12:
                t_s = delta_u_t / slip_norm
            else:
                t_s = np.zeros(nDim)

            if is_slipping:
                f_t = -f_t_mag_max * t_s
            else:
                f_t = -self.tangentStiffness * delta_u_t

            self._totalFrictionForce += f_t

            # 3. Assemble residual vectors
            f_contact = f_n_mag * n_s + f_t

            for i in range(len(p_nodes)):
                PExt[p_dofs[i * nDim : (i + 1) * nDim]] -= N_vec[i] * f_contact
            PExt[rp_dofs[0:nDim]] += f_contact
            PExt[rp_dofs[nDim : nDim + nRot]] -= np.cross(r_s, f_contact)

            # 4. Tangent Stiffness matrix building
            K_mat = self._penaltyParameter * np.outer(dg_dU, dg_dU)
            K_geo = np.zeros((self.nDof, self.nDof))

            f_c_hat = np.array(
                [
                    [0.0, -f_contact[2], f_contact[1]],
                    [f_contact[2], 0.0, -f_contact[0]],
                    [-f_contact[1], f_contact[0], 0.0],
                ]
            )
            r_s_hat = np.array([[0.0, -r_s[2], r_s[1]], [r_s[2], 0.0, -r_s[0]], [-r_s[1], r_s[0], 0.0]])
            n_s_hat = np.array([[0.0, -n_s[2], n_s[1]], [n_s[2], 0.0, -n_s[0]], [-n_s[1], n_s[0], 0.0]])

            K_geo[np.ix_(rp_dofs[nDim : nDim + nRot], rp_dofs[nDim : nDim + nRot])] += f_c_hat @ r_s_hat

            for i in range(len(p_nodes)):
                K_geo[np.ix_(p_dofs[i * nDim : (i + 1) * nDim], rp_dofs[nDim : nDim + nRot])] += (
                    -f_n_mag * N_vec[i] * n_s_hat
                )
            K_geo[np.ix_(rp_dofs[0:nDim], rp_dofs[nDim : nDim + nRot])] += f_n_mag * n_s_hat

            # Frictional Stiffness
            K_fric = np.zeros((self.nDof, self.nDof))

            G_t = np.zeros((nDim, self.nDof))
            for i in range(len(p_nodes)):
                G_t[:, p_dofs[i * nDim : (i + 1) * nDim]] = N_vec[i] * P_t
            G_t[:, rp_dofs[0:nDim]] = -P_t
            G_t[:, rp_dofs[nDim : nDim + nRot]] = -P_t @ r_s_hat

            if is_slipping:
                w_t = np.zeros(self.nDof)
                for i in range(len(p_nodes)):
                    w_t[p_dofs[i * nDim : (i + 1) * nDim]] = N_vec[i] * t_s
                w_t[rp_dofs[0:nDim]] = -t_s
                w_t[rp_dofs[nDim : nDim + nRot]] = np.cross(r_s, t_s)

                K_fric += self.frictionCoefficient * self._penaltyParameter * np.outer(w_t, dg_dU)

                if slip_norm > 1e-12:
                    P_slip = np.eye(nDim) - np.outer(t_s, t_s)
                    K_fric += (f_t_mag_max / slip_norm) * (G_t.T @ P_slip @ G_t)
            else:
                K_fric += self.tangentStiffness * (G_t.T @ G_t)

            K_total = K_mat + K_geo + K_fric
            K.K_pp[idx] += K_total[np.ix_(p_dofs, p_dofs)]
            K.K_prp[idx] += K_total[np.ix_(p_dofs, rp_dofs)]
            K.K_rpp[idx] += K_total[np.ix_(rp_dofs, p_dofs)]
            K.K_rprp += K_total[np.ix_(rp_dofs, rp_dofs)]
