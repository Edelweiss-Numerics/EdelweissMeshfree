import numpy as np
import pyvista as pv


class RigidBodyMassProperties:
    """
    Utility to compute mass, center of mass, and inertia tensor from a closed surface mesh.
    Uses PyVista (VTK) for volume and center of mass via the divergence theorem.
    For the inertia tensor, we implement a surface integral over the triangles.
    """

    def __init__(self, filename: str, density: float, initial_offset: np.ndarray = None):
        self.mesh = pv.read(filename)
        if isinstance(self.mesh, pv.MultiBlock):
            self.mesh = self.mesh.combine()
        self.mesh = self.mesh.extract_surface().triangulate()

        if initial_offset is not None:
            self.mesh.points += np.asarray(initial_offset)

        self.density = density

        # PyVista computes exact volume and CM for closed meshes
        self.volume = self.mesh.volume
        self.mass = self.volume * self.density

        if self.volume <= 1e-12:
            raise ValueError("Mesh volume is zero. Ensure it is a closed, watertight surface.")

        self.center_of_mass = np.array(self.mesh.center_of_mass())

        # Compute Inertia Tensor at CM
        self.inertia_tensor = self._compute_inertia_tensor_at_cm()

    def _compute_inertia_tensor_at_cm(self):
        """
        Computes the inertia tensor for a closed triangular surface mesh around its center of mass.
        """
        # Shift points to CM
        pts = self.mesh.points - self.center_of_mass
        faces = self.mesh.faces.reshape(-1, 4)[:, 1:]  # (N, 3) since triangulated

        p1 = pts[faces[:, 0]]
        p2 = pts[faces[:, 1]]
        p3 = pts[faces[:, 2]]

        # Normal scaled by 2 * area (cross product of edges)
        normals = np.cross(p2 - p1, p3 - p1)

        # We need volume integrals of x^2, y^2, z^2, xy, yz, zx.
        # Using divergence theorem: \int_V x^2 dV = 1/3 \int_S x^3 n_x dS
        # For a triangle, we can use a Gaussian quadrature or exact formula.
        # Exact formula for \int_T x^3 n_x dS involves terms like x1^3 + x1^2 x2 + ...
        # A simpler way is to use a 3-point or 4-point quadrature on the triangle.

        # 3-point quadrature (midpoints of edges)
        m1 = (p1 + p2) / 2
        m2 = (p2 + p3) / 2
        m3 = (p3 + p1) / 2
        area = np.linalg.norm(normals, axis=1) / 2
        unit_normals = normals / (2 * area[:, None] + 1e-16)

        # Integrand evaluated at midpoints
        # I_xx = \int (y^2 + z^2) dV = 1/3 \int (y^3 n_y + z^3 n_z) dS
        # Actually divergence of (0, y^3/3, z^3/3) is y^2 + z^2.

        # Let's use exact volume integration by sum of signed tetrahedra from origin to triangle
        I = np.zeros((3, 3))
        for i in range(len(faces)):
            # Tetrahedron formed by origin, p1, p2, p3
            # Volume of tetra = 1/6 * det([p1, p2, p3])
            det = np.dot(p1[i], np.cross(p2[i], p3[i]))
            if abs(det) < 1e-14:
                continue
            v = det / 6.0

            # For a tetrahedron with one vertex at origin, the inertia tensor components are:
            # I_xx = v/10 * (y1^2 + y2^2 + y3^2 + y1*y2 + y2*y3 + y3*y1 + z1^2 + z2^2 + z3^2 + z1*z2 + z2*z3 + z3*z1)
            # etc. (See "Fast and Accurate Computation of Polyhedral Mass Properties", Mirtich 1996)

            x1, y1, z1 = p1[i]
            x2, y2, z2 = p2[i]
            x3, y3, z3 = p3[i]

            f_xx = x1 * x1 + x2 * x2 + x3 * x3 + x1 * x2 + x2 * x3 + x3 * x1
            f_yy = y1 * y1 + y2 * y2 + y3 * y3 + y1 * y2 + y2 * y3 + y3 * y1
            f_zz = z1 * z1 + z2 * z2 + z3 * z3 + z1 * z2 + z2 * z3 + z3 * z1

            f_xy = x1 * y1 + x2 * y2 + x3 * y3 + 0.5 * (x1 * y2 + x2 * y1 + x2 * y3 + x3 * y2 + x3 * y1 + x1 * y3)
            f_yz = y1 * z1 + y2 * z2 + y3 * z3 + 0.5 * (y1 * z2 + y2 * z1 + y2 * z3 + y3 * z2 + y3 * z1 + y1 * z3)
            f_zx = z1 * x1 + z2 * x2 + z3 * x3 + 0.5 * (z1 * x2 + z2 * x1 + z2 * x3 + z3 * x2 + z3 * x1 + z1 * x3)

            I[0, 0] += v * (f_yy + f_zz) / 10.0
            I[1, 1] += v * (f_xx + f_zz) / 10.0
            I[2, 2] += v * (f_xx + f_yy) / 10.0
            I[0, 1] -= v * f_xy / 10.0
            I[1, 2] -= v * f_yz / 10.0
            I[2, 0] -= v * f_zx / 10.0

        I[1, 0] = I[0, 1]
        I[2, 1] = I[1, 2]
        I[0, 2] = I[2, 0]

        return I * self.density


if __name__ == "__main__":
    props = RigidBodyMassProperties("rigid_body.exo", density=100.0)
    print("Mass:", props.mass)
    print("CM:", props.center_of_mass)
    print("Inertia Tensor:\\n", props.inertia_tensor)
