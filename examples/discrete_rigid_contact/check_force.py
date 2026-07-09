import pyvista as pv
import numpy as np

reader = pv.get_reader("explicit_sim_out_2026-07-08-22-24-04.case")
reader.set_active_time_set(1)
reader.set_active_time_value(reader.time_values[-1])
mesh = reader.read()
particles = mesh["PSET_mesh_particles_all"]
print("Max vertex displacement:", np.max(np.abs(particles.point_data["vertex_displacements"])))
