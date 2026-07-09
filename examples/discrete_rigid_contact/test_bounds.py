import pyvista as pv

mesh = pv.read("rigid_body.exo").extract_surface()
print(mesh.bounds)
