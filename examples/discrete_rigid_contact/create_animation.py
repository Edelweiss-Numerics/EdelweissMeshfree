import argparse
import os
import sys

import numpy as np
import pyvista as pv


def animate_case():
    parser = argparse.ArgumentParser(description="Interactive PyVista visualizer for Ensight Gold .case files.")
    parser.add_argument("case_file", nargs="?", help="Path to the Ensight Gold .case file to visualize.")
    args = parser.parse_args()

    case_file = args.case_file
    if not case_file:
        import glob

        case_files = glob.glob("*.case")
        if not case_files:
            print("Error: No .case file specified and none found in the current directory.")
            print("Usage: python create_animation.py [path_to_case_file.case]")
            sys.exit(1)
        case_file = sorted(case_files)[-1]
        print(f"No case file specified. Auto-detecting latest: {case_file}")

    if not os.path.exists(case_file):
        print(f"Error: Specified case file '{case_file}' does not exist.")
        sys.exit(1)

    print(f"Reading Ensight Gold case file: {case_file}...")
    reader = pv.get_reader(case_file)
    reader.set_active_time_set(1)
    time_values = reader.time_values
    print(f"Found {len(time_values)} time steps (t = {time_values[0]:.4f} to {time_values[-1]:.4f} s).")

    # ------------------------------------------------------------------ helpers
    def read_step(t_val):
        """Read one time step and return (rigid_surface, particles) blocks."""
        reader.set_active_time_value(t_val)
        mesh = reader.read()
        return mesh["rigid_body_surface"], mesh["PSET_mesh_particles_all"]

    def warp(block, vector_name):
        """Return a new mesh with points shifted by the named point-data vector."""
        pts = block.points + block.point_data[vector_name]
        out = block.copy()
        out.points = pts
        return out

    # ------------------------------------------------------------------ initial geometry
    pv.set_plot_theme("document")
    off_screen = "HEADLESS_TEST" in os.environ
    plotter = pv.Plotter(notebook=False, off_screen=off_screen, window_size=[1024, 768])

    rigid0, particles0 = read_step(time_values[0])
    warped_rigid0 = warp(rigid0, "rigid_displacement")
    warped_particles0 = warp(particles0, "vertex_displacements")

    vel_mag0 = np.linalg.norm(particles0.cell_data["velocity"], axis=1)
    warped_particles0.cell_data["velocity_magnitude"] = vel_mag0

    rigid_actor = plotter.add_mesh(
        warped_rigid0,
        color="#a0a0a0",
        opacity=0.5,
        show_edges=True,
        edge_color="#404040",
        line_width=1,
        ambient=0.3,
        diffuse=0.7,
        specular=0.2,
    )

    particle_actor = plotter.add_mesh(
        warped_particles0,
        scalars="velocity_magnitude",
        cmap="turbo",
        clim=[0, 10.0],
        scalar_bar_args={
            "title": "Velocity Magnitude (m/s)",
            "vertical": True,
            "position_x": 0.88,
            "position_y": 0.15,
            "width": 0.05,
            "height": 0.7,
        },
        ambient=0.4,
        diffuse=0.6,
    )

    # ------------------------------------------------------------------ camera / overlays
    plotter.camera_position = [
        (18.0, 10.0, 18.0),
        (0.0, 2.5, 0.0),
        (0.0, 1.0, 0.0),
    ]
    plotter.add_axes()
    plotter.show_grid()

    text_actor = plotter.add_text(
        f"Time: {time_values[0]:.4f} s",
        position="upper_left",
        font_size=14,
        color="black",
        shadow=True,
    )
    plotter.add_text(
        "Discrete Rigid Body Impact (MPM Simulation)",
        position="upper_edge",
        font_size=16,
        color="darkblue",
        shadow=True,
    )

    print("Saving animation to crash_animation.gif...")
    plotter.open_gif("crash_animation.gif")

    # ------------------------------------------------------------------ animation loop
    for i, time_val in enumerate(time_values):
        rigid, particles = read_step(time_val)

        # Rigid surface: shift reference points by rigid_displacement (per node)
        rigid_actor.mapper.dataset.points = rigid.points + rigid.point_data["rigid_displacement"]

        # Particle block: shift reference points by vertex_displacements (per node)
        particle_actor.mapper.dataset.points = particles.points + particles.point_data["vertex_displacements"]

        # Update velocity magnitude colour
        vel_mag = np.linalg.norm(particles.cell_data["velocity"], axis=1)
        particle_actor.mapper.dataset.cell_data["velocity_magnitude"] = vel_mag

        text_actor.set_text(0, f"Time: {time_val:.4f} s  (Step {i+1}/{len(time_values)})")

        plotter.write_frame()

    print("Animation completed.")
    plotter.close()


if __name__ == "__main__":
    animate_case()
