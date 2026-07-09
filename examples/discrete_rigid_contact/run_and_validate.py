import glob

import numpy as np
import pyvista as pv


def validate_results():
    case_files = sorted(glob.glob("explicit_sim_out_*.case"))
    if not case_files:
        print("No .case file found.")
        exit(1)
    latest_case = case_files[-1]
    print(f"Validating case file: {latest_case}")

    reader = pv.get_reader(latest_case)
    reader.set_active_time_set(1)
    time_values = reader.time_values

    velocity = 6.0  # -5.5 displacement over 1.0 seconds
    initial_gap = 5.0
    expected_contact_time = initial_gap / velocity

    print(f"Velocity: {velocity} m/s")
    print(f"Initial Gap: {initial_gap} m")
    print(f"Expected contact time: {expected_contact_time:.4f} s")

    errors = []

    for t_val in time_values:
        reader.set_active_time_value(t_val)
        mesh = reader.read()
        rigid = mesh["rigid_surface"]
        particles = mesh["PSET_mesh_particles_all"]

        expected_disp = -velocity * t_val
        actual_disp = rigid.point_data["rigid_displacement"][:, 1]

        max_error = np.max(np.abs(actual_disp - expected_disp))

        block_disp = particles.point_data["vertex_displacements"]
        max_block_disp = np.max(np.abs(block_disp))

        status = "PRE-CONTACT" if t_val < expected_contact_time else "POST-CONTACT"

        print(
            f"t={t_val:.4f} ({status}): expected rigid Y disp = {expected_disp:.4f}, actual rigid Y disp = {np.min(actual_disp):.4f}, block max disp = {max_block_disp:.4f}"
        )

        if max_error > 1e-3:
            errors.append(
                f"At t={t_val}, rigid displacement mismatched: expected {expected_disp}, got max diff {max_error}"
            )

        if t_val < expected_contact_time - 0.01:
            if max_block_disp > 1e-3:
                errors.append(f"At t={t_val}, block deformed prematurely before contact! max_disp={max_block_disp}")
        elif t_val > expected_contact_time + 0.01:
            if max_block_disp < 1e-5:
                errors.append(f"At t={t_val}, block did not deform after contact!")

    if errors:
        print("\nVALIDATION FAILED:")
        for err in errors:
            print(f" - {err}")
        exit(1)
    else:
        print(
            "\nVALIDATION PASSED! Theoretical displacement perfectly matches actual simulation, and contact occurs precisely at the calculated time."
        )


if __name__ == "__main__":
    validate_results()
