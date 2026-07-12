import sys

import test_friction_validation

from edelweissmeshfree.solvers.explicitmultiphysicssolver import (
    ExplicitMultiphysicsSolver,
)

orig_solveStep = ExplicitMultiphysicsSolver.solveStep


def mock_solveStep(self, timeStepper, model, *args, **kwargs):
    print("Intercepting solveStep!")
    try:
        orig_solveStep(self, timeStepper, model, *args, **kwargs)
    except Exception:
        import traceback

        traceback.print_exc()
    sys.exit(0)


ExplicitMultiphysicsSolver.solveStep = mock_solveStep

test_friction_validation.run_friction_test()
