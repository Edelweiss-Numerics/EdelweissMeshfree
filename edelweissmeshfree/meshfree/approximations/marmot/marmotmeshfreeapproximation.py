from edelweissfe.utils.exceptions import ModuleNotBuilt


# Mock for marmotmeshfreeapproximation.pyx
class MarmotMeshfreeApproximationWrapper:
    def __init__(self, *args, **kwargs):
        raise ModuleNotBuilt("marmotmeshfreeapproximation is not built in this environment")
