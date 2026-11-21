from edelweissfe.utils.exceptions import ModuleNotBuilt


# Mock for mp.pyx
class MarmotMaterialPointWrapper:
    def __init__(self, *args, **kwargs):
        raise ModuleNotBuilt("mp is not built in this environment")
