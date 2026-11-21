from edelweissfe.utils.exceptions import ModuleNotBuilt


# Mock for marmotcell.pyx
class MarmotCellWrapper:
    def __init__(self, *args, **kwargs):
        raise ModuleNotBuilt("marmotcell is not built in this environment")
