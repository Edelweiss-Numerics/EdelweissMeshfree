from edelweissfe.utils.exceptions import ModuleNotBuilt

from edelweissmeshfree.cells.marmotcell.marmotcell import MarmotCellWrapper


# Mock for bsplinemarmotcell.pyx
class BSplineMarmotCellWrapper(MarmotCellWrapper):
    def __init__(self, *args, **kwargs):
        raise ModuleNotBuilt("bsplinemarmotcell is not built in this environment")
