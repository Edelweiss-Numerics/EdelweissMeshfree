from edelweissfe.utils.exceptions import ModuleNotBuilt

from edelweissmeshfree.cells.marmotcell.marmotcell import MarmotCellWrapper


# Mock for marmotcellelement.pyx
class MarmotCellElementWrapper(MarmotCellWrapper):
    def __init__(self, *args, **kwargs):
        raise ModuleNotBuilt("marmotcellelement is not built in this environment")
