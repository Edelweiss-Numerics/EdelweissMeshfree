from edelweissfe.utils.exceptions import ModuleNotBuilt

from edelweissmeshfree.cells.marmotcell.marmotcell import MarmotCellWrapper


# Mock for lagrangianmarmotcell.pyx
class LagrangianMarmotCellWrapper(MarmotCellWrapper):
    def __init__(self, *args, **kwargs):
        raise ModuleNotBuilt("lagrangianmarmotcell is not built in this environment")
