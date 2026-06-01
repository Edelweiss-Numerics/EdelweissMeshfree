from edelweissmeshfree.cells.marmotcell.marmotcell import MarmotCellWrapper


# Mock for lagrangianmarmotcell.pyx
class LagrangianMarmotCellWrapper(MarmotCellWrapper):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("lagrangianmarmotcell not available in this environment")
