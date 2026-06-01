from edelweissmeshfree.cellelements.marmotcellelement import MarmotCellElementWrapper


# Mock for lagrangianmarmotcellelement.pyx
class LagrangianMarmotCellElementWrapper(MarmotCellElementWrapper):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("lagrangianmarmotcellelement not available in this environment")
