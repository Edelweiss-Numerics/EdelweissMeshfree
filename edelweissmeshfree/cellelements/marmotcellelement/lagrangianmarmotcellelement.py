from edelweissfe.utils.exceptions import ModuleNotBuilt

from edelweissmeshfree.cellelements.marmotcellelement import MarmotCellElementWrapper


# Mock for lagrangianmarmotcellelement.pyx
class LagrangianMarmotCellElementWrapper(MarmotCellElementWrapper):
    def __init__(self, *args, **kwargs):
        raise ModuleNotBuilt("lagrangianmarmotcellelement is not built in this environment")
