from edelweissfe.utils.exceptions import ModuleNotBuilt


# Mock for utils.pyx
class BoundingBox:
    def __init__(self, *args, **kwargs):
        raise ModuleNotBuilt("utils is not built in this environment")


class BoundedCell:
    def __init__(self, *args, **kwargs):
        raise ModuleNotBuilt("utils is not built in this environment")


class _KDTreeImpl:
    def __init__(self, *args, **kwargs):
        raise ModuleNotBuilt("utils is not built in this environment")


class KDTree:
    def __init__(self, *args, **kwargs):
        raise ModuleNotBuilt("utils is not built in this environment")
