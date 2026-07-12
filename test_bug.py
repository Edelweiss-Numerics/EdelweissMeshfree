import numpy as np
from edelweissfe.elements.discreterigid import DiscreteRigidElement
from edelweissfe.points.node import Node

if __name__ == "__main__":
    node = Node(1, np.array([0, 0, 0]))
    el = DiscreteRigidElement(1, [node], None, "tria3")
    fields_on_ent = el.fields
    for iNode, n in enumerate(el.nodes):
        print(fields_on_ent[iNode])
