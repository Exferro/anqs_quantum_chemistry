from .hilbert_space import HilbertSpace
from .abstract_hilbert_space_object import AbstractHilbertSpaceObject
from .abstract_quantum_state import AbstractQuantumState
from .abstract_observable import AbstractObservable

from enum import Enum
MaskingMode = Enum('MaskingMode', 'logits part_base_vecs')
