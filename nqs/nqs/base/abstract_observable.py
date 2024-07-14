import torch as pt

from typing import Tuple

from .abstract_hilbert_space_object import AbstractHilbertSpaceObject
from abc import abstractmethod

from .abstract_quantum_state import AbstractQuantumState


class AbstractObservable(AbstractHilbertSpaceObject):
    def __init__(self, *args, **kwargs):
        super(AbstractObservable, self).__init__(*args, **kwargs)

    @abstractmethod
    def compute_local_energies(self,
                               wf: AbstractQuantumState = None,
                               sampled_indices: pt.Tensor = None,
                               sampled_amps: pt.Tensor = None,
                               verbose: bool = False) -> Tuple[pt.Tensor, pt.Tensor]:
        ...
