import torch as pt

from abc import abstractmethod

from ...base.abstract_hilbert_space_object import AbstractHilbertSpaceObject


class AbstractMasker(AbstractHilbertSpaceObject):
    def __init__(self,
                 *args,
                 **kwargs):
        super(AbstractMasker, self).__init__(*args, **kwargs)

    @abstractmethod
    def mask(self, base_vec: pt.Tensor) -> pt.Tensor:
        ...
