from abc import abstractmethod

import numpy as np
import torch as pt

from typing import Tuple, Union

from ....base.constants import NEGINF
from ....base.constants import BASE_REAL_TYPE, BASE_COMPLEX_TYPE

from ....base import AbstractQuantumState
from ....base.qubit_grouping import QubitGroupingConfig, QubitGrouping
from ....infrastructure.nested_data import Config

from .mlp import MLPConfig
from ...maskers import LocallyDecomposableMasker

LOCAL_SAMPLING_STRATEGIES = ('DU', 'MU',)


class LocalSamplingConfig(Config):
    ALLOWED_LOCAL_SAMPLING_PATTERNS = ('uniform',)
    FIELDS = (
        'pattern_type',
        'strategy',
        'masking_depth',
    )

    def __init__(self,
                 *args,
                 pattern_type: str = 'uniform',
                 strategy: str = 'MU',
                 masking_depth: int = 0,
                 **kwargs):
        self.pattern_type = pattern_type
        self.strategy = strategy
        self.masking_depth = masking_depth

        super().__init__(*args, **kwargs)

    def create_local_sampling_pattern(self, qudit_num: int = None):
        assert self.pattern_type in self.ALLOWED_LOCAL_SAMPLING_PATTERNS
        if self.pattern_type == 'uniform':
            assert self.strategy in LOCAL_SAMPLING_STRATEGIES
            local_sampling_pattern = (self.strategy, ) * (qudit_num - self.masking_depth)
            local_sampling_pattern = local_sampling_pattern + ('DU', ) * self.masking_depth
        else:
            raise RuntimeError(f'Wrong local sampling pattern requested: {self.pattern_type}')

        return local_sampling_pattern


class SpinFlipSymmetryConfig(Config):
    FIELDS = (
        'abs',
        'phase',
    )

    def __init__(self,
                 *args,
                 abs: bool = False,
                 phase: bool = False,
                 **kwargs):
        self.abs = abs
        self.phase = phase

        super().__init__(*args, **kwargs)


class ANQSConfig(Config):
    ALLOWED_DE_MODES = ('MADE', 'NADE',)
    FIELDS = (
        'dtype',
        'de_mode',
        'qubit_grouping_config',
        'local_sampling_config',
        'spin_flip_symmetry_config',
        'subtract_mean',
        'main_subnet_config',
        'aux_subnet_config',
        'use_sign_structure',
    )
    NON_JSONABLE_FIELDS = (
        'dtype',
    )

    def __init__(self,
                 *args,
                 dtype=BASE_REAL_TYPE,
                 de_mode: str = 'NADE',
                 qubit_grouping_config: QubitGroupingConfig = None,
                 local_sampling_config: LocalSamplingConfig = None,
                 spin_flip_symmetry_config: SpinFlipSymmetryConfig = None,
                 subtract_mean: bool = True,
                 main_subnet_config: MLPConfig = None,
                 aux_subnet_config: MLPConfig = None,
                 use_sign_structure: bool = False,
                 **kwargs):
        self.dtype = dtype
        self.de_mode = de_mode
        self.qubit_grouping_config = qubit_grouping_config if qubit_grouping_config is not None else QubitGroupingConfig()
        self.local_sampling_config = local_sampling_config if local_sampling_config is not None else LocalSamplingConfig()
        self.spin_flip_symmetry_config = spin_flip_symmetry_config if spin_flip_symmetry_config is not None else SpinFlipSymmetryConfig()
        self.subtract_mean = subtract_mean
        self.main_subnet_config = main_subnet_config if main_subnet_config is not None else MLPConfig()
        self.aux_subnet_config = aux_subnet_config if aux_subnet_config is not None else MLPConfig()
        self.use_sign_structure = use_sign_structure

        super().__init__(*args, **kwargs)


class AbstractANQS(AbstractQuantumState):
    def __init__(self,
                 *args,
                 config: ANQSConfig = None,
                 masker: LocallyDecomposableMasker = None,
                 sign_structure: pt.Tensor = None,
                 **kwargs):
        super(AbstractANQS, self).__init__(*args, **kwargs)
        #self.width = 512

        self.config = config if config is not None else ANQSConfig()
        assert self.config.dtype in (BASE_REAL_TYPE, BASE_COMPLEX_TYPE)
        self.dtype = self.config.dtype

        self.qubit_grouping_config = self.config.qubit_grouping_config
        self.qubit_grouping = QubitGrouping.create(hs=self.hilbert_space,
                                                   config=self.qubit_grouping_config,
                                                   masker=masker)
        self.max_qudit_dim = max(self.qudit_dims)

        assert config.de_mode in ANQSConfig.ALLOWED_DE_MODES
        self.de_mode = self.config.de_mode

        self.masker = masker

        self.local_sampling_config = self.config.local_sampling_config
        self.local_sampling_pattern = self.local_sampling_config.create_local_sampling_pattern(qudit_num=self.qudit_num)

        self.made_qudit_dim_mask = pt.less(pt.tile(pt.arange(self.max_qudit_dim,
                                                             device=self.device,
                                                             dtype=self.idx_dtype),
                                                   (self.qudit_num, 1)),
                                           pt.unsqueeze(self.qudit_dims, dim=-1))

        if self.config.use_sign_structure:
            assert sign_structure is not None
            assert sign_structure.shape[0] == 2**self.qubit_num
            assert self.qubit_num < 32
            self.sign_structure = sign_structure

        # # Orbitals processing
        self.spin_flip_symmetry_config = self.config.spin_flip_symmetry_config
        self._param_num = None
        self._param_shapes = None
        self._cat_grad_splits = None
        self._name_idx2name = None
        self._name_idx2dtype = None
        #
        # if self.spin_flip_symmetry or self.spin_flip_phase_symmetry:
        #     self.qubit_per_orbit = 2
        #     assert (self.qubit_num % 2) == 0
        #     self.orbit_num = self.qubit_num // self.qubit_per_orbit
        #
        #     self.orbit_starts = tuple(orbit_idx * self.qubit_per_orbit for orbit_idx in range(self.orbit_num))
        #     self.orbit_ends = self.orbit_starts[1:] + (self.qubit_num, )
        #     self.orbit_starts = pt.tensor(self.orbit_starts, device=self.device)
        #     self.orbit_ends = pt.tensor(self.orbit_ends, device=self.device)
        #
        #     self.orbit_dims = tuple(2 ** (self.orbit_ends[orbit_idx] - self.orbit_starts[orbit_idx])
        #                             for orbit_idx in range(self.orbit_num))
        #     self.orbit_dims = pt.tensor(self.orbit_dims, device=self.device)
        #     self.qubit_idx2orbit_two_power = ()
        #     self.qubit_idx2orbit_idx = ()
        #     for orbit_idx in range(self.orbit_num):
        #         self.qubit_idx2orbit_two_power += tuple(2**qubit_idx
        #                                                 for qubit_idx in range(self.orbit_ends[orbit_idx]
        #                                                                        - self.orbit_starts[orbit_idx]))
        #         self.qubit_idx2orbit_idx += tuple([orbit_idx] * (self.orbit_ends[orbit_idx] - self.orbit_starts[orbit_idx]))
        #
        #     self.qubit_idx2orbit_two_power = pt.tensor(self.qubit_idx2orbit_two_power, device=self.device)
        #     self.qubit_idx2orbit_idx = pt.tensor(self.qubit_idx2orbit_idx, device=self.device)
        #
        #     self.sf_qudit_couplings = []
        #     for qudit_idx in range(self.qudit_num):
        #         self.sf_qudit_couplings.append(self.qubits2single_qudit(self.spin_flip_base_vec(self.qudit_continuations[qudit_idx]),
        #                                                                 qubit_per_qudit=self.qudit_ends[qudit_idx]
        #                                                                                 - self.qudit_starts[qudit_idx]))
        #
        #     self.sf_qudit_couplings = tuple(self.sf_qudit_couplings)

    @property
    def qudit_num(self):
        return self.qubit_grouping.qudit_num

    @property
    def qudit_dims(self):
        return self.qubit_grouping.qudit_dims

    @property
    def qudit_starts(self):
        return self.qubit_grouping.qudit_starts

    @property
    def qudit_ends(self):
        return self.qubit_grouping.qudit_ends

    @property
    def name_idx2name(self):
        if self._name_idx2name is None:
            self._name_idx2name = {name_idx: name for name_idx, name in enumerate(dict(self.named_parameters()).keys())}

        return self._name_idx2name

    @property
    def name_idx2dtype(self):
        if self._name_idx2dtype is None:
            self._name_idx2dtype = {name_idx: param.dtype for name_idx, (name, param) in enumerate(dict(self.named_parameters()).items())}

        return self._name_idx2dtype

    @property
    def param_num(self):
        if self._param_num is None:
            self._param_num = 0
            for param in self.parameters():
                self._param_num += param.numel()

        return self._param_num

    @property
    def param_shapes(self):
        if self._param_shapes is None:
            self._param_shapes = []
            for param in self.parameters():
                self._param_shapes.append(param.shape)
            self._param_shapes = tuple(self._param_shapes)

        return self._param_shapes

    @property
    def cat_grad_splits(self):
        if self._cat_grad_splits is None:
            self._cat_grad_splits = []
            for param in self.parameters():
                if param.dtype == BASE_REAL_TYPE:
                    self._cat_grad_splits.append(param.numel())
                elif param.dtype == BASE_COMPLEX_TYPE:
                    self._cat_grad_splits.append(2 * param.numel())
                else:
                    raise RuntimeError(f'Wrong param dtype: {param.dtype}')
            self._cat_grad_splits = tuple(self._cat_grad_splits)
        return self._cat_grad_splits

    @property
    def cat_grad(self):
        cat_grad = []
        for param in self.parameters():
            if param.grad is not None:
                grad_tensor = param.grad.data
            else:
                grad_tensor = pt.zeros_like(param)
            if param.dtype == BASE_REAL_TYPE:
                cat_grad.append(pt.reshape(grad_tensor, (-1, )))
            elif param.dtype == BASE_COMPLEX_TYPE:
                cat_grad.append(pt.reshape(pt.view_as_real(grad_tensor), (-1, )))
            else:
                raise RuntimeError(f'Wrong param dtype: {param.dtype}')

        return pt.cat(cat_grad)

    @cat_grad.setter
    def cat_grad(self, cat_grad: pt.Tensor):
        cat_grad = pt.split(cat_grad, self.cat_grad_splits)
        for param_idx, param in enumerate(self.parameters()):
            if param.dtype == BASE_REAL_TYPE:
                grad_tensor = pt.reshape(cat_grad[param_idx],
                                         self.param_shapes[param_idx])
            elif param.dtype == BASE_COMPLEX_TYPE:
                grad_tensor = pt.reshape(cat_grad[param_idx],
                                         (*self.param_shapes[param_idx], 2))
                grad_tensor = pt.view_as_complex(grad_tensor.contiguous())
            else:
                raise RuntimeError(f'Wrong param dtype: {param.dtype}')

            param.grad = grad_tensor

    @abstractmethod
    def clip_grad_norm(self, value: float = None):
        ...

    @abstractmethod
    def _cond_log_psi(self,
                      qudit_idx: Union[int, None] = None,
                      base_vec: pt.Tensor = None) -> pt.Tensor:
        """
        The core function of any ANQS: it prescribes how the anqs vector of {0, 1}^n (where n is the number of qubits
        considered so far) is transformed into a conditional log_psi = \log \psi(x_{n+1}|x_1 x_2 ... x_n). In case of
        NADE we expect the output to be of shape (C, B, 2**qubits_per_group), while for MADE we expect it to be
        of shape (C, B, self.qubit_num, 2**qubits_per_group),
        where C is the number of channels (heads) and B is the batch size. If self.head_num == 1, we don't expect C to
        be in the final shape.

        If NADE, qubit_idx should indicate which qubit we are computing the conditional probability for; if MADE,
         qubit_idx should be None.
        """
        ...

    def cond_log_psi(self,
                     qudit_idx: int = None,
                     base_vec: pt.Tensor = None,
                     return_all_if_made: bool = False,
                     mask: pt.Tensor = None) -> pt.Tensor:
        if self.de_mode == 'NADE':
            assert qudit_idx is not None
            cond_log_psi = self._cond_log_psi(qudit_idx=qudit_idx, base_vec=base_vec)
            if self.config.subtract_mean:
                cond_log_psi = pt.complex(cond_log_psi.real - pt.mean(cond_log_psi.real, dim=-1, keepdim=True),
                                          cond_log_psi.imag)
            if self.spin_flip_symmetry_config.abs:
                sf_base_vec = self.spin_flip_base_vec(pt.clone(base_vec))
                sf_cond_log_psi = self._cond_log_psi(qudit_idx=qudit_idx, base_vec=sf_base_vec)
                if self.config.subtract_mean:
                    sf_cond_log_psi = sf_cond_log_psi - pt.mean(sf_cond_log_psi, dim=-1, keepdim=True)
                cond_log_psi = pt.complex(0.5 * (cond_log_psi.real + sf_cond_log_psi[..., self.sf_qudit_couplings[qudit_idx]].real),
                                          cond_log_psi.imag)

            assert mask is not None
            cond_log_psi = pt.where(mask,
                                    cond_log_psi,
                                    pt.full(cond_log_psi.shape, NEGINF, dtype=cond_log_psi.dtype,
                                            device=cond_log_psi.device))

            cond_log_psi = self.normalise_cond_log(cond_log_psi)

        elif self.de_mode == 'MADE':
            cond_log_psi = self._cond_log_psi(qudit_idx=None, base_vec=base_vec)
            if self.config.subtract_mean:
                cond_log_psi = pt.complex(cond_log_psi.real - pt.mean(cond_log_psi.real, dim=-1, keepdim=True),
                                  cond_log_psi.imag)
            # if self.spin_flip_symmetry_config.abs:
            #     sf_base_vec = self.spin_flip_base_vec(pt.clone(base_vec))
            #     sf_cond_log_psi = self._cond_log_psi(qudit_idx=None, base_vec=sf_base_vec)
            #     if self.config.subtract_mean:
            #         sf_cond_log_psi = pt.complex(sf_cond_log_psi.real - pt.mean(sf_cond_log_psi.real, dim=-1, keepdim=True),
            #                                      sf_cond_log_psi.imag)
            #     cond_log_psi = pt.complex(0.5 * (cond_log_psi.real + sf_cond_log_psi[..., self.sf_qudit_couplings[0]].real),
            #                               cond_log_psi.imag)
            # cond_log_psi = pt.where(self.made_qudit_dim_mask,
            #                         cond_log_psi,
            #                         pt.full(cond_log_psi.shape, NEGINF, dtype=cond_log_psi.dtype,
            #                                 device=cond_log_psi.device))
            if return_all_if_made:
                cond_log_psi = cond_log_psi[..., :qudit_idx + 1, :]
            else:
                cond_log_psi = cond_log_psi[..., qudit_idx, :]

            assert mask is not None
            cond_log_psi = pt.where(mask,
                                    cond_log_psi,
                                    pt.full(cond_log_psi.shape, NEGINF, dtype=cond_log_psi.dtype,
                                            device=cond_log_psi.device))

            cond_log_psi = self.normalise_cond_log(cond_log_psi)
        else:
            raise RuntimeError(f'Somehow we have wrong de_mode: {self.de_mode}')

        return cond_log_psi

    def cond_log_abs(self,
                     qudit_idx: int = None,
                     base_vec: pt.Tensor = None,
                     return_all_if_made: bool = False,
                     mask: pt.Tensor = None) -> pt.Tensor:

        return self.cond_log_psi(qudit_idx=qudit_idx,
                                 base_vec=base_vec,
                                 return_all_if_made=return_all_if_made,
                                 mask=mask).real

    def cond_phase(self,
                   qudit_idx: int = None,
                   base_vec: pt.Tensor = None,
                   return_all_if_made: bool = False,
                   mask: pt.Tensor = None) -> pt.Tensor:

        return self.cond_log_psi(qudit_idx=qudit_idx,
                                 base_vec=base_vec,
                                 return_all_if_made=return_all_if_made,
                                 mask=mask).imag

    def normalise_cond_log(self, cond_log: pt.Tensor) -> pt.Tensor:
        if cond_log.dtype == BASE_COMPLEX_TYPE:
            # We presume we are normalising complex log amplitudes
            cond_log = cond_log - 0.5 * pt.complex(pt.logsumexp(2 * cond_log.real, dim=-1, keepdim=True),
                                                   pt.zeros_like(cond_log.imag))
            cond_log = pt.complex(
                pt.nan_to_num(cond_log.real, nan=NEGINF.type(self.rdtype), neginf=NEGINF.type(self.rdtype)),
                pt.nan_to_num(cond_log.imag, nan=0.0, neginf=NEGINF.type(self.rdtype)))
        else:
            # We presume we are normalising real log absolute values
            cond_log = cond_log - 0.5 * pt.logsumexp(2 * cond_log, dim=-1, keepdim=True)
            cond_log = pt.nan_to_num(cond_log, nan=NEGINF.type(self.rdtype), neginf=NEGINF.type(self.rdtype))

        return cond_log

    def log_psi(self, base_vec: pt.Tensor, just_return: bool = False) -> pt.Tensor:
        qudit_continuation_masks = self.qubit_grouping.base_vec2qudit_continuation_masks(base_vec)
        qudits = self.qubit_grouping.base_vec2qudit_base_vec(base_vec)
        if self.de_mode == 'NADE':
            log_psi = pt.zeros(*base_vec.shape[:-1],
                               dtype=BASE_COMPLEX_TYPE,
                               device=self.device)
            for qudit_idx in range(self.qudit_num):
                mask = pt.reshape(qudit_continuation_masks[qudit_idx],
                                  (-1, self.qudit_dims[qudit_idx]))
                if self.local_sampling_pattern[qudit_idx] == 'DU':
                    mask = pt.ones_like(mask)
                cond_log_psi = self.cond_log_psi(qudit_idx=qudit_idx,
                                                 base_vec=base_vec[..., :self.qudit_starts[qudit_idx]],
                                                 mask=mask)
                log_psi = log_psi + pt.squeeze(pt.gather(cond_log_psi,
                                                         dim=1,
                                                         index=qudits[:, qudit_idx:qudit_idx + 1]),
                                               dim=-1)
                log_psi = pt.complex(
                    pt.nan_to_num(log_psi.real, nan=NEGINF.type(self.rdtype), neginf=NEGINF.type(self.rdtype)),
                    pt.nan_to_num(log_psi.imag, nan=NEGINF.type(self.rdtype), neginf=NEGINF.type(self.rdtype)))

        elif self.de_mode == 'MADE':
            masks = []
            for qudit_idx in range(self.qudit_num):
                masks.append(pt.reshape(qudit_continuation_masks[qudit_idx],
                                        (-1, self.qudit_dims[qudit_idx])))
                masks[qudit_idx] = pt.cat((masks[qudit_idx],
                                           pt.zeros((*masks[qudit_idx].shape[:-1],
                                                     self.max_qudit_dim - masks[qudit_idx].shape[-1]),
                                                    device=self.device,
                                                    dtype=pt.bool)),
                                          dim=-1)
                if self.local_sampling_pattern[qudit_idx] == 'DU':
                    masks[qudit_idx] = pt.ones_like(masks[qudit_idx])
            mask = pt.stack(masks, dim=1)

            cond_log_psi = self.cond_log_psi(qudit_idx=self.qudit_num - 1,
                                             base_vec=base_vec,
                                             return_all_if_made=True,
                                             mask=mask)
            
            log_psi = pt.sum(pt.squeeze(pt.gather(cond_log_psi,
                                               -1,
                                               pt.unsqueeze(qudits, dim=-1)),
                                     dim=-1),
                          dim=-1)
        else:
            raise RuntimeError(f'Wrong DE mode: {self.de_mode}')
        if just_return:
            if self.config.use_sign_structure:
                print(f'We are using sign structure')
                base_idx = pt.squeeze(self.base_vec2base_idx(base_vec), dim=-1)
                log_psi = pt.complex(log_psi.real,
                                     self.sign_structure[base_idx])
            print(self.config.use_sign_structure)
            return log_psi
        else:
            if self.spin_flip_symmetry_config.phase:
                sf_base_vec = self.spin_flip_base_vec(pt.clone(base_vec))
                sf_cano_repr = self.qubits2sf_cano_repr(pt.clone(base_vec))
                sf_log_psi = self.log_psi(base_vec=sf_base_vec, just_return=True)
                pi_mult = (base_vec - sf_cano_repr).abs().sum(dim=-1) // 4 
                phases = pt.abs(pt.tensor(pt.pi, dtype=self.rdtype) * (pi_mult % 2))
                log_psi = pt.complex(log_psi.real,
                                     0.5 * (log_psi.imag + sf_log_psi.imag) + phases)                
                return log_psi
            else:
                if self.config.use_sign_structure:
                    base_idx = pt.squeeze(self.base_vec2base_idx(base_vec), dim=-1)

                    log_psi = pt.complex(log_psi.real,
                                         self.sign_structure[base_idx])
                return log_psi

    def amplitude(self, base_idx: pt.Tensor):
        base_vec = self.base_idx2base_vec(base_idx).to(self.device)
        return pt.exp(self.log_psi(base_vec))

    def phase(self, base_idx: pt.Tensor):
        base_vec = self.base_idx2base_vec(base_idx).to(self.device)
        return self.log_psi(base_vec).imag

    def forward(self, base_idx: pt.Tensor):
        return self.amplitude(base_idx)

    @pt.no_grad()
    def sample_stats(self,
                     sample_num: int) -> Tuple[pt.Tensor, pt.Tensor]:
        unq_base_vecs = pt.zeros((1, 0),
                                 dtype=self.idx_dtype,
                                 device=self.device)
        unq_counts = pt.tensor([sample_num],
                               dtype=self.idx_dtype,
                               device=self.device)
        unq_acc_eigs = []
        for sym_idx in range(self.masker.sym_num):
            unq_acc_eigs.append(pt.zeros(unq_base_vecs.shape[:-1],
                                         dtype=self.idx_dtype,
                                         device=self.device) + self.masker.symmetries[sym_idx].start_eig)
        unq_acc_eigs = pt.stack(unq_acc_eigs, dim=-1)
        unq_memo_idx = self.masker.acc_eigs2memo_idx(unq_acc_eigs)
        for qudit_idx in range(0, self.qudit_num):
            unq_base_vecs, unq_counts, unq_memo_idx = self.sample_next_qudit_stats(qudit_idx=qudit_idx,
                                                                                   unq_base_vecs=unq_base_vecs,
                                                                                   unq_counts=unq_counts,
                                                                                   unq_memo_idx=unq_memo_idx,
                                                                                   rng=self.rng)
            phys_mask = self.masker.memo[unq_base_vecs.shape[-1], unq_memo_idx]
            # print(f'We have {phys_mask.sum()} physical samples out of {phys_mask.shape[0]}')
            unq_base_vecs = unq_base_vecs[phys_mask]
            unq_counts = unq_counts[phys_mask]
            unq_memo_idx = unq_memo_idx[phys_mask]

        unq_indices = self.base_vec2base_idx(unq_base_vecs)
        unq_counts = unq_counts.type(self.cdtype).to(self.device)

        return unq_indices, unq_counts

    @staticmethod
    def sample_mult_new(logits, sample_nums, qubit_num):
        sampled_counts = sample_nums
        cur_probs = pt.softmax(2 * logits, dim=-1)
        cur_probs = pt.nan_to_num(cur_probs, nan=0.0, neginf=0.0)
        indices = pt.unsqueeze(pt.arange(logits.shape[-1]), dim=0)

        for qubit_idx in range(qubit_num):
            cur_length = sampled_counts.shape[0]
            cur_distr_size = cur_probs.shape[-1]

            indices = pt.cat((indices[..., :cur_distr_size // 2],
                              indices[..., cur_distr_size // 2:]),
                             dim=0)

            success_prob = pt.sum(cur_probs[..., :cur_distr_size // 2], dim=-1)
            failure_prob = pt.sum(cur_probs[..., cur_distr_size // 2:], dim=-1)

            bernoulli_probs = success_prob / (success_prob + failure_prob)
            bernoulli_probs = pt.nan_to_num(bernoulli_probs, nan=0.0, neginf=0.0)

            cur_counts = pt.distributions.Binomial(sampled_counts, bernoulli_probs).sample()
            sampled_counts = pt.cat((cur_counts, sampled_counts - cur_counts))
            cur_probs = pt.cat((cur_probs[..., :cur_distr_size // 2],
                                cur_probs[..., cur_distr_size // 2:]),
                               dim=0)
        indices = pt.reshape(indices, (-1,))

        return pt.transpose(sampled_counts.reshape((-1, sample_nums.shape[0])), 0, 1)[..., indices].reshape((-1, ))

    @staticmethod
    def sample_mult_new_new(logits, sample_nums, qubit_num):
        sampled_counts = sample_nums
        cur_probs = pt.softmax(2 * logits, dim=-1)
        cur_probs = pt.nan_to_num(cur_probs, nan=0.0, neginf=0.0)
        cur_cumsum_probs = pt.cumsum(cur_probs, dim=-1)
        cur_cumsum_probs = pt.cat((pt.zeros_like(cur_cumsum_probs[:, 0:1]),
                                   cur_cumsum_probs),
                                  dim=-1)
        indices = pt.unsqueeze(pt.arange(logits.shape[-1]), dim=0)

        for qubit_idx in range(qubit_num):
            cur_distr_size = cur_cumsum_probs.shape[-1]
            half_idx = cur_distr_size // 2

            indices = pt.cat((indices[..., :(cur_distr_size - 1) // 2],
                              indices[..., (cur_distr_size - 1) // 2:]),
                             dim=0)

            success_prob = cur_cumsum_probs[..., half_idx] - cur_cumsum_probs[..., 0]
            failure_prob = cur_cumsum_probs[..., cur_distr_size - 1] - cur_cumsum_probs[..., half_idx]
            bernoulli_probs = success_prob / (success_prob + failure_prob)
            bernoulli_probs = pt.nan_to_num(bernoulli_probs, nan=0.0, neginf=0.0)

            cur_counts = pt.distributions.Binomial(sampled_counts, bernoulli_probs).sample()
            sampled_counts = pt.cat((cur_counts, sampled_counts - cur_counts))
            cur_cumsum_probs = pt.cat((cur_cumsum_probs[..., :half_idx + 1],
                                       cur_cumsum_probs[..., half_idx:]),
                                      dim=0)

        indices = pt.reshape(indices, (-1,))

        result = pt.transpose(sampled_counts.reshape((-1, sample_nums.shape[0])), 0, 1)[..., indices].reshape((-1, ))

        return result

    def sample_next_qudit_stats(self,
                                qudit_idx: int = None,
                                unq_base_vecs: pt.Tensor = None,
                                unq_counts: pt.Tensor = None,
                                unq_memo_idx: pt.Tensor = None,
                                rng: np.random.Generator = None) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor]:
        new_unq_memo_idx = self.qubit_grouping.qudit_idx2memo_idx_mul_table[qudit_idx][unq_memo_idx]
        new_unq_memo_idx = pt.reshape(new_unq_memo_idx, (-1,))
        mask = self.qubit_grouping.qudit_idx2cont_mask_mul_table[qudit_idx][unq_memo_idx]
        reshaped_mask = mask
        mask = pt.reshape(mask, (-1,))
        # reshaped_mask = pt.reshape(mask, (-1, self.qudit_dims[qudit_idx]))
        if self.local_sampling_pattern[qudit_idx] == 'DU':
            reshaped_mask = pt.ones_like(reshaped_mask)

        if self.de_mode == 'MADE':
            interm_mask = pt.zeros((*reshaped_mask.shape[:-1],
                                    self.max_qudit_dim),
                                   device=self.device,
                                   dtype=pt.bool)
            interm_mask[..., :reshaped_mask.shape[-1]] = reshaped_mask
            reshaped_mask = interm_mask
        cond_log_abs = self.cond_log_abs(qudit_idx=qudit_idx,
                                         base_vec=unq_base_vecs,
                                         return_all_if_made=False,
                                         mask=reshaped_mask)
        if self.de_mode == 'MADE':
            cond_log_abs = cond_log_abs[..., :self.qudit_dims[qudit_idx]]

        sampled_out_counts = self.sample_mult_new_new(logits=cond_log_abs,
                                                  sample_nums=unq_counts,
                                                  qubit_num=self.qubit_grouping.qubits_per_qudit[qudit_idx])
        # Binomial sampling
        # for out_idx in range(self.qudit_dims[qudit_idx] - 1):
        #     probs = pt.exp(2 * cond_log_abs[..., out_idx:])
        #     probs = probs / pt.sum(probs, dim=-1, keepdim=True)
        #     probs = pt.nan_to_num(probs, nan=0.0, neginf=0.0)
        #     # cur_out_counts = rng.binomial(unq_counts.detach().cpu().numpy(),
        #     #                               probs[..., 0].detach().cpu())
        #     cur_out_counts = pt.distributions.Binomial(unq_counts, probs[..., 0]).sample()
        #     #cur_out_counts = pt.tensor(cur_out_counts, dtype=unq_counts.dtype, device=unq_counts.device)
        #     sampled_out_counts.append(cur_out_counts)
        #     unq_counts = unq_counts - cur_out_counts
        # sampled_out_counts.append(unq_counts)
        # print(sampled_out_counts.shape)
        # sampled_out_counts = pt.cat([pt.unsqueeze(a, dim=-1) for a in sampled_out_counts], dim=-1)
        # print(sampled_out_counts.shape)


        #sampled_out_counts = pt.reshape(sampled_out_counts, (-1,))
        #print(sampled_out_counts.shape)

        continuations = pt.tile(self.qubit_grouping.qudit_idx2local_base_vecs[qudit_idx], (unq_base_vecs.shape[0], 1))
        unq_base_vecs = pt.unsqueeze(unq_base_vecs, dim=1)
        unq_base_vecs = pt.tile(unq_base_vecs, (1, self.qudit_dims[qudit_idx], 1))
        unq_base_vecs = pt.reshape(unq_base_vecs, (self.qudit_dims[qudit_idx] * unq_base_vecs.shape[0], -1))

        unq_base_vecs = pt.cat((unq_base_vecs, continuations), dim=-1)
        unq_counts = sampled_out_counts

        unq_base_vecs = unq_base_vecs[mask]
        unq_counts = unq_counts[mask]
        unq_memo_idx = new_unq_memo_idx[mask]

        non_zero_count_mask = (unq_counts > 0)
        unq_base_vecs = unq_base_vecs[non_zero_count_mask]
        unq_counts = unq_counts[non_zero_count_mask]
        unq_memo_idx = unq_memo_idx[non_zero_count_mask]

        return unq_base_vecs, unq_counts, unq_memo_idx

    @staticmethod
    def log1mexp(x):
        return pt.where(pt.greater(x, -0.693),
                        pt.log(-pt.expm1(x)),
                        pt.log1p(-pt.exp(x)))

    @staticmethod
    def log1pexp(x):
        return pt.where(pt.less(x, 18.),
                        pt.log1p(pt.exp(x)),
                        x + pt.exp(-x))

    def sample_gumbels_given_max(self, centres, maxes):
        # gumbels = pt.reshape(pt.from_numpy(self.rng.gumbel(pt.reshape(centres,
        #                                                                (-1,)).cpu().numpy())).to(centres.device),
        #                      centres.shape)
        gumbels = centres - pt.log(-pt.log(pt.rand(centres.shape, device=centres.device, dtype=centres.dtype)))
        observed_maxes, _ = pt.max(gumbels, dim=-1)
        v_variables = pt.unsqueeze(maxes, dim=-1) - gumbels + AbstractANQS.log1mexp(
            gumbels - pt.unsqueeze(observed_maxes, dim=-1))
        cond_gumbels = (pt.unsqueeze(maxes, dim=-1)
                        - pt.maximum(v_variables, pt.zeros_like(v_variables))
                        - AbstractANQS.log1pexp(-pt.abs(v_variables)))

        return cond_gumbels

    @pt.no_grad()
    def sample_next_qudit_indices_gumbel(self,
                                         sample_num: int = None,
                                         qudit_idx: int = None,
                                         unq_base_vecs: pt.Tensor = None,
                                         unq_log_probs: pt.Tensor = None,
                                         unq_gumbels: pt.Tensor = None,
                                         unq_memo_idx: pt.Tensor = None) -> Tuple[
        pt.Tensor, pt.Tensor, pt.Tensor, pt.Tensor]:
        # new_unq_acc_eigs, mask = self.qubit_grouping.qudit_acc_eigs2continuation_mask(qudit_idx=qudit_idx,
        #                                                                               qudit_acc_eigs=unq_acc_eigs)
        # print(mask.shape)
        new_unq_memo_idx = self.qubit_grouping.qudit_idx2memo_idx_mul_table[qudit_idx][unq_memo_idx]
        new_unq_memo_idx = pt.reshape(new_unq_memo_idx, (-1,))
        mask = self.qubit_grouping.qudit_idx2cont_mask_mul_table[qudit_idx][unq_memo_idx]
        reshaped_mask = mask
        mask = pt.reshape(mask, (-1,))
        #reshaped_mask = pt.reshape(mask, (-1, self.qudit_dims[qudit_idx]))
        if self.local_sampling_pattern[qudit_idx] == 'DU':
            reshaped_mask = pt.ones_like(reshaped_mask)

        if self.de_mode == 'MADE':
            interm_mask = pt.zeros((*reshaped_mask.shape[:-1],
                                    self.max_qudit_dim),
                                   device=self.device,
                                   dtype=pt.bool)
            interm_mask[..., :reshaped_mask.shape[-1]] = reshaped_mask
            reshaped_mask = interm_mask
        cond_log_abs = self.cond_log_abs(qudit_idx=qudit_idx,
                                         base_vec=unq_base_vecs,
                                         return_all_if_made=False,
                                         mask=reshaped_mask)
        if self.de_mode == 'MADE':
            cond_log_abs = cond_log_abs[..., :self.qudit_dims[qudit_idx]]
        unq_log_probs = pt.unsqueeze(unq_log_probs, dim=-1)
        unq_log_probs = unq_log_probs + 2 * cond_log_abs
        unq_log_probs = pt.nan_to_num(unq_log_probs, nan=NEGINF.type(self.rdtype), neginf=NEGINF.type(self.rdtype))

        new_unq_gumbels = self.sample_gumbels_given_max(unq_log_probs, unq_gumbels)
        new_unq_gumbels = pt.reshape(new_unq_gumbels, (-1,))
        new_unq_log_probs = pt.reshape(unq_log_probs, (-1,))
        new_unq_gumbels = pt.nan_to_num(new_unq_gumbels, nan=NEGINF.type(self.rdtype), neginf=NEGINF.type(self.rdtype))

        sorted_gumbels, sorted_indices = pt.sort(new_unq_gumbels, descending=True)

        new_unq_log_probs = new_unq_log_probs[sorted_indices[:sample_num]]
        new_unq_gumbels = new_unq_gumbels[sorted_indices[:sample_num]]
        new_unq_memo_idx = new_unq_memo_idx[sorted_indices[:sample_num]]

        init_indices = pt.arange(unq_base_vecs.shape[0], device=self.device)
        init_indices = pt.repeat_interleave(init_indices, self.qudit_dims[qudit_idx], dim=0)
        cont_indices = pt.arange(self.qudit_dims[qudit_idx], device=self.device)
        cont_indices = pt.tile(cont_indices, (unq_base_vecs.shape[0], ))

        new_unq_base_vecs_starts = unq_base_vecs[init_indices[sorted_indices[:sample_num]]]
        new_unq_base_vecs_ends = self.qubit_grouping.qudit_idx2local_base_vecs[qudit_idx][cont_indices[sorted_indices[:sample_num]]]
        new_unq_base_vecs = pt.cat((new_unq_base_vecs_starts, new_unq_base_vecs_ends), dim=-1)
        # print(zero_outcome)
        # print(one_outcome)
        # continuations = pt.tile(self.qubit_grouping.qudit_idx2local_base_vecs[qudit_idx], (unq_base_vecs.shape[0], 1))
        # unq_base_vecs = pt.unsqueeze(unq_base_vecs, dim=1)
        # unq_base_vecs = pt.tile(unq_base_vecs, (1, self.qudit_dims[qudit_idx], 1))
        # unq_base_vecs = pt.reshape(unq_base_vecs, (self.qudit_dims[qudit_idx] * unq_base_vecs.shape[0], -1))
        #
        # unq_base_vecs = pt.cat((unq_base_vecs, continuations), dim=-1)
        # print(pt.stack((zero_outcome, one_outcome), dim=1))
        # print(unq_base_vecs.shape)
        # new_unq_gumbels = self.sample_gumbels_given_max(unq_log_probs, unq_gumbels)
        # new_unq_log_probs = pt.reshape(unq_log_probs, (-1,))
        # new_unq_gumbels = pt.reshape(new_unq_gumbels, (-1,))
        # new_unq_gumbels = pt.nan_to_num(new_unq_gumbels, nan=NEGINF.type(self.rdtype), neginf=NEGINF.type(self.rdtype))

        # if qudit_idx == self.qudit_num - 1:
        #     phys_mask = self.masker.memo[unq_base_vecs.shape[-1], new_unq_memo_idx]
        #     unq_base_vecs = unq_base_vecs[phys_mask]
        #     new_unq_gumbels = new_unq_gumbels[phys_mask]
        #     new_unq_log_probs = new_unq_log_probs[phys_mask]
        #     new_unq_memo_idx = new_unq_memo_idx[phys_mask]

        # sorted_gumbels, sorted_indices = pt.sort(new_unq_gumbels, descending=True)
        #
        # unq_base_vecs = unq_base_vecs[sorted_indices[:sample_num]]
        # new_unq_log_probs = new_unq_log_probs[sorted_indices[:sample_num]]
        # new_unq_gumbels = new_unq_gumbels[sorted_indices[:sample_num]]
        # new_unq_memo_idx = new_unq_memo_idx[sorted_indices[:sample_num]]

        return new_unq_base_vecs, new_unq_log_probs, new_unq_gumbels, new_unq_memo_idx

    def sample_indices_gumbel(self,
                              sample_num: int):
        unq_base_vecs = pt.zeros((1, 0),
                                 dtype=self.idx_dtype,
                                 device=self.device)
        unq_log_probs = pt.zeros((1,),
                                 dtype=BASE_REAL_TYPE,
                                 device=self.device)
        unq_gumbels = pt.zeros((1,),
                               dtype=BASE_REAL_TYPE,
                               device=self.device)
        unq_acc_eigs = []
        for sym_idx in range(self.masker.sym_num):
            unq_acc_eigs.append(pt.zeros(unq_base_vecs.shape[:-1],
                                         dtype=self.idx_dtype,
                                         device=self.device) + self.masker.symmetries[sym_idx].start_eig)
        unq_acc_eigs = pt.stack(unq_acc_eigs, dim=-1)
        unq_memo_idx = self.masker.acc_eigs2memo_idx(unq_acc_eigs)
        for qudit_idx in range(self.qudit_num):
            unq_base_vecs, unq_log_probs, unq_gumbels, unq_memo_idx = self.sample_next_qudit_indices_gumbel(
                sample_num=sample_num,
                qudit_idx=qudit_idx,
                unq_base_vecs=unq_base_vecs,
                unq_log_probs=unq_log_probs,
                unq_gumbels=unq_gumbels,
                unq_memo_idx=unq_memo_idx)
            phys_mask = self.masker.memo[unq_base_vecs.shape[-1], unq_memo_idx]
            # print(f'We have {phys_mask.sum()} physical samples out of {phys_mask.shape[0]}')
            unq_base_vecs = unq_base_vecs[phys_mask]
            unq_log_probs = unq_log_probs[phys_mask]
            unq_gumbels = unq_gumbels[phys_mask]
            unq_memo_idx = unq_memo_idx[phys_mask]
        # phys_mask = self.masker.mask(unq_base_vecs)

        # phys_mask = self.masker.memo[unq_base_vecs.shape[-1], self.masker.acc_eigs2memo_idx(unq_acc_eigs)]

        unq_indices = self.base_vec2base_idx(unq_base_vecs)
        unq_log_probs = unq_log_probs - pt.logsumexp(unq_log_probs, dim=0)
        freqs = pt.exp(unq_log_probs)

        return unq_indices, freqs

    def compute_cat_log_jac(self, indices):
        def params2amps(*params):
            named_params = {self.name_idx2name[name_idx]: param if self.name_idx2dtype[name_idx] == BASE_REAL_TYPE else pt.view_as_complex(param)
                            for name_idx, param in enumerate(params)}

            return pt.view_as_real(pt.log(pt.conj(pt.func.functional_call(self,
                                                                          named_params,
                                                                          indices))))
        param_tuple = tuple(param if param.dtype == BASE_REAL_TYPE else pt.view_as_real(param)
                            for param_name, param in dict(self.named_parameters()).items())
        log_jacs = pt.autograd.functional.jacobian(params2amps,
                                                   param_tuple,
                                                   vectorize=True)
        cat_log_jac = []
        for log_jac in log_jacs:
            cat_log_jac.append(pt.reshape(pt.complex(log_jac[:, 0, ...],
                                                     log_jac[:, 1, ...]),
                                          (log_jac.shape[0], -1)))

        return pt.cat(cat_log_jac, dim=-1)

    @staticmethod
    def spin_flip_base_vec(base_vec):
        assert (base_vec.shape[-1] % 2) == 0
        alpha_subvecs = base_vec[..., ::2]
        beta_subvecs = base_vec[..., 1::2]

        result = pt.cat((pt.unsqueeze(beta_subvecs, dim=-1),
                         pt.unsqueeze(alpha_subvecs, dim=-1)), dim=-1)

        return pt.reshape(result, base_vec.shape)

    def spin_flip_base_idx(self, base_idx):
        return self.base_vec2base_idx(self.spin_flip_base_vec(self.base_idx2base_vec(base_idx)))
    # def qubits2orbits(self, base_vec: pt.Tensor) -> pt.Tensor:
    #     assert (base_vec.shape[-1] % 2) == 0
    #     orbit_num = base_vec.shape[-1] // 2
    #     return pt.scatter_add(pt.zeros((base_vec.shape[0], orbit_num),
    #                                    dtype=self.idx_dtype,
    #                                    device=self.device),
    #                           dim=1,
    #                           index=pt.broadcast_to(self.qubit_idx2orbit_idx[..., :base_vec.shape[-1]], base_vec.shape),
    #                           src=base_vec * self.qubit_idx2orbit_two_power[..., :base_vec.shape[-1]])
    #
    # def orbits2qubits(self, orbits: pt.Tensor) -> pt.Tensor:
    #     shifts = pt.arange(0,
    #                        self.qubit_per_orbit,
    #                        dtype=orbits.dtype,
    #                        device=orbits.device)
    #     return pt.reshape((pt.unsqueeze(orbits, dim=-1) >> shifts).remainder_(2),
    #                       (orbits.shape[0], orbits.shape[-1] * 2))
    #
    # def qubits2sf_cano_repr(self, base_vec: pt.Tensor) -> pt.Tensor:
    #     sf_base_vec = self.spin_flip_base_vec(base_vec)
    #
    #     base_idx = self.base_vec2base_idx(base_vec)
    #     sf_base_idx = self.base_vec2base_idx(sf_base_vec)
    #
    #     diff = base_idx - sf_base_idx
    #
    #     first_non_zero_diff_pos = pt.argmin(-pt.isclose(diff, pt.zeros_like(diff)).type(pt.int), dim=-1, keepdim=True)
    #     first_non_zero_diffs = pt.squeeze(pt.gather(diff, dim=1, index=first_non_zero_diff_pos))
    #
    #     sf_cano_repr = pt.where(pt.unsqueeze(first_non_zero_diffs > 0, dim=-1),
    #                             base_vec,
    #                             sf_base_vec)
    #
    #     return sf_cano_repr




