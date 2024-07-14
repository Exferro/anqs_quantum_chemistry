import torch as pt
import numpy as np

from .constants import BASE_INT_TYPE, BASE_REAL_TYPE, BASE_COMPLEX_TYPE
from ..utils.popcount import popcount
from ..utils.custom_popcount import cuda_int64_popcount, cuda_int64_popcount_


class HilbertSpace:
    SUPPORTED_IDX_DTYPES = (pt.int32, pt.int64)
    SUPPORTED_RDTYPES = (pt.float, pt.double)
    SUPPORTED_CDTYPES = (pt.cfloat, pt.cdouble)
    IDX_DTYPE_TO_BIT_DEPTH = {
        pt.int32: 32,
        pt.int64: 64,
    }
    IDX_DTYPE_TO_ALPHA_POS_MASK = {
        pt.int32: 0x3AAAAAAA,
        pt.int64: 0x3AAAAAAAAAAAAAAA,
    }
    IDX_DTYPE_TO_BETA_POS_MASK = {
        pt.int32: 0x35555555,
        pt.int64: 0x3555555555555555,
    }

    DEFAULT_GPU_MEMORY_LIMIT = 6 * 10**9

    ALLOWED_PERM_TYPES = ('direct', 'inverse')
    ALLOWED_POPCOUNT_MODES = ('compute_efficient', 'memory_efficient', 'custom')

    def __init__(self,
                 *,
                 qubit_num: int = 0,
                 device=None,
                 rdtype=BASE_REAL_TYPE,
                 cdtype=BASE_COMPLEX_TYPE,
                 idx_dtype=BASE_INT_TYPE,
                 parent_dir: str = None,
                 rng_seed: int = None,
                 rng=None,
                 gpu_memory_limit: int = DEFAULT_GPU_MEMORY_LIMIT,
                 perm_type: str = 'direct',
                 popcount_mode: str = 'custom'):
        assert idx_dtype in HilbertSpace.SUPPORTED_IDX_DTYPES
        self.idx_dtype = idx_dtype

        assert device is not None
        self.device = device

        self.qubit_num = qubit_num

        self.bit_depth = HilbertSpace.IDX_DTYPE_TO_BIT_DEPTH[idx_dtype]
        self.int_per_idx = (self.qubit_num // self.bit_depth) + 1 * ((self.qubit_num % self.bit_depth) > 0)

        # Internal variables required for base_idx2base_vec and base_vec2base_idx calculations
        self.split_sizes = [min(self.bit_depth, self.qubit_num - int_idx * self.bit_depth) for int_idx in range(self.int_per_idx)]
        self.split_starts = []
        self.split_ends = []
        self.shifts = []
        self.two_powers = []
        for int_idx in range(self.int_per_idx):
            if int_idx == 0:
                self.split_starts.append(0)
            else:
                self.split_starts.append(self.split_starts[int_idx - 1] + self.split_sizes[int_idx - 1])
            self.split_ends.append(self.split_starts[int_idx] + self.split_sizes[int_idx])

            self.shifts.append(pt.arange(0,
                                         self.split_sizes[int_idx],
                                         dtype=self.idx_dtype,
                                         device=self.device))
            self.two_powers.append(pt.reshape(pt.tensor([1],
                                                        dtype=self.idx_dtype,
                                                        device=self.device),
                                              (1, )) << self.shifts[int_idx])

        assert rdtype in HilbertSpace.SUPPORTED_RDTYPES
        self.rdtype = rdtype

        assert cdtype in HilbertSpace.SUPPORTED_CDTYPES
        self.cdtype = cdtype

        assert parent_dir is not None
        self.parent_dir = parent_dir

        assert rng_seed is not None
        self.rng_seed = rng_seed
        if rng is None:
            self.rng = np.random.default_rng(seed=self.rng_seed)
            pt.manual_seed(self.rng_seed)
        else:
            self.rng = rng

        self.gpu_memory_limit = gpu_memory_limit
        self.max_idx_num_per_mask = self.gpu_memory_limit // (8 * self.qubit_num)

        self.perm = pt.arange(self.qubit_num, dtype=self.idx_dtype, device=self.device)
        self.inv_perm = pt.arange(self.qubit_num, dtype=self.idx_dtype, device=self.device)
        assert perm_type in self.ALLOWED_PERM_TYPES
        self.perm_type = perm_type

        if self.perm_type == 'inverse':
            self.perm = pt.arange(self.qubit_num - 1, -1, -1, dtype=self.idx_dtype, device=self.device)
            self.inv_perm = pt.arange(self.qubit_num - 1, -1, -1, dtype=self.idx_dtype, device=self.device)

        assert popcount_mode in self.ALLOWED_POPCOUNT_MODES
        self.popcount_mode = popcount_mode
        if self.popcount_mode == 'compute_efficient':
            self.popcounts32 = []
            chunk_num = 16
            chunk_size = 2**32 // chunk_num
            for chunk_idx in range(chunk_num):
                chunk_start = chunk_idx * chunk_size
                self.popcounts32.append(popcount(pt.arange(chunk_start, chunk_start + chunk_size, device=self.device, dtype=pt.int32), bit_depth=32).type(pt.int8))
            self.popcounts32 = pt.cat(self.popcounts32)
        else:
            self.popcounts32 = None
        self.first_32_bits = 0x00000000FFFFFFFF

    # noinspection DuplicatedCode
    def base_idx2base_vec(self, base_idx: pt.Tensor) -> pt.Tensor:
        if not pt.is_tensor(base_idx):
            base_idx = pt.tensor(base_idx, dtype=self.idx_dtype, device=self.device)
        assert len(base_idx.shape) == 2
        assert base_idx.shape[-1] == self.int_per_idx
        assert base_idx.device == self.device

        base_vecs = []
        for int_idx in range(self.int_per_idx):
            base_vecs.append((base_idx[:, int_idx].reshape(-1, 1) >> self.shifts[int_idx]).remainder_(2))

        return pt.cat(base_vecs, dim=-1)

    # noinspection DuplicatedCode
    def base_vec2base_idx(self, base_vec: pt.Tensor) -> pt.Tensor:
        if not pt.is_tensor(base_vec):
            base_vec = pt.tensor(base_vec, dtype=self.idx_dtype, device=self.device)
        if base_vec.dtype != self.idx_dtype:
            base_vec = base_vec.type(self.idx_dtype)
        assert base_vec.device == self.device
        base_idxs = []
        for int_idx in range(self.int_per_idx):
            base_idxs.append(pt.sum(pt.mul(base_vec[:, self.split_starts[int_idx]:self.split_ends[int_idx]],
                                           self.two_powers[int_idx]),
                                    dim=-1, keepdim=True))

        return pt.cat(base_idxs, dim=-1)

    def compute_efficient_popcount(self, base_idx: pt.Tensor, bit_depth: int = 64) -> pt.Tensor:
        if bit_depth == 64:
            return pt.add(self.popcounts32[pt.bitwise_and(base_idx, self.first_32_bits)],
                    self.popcounts32[pt.bitwise_right_shift(base_idx, 32)]).type(base_idx.dtype)
        elif bit_depth == 32:
            raise NotImplementedError
        else:
            raise RuntimeError(f'Wrong bit depth: {bit_depth}')

    def popcount(self, base_idx: pt.Tensor) -> pt.Tensor:
        result = None
        if self.popcount_mode == 'custom':
            return cuda_int64_popcount(base_idx.view((-1, ))).view((-1, self.int_per_idx)).sum(dim=-1)
        else:
            for int_idx in range(self.int_per_idx):
                if self.popcount_mode == 'compute_efficient':
                    cur_popcounts = self.compute_efficient_popcount(base_idx[:, int_idx], bit_depth=self.bit_depth)
                elif self.popcount_mode == 'memory_efficient':
                    cur_popcounts = popcount(base_idx[:, int_idx], bit_depth=self.bit_depth)
                else:
                    raise ValueError(f'Wrong popcount type: {self.popcount_mode}')
                if result is None:
                    result = cur_popcounts
                else:
                    result = result.add_(cur_popcounts)
            return result

    def popcount_(self, base_idx: pt.Tensor) -> pt.Tensor:
        result = None
        if self.popcount_mode == 'custom':
            return cuda_int64_popcount_(base_idx.view((-1, ))).view((-1, self.int_per_idx)).sum(dim=-1)
        else:
            for int_idx in range(self.int_per_idx):
                if self.popcount_mode == 'compute_efficient':
                    cur_popcounts = self.compute_efficient_popcount(base_idx[:, int_idx], bit_depth=self.bit_depth)
                elif self.popcount_mode == 'memory_efficient':
                    cur_popcounts = popcount(base_idx[:, int_idx], bit_depth=self.bit_depth)
                else:
                    raise ValueError(f'Wrong popcount type: {self.popcount_mode}')
                if result is None:
                    result = cur_popcounts
                else:
                    result = result.add_(cur_popcounts)
            return result

    def old_popcount(self, base_idx: pt.Tensor) -> pt.Tensor:
        popcounts = []
        for int_idx in range(self.int_per_idx):
            popcounts.append(popcount(base_idx[:, int_idx], bit_depth=self.bit_depth))
        return pt.sum(pt.stack(popcounts, dim=-1), dim=-1)

    @staticmethod
    def two_unique2cat_unique(unq_1, unq_inv_1, unq_2, unq_inv_2):
        if len(unq_1.shape) == 1:
            unq_1 = pt.reshape(unq_1, (-1, 1))
        if len(unq_2.shape) == 1:
            unq_2 = pt.reshape(unq_2, (-1, 1))

        assert unq_inv_1.shape[0] == unq_inv_2.shape[0]
        ordinals = unq_inv_1 + unq_inv_2 * unq_1.shape[0]
        unq_ordinals, unq_inv = pt.unique(ordinals, return_inverse=True)

        merge_unq = pt.cat([unq_1[unq_ordinals % unq_1.shape[0]], unq_2[unq_ordinals // unq_1.shape[0]]], dim=-1)

        return merge_unq, unq_inv

    def compute_unique_indices(self, base_idx):
        assert len(base_idx.shape) == 2
        assert base_idx.shape[-1] == self.int_per_idx

        prev_unq, prev_unq_inv = pt.unique(base_idx[..., 0], return_inverse=True)
        prev_unq = pt.reshape(prev_unq, (-1, 1))
        for int_idx in range(1, self.int_per_idx):
            new_unq, new_unq_inv = pt.unique(base_idx[..., int_idx], return_inverse=True)
            prev_unq, prev_unq_inv = self.two_unique2cat_unique(unq_1=prev_unq,
                                                                unq_inv_1=prev_unq_inv,
                                                                unq_2=new_unq,
                                                                unq_inv_2=new_unq_inv)

        return prev_unq, prev_unq_inv

    def init_perm(self,
                  perm_type: str = 'direct'):
        assert perm_type in self.ALLOWED_PERM_TYPES
        if perm_type == 'direct':
            self.perm = pt.arange(self.qubit_num, dtype=self.idx_dtype, device=self.device)
            self.inv_perm = pt.arange(self.qubit_num, dtype=self.idx_dtype, device=self.device)

        return self.perm, self.inv_perm

    def sort_base_idx(self,
                      base_idx: pt.Tensor = None,
                      descending: bool = False):
        if descending:
            raise NotImplementedError
        else:
            sorted_base_idx = base_idx
            sort_perm = pt.arange(base_idx.shape[0], device=self.device)
            for int_idx in range(self.int_per_idx):
                _, cur_sort_perm = pt.sort(sorted_base_idx[:, int_idx], stable=True, descending=False)
                sorted_base_idx = sorted_base_idx[cur_sort_perm]
                sort_perm = sort_perm[cur_sort_perm]

                neg_mask = sorted_base_idx[:, int_idx] < 0
                arange = pt.arange(sorted_base_idx.shape[0], device=self.device)
                neg_indices = arange[neg_mask]
                non_neg_indices = arange[~neg_mask]
                cur_neg_perm = pt.cat((non_neg_indices, neg_indices), dim=0)

                sorted_base_idx = sorted_base_idx[cur_neg_perm]
                sort_perm = sort_perm[cur_neg_perm]

            return sorted_base_idx, sort_perm

    def find_a_in_b(self, a: pt.Tensor, b: pt.Tensor):
        assert len(a.shape) <= 2
        assert len(b.shape) <= 2
        assert len(a.shape) == len(b.shape)
        if len(a.shape) == 2:
            assert a.shape[1] == b.shape[1]
            assert a.shape[1] == self.int_per_idx

        a_cat_b = pt.cat((a, b), dim=0)
        unq_a_cat_b, a_cat_b_as_unq_a_cat_b_ptr = self.compute_unique_indices(a_cat_b)

        unq_a_cat_b_as_b_ptr = -pt.ones((unq_a_cat_b.shape[0], ),
                                        dtype=self.idx_dtype,
                                        device=self.device)
        unq_a_cat_b_as_b_ptr.scatter_(dim=0,
                                      index=a_cat_b_as_unq_a_cat_b_ptr[a.shape[0]:],
                                      src=pt.arange(b.shape[0], device=self.device))
        a_as_b_ptr = unq_a_cat_b_as_b_ptr[a_cat_b_as_unq_a_cat_b_ptr[:a.shape[0]]]

        a_in_b_mask = (a_as_b_ptr != -1)

        return a_in_b_mask, a_as_b_ptr
