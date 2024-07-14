import os

import numpy as np
import torch as pt
from typing import Tuple, Any

import time

from openfermion import QubitOperator
from openfermion.utils import count_qubits

from ...base.constants import BASE_INT_TYPE, BASE_COMPLEX_TYPE

from ...base import AbstractObservable
from ...base import AbstractQuantumState

from ...utils.trie import Trie

from ...utils.misc import verbosify_generator, compute_chunk_boundaries

from ...infrastructure.nested_data import Config
from ...infrastructure.timed_decorator import timed


class LocalEnergyMetrics(Config):
    FIELDS = (
        'candidate_x_primes_num',
        'sampled_unq_x_primes_num',
        'sampled_x_primes_num',
        'sampled_coupled_yz_num',
        'non_sampled_unq_x_primes_num',
        'non_sampled_x_primes_num',
        'non_sampled_coupled_yz_num',
        'candidates_time',
        'filter_candidates_time',
        'find_a_in_b_time',
        'find_sampled_and_coupled_time',
        'sampled_matrix_elements_time',
        'sampled_scatter_time',
        'non_sampled_matrix_elements_time',
        'non_sampled_scatter_time',
        'eval_non_sampled_amps_time',
    )

    def __init__(self,
                 *args,
                 candidate_x_primes_num: int = np.nan,
                 sampled_unq_x_primes_num: int = np.nan,
                 sampled_x_primes_num: int = np.nan,
                 sampled_coupled_yz_num: int = np.nan,
                 non_sampled_unq_x_primes_num: int = np.nan,
                 non_sampled_x_primes_num: int = np.nan,
                 non_sampled_coupled_yz_num: int = np.nan,
                 candidates_time: float = np.nan,
                 filter_candidates_time: float = np.nan,
                 find_a_in_b_time: float = np.nan,
                 find_sampled_and_coupled: float = np.nan,
                 sampled_matrix_elements_time: float = np.nan,
                 sampled_scatter_time: float = np.nan,
                 non_sampled_matrix_elements_time: float = np.nan,
                 non_sampled_scatter_time: float = np.nan,
                 eval_non_sampled_amps_time: float = np.nan,
                 **kwargs):
        self.candidate_x_primes_num = candidate_x_primes_num

        self.sampled_unq_x_primes_num = sampled_unq_x_primes_num
        self.sampled_x_primes_num = sampled_x_primes_num
        self.sampled_coupled_yz_num = sampled_coupled_yz_num

        self.non_sampled_unq_x_primes_num = non_sampled_unq_x_primes_num
        self.non_sampled_x_primes_num = non_sampled_x_primes_num
        self.non_sampled_coupled_yz_num = non_sampled_coupled_yz_num

        self.candidates_time = candidates_time
        self.filter_candidates_time = filter_candidates_time
        self.find_a_in_b_time = find_a_in_b_time
        self.find_sampled_and_coupled_time = find_sampled_and_coupled

        self.sampled_matrix_elements_time = sampled_matrix_elements_time
        self.sampled_scatter_time = sampled_scatter_time

        self.non_sampled_matrix_elements_time = non_sampled_matrix_elements_time
        self.non_sampled_scatter_time = non_sampled_scatter_time
        self.eval_non_sampled_amps_time = eval_non_sampled_amps_time

        super().__init__(*args, **kwargs)


class PauliObservable(AbstractObservable):
    ALLOWED_COUPLING_METHODS = ('ham', 'all_to_all', 'hamming_ball', 'trie')

    MEMORY_MAGIC_CONSTANT = 25

    def __init__(self,
                 *args,
                 of_qubit_operator: QubitOperator = None,
                 **kwargs):
        super(PauliObservable, self).__init__(*args, **kwargs)
        assert self.qubit_num == count_qubits(of_qubit_operator)
        self.of_qubit_operator = of_qubit_operator
        self.term_num = len(self.of_qubit_operator.terms)

        self.unq_xy_masks = None
        self.unq_xy_masks_num = None
        self.unq_xy_to_yz_num = None
        self.unq_xy_to_yz_start = None
        self.rearranged_yz = None
        self.rearranged_weights = None

        self.local_energy_structure_tensor_names = ('unq_xy_masks',
                                                    'unq_xy_masks_inv',
                                                    'unq_xy_to_yz_num',
                                                    'unq_xy_to_yz_start',
                                                    'rearranged_yz',
                                                    'rearranged_weights')
        self.tensor_name2tensor_path = {tensor_name: os.path.join(self.hilbert_space.parent_dir,
                                                                  f'{tensor_name}.npy')
                                        for tensor_name in self.local_energy_structure_tensor_names}
        all_tensors_are_cached = True
        for tensor_name in self.tensor_name2tensor_path:
            if not os.path.exists(self.tensor_name2tensor_path[tensor_name]):
                all_tensors_are_cached = False
                break
        if all_tensors_are_cached:
            for tensor_name in self.tensor_name2tensor_path:
                tensor = np.load(self.tensor_name2tensor_path[tensor_name])
                tensor = pt.from_numpy(tensor).to(self.device)
                setattr(self, tensor_name, tensor)
            self.unq_xy_masks_num = self.unq_xy_masks.shape[0]
        else:
            weights, xy_masks, yz_masks = self.parse_of_qubit_operator(of_qubit_operator=self.of_qubit_operator,
                                                                       dtype=self.cdtype,
                                                                       idx_dtype=self.idx_dtype,
                                                                       device=self.device)
            self.unq_xy_masks, self.unq_xy_masks_inv = self.hilbert_space.compute_unique_indices(xy_masks)

            # Filling data structures required for sample-aware local energy calculations
            self.unq_xy_masks_num = self.unq_xy_masks.shape[0]
            self.unq_xy_to_yz_num, self.unq_xy_to_yz_start, self.rearranged_yz, self.rearranged_weights = self.compute_local_energy_structures(unq_xy_masks=self.unq_xy_masks,
                                                                                                                                               unq_xy_masks_inv=self.unq_xy_masks_inv,
                                                                                                                                               yz_masks=yz_masks,
                                                                                                                                               weights=weights)
            for tensor_name in self.tensor_name2tensor_path:
                np.save(self.tensor_name2tensor_path[tensor_name],
                        getattr(self, tensor_name).cpu().numpy())
        self.unq_ham_xy_trie = Trie(hilbert_space=self.hilbert_space,
                                    batch_as_base_indices=self.unq_xy_masks)
        _, self.unq_ham_xy_trie_node_idx2unq_ham_xy_ptr = pt.sort(self.unq_ham_xy_trie.base_vec2all_lvl_node_indices[:, -1])

    def parse_of_qubit_operator(self,
                                of_qubit_operator: QubitOperator = None,
                                dtype=BASE_COMPLEX_TYPE,
                                idx_dtype=BASE_INT_TYPE,
                                device=pt.device('cpu')):
        qubit_num = count_qubits(of_qubit_operator)
        weights, xy_masks, yz_masks = [], [], []
        for qubit_ops, weight in of_qubit_operator.terms.items():
            weights.append(weight + 0j)
            xy_mask = [pt.tensor(0, dtype=idx_dtype)] * self.hilbert_space.int_per_idx
            yz_mask = [pt.tensor(0, dtype=idx_dtype)] * self.hilbert_space.int_per_idx
            for qubit_op in qubit_ops:
                pos = qubit_num - qubit_op[0] - 1
                mod_pos = pos % self.hilbert_space.bit_depth
                if mod_pos == (self.hilbert_space.bit_depth - 1):
                    two_power = -2**mod_pos
                else:
                    two_power = 2**mod_pos
                if qubit_op[1] == 'X' or qubit_op[1] == 'Y':
                    xy_mask[pos // self.hilbert_space.bit_depth] = pt.bitwise_or(xy_mask[pos // self.hilbert_space.bit_depth],
                                                                                 two_power)

                if qubit_op[1] == 'Y' or qubit_op[1] == 'Z':
                    yz_mask[pos // self.hilbert_space.bit_depth] = pt.bitwise_or(yz_mask[pos // self.hilbert_space.bit_depth],
                                                                                 two_power)

                if qubit_op[1] == 'Y':
                    weights[-1] *= 1j
            xy_masks.append(xy_mask)
            yz_masks.append(yz_mask)

        return (pt.tensor(weights, dtype=dtype, device=device),
                pt.tensor(xy_masks, dtype=idx_dtype, device=device),
                pt.tensor(yz_masks, dtype=idx_dtype, device=device))

    def compute_local_energy_structures(self,
                                        unq_xy_masks: pt.Tensor = None,
                                        unq_xy_masks_inv: pt.Tensor = None,
                                        yz_masks: pt.Tensor = None,
                                        weights: pt.Tensor = None):
        unq_xy_masks_num = unq_xy_masks.shape[0]
        unq_xy_to_yz_num = [0 for _ in range(unq_xy_masks_num)]
        rearranged_yz = [[] for _ in range(unq_xy_masks_num)]
        rearranged_weights = [[] for _ in range(unq_xy_masks_num)]
        for term_idx, unq_xy_mask_idx in enumerate(unq_xy_masks_inv):
            unq_xy_to_yz_num[unq_xy_mask_idx.item()] += 1
            rearranged_yz[unq_xy_mask_idx.item()].append(yz_masks[term_idx:term_idx + 1])
            rearranged_weights[unq_xy_mask_idx.item()].append(weights[term_idx:term_idx + 1])

        unq_xy_to_yz_num = pt.tensor(unq_xy_to_yz_num, device=self.device)
        unq_xy_to_yz_start = pt.cumsum(unq_xy_to_yz_num, dim=0)
        unq_xy_to_yz_start = pt.roll(unq_xy_to_yz_start, 1, dims=0)
        unq_xy_to_yz_start[0] = 0

        for unq_xy_mask_idx in range(unq_xy_masks_num):
            rearranged_yz[unq_xy_mask_idx] = pt.cat(rearranged_yz[unq_xy_mask_idx], dim=0)
            rearranged_weights[unq_xy_mask_idx] = pt.cat(rearranged_weights[unq_xy_mask_idx], dim=0)

        rearranged_yz = pt.cat(rearranged_yz, dim=0)
        rearranged_weights = pt.cat(rearranged_weights, dim=0)

        return unq_xy_to_yz_num, unq_xy_to_yz_start, rearranged_yz, rearranged_weights

    def compute_ham_xy_pointers(self, coupling_xys: pt.Tensor = None):
        coupling_xys_num = coupling_xys.shape[0]

        coupling_or_ham_xys = pt.cat((coupling_xys, self.unq_xy_masks), dim=0)
        unq_coupling_or_ham_xys, unq_coupling_or_ham_xys_inv = self.hilbert_space.compute_unique_indices(coupling_or_ham_xys)

        unq_coupling_or_ham_xy_is_ham_xy = pt.zeros((unq_coupling_or_ham_xys.shape[0], ),
                                                   dtype=pt.bool,
                                                   device=self.device)
        unq_coupling_or_ham_xy_is_ham_xy.scatter_add_(dim=0,
                                                      index=unq_coupling_or_ham_xys_inv[coupling_xys_num:],
                                                      src=pt.ones((self.unq_xy_masks_num,),
                                                                  dtype=pt.bool,
                                                                  device=self.device))
        coupling_xy_is_ham_xy = unq_coupling_or_ham_xy_is_ham_xy[unq_coupling_or_ham_xys_inv[:coupling_xys_num]]

        unq_coupling_or_ham_xy_ham_xy_ordinal = pt.zeros((unq_coupling_or_ham_xys.shape[0], ),
                                                         dtype=self.idx_dtype,
                                                         device=self.device)
        unq_coupling_or_ham_xy_ham_xy_ordinal.scatter_add_(dim=0,
                                                           index=unq_coupling_or_ham_xys_inv[coupling_xys_num:],
                                                           src=pt.arange(self.unq_xy_masks_num, device=self.device))
        ham_xy_pointers = unq_coupling_or_ham_xy_ham_xy_ordinal[unq_coupling_or_ham_xys_inv[:coupling_xys_num]]
        ham_xy_pointers = ham_xy_pointers[coupling_xy_is_ham_xy]

        return ham_xy_pointers, coupling_xy_is_ham_xy

    @staticmethod
    def expand_pointers(starts: pt.Tensor = None,
                        nums: pt.Tensor = None):
        assert len(starts.shape) == 1
        assert len(nums.shape) == 1
        assert starts.shape[0] == nums.shape[0]
        pointer_borders = pt.roll(pt.cumsum(nums, dim=0), 1, dims=0)
        pointer_borders[0] = 0
        pointer_borders = pt.repeat_interleave(pointer_borders - starts, nums)

        pointer_increments = pt.ones_like(pointer_borders)
        pointer_increments = pt.cumsum(pointer_increments, dim=0) - 1

        return pointer_increments - pointer_borders

    @timed
    def compute_matrix_elements(self,
                                x_primes: pt.Tensor = None,
                                ham_xy_pointers: pt.Tensor = None,
                                chunk_size: int = np.inf):
        """
        A function which calculates the matrix elements of the Hamiltonian.
        We assume that we have flattened all the base indices xprimes coupled to input base indices x (which are not
        required in this calculations. So, for each xprime we have a corresponding unique coupling xy mask from
        Hamiltonian.

        To calculate the matrix element, we expand every xy_mask to its corresponding yz masks. Then, we calculate the
        sign of the contribution to the matrix element by finding bitwise AND between the yz mask and the xprime.
        Then, we multiply the weights of the yz masks by these signs and scatter add them into a singular float
        corresponding to each xprime.
        """
        assert len(x_primes.shape) == 2
        assert len(ham_xy_pointers.shape) == 1
        assert x_primes.shape[0] == ham_xy_pointers.shape[0]


        chunk_num = None
        chunk_boundaries = None
        
        if chunk_size == np.inf:
            chunk_num = 1
            chunk_boundaries = [0, x_primes.shape[0]]
        else:
            chunk_boundaries = compute_chunk_boundaries(array_len=x_primes.shape[0],
                                                        chunk_size=chunk_size)
            chunk_num = len(chunk_boundaries) - 1
            
        matrix_elements = pt.tensor([], dtype=self.rearranged_weights.dtype, device=self.device)
        yz_num = 0
        
        for chunk_idx in range(chunk_num):
            chunk_start = chunk_boundaries[chunk_idx]
            chunk_end = chunk_boundaries[chunk_idx + 1]
            
            cur_x_primes = x_primes[chunk_start:chunk_end]
            cur_ham_xy_pointers = ham_xy_pointers[chunk_start:chunk_end]                                                      
        
            # First, we calculate the starting positions and the number of ham_yzs corresponding to each ham_xy.
            # We have precalculated this at the start of simulation, and so now we just need to fetch these
            # numbers from the hamiltonian since we already know the "ordinals" of coupling unq_ham_xy.
            ham_yz_starts = self.unq_xy_to_yz_start[cur_ham_xy_pointers]
            ham_yz_nums = self.unq_xy_to_yz_num[cur_ham_xy_pointers]

            # Given this information we do our usual trick where we basically calculate the list of pointers to
            # ham_yzs and weights (or, more generally, to terms).
            ham_term_pointers = self.expand_pointers(ham_yz_starts, ham_yz_nums)


            signs = ((-1) ** (self.popcount_(pt.repeat_interleave(cur_x_primes, ham_yz_nums, dim=0).bitwise_and_(self.rearranged_yz[ham_term_pointers])))).to(self.cdtype)
            
            signs_mul_weights = signs.mul_(self.rearranged_weights[ham_term_pointers])
            
            ham_yz_to_xprime_scatter_idx = pt.repeat_interleave(pt.arange(cur_x_primes.shape[0], device=self.device),
                                                                ham_yz_nums)
            
            cur_matrix_elements = pt.zeros((cur_x_primes.shape[0],), device=self.device, dtype=self.cdtype)
            cur_matrix_elements.scatter_add_(dim=0,
                                         index=ham_yz_to_xprime_scatter_idx,
                                         src=signs_mul_weights)
            
            matrix_elements = pt.cat((matrix_elements, cur_matrix_elements))
            
            yz_num += signs.shape[0]
            
        return matrix_elements, signs.shape[0]

    @pt.no_grad()
    def compute_local_energies(self,
                               wf: AbstractQuantumState = None,
                               sampled_indices: pt.Tensor = None,
                               sampled_amps: pt.Tensor = None,
                               verbose: bool = False,
                               use_tree_for_candidates: bool = False,
                               chunk_size: int = 20000,
                               sample_aware: bool = False,
                               compute_via_ham_xy_coupling: bool = True) -> Tuple[
        pt.Tensor, pt.Tensor, LocalEnergyMetrics]:
        sampled_indices_num = sampled_indices.shape[0]
        if use_tree_for_candidates:
            assert sample_aware is True
            assert compute_via_ham_xy_coupling is False
            return self.compute_local_energies_via_indices_coupling(
                sampled_base_idx_idx_to=pt.arange(sampled_indices_num, device=self.device),
                sampled_base_idx_idx_from=pt.arange(sampled_indices_num, device=self.device),
                sampled_indices=sampled_indices,
                sampled_amps=sampled_amps,
                use_tree_for_candidates=True,
                symmetric=True)
        else:
            chunk_boundaries = compute_chunk_boundaries(array_len=sampled_indices_num,
                                                        chunk_size=chunk_size)
            chunk_num = len(chunk_boundaries) - 1
            if (sample_aware is True) and (compute_via_ham_xy_coupling is False) and (chunk_num == 1):
                return self.compute_local_energies_via_indices_coupling(
                    sampled_base_idx_idx_to=pt.arange(sampled_indices_num, device=self.device),
                    sampled_base_idx_idx_from=pt.arange(sampled_indices_num, device=self.device),
                    sampled_indices=sampled_indices,
                    sampled_amps=sampled_amps,
                    symmetric=True)
            else:
                metrics = LocalEnergyMetrics()
                local_energies = pt.tensor([], dtype=wf.dtype, device=wf.device)
                var_local_energies = pt.tensor([], dtype=wf.dtype, device=wf.device)
                for chunk_idx in verbosify_generator(range(chunk_num),
                                                     verbose=verbose,
                                                     activity_descr='Computing local energies'):
                    chunk_start = chunk_boundaries[chunk_idx]
                    chunk_end = chunk_boundaries[chunk_idx + 1]
                    if (sample_aware is True) and (compute_via_ham_xy_coupling is False):
                        cur_local_energies, cur_var_local_energies, cur_metrics = self.compute_local_energies_via_indices_coupling(
                            sampled_base_idx_idx_to=pt.arange(start=chunk_start, end=chunk_end, device=self.device),
                            sampled_base_idx_idx_from=pt.arange(sampled_indices_num, device=self.device),
                            sampled_indices=sampled_indices,
                            sampled_amps=sampled_amps,
                            symmetric=False)
                    else:
                        cur_local_energies, cur_var_local_energies, cur_metrics = self.compute_local_energies_via_ham_xy_coupling(
                            sampled_base_idx_idx_to=pt.arange(start=chunk_start, end=chunk_end, device=self.device),
                            sampled_indices=sampled_indices,
                            sampled_amps=sampled_amps,
                            wf=wf,
                            sample_aware=sample_aware)

                    local_energies = pt.cat([local_energies, cur_local_energies])
                    var_local_energies = pt.cat([var_local_energies, cur_var_local_energies])
                    # Update the statistics
                    for field_name in metrics.FIELDS:
                        field_value = getattr(metrics, field_name)
                        if np.isnan(field_value):
                            setattr(metrics, field_name, getattr(cur_metrics, field_name))
                        else:
                            setattr(metrics, field_name, field_value + getattr(cur_metrics, field_name))

                return local_energies, var_local_energies, metrics

    @pt.no_grad()
    def compute_var_local_energy_proxy(self,
                                       unq_batch_as_base_indices: pt.Tensor = None,
                                       unq_batch_as_amps: pt.Tensor = None,
                                       coupling_method: str = None,
                                       chunk_size: int = 20000,
                                       alpha_num: int = None,
                                       beta_num: int = None,
                                       matrix_element_chunk_size: int = np.inf) -> Tuple[pt.Tensor, pt.Tensor, LocalEnergyMetrics]:
        assert coupling_method in self.ALLOWED_COUPLING_METHODS

        chunk_num = None
        chunk_boundaries = None
        symmetric = False

        unq_batch_size = unq_batch_as_base_indices.shape[0]
        if (coupling_method == 'trie') or (coupling_method == 'hamming_ball'):
            chunk_num = 1
            chunk_boundaries = [0, unq_batch_size]
            symmetric = True
        else:
            chunk_boundaries = compute_chunk_boundaries(array_len=unq_batch_size,
                                                        chunk_size=chunk_size)
            chunk_num = len(chunk_boundaries) - 1
            if (coupling_method == 'all_to_all') and (chunk_num == 1):
                symmetric = True

        metrics = LocalEnergyMetrics()
        local_proxy_energies = pt.tensor([], dtype=unq_batch_as_amps.dtype, device=self.device)

        unq_batch_ptrs = pt.arange(unq_batch_size, dtype=self.idx_dtype, device=self.device)
        for chunk_idx in range(chunk_num):
            cur_chunk_metrics = LocalEnergyMetrics()
            chunk_start = chunk_boundaries[chunk_idx]
            chunk_end = chunk_boundaries[chunk_idx + 1]

            chunk_as_unq_batch_ptrs = unq_batch_ptrs[chunk_start:chunk_end]

            (dest_as_chunk_ptrs,
             src_as_unq_batch_ptrs,
             src_as_base_indices,
             coupling_xy_as_unq_ham_xy_ptrs,
             cur_chunk_metrics,
             find_sampled_and_coupled_time) = self.find_sampled_and_coupled(
                chunk_as_unq_batch_ptrs=chunk_as_unq_batch_ptrs,
                unq_batch_as_base_indices=unq_batch_as_base_indices,
                coupling_method=coupling_method,
                symmetric=symmetric,
                alpha_num=alpha_num,
                beta_num=beta_num,
                metrics=cur_chunk_metrics)
            cur_chunk_metrics.sampled_x_primes_num = src_as_base_indices.shape[0]
            cur_chunk_metrics.find_sampled_and_coupled_time = find_sampled_and_coupled_time

            dest_as_unq_batch_ptrs = chunk_as_unq_batch_ptrs[dest_as_chunk_ptrs]
            ham_elements, cur_chunk_metrics.sampled_coupled_yz_num, cur_chunk_metrics.sampled_matrix_elements_time = self.compute_matrix_elements(x_primes=src_as_base_indices,
                                                                                                                                                  ham_xy_pointers=coupling_xy_as_unq_ham_xy_ptrs,
                                                                                                                                                  chunk_size=matrix_element_chunk_size)
            cur_chunk_local_proxy_energies = pt.zeros((chunk_end - chunk_start),
                                                      dtype=ham_elements.dtype,
                                                      device=self.device)
            scatter_start_time = time.time()
            if symmetric:
                ham_elements[dest_as_unq_batch_ptrs == src_as_unq_batch_ptrs] = ham_elements[dest_as_unq_batch_ptrs == src_as_unq_batch_ptrs] / 2
                src_amps = pt.cat((unq_batch_as_amps[src_as_unq_batch_ptrs],
                                   unq_batch_as_amps[dest_as_unq_batch_ptrs]),
                                  dim=0)
                cur_chunk_local_proxy_energies = cur_chunk_local_proxy_energies.scatter_add_(dim=0,
                                                                                             index=pt.cat((dest_as_chunk_ptrs,
                                                                                                           src_as_unq_batch_ptrs),
                                                                                                          dim=0),
                                                                                             src=pt.cat((ham_elements,
                                                                                                         pt.conj(ham_elements)),
                                                                                                        dim=0) * src_amps)
            else:
                src_amps = unq_batch_as_amps[src_as_unq_batch_ptrs]
                cur_chunk_local_proxy_energies = cur_chunk_local_proxy_energies.scatter_add_(dim=0,
                                                                                             index=dest_as_chunk_ptrs,
                                                                                             src=ham_elements * src_amps)
            scatter_end_time = time.time()
            cur_chunk_metrics.sampled_scatter_time = scatter_end_time - scatter_start_time

            cur_chunk_local_proxy_energies = cur_chunk_local_proxy_energies / unq_batch_as_amps[chunk_as_unq_batch_ptrs]
            local_proxy_energies = pt.cat((local_proxy_energies, cur_chunk_local_proxy_energies))

            for field_name in metrics.FIELDS:
                field_value = getattr(metrics, field_name)
                if np.isnan(field_value):
                    setattr(metrics, field_name, getattr(cur_chunk_metrics, field_name))
                else:
                    setattr(metrics, field_name, field_value + getattr(cur_chunk_metrics, field_name))

        return local_proxy_energies, local_proxy_energies, metrics

    def find_sampled_and_coupled(self,
                                 chunk_as_unq_batch_ptrs: pt.Tensor = None,
                                 unq_batch_as_base_indices: pt.Tensor = None,
                                 coupling_method: str = None,
                                 symmetric: bool = None,
                                 alpha_num: int = None,
                                 beta_num: int = None,
                                 metrics: LocalEnergyMetrics = None):
            assert coupling_method in self.ALLOWED_COUPLING_METHODS
            match coupling_method:
                case 'ham':
                    return self.find_sampled_and_coupled_via_ham(chunk_as_unq_batch_ptrs=chunk_as_unq_batch_ptrs,
                                                                 unq_batch_as_base_indices=unq_batch_as_base_indices,
                                                                 alpha_num=alpha_num,
                                                                 beta_num=beta_num,
                                                                 metrics=metrics)
                case 'all_to_all':
                    return self.find_sampled_and_coupled_via_all_to_all(chunk_as_unq_batch_ptrs=chunk_as_unq_batch_ptrs,
                                                                        unq_batch_as_base_indices=unq_batch_as_base_indices,
                                                                        symmetric=symmetric,
                                                                 metrics=metrics)
                case 'hamming_ball':
                    return self.find_sampled_and_coupled_via_hamming_ball(chunk_as_unq_batch_ptrs=chunk_as_unq_batch_ptrs,
                                                                          unq_batch_as_base_indices=unq_batch_as_base_indices,
                                                                 metrics=metrics)
                case 'trie':
                    return self.find_sampled_and_coupled_via_trie(chunk_as_unq_batch_ptrs=chunk_as_unq_batch_ptrs,
                                                                  unq_batch_as_base_indices=unq_batch_as_base_indices,
                                                                 metrics=metrics)
                case _:
                    raise ValueError(f'Wrong method: {coupling_method}')

    @timed
    def timed_find_a_in_b(self, a: pt.Tensor = None, b: pt.Tensor = None):
        return self.find_a_in_b(a=a,
                                b=b)

    @timed
    def compute_candidates_for_coupling_via_ham(self,
                                                chunk_as_unq_batch_ptrs: pt.Tensor = None,
                                                unq_batch_as_base_indices: pt.Tensor = None):
        chunk_size = chunk_as_unq_batch_ptrs.shape[0]
        dest_as_chunk_ptrs = pt.reshape(pt.tile(pt.unsqueeze(pt.arange(chunk_size, device=self.device),
                                                     dim=-1),
                                        (1, self.unq_xy_masks_num)),
                                (-1,))
        coupling_xy_as_unq_ham_xy_ptrs = pt.reshape(pt.tile(pt.unsqueeze(pt.arange(self.unq_xy_masks_num, device=self.device),
                                                       dim=0),
                                          (chunk_size, 1)),
                                  (-1,))

        src_as_base_indices = pt.bitwise_xor(unq_batch_as_base_indices[chunk_as_unq_batch_ptrs[dest_as_chunk_ptrs]],
                                             self.unq_xy_masks[coupling_xy_as_unq_ham_xy_ptrs])

        return dest_as_chunk_ptrs, src_as_base_indices, coupling_xy_as_unq_ham_xy_ptrs

    @timed
    def filter_candidates_for_coupling_via_ham(self,
                                               dest_as_chunk_ptrs: pt.Tensor = None,
                                               src_as_base_indices: pt.Tensor = None,
                                               coupling_xy_as_unq_ham_xy_ptrs: pt.Tensor = None,
                                               alpha_num: int = None,
                                               beta_num: int = None):
        # Filter out unphysical candidate x_primes
        ALPHA_MASK = pt.tensor(0x5555555555555555, device=self.device)
        BETA_MASK = pt.bitwise_xor(ALPHA_MASK, pt.tensor(-1, device=self.device))
        alpha_nums = self.popcount_(pt.bitwise_and(src_as_base_indices, ALPHA_MASK))
        correct_alpha_num_mask = (alpha_nums == alpha_num)
        dest_as_chunk_ptrs = dest_as_chunk_ptrs[correct_alpha_num_mask]
        coupling_xy_as_unq_ham_xy_ptrs = coupling_xy_as_unq_ham_xy_ptrs[correct_alpha_num_mask]
        src_as_base_indices = src_as_base_indices[correct_alpha_num_mask]

        beta_nums = self.popcount_(pt.bitwise_and(src_as_base_indices, BETA_MASK))
        correct_beta_num_mask = (beta_nums == beta_num)
        dest_as_chunk_ptrs = dest_as_chunk_ptrs[correct_beta_num_mask]
        coupling_xy_as_unq_ham_xy_ptrs = coupling_xy_as_unq_ham_xy_ptrs[correct_beta_num_mask]
        src_as_base_indices = src_as_base_indices[correct_beta_num_mask]

        return dest_as_chunk_ptrs, src_as_base_indices, coupling_xy_as_unq_ham_xy_ptrs

    @timed
    def find_sampled_and_coupled_via_ham(self,
                                         chunk_as_unq_batch_ptrs: pt.Tensor = None,
                                         unq_batch_as_base_indices: pt.Tensor = None,
                                         alpha_num: int = None,
                                         beta_num: int = None,
                                         metrics: LocalEnergyMetrics = None):
        (dest_as_chunk_ptrs,
         src_as_base_indices,
         coupling_xy_as_unq_ham_xy_ptrs,
         metrics.candidates_time) = self.compute_candidates_for_coupling_via_ham(chunk_as_unq_batch_ptrs=chunk_as_unq_batch_ptrs,
                                                                                        unq_batch_as_base_indices=unq_batch_as_base_indices)
        (dest_as_chunk_ptrs,
         src_as_base_indices,
         coupling_xy_as_unq_ham_xy_ptrs,
         metrics.filter_candidates_time) = self.filter_candidates_for_coupling_via_ham(dest_as_chunk_ptrs=dest_as_chunk_ptrs,
                                                                                       src_as_base_indices=src_as_base_indices,
                                                                                       coupling_xy_as_unq_ham_xy_ptrs=coupling_xy_as_unq_ham_xy_ptrs,
                                                                                       alpha_num=alpha_num,
                                                                                       beta_num=beta_num)
        metrics.candidate_x_primes_num = src_as_base_indices.shape[0]
        src_in_unq_batch_mask, srs_as_unq_batch_ptrs, metrics.find_a_in_b_time = self.timed_find_a_in_b(a=src_as_base_indices,
                                                                              b=unq_batch_as_base_indices)
        dest_as_chunk_ptrs = dest_as_chunk_ptrs[src_in_unq_batch_mask]
        src_as_unq_batch_ptrs = srs_as_unq_batch_ptrs[src_in_unq_batch_mask]
        src_as_base_indices = src_as_base_indices[src_in_unq_batch_mask]
        coupling_xy_as_unq_ham_xy_ptrs = coupling_xy_as_unq_ham_xy_ptrs[src_in_unq_batch_mask]

        return (dest_as_chunk_ptrs,
                src_as_unq_batch_ptrs,
                src_as_base_indices,
                coupling_xy_as_unq_ham_xy_ptrs, metrics)

    @timed
    def compute_candidates_for_coupling_via_all_to_all(self,
                                                       chunk_as_unq_batch_ptrs: pt.Tensor = None,
                                                       unq_batch_as_base_indices: pt.Tensor = None,
                                                       symmetric: bool = False):
        to_num = chunk_as_unq_batch_ptrs.shape[0]
        from_num = unq_batch_as_base_indices.shape[0]
        if symmetric:
            assert to_num == from_num
            #assert pt.allclose(chunk_as_unq_batch_ptrs, unq_batch_as_base_indices)
            triu_indices = pt.triu_indices(to_num, from_num, device=self.device)
            dest_as_chunk_ptrs = triu_indices[0, :]
            src_as_unq_batch_ptrs = triu_indices[1, :]
            del triu_indices
        else:
            dest_as_chunk_ptrs = pt.reshape(pt.tile(pt.unsqueeze(pt.arange(to_num, device=self.device),
                                                         dim=-1),
                                            (1, from_num)),
                                    (-1,))
            src_as_unq_batch_ptrs = pt.reshape(pt.tile(pt.unsqueeze(pt.arange(from_num, device=self.device),
                                                           dim=0),
                                              (to_num, 1)),
                                      (-1,))
        src_as_base_indices = unq_batch_as_base_indices[src_as_unq_batch_ptrs]
        coupling_xys = (unq_batch_as_base_indices[chunk_as_unq_batch_ptrs[dest_as_chunk_ptrs]]).bitwise_xor_(src_as_base_indices)

        return dest_as_chunk_ptrs, src_as_unq_batch_ptrs, src_as_base_indices, coupling_xys

    @timed
    def filter_candidates_for_coupling_via_all_to_all(self,
                                                      dest_as_chunk_ptrs: pt.Tensor = None,
                                                      src_as_unq_batch_ptrs: pt.Tensor = None,
                                                      src_as_base_indices: pt.Tensor = None,
                                                      coupling_xys: pt.Tensor = None):
        popcounts = self.popcount(coupling_xys)
        valid_excitation_mask = pt.le(popcounts, 4)

        dest_as_chunk_ptrs = dest_as_chunk_ptrs[valid_excitation_mask]
        src_as_unq_batch_ptrs = src_as_unq_batch_ptrs[valid_excitation_mask]
        src_as_base_indices = src_as_base_indices[valid_excitation_mask]
        coupling_xys = coupling_xys[valid_excitation_mask]

        return dest_as_chunk_ptrs, src_as_unq_batch_ptrs, src_as_base_indices, coupling_xys

    @timed
    def find_sampled_and_coupled_via_all_to_all(self,
                                                chunk_as_unq_batch_ptrs: pt.Tensor = None,
                                                unq_batch_as_base_indices: pt.Tensor = None,
                                                symmetric: bool = None,
                                                metrics: LocalEnergyMetrics = None):
        (dest_as_chunk_ptrs,
         src_as_unq_batch_ptrs,
         src_as_base_indices,
         coupling_xys, metrics.candidates_time) = self.compute_candidates_for_coupling_via_all_to_all(chunk_as_unq_batch_ptrs=chunk_as_unq_batch_ptrs,
                                                                             unq_batch_as_base_indices=unq_batch_as_base_indices,
                                                                             symmetric=symmetric)
        metrics.candidate_x_primes_num = src_as_base_indices.shape[0]

        (dest_as_chunk_ptrs,
         src_as_unq_batch_ptrs,
         src_as_base_indices,
         coupling_xys, metrics.filter_candidates_time) = self.filter_candidates_for_coupling_via_all_to_all(dest_as_chunk_ptrs=dest_as_chunk_ptrs,
                                                                            src_as_unq_batch_ptrs=src_as_unq_batch_ptrs,
                                                                            src_as_base_indices=src_as_base_indices,
                                                                            coupling_xys=coupling_xys)

        coupling_xy_in_unq_ham_xy_mask, coupling_xy_as_unq_ham_xy_ptrs, metrics.find_a_in_b_time = self.timed_find_a_in_b(a=coupling_xys,
                                                                                          b=self.unq_xy_masks)

        dest_as_chunk_ptrs = dest_as_chunk_ptrs[coupling_xy_in_unq_ham_xy_mask]
        src_as_unq_batch_ptrs = src_as_unq_batch_ptrs[coupling_xy_in_unq_ham_xy_mask]
        src_as_base_indices = src_as_base_indices[coupling_xy_in_unq_ham_xy_mask]
        coupling_xy_as_unq_ham_xy_ptrs = coupling_xy_as_unq_ham_xy_ptrs[coupling_xy_in_unq_ham_xy_mask]

        return (dest_as_chunk_ptrs,
                src_as_unq_batch_ptrs,
                src_as_base_indices,
                coupling_xy_as_unq_ham_xy_ptrs, metrics)

    @timed
    def find_sampled_and_coupled_via_trie(self,
                                          chunk_as_unq_batch_ptrs: pt.Tensor = None,
                                          unq_batch_as_base_indices: pt.Tensor = None,
                                 metrics: LocalEnergyMetrics = None):
        assert chunk_as_unq_batch_ptrs.shape[0] == unq_batch_as_base_indices.shape[0]
        unq_batch_trie = Trie(hilbert_space=self.hilbert_space,
                              batch_as_base_indices=unq_batch_as_base_indices)
        (dest_as_chunk_ptrs,
         src_as_unq_batch_ptrs,
         unq_ham_xy_trie_node_indices) = unq_batch_trie.compute_pairs_allowed_by_other_trie(other_trie=self.unq_ham_xy_trie)

        return (dest_as_chunk_ptrs,
                src_as_unq_batch_ptrs,
                unq_batch_as_base_indices[src_as_unq_batch_ptrs],
                self.unq_ham_xy_trie_node_idx2unq_ham_xy_ptr[unq_ham_xy_trie_node_indices], metrics)

    def find_sampled_and_coupled_via_hamming_ball(self,
                                          chunk_as_unq_batch_ptrs: pt.Tensor = None,
                                          unq_batch_as_base_indices: pt.Tensor = None,
                                                  metrics: LocalEnergyMetrics = None):
        assert chunk_as_unq_batch_ptrs.shape[0] == unq_batch_as_base_indices.shape[0]
        unq_batch_trie = Trie(hilbert_space=self.hilbert_space,
                              batch_as_base_indices=unq_batch_as_base_indices)
        (dest_as_chunk_ptrs,
         src_as_unq_batch_ptrs) = unq_batch_trie.compute_hamming_balls(hamming_dist=4)

        src_as_base_indices = unq_batch_as_base_indices[src_as_unq_batch_ptrs]
        coupling_xys = (unq_batch_as_base_indices[chunk_as_unq_batch_ptrs[dest_as_chunk_ptrs]]).bitwise_xor_(src_as_base_indices)
        metrics.candidate_x_primes_num = src_as_base_indices.shape[0]

        coupling_xy_in_unq_ham_xy_mask, coupling_xy_as_unq_ham_xy_ptrs, metrics.find_a_in_b_time = self.timed_find_a_in_b(a=coupling_xys,
                                                                                          b=self.unq_xy_masks)

        dest_as_chunk_ptrs = dest_as_chunk_ptrs[coupling_xy_in_unq_ham_xy_mask]
        src_as_unq_batch_ptrs = src_as_unq_batch_ptrs[coupling_xy_in_unq_ham_xy_mask]
        src_as_base_indices = src_as_base_indices[coupling_xy_in_unq_ham_xy_mask]
        coupling_xy_as_unq_ham_xy_ptrs = coupling_xy_as_unq_ham_xy_ptrs[coupling_xy_in_unq_ham_xy_mask]

        return (dest_as_chunk_ptrs,
                src_as_unq_batch_ptrs,
                src_as_base_indices,
                coupling_xy_as_unq_ham_xy_ptrs, metrics)

    @timed
    def compute_candidates_for_indices_coupling(self,
                                                sampled_base_idx_idx_to: pt.Tensor = None,
                                                sampled_base_idx_idx_from: pt.Tensor = None,
                                                sampled_indices: pt.Tensor = None,
                                                symmetric: bool = False):
        to_num = sampled_base_idx_idx_to.shape[0]
        from_num = sampled_base_idx_idx_from.shape[0]
        if symmetric:
            assert to_num == from_num
            assert pt.allclose(sampled_base_idx_idx_to, sampled_base_idx_idx_from)
            triu_indices = pt.triu_indices(to_num, from_num, device=self.device)
            scatter_to = triu_indices[0, :]
            scatter_from = triu_indices[1, :]
            del triu_indices
        else:
            scatter_to = pt.reshape(pt.tile(pt.unsqueeze(pt.arange(to_num, device=self.device),
                                                         dim=-1),
                                            (1, from_num)),
                                    (-1,))
            scatter_from = pt.reshape(pt.tile(pt.unsqueeze(pt.arange(from_num, device=self.device),
                                                           dim=0),
                                              (to_num, 1)),
                                      (-1,))
        coupling_xy_masks = (sampled_indices[sampled_base_idx_idx_to[scatter_to]]).bitwise_xor_(sampled_indices[sampled_base_idx_idx_from[scatter_from]])

        return scatter_to, scatter_from, coupling_xy_masks

    @timed
    def filter_candidates_for_indices_coupling(self,
                                               coupling_xy_masks: pt.Tensor = None,
                                               scatter_to: pt.Tensor = None,
                                               scatter_from: pt.Tensor = None):
        # Filter out candidate xy_masks which have wrong number of excitations
        popcounts = self.popcount(coupling_xy_masks)
        valid_ex_mask = pt.le(popcounts, 4)

        scatter_to = scatter_to[valid_ex_mask]
        scatter_from = scatter_from[valid_ex_mask]
        coupling_xy_masks = coupling_xy_masks[valid_ex_mask]

        # Filter out candidate xy_masks which do not belong to the Hamiltonian
        ham_xy_pointers, coupling_xy_is_ham_xy = self.compute_ham_xy_pointers(coupling_xys=coupling_xy_masks)
        scatter_to = scatter_to[coupling_xy_is_ham_xy]
        scatter_from = scatter_from[coupling_xy_is_ham_xy]

        return scatter_to, scatter_from, ham_xy_pointers

    @timed
    def scatter_sampled_for_indices_coupling(self,
                                             scatter_to: pt.Tensor = None,
                                             scatter_from: pt.Tensor = None,
                                             ham_elements: pt.Tensor = None,
                                             sampled_base_idx_idx_to: pt.Tensor = None,
                                             sampled_base_idx_idx_from: pt.Tensor = None,
                                             to_num: int = None,
                                             sampled_amps: pt.Tensor = None,
                                             symmetric: bool = True):
        if symmetric:
            # IMPORTANT: Might have fucked up here
            ham_elements[scatter_to == scatter_from] = ham_elements[scatter_to == scatter_from] / 2

            from_amps = pt.cat((sampled_amps[sampled_base_idx_idx_from[scatter_from]],
                                     sampled_amps[sampled_base_idx_idx_to[scatter_to]]), dim=0)
            eloc = pt.zeros((to_num,), dtype=ham_elements.dtype, device=self.device)
            eloc = eloc.scatter_add_(dim=0,
                                     index=pt.cat((scatter_to, scatter_from), dim=0),
                                     src=pt.cat((ham_elements, pt.conj(ham_elements)), dim=0) * from_amps)
        else:
            from_amps = sampled_amps[sampled_base_idx_idx_from[scatter_from]]
            eloc = pt.zeros((to_num, ), dtype=ham_elements.dtype, device=self.device)
            eloc = eloc.scatter_add_(dim=0,
                                     index=scatter_to,
                                     src=ham_elements * from_amps)

        return eloc

    def sort_indices(self, indices):
        sorted_indices = indices
        for int_idx in range(self.hilbert_space.int_per_idx):
            _, sort_inv = pt.sort(sorted_indices[:, int_idx], stable=True, descending=False)
            sorted_indices = sorted_indices[sort_inv]

            neg_mask = sorted_indices[:, int_idx] < 0
            arange = pt.arange(sorted_indices.shape[0], device=self.device)
            neg_indices = arange[neg_mask]
            non_neg_indices = arange[~neg_mask]

            sorted_indices = pt.cat((sorted_indices[non_neg_indices], sorted_indices[neg_indices]), dim=0)

        return sorted_indices

    def compute_unique_composite_bits(self, composite_bits):
        assert len(composite_bits.shape) == 2

        prev_unq, prev_unq_inv = pt.unique(composite_bits[..., 0], return_inverse=True)
        prev_unq = pt.reshape(prev_unq, (-1, 1))
        for int_idx in range(1, 2):
            new_unq, new_unq_inv = pt.unique(composite_bits[..., int_idx], return_inverse=True)
            prev_unq, prev_unq_inv = self.hilbert_space.two_unique2cat_unique(unq_1=prev_unq,
                                                                              unq_inv_1=prev_unq_inv,
                                                                              unq_2=new_unq,
                                                                              unq_inv_2=new_unq_inv)

        return prev_unq, prev_unq_inv

    def construct_tree(self, indices):
        base_vec = self.base_idx2base_vec(indices)
        base_vec = pt.flip(base_vec, dims=(-1,))

        tree = []
        parents_to_children = []
        prev_layer_parent_idx = pt.zeros_like(base_vec[:, 0])
        base_vec_as_node_idx = []
        for qubit_idx in range(self.qubit_num):
            composite_bits = pt.stack((base_vec[:, qubit_idx], prev_layer_parent_idx), dim=1)
            unq_composite_bits, unq_composite_bits_inv = self.compute_unique_composite_bits(composite_bits)
            base_vec_as_node_idx.append(unq_composite_bits_inv)
            tree.append(unq_composite_bits)

            prev_layer_parent_idx = unq_composite_bits_inv
            cur_parents_to_children = -1 * pt.ones_like(tree[qubit_idx - 1])
            cur_parents_to_children[unq_composite_bits[:, 1], unq_composite_bits[:, 0]] = pt.arange(
                unq_composite_bits.shape[0], device=self.device)
            parents_to_children.append(cur_parents_to_children)
        # parents_to_children[0] = pt.tensor([[0, 1]], device=self.device)
        base_vec_as_node_idx = pt.stack(base_vec_as_node_idx, dim=-1)

        return base_vec, tree, parents_to_children, base_vec_as_node_idx

    def compute_scatter_to_and_from_from_tree(self,
                                              sampled_indices,
                                              base_vec,
                                              tree,
                                              parents_to_children,
                                              base_vec_as_node_idx):
        scatter_to = pt.arange(sampled_indices.shape[0], device=self.device)
        scatter_from = pt.zeros_like(scatter_to)
        budgets = 4 * pt.ones_like(scatter_to)
        pt.cuda.synchronize()
        for qubit_idx in range(self.qubit_num):
            scatter_to = pt.tile(pt.unsqueeze(scatter_to, dim=-1), (1, 2))
            budgets = pt.tile(pt.unsqueeze(budgets, dim=-1), (1, 2))
            scatter_from = parents_to_children[qubit_idx][scatter_from]

            scatter_to = pt.reshape(scatter_to, (-1,))
            budgets = pt.reshape(budgets, (-1,))
            scatter_from = pt.reshape(scatter_from, (-1,))
            scatter_from_mask = (scatter_from != -1)

            scatter_to = scatter_to[scatter_from_mask]
            budgets = budgets[scatter_from_mask]
            scatter_from = scatter_from[scatter_from_mask]
            cur_bits = base_vec[scatter_to, qubit_idx]
            cand_bits = tree[qubit_idx][scatter_from, 0]

            bit_eq_mask = cur_bits == cand_bits
            # budgets[bit_neq_mask].sub_(1)
            budgets = pt.where(bit_eq_mask,
                               budgets,
                               budgets - 1)

            non_neg_budget_mask = (budgets >= 0).logical_and_(
                (base_vec_as_node_idx[scatter_to, qubit_idx].greater_equal_(scatter_from)))

            scatter_to = pt.masked_select(scatter_to, non_neg_budget_mask)
            scatter_from = pt.masked_select(scatter_from, non_neg_budget_mask)
            budgets = pt.masked_select(budgets, non_neg_budget_mask)

        return scatter_to, scatter_from

    @pt.no_grad()
    def compute_local_energies_via_indices_coupling(self,
                                                    sampled_base_idx_idx_to: pt.Tensor = None,
                                                    sampled_base_idx_idx_from: pt.Tensor = None,
                                                    sampled_indices: pt.Tensor = None,
                                                    sampled_amps: pt.Tensor = None,
                                                    use_tree_for_candidates: bool = False,
                                                    symmetric: bool = False) -> Tuple[pt.Tensor, pt.Tensor, LocalEnergyMetrics]:
        metrics = LocalEnergyMetrics()
        if use_tree_for_candidates:
            to_num = sampled_indices.shape[0]
            start_time = time.time()
            base_vec, tree, parents_to_children, base_vec_as_node_idx = self.construct_tree(sampled_indices)

            scatter_to, scatter_from = self.compute_scatter_to_and_from_from_tree(sampled_indices, base_vec, tree, parents_to_children, base_vec_as_node_idx)
            end_time = time.time()
            metrics.candidates_time = 0.0
            metrics.filter_candidates_time = end_time - start_time
            coupling_xy_masks = pt.bitwise_xor(sampled_indices[scatter_to], sampled_indices[scatter_from])

            ham_xy_pointers, coupling_xy_is_ham_xy = self.compute_ham_xy_pointers(coupling_xys=coupling_xy_masks)
            scatter_to = scatter_to[coupling_xy_is_ham_xy]
            scatter_from = scatter_from[coupling_xy_is_ham_xy]
            coupling_xy_masks = None
            del coupling_xy_masks
        else:
            to_num = sampled_base_idx_idx_to.shape[0]
            scatter_to, scatter_from, coupling_xy_masks, metrics.candidates_time = self.compute_candidates_for_indices_coupling(sampled_base_idx_idx_to=sampled_base_idx_idx_to,
                                                                                                                                sampled_base_idx_idx_from=sampled_base_idx_idx_from,
                                                                                                                                sampled_indices=sampled_indices,
                                                                                                                                symmetric=symmetric)
            metrics.candidate_x_primes_num = scatter_to.shape[0]

            scatter_to, scatter_from, ham_xy_pointers, metrics.filter_candidates_time = self.filter_candidates_for_indices_coupling(coupling_xy_masks=coupling_xy_masks,
                                                                                                                                    scatter_to=scatter_to,
                                                                                                                                    scatter_from=scatter_from)
        metrics.sampled_x_primes_num = scatter_to.shape[0]

        ham_elements, ham_yzs_num, metrics.sampled_matrix_elements_time = self.compute_matrix_elements(x_primes=sampled_indices[sampled_base_idx_idx_from[scatter_from]],
                                                                                                       ham_xy_pointers=ham_xy_pointers)
        metrics.sampled_coupled_yz_num = ham_yzs_num

        eloc, metrics.sampled_scatter_time = self.scatter_sampled_for_indices_coupling(scatter_to=scatter_to,
                                                                                       scatter_from=scatter_from,
                                                                                       ham_elements=ham_elements,
                                                                                       sampled_base_idx_idx_to=sampled_base_idx_idx_to,
                                                                                       sampled_base_idx_idx_from=sampled_base_idx_idx_from,
                                                                                       to_num=to_num,
                                                                                       sampled_amps=sampled_amps,
                                                                                       symmetric=symmetric)

        eloc = eloc / sampled_amps[sampled_base_idx_idx_to]

        return eloc, eloc, metrics

    @timed
    def compute_candidates_for_ham_xy_coupling(self,
                                               dest_base_idx_ptrs: pt.Tensor = None,
                                               sampled_indices: pt.Tensor = None):
        to_num = dest_base_idx_ptrs.shape[0]
        scatter_to = pt.reshape(pt.tile(pt.unsqueeze(pt.arange(to_num, device=self.device),
                                                     dim=-1),
                                        (1, self.unq_xy_masks_num)),
                                (-1,))
        ham_pointers = pt.reshape(pt.tile(pt.unsqueeze(pt.arange(self.unq_xy_masks_num, device=self.device),
                                                       dim=0),
                                          (to_num, 1)),
                                  (-1,))
        x_primes = pt.bitwise_xor(sampled_indices[dest_base_idx_ptrs[scatter_to]],
                                  self.unq_xy_masks[ham_pointers])

        return scatter_to, x_primes, ham_pointers

    @timed
    def filter_candidates_for_ham_xy_coupling(self,
                                              scatter_to: pt.Tensor = None,
                                              x_primes: pt.Tensor = None,
                                              ham_pointers: pt.Tensor = None,
                                              wf: AbstractQuantumState = None):
        # Filter out unphysical candidate x_primes
        ALPHA_MASK = pt.tensor(0x5555555555555555, device=self.device)
        BETA_MASK = pt.bitwise_xor(ALPHA_MASK, pt.tensor(-1, device=self.device))
        alpha_nums = self.popcount_(pt.bitwise_and(x_primes, ALPHA_MASK))
        correct_alpha_num_mask = (alpha_nums == (wf.masker.symmetries[0].particle_num // 2))
        scatter_to = scatter_to[correct_alpha_num_mask]
        ham_pointers = ham_pointers[correct_alpha_num_mask]
        x_primes = x_primes[correct_alpha_num_mask]

        beta_nums = self.popcount_(pt.bitwise_and(x_primes, BETA_MASK))
        correct_beta_num_mask = (beta_nums == (wf.masker.symmetries[0].particle_num // 2))
        scatter_to = scatter_to[correct_beta_num_mask]
        ham_pointers = ham_pointers[correct_beta_num_mask]
        x_primes = x_primes[correct_beta_num_mask]

        return scatter_to, x_primes, ham_pointers

    @pt.no_grad()
    def compute_local_energies_via_ham_xy_coupling(self,
                                                   sampled_base_idx_idx_to: pt.Tensor = None,
                                                   sampled_indices: pt.Tensor = None,
                                                   sampled_amps: pt.Tensor = None,
                                                   wf: AbstractQuantumState = None,
                                                   sample_aware: bool = False,
                                                   amps_chunk_size: int = 100000) -> Tuple[pt.Tensor, pt.Tensor, LocalEnergyMetrics]:
        metrics = LocalEnergyMetrics()
        to_num = sampled_base_idx_idx_to.shape[0]

        scatter_to, x_primes, pointers, metrics.candidates_time = self.compute_candidates_for_ham_xy_coupling(
            dest_base_idx_ptrs=sampled_base_idx_idx_to,
            sampled_indices=sampled_indices)
        metrics.candidate_x_primes_num = x_primes.shape[0]

        scatter_to, x_primes, pointers, metrics.filter_candidates_time = self.filter_candidates_for_ham_xy_coupling(scatter_to=scatter_to,
                                                                                                                        x_primes=x_primes,
                                                                                                                        ham_pointers=pointers,
                                                                                                                        wf=wf)
        x_primes_num = x_primes.shape[0]

        # We find unique indices among a concatenation between the candidate x_primes and sampled indices
        # Then, we figure out which x_primes were sampled and what is their position in the array of all sampled indices
        unq_or_sampled_x_primes, unq_or_sampled_x_primes_inv = self.hilbert_space.compute_unique_indices(pt.cat((x_primes, sampled_indices)))
        unq_or_sampled_x_prime_is_sampled = pt.zeros((unq_or_sampled_x_primes.shape[0]),
                                                     dtype=pt.bool,
                                                     device=self.device)
        unq_or_sampled_x_prime_is_sampled.scatter_add_(dim=0,
                                                       index=unq_or_sampled_x_primes_inv[x_primes_num:],
                                                       src=pt.ones((sampled_indices.shape[0],),
                                                                    dtype=pt.bool,
                                                                    device=self.device))
        x_prime_is_sampled = unq_or_sampled_x_prime_is_sampled[unq_or_sampled_x_primes_inv[:x_primes_num]]

        unq_or_sampled_x_prime_sampled_idx = pt.zeros((unq_or_sampled_x_primes.shape[0]),
                                                      dtype=pt.int64,
                                                      device=self.device)
        unq_or_sampled_x_prime_sampled_idx.scatter_add_(dim=0,
                                                        index=unq_or_sampled_x_primes_inv[x_primes_num:],
                                                        src=pt.arange(sampled_indices.shape[0],
                                                                      dtype=pt.int64,
                                                                      device=self.device))
        sampled_x_prime_sampled_idx = unq_or_sampled_x_prime_sampled_idx[unq_or_sampled_x_primes_inv[:x_primes_num]]

        sampled_x_primes = x_primes[x_prime_is_sampled]
        sampled_scatter_to = scatter_to[x_prime_is_sampled]
        sampled_x_prime_sampled_idx = sampled_x_prime_sampled_idx[x_prime_is_sampled]
        sampled_ham_pointers = pointers[x_prime_is_sampled]
        metrics.sampled_unq_x_primes_num = unq_or_sampled_x_prime_is_sampled.sum().item()
        metrics.sampled_x_primes_num = sampled_x_primes.shape[0]

        sampled_ham_elements, sampled_ham_yz_num, metrics.sampled_matrix_elements_time = self.compute_matrix_elements(x_primes=sampled_x_primes,
                                                                                                                      ham_xy_pointers=sampled_ham_pointers)
        metrics.sampled_coupled_yz_num = sampled_ham_yz_num

        sampled_scatter_start_time = time.time()
        sampled_eloc = pt.zeros((to_num,), dtype=sampled_ham_elements.dtype, device=self.device)
        sampled_from_amps = sampled_amps[sampled_x_prime_sampled_idx]
        sampled_eloc = sampled_eloc.scatter_add_(dim=0,
                                         index=sampled_scatter_to,
                                         src=sampled_ham_elements * sampled_from_amps)
        sampled_scatter_end_time = time.time()
        metrics.sampled_scatter_time = sampled_scatter_end_time - sampled_scatter_start_time

        sampled_eloc = sampled_eloc / sampled_amps[sampled_base_idx_idx_to]

        if sample_aware:
            return sampled_eloc, sampled_eloc, metrics
        else:
            non_sampled_x_primes = x_primes[~x_prime_is_sampled]
            non_sampled_scatter_to = scatter_to[~x_prime_is_sampled]
            non_sampled_ham_pointers = pointers[~x_prime_is_sampled]
            non_sampled_ham_elements, non_sampled_yz_num, metrics.non_sampled_matrix_elements_time = self.compute_matrix_elements(x_primes=non_sampled_x_primes,
                                                                                                                                  ham_xy_pointers=non_sampled_ham_pointers)
            metrics.non_sampled_x_primes_num = non_sampled_x_primes.shape[0]
            metrics.non_sampled_coupled_yz_num = non_sampled_yz_num

            non_sampled_eloc = pt.zeros((to_num,), dtype=sampled_ham_elements.dtype, device=self.device)

            unq_non_sampled_x_primes = unq_or_sampled_x_primes[~unq_or_sampled_x_prime_is_sampled]
            metrics.non_sampled_unq_x_primes_num = unq_non_sampled_x_primes.shape[0]
            dummy_inv = pt.zeros((unq_or_sampled_x_primes.shape[0]),
                                 dtype=pt.int64,
                                 device=self.device)
            dummy_inv[~unq_or_sampled_x_prime_is_sampled] = pt.arange(unq_non_sampled_x_primes.shape[0],
                                                                      device=self.device)
            unq_non_sampled_x_primes_inv = dummy_inv[unq_or_sampled_x_primes_inv[:x_primes_num]]
            unq_non_sampled_x_primes_inv = unq_non_sampled_x_primes_inv[~x_prime_is_sampled]

            eval_non_sampled_amps_start_time = time.time()
            unq_non_sampled_amps = []
            amps_chunk_boundaries = compute_chunk_boundaries(array_len=unq_non_sampled_x_primes.shape[0],
                                                             chunk_size=amps_chunk_size)
            chunk_num = len(amps_chunk_boundaries) - 1
            for chunk_idx in range(chunk_num):
                chunk_start = amps_chunk_boundaries[chunk_idx]
                chunk_end = amps_chunk_boundaries[chunk_idx + 1]
                unq_non_sampled_amps.append(wf.amplitude(unq_non_sampled_x_primes[chunk_start:chunk_end]))
            unq_non_sampled_amps = pt.cat(unq_non_sampled_amps, dim=0)
            non_sampled_amps = unq_non_sampled_amps[unq_non_sampled_x_primes_inv]
            eval_non_sampled_amps_end_time = time.time()
            metrics.eval_non_sampled_amps_time = eval_non_sampled_amps_end_time - eval_non_sampled_amps_start_time

            non_sampled_scatter_start_time = time.time()
            non_sampled_eloc.scatter_add_(dim=0,
                              index=non_sampled_scatter_to,
                              src=non_sampled_ham_elements * non_sampled_amps)
            non_sampled_scatter_end_time = time.time()
            metrics.non_sampled_scatter_time = non_sampled_scatter_end_time - non_sampled_scatter_start_time

            non_sampled_eloc = non_sampled_eloc / sampled_amps[sampled_base_idx_idx_to]

            return non_sampled_eloc + sampled_eloc, sampled_eloc, metrics
