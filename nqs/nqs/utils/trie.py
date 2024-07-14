from __future__ import annotations

import torch as pt

from ..base.abstract_hilbert_space_object import AbstractHilbertSpaceObject


class Trie(AbstractHilbertSpaceObject):
    def __init__(self,
                 *args,
                 batch_as_base_indices: pt.Tensor = None,
                 **kwargs):
        super(Trie, self).__init__(*args, **kwargs)

        self.batch_as_base_indices = batch_as_base_indices

        self.lvls, self.co_lvls, self.batch_as_base_vecs, self.base_vec2all_lvl_node_indices = self.construct()

    def compute_unique_cur_lvl_nodes(self, cur_lvl_nodes):
        assert len(cur_lvl_nodes.shape) == 2

        unq_node_bits, unq_node_bits_inv = pt.unique(cur_lvl_nodes[..., 0], return_inverse=True)
        unq_node_parents, unq_node_parents_inv = pt.unique(cur_lvl_nodes[..., 1], return_inverse=True)

        unq_nodes, unq_nodes_inv = self.hilbert_space.two_unique2cat_unique(unq_1=unq_node_bits,
                                                                            unq_inv_1=unq_node_bits_inv,
                                                                            unq_2=unq_node_parents,
                                                                            unq_inv_2=unq_node_parents_inv)

        return unq_nodes, unq_nodes_inv

    def construct(self):
        base_vec = self.base_idx2base_vec(self.batch_as_base_indices)
        base_vec = pt.flip(base_vec, dims=(-1,))

        lvls = [pt.tensor([[-1, -1]], device=self.device)]
        co_lvls = []

        cur_lvl_node_parents = pt.zeros_like(base_vec[:, 0])
        base_vec2cur_lvl_node_idx = []

        for qubit_idx in range(1, self.qubit_num + 1):
            # Evaluating next level of trie
            cur_lvl_nodes = pt.stack((base_vec[:, qubit_idx - 1], cur_lvl_node_parents), dim=1)
            cur_lvl_unq_nodes, cur_lvl_unq_nodes_inv = self.compute_unique_cur_lvl_nodes(cur_lvl_nodes)
            base_vec2cur_lvl_node_idx.append(cur_lvl_unq_nodes_inv)
            lvls.append(cur_lvl_unq_nodes)
            cur_lvl_node_parents = cur_lvl_unq_nodes_inv

            # Evaluating next level of co-trie
            cur_lvl_node_parents2children_nodes = -1 * pt.ones_like(lvls[qubit_idx - 1])

            cur_lvl_node_parents2children_nodes[cur_lvl_unq_nodes[:, 1], cur_lvl_unq_nodes[:, 0]] = pt.arange(
                cur_lvl_unq_nodes.shape[0], device=self.device)
            co_lvls.append(cur_lvl_node_parents2children_nodes)
        base_vec2all_lvl_node_indices = pt.stack(base_vec2cur_lvl_node_idx, dim=-1)

        return lvls[1:], co_lvls, base_vec, base_vec2all_lvl_node_indices

    def compute_hamming_balls(self, hamming_dist: int = 4):
        scatter_to = pt.arange(self.batch_as_base_indices.shape[0], device=self.device)
        scatter_from = pt.zeros_like(scatter_to)
        budgets = hamming_dist * pt.ones_like(scatter_to)

        for qubit_idx in range(self.qubit_num):
            scatter_to = pt.tile(pt.unsqueeze(scatter_to, dim=-1), (1, 2))
            budgets = pt.tile(pt.unsqueeze(budgets, dim=-1), (1, 2))
            scatter_from = self.co_lvls[qubit_idx][scatter_from]

            scatter_to = pt.reshape(scatter_to, (-1,))
            budgets = pt.reshape(budgets, (-1,))
            scatter_from = pt.reshape(scatter_from, (-1,))
            scatter_from_mask = (scatter_from != -1)

            scatter_to = scatter_to[scatter_from_mask]
            budgets = budgets[scatter_from_mask]
            scatter_from = scatter_from[scatter_from_mask]
            cur_bits = self.batch_as_base_vecs[scatter_to, qubit_idx]
            cand_bits = self.lvls[qubit_idx][scatter_from, 0]

            bit_eq_mask = (cur_bits == cand_bits)
            budgets = pt.where(bit_eq_mask,
                               budgets,
                               budgets - 1)

            non_neg_budget_mask = (budgets >= 0).logical_and_((self.base_vec2all_lvl_node_indices[scatter_to, qubit_idx].greater_equal_(scatter_from)))

            scatter_to = pt.masked_select(scatter_to, non_neg_budget_mask)
            scatter_from = pt.masked_select(scatter_from, non_neg_budget_mask)
            budgets = pt.masked_select(budgets, non_neg_budget_mask)

        return scatter_to, scatter_from

    def compute_pairs_allowed_by_other_trie(self, other_trie: Trie = None):
        frst_elem_cur_lvl_node_idx = pt.arange(self.batch_as_base_indices.shape[0], device=self.device)
        scnd_elem_cur_lvl_node_idx = pt.zeros_like(frst_elem_cur_lvl_node_idx)
        other_trie_cur_lvl_node_idx = pt.zeros_like(frst_elem_cur_lvl_node_idx)

        for qubit_idx in range(self.qubit_num):
            cur_bits = pt.stack((self.batch_as_base_vecs[frst_elem_cur_lvl_node_idx, qubit_idx],
                                 (self.batch_as_base_vecs[frst_elem_cur_lvl_node_idx, qubit_idx] + 1) % 2),
                                dim=-1)
            frst_elem_cur_lvl_node_idx = pt.tile(pt.unsqueeze(frst_elem_cur_lvl_node_idx, dim=-1), (1, 2))
            scnd_elem_cur_lvl_node_idx = self.co_lvls[qubit_idx][scnd_elem_cur_lvl_node_idx]

            other_trie_cur_lvl_node_idx = pt.gather(other_trie.co_lvls[qubit_idx][other_trie_cur_lvl_node_idx], -1, cur_bits)

            frst_elem_cur_lvl_node_idx = pt.reshape(frst_elem_cur_lvl_node_idx, (-1,))
            scnd_elem_cur_lvl_node_idx = pt.reshape(scnd_elem_cur_lvl_node_idx, (-1,))
            this_trie_child_exists_mask = (scnd_elem_cur_lvl_node_idx != -1)

            other_trie_cur_lvl_node_idx = pt.reshape(other_trie_cur_lvl_node_idx, (-1,))
            other_trie_child_exists_mask = (other_trie_cur_lvl_node_idx != -1)

            mask = pt.logical_and(this_trie_child_exists_mask, other_trie_child_exists_mask)

            # Make sure we avoid duplicate pairs
            mask = mask.logical_and_(
                (self.base_vec2all_lvl_node_indices[frst_elem_cur_lvl_node_idx, qubit_idx].less_equal_(scnd_elem_cur_lvl_node_idx)))

            frst_elem_cur_lvl_node_idx = frst_elem_cur_lvl_node_idx[mask]
            scnd_elem_cur_lvl_node_idx = scnd_elem_cur_lvl_node_idx[mask]
            other_trie_cur_lvl_node_idx = other_trie_cur_lvl_node_idx[mask]

        return frst_elem_cur_lvl_node_idx, scnd_elem_cur_lvl_node_idx, other_trie_cur_lvl_node_idx

