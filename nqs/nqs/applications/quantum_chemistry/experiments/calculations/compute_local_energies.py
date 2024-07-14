import numpy as np
import torch as pt

from .....infrastructure.nested_data import Config

from .....stochastic.ansatzes.anqs import AbstractANQS
from .....stochastic.observables.pauli_observable import PauliObservable

from .sample import SamplingResult

from .....utils.misc import compute_chunk_boundaries, verbosify_generator


class LocalEnergyCalculationConfig(Config):
    FIELDS = (
        'use_theor_freqs',
        'use_tree_for_candidates',
        'sampled_indices_chunk_size',
        'amps_chunk_size',
        'code_version',
        'matrix_element_chunk_size',
    )
    ALLOWED_CODE_VERSIONS = ('old', 'new')

    def __init__(self,
                 *args,
                 use_theor_freqs: bool = True,
                 use_tree_for_candidates: str = 'all_to_all',
                 sampled_indices_chunk_size: int = 20000,
                 amps_chunk_size: int = 100000,
                 code_version: str = 'new',
                 matrix_element_chunk_size: int = np.inf,
                 **kwargs):
        self.use_theor_freqs = use_theor_freqs

        self.use_tree_for_candidates = use_tree_for_candidates

        self.sampled_indices_chunk_size = sampled_indices_chunk_size
        self.amps_chunk_size = amps_chunk_size
        self.matrix_element_chunk_size = matrix_element_chunk_size

        assert code_version in self.ALLOWED_CODE_VERSIONS
        self.code_version = code_version

        super().__init__(*args, **kwargs)


class MonteCarloEstimator:
    def __init__(self,
                 *args,
                 values: pt.Tensor = None,
                 counts: pt.Tensor = None,
                 **kwargs):
        self.values = values
        self.freqs = counts / pt.sum(counts)

        if (self.values is not None) and (self.freqs is not None):
            self.mean = pt.dot(self.values, self.freqs)
            self.var = pt.dot(pt.pow(self.values - self.mean, 2), self.freqs)
        else:
            self.mean = None
            self.var = None


class LocalEnergyResult:
    def __init__(self,
                 *args,
                 full_e_loc_mc_est: MonteCarloEstimator = None,
                 sample_aware_e_loc_mc_est: MonteCarloEstimator = None,
                 **kwargs):
        self.full_e_loc_mc_est = full_e_loc_mc_est
        self.sample_aware_e_loc_mc_est = sample_aware_e_loc_mc_est


@pt.no_grad()
def compute_local_energies(wf: AbstractANQS = None,
                           sampling_result: SamplingResult = None,
                           sampled_amps: pt.Tensor = None,
                           ham: PauliObservable = None,
                           config: LocalEnergyCalculationConfig = None,
                           sample_aware: bool = True,
                           verbose: bool = False):
    config = config if config is not None else LocalEnergyCalculationConfig()
    if config.code_version == 'old':
        return compute_old_local_energies(wf=wf,
                                   sampling_result=sampling_result,
                                   sampled_amps=sampled_amps,
                                   ham=ham,
                                   config=config,
                                   sample_aware=sample_aware,
                                   verbose=verbose)
    elif config.code_version == 'new':
        if sample_aware is False:
            raise NotImplementedError(f'There is no new code version for conventional local energy calculations')
        else:
            sampled_indices = sampling_result.indices
            alpha_num = wf.masker.symmetries[0].particle_num // 2
            beta_num = wf.masker.symmetries[0].particle_num // 2

            full_local_energies, sample_aware_local_energies, metrics = ham.compute_var_local_energy_proxy(unq_batch_as_base_indices=sampled_indices,
                                                                                                           unq_batch_as_amps=sampled_amps,
                                                                                                           coupling_method=config.use_tree_for_candidates,
                                                                                                           chunk_size=config.sampled_indices_chunk_size,
                                                                                                           alpha_num=alpha_num,
                                                                                                           beta_num=beta_num,
                                                                                                           matrix_element_chunk_size=config.matrix_element_chunk_size)
            theor_freqs = pt.conj(sampled_amps) * sampled_amps
            theor_freqs = theor_freqs / pt.sum(theor_freqs)

            return LocalEnergyResult(full_e_loc_mc_est=MonteCarloEstimator(values=full_local_energies,
                                                                           counts=theor_freqs if config.use_theor_freqs else sampling_result.counts),
                                     sample_aware_e_loc_mc_est=MonteCarloEstimator(values=sample_aware_local_energies,
                                                                                   counts=theor_freqs)), metrics
    else:
        raise ValueError(f'Wrong local energy code version: {config.code_version}')

@pt.no_grad()
def compute_old_local_energies(wf: AbstractANQS = None,
                           sampling_result: SamplingResult = None,
                           sampled_amps: pt.Tensor = None,
                           ham: PauliObservable = None,
                           config: LocalEnergyCalculationConfig = None,
                           sample_aware: bool = True,
                           verbose: bool = False):
    config = config if config is not None else LocalEnergyCalculationConfig()
    sampled_indices = sampling_result.indices
    if config.use_tree_for_candidates == 'trie':
        assert sample_aware is True
        full_local_energies, sample_aware_local_energies, metrics = ham.compute_local_energies(wf=wf,
                                                                                               sampled_indices=sampled_indices,
                                                                                               sampled_amps=sampled_amps,
                                                                                               verbose=verbose,
                                                                                               sample_aware=sample_aware,
                                                                                               use_tree_for_candidates=True,
                                                                                               chunk_size=config.sampled_indices_chunk_size,
                                                                                               compute_via_ham_xy_coupling=False)
    elif config.use_tree_for_candidates == 'all_to_all':
        full_local_energies, sample_aware_local_energies, metrics = ham.compute_local_energies(wf=wf,
                                                                                               sampled_indices=sampled_indices,
                                                                                               sampled_amps=sampled_amps,
                                                                                               verbose=verbose,
                                                                                               sample_aware=sample_aware,
                                                                                               use_tree_for_candidates=False,
                                                                                               chunk_size=config.sampled_indices_chunk_size,
                                                                                               compute_via_ham_xy_coupling=False)
    elif config.use_tree_for_candidates == 'ham':
        full_local_energies, sample_aware_local_energies, metrics = ham.compute_local_energies(wf=wf,
                                                                                               sampled_indices=sampled_indices,
                                                                                               sampled_amps=sampled_amps,
                                                                                               verbose=verbose,
                                                                                               sample_aware=sample_aware,
                                                                                               chunk_size=config.sampled_indices_chunk_size,
                                                                                               compute_via_ham_xy_coupling=True)
    else:
        raise ValueError(f'Wrong coupling mode: {config.use_tree_for_candidates}')

    theor_freqs = pt.conj(sampled_amps) * sampled_amps
    theor_freqs = theor_freqs / pt.sum(theor_freqs)

    return LocalEnergyResult(full_e_loc_mc_est=MonteCarloEstimator(values=full_local_energies,
                                                                   counts=theor_freqs if config.use_theor_freqs else sampling_result.counts),
                             sample_aware_e_loc_mc_est=MonteCarloEstimator(values=sample_aware_local_energies,
                                                                           counts=theor_freqs)), metrics
