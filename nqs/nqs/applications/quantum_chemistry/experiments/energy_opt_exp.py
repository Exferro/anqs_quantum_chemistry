import os
import logging

import numpy as np
import torch as pt
import time


import sys


# Function to detect if the environment is Jupyter
def in_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # Check if not notebook
            return False
    except Exception:
        return False
    return True


# Import the appropriate tqdm based on the environment
if in_notebook():
    from tqdm.notebook import tqdm as env_dependent_tqdm
else:
    from tqdm import tqdm as env_dependent_tqdm
# import tqdm
# from tqdm.notebook import tqdm as tqdm_notebook

import pickle

import pandas as pd

from ....infrastructure.nested_data import Config, Schedule
from ....infrastructure import create_dir
from ....infrastructure.timed_decorator import timed

from .. import CHEMICAL_ACCURACY
from ..molecule import MolConfig, MolInitConfig, Molecule
from .preparation import create_mol

from ....base import HilbertSpace
from ....stochastic.observables.pauli_observable import LocalEnergyMetrics, PauliObservable

from .preparation import MaskerConfig, create_masker
from .preparation import MetaAnsatzConfig, create_ansatz
from .preparation import OptConfig, create_opt

from .calculations import SamplingConfig, SamplingResult, sample
from .calculations import LocalEnergyCalculationConfig, LocalEnergyResult, compute_local_energies
from .calculations import ProcessGradConfig, ProcessGradMetrics, process_grad

from ..experiments import bin_search_schedule

from typing import Tuple


class SamplingMetrics(Config):
    FIELDS = (
        'sample_num',
        'unq_num',
        'actual_unq_num',
        'repetition_num',
        'next_rep_sample_num',
        'sampling_time',
    )
    OPTIONAL_FIELDS = (
        'sample_num',
        'unq_num',
        'actual_unq_num',
        'repetition_num',
        'next_rep_sample_num',
        'sampling_time',
    )

    def __init__(self,
                 *args,
                 sample_num: float = np.nan,
                 unq_num: float = np.nan,
                 actual_unq_num: float = np.nan,
                 repetition_num: float = np.nan,
                 next_rep_sample_num: float = np.nan,
                 sampling_time: float = np.nan,
                 **kwargs):
        self.sample_num = sample_num
        self.unq_num = unq_num
        self.actual_unq_num = actual_unq_num
        self.repetition_num = repetition_num
        self.next_rep_sample_num = next_rep_sample_num

        self.sampling_time = sampling_time

        super().__init__(*args, **kwargs)


class EvalAmpsMetrics(Config):
    FIELDS = (
        'max_amp',
        'min_amp',
        'sampled_prob',
        'eval_amps_time',
        'ipr',
        'renorm_ipr',
    )
    OPTIONAL_FIELDS = (
        'max_amp',
        'min_amp',
        'sampled_prob',
        'eval_amps_time',
    )

    def __init__(self,
                 *args,
                 max_amp: float = np.nan,
                 min_amp: float = np.nan,
                 sampled_prob: float = np.nan,
                 eval_amps_time: float = np.nan,
                 ipr: float = np.nan,
                 renorm_ipr: float = np.nan,
                 **kwargs):
        self.max_amp = max_amp
        self.min_amp = min_amp
        self.sampled_prob = sampled_prob

        self.eval_amps_time = eval_amps_time

        self.ipr = ipr
        self.renorm_ipr = renorm_ipr

        super().__init__(*args, **kwargs)


class EvalLossMetrics(Config):
    FIELDS = (
        'eval_amps_metrics',
        'local_energy_metrics',
        'full_energy',
        'sample_aware_energy',
        'hf_proj_energy_real',
        'hf_proj_energy_imag',
        'full_hf_proj_energy_real',
        'full_hf_proj_energy_imag',
        'eval_loss_time'
    )

    OPTIONAL_FIELDS = (
        'eval_amps_metrics',
        'local_energy_metrics',
        'full_energy',
        'sample_aware_energy',
        'hf_proj_energy_real',
        'hf_proj_energy_imag',
        'full_hf_proj_energy_real',
        'full_hf_proj_energy_imag',
        'eval_loss_time'
    )

    def __init__(self,
                 *args,
                 eval_amps_metrics: EvalAmpsMetrics = None,
                 local_energy_metrics: LocalEnergyMetrics = None,
                 full_energy: float = np.nan,
                 sample_aware_energy: float = np.nan,
                 hf_proj_energy_real: float = np.nan,
                 hf_proj_energy_imag: float = np.nan,
                 full_hf_proj_energy_real: float = np.nan,
                 full_hf_proj_energy_imag: float = np.nan,
                 eval_loss_time: float = np.nan,
                 **kwargs):
        self.eval_amps_metrics = eval_amps_metrics if eval_amps_metrics is not None else EvalAmpsMetrics()
        self.local_energy_metrics = local_energy_metrics if local_energy_metrics is not None else LocalEnergyMetrics()
        self.full_energy = full_energy
        self.sample_aware_energy = sample_aware_energy
        self.hf_proj_energy_real = hf_proj_energy_real
        self.hf_proj_energy_imag = hf_proj_energy_imag
        self.full_hf_proj_energy_real = full_hf_proj_energy_real
        self.full_hf_proj_energy_imag = full_hf_proj_energy_imag

        self.eval_loss_time = eval_loss_time

        super().__init__(*args, **kwargs)


class IterResult(Config):
    FIELDS = (
        'iter_idx',
        'sampling_metrics',
        'eval_loss_metrics',
        'backward_time',
        'proc_grad_metrics',
        'iter_time',
    )
    OPTIONAL_FIELDS = (
        'sampling_metrics',
        'eval_loss_metrics',
        'backward_time',
        'proc_grad_metrics',
        'iter_time',
    )

    def __init__(self,
                 *args,
                 iter_idx: int = np.nan,
                 sampling_metrics: SamplingMetrics = None,
                 eval_loss_metrics: EvalLossMetrics = None,
                 backward_time: float = np.nan,
                 proc_grad_metrics: ProcessGradMetrics = None,
                 iter_time: float = np.nan,
                 **kwargs):
        self.iter_idx = iter_idx
        self.sampling_metrics = sampling_metrics if sampling_metrics is not None else SamplingMetrics()
        self.eval_loss_metrics = eval_loss_metrics if eval_loss_metrics is not None else EvalLossMetrics()
        self.backward_time = backward_time
        self.proc_grad_metrics = proc_grad_metrics if proc_grad_metrics is not None else ProcessGradMetrics()

        self.iter_time = iter_time
        super().__init__(*args, **kwargs)


class EnergyOptExpConfig(Config):
    ALLOWED_LOSS_TYPES = (
        'full_e_loc',
        'sample_aware_e_loc',
    )
    ALLOWED_POPCOUNT_MODES = (
        'memory_efficient',
        'compute_efficient',
        'custom',
    )
    FIELDS = (
        'mols_root_dir',
        'mol_config',
        'mol_init_config',
        'series_name',
        'device',
        'rng_seed',
        'perm_type',
        'popcount_mode',
        'masker_config',
        'meta_ansatz_config',
        'use_sign_structure',
        'opt_schedule',
        'sampling_schedule',
        'loss_type',
        'full_energy_period',
        'eval_sampled_amps_twice',
        'local_energy_config',
        'proc_grad_schedule',
    )
    OPTIONAL_FIELDS = ('full_energy_period',)

    def __init__(self,
                 *args,
                 mols_root_dir: str = None,
                 mol_config: MolConfig = None,
                 mol_init_config: MolInitConfig = None,
                 series_name: str = None,
                 device: str = 'gpu',
                 rng_seed: int = 0,
                 perm_type: str = 'direct',
                 popcount_mode: str = 'custom',
                 masker_config: MaskerConfig = None,
                 meta_ansatz_config: MetaAnsatzConfig = None,
                 use_sign_structure: bool = False,
                 opt_schedule: Tuple[int, OptConfig] = None,
                 sampling_schedule: Tuple[int, SamplingConfig] = None,
                 loss_type: str = 'sample_aware_e_loc',
                 full_energy_period: int = None,
                 eval_sampled_amps_twice: bool = False,
                 local_energy_config: LocalEnergyCalculationConfig = None,
                 proc_grad_schedule: Tuple[int, ProcessGradConfig] = None,
                 **kwargs):
        assert mols_root_dir is not None
        self.mols_root_dir = mols_root_dir
        self.mol_config = mol_config
        self.mol_init_config = mol_init_config if mol_init_config is not None else MolInitConfig()

        assert series_name is not None
        self.series_name = series_name

        self.device = device
        self.rng_seed = rng_seed

        self.perm_type = perm_type
        assert popcount_mode in self.ALLOWED_POPCOUNT_MODES
        self.popcount_mode = popcount_mode

        self.masker_config = masker_config if masker_config is not None else MaskerConfig()
        self.meta_ansatz_config = meta_ansatz_config if meta_ansatz_config is not None else MetaAnsatzConfig()
        self.use_sign_structure = use_sign_structure
        self.meta_ansatz_config.ansatz_config.use_sign_structure = self.use_sign_structure

        self.opt_schedule = opt_schedule if opt_schedule is not None else Schedule(schedule=((0, OptConfig()), ))
        self.sampling_schedule = sampling_schedule if sampling_schedule is not None else Schedule(schedule=((0, SamplingConfig()),))

        assert loss_type in self.ALLOWED_LOSS_TYPES
        self.loss_type = loss_type
        self.full_energy_period = full_energy_period
        self.eval_sampled_amps_twice = eval_sampled_amps_twice
        self.local_energy_config = local_energy_config if local_energy_config is not None else LocalEnergyCalculationConfig()

        self.proc_grad_schedule = proc_grad_schedule if proc_grad_schedule is not None else Schedule(schedule=((0, ProcessGradConfig()),))

        super().__init__(*args, **kwargs)


class EnergyOptExp:
    def __init__(self,
                 *args,
                 config: EnergyOptExpConfig = None,
                 mol: Molecule = None,
                 hs: HilbertSpace = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config if EnergyOptExpConfig is not None else EnergyOptExpConfig()

        # We set up the RNG seed for reproducibility
        self.rng = None
        pt.manual_seed(self.config.rng_seed)

        if mol is not None:
            self.mol = mol
            logging.info(f'The molecule was provided from outside')
        else:
            self.mol = create_mol(config=self.config.mol_config,
                             init_config=self.config.mol_init_config,
                             mols_root_dir=self.config.mols_root_dir)
            logging.info(f'We created the molecule')

        self.series_dir = create_dir(os.path.join(self.mol.dir,
                                                  'exp_series',
                                                  self.config.series_name))

        self.dir = create_dir(os.path.join(self.series_dir,
                                           f'{self.config.to_sha256_str()}'))
        self.config.to_json(os.path.join(self.dir, 'config.json'))

        # We set up the RNG seed for reproducibility
        self.rng = np.random.default_rng(seed=self.config.rng_seed)
        np.random.seed(seed=self.config.rng_seed)
        pt.manual_seed(self.config.rng_seed)

        # We create Hilbert space to encompass all further objects
        if hs is not None:
            self.hs = hs
            logging.info('Hilbert space was provided from outside')
        self.hs = HilbertSpace(qubit_num=self.mol.n_qubits,
                               parent_dir=self.mol.dir,
                               device=pt.device('cuda:0') if self.config.device == 'gpu' else pt.device('cpu'),
                               rng_seed=self.config.rng_seed,
                               rng=self.rng,
                               popcount_mode=self.config.popcount_mode)
        self.hs.init_perm()
        logging.info('We have created the Hilbert space')

        # Creating the Hamiltonian
        self.ham = PauliObservable(hilbert_space=self.hs,
                                   of_qubit_operator=self.mol.qubit_ham)
        logging.info('We have created the Hamiltonian')


        self.masker = create_masker(config=self.config.masker_config,
                                    hs=self.hs,
                                    mol=self.mol)
        logging.info('We have created the masker')

        if self.config.use_sign_structure:
            sign_structure = np.load(os.path.join(self.mol.dir, 'sign_structure.npy'))
            sign_structure = pt.from_numpy(sign_structure).to(self.hs.device)
        else:
            sign_structure = None
        self.wf = create_ansatz(config=self.config.meta_ansatz_config,
                                hs=self.hs,
                                masker=self.masker,
                                sign_structure=sign_structure)
        self.repetition_num = None
        self.next_rep_sample_num = 2 * self.config.sampling_schedule[0][1].sample_num
        self.actual_unq_num = None

        self.ansatz_dir = os.path.join(self.hs.parent_dir,
                                       'ansatzes',
                                       f'{self.config.meta_ansatz_config.to_sha256_str()}')
        self.ansatz_filename = os.path.join(self.ansatz_dir,
                                            f'rng_seed={self.config.rng_seed}')
        if os.path.exists(self.ansatz_filename):
            self.wf.load_state_dict(pt.load(self.ansatz_filename))
            logging.info('Ansatz existed, we loaded it')
        else:
            if not os.path.exists(self.ansatz_dir):
                os.makedirs(self.ansatz_dir)
            pt.save(self.wf.state_dict(), self.ansatz_filename)
            self.config.meta_ansatz_config.to_json(os.path.join(self.ansatz_dir, 'meta_ansatz_config.json'))
            logging.info('Ansatz did not exist, we created it')
        self.wf = self.wf.to(self.hs.device)

        self.opt = create_opt(wf=self.wf, opt_config=self.config.opt_schedule[0][1])
        logging.info('We created the optimizer')

        self.min_energy = np.inf
        self.min_energy_iter = None
        self.min_energy_time = None

        self.chem_acc_iter = None
        self.chem_acc_time = None

        self.result_filename = os.path.join(self.dir, 'result.csv')
        result_df = pd.DataFrame(columns=IterResult().to_flat_dict().keys())
        result_df.to_csv(self.result_filename,
                              mode='w',
                              header=True)
        logging.info('We prepared empty result file')

        self.best_loss_level2loss_dir = {
            'exp': self.dir,
            'series': self.series_dir,
            'mol': self.mol.dir,
        }

        self.best_loss_level2loss_value_filename = {
            best_loss_level: os.path.join(self.best_loss_level2loss_dir[best_loss_level],
                                          f'best_{self.config.loss_type}.npy')
            for best_loss_level in self.best_loss_level2loss_dir
        }
        self.best_loss_level2readable_loss_value_filename = {
            best_loss_level: os.path.join(self.best_loss_level2loss_dir[best_loss_level],
                                          f'best_{self.config.loss_type}.txt')
            for best_loss_level in self.best_loss_level2loss_dir
        }
        self.best_loss_level2loss_value = {
            best_loss_level: np.inf
            for best_loss_level in self.best_loss_level2loss_dir
        }

        self.best_loss_level2iter_idx_filename = {
            best_loss_level: os.path.join(self.best_loss_level2loss_dir[best_loss_level],
                                          f'best_{self.config.loss_type}_iter_idx.npy')
            for best_loss_level in self.best_loss_level2loss_dir
        }
        self.best_loss_level2readable_iter_idx_filename = {
            best_loss_level: os.path.join(self.best_loss_level2loss_dir[best_loss_level],
                                          f'best_{self.config.loss_type}_iter_idx.txt')
            for best_loss_level in self.best_loss_level2loss_dir
        }

        self.best_loss_level2sampled_indices_filename = {
            best_loss_level: os.path.join(self.best_loss_level2loss_dir[best_loss_level],
                                          f'best_{self.config.loss_type}_sampled_indices.npy')
            for best_loss_level in self.best_loss_level2loss_dir
        }
        self.best_loss_level2sampled_counts_filename = {
            best_loss_level: os.path.join(self.best_loss_level2loss_dir[best_loss_level],
                                          f'best_{self.config.loss_type}_sampled_counts.npy')
            for best_loss_level in self.best_loss_level2loss_dir
        }

        self.best_loss_level2exp_config_filename = {
            best_loss_level: os.path.join(self.best_loss_level2loss_dir[best_loss_level],
                                          f'best_{self.config.loss_type}_exp_config.pickle')
            for best_loss_level in self.best_loss_level2loss_dir
        }
        self.best_loss_level2readable_exp_config_filename = {
            best_loss_level: os.path.join(self.best_loss_level2loss_dir[best_loss_level],
                                          f'best_{self.config.loss_type}_exp_config.json')
            for best_loss_level in self.best_loss_level2loss_dir
        }

        self.best_loss_level2ansatz_filename = {
            best_loss_level: os.path.join(self.best_loss_level2loss_dir[best_loss_level],
                                          f'best_{self.config.loss_type}_ansatz_state_dict')
            for best_loss_level in self.best_loss_level2loss_dir
        }

        for best_loss_level, loss_filename in self.best_loss_level2loss_value_filename.items():
            if os.path.exists(loss_filename):
                self.best_loss_level2loss_value[best_loss_level] = np.load(loss_filename)
        logging.info(f'We initialise the best losses. They are:')
        for best_loss_level, loss_value in self.best_loss_level2loss_value.items():
            logging.info(f'{best_loss_level}: {loss_value}')

        self.best_loss_sampling_result = None

    def update_opt(self,
                   iter_idx: int = None,
                   wf=None,
                   opt=None):
        if len(self.config.opt_schedule) == 1:
            return opt
        elif iter_idx == 0:
            return opt
        else:
            new_opt_config = bin_search_schedule(self.config.opt_schedule, iter_idx=iter_idx)
            old_opt_config = bin_search_schedule(self.config.opt_schedule, iter_idx=iter_idx - 1)
            if new_opt_config.opt_type != old_opt_config.opt_type:
                return create_opt(wf=wf, opt_config=new_opt_config)
            elif new_opt_config.lr != old_opt_config.lr:
                for g in opt.param_groups:
                    g['lr'] = new_opt_config.lr
                return opt
            else:
                return opt

    @timed
    def sample(self, iter_idx: int = None):
        sampling_metrics = SamplingMetrics()
        sampling_config = bin_search_schedule(self.config.sampling_schedule, iter_idx=iter_idx)
        sampling_result, self.actual_unq_num, self.repetition_num, self.next_rep_sample_num = sample(wf=self.wf,
                                                                      config=sampling_config,
                                                                      starting_sample_num=self.next_rep_sample_num)

        sorted_indices, sort_perm = self.wf.sort_base_idx(base_idx=sampling_result.indices)
        sampling_result.indices = sorted_indices
        sampling_result.counts = sampling_result.counts[sort_perm]
        
        sampling_metrics.unq_num = sampling_result.indices.shape[0]
        sampling_metrics.sample_num = sampling_result.counts.sum()

        sampling_metrics.actual_unq_num = self.actual_unq_num
        sampling_metrics.repetition_num = self.repetition_num
        sampling_metrics.next_rep_sample_num = self.next_rep_sample_num

        return sampling_result, sampling_metrics

    @timed
    def eval_amps(self,
                  sampling_result: SamplingResult = None,
                  detached: bool = False):
        eval_amps_metrics = EvalAmpsMetrics()
        if detached:
            with pt.no_grad():
                sampled_amps = self.wf.amplitude(sampling_result.indices)
        else:
            sampled_amps = self.wf.amplitude(sampling_result.indices)
            sorted_abses, sorted_abses_inv = pt.sort(pt.abs(sampled_amps), descending=True)
            eval_amps_metrics.max_amp = sampled_amps[sorted_abses_inv[0]]
            eval_amps_metrics.min_amp = sampled_amps[sorted_abses_inv[-1]]
            eval_amps_metrics.sampled_prob = pt.dot(pt.conj(sampled_amps), sampled_amps).real

            sampled_probs = (pt.conj(sampled_amps) * sampled_amps).real
            eval_amps_metrics.ipr = (sampled_probs * sampled_probs).sum()

            renorm_probs = sampled_probs / pt.sum(sampled_probs)
            eval_amps_metrics.renorm_ipr = pt.sum(renorm_probs * renorm_probs).sum()

        return sampled_amps, eval_amps_metrics

    @timed
    def compute_loss(self,
                     iter_idx: int = None,
                     sampling_result: SamplingResult = None):
        eval_loss_metrics = EvalLossMetrics()
        sampled_amps, eval_loss_metrics.eval_amps_metrics, eval_amps_time = self.eval_amps(sampling_result=sampling_result,
                                                                                           detached=True if self.config.eval_sampled_amps_twice else False)
        eval_loss_metrics.eval_amps_metrics.eval_amps_time = eval_amps_time

        if self.config.loss_type == 'full_e_loc':
            local_energy_result, local_energy_metrics = compute_local_energies(wf=self.wf,
                                                                  sampling_result=sampling_result,
                                                                  sampled_amps=sampled_amps.detach(),
                                                                  ham=self.ham,
                                                                  config=self.config.local_energy_config,
                                                                  sample_aware=False)
            loss_local_energies = local_energy_result.full_e_loc_mc_est.values - local_energy_result.full_e_loc_mc_est.mean
            loss_freqs = local_energy_result.full_e_loc_mc_est.freqs
            eval_loss_metrics.full_energy = local_energy_result.full_e_loc_mc_est.mean.real

        elif self.config.loss_type == 'sample_aware_e_loc':
            if (self.config.full_energy_period is not None) and ((iter_idx % self.config.full_energy_period) == 0) and (iter_idx != 0):
                local_energy_result, local_energy_metrics = compute_local_energies(wf=self.wf,
                                                                      sampling_result=sampling_result,
                                                                      sampled_amps=sampled_amps.detach(),
                                                                      ham=self.ham,
                                                                      config=self.config.local_energy_config,
                                                                      sample_aware=False)
                eval_loss_metrics.full_energy = local_energy_result.full_e_loc_mc_est.mean.real

            else:
                local_energy_result, local_energy_metrics = compute_local_energies(wf=self.wf,
                                                                      sampling_result=sampling_result,
                                                                      sampled_amps=sampled_amps.detach(),
                                                                      ham=self.ham,
                                                                      config=self.config.local_energy_config,
                                                                      sample_aware=True)
            loss_local_energies = local_energy_result.sample_aware_e_loc_mc_est.values - local_energy_result.sample_aware_e_loc_mc_est.mean
            loss_freqs = local_energy_result.sample_aware_e_loc_mc_est.freqs

        else:
            raise RuntimeError(f'Wrong loss type: {self.config.loss_type}')

        eval_loss_metrics.local_energy_metrics = local_energy_metrics
        eval_loss_metrics.sample_aware_energy = local_energy_result.sample_aware_e_loc_mc_est.mean.real
        eval_loss_metrics.hf_proj_energy_real = local_energy_result.sample_aware_e_loc_mc_est.values[-1].real
        eval_loss_metrics.hf_proj_energy_imag = local_energy_result.sample_aware_e_loc_mc_est.values[-1].imag
        #hf_sampling_result = SamplingResult(indices=sampling_result.indices[-1:], counts=sampling_result.counts[-1:])
        # full_hf_proj_energy_e_loc, _ = compute_local_energies(wf=self.wf,
        #                                                       sampling_result=hf_sampling_result,
        #                                                       sampled_amps=sampled_amps[-1:].detach(),
        #                                                       ham=self.ham,
        #                                                       config=self.config.local_energy_config,
        #                                                       sample_aware=False)
        # eval_loss_metrics.full_hf_proj_energy_real = full_hf_proj_energy_e_loc.full_e_loc_mc_est.values[-1].real
        # eval_loss_metrics.full_hf_proj_energy_imag = full_hf_proj_energy_e_loc.full_e_loc_mc_est.values[-1].imag
        if self.config.eval_sampled_amps_twice:
            sampled_amps, second_eval_amps_metrics, second_eval_amps_time = self.eval_amps(sampling_result=sampling_result,
                                                                                           detached=False)
            second_eval_amps_metrics.eval_amps_time = eval_loss_metrics.eval_amps_metrics.eval_amps_time + second_eval_amps_time
            eval_loss_metrics.eval_amps_metrics = second_eval_amps_metrics

        loss = 2 * (loss_freqs * pt.log(pt.conj(sampled_amps)) * loss_local_energies).sum().real

        return loss, eval_loss_metrics

    @timed
    def backward(self, loss=None):
        loss.backward()

    @timed
    def process_grad(self, iter_idx: int = None, sampling_result: SamplingResult = None):
        proc_grad_config = bin_search_schedule(self.config.proc_grad_schedule, iter_idx)

        return process_grad(wf=self.wf,
                            sampling_result=sampling_result,
                            config=proc_grad_config)

    @timed
    def iter(self,
             iter_idx: int = None):
        # Update the state of the optimizer (e.g. change the learning rate or its type (e.g. SGD to Adam))
        iter_result = IterResult(iter_idx=iter_idx)
        self.opt.zero_grad()
        self.opt = self.update_opt(iter_idx=iter_idx, wf=self.wf, opt=self.opt)

        sampling_result, iter_result.sampling_metrics, sampling_time = self.sample(iter_idx=iter_idx)
        iter_result.sampling_metrics.sampling_time = sampling_time

        loss, iter_result.eval_loss_metrics, eval_loss_time = self.compute_loss(iter_idx=iter_idx,
                                                                                sampling_result=sampling_result)
        iter_result.eval_loss_metrics.eval_loss_time = eval_loss_time

        _, iter_result.backward_time = self.backward(loss)

        iter_result.proc_grad_metrics, proc_grad_time = self.process_grad(iter_idx=iter_idx,
                                                                          sampling_result=sampling_result)
        iter_result.proc_grad_metrics.proc_grad_time = proc_grad_time

        # Here we might update the best updated loss values, and we have to do it precisely here,
        # because otherwise we do a gradient update step and it shifts us from the optimal wf
        if self.config.loss_type == 'full_e_loc':
            cur_loss = iter_result.eval_loss_metrics.full_energy.detach().cpu().numpy().item()
        elif self.config.loss_type == 'sample_aware_e_loc':
            cur_loss = iter_result.eval_loss_metrics.sample_aware_energy.detach().cpu().numpy().item()
        else:
            raise RuntimeError(f'Wrong loss type was asked to update the best loss metrics: {self.config.loss_type}')

        for best_loss_level, loss_value in self.best_loss_level2loss_value.items():
            if cur_loss < loss_value:
                self.best_loss_level2loss_value[best_loss_level] = cur_loss
                np.save(self.best_loss_level2loss_value_filename[best_loss_level], cur_loss)
                with open(self.best_loss_level2readable_loss_value_filename[best_loss_level], 'w') as f:
                    f.write(f'{cur_loss}')

                np.save(self.best_loss_level2iter_idx_filename[best_loss_level], iter_idx)
                with open(self.best_loss_level2readable_iter_idx_filename[best_loss_level], 'w') as f:
                    f.write(f'{iter_idx}')

                np.save(self.best_loss_level2sampled_indices_filename[best_loss_level],
                        sampling_result.indices.cpu().numpy())
                np.save(self.best_loss_level2sampled_counts_filename[best_loss_level],
                        sampling_result.counts.cpu().numpy())

                with open(self.best_loss_level2exp_config_filename[best_loss_level], 'wb') as handle:
                    pickle.dump(self.config, handle, protocol=pickle.HIGHEST_PROTOCOL)
                self.config.to_json(self.best_loss_level2readable_exp_config_filename[best_loss_level])

                pt.save(self.wf.state_dict(), self.best_loss_level2ansatz_filename[best_loss_level])

        self.opt.step()

        return iter_result

    def run(self, iter_num: int = 10000):
        run_start_time = time.time()
        for iter_idx in (pbar := env_dependent_tqdm(range(iter_num))):
            iter_result, iter_time = self.iter(iter_idx=iter_idx)
            iter_result.iter_time = iter_time

            # Here we do some processing of the iter_result to save it
            iter_result_flat_dict = iter_result.to_flat_dict()
            for key, value in iter_result_flat_dict.items():
                if pt.is_tensor(value):
                    iter_result_flat_dict[key] = value.detach().cpu().item()
            result_df = pd.DataFrame(iter_result_flat_dict, index=[iter_result.iter_idx])
            result_df.to_csv(self.result_filename,
                             mode='a',
                             header=False)

            pbar.set_postfix(self.iter_result2pbar_postfix(iter_result=iter_result, cur_time=time.time() - run_start_time))
            if ((iter_idx % 1000) == 0) and (iter_idx != 0):
                pt.save(self.wf.state_dict(), os.path.join(self.dir, f'ansatz_state_dict_after_iter_{iter_idx}'))
                pt.save(self.opt.state_dict(), os.path.join(self.dir, f'opt_state_dict_after_iter_{iter_idx}'))

        pt.save(self.wf.state_dict(), os.path.join(self.dir, f'ansatz_state_dict_after_iter_{iter_num}'))
        pt.save(self.opt.state_dict(), os.path.join(self.dir, f'opt_state_dict_after_iter_{iter_num}'))

    def iter_result2pbar_postfix(self,
                                 iter_result: IterResult = None,
                                 cur_time: float = None):
        cur_energy = None
        if self.config.loss_type == 'full_e_loc':
            cur_energy = iter_result.eval_loss_metrics.full_energy.cpu().numpy()
        elif self.config.loss_type == 'sample_aware_e_loc':
            cur_energy = iter_result.eval_loss_metrics.sample_aware_energy

        if cur_energy < self.min_energy:
            self.min_energy = cur_energy
            self.min_energy_iter = iter_result.iter_idx
            self.min_energy_time = cur_time
        ref_energy = self.mol.fci_energy if self.mol.fci_energy is not None else self.mol.ccsd_t_energy
        if (cur_energy - ref_energy < CHEMICAL_ACCURACY) and (self.chem_acc_iter is None):
            self.chem_acc_iter = iter_result.iter_idx
            self.chem_acc_time = cur_time

        postfix = {
            '<E>': iter_result.eval_loss_metrics.full_energy.cpu().numpy() if self.config.loss_type == 'full_e_loc' else iter_result.eval_loss_metrics.full_energy,
            'SA <E>': f'{iter_result.eval_loss_metrics.sample_aware_energy}',
            'N_unq': iter_result.sampling_metrics.unq_num,
            'Act. N_unq': self.actual_unq_num,
            'Rep. num': self.repetition_num,
            'Next N_s': self.next_rep_sample_num,
            'min <E>': f'{self.min_energy} in {self.min_energy_time}s ({self.min_energy_iter} iter)',
            'Chem. Acc.': f'in {self.chem_acc_time}s ({self.chem_acc_iter} iter)',
        }

        return postfix
