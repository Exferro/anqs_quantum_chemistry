import os
os.environ['OMP_NUM_THREADS'] = "16"
os.environ['MKL_NUM_THREADS'] = "16"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import json
import pickle
import time

import numpy as np
import torch as pt
pt.use_deterministic_algorithms(False)

import pandas as pd

from matplotlib import pyplot as plt

from tqdm import tqdm

from nqs.applications.quantum_chemistry.molecule import GeometryConfig, MolConfig, MolInitConfig

from nqs.infrastructure.nested_data import Schedule
from nqs.applications.quantum_chemistry.experiments.preparation import OptConfig
from nqs.applications.quantum_chemistry.experiments.calculations import SamplingConfig
from nqs.applications.quantum_chemistry.experiments.calculations import ProcessGradConfig
from nqs.applications.quantum_chemistry.experiments.calculations import SRConfig
from nqs.applications.quantum_chemistry.experiments.energy_opt_exp import EvalLossMetrics, IterResult, EnergyOptExpConfig, EnergyOptExp

import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

from nqs.applications.quantum_chemistry.experiments.preparation import create_mol, create_masker, create_ansatz
from nqs.applications.quantum_chemistry import CHEMICAL_ACCURACY

mols_root_dir = './../../quantum_chemistry/paper_mols'
geom_config = GeometryConfig(type='diss_curve_test', idx=0)
mol_config = MolConfig(name='Li2O', geom_config=geom_config, basis='sto-3g')
mol_init_config = MolInitConfig(run_fci=False)

mol = create_mol(config=mol_config,
                 init_config=mol_init_config,
                 mols_root_dir=mols_root_dir)

series_name = 'tree_fixing_attempt_profile2_testing_schedule'
series_dir = os.path.join(mol.dir, 'exp_series', series_name)
if not os.path.exists(series_dir):
    os.makedirs(series_dir)

started_exp_configs_filename = os.path.join(series_dir, 'started_exp_configs.pickle')
if os.path.exists(started_exp_configs_filename):
    with open(started_exp_configs_filename, 'rb') as handle:
        started_exp_configs = pickle.load(handle)
else:
    started_exp_configs = []

finished_exp_configs_filename = os.path.join(series_dir, 'finished_exp_configs.pickle')
if os.path.exists(finished_exp_configs_filename):
    with open(finished_exp_configs_filename, 'rb') as handle:
        finished_exp_configs = pickle.load(handle)
else:
    finished_exp_configs = []

started_exp_configs_set = set(started_exp_configs)
finished_exp_configs_set = set(finished_exp_configs)

for rng_seed in range(1):
    exp_config = EnergyOptExpConfig(mols_root_dir=mols_root_dir,
                                    mol_config=mol_config,
                                    mol_init_config=mol_init_config,
                                    series_name=series_name)
    exp_config.rng_seed = rng_seed
    exp_config.popcount_mode = 'custom'
    exp_config.meta_ansatz_config.ansatz_config.qubit_grouping_config.qubit_per_qudit = 6

    opt_schedule = Schedule(schedule=((0, OptConfig(lr=3e-4)),))
    exp_config.opt_schedule = opt_schedule

    sampling_schedule = Schedule(schedule=((0, SamplingConfig(sample_num=10000, sample_indices=True, sample_precisely=True, upscale_factor=5)),))
    exp_config.sampling_schedule = sampling_schedule

    exp_config.loss_type = 'sample_aware_e_loc'
    exp_config.local_energy_config.sampled_indices_chunk_size = 500000
    exp_config.local_energy_config.use_tree_for_candidates = 'ham'
    proc_grad_schedule = Schedule(schedule=((0, ProcessGradConfig(renorm_grad=False,
                                                                  clip_grad_norm=True,
                                                                  use_sr=True,
                                                                  sr_config=SRConfig(max_indices_num=50))),))
    exp_config.proc_grad_schedule = proc_grad_schedule

    print(f'RNG seed: {rng_seed}')

    if exp_config in finished_exp_configs_set:
        assert exp_config in started_exp_configs_set
        print(f'This experiment was finished before, we do not start it again\n')
    else:
        if exp_config not in started_exp_configs_set:
            print(f'This experiment is a fresh one\n')
            started_exp_configs.append(exp_config)
            started_exp_configs_set.add(exp_config)
            with open(started_exp_configs_filename, 'wb') as handle:
                pickle.dump(started_exp_configs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print(f'This experiment was started at some point\n')

        energy_opt_exp = EnergyOptExp(config=exp_config, mol=mol)
        print(energy_opt_exp.ham.device)
        for method in ('hf', 'cisd', 'ccsd', 'ccsd_t', 'fci'):
            print(f'{method} energy: {getattr(energy_opt_exp.mol, f"{method}_energy")}')
        if energy_opt_exp.mol.fci_energy is not None:
            print(f'fci energy up to chem. acc.: {energy_opt_exp.mol.fci_energy + CHEMICAL_ACCURACY}')
        print()
        energy_opt_exp.run(iter_num=5000)
        finished_exp_configs.append(exp_config)
        finished_exp_configs_set.add(exp_config)
        with open(finished_exp_configs_filename, 'wb') as handle:
            pickle.dump(finished_exp_configs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'The experiment was just finished')
        print()
    print()


