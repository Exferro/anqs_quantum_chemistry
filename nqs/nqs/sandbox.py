import os
os.environ['OMP_NUM_THREADS'] = "16"
os.environ['MKL_NUM_THREADS'] = "16"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import os
import json
import pickle
import time

import numpy as np
import torch as pt

from matplotlib import pyplot as plt

from tqdm import tqdm

from nqs.base.constants import BASE_COMPLEX_TYPE, BASE_REAL_TYPE

from nqs.applications.quantum_chemistry.molecule import GeometryConfig, MolConfig, MolInitConfig, Molecule
from nqs.applications.quantum_chemistry.experiments.preparation import MaskerConfig, MetaAnsatzConfig
from nqs.applications.quantum_chemistry.experiments.exp_config import ExpConfig
from nqs.applications.quantum_chemistry.experiments.calculations import SamplingConfig, LocalEnergyCalculationConfig


from nqs.applications.quantum_chemistry import CHEMICAL_ACCURACY

from nqs.base.hilbert_space import HilbertSpace
from nqs.stochastic.observables.pauli_observable import PauliObservable
from nqs.stochastic.ansatzes.anqs import ANQSConfig

from nqs.applications.quantum_chemistry.experiments.preparation import create_mol, create_masker, create_ansatz
from nqs.applications.quantum_chemistry.experiments import bin_search_schedule
from nqs.applications.quantum_chemistry.experiments.calculations import sample, SamplingResult, compute_local_energies, SRConfig, sr, ProcessGradConfig, process_grad

geom_config = GeometryConfig(type='paper', idx=0)
mol_config = MolConfig(name='MgI2', geom_config=geom_config, basis='sto-3g')
masker_config = MaskerConfig(symmetry_level='z2')

exp_config = ExpConfig(mol_config=mol_config,
                       perm_type='direct',
                       meta_ansatz_config=None,
                       masker_config=masker_config,
                       rng_seed=0)

mol = create_mol(config=mol_config,
                 init_config=MolInitConfig(run_fci=False),
                 mols_root_dir='./../../quantum_chemistry/mols')
exit(-1)
# Creating the RNG with a set seed for reproducibility
exp_rng = np.random.default_rng(seed=exp_config.rng_seed)
pt.manual_seed(exp_config.rng_seed)

hs = HilbertSpace(qubit_num=mol.n_qubits,
                  parent_dir=mol.dir,
                  device=pt.device('cuda:0'),
                  rng_seed=exp_config.rng_seed,
                  rng=exp_rng)
hs.init_perm()

masker = create_masker(config=masker_config,
                       hs=hs,
                       mol=mol)
print(masker.memo.shape)
print(len(mol.z2_generators))
config = MetaAnsatzConfig()
config.ansatz_config.de_mode = 'MADE'
config.ansatz_config.qubit_grouping_config.qubit_per_qudit = 6
config.ansatz_config.local_sampling_config.strategy = 'MU'
config.ansatz_config.local_sampling_config.masking_depth = 0

wf = create_ansatz(config=config,
                   hs=hs,
                   masker=masker)
wf = wf.to(hs.device)
print(f'Number of parameters: {wf.param_num}')

sampling_schedule = ((0, SamplingConfig(sample_indices=True, sample_num=10000, couple_spin_flip=False)),)
proc_grad_schedule = ((0, ProcessGradConfig(use_sr=True,
                                            sr_config=SRConfig(max_indices_num=100),
                                            clip_grad_norm=False, renorm_grad=True)),
                      (20000, ProcessGradConfig(use_sr=False, clip_grad_norm=False, renorm_grad=True)),)

for method in ('hf', 'cisd', 'ccsd', 'ccsd_t', 'fci'):
    print(f'{method} energy: {getattr(mol, f"{method}_energy")}')
if mol.fci_energy is not None:
    print(f'fci energy up to chem. acc.: {mol.fci_energy + CHEMICAL_ACCURACY}')

ham = PauliObservable(hilbert_space=hs,
                      of_qubit_operator=mol.qubit_ham)
print(f'Ham unq_xy_num: {ham.unq_xy_masks_num}')

opt = pt.optim.Adam(wf.parameters())
local_energy_config = LocalEnergyCalculationConfig(use_theor_freqs=True)
local_energy_config.use_sampled_only = True
local_energy_config.indices_split_size = 25000

losses = []
start_time = time.time()

min_energy = np.inf
min_energy_iter = None
min_energy_time = None

chem_acc_iter = None
chem_acc_time = None

for iter_idx in (pbar := tqdm(range(20000))):
    if iter_idx == 1500:
        for g in opt.param_groups:
            g['lr'] = 3e-4
    if iter_idx == 40000:
        opt = pt.optim.Adam(wf.parameters(), lr=3e-4)
    sampling_config = bin_search_schedule(sampling_schedule, iter_idx)
    sampling_result = sample(wf=wf,
                             config=sampling_config)
    amps = wf.amplitude(sampling_result.indices)
    local_energy_result = compute_local_energies(wf=wf,
                                                 sampling_result=sampling_result,
                                                 sampled_amps=amps.detach(),
                                                 ham=ham,
                                                 config=local_energy_config)

    opt.zero_grad()
    e_loc_mc_est = local_energy_result.full_e_loc_mc_est
    if e_loc_mc_est is not None:
        local_energies = e_loc_mc_est.values - e_loc_mc_est.mean
        loss = 2 * (e_loc_mc_est.freqs * pt.log(pt.conj(amps)) * local_energies).sum().real
        loss.backward()

        proc_grad_config = bin_search_schedule(proc_grad_schedule, iter_idx)
        process_grad(wf=wf,
                     sampling_result=sampling_result,
                     config=proc_grad_config)
        opt.step()
        if e_loc_mc_est.mean.real.detach().cpu().numpy() < min_energy:
            min_energy = e_loc_mc_est.mean.real.detach().cpu().numpy()
            min_energy_iter = iter_idx
            min_energy_time = time.time() - start_time
        if mol.fci_energy is not None:
            if (e_loc_mc_est.mean.real.detach().cpu().numpy() - mol.fci_energy < CHEMICAL_ACCURACY) and (
                    chem_acc_iter is None):
                chem_acc_iter = iter_idx
                chem_acc_time = time.time() - start_time
        else:
            if (e_loc_mc_est.mean.real.detach().cpu().numpy() - mol.ccsd_t_energy < CHEMICAL_ACCURACY) and (
                    chem_acc_iter is None):
                chem_acc_iter = iter_idx
                chem_acc_time = time.time() - start_time

    pbar.set_postfix(
        {
            '<E>': e_loc_mc_est.mean.real.detach().cpu().numpy(),
            'N_unq': sampling_result.indices.shape[0],
            'min <E>': min_energy,
            'min <E> time': min_energy_time,
            'min <E> iter': min_energy_iter,
            'Chem. Acc. time': chem_acc_time,
            'Chem. Acc. iter': chem_acc_iter,
        }
    )

    # print(f'Iteration #{iter_idx}, indices sampled: {}, <E> = {e_loc_mc_est.mean.real}')
    losses.append(e_loc_mc_est.mean.real.detach().cpu().numpy())
end_time = time.time()
print(f'Time elapsed: {end_time - start_time}')