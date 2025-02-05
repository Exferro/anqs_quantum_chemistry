import os

import torch as pt
import numpy as np
import pandas as pd

import json

import time

from tqdm import tqdm

from collections import namedtuple

from typing import Tuple

import pennylane as qml
from pennylane.qchem.convert import import_operator

from openfermion import QubitOperator

from nqs.applications.quantum_chemistry import GEOM_TYPES
from nqs.applications.quantum_chemistry import SYMMETRY_LEVELS, SAMPLING_MODES
from nqs.applications.quantum_chemistry import EXPERIMENTS_ROOT_DIR, RESULTS_ROOT_DIR
from nqs.applications.quantum_chemistry import SPLIT_SIZES_ROOT_DIR, SPLIT_SIZE_FILENAME
from nqs.applications.quantum_chemistry import STATE_DICTS_ROOT_DIR, STATE_DICT_FILENAME


from nqs.infrastructure import create_dir
from nqs.applications.quantum_chemistry.experiments.molecular_data import MolecularData
from nqs.applications.quantum_chemistry.experiments.run_pyscf import run_pyscf

from nqs.applications.quantum_chemistry import CHEMICAL_ACCURACY

from nqs.base.hilbert_space import HilbertSpace
from nqs.base.abstract_quantum_state import AbstractQuantumState
from nqs.stochastic.observables.pauli_observable import PauliObservable

from nqs.stochastic.maskers.locally_decomposable_masker import LocallyDecomposableMasker

from nqs.stochastic.ansatzes.legacy.anqs_primitives import activation_combo2activations
from nqs.stochastic.ansatzes.legacy.anqs_primitives import MaskingMode

from nqs.stochastic.ansatzes.legacy.nade.log_abs_global_phase_nade import LogAbsGlobalPhaseNADE
from nqs.stochastic.ansatzes.legacy.nade import ComplexLogPsiNADE
from nqs.stochastic.ansatzes.legacy.nade.real_log_psi_nade import RealLogPsiNADE
from nqs.stochastic.ansatzes.legacy.nade.log_abs_phase_nade import LogAbsPhaseNADE

from nqs.stochastic.ansatzes.legacy.made import LogAbsPhaseMADE
from nqs.stochastic.ansatzes.legacy.made import RealLogPsiMADE

from nqs.stochastic.ansatzes.base.log_abs_phase_anqs import LogAbsPhaseANQS

from nqs.stochastic.ansatzes import compute_log_jacobians, compute_log_grad
from nqs.utils.misc import hermitian_matmul, soft_matrix_inv

from nqs.stochastic.symmetries.particle_number_symmetry import ParticleNumberSymmetry
from nqs.stochastic.symmetries.spin_half_projection_symmetry import SpinHalfProjectionSymmetry
from nqs.stochastic.symmetries.z2_symmetry import Z2Symmetry


from nqs.utils.misc import verbosify_generator, compute_chunk_boundaries

MOL_DESCR_FIELDS = ('mol_name',
                    'geom_type',
                    'geom_idx',
                    'basis',
                    'multiplicity',
                    'charge')
MolDescr = namedtuple('MolDescr', MOL_DESCR_FIELDS)

WF_DESCR_FIELDS = ('depth',
                 'width',
                 'phase_depth',
                 'phase_width')

EXP_DESCR_FIELDS = ('name',)
EXP_DESCR_FIELDS = EXP_DESCR_FIELDS + MOL_DESCR_FIELDS
EXP_DESCR_FIELDS = EXP_DESCR_FIELDS + ('device',
                                       'ansatz',
                                       'dtype',
                                       'de_mode',
                                       'head_num',
                                       'head_agg_mode',
                                       'depth',
                                       'width',
                                       'activation_combo',
                                       'add_bias',
                                       'use_residuals',
                                       'phase_depth',
                                       'phase_width',
                                       'phase_activation_combo',
                                       'phase_add_bias',
                                       'phase_use_residuals',
                                       'iter_num',
                                       'sample_num',
                                       'symmetry_level',
                                       'sampling_mode',
                                       'masking_depth',
                                       'opt_class',
                                       'lr',
                                       'indices_split_size',
                                       'use_sr',
                                       'max_sr_indices',
                                       'use_reg_sr',
                                       'sr_reg',
                                       'renorm_grad',
                                       'clip_grad_norm',
                                       'rng_seed')
ExpDescr = namedtuple('ExpDescr', EXP_DESCR_FIELDS)
SeedlessExpDescr = namedtuple('SeedlessExpDescr', EXP_DESCR_FIELDS[:-1])

KPI_COLS = ('iter_idx',
            'energy',
            'var_est',
            'ste_est',
            'iter_time',
            'counts_sum',
            'wcounts_sum',
            'unique_num',
            'var_energy',
            'var_var_est',
            'var_ste_est')


def run_exp(exp_descr: ExpDescr = None,
            mols_dir: str = None,
            verbose: bool = True,
            save_period: int = 20):
    assert exp_descr is not None
    assert mols_dir is not None

    mol_descr = MolDescr(mol_name=exp_descr.mol_name,
                         geom_type=exp_descr.geom_type,
                         geom_idx=exp_descr.geom_idx,
                         basis=exp_descr.basis,
                         multiplicity=exp_descr.multiplicity,
                         charge=exp_descr.charge)
    mol, mol_dir = create_mol(mol_descr=mol_descr,
                              mols_dir=mols_dir)
    if verbose:
        print(f'We are starting experiment for a molecule at {mol_dir}')
        for method in ('hf', 'cisd', 'ccsd', 'ccsd_t', 'fci'):
            print(f'{method} energy: {getattr(mol, f"{method}_energy")}')
        if mol.fci_energy is not None:
            print(f'fci energy up to chem. acc.: {mol.fci_energy + CHEMICAL_ACCURACY}')

    # ######################CREATE FOLDERS FOR THE RESULTS#############################
    exps_dir = create_dir(os.path.join(mol_dir, EXPERIMENTS_ROOT_DIR))
    exp_dir = create_dir(os.path.join(exps_dir,
                                      *[f'{descr_field}={getattr(exp_descr, descr_field)}'
                                        for descr_field in EXP_DESCR_FIELDS
                                        if descr_field not in MOL_DESCR_FIELDS]))
    results_dir = create_dir(os.path.join(exp_dir, RESULTS_ROOT_DIR))

    # #####################INITIALIZE EMPTY DATAFRAME FOR THE RESULT###################
    result_df = pd.DataFrame(columns=KPI_COLS)
    result_df.to_csv(os.path.join(results_dir, 'result'),
                     mode='w',
                     header=True)

    # ###############################CALCULATION PART##################################

    # ###########################CHOOSE A DEVICE#######################################
    assert exp_descr.device in ('cpu', 'gpu')
    cpu_device = pt.device('cpu')
    gpu_device = pt.device('cuda:0')
    device = cpu_device if exp_descr.device == 'cpu' else gpu_device
    # #######################CREATE A HILBERT SPACE OBJECT#############################
    hs = HilbertSpace(qubit_num=mol.n_qubits,
                      device=device,
                      parent_dir=mol.data_directory,
                      rng_seed=exp_descr.rng_seed)
    # And immediately set an RNG seed!
    pt.manual_seed(exp_descr.rng_seed)

    # ###########################CREATE A MASKER#######################################
    masker = create_masker(exp_descr=exp_descr,
                           hs=hs,
                           mol=mol)

    # ###########################CREATE A WAVEFUNCTION#################################
    wf = create_wf(exp_descr=exp_descr,
                   hs=hs,
                   masker=masker,
                   mol_dir=mol_dir)

    # ###############CREATE A PAULI OBSERVABLE FOR THE HAMILTONIAN#####################
    ham = PauliObservable(hilbert_space=hs,
                          of_qubit_operator=mol.qubit_ham)
    # #################################CREATE AN OPTIMIZER#############################
    from_power = 2.0
    to_power = 5.0
    powers = np.linspace(from_power, to_power, hs.qubit_num)
    coeffs = 2 ** np.linspace(0.5, 1, hs.qubit_num)
    if exp_descr.opt_class == 'Adam':
        opt = pt.optim.Adam(wf.parameters(), lr=exp_descr.lr)
    elif exp_descr.opt_class == 'SGD':
        opt = pt.optim.SGD(wf.parameters(), lr=exp_descr.lr)
    else:
        raise RuntimeError(f'Wrong optimizer class: {exp_descr.opt_class}')

    assert isinstance(exp_descr.sample_num, tuple)
    assert isinstance(exp_descr.sample_num[0], str)

    sample_schedule = None
    sample_config = {}

    if exp_descr.sample_num[0] == 'schedule':
        sample_schedule = exp_descr.sample_num[1:]
        sample_schedule = sorted(sample_schedule, key=lambda tup: tup[0])
        sample_schedule_stage_idx = 0
    elif exp_descr.sample_num[0] == 'adaptive':
        for key, value in exp_descr.sample_num[1:]:
            sample_config[key] = value
    else:
        raise RuntimeError(f'Wrong exp_descr.sample_num[0] is str but has wrong value: {exp_descr.sample_num[0]}')

    stop_early = False
    reached_hf = False
    # #################################TRAINING LOOP###################################
    for iter_idx in tqdm(range(exp_descr.iter_num)):
        if exp_descr.sample_num[0] == 'schedule':
            if (sample_schedule_stage_idx < len(sample_schedule) - 1) and (
                    iter_idx >= sample_schedule[sample_schedule_stage_idx + 1][0]):
                sample_schedule_stage_idx += 1
            cur_sample_num = sample_schedule[sample_schedule_stage_idx][1]
        elif exp_descr.sample_num[0] == 'adaptive':
            if iter_idx == 0:
                cur_sample_num = sample_config['start_sample_num']

        result_dict = {}
        start_time = time.time()

        if verbose:
            print(f'Iteration #{iter_idx}')
        kpi_tuple, unique_num, var_tuple, final_sample_num = opt_step(wf=wf,
                                                                      sample_num=cur_sample_num,
                                                                      min_unq_sample_num=sample_config.get(
                                                                          'min_unq_sample_num', None),
                                                                      max_unq_sample_num=sample_config.get(
                                                                          'max_unq_sample_num', None),
                                                                      min_sample_num=sample_config.get('min_sample_num',
                                                                                                       None),
                                                                      max_sample_num=sample_config.get('max_sample_num',
                                                                                                       None),
                                                                      ham=ham,
                                                                      opt=opt,
                                                                      indices_split_size=exp_descr.indices_split_size,
                                                                      use_sr=exp_descr.use_sr,
                                                                      max_sr_indices=exp_descr.max_sr_indices,
                                                                      use_reg_sr=exp_descr.use_reg_sr,
                                                                      sr_reg=exp_descr.sr_reg,
                                                                      renorm_grad=exp_descr.renorm_grad,
                                                                      clip_grad_norm=exp_descr.clip_grad_norm,
                                                                      verbose=verbose)

        energy, var_est, ste_est, counts_sum, wcounts_sum = kpi_tuple
        var_energy, var_var_est, var_ste_est = var_tuple

        if var_energy - mol.fci_energy < 1e-5:
            stop_early = True

        if final_sample_num > cur_sample_num:
            cur_sample_num = final_sample_num
            print(f'For next iteration we set sample_num to {cur_sample_num}')

        if verbose:
            print(f'<E> = {energy}\n')
        iter_time = time.time() - start_time

        cur_locals = locals()
        result_dict.update({kpi_col: cur_locals[kpi_col] for kpi_col in KPI_COLS})

        result_df = pd.concat([result_df,
                               pd.DataFrame(result_dict, index=[0])],
                              ignore_index=True)
        if ((iter_idx + 1) % save_period) == 0:
            result_df.to_csv(os.path.join(results_dir, 'result'),
                             mode='a',
                             header=False)
            result_df = pd.DataFrame(columns=KPI_COLS)
            if stop_early:
                print('We are stopping early since we came within 1e-5 Ha to the FCI energy')
                break


def create_mol(mol_descr: MolDescr = None,
               mols_dir: str = None,
               run_scf=True,
               run_cisd=True,
               run_ccsd=True,
               run_fci=True):
    assert mols_dir is not None
    assert mol_descr.geom_type in GEOM_TYPES
    if mol_descr.geom_type in ('carleo', 'pubchem'):
        assert mol_descr.geom_idx == 0

    mol_dir = create_dir(os.path.join(mols_dir,
                                      *[f'{descr_field}={getattr(mol_descr, descr_field)}'
                                        for descr_field in MOL_DESCR_FIELDS]))
    geom = load_geom(mols_dir=mols_dir,
                     mol_name=mol_descr.mol_name,
                     geom_type=mol_descr.geom_type,
                     geom_idx=mol_descr.geom_idx)
    mol = MolecularData(geometry=geom,
                        basis=mol_descr.basis,
                        multiplicity=mol_descr.multiplicity,
                        charge=mol_descr.charge,
                        data_directory=mol_dir)
    if os.path.exists("{}.hdf5".format(mol.filename)):
        mol.load()
    else:
        run_pyscf(molecule=mol,
                  run_scf=run_scf,
                  run_cisd=run_cisd,
                  run_ccsd=run_ccsd,
                  run_fci=run_fci)
        mol.save()

    return mol, mol_dir


def load_geom(mols_dir: str = None,
              mol_name: str = None,
              geom_type: str = None,
              geom_idx: int = None):
    geom_dir = os.path.join(mols_dir,
                            f'mol_name={mol_name}',
                            'geometries',
                            f'{geom_type}')
    geom_filename = os.path.join(geom_dir,
                                 '0.json' if geom_type in ('carleo', 'pubchem') else f'{geom_idx}.json')

    if os.path.exists(geom_filename):
        with open(geom_filename, 'r') as f:
            geom = json.load(f)
            return geom
    else:
        raise RuntimeError(f'There is no existing geometry at {geom_filename}')


def create_z2_symmetries(hilbert_space: HilbertSpace = None,
                         mol: MolecularData = None,
                         ham: QubitOperator = None,
                         perm: pt.Tensor = None,
                         inv_perm: pt.Tensor = None):
    qml_ham = import_operator(mol.qubit_ham)
    hf_base_vec = [0] * (mol.n_qubits - mol.n_electrons) + [1] * mol.n_electrons
    hf_base_vec = pt.tensor([hf_base_vec],
                            dtype=hilbert_space.idx_dtype,
                            device=hilbert_space.device)
    hf_base_vec = hf_base_vec[..., inv_perm]
    print(hf_base_vec)

    generators = qml.symmetry_generators(qml_ham)

    z2_symmetries = []
    for gen in generators:
        pauli_z_positions = hilbert_space.qubit_num - pt.tensor(list(gen.ops[0].wires)[::-1],
                                                                dtype=hilbert_space.idx_dtype,
                                                                device=hilbert_space.device) - 1

        pauli_z_positions = perm[pauli_z_positions]
        z2_symmetry = Z2Symmetry(hilbert_space=hilbert_space,
                                 pauli_z_positions=pauli_z_positions,
                                 value=None)
        z2_symmetry.value = z2_symmetry.compute_acc_eig(hf_base_vec)
        z2_symmetries.append(z2_symmetry)

    return tuple(z2_symmetries)


def create_masker(exp_descr: ExpDescr = None,
                  hs: HilbertSpace = None,
                  mol: MolecularData = None,
                  ham: QubitOperator = None,
                  perm: pt.Tensor = None,
                  inv_perm: pt.Tensor = None) -> LocallyDecomposableMasker:
    assert exp_descr.symmetry_level in SYMMETRY_LEVELS
    if exp_descr.symmetry_level == 'no_sym':
        masker = IdleMasker(hilbert_space=hs)
    else:
        if exp_descr.symmetry_level == 'e_num_spin' or exp_descr.symmetry_level == 'z2':
            symmetries = (ParticleNumberSymmetry(hilbert_space=hs,
                                                 particle_num=mol.n_electrons),
                          SpinHalfProjectionSymmetry(hilbert_space=hs,
                                                     spin=mol.multiplicity - 1,
                                                     perm=perm,
                                                     inv_perm=inv_perm))

            if exp_descr.symmetry_level == 'z2':
                symmetries = symmetries + create_z2_symmetries(hilbert_space=hs,
                                                               mol=mol,
                                                               ham=ham,
                                                               perm=perm,
                                                               inv_perm=inv_perm)
            masker = LocallyDecomposableMasker(hilbert_space=hs,
                                               symmetries=symmetries,
                                               perm=perm)
        else:
            raise RuntimeError(f'Wrong level of symmetry: {exp_descr.symmetry_level}')

    return masker


def create_wf(exp_descr: ExpDescr = None,
              hs: HilbertSpace = None,
              masker: LocallyDecomposableMasker = None,
              mol_dir: str = None):
    assert exp_descr.sampling_mode in SAMPLING_MODES

    split_size = np.inf
    split_size_dir = create_dir(os.path.join(mol_dir,
                                             SPLIT_SIZES_ROOT_DIR,
                                             exp_descr.ansatz,
                                             *[f'{descr_field}={getattr(exp_descr, descr_field)}'
                                               for descr_field in WF_DESCR_FIELDS]))
    split_size_filename = os.path.join(split_size_dir, SPLIT_SIZE_FILENAME)
    if os.path.exists(split_size_filename):
        with open(split_size_filename, 'r') as f:
            split_size = json.load(split_size_filename)
        print(f'Split size file exists so we use the value {split_size}')
    else:
        print(f'Split size file doesn\'t exist so we use the value {split_size}')

    perseverant_sampling = False
    if exp_descr.sampling_mode == 'vanilla':
        masking_mode = None
    elif exp_descr.sampling_mode == 'masked_logits':
        masking_mode = MaskingMode.logits
    elif exp_descr.sampling_mode == 'masked_part_base_vecs':
        masking_mode = MaskingMode.part_base_vecs
    elif exp_descr.sampling_mode == 'perseverant':
        masking_mode = MaskingMode.part_base_vecs
        perseverant_sampling = True
    elif exp_descr.sampling_mode == 'renorm':
        masking_mode = MaskingMode.part_base_vecs
        perseverant_sampling = 'renorm'

    assert exp_descr.ansatz in ('LogAbsGlobalPhaseNADE',
                                 'ComplexLogPsiNADE',
                                 'RealLogPsiNADE',
                                 'LogAbsPhaseNADE',
                                 'LogAbsPhaseMADE',
                                 'RealLogPsiMADE',
                                 'LogAbsPhaseANQS',
                                 'LogPsiANQS')
    activations = activation_combo2activations(activation_combo=exp_descr.activation_combo,
                                               depth=exp_descr.depth)
    phase_activations = activation_combo2activations(activation_combo=exp_descr.phase_activation_combo,
                                                     depth=exp_descr.phase_depth)

    if exp_descr.ansatz == 'LogAbsGlobalPhaseNADE':
        wf = LogAbsGlobalPhaseNADE(hilbert_space=hs,
                                   depth=exp_descr.depth,
                                   width=exp_descr.width,
                                   log_abs_activations=activations,
                                   phase_depth=exp_descr.phase_depth,
                                   phase_width=exp_descr.phase_width,
                                   phase_activations=phase_activations,
                                   masker=masker,
                                   masking_mode=masking_mode,
                                   masking_depth=exp_descr.masking_depth,
                                   perseverant_sampling=perseverant_sampling,
                                   split_size=split_size)
    elif exp_descr.ansatz == 'ComplexLogPsiNADE':
        wf = ComplexLogPsiNADE(hilbert_space=hs,
                               depth=exp_descr.depth,
                               width=exp_descr.width,
                               activations=activations,
                               masker=masker,
                               masking_mode=masking_mode,
                               masking_depth=exp_descr.masking_depth,
                               perseverant_sampling=perseverant_sampling,
                               split_size=split_size)
    elif exp_descr.ansatz == 'RealLogPsiNADE':
        wf = RealLogPsiNADE(hilbert_space=hs,
                            depth=exp_descr.depth,
                            width=exp_descr.width,
                            activations=activations,
                            masker=masker,
                            masking_mode=masking_mode,
                            masking_depth=exp_descr.masking_depth,
                            perseverant_sampling=perseverant_sampling,
                            split_size=split_size)
    elif exp_descr.ansatz == 'LogAbsPhaseNADE':
        wf = LogAbsPhaseNADE(hilbert_space=hs,
                             depth=exp_descr.depth,
                             width=exp_descr.width,
                             log_abs_activations=activations,
                             phase_depth=exp_descr.phase_depth,
                             phase_width=exp_descr.phase_width,
                             phase_activations=phase_activations,
                             masker=masker,
                             masking_mode=masking_mode,
                             masking_depth=exp_descr.masking_depth,
                             perseverant_sampling=perseverant_sampling,
                             split_size=split_size)
    elif exp_descr.ansatz == 'LogAbsPhaseMADE':
        wf = LogAbsPhaseMADE(hilbert_space=hs,
                             depth=exp_descr.depth,
                             width=exp_descr.width,
                             log_abs_activations=activations,
                             phase_depth=exp_descr.phase_depth,
                             phase_width=exp_descr.phase_width,
                             phase_activations=phase_activations,
                             masker=masker,
                             masking_mode=masking_mode,
                             masking_depth=exp_descr.masking_depth,
                             perseverant_sampling=perseverant_sampling,
                             split_size=split_size)
    elif exp_descr.ansatz == 'RealLogPsiMADE':
        wf = RealLogPsiMADE(hilbert_space=hs,
                            depth=exp_descr.depth,
                            width=exp_descr.width,
                            activations=activations,
                            masker=masker,
                            masking_mode=masking_mode,
                            masking_depth=exp_descr.masking_depth,
                            perseverant_sampling=perseverant_sampling,
                            split_size=split_size)
    elif exp_descr.ansatz == 'LogAbsPhaseANQS':
        wf = LogAbsPhaseANQS(hilbert_space=hs,
                             dtype=hs.rdtype,
                             de_mode=exp_descr.de_mode,
                             head_num=exp_descr.head_num,
                             head_agg_mode=exp_descr.head_agg_mode,
                             log_abs_depth=exp_descr.depth)
    else:
        raise RuntimeError(f'Wrong ansatz type: {exp_descr.ansatz}')

    state_dict_dir = create_dir(os.path.join(mol_dir,
                                             STATE_DICTS_ROOT_DIR,
                                             f'{wf.__class__.__name__}',
                                             *[f'{descr_field}={getattr(exp_descr, descr_field)}'
                                               for descr_field in WF_DESCR_FIELDS]))

    state_dict_filename = os.path.join(state_dict_dir,
                                       STATE_DICT_FILENAME + f'_rng_seed_{wf.rng_seed}')
    if not os.path.exists(state_dict_filename):
        pt.save(wf.state_dict(), state_dict_filename)

    wf.load_state_dict(pt.load(state_dict_filename))
    wf = wf.to(hs.device)
    wf.init_shapes_and_splits()

    return wf


def sample_and_postselect(wf: AbstractQuantumState = None,
                          sample_unique: bool = False,
                          sample_num: int = None,
                          mask_unphysical: bool = False,
                          min_unq_sample_num: int = None,
                          max_unq_sample_num: int = None,
                          min_sample_num: int = None,
                          max_sample_num: int = None,
                          verbose: bool = False):
    if ((min_unq_sample_num is not None)
            and (max_unq_sample_num is not None)
            and (min_sample_num is not None)
            and (max_sample_num is not None)):
        keep_sampling = True
        while keep_sampling:
            unq_indices, counts, weights = wf.sample_stats(sample_num)
            if unq_indices.shape[0] < min_unq_sample_num:
                if 2 * sample_num < max_sample_num:
                    sample_num *= 2
                elif sample_num < max_sample_num:
                    sample_num = max_sample_num
                else:
                    keep_sampling = False
            elif unq_indices.shape[0] > max_unq_sample_num:
                if sample_num > 2 * min_sample_num:
                    sample_num //= 2
                elif sample_num > min_sample_num:
                    sample_num = min_sample_num
                else:
                    keep_sampling = False
            else:
                keep_sampling = False
    else:
#         unq_indices, counts, weights = wf.sample_stats(sample_num)
#         weights = weights.type(wf.cdtype).to(wf.device)
        if sample_unique:
            unq_indices, counts = wf.sample_indices_gumbel(sample_num)
        else:
            unq_indices, counts = wf.sample_stats(sample_num)
        #unq_indices, counts = wf.sample_stats_new(sample_num)
        # amps = wf.amplitude(unq_indices)
        # freqs = (pt.conj(amps) * amps).detach().real
        # sum_prob = pt.sum(freqs)
        # print(f'Probability sampled: {sum_prob.item()}')
        # print(f'Entropy: {pt.sum(-freqs * pt.log2(freqs)) + -(1.0 - sum_prob) * pt.log(1.0 - sum_prob)}')
        # norm_freqs = freqs / pt.sum(freqs)
        # print(f'Renorm entropy: {pt.sum(-norm_freqs * pt.log2(norm_freqs))}')
        counts = counts.type(wf.cdtype).to(wf.device)
        weights = pt.ones_like(counts)
        
    counts_sum = counts.sum().detach().cpu().numpy().real.item()
    wcounts_sum = (weights * counts).sum().detach().cpu().numpy().real.item()

    if verbose:
        print(f'We have sampled {unq_indices.shape[0]} indices')
        print(f'They amounted for a total of {counts_sum} samples')
        print(
            f'Which is a {counts_sum / sample_num} fraction of the required number of samples')
        print(f'The weighted number of samples is: {wcounts_sum}\n')

    if mask_unphysical:
        # Mask off unphysical samples
        phys_mask = wf.masker.mask(wf.base_idx2base_vec(unq_indices))

        phys_indices = unq_indices[phys_mask]
        phys_counts = counts[phys_mask]
        phys_weights = weights[phys_mask]
    else:
        phys_indices = unq_indices
        phys_counts = counts
        phys_weights = weights

    phys_counts_sum = phys_counts.sum().detach().cpu().numpy().real.item()
    phys_wcounts_sum = (phys_weights * phys_counts).sum().detach().cpu().numpy().real.item()
    if verbose:
        print(f'Of them {phys_indices.shape[0]} were physical')
        print(f'They amounted for a total of {phys_counts_sum} samples')
        print(
            f'Which is a {phys_counts_sum / sample_num} fraction of the required number of samples')
        print(f'The weighted number of samples is: {phys_wcounts_sum}')

    return phys_indices, phys_counts, phys_weights, sample_num


def compute_var_and_ste(values: pt.Tensor = None,
                        counts: pt.Tensor = None) -> Tuple[pt.Tensor, pt.Tensor]:
    sample_num = counts.sum()
    values_mean = (values * counts).sum() / sample_num
    var_est = pt.mul(pt.pow((values - values_mean).real, 2), counts).sum() / (sample_num - 1)
    ste_est = pt.sqrt(var_est / sample_num)

    return var_est.detach().cpu().numpy().item(), ste_est.detach().cpu().numpy().item()


def sample_and_compute_local_energies(wf: AbstractQuantumState = None,
                                      sample_num: int = None,
                                      sample_unique: bool = False,
                                      sampled_only: bool = True,
                                      min_unq_sample_num: int = None,
                                      max_unq_sample_num: int = None,
                                      min_sample_num: int = None,
                                      max_sample_num: int = None,
                                      ham: PauliObservable = None,
                                      indices_split_size: int = None,
                                      use_theor_freqs: bool = True,
                                      verbose: bool = True):
    # Step 1. Sample and postselect only physical indices
    sampled_indices, counts, weights, final_sample_num = sample_and_postselect(wf=wf,
                                                                               sample_num=sample_num,
                                                                               sample_unique=sample_unique,
                                                                               min_unq_sample_num=min_unq_sample_num,
                                                                               max_unq_sample_num=max_unq_sample_num,
                                                                               min_sample_num=min_sample_num,
                                                                               max_sample_num=max_sample_num,
                                                                               verbose=verbose)

    if len(sampled_indices) == 0:
        return [], [], [], (np.nan, np.nan, np.nan, np.nan, np.nan), (np.nan, np.nan, np.nan), sample_num
    else:
        with pt.no_grad():
            sampled_indices_boundaries = compute_chunk_boundaries(array_len=sampled_indices.shape[0],
                                                                  chunk_size=indices_split_size)
            sampled_amps = pt.tensor([], dtype=wf.cdtype, device=wf.device)
            local_energies = pt.tensor([], dtype=wf.cdtype, device=wf.device)
            var_local_energies = pt.tensor([], dtype=wf.cdtype, device=wf.device)
            for chunk_idx in verbosify_generator(range(len(sampled_indices_boundaries) - 1),
                                                  verbose=verbose,
                                                  activity_descr=' metacomputing local energies'):
                # Step 2. Differentiably evaluate amplitudes for the physical indices
                chunk_start = sampled_indices_boundaries[chunk_idx]
                chunk_end = sampled_indices_boundaries[chunk_idx + 1]
                cur_sampled_amps = pt.tensor([], dtype=wf.cdtype, device=wf.device)
                max_amp_indices_num = wf.hilbert_space.gpu_memory_limit // (16 * wf.width ** 2)
                amp_indices_boundaries = compute_chunk_boundaries(array_len=sampled_indices[chunk_start:chunk_end].shape[0],
                                                                  chunk_size=max_amp_indices_num)
                for mini_chunk_idx in verbosify_generator(range(len(amp_indices_boundaries) - 1),
                                                          verbose=verbose,
                                                          activity_descr='computing no-grad sampled amplitudes'):
                    mini_chunk_start = amp_indices_boundaries[mini_chunk_idx]
                    mini_chunk_end = amp_indices_boundaries[mini_chunk_idx + 1]
                    cur_sampled_amps = pt.cat([cur_sampled_amps,
                                               wf.amplitude(sampled_indices[chunk_start:chunk_end][mini_chunk_start:mini_chunk_end])])

                # Step 3. Evaluate local energy elements
                e_locs, var_e_locs = ham.compute_local_energies(wf=wf,
                                                                sampled_indices=sampled_indices[chunk_start:chunk_end],
                                                                sampled_amps=cur_sampled_amps,
                                                                verbose=verbose,
                                                                sampled_only=sampled_only)
                sampled_amps = pt.cat([sampled_amps, cur_sampled_amps])
                local_energies = pt.cat([local_energies, e_locs])
                var_local_energies = pt.cat([var_local_energies, var_e_locs])

        # Step 4. Calculate probabilities (we call them frequencies to reflect that they are not
        # exact Born probabilities)
        if use_theor_freqs:
            freqs = pt.conj(sampled_amps.detach()) * sampled_amps.detach()
        else:
            freqs = (counts * weights)
        freqs = freqs / pt.sum(freqs)

        # Step 5. Evaluate KPIs
        mean_local_energy = pt.dot(local_energies, freqs)
        mean_local_energy = mean_local_energy.real.detach().cpu().numpy().item()
        var_est, ste_est = compute_var_and_ste(local_energies, (counts * weights).real)

        counts_sum = counts.sum().detach().cpu().numpy().real.item()
        wcounts_sum = (weights * counts).sum().detach().cpu().numpy().real.item()

        with pt.no_grad():
            analytic_freqs = pt.conj(sampled_amps) * sampled_amps
        analytic_freqs = analytic_freqs / pt.sum(analytic_freqs)
        mean_var_local_energy = pt.dot(var_local_energies, analytic_freqs)

        var_var_est = pt.dot(analytic_freqs.real,
                             pt.pow(var_local_energies - mean_var_local_energy, 2).real)
        var_ste_est = pt.sqrt(var_var_est)
        mean_var_local_energy = mean_var_local_energy.real.detach().cpu().numpy().item()

        return local_energies, sampled_indices, freqs, (mean_local_energy, var_est, ste_est,
                                             counts_sum, wcounts_sum), (mean_var_local_energy,
                                                                        var_var_est.detach().cpu().numpy().item(),
                                                                        var_ste_est.detach().cpu().numpy().item()), final_sample_num


def opt_step(wf: AbstractQuantumState,
             sample_num: int = None,
             min_unq_sample_num: int = None,
             max_unq_sample_num: int = None,
             min_sample_num: int = None,
             max_sample_num: int = None,
             ham: PauliObservable = None,
             opt: pt.optim.Optimizer = None,
             indices_split_size: int = None,
             use_sr: bool = False,
             max_sr_indices: int = None,
             use_reg_sr: bool = True,
             sr_reg: float = 1e-4,
             renorm_grad: bool = False,
             clip_grad_norm: float = None,
             verbose: bool = False,
             sampled_only: bool = True):
    opt.zero_grad()

    # Step 1. Sample the desired number of samples and evaluate local energy elements
    local_energies, sampled_indices, freqs, kpi_tuple, var_tuple, final_sample_num = sample_and_compute_local_energies(wf=wf,
                                                                                                                       sample_num=sample_num,
                                                                                                                       min_unq_sample_num=min_unq_sample_num,
                                                                                                                       max_unq_sample_num=max_unq_sample_num,
                                                                                                                       min_sample_num=min_sample_num,
                                                                                                                       max_sample_num=max_sample_num,
                                                                                                                       ham=ham,
                                                                                                                       indices_split_size=indices_split_size,
                                                                                                                       verbose=verbose,
                                                                                                                       sampled_only=sampled_only)
    if len(local_energies) > 0:
        # Step 2. Evaluate the loss estimator
        # Subtract mean local energy (because the estimator tells to do so
        local_energies = local_energies - pt.dot(local_energies, freqs)
        max_amp_indices_num = wf.hilbert_space.gpu_memory_limit // (2 * 16 * wf.width ** 2)
        amp_indices_boundaries = compute_chunk_boundaries(array_len=sampled_indices.shape[0],
                                                          chunk_size=max_amp_indices_num)
        for chunk_idx in verbosify_generator(range(len(amp_indices_boundaries) - 1),
                                             verbose=verbose,
                                             activity_descr='computing grad sampled amplitudes'):
            chunk_start = amp_indices_boundaries[chunk_idx]
            chunk_end = amp_indices_boundaries[chunk_idx + 1]
            cur_sampled_amps = wf.amplitude(sampled_indices[chunk_start:chunk_end])
            loss = 2 * (freqs[chunk_start:chunk_end] * pt.log(pt.conj(cur_sampled_amps)) * local_energies[chunk_start:chunk_end]).sum().real

            # Step 3. Backpropagate
            loss.backward()

        cent_cat_grads = []
        for param in wf.parameters():
            cent_cat_grads.append(pt.reshape(param.grad, (-1, )))
        cent_cat_grad = pt.cat(cent_cat_grads)
        cent_cat_grad = pt.complex(cent_cat_grad,
                                   pt.zeros_like(cent_cat_grad))

        # Step 4. Potentially use SR
        if use_sr:
            sorted_freqs, sorted_freq_indices = pt.sort(freqs.real, descending=True)
            sr_freqs = freqs[sorted_freq_indices[:max_sr_indices]]
            sr_freqs = sr_freqs / pt.sum(sr_freqs)
            sr_indices = sampled_indices[sorted_freq_indices[:max_sr_indices]]

            log_jacobians = [pt.complex(jac[:, 0, ...], jac[:, 1, ...]) for jac in compute_log_jacobians(wf, sr_indices)]
            log_grad = compute_log_grad(log_jacobians)
            log_grad = log_grad - (log_grad.T * sr_freqs).sum(dim=-1, keepdim=True).T

            if use_reg_sr:
                s_reg_inv = 1 / sr_reg
                o_matrix = s_reg_inv * pt.diag(pt.sqrt(sr_freqs)) @ log_grad.conj()

                t_matrix = hermitian_matmul(o_matrix)
                t_matrix_reg = pt.diag(pt.ones_like(pt.diag(t_matrix))) + sr_reg * t_matrix

                cent_cat_grad = s_reg_inv * cent_cat_grad - (o_matrix.T.conj() @ pt.linalg.solve(t_matrix_reg,
                                                                                                 o_matrix @ cent_cat_grad))
            else:
                o_matrix = pt.diag(pt.sqrt(sr_freqs)) @ log_grad.conj()

                t_matrix = hermitian_matmul(o_matrix)
                t_matrix_inv = soft_matrix_inv(t_matrix)

                cent_cat_grad = (o_matrix.T.conj() @ (t_matrix_inv.T.conj() @ (t_matrix_inv @ (o_matrix @ cent_cat_grad))))

        # Step 5. Apply gradients
        cent_cat_grad = cent_cat_grad.real
        if renorm_grad:
            cent_cat_grad = cent_cat_grad / pt.linalg.norm(cent_cat_grad)

        cent_cat_grads = pt.split(cent_cat_grad, wf.splits)
        for param_idx, param in enumerate(wf.parameters()):
            param.grad = pt.reshape(cent_cat_grads[param_idx], wf.shapes[param_idx]).type(param.dtype)
        if clip_grad_norm is not None:
            if isinstance(wf, LogAbsPhaseMADE):
                pt.nn.utils.clip_grad_norm_(wf.log_abs_mlp.parameters(), clip_grad_norm)
                pt.nn.utils.clip_grad_norm_(wf.phase_mlp.parameters(), clip_grad_norm)
            else:
                pt.nn.utils.clip_grad_norm_(wf.parameters(), clip_grad_norm)
        opt.step()

        return kpi_tuple, sampled_indices.shape[0], var_tuple, final_sample_num
    else:
        return (np.nan, np.nan, np.nan, np.nan, np.nan), np.nan, (np.nan, np.nan, np.nan), sample_num



