import torch as pt
import time

from nqs.base.hilbert_space import HilbertSpace
from nqs.base.constants import BASE_REAL_TYPE

from nqs.applications.quantum_chemistry.experiments.preparation.infrastructure import ExpDescr
from nqs.applications.quantum_chemistry.experiments.preparation.infrastructure import MolDescr
from nqs.applications.quantum_chemistry.experiments.preparation.infrastructure import create_mol, create_masker
from nqs.applications.quantum_chemistry.experiments.preparation.infrastructure import opt_step
from nqs.applications.quantum_chemistry import CHEMICAL_ACCURACY

from nqs.stochastic.observables.pauli_observable import PauliObservable

from nqs.stochastic.ansatzes.base.log_abs_phase_anqs import LogAbsPhaseANQS
from nqs.stochastic.ansatzes.base.abstract_anqs import DEMode, HeadAggMode
from nqs.stochastic.ansatzes.base.abstract_anqs import MaskingMode as NewMaskingMode

mol_descr = MolDescr(mol_name='NH3',
                     geom_type='carleo',
                     geom_idx=0,
                     basis='sto-3g',
                     multiplicity=1,
                     charge=0)

mols_dir = './../../quantum_chemistry/mols'
mol, mol_dir = create_mol(mol_descr=mol_descr,
                          mols_dir=mols_dir)

exp_descr = ExpDescr(name='sandbox',
                     mol_name=mol_descr.mol_name,
                     geom_type=mol_descr.geom_type,
                     geom_idx=0,
                     basis='sto-3g',
                     multiplicity=1,
                     charge=0,
                     device='gpu',
                     ansatz='LogAbsPhaseANQS',
                     dtype=BASE_REAL_TYPE,
                     de_mode=DEMode.MADE,
                     head_num=2,
                     head_agg_mode=HeadAggMode.exp,
                     depth=2,
                     width=64,
                     activation_combo=None,
                     add_bias=True,
                     use_residuals=True,
                     phase_depth=2,
                     phase_width=256,
                     phase_activation_combo=None,
                     phase_add_bias=True,
                     phase_use_residuals=True,
                     iter_num=100,
                     sample_num=10**8,
                     symmetry_level='z2',
                     sampling_mode='masked_part_base_vecs',
                     masking_depth=0,
                     opt_class='Adam',
                     lr=1e-4,
                     indices_split_size=15000,
                     use_sr=False,
                     max_sr_indices=25,
                     use_reg_sr=True,
                     sr_reg=1e-4,
                     renorm_grad=False,
                     clip_grad_norm=True,
                     rng_seed=2)

hs = HilbertSpace(qubit_num=mol.n_qubits,
                  parent_dir=mol.data_directory,
                  device=pt.device('cpu'),
                  rng_seed=0)

masker = create_masker(exp_descr=exp_descr,
                       hs=hs,
                       mol=mol)

wf = LogAbsPhaseANQS(hilbert_space=hs,
                     dtype=BASE_REAL_TYPE,
                     de_mode=DEMode.MADE,
                     head_num=2,
                     head_agg_mode=HeadAggMode.exp,
                     normalise_heads_before_agg=True,
                     log_abs_depth=4,
                     log_abs_widthes=2*hs.qubit_num,
                     log_abs_use_residuals=True,
                     log_abs_add_bias=True,
                     log_abs_activations=pt.tanh,
                     phase_depth=4,
                     phase_widthes=2*hs.qubit_num,
                     phase_use_residuals=True,
                     phase_add_bias=True,
                     phase_activations=pt.tanh,
                     masker=masker,
                     masking_mode=NewMaskingMode.part_base_vecs,
                     masking_depth=2,
                     perseverant_sampling='renorm',
                     split_size=250000)

wf.to(hs.device)
wf.init_shapes_and_splits()

numel = 0
for param in wf.parameters():
    numel += param.numel()
   # param.data = param.data / 10
print(numel)

# for iter_idx in range(50):
#     indices, _, _ = wf.sample_stats(10**8)

# print(indices.shape)

opt = pt.optim.Adam(wf.parameters(), lr=1e-3)
ham = PauliObservable(hilbert_space=hs,
                      of_qubit_operator=mol.qubit_ham)
for method in ('hf', 'cisd', 'ccsd', 'ccsd_t', 'fci'):
    print(f'{method} energy: {getattr(mol, f"{method}_energy")}')
if mol.fci_energy is not None:
    print(f'fci energy up to chem. acc.: {mol.fci_energy + CHEMICAL_ACCURACY}')
#print(masker.local_eigs)


start_time = time.time()
for iter_idx in range(100):
    print(f'Iteration #{iter_idx}')
#     sample_num = 10**6
#     if iter_idx == 50:
#         sample_num = 10**7
#     if iter_idx == 100:
#         sample_num = 10**8
#     if iter_idx == 200:
#         sample_num = 10**9
    kpi_tuple, unique_num, var_tuple, final_sample_num = opt_step(wf=wf,
                                                                  sample_num=10**8,
                                                                  ham=ham,
                                                                  opt=opt,
                                                                  indices_split_size=250000,
                                                                  use_sr=True,
                                                                  use_reg_sr=True,
                                                                  max_sr_indices=25,
                                                                  clip_grad_norm=True,
                                                                  renorm_grad=False,
                                                                  sr_reg=1e-4,
                                                                  verbose=False)
    energy, var_est, ste_est, counts_sum, wcounts_sum = kpi_tuple
    var_energy, var_var_est, var_ste_est = var_tuple
    sample_num = final_sample_num
    print(f'For next iteration we set sample_num to {sample_num}')
    print(f'<E> = {energy}, unq num: {unique_num}, Hilbert space sampled: {wcounts_sum / sample_num}')
    print(f'Var <E> = {var_energy}\n')
print(f'Time elapsed: {time.time() - start_time}')