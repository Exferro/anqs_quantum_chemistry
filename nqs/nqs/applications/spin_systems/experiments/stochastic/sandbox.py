import torch as pt

import time

from openfermion.ops import QubitOperator

import pennylane as qml
from pennylane.qchem.convert import import_operator

from nqs.base.hilbert_space import HilbertSpace

from nqs.stochastic.symmetries.spin_half_projection_symmetry import SpinHalfProjectionSymmetry
from nqs.stochastic.symmetries.z2_symmetry import Z2Symmetry
from nqs.stochastic.maskers.locally_decomposable_masker import LocallyDecomposableMasker


from nqs.stochastic.observables.pauli_observable import PauliObservable

from nqs.stochastic.ansatzes.legacy.anqs_primitives import activation_combo2activations
from nqs.stochastic.ansatzes.legacy.anqs_primitives import MaskingMode
from nqs.stochastic.ansatzes.legacy.nade.log_abs_phase_nade import LogAbsPhaseNADE

from nqs.applications.quantum_chemistry.experiments.preparation.infrastructure import opt_step

qubit_num = 12

hs = HilbertSpace(qubit_num=qubit_num,
                  parent_dir='./',
                  device=pt.device('cpu'),
                  rng_seed=0)

tfi_ham = QubitOperator()
for qubit_idx in range(qubit_num):
    tfi_ham += QubitOperator(f'Z{qubit_idx} Z{(qubit_idx + 1) % qubit_num}', -1.0)
    #tfi_ham += QubitOperator(f'X{qubit_idx}', -0.5)

qml_ham = import_operator(tfi_ham)
generators = qml.symmetry_generators(qml_ham)

symmetries = [SpinHalfProjectionSymmetry(hilbert_space=hs,
                                        spin=0)]
for gen in generators[0:1]:
    z2_symmetry = Z2Symmetry(hilbert_space=hs,
                             pauli_z_positions=hs.qubit_num - pt.tensor(list(gen.ops[0].wires)[::-1],
                                                                        dtype=hs.idx_dtype,
                                                                        device=hs.device) - 1,
                             value=None)
    z2_symmetry.value = 1
    symmetries.append(z2_symmetry)

masker = LocallyDecomposableMasker(hilbert_space=hs,
                                   symmetries=tuple(symmetries))

activations = activation_combo2activations(activation_combo='tanh_layer_norm_then_leaky_relu',
                                           depth=2)
phase_activations = activation_combo2activations(activation_combo='tanh_layer_norm_then_leaky_relu',
                                                 depth=2)
wf = LogAbsPhaseNADE(hilbert_space=hs,
                     depth=2,
                     width=32,
                     log_abs_activations=activations,
                     phase_depth=2,
                     phase_width=32,
                     phase_activations=phase_activations,
                     masker=masker,
                     masking_mode=MaskingMode.part_base_vecs,
                     masking_depth=2,
                     perseverant_sampling='renorm',
                     split_size=250000)
wf.init_shapes_and_splits()

ham = PauliObservable(hilbert_space=hs,
                      of_qubit_operator=tfi_ham)

opt = pt.optim.Adam(wf.parameters(), lr=3e-4)

start_time = time.time()
for iter_idx in range(100000):
    print(f'Iteration #{iter_idx}')
    if iter_idx < 1000:
        kpi_tuple, unique_num, var_tuple, final_sample_num = opt_step(wf=wf,
                                                                      sample_num=10**3,
                                                                      ham=ham,
                                                                      opt=opt,
                                                                      indices_split_size=250000,
                                                                      use_sr=False,
                                                                      max_sr_indices=25,
                                                                      clip_grad_norm=True,
                                                                      verbose=False)
    else:
        kpi_tuple, unique_num, var_tuple, final_sample_num = opt_step(wf=wf,
                                                                      sample_num=10**4,
                                                                      ham=ham,
                                                                      opt=opt,
                                                                      indices_split_size=250000,
                                                                      use_sr=True,
                                                                      max_sr_indices=25,
                                                                      clip_grad_norm=True,
                                                                      verbose=False)
    energy, var_est, ste_est, counts_sum, wcounts_sum = kpi_tuple
    var_energy, var_var_est, var_ste_est = var_tuple
    sample_num = final_sample_num
    print(f'For next iteration we set sample_num to {sample_num}')
    print(f'<E> = {energy}, unq num: {unique_num}, Hilbert space sampled: {wcounts_sum / sample_num}')
    print(f'Var <E> = {var_energy}\n')
print(f'Time elapsed: {time.time() - start_time}')
