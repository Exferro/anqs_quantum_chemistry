import torch as pt

from .....base.hilbert_space import HilbertSpace
from .....infrastructure.nested_data import Config

from .....stochastic.maskers import LocallyDecomposableMasker
from .....stochastic.maskers import ALLOWED_SYMMETRY_LEVELS

from ...molecule import Molecule
from .....stochastic.symmetries import IdleSymmetry, ParticleNumberSymmetry, SpinHalfProjectionSymmetry, Z2Symmetry


class MaskerConfig(Config):
    FIELDS = (
        'symmetry_level',
    )

    def __init__(self,
                 *args,
                 symmetry_level: str = 'z2',
                 **kwargs):
        self.symmetry_level = symmetry_level

        super().__init__(*args, **kwargs)


def create_z2_symmetries(hs: HilbertSpace = None,
                         mol: Molecule = None,):
    hf_base_vec = [0] * (mol.n_qubits - mol.n_electrons) + [1] * mol.n_electrons
    hf_base_vec = pt.tensor([hf_base_vec],
                            dtype=hs.idx_dtype,
                            device=hs.device)
    hf_base_vec = hf_base_vec[..., hs.inv_perm]

    generators = mol.z2_generators

    z2_symmetries = []
    for gen in generators:
        pauli_z_positions = hs.qubit_num - pt.tensor(list(gen.ops[0].wires)[::-1],
                                                     dtype=hs.idx_dtype,
                                                     device=hs.device) - 1

        pauli_z_positions = hs.perm[pauli_z_positions]
        z2_symmetry = Z2Symmetry(hilbert_space=hs,
                                 pauli_z_positions=pauli_z_positions,
                                 value=None)
        z2_symmetry.value = z2_symmetry.compute_acc_eig(hf_base_vec)
        z2_symmetries.append(z2_symmetry)

    return tuple(z2_symmetries)


def create_masker(config: MaskerConfig = None,
                  hs: HilbertSpace = None,
                  mol: Molecule = None) -> LocallyDecomposableMasker:
    config = config if config is not None else MaskerConfig()
    assert config.symmetry_level in ALLOWED_SYMMETRY_LEVELS
    if config.symmetry_level == 'no_sym':
        symmetries = (IdleSymmetry(hilbert_space=hs),)
    else:
        if config.symmetry_level == 'e_num_spin' or config.symmetry_level == 'z2':
            symmetries = (ParticleNumberSymmetry(hilbert_space=hs,
                                                 particle_num=mol.n_electrons),
                          SpinHalfProjectionSymmetry(hilbert_space=hs,
                                                     spin=mol.config.multiplicity - 1))

            if config.symmetry_level == 'z2':
                symmetries = symmetries + create_z2_symmetries(hs=hs,
                                                               mol=mol)

        else:
            raise RuntimeError(f'Wrong level of symmetry: {config.symmetry_level}')
    masker = LocallyDecomposableMasker(hilbert_space=hs,
                                       symmetries=symmetries)

    return masker
