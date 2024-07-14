from __future__ import annotations

import numpy as np

import os
import json
import pickle

import subprocess

from jinja2 import Environment, PackageLoader

import pennylane as qml
from pennylane.qchem.convert import import_operator

from openfermion.transforms import jordan_wigner
from openfermion.chem import MolecularData as Psi4MolecularData
from openfermionpsi4 import run_psi4

from pennylane.pauli import PauliSentence, PauliWord, pauli_sentence, simplify
from pennylane.pauli.utils import _binary_matrix_from_pws
from pennylane.qchem.tapering import _reduced_row_echelon, _kernel
from pennylane.operation import active_new_opmath

from ...infrastructure import create_dir
from ...infrastructure.nested_data import Config
from .molecular_data import MolecularData
from .run_pyscf import run_pyscf


class GeometryConfig(Config):
    CLASS_SHORTHAND = 'geom'
    FIELDS = ('type', 'idx')

    def __init__(self,
                 *args,
                 type: str = 'paper',
                 idx: int = 0,
                 **kwargs):
        self.type = type
        self.idx = idx

        super().__init__(*args, **kwargs)


class MolConfig(Config):
    CLASS_SHORTHAND = 'mol'
    FIELDS = (
        'name',
        'geom_config',
        'basis',
        'multiplicity',
        'charge',
        'driver',
    )

    def __init__(self,
                 *args,
                 name: str = None,
                 geom_config: GeometryConfig = None,
                 basis: str = 'sto-3g',
                 multiplicity: int = 1,
                 charge: int = 0,
                 driver: str = 'pyscf',
                 **kwargs):
        self.name = name
        self.geom_config = geom_config if geom_config is not None else GeometryConfig()
        self.basis = basis
        self.multiplicity = multiplicity
        self.charge = charge
        assert driver in ('pyscf', 'psi4')
        self.driver = driver

        super().__init__(*args, **kwargs)


class MolInitConfig(Config):
    FIELDS = (
        'run_scf',
        'run_cisd',
        'run_ccsd',
        'run_fci',
    )

    def __init__(self,
                 *args,
                 run_scf: bool = True,
                 run_cisd: bool = True,
                 run_ccsd: bool = True,
                 run_fci: bool = True,
                 reinit: bool = False,
                 **kwargs):
        self.run_scf = run_scf
        self.run_cisd = run_cisd
        self.run_ccsd = run_ccsd
        self.run_fci = run_fci
        self.reinit = reinit

        super().__init__(*args, **kwargs)


class Molecule:
    FCI_NDET_FILENAME = 'psi_4_fci_ndet'

    SYM_LEVEL_TO_PSI_4 = {
        'z2': '',
        'e_num_spin': 'symmetry c1',
    }

    def __init__(self,
                 config: MolConfig,
                 mols_root_dir: str = '.',
                 init_config: MolInitConfig = None):
        if not isinstance(config, MolConfig):
            raise RuntimeError(
                f'A mol_config of the wrong type {config.__class__} was passed to the constructor of {self.__class__}')

        self.config = config

        self.root_dir = mols_root_dir
        self.dir = create_dir(os.path.join(self.root_dir, self.config.to_path_suffix()))
        self.filename = os.path.join(self.dir, 'molecule.pickle')

        self.geometry = self.load_geom(mol_config=self.config,
                                       mols_root_dir=self.root_dir)

        if not isinstance(init_config, MolInitConfig):
            raise RuntimeError(
                f'A mol_init_config of the wrong type {init_config.__class__} was passed to the constructor of {self.__class__}')
        self.init_config = init_config if init_config is not None else MolInitConfig()

        self.n_qubits = None
        self.n_electrons = None

        self.hf_energy = None
        self.cisd_energy = None
        self.ccsd_energy = None
        self.ccsd_t_energy = None
        self.fci_energy = None
        self.fci_wf_ipr = None

        self._qubit_ham = None
        self._ham_binary_matrix = None
        self._z2_generators = None

        self.initialise()

        if (self.config.basis == 'vdz') or (self.config.basis == 'SV'):
            self.z2_fci_ndet = None
            self.e_num_spin_fci_ndet = None
            print(f'We fucked up Psi4, but we do PySCF anyway')

        else:
            # self.z2_fci_ndet = self.sym_level_to_fci_ndet('z2')
            # self.e_num_spin_fci_ndet = self.sym_level_to_fci_ndet('e_num_spin')
            self.z2_fci_ndet = None
            self.e_num_spin_fci_ndet = None

        self.save()

    @property
    def qubit_ham(self):
        if self._qubit_ham is None:
            qubit_ham_filename = os.path.join(self.dir, 'qubit_ham.pickle')
            assert os.path.exists(qubit_ham_filename)
            with open(qubit_ham_filename, 'rb') as handle:
                self._qubit_ham = pickle.load(handle)

        return self._qubit_ham

    @property
    def ham_binary_matrix(self):
        if self._ham_binary_matrix is None:
            ham_binary_matrix_filename = os.path.join(self.dir, 'ham_binary_matrix.npy')
            if os.path.exists(ham_binary_matrix_filename):
                self._ham_binary_matrix = np.load(ham_binary_matrix_filename)
            else:
                self._ham_binary_matrix = self.qubit_ham2ham_binary_matrix(self.qubit_ham)
                np.save(ham_binary_matrix_filename, self._ham_binary_matrix)

        return self._ham_binary_matrix

    def qubit_ham2ham_binary_matrix(self, qubit_ham=None):
        num_qubits = self.n_qubits
        binary_matrix = np.zeros((len(qubit_ham.terms), 2 * num_qubits), dtype=int)
        for idx, term in enumerate(qubit_ham.terms):
            for (qubit_idx, pauli_op) in term:
                if pauli_op in ["X", "Y"]:
                    binary_matrix[idx][qubit_idx + num_qubits] = 1
                if pauli_op in ["Z", "Y"]:
                    binary_matrix[idx][qubit_idx] = 1

        return binary_matrix

    @property
    def z2_generators(self):
        if self._z2_generators is None:
            z2_generators_filename = os.path.join(self.dir, 'z2_generators.pickle')
            if os.path.exists(z2_generators_filename):
                with open(z2_generators_filename, 'rb') as handle:
                    self._z2_generators = pickle.load(handle)
            else:
                self._z2_generators = self.ham_binary_matrix2z2_generators(self.ham_binary_matrix)
                with open(z2_generators_filename, 'wb') as handle:
                    pickle.dump(self._z2_generators, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return self._z2_generators

    def ham_binary_matrix2z2_generators(self, ham_binary_matrix=None):
        num_qubits = self.n_qubits
        wires = qml.wires.Wires(tuple(range(num_qubits)))

        rref_binary_matrix = _reduced_row_echelon(ham_binary_matrix)
        rref_binary_matrix_red = rref_binary_matrix[
            ~np.all(rref_binary_matrix == 0, axis=1)
        ]  # remove all-zero rows

        # Get kernel (i.e., nullspace) for trimmed binary matrix using gaussian elimination
        nullspace = _kernel(rref_binary_matrix_red)

        generators = []
        pauli_map = {"00": "I", "10": "X", "11": "Y", "01": "Z"}

        for null_vector in nullspace:
            tau = {}
            for idx, op in enumerate(zip(null_vector[:num_qubits], null_vector[num_qubits:])):
                x, z = op
                tau[idx] = pauli_map[f"{x}{z}"]

            ham = qml.pauli.PauliSentence({qml.pauli.PauliWord(tau): 1.0})
            ham = ham.operation(wires) if active_new_opmath() else ham.hamiltonian(wires)
            generators.append(ham)

        return generators

    @staticmethod
    def load_geom(mol_config: MolConfig = None,
                  mols_root_dir: str = None):
        geom_dir = os.path.join(mols_root_dir,
                                f'name={mol_config.name}',
                                'geometries',
                                mol_config.geom_config.to_path_suffix())
        geom_filename = os.path.join(geom_dir, 'geom.json')

        if os.path.exists(geom_filename):
            with open(geom_filename, 'r') as f:
                geom = json.load(f)
                return geom
        else:
            raise RuntimeError(f'There is no existing geometry at {geom_filename}')

    def initialise(self, init_config: MolInitConfig = None):
        if init_config is not None:
            self.init_config = init_config

        if self.config.driver == 'pyscf':
            my_old_mol = MolecularData(geometry=self.geometry,
                                       basis=self.config.basis,
                                       multiplicity=self.config.multiplicity,
                                       charge=self.config.charge,
                                       data_directory=self.dir)

            run_pyscf(molecule=my_old_mol,
                      run_scf=self.init_config.run_scf,
                      run_cisd=self.init_config.run_cisd,
                      run_ccsd=self.init_config.run_ccsd,
                      run_fci=self.init_config.run_fci)
        else:
            raise NotImplementedError(f'We do not know how to run calculations with Psi4 yet.')

        self.n_qubits = my_old_mol.n_qubits
        self.n_electrons = my_old_mol.n_electrons

        self.hf_energy = my_old_mol.hf_energy
        self.cisd_energy = my_old_mol.cisd_energy
        self.ccsd_energy = my_old_mol.ccsd_energy
        self.ccsd_t_energy = my_old_mol.ccsd_t_energy
        self.fci_energy = my_old_mol.fci_energy
        self.fci_wf_ipr = my_old_mol.ipr

        self._qubit_ham = my_old_mol.qubit_ham

    def compute_symmetry_generators(self):
        wires = qml.wires.Wires(tuple(range(self.n_qubits)))

        num_qubits = len(wires)

        # Generate binary matrix for qubit_op
        ps = PauliSentence()
        term_num = len(self.qubit_ham.terms)
        term_idx = 0
        for term, coeff in self.qubit_ham.terms.items():
            print(f'Transferring term #{term_idx}/{term_num}')
            term_idx += 1
            pw = {}
            for qubit_idx, pauli_op in term:
                pw[qubit_idx] = pauli_op
            pw = PauliWord(pw)
            ps[pw] = coeff
        # ps = pauli_sentence(h)
        binary_matrix = _binary_matrix_from_pws(list(ps), num_qubits)

        # Get reduced row echelon form of binary matrix
        rref_binary_matrix = _reduced_row_echelon(binary_matrix)
        rref_binary_matrix_red = rref_binary_matrix[
            ~np.all(rref_binary_matrix == 0, axis=1)
        ]  # remove all-zero rows

        # Get kernel (i.e., nullspace) for trimmed binary matrix using gaussian elimination
        nullspace = _kernel(rref_binary_matrix_red)

        generators = []
        pauli_map = {"00": "I", "10": "X", "11": "Y", "01": "Z"}

        for null_vector in nullspace:
            tau = {}
            for idx, op in enumerate(zip(null_vector[:num_qubits], null_vector[num_qubits:])):
                x, z = op
                tau[idx] = pauli_map[f"{x}{z}"]

            ham = qml.pauli.PauliSentence({qml.pauli.PauliWord(tau): 1.0})
            ham = ham.operation(wires) if active_new_opmath() else ham.hamiltonian(wires)
            generators.append(ham)

        return generators

    def save(self):
        self._qubit_ham = None
        self._ham_binary_matrix = None
        with open(self.filename, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls,
             config: MolConfig = None,
             mols_root_dir: str = None) -> Molecule:
        mol_dir = create_dir(os.path.join(mols_root_dir, config.to_path_suffix()))
        mol_filename = os.path.join(mol_dir, 'molecule.pickle')
        with open(mol_filename, 'rb') as handle:
            return pickle.load(handle)

    @classmethod
    def create(cls,
               config: MolConfig = None,
               init_config: MolInitConfig = None,
               mols_root_dir: str = None):
        if init_config is None:
            init_config = MolInitConfig()
        mols_root_dir = os.path.abspath(mols_root_dir)
        mol_dir = create_dir(os.path.join(mols_root_dir, config.to_path_suffix()))
        mol_filename = os.path.join(mol_dir, 'molecule.pickle')
        if os.path.exists(mol_filename):
            print(f'The molecule filename exists')
            mol = Molecule.load(config=config,
                                mols_root_dir=mols_root_dir)
            if init_config.reinit:
                print(f'Init config prescribes us to reinit the Molecule, and so we do')
                mol.init_config = init_config
                mol.initialise()
                mol.save()
            else:
                print(f'The init config does not prescribe to reinit the Molecule')
        else:
            print(f'The molecule didn\'t exist')
            mol = Molecule(config=config,
                           init_config=init_config,
                           mols_root_dir=mols_root_dir)
            mol.save()

        return mol

    @staticmethod
    def generate_geometry_string(geometry):
        geo_string = ''
        for item in geometry:
            atom = item[0]
            coordinates = item[1]
            line = '    {} {} {} {}'.format(atom,
                                        coordinates[0],
                                        coordinates[1],
                                        coordinates[2])
            if len(geo_string) > 0:
                geo_string += '\n'
            geo_string += line

        return geo_string

    def sym_level_to_fci_ndet(self,
                               sym_level: str = 'z2'):
        assert sym_level in self.SYM_LEVEL_TO_PSI_4

        env = Environment(loader=PackageLoader('nqs', 'templates'))
        template = env.get_template(self.FCI_NDET_FILENAME)
        dump_filename = os.path.join(self.dir, f'{sym_level}_fci_ndet')
        if os.path.exists(dump_filename):
            with open(dump_filename, 'r') as f:
                return int(f.read())
        else:
            lines = template.render(mol_config=self.config,
                                    geo_string=self.generate_geometry_string(self.geometry),
                                    symmetry_level=self.SYM_LEVEL_TO_PSI_4[sym_level],
                                    dump_filename=dump_filename)
            input_filename = os.path.join(self.dir,
                                          f'{self.FCI_NDET_FILENAME}_{sym_level}' + '.in')

            with open(input_filename, 'w') as input_file:
                input_file.write(''.join(lines))

            output_filename = os.path.join(self.dir,
                                           f'{self.FCI_NDET_FILENAME}_{sym_level}' + '.out')
            try:
                process = subprocess.Popen(['psi4', input_filename, output_filename])
                process.wait()
            except:
                print(f'Psi4 calculation for {self.config} '
                                   f'geometry failed')
                process.kill()

            finally:
                cur_dir = os.getcwd()
                for local_file in os.listdir(cur_dir):
                    if local_file.endswith('.clean'):
                        os.remove(os.path.join(cur_dir, local_file))
                try:
                    os.remove(os.path.join(cur_dir, 'timer.dat'))
                except Exception:
                    pass
                os.remove(input_filename)
                os.remove(output_filename)

                if os.path.exists(dump_filename):
                    with open(dump_filename, 'r') as f:
                        return int(f.read())
                else:
                    return None
