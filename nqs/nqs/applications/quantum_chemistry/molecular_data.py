import numpy as np

import os
import uuid
import shutil
import h5py
import pickle

from openfermionpyscf import PyscfMolecularData

from openfermion import get_fermion_operator
from openfermion.transforms import jordan_wigner
from openfermion import get_sparse_operator

from scipy.sparse.linalg import eigsh as sparse_eigsh


from typing import List, Tuple


class MolecularData(PyscfMolecularData):
    def __init__(self,
                 geometry: List[Tuple[str, Tuple[float, float, float]]] = '',
                 basis: str = '',
                 multiplicity: int = 1,
                 charge: int = 0,
                 description: str = '',
                 filename: str = None,
                 data_directory: str = None,
                 max_fci_qubits: int = 20):
        super(MolecularData, self).__init__(geometry,
                                            basis,
                                            multiplicity,
                                            charge,
                                            description,
                                            filename,
                                            data_directory)
        self.data_directory = data_directory
        self.qubit_ham_filename = os.path.join(self.data_directory,
                                               'qubit_ham.pickle')

        self.ccsd_t_energy = None
        self._molecular_ham = None
        self._qubit_ham = None
        self.ipr = None

        self._sparse_ham = None

        self.fci_data = {}
        self.max_fci_qubits = max_fci_qubits

    @property
    def molecular_ham(self):
        if self._molecular_ham is None:
            assert (self.one_body_integrals is not None) and (self.two_body_integrals is not None)
            self._molecular_ham = self.get_molecular_hamiltonian()

        return self._molecular_ham

    @property
    def qubit_ham(self):
        if self._qubit_ham is None:
            self._qubit_ham = jordan_wigner(self.molecular_ham)
            self.save()

        return self._qubit_ham

    @property
    def sparse_ham(self):
        assert self.n_qubits <= self.max_fci_qubits
        if self._sparse_ham is None:
            self._sparse_ham = get_sparse_operator(self.qubit_ham)

        return self._sparse_ham

    @property
    def fci_wf(self):
        if self.fci_data.get('wf', None) is None:
            self.run_bf_fci()

        return self.fci_data['wf']

    def run_bf_fci(self):
        assert self.n_qubits <= self.max_fci_qubits
        v, w = sparse_eigsh(self.sparse_ham, which='SA')
        self.fci_energy = float(np.min(v))
        self.fci_data['wf'] = w[:, np.argmin(v)]

        return self.fci_energy, self.fci_wf

    def save(self):
        """Method to save the class under a systematic name."""
        # Create a temporary file and swap it to the original name in case
        # data needs to be loaded while saving
        tmp_name = uuid.uuid4()
        with h5py.File("{}.hdf5".format(tmp_name), "w") as f:
            # Save geometry (atoms and positions need to be separate):
            d_geom = f.create_group("geometry")
            if not isinstance(self.geometry, str):
                atoms = [np.string_(item[0]) for item in self.geometry]
                positions = np.array(
                    [list(item[1]) for item in self.geometry])
            else:
                atoms = np.string_(self.geometry)
                positions = None
            d_geom.create_dataset("atoms",
                                  data=(atoms if atoms is not None else False))
            d_geom.create_dataset(
                "positions",
                data=(positions if positions is not None else False))
            # Save basis:
            f.create_dataset("basis", data=np.string_(self.basis))
            # Save multiplicity:
            f.create_dataset("multiplicity", data=self.multiplicity)
            # Save charge:
            f.create_dataset("charge", data=self.charge)
            # Save description:
            f.create_dataset("description",
                             data=np.string_(self.description))
            # Save name:
            f.create_dataset("name", data=np.string_(self.name))
            # Save n_atoms:
            f.create_dataset("n_atoms", data=self.n_atoms)
            # Save atoms:
            f.create_dataset("atoms", data=np.string_(self.atoms))
            # Save protons:
            f.create_dataset("protons", data=self.protons)
            # Save n_electrons:
            f.create_dataset("n_electrons", data=self.n_electrons)
            # Save generic attributes from calculations:
            f.create_dataset("n_orbitals",
                             data=(self.n_orbitals
                                   if self.n_orbitals is not None else False))
            f.create_dataset(
                "n_qubits",
                data=(self.n_qubits if self.n_qubits is not None else False))
            f.create_dataset(
                "nuclear_repulsion",
                data=(self.nuclear_repulsion
                      if self.nuclear_repulsion is not None else False))
            # Save attributes generated from SCF calculation.
            f.create_dataset(
                "hf_energy",
                data=(self.hf_energy if self.hf_energy is not None else False))
            f.create_dataset(
                "canonical_orbitals",
                data=(self.canonical_orbitals
                      if self.canonical_orbitals is not None else False),
                compression=("gzip"
                             if self.canonical_orbitals is not None else None))
            f.create_dataset(
                "overlap_integrals",
                data=(self.overlap_integrals
                      if self.overlap_integrals is not None else False),
                compression=("gzip"
                             if self.overlap_integrals is not None else None))
            f.create_dataset(
                "orbital_energies",
                data=(self.orbital_energies
                      if self.orbital_energies is not None else False))
            # Save attributes generated from integrals.
            f.create_dataset(
                "one_body_integrals",
                data=(self.one_body_integrals
                      if self.one_body_integrals is not None else False),
                compression=("gzip"
                             if self.one_body_integrals is not None else None))
            f.create_dataset(
                "two_body_integrals",
                data=(self.two_body_integrals
                      if self.two_body_integrals is not None else False),
                compression=("gzip"
                             if self.two_body_integrals is not None else None))
            # Save attributes generated from MP2 calculation.
            f.create_dataset("mp2_energy",
                             data=(self.mp2_energy
                                   if self.mp2_energy is not None else False))
            # Save attributes generated from CISD calculation.
            f.create_dataset("cisd_energy",
                             data=(self.cisd_energy
                                   if self.cisd_energy is not None else False))
            f.create_dataset(
                "cisd_one_rdm",
                data=(self.cisd_one_rdm
                      if self.cisd_one_rdm is not None else False),
                compression=("gzip" if self.cisd_one_rdm is not None else None))
            f.create_dataset(
                "cisd_two_rdm",
                data=(self.cisd_two_rdm
                      if self.cisd_two_rdm is not None else False),
                compression=("gzip" if self.cisd_two_rdm is not None else None))
            # Save attributes generated from exact diagonalization.
            f.create_dataset("fci_energy",
                             data=(self.fci_energy
                                   if self.fci_energy is not None else False))
            f.create_dataset(
                "fci_one_rdm",
                data=(self.fci_one_rdm
                      if self.fci_one_rdm is not None else False),
                compression=("gzip" if self.fci_one_rdm is not None else None))
            f.create_dataset(
                "fci_two_rdm",
                data=(self.fci_two_rdm
                      if self.fci_two_rdm is not None else False),
                compression=("gzip" if self.fci_two_rdm is not None else None))
            # Save attributes generated from CCSD calculation.
            f.create_dataset("ccsd_energy",
                             data=(self.ccsd_energy
                                   if self.ccsd_energy is not None else False))
            f.create_dataset(
                "ccsd_single_amps",
                data=(self.ccsd_single_amps
                      if self.ccsd_single_amps is not None else False),
                compression=("gzip"
                             if self.ccsd_single_amps is not None else None))
            f.create_dataset(
                "ccsd_double_amps",
                data=(self.ccsd_double_amps
                      if self.ccsd_double_amps is not None else False),
                compression=("gzip"
                             if self.ccsd_double_amps is not None else None))
            # Save attributes generated from CCSD(T) calculation.
            f.create_dataset("ccsd_t_energy",
                             data=(self.ccsd_t_energy
                                   if self.ccsd_t_energy is not None else False))
            # Save general calculation data
            key_list = list(self.general_calculations.keys())
            f.create_dataset("general_calculations_keys",
                             data=([np.string_(key) for key in key_list]
                                   if len(key_list) > 0 else False))
            f.create_dataset(
                "general_calculations_values",
                data=([self.general_calculations[key] for key in key_list]
                      if len(key_list) > 0 else False))

            # Save data from Jordan-Wigner calculation.
            if self._qubit_ham is not None:
                with open(self.qubit_ham_filename, 'wb') as handle:
                    pickle.dump(self._qubit_ham, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Remove old file first for compatibility with systems that don't allow
        # rename replacement.  Catching OSError for when file does not exist
        # yet
        try:
            os.remove("{}.hdf5".format(self.filename))
        except OSError:
            pass

        shutil.move("{}.hdf5".format(tmp_name), "{}.hdf5".format(self.filename))

    def load(self):
        geometry = []
        with h5py.File("{}.hdf5".format(self.filename), "r") as f:
            # Load geometry:
            data = f["geometry/atoms"]
            if data.shape != (()):
                for atom, pos in zip(f["geometry/atoms"][...],
                                     f["geometry/positions"][...]):
                    geometry.append((atom.tobytes().decode('utf-8'), list(pos)))
                self.geometry = geometry
            else:
                self.geometry = data[...].tobytes().decode('utf-8')
            # Load basis:
            self.basis = f["basis"][...].tobytes().decode('utf-8')
            # Load multiplicity:
            self.multiplicity = int(f["multiplicity"][...])
            # Load charge:
            self.charge = int(f["charge"][...])
            # Load description:
            self.description = f["description"][...].tobytes().decode(
                'utf-8').rstrip(u'\x00')
            # Load name:
            self.name = f["name"][...].tobytes().decode('utf-8')
            # Load n_atoms:
            self.n_atoms = int(f["n_atoms"][...])
            # Load atoms:
            self.atoms = f["atoms"][...]
            # Load protons:
            self.protons = f["protons"][...]
            # Load n_electrons:
            self.n_electrons = int(f["n_electrons"][...])
            # Load generic attributes from calculations:
            data = f["n_orbitals"][...]
            self.n_orbitals = int(data) if data.dtype.num != 0 else None
            data = f["n_qubits"][...]
            self.n_qubits = int(data) if data.dtype.num != 0 else None
            data = f["nuclear_repulsion"][...]
            self.nuclear_repulsion = (float(data)
                                      if data.dtype.num != 0 else None)
            # Load attributes generated from SCF calculation.
            data = f["hf_energy"][...]
            self.hf_energy = data if data.dtype.num != 0 else None
            data = f["orbital_energies"][...]
            self.orbital_energies = data if data.dtype.num != 0 else None
            # Load attributes generated from MP2 calculation.
            data = f["mp2_energy"][...]
            self.mp2_energy = data if data.dtype.num != 0 else None
            # Load attributes generated from CISD calculation.
            data = f["cisd_energy"][...]
            self.cisd_energy = data if data.dtype.num != 0 else None
            # Load attributes generated from exact diagonalization.
            data = f["fci_energy"][...]
            self.fci_energy = data if data.dtype.num != 0 else None
            # Load attributes generated from CCSD calculation.
            data = f["ccsd_energy"][...]
            self.ccsd_energy = data if data.dtype.num != 0 else None
            # Load attributes generated from CCSD(T) calculation.
            data = f["ccsd_t_energy"][...]
            self.ccsd_t_energy = data if data.dtype.num != 0 else None
            # Load general calculations
            if ("general_calculations_keys" in f and
                    "general_calculations_values" in f):
                keys = f["general_calculations_keys"]
                values = f["general_calculations_values"]
                if keys.shape != (()):
                    self.general_calculations = {
                        key.tobytes().decode('utf-8'): value
                        for key, value in zip(keys[...], values[...])
                    }
            else:
                # TODO: test the no cover
                # no coverage here because pathway is check on
                # bad user generated file
                self.general_calculations = None  # pragma: nocover

            # Load data from Jordan-Wigner calculation.
            if os.path.exists(self.qubit_ham_filename):
                with open(self.qubit_ham_filename, 'rb') as handle:
                    self._qubit_ham = pickle.load(handle)
