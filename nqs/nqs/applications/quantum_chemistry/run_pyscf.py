#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""Driver to initialize molecular object from pyscf program."""

from __future__ import absolute_import

from functools import reduce

import numpy
import pyscf
from pyscf import gto, scf, ao2mo, ci, cc, fci, mp

from openfermion import MolecularData
from openfermionpyscf import PyscfMolecularData

cr_vdz_basis_string = '''
BASIS "ao basis" SPHERICAL PRINT
#BASIS SET: (14s,8p,5d) -> [5s,2p,2d]
Cr    S
  51528.086349               0.14405823106E-02
   7737.2103487              0.11036202287E-01
   1760.3748470              0.54676651806E-01
    496.87706544             0.18965038103
    161.46520598             0.38295412850
     55.466352268            0.29090050668
Cr    S
    107.54732999            -0.10932281100
     12.408671897            0.64472599471
      5.0423628826           0.46262712560
Cr    S
      8.5461640165          -0.22711013286
      1.3900441221           0.73301527591
      0.56066602876          0.44225565433
Cr    S
      0.71483705972E-01      1.0000000000
Cr    S
      0.28250687604E-01      1.0000000000
Cr    P
    640.48536096             0.96126715203E-02
    150.69711194             0.70889834655E-01
     47.503755296            0.27065258990
     16.934120165            0.52437343414
      6.2409680590           0.34107994714
Cr    P
      3.0885463206           0.33973986903
      1.1791047769           0.57272062927
      0.43369774432          0.24582728206
Cr    D
     27.559479426            0.30612488044E-01
      7.4687020327           0.15593270944
      2.4345903574           0.36984421276
      0.78244754808          0.47071118077
Cr    D
      0.21995774311          0.33941649889
END'''

sv_basis = {
    'Cr': gto.basis.parse('''
    Cr S
    51528.086349 0.0014405823106
    7737.2103487 0.011036202287
    1760.3748470 0.054676651806
    496.87706544 0.18965038103
    161.46520598 0.38295412850
    55.466352268 0.29090050668
    Cr S
    107.54732999 -0.10932281100
    12.408671897 0.64472599471
    5.0423628826 0.46262712560
    Cr S
    8.5461640165 -0.22711013286
    1.3900441221 0.73301527591
    0.56066602876 0.44225565433
    Cr S
    0.071483705972 1.0000000000
    Cr S
    0.028250687604 1.0000000000
    Cr P
    640.48536096 0.0096126715203
    150.69711194 0.070889834655
    47.503755296 0.27065258990
    16.934120165 0.52437343414
    6.2409680590 0.34107994714
    Cr P
    3.0885463206 0.33973986903
    1.1791047769 0.57272062927
    0.43369774432 0.24582728206
    Cr D
    27.559479426 0.030612488044
    7.4687020327 0.15593270944
    2.4345903574 0.36984421276
    0.78244754808 0.47071118077
    Cr D
    0.21995774311 0.33941649889
    ''')
}

def prepare_pyscf_molecule(molecule):
    """
    This function creates and saves a pyscf input file.

    Args:
        molecule: An instance of the MolecularData class.

    Returns:
        pyscf_molecule: A pyscf molecule instance.
    """
    pyscf_molecule = pyscf.M()
    pyscf_molecule.atom = molecule.geometry
    print(pyscf_molecule.atom)
    pyscf_molecule.basis = molecule.basis
    if molecule.basis == 'vdz':
        assert len(molecule.geometry) == 2
        assert molecule.geometry[0][0] == 'Cr'
        assert molecule.geometry[1][0] == 'Cr'
        pyscf_molecule.basis = {'CR': gto.basis.parse(cr_vdz_basis_string),}
    elif molecule.basis == 'SV':
        assert len(molecule.geometry) == 2
        assert molecule.geometry[0][0] == 'Cr'
        assert molecule.geometry[1][0] == 'Cr'
        pyscf_molecule.basis = sv_basis

    pyscf_molecule.spin = molecule.multiplicity - 1
    pyscf_molecule.charge = molecule.charge
    pyscf_molecule.symmetry = True
    pyscf_molecule.build()
    print(pyscf_molecule.atom)

    return pyscf_molecule


def compute_scf(pyscf_molecule):
    """
    Perform a Hartree-Fock calculation.

    Args:
        pyscf_molecule: A pyscf molecule instance.

    Returns:
        pyscf_scf: A PySCF "SCF" calculation object.
    """
    if pyscf_molecule.spin:
        pyscf_scf = scf.ROHF(pyscf_molecule)
    else:
        pyscf_scf = scf.RHF(pyscf_molecule)
    return pyscf_scf


def compute_integrals(pyscf_molecule, pyscf_scf):
    """
    Compute the 1-electron and 2-electron integrals.

    Args:
        pyscf_molecule: A pyscf molecule instance.
        pyscf_scf: A PySCF "SCF" calculation object.

    Returns:
        one_electron_integrals: An N by N array storing h_{pq}
        two_electron_integrals: An N by N by N by N array storing h_{pqrs}.
    """
    # Get one electrons integrals.
    n_orbitals = pyscf_scf.mo_coeff.shape[1]
    one_electron_compressed = reduce(numpy.dot, (pyscf_scf.mo_coeff.T,
                                                 pyscf_scf.get_hcore(),
                                                 pyscf_scf.mo_coeff))
    one_electron_integrals = one_electron_compressed.reshape(
        n_orbitals, n_orbitals).astype(float)

    # Get two electron integrals in compressed format.
    two_electron_compressed = ao2mo.kernel(pyscf_molecule,
                                           pyscf_scf.mo_coeff)

    two_electron_integrals = ao2mo.restore(
        1, # no permutation symmetry
        two_electron_compressed, n_orbitals)
    # See PQRS convention in OpenFermion.hamiltonians._molecular_data
    # h[p,q,r,s] = (ps|qr)
    two_electron_integrals = numpy.asarray(
        two_electron_integrals.transpose(0, 2, 3, 1), order='C')

    # Return.
    return one_electron_integrals, two_electron_integrals


def run_pyscf(molecule,
              run_scf=True,
              run_mp2=False,
              run_cisd=False,
              run_ccsd=False,
              run_fci=False,
              verbose=False):
    """
    This function runs a pyscf calculation.

    Args:
        molecule: An instance of the MolecularData or PyscfMolecularData class.
        run_scf: Optional boolean to run SCF calculation.
        run_mp2: Optional boolean to run MP2 calculation.
        run_cisd: Optional boolean to run CISD calculation.
        run_ccsd: Optional boolean to run CCSD calculation.
        run_fci: Optional boolean to FCI calculation.
        verbose: Boolean whether to print calculation results to screen.

    Returns:
        molecule: The updated PyscfMolecularData object. Note the attributes
        of the input molecule are also updated in this function.
    """
    # Prepare pyscf molecule.
    pyscf_molecule = prepare_pyscf_molecule(molecule)
    molecule.n_orbitals = int(pyscf_molecule.nao_nr())
    molecule.n_qubits = 2 * molecule.n_orbitals
    molecule.nuclear_repulsion = float(pyscf_molecule.energy_nuc())

    # Run SCF.
    # pyscf_scf = compute_scf(pyscf_molecule)
    # pyscf_scf.verbose = 0
    # pyscf_scf.run()
    mf = pyscf.scf.RHF(pyscf_molecule).run()
    mycc = pyscf.ci.CISD(mf).run()
    mycc_cc = pyscf.cc.CCSD(mf).run()
    mycc_cc.e_tot + mycc_cc.ccsd_t()
    #print('HUI')

    pyscf_scf = pyscf.scf.HF(pyscf_molecule).run()

    molecule.hf_energy = float(pyscf_scf.e_tot)
    if verbose:
        print('Hartree-Fock energy for {} ({} electrons) is {}.'.format(
            molecule.name, molecule.n_electrons, molecule.hf_energy))

    # Hold pyscf data in molecule. They are required to compute density
    # matrices and other quantities.
    molecule._pyscf_data = pyscf_data = {}
    pyscf_data['mol'] = pyscf_molecule
    pyscf_data['scf'] = pyscf_scf

    # Populate fields.
    molecule._canonical_orbitals = pyscf_scf.mo_coeff.astype(float)
    molecule._orbital_energies = pyscf_scf.mo_energy.astype(float)

    # Get integrals.
    one_body_integrals, two_body_integrals = compute_integrals(
        pyscf_molecule, pyscf_scf)
    molecule._one_body_integrals = one_body_integrals
    molecule._two_body_integrals = two_body_integrals
    molecule._overlap_integrals = pyscf_scf.get_ovlp()

    # Run MP2.
    if run_mp2:
        if molecule.multiplicity != 1:
            print("WARNING: RO-MP2 is not available in PySCF.")
        else:
            pyscf_mp2 = mp.MP2(pyscf_scf)
            pyscf_mp2.verbose = 0
            pyscf_mp2.run()
            # molecule.mp2_energy = pyscf_mp2.e_tot  # pyscf-1.4.4 or higher
            molecule.mp2_energy = pyscf_scf.e_tot + pyscf_mp2.e_corr
            pyscf_data['mp2'] = pyscf_mp2
            if verbose:
                print('MP2 energy for {} ({} electrons) is {}.'.format(
                    molecule.name, molecule.n_electrons, molecule.mp2_energy))

    # Run CISD.
    if run_cisd:
        pyscf_cisd = ci.CISD(pyscf_scf)
        pyscf_cisd.verbose = 0
        pyscf_cisd.run()
        molecule.cisd_energy = pyscf_cisd.e_tot
        pyscf_data['cisd'] = pyscf_cisd
        if verbose:
            print('CISD energy for {} ({} electrons) is {}.'.format(
                molecule.name, molecule.n_electrons, molecule.cisd_energy))

    # Run CCSD.
    if run_ccsd:
        pyscf_ccsd = cc.CCSD(pyscf_scf).run()

        molecule.ccsd_energy = pyscf_ccsd.e_tot
        ccsd_t_corr = pyscf_ccsd.ccsd_t()
        molecule.ccsd_t_energy = molecule.ccsd_energy + ccsd_t_corr
        pyscf_data['ccsd'] = pyscf_ccsd
        if verbose:
            print('CCSD energy for {} ({} electrons) is {}.'.format(
                molecule.name, molecule.n_electrons, molecule.ccsd_energy))

    # Run FCI.
    if run_fci:
        #molecule.fci_energy = -108.606226 - 0.479983 #pyscf_fci.kernel()[0]
        #pyscf_data['fci'] = None
        
        pyscf_fci = fci.FCI(pyscf_molecule, pyscf_scf.mo_coeff)
        pyscf_fci.verbose = 0
        molecule.fci_energy, fci_vec = pyscf_fci.kernel()
        molecule.ipr = (fci_vec ** 4).sum()
        pyscf_data['fci'] = pyscf_fci
        if verbose:
            print('FCI energy for {} ({} electrons) is {}.'.format(
                molecule.name, molecule.n_electrons, molecule.fci_energy))

    # Return updated molecule instance.
    pyscf_molecular_data = PyscfMolecularData.__new__(PyscfMolecularData)
    pyscf_molecular_data.__dict__.update(molecule.__dict__)
    pyscf_molecular_data.save()
    return pyscf_molecular_data


def generate_molecular_hamiltonian(
        geometry,
        basis,
        multiplicity,
        charge=0,
        n_active_electrons=None,
        n_active_orbitals=None):
    """Generate a molecular Hamiltonian with the given properties.

    Args:
        geometry: A list of tuples giving the coordinates of each atom.
            An example is [('H', (0, 0, 0)), ('H', (0, 0, 0.7414))].
            Distances in angstrom. Use atomic symbols to
            specify atoms.
        basis: A string giving the basis set. An example is 'cc-pvtz'.
            Only optional if loading from file.
        multiplicity: An integer giving the spin multiplicity.
        charge: An integer giving the charge.
        n_active_electrons: An optional integer specifying the number of
            electrons desired in the active space.
        n_active_orbitals: An optional integer specifying the number of
            spatial orbitals desired in the active space.

    Returns:
        The Hamiltonian as an InteractionOperator.
    """

    # Run electronic structure calculations
    molecule = run_pyscf(
            MolecularData(geometry, basis, multiplicity, charge)
    )

    # Freeze core orbitals and truncate to active space
    if n_active_electrons is None:
        n_core_orbitals = 0
        occupied_indices = None
    else:
        n_core_orbitals = (molecule.n_electrons - n_active_electrons) // 2
        occupied_indices = list(range(n_core_orbitals))

    if n_active_orbitals is None:
        active_indices = None
    else:
        active_indices = list(range(n_core_orbitals,
                                    n_core_orbitals + n_active_orbitals))

    return molecule.get_molecular_hamiltonian(
            occupied_indices=occupied_indices,
            active_indices=active_indices)
