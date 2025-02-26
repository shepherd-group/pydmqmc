#!/usr/bin/env python

import numpy as np
import pandas as pd

from time import time
from scipy.linalg import block_diag

from local_utils import (
    getmem,
    read_psi4_basis,
    read_fci_output,
    read_hamiltonian,
    read_wavefunctions,
    read_determinants,
    read_truncated_wavefunctions,
    generate_binary_arrays,
    generate_psi4_orbital_map,
    D2h_dipole_irrep_to_array,
)


class D2hSystem:
    ''' TODO: Write a docstring.
    '''
    irrep_label_to_int = {
        'Ag': 1,
        'B3u': 2,
        'B2u': 3,
        'B1g': 4,
        'B1u': 5,
        'B2g': 6,
        'B3g': 7,
        'Au': 8,
    }

    irrep_int_to_label = {
        v: k
        for k, v in irrep_label_to_int.items()
    }

    def __init__(
            self,
            dump: str,
            output: str,
            psi4_output: str,
            psi4_rhf_output: str,
            verbose: bool,
            cas_orbitals: list[str],
            hande_fci_files: dict,
            dipole: dict,
            overlap: dict,
            coefficients: dict,
        ) -> None:
        ''' TODO: Write a docstring.
        '''
        ti = time()

        self.dump = dump
        self.output = output
        self.psi4_output = psi4_output
        self.psi4_rhf_output = psi4_rhf_output
        self.verbose = verbose
        self.cas_orbitals = cas_orbitals
        self.hande_fci_files = hande_fci_files
        self.dipole = dipole
        self.overlap = overlap
        self.coefficients = coefficients

        self.N: int
        self.M: int
        self.basis: pd.DataFrame

        self.fci: dict = {}
        self.ham: dict = {}
        self.wfn: dict = {}
        self.det: dict = {}
        self.bas: dict = {}
        self.ndets: dict = {}

        self.orbsyms: dict = {}
        self.orbmap: dict = {}
        self.somu: dict = {}
        self.sos: dict = {}
        self.soc: dict = {}
        self.cmat: Array
        self.cmatt: Array

        if len(self.cas_orbitals) > 0:
            self.print(' === reading in CAS basis ===')
            self.read_cas_basis()

        self.print(' === generating orbital map ===')
        self.orbsyms, self.orbmap = generate_psi4_orbital_map(
            self.psi4_rhf_output,
            self.irrep_label_to_int,
            verbose=self.verbose,
        )

        self.print(' === reading in FCI data ===')
        self.readin(output, hande_fci_files)
        self.print(' === reading in dipole integrals ===')
        self.read_dipole(dipole)
        self.print(' === reading in overlap matrices ===')
        self.read_overlap(overlap)
        self.print(' === reading in coefficient matrices ===')
        self.read_coefficients(coefficients)

        tf = time()
        self.tinit = tf - ti
        self.print(
            f' === finished, total time to initialize: {tf-ti:>.8f} (sec.) ==='
        )

        return

    def print(self, text: str) -> None:
        ''' TODO: Write a docstring.
        '''
        if self.verbose:
            print(text)
        return

    def read_cas_basis(self) -> None:
        ''' TODO: Write a docstring.
        '''
        tmp = read_psi4_basis(self.psi4_rhf_output)

        self.cas_basis = tmp.copy().loc[np.isin(
            tmp.label.values,
            self.cas_orbitals,
        )]

        del tmp

        self.cas_basis.loc[:, 'inew'] = np.arange(self.cas_basis.shape[0])

        return

    def readin(self, output: str, hande_fci_files: dict) -> None:
        ''' TODO: Write a docstring.
        '''
        # Read in and store the system information and FCI energies.
        self.print(f'    reading: {output}')
        info = read_fci_output(output)

        self.N = info['N']
        self.M = info['M']
        self.basis = pd.DataFrame(info['basis'])

        # Read in and store the Hamiltonians, wavefunctions, and determinants.
        for isym, files in hande_fci_files.items():
            self.fci[isym] = np.array(info['fci'][isym])
            self.ndets[isym] = self.fci[isym].shape[0]

            if 'det' in files:
                self.print(f'    reading: {files["det"]}')
                self.det[isym] = read_determinants(files['det'])
                self.bas[isym] = generate_binary_arrays(
                    self.M,
                    self.ndets[isym],
                    self.det[isym],
                )
                assert self.fci[isym].shape[0] == self.det[isym].shape[0]

            if 'ham' in files:
                if 'diag' in files:
                    diag = files['diag']
                else:
                    diag = False
                self.print(f'    reading: {files["ham"]}')
                self.ham[isym] = read_hamiltonian(
                    files['ham'],
                    diag=diag,
                    verbose=self.verbose,
                )
                assert self.fci[isym].shape[0] == self.ham[isym].shape[0]

            if 'wfn' in files and 'nwfn' in files:
                self.print(
                    f'    reading: {files["wfn"]} (nmax = {files["nwfn"]})'
                )
                self.wfn[isym] = read_truncated_wavefunctions(
                    files['wfn'],
                    nmax=files['nwfn'],
                    ndets=self.ndets[isym],
                    verbose=self.verbose,
                )
                assert self.fci[isym].shape[0] == self.wfn[isym].shape[1]
                assert self.wfn[isym].shape[0] == files['nwfn']
            elif 'wfn' in files:
                self.print(f'    reading: {files["wfn"]}')
                self.wfn[isym] = read_wavefunctions(
                    files['wfn'],
                    verbose=self.verbose,
                )
                assert self.fci[isym].shape[0] == self.wfn[isym].shape[0]
                assert self.wfn[isym].shape[0] == self.wfn[isym].shape[1]

        return

    def read_dipole(self, dipole: dict) -> None:
        ''' TODO: Write a docstring.
        '''
        for coord, files in dipole.items():
            self.somu[coord] = {}

            for irrep, f in files.items():
                m = np.load(f)

                if m.shape[0] == 0 or m.shape[1] == 0:
                    continue

                self.somu[coord][irrep] = m

        return

    def read_overlap(self, overlap: dict) -> None:
        ''' TODO: Write a docstring.
        '''
        for irrep, f in overlap.items():
            m = np.load(f)

            if m.shape[0] == 0 or m.shape[1] == 0:
                continue

            self.sos[irrep] = m

        return

    def read_coefficients(self, coefficients: dict) -> None:
        ''' TODO: Write a docstring.
        '''
        if len(coefficients) == 0:
            return

        for irrep, f in coefficients.items():
            m = np.load(f)

            if m.shape[0] == 0 or m.shape[1] == 0:
                continue

            self.soc[irrep] = m

        # Get the M, M coefficient matrix (and transpose)
        # This is just a block diagonal matrix where the irreps
        # are the various blocks, e.g.
        #   C = [  Ag   0   0 ... ]
        #       [   0 B1g   0 ... ]
        #       [   0   0   . ... ]
        #       [   .   .   . ... ]
        #       [   .   .   . ... ]
        self.cmat = block_diag(*[v for _, v in self.soc.items()])
        self.cmatt = np.transpose(self.cmat)

        return

    def __repr__(self) -> str:
        ''' TODO: Write a docstring.
        '''
        report = ' === system report ===\n'
        report += f' tinit: {self.tinit/60.0:.4f} (min.)\n'
        report += f' output: {self.output}\n'
        report += f' Psi4 output: {self.psi4_output}\n'
        report += f' Psi4 RHF output: {self.psi4_rhf_output}\n'

        for isym, files in self.hande_fci_files.items():
            report += f' symmetry {isym} files:\n'
            for k, f in files.items():
                report += f'     {k}: {f}\n'

        report += f' electrons: {self.N}\n'
        report += f' basis size: {self.M}\n'
        report += ' spin basis table:\n'
        report += self.basis.to_string()

        report += '\n'
        report += ' CAS spatial basis table:\n'
        report += self.cas_basis.to_string()

        report += '\n'
        report += ' symmetry information:\n'
        report += ' === assumed symmetry labels ===\n'
        for l, i in self.irrep_label_to_int.items():
            report += f'     {l:>4}: {i:>3}\n'
        report += ' === symmetry FCI data ===\n'
        for isym, ndets in self.ndets.items():
            report += f'     isym: {isym}\n'
            report += f'         ndets: {ndets}\n'
            report += f'          Emin: {self.fci[isym].min():> 22.12f}\n'
            report += f'{" "*12}memory usage (per processor)\n'
            if isym in self.bas:
                report += f'{" "*11}bas: {getmem(self.bas[isym]):.3f} (GB)\n'
            if isym in self.det:
                report += f'{" "*11}det: {getmem(self.det[isym]):.3f} (GB)\n'
            if isym in self.ham:
                report += f'{" "*11}ham: {getmem(self.ham[isym]):.3f} (GB)\n'
            if isym in self.wfn:
                report += f'{" "*11}wfn: {getmem(self.wfn[isym]):.3f} (GB)\n'

        report += '\n\n'

        return report


def get_dipole_matrices(sys: D2hSystem) -> tuple[np.array, ...]:
    r''' TODO: Docstring.
    '''
    # Get dipole integrals as a dict with keys x, y, and z which have values
    # of M, M arrays for the dipole values. These are still in a symmetrized
    # atomic orbital form.
    so_muij_xyz = D2h_dipole_irrep_to_array(
        len(sys.orbmap),
        sys.somu,
        sys.orbsyms,
    )

    # Convert the atomic orbital form to the molecular orbital form.
    mo_muij_xyz = {
        c: (sys.cmatt @ m) @ sys.cmat
        for c, m in so_muij_xyz.items()
    }

    Mcas = sys.cas_basis.shape[0]

    muij_xyz = {
        'x': np.zeros((Mcas, Mcas), dtype=float),
        'y': np.zeros((Mcas, Mcas), dtype=float),
        'z': np.zeros((Mcas, Mcas), dtype=float),
    }

    # Use the CAS basis table to downsample the M, M dipole matrices
    # to Mcas, Mcas forms.
    for irow in sys.cas_basis.itertuples():
        i = sys.orbmap[irow.label]
        icas = irow.inew

        for jrow in sys.cas_basis.itertuples():
            j = sys.orbmap[jrow.label]
            jcas = jrow.inew

            for c in muij_xyz:
                muij_xyz[c][icas, jcas] = mo_muij_xyz[c][i, j]

    muxmat = muij_xyz['x']
    muymat = muij_xyz['y']
    muzmat = muij_xyz['z']

    return muxmat, muymat, muzmat
