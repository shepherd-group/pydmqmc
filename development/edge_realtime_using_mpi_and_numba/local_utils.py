#!/usr/bin/env python

import numpy as np
import pandas as pd

from mpi import ParallelHelper
from os import path, remove
from numpy.typing import NDArray as Array


def getmem(a: Array) -> float:
    r''' A rough estimate for GB of memory an array is using.
    '''
    return a.nbytes*(1e-9)


def read_comm_file(comm_file: str, ph: ParallelHelper) -> dict:
    r''' Read the comm file and carry out instructions.
    '''
    comm_cmds = {}

    if not path.isfile(comm_file):
        return comm_cmds
    else:
        ph.print(f'# Encountered communication file: {comm_file}')
        ph.print('# Reading contents and updating calculation accordingly.')

    with open(comm_file, 'rt') as stream:
        for line in stream:
            cmd = line.replace(' ', '').strip().split('=')

            if len(cmd) == 2:
                ph.print(f'# Accepting command: "{line.strip()}"')
                comm_cmds[cmd[0].lower()] = cmd[1].lower()

    # Make sure every processor read the file bfore removing.
    ph.barrier()

    ph.print(
        f'# Finished reading {comm_file}, removing file to prevent collisions!'
    )
    if ph.parent:
        remove(comm_file)

    for key in comm_cmds:
        val = comm_cmds[key]

        if val == 'true':
            comm_cmds[key] = True
        elif val == 'false':
            comm_cmds[key] = False
    #if 'softexit' in comm_cmds:
    #    if comm_cmds['softexit'] == 'true':
    #        print('# Executing soft exit, have a good day!')
    #        ph.abort(0)

    return comm_cmds


def generate_binary_arrays(M: int, ndets: int, dets: Array) -> Array:
    ''' TODO: Write a docstring.
    '''
    bas = np.zeros((ndets, M), dtype=int)

    for idet in range(ndets):
        bas[idet][dets[idet] - 1] = 1

    return bas


def read_fci_output(fname: str) -> dict:
    ''' TODO: Write a docstring.
    '''
    energies_header = 'State     Energy'
    basis_header = 'index  spatial symmetry sym_index lz     ms       <i|f|i>'
    electron_line = 'Number of electrons:'
    basis_line = 'Number of basis functions:'
    isym_line = '"symmetry":'

    store_basis = False
    basis_stored = False
    store_energies = False
    info = {
        'N': None,
        'M': None,
        'basis': {
            'indx': [],
            'isym': [],
            'isym_indx': [],
            'ms': [],
            'fi': [],
        },
        'fci': {},
    }

    with open(fname, 'rt') as stream:
        for line in stream:
            if '--' in line:
                continue

            # Basis storage
            if basis_header in line and not basis_stored:
                store_basis = True
            elif store_basis:
                ld = line.split()

                if len(ld) == 0:
                    store_basis = False
                    basis_stored = True
                elif len(ld) == 7:
                    info['basis']['indx'].append(int(ld[0]))
                    info['basis']['isym'].append(int(ld[2]) + 1)
                    info['basis']['isym_indx'].append((int(ld[3]) - 1)//2)
                    info['basis']['ms'].append(int(ld[5]))
                    info['basis']['fi'].append(float(ld[6]))
                else:
                    raise RuntimeError('Read in failed for basis set!')

            # N/M storage
            if electron_line in line:
                N = int(line.split()[-1])
                if info['N'] is None:
                    info['N'] = N
                else:
                    assert info['N'] == N
            elif basis_line in line:
                M = int(line.split()[-1])
                if info['M'] is None:
                    info['M'] = M
                else:
                    assert info['M'] == M

            # Get the isym key for storing fci energies.
            if isym_line in line:
                isym = int(line.split()[-1][:-1]) + 1
                info['fci'][isym] = []

            # Store the energies.
            if energies_header in line:
                store_energies = True
            elif store_energies:
                ld = line.split()

                if len(ld) == 0:
                    store_energies = False
                elif len(ld) == 2:
                    info['fci'][isym].append(float(ld[1]))

    return info


def read_hamiltonian(
        fname: str,
        diag: bool = False,
        verbose: bool = False,
    ) -> np.array:
    ''' TODO: Write a docstring.
    '''
    if path.isfile(f'{fname}.npy'):
        if verbose:
            print(f'WARNING: reading {fname}.npy cache.')
        hamil = np.load(f'{fname}.npy')
    else:
        ndets = 0
        raw = {}

        with open(fname, 'rt') as stream:
            for line in stream:
                i, j, hij = line.split()

                i = int(i)
                j = int(j)
                hij = float(hij)

                ndets = max(i, j, ndets)

                raw[i,j] = hij

        hamil = np.zeros((ndets, ndets), dtype=float)

        for (i, j), hij in raw.items():
            hamil[i-1,j-1] = hij
            hamil[j-1,i-1] = hij

        del raw

    if diag:
        hamil = np.einsum('ii->i', hamil)

    return hamil


def read_wavefunctions(fname: str, verbose: bool = False) -> Array:
    ''' TODO: Write a docstring.
    '''
    if path.isfile(f'{fname}.npy'):
        if verbose:
            print(f'WARNING: reading {fname}.npy cache.')
        wfns = np.load(f'{fname}.npy')
    else:
        ndets = 0
        raw = []

        with open(fname, 'rt') as stream:
            for line in stream:
                i, fi, Ci = line.split()

                i = int(i)
                Ci = float(Ci)

                ndets = max(i, ndets)

                raw.append(Ci)

        wfns = np.array(raw).reshape((ndets, ndets))

        del raw

    return wfns


def read_determinants(fname: str) -> Array:
    ''' TODO: Write a docstring.
    '''
    dets = []

    with open(fname, 'rt') as stream:
        for line in stream:
            line = line.split('|')[1].replace('>', '')

            det = np.array(line.split()).astype(int)

            dets.append(det)

    dets = np.array(dets)

    return dets


def read_truncated_wavefunctions(
        fname: str,
        nmax: int,
        ndets: int,
        verbose: bool = False,
    ) -> Array:
    ''' TODO: Write a docstring.
    '''
    if path.isfile(f'{fname}.npy'):
        if verbose:
            print(f'WARNING: reading {fname}.npy cache.')
        wfns = np.load(f'{fname}.npy')[:nmax,:]
    else:
        ndet = 0
        wfns = np.zeros((nmax, ndets), dtype=float)

        with open(fname, 'rt') as stream:
            for line in stream:
                i, fi, Ci = line.split()

                i = int(i)
                Ci = float(Ci)

                wfns[ndet, i - 1] = Ci

                if i == ndets:
                    ndet += 1

                if ndet >= nmax:
                    break

    return wfns


def read_psi4_basis(output: str) -> pd.DataFrame:
    r''' Given a Psi4 output file read and store the orbital basis contained
    within.

    Parameters
    ----------
    output : str
        The Psi4 output file to read the basis from.

    Returns
    -------
    basis : pd.DataFrame
        The read in orbital basis, where each column corresponds with
        a spatial orbital.
    '''

    table_start = 'Orbital Energies [Eh]'

    table_ignore = [
        '---------------------',
        'Doubly Occupied:',
        'Virtual:',
    ]

    table_end = 'Final Occupation by Irrep:'

    store = False
    occupied = 1
    basis = {
        'indx': [],
        'label': [],
        'energy': [],
        'occupied': [],
    }

    with open(output, 'rt') as stream:
        for line in stream:
            ld = line.split()

            if len(ld) == 0:
                continue

            if table_start in line:
                store = True
            elif table_end in line:
                store = False
            elif any(ti in line for ti in table_ignore):
                if 'Virtual:' in line:
                    occupied = 0
                continue
            elif store and len(ld) % 2 == 0:
                for ipair in range(len(ld)//2):
                    l = ld[2*ipair + 0]
                    e = ld[2*ipair + 1]
                    basis['indx'].append(len(basis['indx']))
                    basis['label'].append(l)
                    basis['energy'].append(float(e))
                    basis['occupied'].append(occupied)

    return pd.DataFrame(basis)


def generate_psi4_orbital_map(
        output: str,
        label_to_int: dict,
        verbose: bool = False,
    ) -> tuple[dict,]:
    r''' Given a Psi4 output file, work out the map between the orbital
    labels in the basis table and the one-particle integrals (spatial) index.

    Parameters
    ----------
    output : str
        The Psi4 output to read the information from.
    label_to_int : dict
        A dictionary with label keys and integer values for the
        irreducible representations.
    verbose : bool, default = False
        Whether we should report the orbital symmetries to stdout.

    Returns
    -------
    orbsyms : dict
        Contains keys for the integer irreducible representation which
        returns values for the orbital indices for that irrep.
    orbmap : dict
        Contains keys like {X}{Y}, where X is a number for the orbital
        in irreducible lable Y, and values of the spatial index in the
        one-electron integrals.
    '''
    basis_count_header = 'Irrep   Nso     Nmo'
    basis_count_terminus = 'Total'
    basis_count_ignore = ['-------------------------']

    store = False
    M = {}

    with open(output, 'rt') as stream:
        for line in stream:
            if basis_count_header in line:
                store = True
            elif store and basis_count_terminus in line:
                store = False
            elif store and any(bci in line for bci in basis_count_ignore):
                continue
            elif store:
                label, cntso, cntmo = line.split()
                assert cntso == cntmo
                M[label] = int(cntso)

    edges = np.cumsum(
        [0]
        + [v for k, v in M.items()]
    )

    orbsyms = {
        label_to_int[label]: np.arange(edges[i], edges[i + 1])
        for i, label in enumerate(M)
    }

    int_to_label = {
        v: k
        for k, v in label_to_int.items()
    }

    if verbose:
        print(' === read in orbital symmetries === ')
        for irrep, orbs in orbsyms.items():
            if len(orbs) != 0:
                print(f'irrep: {irrep} | {int_to_label[irrep]}')
                print(f'    min(orbs): {min(orbs)}')
                print(f'    max(orbs): {max(orbs)}')
                print(f'    len(orbs): {len(orbs)}')
                print()
            else:
                print(f'irrep: {irrep} | {int_to_label[irrep]}')
                print(f'    min(orbs): {np.nan}')
                print(f'    max(orbs): {np.nan}')
                print(f'    len(orbs): {len(orbs)}')
                print()

    # Maps, Psi4 label: index
    orbmap = {}
    for irrep, orbs in orbsyms.items():
        label = int_to_label[irrep]

        for i, orb in enumerate(orbs):
            orbmap[f'{i+1}{label}'] = len(orbmap)

    return orbsyms, orbmap


def matrix_print(
        label: str,
        mat: Array,
        wpad: int = 8,
        dpad: int = None,
    ) -> None:
    ''' TODO: Write a docstring.
    '''
    if dpad is None:
        dpad = wpad//2

    if len(mat.shape) > 1:
        #print()
        print()
        print(label)
        for i, row in enumerate(mat):
            hstr = '      '
            rstr = f' {i+1:<5}'
            for j, v in enumerate(row):
                hstr += f' {j+1:>{wpad}}'
                if isinstance(v, np.float64):
                    rstr += f' {v:> {wpad}.{dpad}f}'
                elif isinstance(v, np.int64):
                    rstr += f' {v:> {wpad}d}'
                else:
                    rstr += f' {str(v):>{wpad}}'
            if i == 0:
                print(hstr)
            print(rstr)
    else:
        tmp = mat.copy().reshape((mat.shape[0], 1))
        matrix_print(label, tmp, wpad, dpad)

    return


def D2h_product(a: str, b: str) -> str:
    r''' Find the cross product between irrep a and irrep b as the operation
        a (X) b
    where (x) represents the cross product.

    Parameters
    ----------
    a : str
        The left hand side for the operator.
    b : str
        The right hand side for the operator.

    Returns
    -------
    x : str
        The resulting irrep from a (x) b.
    '''
    nirreps = 8

    table = {
        ('Ag', 'Ag'): 'Ag',
        ('Ag', 'B1g'): 'B1g',
        ('Ag', 'B2g'): 'B2g',
        ('Ag', 'B3g'): 'B3g',
        ('Ag', 'Au'): 'Au',
        ('Ag', 'B1u'): 'B1u',
        ('Ag', 'B2u'): 'B2u',
        ('Ag', 'B3u'): 'B3u',
        ('B1g', 'Ag'): 'B1g',
        ('B1g', 'B1g'): 'Ag',
        ('B1g', 'B2g'): 'B3g',
        ('B1g', 'B3g'): 'B2g',
        ('B1g', 'Au'): 'B1u',
        ('B1g', 'B1u'): 'Au',
        ('B1g', 'B2u'): 'B3u',
        ('B1g', 'B3u'): 'B2u',
        ('B2g', 'Ag'): 'B2g',
        ('B2g', 'B1g'): 'B3g',
        ('B2g', 'B2g'): 'Ag',
        ('B2g', 'B3g'): 'B1g',
        ('B2g', 'Au'): 'B2u',
        ('B2g', 'B1u'): 'B3u',
        ('B2g', 'B2u'): 'Au',
        ('B2g', 'B3u'): 'B1u',
        ('B3g', 'Ag'): 'B3g',
        ('B3g', 'B1g'): 'B2g',
        ('B3g', 'B2g'): 'B1g',
        ('B3g', 'B3g'): 'Ag',
        ('B3g', 'Au'): 'B3u',
        ('B3g', 'B1u'): 'B2u',
        ('B3g', 'B2u'): 'B1u',
        ('B3g', 'B3u'): 'Au',
        ('Au', 'Ag'): 'Au',
        ('Au', 'B1g'): 'B1u',
        ('Au', 'B2g'): 'B2u',
        ('Au', 'B3g'): 'B3u',
        ('Au', 'Au'): 'Ag',
        ('Au', 'B1u'): 'B1g',
        ('Au', 'B2u'): 'B2g',
        ('Au', 'B3u'): 'B3g',
        ('B1u', 'Ag'): 'B1u',
        ('B1u', 'B1g'): 'Au',
        ('B1u', 'B2g'): 'B3u',
        ('B1u', 'B3g'): 'B2u',
        ('B1u', 'Au'): 'B1g',
        ('B1u', 'B1u'): 'Ag',
        ('B1u', 'B2u'): 'B3g',
        ('B1u', 'B3u'): 'B2g',
        ('B2u', 'Ag'): 'B2u',
        ('B2u', 'B1g'): 'B3u',
        ('B2u', 'B2g'): 'Au',
        ('B2u', 'B3g'): 'B1u',
        ('B2u', 'Au'): 'B2g',
        ('B2u', 'B1u'): 'B3g',
        ('B2u', 'B2u'): 'Ag',
        ('B2u', 'B3u'): 'B1g',
        ('B3u', 'Ag'): 'B3u',
        ('B3u', 'B1g'): 'B2u',
        ('B3u', 'B2g'): 'B1u',
        ('B3u', 'B3g'): 'Au',
        ('B3u', 'Au'): 'B3g',
        ('B3u', 'B1u'): 'B2g',
        ('B3u', 'B2u'): 'B1g',
        ('B3u', 'B3u'): 'Ag',
    }

    if False:
        assert len(table) == nirreps**2

        cnts = {}

        for (t1, t2), v in table.items():
            if v not in cnts:
                cnts[v] = 0
            cnts[v] += 1

        assert len(cnts) == nirreps
        assert all(v == nirreps for _, v in cnts.items())

    return table[a, b]


def D2h_dipole_irrep_to_array(
        M: int,
        mu_xyz: dict,
        orbsyms: dict,
    ) -> dict:
    r''' Given the dipole integrals for each coordinate (x, y, z) as a
    function of irreducible representation, work out the corresponding
    location for the integrals in a matrix indexed by the spatial orbitals.

    For a given irreducible representation one can find the corresponding
    irreducible representation that corresponds to a valid excitation.
    Then, given these two labels we can find the corresponding orbitals
    that represent these two and use those to determine where to insert
    the integrals into the matrix.

    This is because for a given irreducible representation i, the following
    must be true for their to be a valid excitation to the irreducible
    representation for j:
        i_irrep (x) cart_irrep (x) j_irrep = Ag
    Where we are implicity assuming a D2h symmetry. This expression can
    be rearranged to give
        j_irrep = Ag (x) i_irrep (x) cart_irrep
    and since the product of any label with Ag is always the given label
    we have
        j_irrep = i_irrep (x) cart_irrep

    Then all we need to do is loop through the coordinates and then given
    integrals with their corresponding irreducible representation and solve
    for the corresponding j representation. Once we have this we map the
    irreducible representations to the orbitals and insert the integrals
    in the spatial indices corresponding to the rows as the ith label orbitals
    and the columns as the jth label orbitals.

    Parameters
    ----------
    M : int
        The number of spatial orbitals in the system.
    mu_xyz : dict
        A dictionary with coordinates (x, y, z) as keys and dictionary
        with the dipole integrals as the values. The values dictionary has
        irreducible representations as keys and the dipole integrals as the
        values.
    orbsyms : dict
        A dictionary with the systems irreducible representation index as
        the key and the corresponding spatial orbital indices for the values.

    Returns
    -------
    muij_xyz : dict
        A dictionary with the coordinates (x, y, z) as the keys and the
        dipole integral matrices (with orbital indices) as the values.
    '''
    cart_labels = {
        'x': 'B3u',
        'y': 'B2u',
        'z': 'B1u',
    }

    index_to_label = {
        1: 'Ag',
        2: 'B3u',
        3: 'B2u',
        4: 'B1g',
        5: 'B1u',
        6: 'B2g',
        7: 'B3g',
        8: 'Au',
    }

    label_to_index = {v: k for k, v in index_to_label.items()}

    muij_xyz = {}

    for cart, imats in mu_xyz.items():
        cart_label = cart_labels[cart]
        muij = np.zeros((M, M), dtype=float)

        for iirrep, mat in imats.items():
            ilabel = index_to_label[iirrep]

            # Work out the only valid irrep given our coordinate (x, y, z)
            # and the current ith irrep.
            jlabel = D2h_product(ilabel, cart_label)
            jirrep = label_to_index[jlabel]

            iorbs = orbsyms[iirrep]
            jorbs = orbsyms[jirrep]

            # Get the location on the matrix indexed by the orbitals.
            min_iorb = min(iorbs)
            max_iorb = max(iorbs) + 1
            min_jorb = min(jorbs)
            max_jorb = max(jorbs) + 1

            # Insert the integrals to the relevant orbitals.
            muij[min_iorb:max_iorb, min_jorb:max_jorb] = mat

        muij_xyz[cart] = muij

    return muij_xyz
