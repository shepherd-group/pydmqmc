#!/usr/bin/env python

import numpy as np
import pandas as pd
from numpy import linalg as LA


def FCI(hamil):

    '''
    Do FCI on a Hamiltonian.

        In:
            hamil:
                The Hamiltonian we want to diagonalize.
                Assumed to be hermetian.
        Out:
            eigs:
                The eigenspectrum from diagonalization.
            vec:
                The corresponding eigenvectors.
    '''

    eigs, vec = LA.eigh(hamil)

    return eigs, vec

def sum_of_states(eigenspectrum, beta):

    '''
    Calculate the energy at a beta using the sum of states method
    E = \sum(\epsilon_i*e^{-\epsilon_i * beta}) / \sum(e^{-\epsilon_i * beta})
    Use numpy longdouble other wise large beta returns infinite because
    of an floating point overflow.

        In:
            eigenspectrum:
                fci eigenvalues for a system
            beta:
                The inverse temperature where we calculate the energy
        Out:
            energy:
                The energy at beta for the system's eigenspectrum we used
    '''

    numerator   = np.exp(-beta * eigenspectrum)
    denominator = np.sum(np.copy(numerator))
    numerator   *= eigenspectrum
    numerator   = np.sum(numerator)

    energy      = np.divide(numerator, denominator)

    return energy


def build_hamiltonian(hamilf):

    '''
        In:
            hamilf:
                The file containing what is assumed to be a triangle of
                our Hamiltonian we are interested in.
        Out:
            hamiltonian:
                A 2D NumPy array representation of our matrix
            hilbert_space:
                The length of our determinant space
            HF:
                The reference energy state, this is assumed to be the zeroth
                element of our matrix.
    '''

    H_data = []

    with open(hamilf, 'r') as f:
        for line in f:
            i, j, hij = line.split()
            i, j, hij = int(i), int(j), float(hij)
            H_data.append((i,j,hij))
    
    hilbert_space = i
    hamiltonian = np.zeros((hilbert_space,hilbert_space))

    for (i,j,hij) in H_data:
        hamiltonian[i-1,j-1] = hij
        hamiltonian[j-1,i-1] = hij
        
    return hamiltonian, hilbert_space, hamiltonian[0,0]


def tmatrix(hamiltonian, shift):

    '''
        In:
            hamiltonian:
                The system we are interested in.
            shift:
                A shift to apply to the diagonal of the hamiltonian
        Out:
            Tpos:
                The positive elements of the T matrix.
            Tneg:
                The negative elements of the T matrix.
                note that the returned matrix is positive.

        T = -(H - S)

        The shift often includes the HF and an additional componant we
        want to add in.
    '''

    hartree_fock = hamiltonian[0,0]
    hilbert_space = len(hamiltonian)
    eyemat = np.eye(hilbert_space)

    T = -(hamiltonian - (hartree_fock*eyemat) - (shift*eyemat))

    Tdiag = np.diag(np.diag(T))
    V = T - Tdiag
    Tpos = V.clip(min=0)
    Tneg = abs(V.clip(max=0))

    return Tpos, Tneg


def unphysical_hamiltonian(hamiltonian):

    '''
    Generates an unphysical Hamiltonian from the physical version.

        In:
            hamiltonian:
                The system Hamiltonian we are interested in investigating.
        Out:
            Htilde:
                The unphysical version of that hamiltonian which should
                not have a sign problem.
    '''

    Htilde = hamiltonian
    Htilde_diags = np.diag(np.diag(Htilde))
    Htilde = Htilde - Htilde_diags
    Htilde_pos = -Htilde.clip(min=0)
    Htilde_neg = Htilde.clip(max=0)
    Htilde = Htilde_pos + Htilde_neg + Htilde_diags
    
    return Htilde


def proj_energy(hamil_matrix, rho_matrix, row):

    '''
        In:
            hamil_matrix:
                THe Hamiltonian to use for the estimation
            rho_matrix:
                The rho matrix to use for energy estimate.
            row:
                The row we are interested in.
        Out:
            Energy: The energy given as
          sum_i { hamil_matrix[i,row] rho_matrix[row,i] } / rho_matrix[row,row]
    '''

    energy = (hamil_matrix[:,row] @ rho_matrix[row,:]).sum()
    energy = energy / rho_matrix[row,row]

    return energy


def expectation(hamil, dm, observable):

    '''
        In:
            hamil
                THe Hamiltonian to use for the estimation
            dm:
                The rho matrix to use for energy estimate.
            observable:
                An observables to calculate and return
        Out:
            expectation: The expectation for an operator "O":
                <O> = Tr(O dm) / Tr(dm)
    '''

    if 'Energy' == observable:
        expectation_num = (hamil @ dm).trace()
        expectation_den = dm.trace()
        expectation = expectation_num/expectation_den

    else:
        print('Unexpected observable:', observable)
        print('Please implement or check spelling!')
        print('Exiting...')
        exit()

    return expectation


def sort_index_by_diagonal(hamil, hilbert):

    '''
        In:
            hamil:
                The hamiltonian we want to sort based on the
                ascending order of the diagonal elements.
            hilbert:
                The hilbert space of the wavefunction, makes
                forming arrays nice and quick.
        Out:
            sorted_hamil:
                The hamiltonian now rearanged to be ascending on the
                diagonal elements.
            sorted_diags:
                A list of the sorted diagonal elements.
            index_map:
                The index map to sort matrix's if we want to later.
    '''

    diags = np.diag(hamil)
    sorted_index = np.argsort(diags)
    sorted_diags = diags[sorted_index]
    index_map = {ii:i for i,ii in enumerate(sorted_index)}

    sorted_hamil = np.zeros((hilbert,hilbert))
    for i in range(hilbert):
        for j in range(hilbert):
            ii = index_map[i]
            jj = index_map[j]
            sorted_hamil[ii,jj] = hamil[i,j]

    return sorted_hamil, sorted_diags, index_map


def system_initialize(hamilf, shift=0):

    '''
    Set up a system and return relevent matrix's for running analytical
    QMC.

        In:
            hamilf:
                The system Hamiltonian we are interested in.
            shift (default=0):
                A shift to apply to the diagonal elements of the Hamiltonian
        Out:
            H:
                The Hamiltonian shifted by the Hartree-Fock and the
                provided shift.
    '''

    H, HS, HF = build_hamiltonian(hamilf)
    H, sorted_diags, sorted_hash = sort_index_by_diagonal(H, HS)
    Heval = np.copy(H)
    H = H - (np.eye(HS)*HF) - (np.eye(HS)*shift)

    return H, Heval, HS


def initialize_ip(init, Nattempts, target, Heval, HS):

    '''
    Initalizes the starting trial density matrix. There are many ways
    to do this so we have a str check to decide.

        In:
            init:
                The initalization method we would like to use.
                Follows naming scheme:
                rows selection - inital weight
                Example: uniform - thermal
                         uniformally generate diagonal elements and populate
                         with the thermal weight of "walkers"
            Nattempt:
                The number of attempts we want to try initalization.
                This is the random selections for stochastic initalizations
                and the number of rows in deterministic initalization.
            Heval:
                The Hamiltonian we are simulating
            HS:
                The hilbert space so we can return the correct density matrix
                dimensionality
        Out:
            f:
                The trial density matrix.
            randomrows:
                The unique random rows selected by the initalization
    '''

    randomrows = []
    f = empty_array(HS)
    if 'thermal' in init:
        thermal_weights = np.exp(-target*np.diag(Heval))
        thermal_weights /= thermal_weights.sum()

    if init == 'deterministic-thermal':
        for ii in range(Nattempts):
            if ii == HS:
                break
            f[ii,ii] = thermal_weights[ii]
        return f, randomrows

    if init == 'deterministic-constant':
        for ii in range(Nattempts):
            if ii == HS:
                break
            f[ii,ii] = 1
        return f, randomrows

    if init == 'uniform-thermal':
        randomrows = np.random.choice(HS, size=Nattempts)
        randomrows = np.unique(randomrows)

        for ii in randomrows:
            f[ii,ii] = thermal_weights[ii]
        return f, randomrows

    if init == 'uniform-constant':
        randomrows = np.random.choice(HS, size=Nattempts)
        randomrows = np.unique(randomrows)

        for ii in randomrows:
            f[ii,ii] = 1
        return f, randomrows

    if init == 'thermal-thermal':
        randomrows = np.random.choice(HS, size=Nattempts, p=thermal_weights)
        randomrows = np.unique(randomrows)

        for ii in randomrows:
            f[ii,ii] = thermal_weights[ii]
        return f, randomrows

    if init == 'thermal-constant':
        randomrows = np.random.choice(HS, size=Nattempts, p=thermal_weights)
        randomrows = np.unique(randomrows)

        for ii in randomrows:
            f[ii,ii] = 1
        return f, randomrows

    else:
        print(' Unknown initalization method:', init)
        print(' Exiting...')
        return exit()


def empty_array(hilbert_space):

    '''
        In:
            hilbert_space:
                The hilbert space of the system.
        Out:
            zeros:
                A 2D NumPy array with zeros.
    '''

    return np.zeros((hilbert_space, hilbert_space))

def write_header():

    '''
    Writes a header so we know what we are looking at from write report.
        In:
            N/A
        Out:
            N/A
    '''

    head = ' {:>10}    {:<18}    {:<18}    {:<18}    {:<18}'
    head = head.format('Beta','Shift','Tr(pH)', 'Tr(p)','Nw')
    print(head)
    return


def write_report(iteration, tau, shift, dm, hamil, df=None, printbool=True):

    '''
        In:
            iteration:
                The current iteration for the data.
            tau:
                The time step of the simulation
            shift:
                The current shift for the data simulation.
            dm:
                The current density matrix estimate.
            hamil:
                The systems Hamiltonian we are simulating.
            df (optional):
                A dictionary where we intend to store information.
            printbool (optional):
                A boolean to print out the data from the iteration.
        Out:
            stdout (optional):
                Prints the beta, shift, energy_numerator, trace
                and "walkers" for the current density matrix.
            df (optional):
                A dictionary of data that has been updated with the current
                data 
    '''

    if iteration == 0:
        write_header()

    energy_numerator = (dm @ hamil).trace()
    trace = dm.trace()
    psips = abs(dm).sum()
    curbeta = round(iteration*tau, abs(int(np.log10(tau))))

    if printbool:
        data  = ' {:>10}   {:< 1.12E}   {:< 2.12E}   {:< 3.12E}   {:< 4.12E} '
        data = data.format(curbeta,shift,energy_numerator,trace,psips)
        print(data)

    if df != None:
        df['Beta'].append(curbeta)
        df['Shift'].append(shift)
        df['Tr(Hp)'].append(energy_numerator)
        df['Tr(p)'].append(trace)
        df['Nw'].append(psips)
        df['<E>'].append(energy_numerator/trace)
        return df

    return

