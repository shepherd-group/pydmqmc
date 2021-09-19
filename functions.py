#!/usr/bin/env python

import numpy as np
import pandas as pd
from numpy import linalg as LA


def symmetric_bloch(A,B,dt):
    r'''
    Calculates the symmetric bloch equation given matrix A and B
    which can be defined as:

        \frac{\delta B}{\delta t} = -\frac{1}{2}(A @ B + B @ A)

    Where "@" is a matrix multiplication operator, and A is related to B by:

        B = e^{-t A}

        In:
            A: The matrix in the exponentiation, of dimension N x N
            B: The matrix being differentiated, of dimension N x N
            dt: The derivative of the constant in exponentiation
        Out:
            dB: The finite difference of matrix A given some time step "dt"
    '''
    return -(dt/2.0)*(A @ B + B @ A)


def bloch(A,B,dt,rows=True):
    r'''
    Calculate the bloch equation given by matrix A and B, where A and B are
    related by:

        B = e^{-t A}

    Then the bloch equation is given as:

        \frac{\delta B}{\delta t} = -(B @ A) = -(A @ B)

    See "symmetric_bloch" for more information.

        In:
            A: The matrix in the exponentiation, of dimension N x N
            B: The matrix being differentiated, of dimension N x N
            dt: The derivative of the constant in exponentiation
            rows (default = True): If we are propagating along rows,
                then we calculate (B @ A), otherwise columns is (A @ B).
        Out:
            dB: The finite difference of matrix A given some time step "dt"
    '''
    if rows:
        return -dt*(B @ A)
    else:
        return -dt*(A @ B)


def ip_bloch(A,B,A0,dt):
    r'''
    Calculate the bloch equation within the interaction picture framework.
    See bloch for more details on the bloch equation.

        \frac{\delta B}{\delta t} = (A0 @ B - B @ A)

        In:
            A: The matrix in the exponentiation, of dimension N x N
            A0: The diagonal terms only of the matrix A, in the matrix
                exponentiation defined in bloch, of dimension N x N
            B: The matrix being differentiated, of dimension N x N
            dt: The derivative of the constant in exponentiation
        Out:
            dB: The finite difference of matrix A given some time step "dt"
    '''
    return dt*(A0 @ B - B @ A)


def piecewise_ip_bloch(A,B,A0,dt,current_t,piecewise_t,symmetric=False):
    '''
    A simple wrapper for piecewise interaction picture DMQMC
    This version of the algorithm runs the interaction picture bloch
    equation until a specific target beta as normal. Then continues
    propagation in temperature with either the symmetric or the asymmetric
    bloch equation.

        In:
            A: The matrix in the exponentiation, of dimension N x N
            A0: The diagonal terms only of the matrix A, in the matrix
                exponentiation defined in bloch, of dimension N x N
            B: The matrix being differentiated, of dimension N x N
            dt: The derivative of the constant in exponentiation
            current_t: The current input value "t"
            piecewise_t: The location in "t" where we switch to a different
                form of the bloch equations
            symmetric (default=True): A boolean for the symmetric of the bloch
                equation once the piecewise_t is reached
        Out:
            dB: The finite difference of matrix A given some time step "dt"
    '''
    if current_t < piecewise_t:
        delta_B = ip_bloch(A,B,A0,dt)
    else:
        if symmetric:
            delta_B = symmetric_bloch(A,B,dt)
        else:
            delta_B = bloch(A,B,dt)
    return delta_B


def FCI(hamil):
    r'''
    Do FCI (full configuration interaction) for a given Hamiltonian
    by performing exact diagonalization.

        In:
            hamil: The Hamiltonian we want to diagonalize.
                Assumed to be hermetian.
        Out:
            eigs: The eigenspectrum from diagonalization.
            vec: The corresponding eigenvectors.
    '''
    eigs, vec = LA.eigh(hamil)
    return eigs, vec


def sum_of_states(eigenspectrum, beta):
    r'''
    Calculate the energy at a beta using the sum of states method

    E = \sum(\epsilon_i*e^{-\epsilon_i * beta}) / \sum(e^{-\epsilon_i * beta})

    Use numpy longdouble other wise large beta returns infinite because
    of an floating point overflow.

        In:
            eigenspectrum: fci eigenvalues for a system
            beta: The inverse temperature(s) where we calculate the energy.
        Out:
            energy: The energy at beta for the system's eigenspectrum we used
    '''
    single_point = False
    if isinstance(beta, (float,int)):
        beta = np.array([beta])
        single_point = True

    energy = []
    for b in beta:
        numerator   = np.exp(-b * eigenspectrum)
        denominator = np.sum(np.copy(numerator))
        numerator   *= eigenspectrum
        numerator   = np.sum(numerator)
        energy.append(np.divide(numerator, denominator))

    if single_point: energy = energy[0]
    return np.array(energy)


def build_hamiltonian(hamilf):
    r'''
        In:
            hamilf: The file containing what is assumed to be a triangle 
                (upper or lower) of the Hamiltonian we are interested in.
        Out:
            hamiltonian: A 2D NumPy array representation of our matrix
            hilbert_space: The length of our determinant space
            HF: The reference energy state, this is assumed to be the zeroth
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


def unphysical_hamiltonian(hamiltonian):
    r'''
    Generates an unphysical Hamiltonian from the physical version.
    Note that unphysical is another way of saying "stoquastic".

        In:
            hamiltonian: The system Hamiltonian we are interested
                in investigating.
        Out:
            V: The unphysical version of that hamiltonian which should
                not have a sign problem.
    '''
    V = np.copy(hamiltonian)
    V_diags = np.diag(np.diag(V))
    V -= V_diags
    V_pos = -V.clip(min=0)
    V_neg = V.clip(max=0)
    V = V_pos + V_neg + V_diags
    return V


def seperate_signs(array, diagonals=False):
    r'''
    Takes a NumPy array and seperates it into two matrix's one for each sign

        In:
            array: The array we would like to decompose into two signed arrays.
            diagonals (optional, defaul=False): If True, generate three
                seperate matrix(s), two for each sign on the off-diagonal and
                a third which is the diagonals with their original signs
        Out:
            array_pos: The positive elements of the original array
            array_neg: The negative elements of the original array, returned
                with a positive sign. (Be warned!!!)
            array_diags: The diagonal elements of the original array as
                a matrix. These are returned with their original sign.
    '''
    if diagonals:
        array_diags = np.diag(np.diag(array))
        array = array - array_diags

    array_pos = array.clip(min=0)
    array_neg = abs(array.clip(max=0))
    if diagonals:
        return array_pos, array_neg, array_diags
    else:
        return array_pos, array_neg


def proj_energy(hamil, rho_matrix, row):
    r'''
        In:
            hamil: THe Hamiltonian to use for the estimation
            rho_matrix: The rho matrix to use for energy estimate.
            row: The row we are interested in.
        Out:
            Energy: The energy given as
          sum_i { hamil[i,row] rho_matrix[row,i] } / rho_matrix[row,row]
    '''
    energy = (hamil[:,row] @ rho_matrix[row,:]).sum()/rho_matrix[row,row]
    return energy


def expectation(hamil, dm, observable):
    r'''
        In:
            hamil: THe Hamiltonian to use for the estimation
            dm: The rho matrix to use for energy estimate.
            observable: An observables to calculate and return
        Out:
            expectation: The expectation for an operator "O":
                <O> = Tr(O dm) / Tr(dm)
            expectation_num: The numerator of the estimator.
            expectation_den: The trace on the density matrix.
    '''
    expectation_den = dm.trace()
    if observable == 'Energy':
        expectation_num = (hamil @ dm).trace()
    elif observable == 'von Neumann':
        expectation_num = -(dm @ np.log(dm)).trace()
    else:
        print('Unexpected observable:', observable)
        print('Please implement or check spelling!')
        print('Exiting...')
        return exit()

    expectation = np.divide(expectation_num, expectation_den)
    return expectation, expectation_num, expectation_den


def sort_index_by_diagonal(hamil, hilbert):
    r'''
        In:
            hamil: The hamiltonian we want to sort based on the
                ascending order of the diagonal elements.
            hilbert: The hilbert space of the wavefunction, makes
                forming arrays nice and quick.
        Out:
            sorted_hamil: The hamiltonian now rearanged to be ascending on the
                diagonal elements.
            sorted_diags: A list of the sorted diagonal elements.
            index_map: The index map to sort matrix's if we want to later.
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


def system_initialize(hamilf, shift=0, return_raw=False, ip=False):
    r'''
    Set up a system and return relevent matrix's for running analytical
    QMC.

        In:
            hamilf: The system Hamiltonian we are interested in.
            shift (default=0): A shift to apply to the diagonal elements of
                the Hamiltonian
            return_raw (optional, default=False): Return the raw Hamiltonian
                and the index map for the sorted?
            ip (optional, defaul=False): A boolean for running the 
                Interaction Picture, if this is true we return the
                non-interacting Hamiltonian.
        Out:
            H: The Hamiltonian shifted by the Hartree-Fock and the
                provided shift.
            Heval: The system Hamiltonian without any shifting so we can
                calculate expectation values.
            HS: The Hilbert space for the system. Very useful for generating
                matrix's and other arrays on the fly that we need for
                many calculations.
            Hraw (optional): Returns the unsorted array if that is needed.
            sorted_hash (optional): Returns the hash map used to sort the
                Hamiltonian, useful for mapping determinants to HANDE.
            H0 (optional): The non-interacting Hamiltonian.
    '''
    Hraw, HS, HF = build_hamiltonian(hamilf)
    H, sorted_diags, sorted_hash = sort_index_by_diagonal(Hraw, HS)
    Heval = np.copy(H)
    H = H - (np.eye(HS)*HF) - (np.eye(HS)*shift)

    if return_raw:
        if ip:
            return H, Heval, HS, np.diag(np.diag(H))
        else:
            return H, Heval, HS, Hraw, sorted_hash
    elif ip:
        return H, Heval, HS, np.diag(np.diag(H))
    else:
        return H, Heval, HS


def initialize_dm(init, Nattempts, target, Heval, HS, rowlist=None,
                  defined_thermal_weights=None):
    r'''
    Initalizes the starting trial density matrix. There are many ways
    to do this so we have a str check to decide.

        In:
            init:
                The initalization method we would like to use.
                Follows naming scheme:

                    rows selection - inital weight

                    Currently Implemented:
                    -------------------------------
                    deterministic-thermal:
                        Rows initialized with the thermal Hartree-Fock
                        weights on the diagonal elements. The canonical
                        starting point for IP-DMQMC

                    deterministic-uniform:
                        Rows initalized with a weight of 1 on the diagonal
                        elements. This works out to be just the identity
                        matrix and is the canonical starting point for DMQMC.

                    uniform-thermal:
                        Uniformly selects random diagonal determinants and
                        initalizes that row with a weight proportional to the
                        thermal weight from FCI Hamiltonian.

                    uniform-uniform:
                        Randomly selects diagonal determinants and adds
                        a weight of 1 to that determinant. This can happen
                        multiple times. This is how HANDE initializes the
                        density matrix.

                    thermal-thermal:
                        Selects random rows with a probability proportional
                        to the thermal weight from the FCI Hamiltonian.
                        This is not the correct way to initalize IP-DMQMC.

                    thermal-uniform:
                        Selects random rows based on probabilities proportional
                        to the thermal weight of the FCI Hamiltonian diagonal
                        elements. Then occupies that determinant with 1 walker.

                    specific-uniform:
                        Takes the optional parameter rowlist and occupies
                        those specific rows with a weight of 1.

            Nattempt: The number of attempts we want to try initalization.
                This is the random selections for stochastic initalizations
                and the number of rows in deterministic initalization.
            Heval: The Hamiltonian we are simulating
            HS: The hilbert space so we can return the correct density matrix
                dimensionality
            rowlist: A list of row index's (Python) that should be occupied.
                Currently unused!
            defined_thermal_weights: Accepts thermal weights to use instead of
                the auto-generated weights, this way the FCI weights can be
                passed instead (for example).
        Out:
            f: The trial density matrix.
            occrows: The unique random rows selected by the initalization
            df: A dictionary for storing accumulated statistics during the
                simulations.
    '''
    df = { 'Beta':[], 'Shift':[], 'Tr(Hp)':[], 'Tr(p)':[],
           'Nw':[], '<E>':[], 'N_rows':[]}
    f = np.zeros((HS, HS))

    if defined_thermal_weights is not None:
        thermal_weights = defined_thermal_weights
    elif 'thermal' in init:
        thermal_weights = np.exp(-target*np.diag(Heval))
        thermal_weights /= thermal_weights.sum()

    if init == 'deterministic-thermal':
        f = np.diag(thermal_weights)
    elif init == 'deterministic-uniform':
        f = np.eye(HS)
    elif init == 'uniform-thermal':
        randomrows = np.random.choice(HS, size=Nattempts)
        randomrows = np.bincount(randomrows, minlength=HS)
        f = np.diag(randomrows*thermal_weights)
    elif init == 'uniform-uniform':
        randomrows = np.random.choice(HS, size=Nattempts)
        randomrows = np.bincount(randomrows, minlength=HS)
        f = np.diag(randomrows)
    elif init == 'thermal-thermal':
        randomrows = np.random.choice(HS, size=Nattempts, p=thermal_weights)
        randomrows = np.bincount(randomrows, minlength=HS)
        f = np.diag(randomrows*thermal_weights)
    elif init == 'thermal-uniform':
        randomrows = np.random.choice(HS, size=Nattempts, p=thermal_weights)
        randomrows = np.bincount(randomrows, minlength=HS)
        f = np.diag(randomrows)
    elif init == 'specific-uniform':
        for ii in rowlist:
            f[ii,ii] += 1
    else:
        print(' Unknown initalization method:', init)
        print(' Exiting...')
        return exit()

    f = f.astype(np.float64)
    occrows = np.count_nonzero(f)
    return f, occrows, df


def update_shift(shift, nw, nw_old, zeta, tau, cycles, hamil, fci_dets):
    r'''
    Update the shift in the classical way.

        In:
            shift: The current shift of the simulation.
            nw: The current particle population.
            nw_old: The previous particle population.
            zeta: The damping parameter for the shift algorithm
            tau: The time step of the simualtion we are performing.
            cycles: How often we are updating the shift.
            hamil: The current system Hamiltonian used for propagation.
            fci_dets: The Hilbert Space of the Hamiltonian we are propagating
        Out:
            hamil: Our Hamiltonian with a new diagonal shift
            new_shift: The new shift estimate

        S(t) = S(t - A*tau) - zeta/(A*tau)*ln(Nw/Nw_old)
    '''
    dshift  = -zeta/(cycles*tau)
    dshift *= np.log(nw/nw_old)
    hamil = hamil - np.eye(fci_dets)*dshift
    new_shift = shift + dshift
    return hamil, new_shift


def stochastic_round_f(f):
    r'''
    Stochastically rounds a single float.

        In:
            f: A float we want to stochastically round.
        Out:
            i: An integer that results from the stochastic rounding.
                The type is still a float though for compatability.
    '''
    i = f + np.sign(f)*np.random.random()
    return np.trunc(f)


def stochastic_round(array, threshold=1.0):
    r'''
    This function performs a stochastic rounding on a NumPy array.

        In:
            array: An array that we want the values to be stochastically
                rounded in.
            threshold (default=1.0): The decimal place we would like to
                perform stochastic rounding too. If threshold is less than 1,
                then we scale and round to that decimal location and rescale
                back after rounding.
        Out:
            stoch_rounded_array: The stochastically rounded version of
                the input array.
    '''
    shape = array.shape
    p_matrix = np.random.random(shape)
    array_sign = np.sign(array)
    p_matrix = np.multiply(p_matrix, array_sign)
    stoch_rounded_array = np.divide(array, threshold)+ p_matrix
    stoch_rounded_array = np.trunc(stoch_rounded_array)
    stoch_rounded_array = np.multiply(stoch_rounded_array, threshold)
    return stoch_rounded_array


def write_header():
    r'''
    Writes a header so we know what we are looking at from write report.

        In:
            N/A
        Out:
            N/A
    '''
    head = ' {:>8}    {:<18}    {:<18}    {:<18}    {:<18}'
    head = head.format('Beta','Shift','Tr(pH)', 'Tr(p)','Nw')
    return print(head)


def write_report(iteration, tau, shift, dm, hamil, df=None, stdout=False,
                 ind_row_evals=False):
    r'''
        In:
            iteration: The current iteration for the data.
            tau: The time step of the simulation
            shift: The current shift for the data simulation.
            dm: The current density matrix estimate.
            hamil: The systems Hamiltonian we are simulating.
            df (optional): A dictionary where we intend to store information.
            stdout (optional): A boolean to print out the data from the
                iteration.
            ind_row_evals (optional): A boolean which flags individual row
                energy estimates. If this is true a loop is performed over
                the entire list of rows on the density matrix, if they are
                non-zero an energy estimate is performed and stored in df.
        Out:
            stdout (optional): Prints the beta, shift, energy_numerator, trace
                and "walkers" for the current density matrix.
            df (optional): A dictionary of data that has been updated with
                the current data.
    '''
    if iteration == 0 and stdout:
        write_header()

    energy, energy_numerator, trace = expectation(hamil, dm, 'Energy')
    psips = abs(dm).sum()
    #repbeta = round(iteration*tau, int(np.ceil(abs(np.log10(tau)))))
    repbeta = round(iteration*tau, 6)
    curbeta = iteration*tau

    if stdout:
        data  = ' {:> 8}   {:< 1.12E}   {:< 2.12E}   {:< 3.12E}   {:< 4.12E}'
        #data = data.format(curbeta,shift,energy_numerator,trace,psips)
        data = data.format(repbeta,shift,energy_numerator,trace,psips)
        print(data)

    if df != None:
        occ_rows = np.unique(np.nonzero(dm)[0])
        df['Beta'].append(curbeta)
        df['Shift'].append(shift)
        df['Tr(Hp)'].append(energy_numerator)
        df['Tr(p)'].append(trace)
        df['Nw'].append(psips)
        df['<E>'].append(energy)
        df['N_rows'].append(len(occ_rows))

        if ind_row_evals:
            for row in occ_rows:
                lab = 'E(p['+str(row+1)+',*])'
                if iteration == 0:
                    df[lab] = []

                row_ene = proj_energy(hamil, dm, row)
                df[lab].append(row_ene)

        return df

    return


def store_data(data, df, betaloop, beta_loops, csv, path=''):
    r'''
    Store data in an array. When the calculation is complete concat it
    and save it as a csv to the specified path. This really should be a
    class object but thats another days problem.

        In:
            df: A Pandas DataFrame to store our data from different
                trajectories.
            betaloop: The current beta loop we are at.
            beta_loops: The number of beta loops we will store.
            csv: The name of the csv we want to store.
            path (defaul = current working directory):
                The path where we want to store the data.
        Out:
            Data: An array of all the data we have accumulated from beta loops.
                Or if the beta loop cycle is compelte it returns an empty
                array after saving the data to a specified location so we
                can do another cycle with a different set of parameters.
            csv (Saved as a file): An array to save as a file.
    '''
    data.append(pd.DataFrame(df))

    if betaloop == beta_loops:
        data = pd.concat(data, ignore_index=True)
        data.to_csv(path + csv + '.csv', index=False)
        data = []

    return data


def average_betaloops(df):
    r'''
    Average the data in a Pandas DataFrame object.

        In:
            df: A data frame of beta loops concat'd together
        Out:
            mean: means of all the data from the Data Frame, and also
                the average energy from the <Tr(Hp)> / <Tr(p)> estimate
                with the appropriate errors.
    '''
    groupdf = df.groupby('Beta')
    count = groupdf.count()
    mean = groupdf.mean()
    se = groupdf.sem()

    cov = groupdf.cov()['Tr(Hp)'].loc[:,'Tr(p)']
    mean_energy = mean['Tr(Hp)']/mean['Tr(p)']
    coverr  = (se['Tr(p)']/mean['Tr(p)'])**2 + (se['Tr(Hp)']/mean['Tr(Hp)'])**2
    coverr -= 2*cov/(count['Tr(Hp)']*mean['Tr(Hp)']*mean['Tr(p)'])
    coverr  = abs(mean_energy*np.sqrt(coverr))

    mean['Tr(Hp)/Tr(p)_error'] = coverr
    mean['Tr(Hp)/Tr(p)'] = mean_energy

    for key in list(se.columns):
        if not(key+'_error' in list(mean.columns)):
            mean[key+'_error'] = se[key]

    return mean


def complex_report(iteration, tau, shift, dm, hamil, df=None, stdout=False,
                   ind_row_evals=False):
    r'''
    See write_report for more information. This is a version of write_report
    which treats complex density matrix's.
    '''
    if iteration == 0 and stdout:
        head  = ' {:>6}    {:<18}    {:<18}    {:<18}    {:<18}    '
        head += '{:<18}    {:<18}    {:<18}'
        head = head.format('Beta','Shift','Re{Tr(pH)}','Im{Tr(pH)}',
                           'Re{Tr(p)}','Im{Tr(p)}','Re{Nw}','Im{Nw}')
        print(head)

    re_dm = np.real(dm)
    im_dm = np.imag(dm)

    re_energy, re_energy_num, re_trace = expectation(hamil, re_dm, 'Energy')
    re_psips = abs(re_dm).sum()

    im_energy, im_energy_num, im_trace = expectation(hamil, im_dm, 'Energy')
    im_psips = abs(im_dm).sum()

    curbeta = round(iteration*tau, abs(int(np.log10(tau))))

    if stdout:
        data  = ' {:> 5}   {:< 1.12E}   {:< 1.12E}   {:< 1.12E}   {:< 1.12E}'
        data += '   {:< 1.12E}   {:< 1.12E}   {:< 1.12E}'
        data = data.format(curbeta,shift,re_energy_num,
                           im_energy_num,re_trace,im_trace,
                           re_psips,im_psips)
        print(data)

    if df != None:
        occ_rows = np.unique(np.nonzero(dm)[0])
        df['Beta'].append(curbeta)
        df['Shift'].append(shift)
        df['Re{Tr(Hp)}'].append(re_energy_num)
        df['Im{Tr(Hp)}'].append(im_energy_num)
        df['Re{Tr(p)}'].append(re_trace)
        df['Im{Tr(p)}'].append(im_trace)
        df['Re{Nw}'].append(re_psips)
        df['Im{Nw}'].append(im_psips)
        df['Re{<E>}'].append(re_energy)
        df['Im{<E>}'].append(im_energy)
        df['N_rows'].append(len(occ_rows))
        return df

    return


