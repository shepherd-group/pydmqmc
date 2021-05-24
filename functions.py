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
                The inverse temperature where we calculate the energy.
                If beta is a list, then we return as list of energies
                at all the beta's.
        Out:
            energy:
                The energy at beta for the system's eigenspectrum we used
    '''

    if isinstance(beta, (list,np.ndarray)):
        energy = []
        for b in beta:
            numerator   = np.exp(-b * eigenspectrum)
            denominator = np.sum(np.copy(numerator))
            numerator   *= eigenspectrum
            numerator   = np.sum(numerator)
            energy.append(np.divide(numerator, denominator))
        return energy
    else:
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


def tmatrix(hamiltonian, shift=0):

    '''
        In:
            hamiltonian:
                The system we are interested in.
            shift (default = 0):
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


def seperate_signs(array, diagonals=False):

    '''
    Takes a NumPy array and seperates it into two matrix's one for each sign

        In:
            array:
                The array we would like to decompose into two signed arrays.
            diagonals (optional, defaul=False):
                If True, generate three matrix's, two for each sign and the
                last is the diagonals with their original signs
        Out:
            array_pos:
                The positive elements of the original array
            array_neg:
                The negative elements of the original array, returned
                with a positive sign. (Be warned!!!)
            array_diags:
                The diagonal elements of the original array as a matrix.
                These are returned with their original sign.
    '''

    if diagonals == True:
        array_diags = np.diag(np.diag(array))
        array = array - array_diags

        array_pos = array.clip(min=0)
        array_neg = abs(array.clip(max=0))

        return array_pos, array_neg, array_diags

    else: 
        array_pos = array.clip(min=0)
        array_neg = abs(array.clip(max=0))

        return array_pos, array_neg


def proj_energy(hamil, rho_matrix, row):

    '''
        In:
            hamil:
                THe Hamiltonian to use for the estimation
            rho_matrix:
                The rho matrix to use for energy estimate.
            row:
                The row we are interested in.
        Out:
            Energy: The energy given as
          sum_i { hamil[i,row] rho_matrix[row,i] } / rho_matrix[row,row]
    '''

    energy = (hamil[:,row] @ rho_matrix[row,:]).sum()/rho_matrix[row,row]

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
            expectation_num:
                The numerator of the estimator.
            expectation_den:
                The trace on the density matrix.
    '''

    if 'Energy' == observable:
        expectation_num = (hamil @ dm).trace()
        expectation_den = dm.trace()
        if expectation_den == 0.0:
            expectation = np.nan
        else:
            expectation = np.divide(expectation_num, expectation_den)
        return expectation, expectation_num, expectation_den

    else:
        print('Unexpected observable:', observable)
        print('Please implement or check spelling!')
        print('Exiting...')
        return exit()


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


def system_initialize(hamilf, shift=0, return_raw=False):

    '''
    Set up a system and return relevent matrix's for running analytical
    QMC.

        In:
            hamilf:
                The system Hamiltonian we are interested in.
            shift (default=0):
                A shift to apply to the diagonal elements of the Hamiltonian
            return_raw (optional, default=False):
                Return the raw Hamiltonian and the index map for the sorted?
        Out:
            H:
                The Hamiltonian shifted by the Hartree-Fock and the
                provided shift.
            Heval:
                The system Hamiltonian without any shifting so we can
                calculate expectation values.
            HS:
                The Hilbert space for the system. Very useful for generating
                matrix's and other arrays on the fly that we need for
                many calculations.
            Hraw (optional):
                Returns the unsorted array if that is needed.
            sorted_hash (optional):
                Returns the hash map used to sort the Hamiltonian, useful
                for mapping determinants to HANDE.
    '''

    Hraw, HS, HF = build_hamiltonian(hamilf)
    H, sorted_diags, sorted_hash = sort_index_by_diagonal(Hraw, HS)
    Heval = np.copy(H)
    H = H - (np.eye(HS)*HF) - (np.eye(HS)*shift)

    if return_raw:
        return H, Heval, HS, Hraw, sorted_hash

    return H, Heval, HS


def non_interacting(hamiltonian):

    '''
    Generate a non-interacting Hamiltonian from an interacting one.
    This just converts the off-diagonal elements to zero.

        In:
            hamiltonian:
                The Hamiltonian we want to generate a non-interacting
                version of.
        Out:
            mean_field_h:
                The non-interacting hamiltonian.
    '''
    return np.diag(np.diag(hamiltonian))


def initialize_dm(init, Nattempts, target, Heval, HS, rowlist=None,
                  thermal_weights=None):

    '''
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

            Nattempt:
                The number of attempts we want to try initalization.
                This is the random selections for stochastic initalizations
                and the number of rows in deterministic initalization.
            Heval:
                The Hamiltonian we are simulating
            HS:
                The hilbert space so we can return the correct density matrix
                dimensionality
            rowlist:
                A list of row index's (Python) that should be occupied.
                Currently unused!
            thermal_weights:
                Accepts thermal weights to use instead of the auto-generated
                weights, this way the FCI weights can be passed instead.
        Out:
            f:
                The trial density matrix.
            occrows:
                The unique random rows selected by the initalization
            df:
                A dictionary for storing accumulated statistics during the
                simulations.
    '''

    df = { 'Beta':[], 'Shift':[], 'Tr(Hp)':[], 'Tr(p)':[],
           'Nw':[], '<E>':[], 'N_rows':[]}

    f = empty_array(HS)
    if 'thermal' in init and not(isinstance(thermal_weights, np.ndarray)):
        thermal_weights = np.exp(-target*np.diag(Heval))
        thermal_weights /= thermal_weights.sum()

    if init == 'deterministic-thermal':
        f = np.diag(thermal_weights)
        occrows = np.count_nonzero(f)
        return f, occrows, df

    if init == 'deterministic-uniform':
        f = np.eye(HS)
        occrows = np.count_nonzero(f)
        return f, occrows, df

    if init == 'uniform-thermal':
        randomrows = np.random.choice(HS, size=Nattempts)
        randomrows = np.bincount(randomrows, minlength=HS)
        occrows = np.count_nonzero(randomrows)

        for ii, nw in enumerate(randomrows):
            f[ii,ii] += thermal_weights[ii]*nw
        return f, occrows, df

    if init == 'uniform-uniform':
        randomrows = np.random.choice(HS, size=Nattempts)
        randomrows = np.bincount(randomrows, minlength=HS)
        occrows = np.count_nonzero(randomrows)

        for ii, nw in enumerate(randomrows):
            f[ii,ii] += nw
        return f, occrows, df

    if init == 'thermal-thermal':
        randomrows = np.random.choice(HS, size=Nattempts, p=thermal_weights)
        randomrows = np.bincount(randomrows, minlength=HS)
        occrows = np.count_nonzero(randomrows)

        for ii, nw in enumerate(randomrows):
            f[ii,ii] += thermal_weights[ii]*nw
        return f, occrows, df

    if init == 'thermal-uniform':
        randomrows = np.random.choice(HS, size=Nattempts, p=thermal_weights)
        randomrows = np.bincount(randomrows, minlength=HS)
        occrows = np.count_nonzero(randomrows)

        for ii, nw in enumerate(randomrows):
            f[ii,ii] += nw
        return f, occrows, df

    if init == 'specific-uniform':
        for ii in rowlist:
            f[ii,ii] += 1
        occrows = np.count_nonzero(f)
        return f, occrows, df

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


def update_shift(H, HS, cycles, tau, df, zeta):

    '''
    Update the shift in the classical way.

        In:
            H:
                The current system Hamiltonian used for propagation.
            HS:
                The Hilbert Space of the Hamiltonian we are propagating
            shift:
                The current shift.
            cycles:
                How often we are updating the shift.
            tau:
                The time step of the simualtion we are performing.
            df:
                The dictionary of data to collect the shift, current particles
                and old particles.
            zeta:
                The damping parameter for the shift algorithm
        Out:
            H:
                Our Hamiltonian with a new diagonal shift
            new_shift:
                The new shift estimate
    '''

    shift   = df['Shift'][-1]
    nw      = df['Nw'][-1]
    nw_old  = df['Nw'][-2]

    dshift  = -zeta/(cycles*tau)
    dshift *= np.log(nw/nw_old)

    H = H - (dshift*np.eye(HS))

    new_shift = shift + dshift

    return H, new_shift


def stochastic_round_f(f):

    '''
    Stochastically rounds a single float.

        In:
            f:
                A float we want to stochastically round.
        Out:
            i:
                An integer that results from the stochastic rounding.
                The type is still a float though for compatability.
    '''

    i = f + np.sign(f)*np.random.random()
    return np.trunc(f)


def stochastic_round(array):

    '''
    This function performs a stochastic rounding on a NumPy array.

        In:
            array:
                An array that we want the values to be stochastically rounded
                in.
        Out:
            stoch_rounded_array:
                The stochastically rounded version of the input array.
    '''

    shape = array.shape
    p_matrix = np.random.random(shape)

    array_sign = np.sign(array)
    p_matrix = np.multiply(p_matrix, array_sign)

    stoch_rounded_array = array + p_matrix
    stoch_rounded_array = np.trunc(stoch_rounded_array)

    return stoch_rounded_array


def deterministic_round(array, round_method, decimals=0):

    '''
    This function performs a deterministic rounding on a NumPy array.
    This is in no way complete and is for preliminary testing of deterministic
    plateaus.

        In:
            array:
                An array that we want the values to be stochastically rounded
                in.
            round_method:
                How do we want to deterministically round the array?
            decimals (optional, default=0):
                round to a specifica decimal place.
        Out:
            rounded_array:
                The stochastically rounded version of the input array.
    '''

    if round_method == 'decimal':
        rounded_array = np.around(array, decimals=decimals)

    if round_method == 'trunc':
        rounded_array = np.trunc(array)

    if round_method == 'rint':
        rounded_array = np.rint(array)

    else:
        print(' Unknown or Unsupplied Rounding Method', round_method)
        print(' Exiting...')
        return exit()

    return rounded_array


def write_header():

    '''
    Writes a header so we know what we are looking at from write report.
        In:
            N/A
        Out:
            N/A
    '''

    head = ' {:>6}    {:<18}    {:<18}    {:<18}    {:<18}'
    head = head.format('Beta','Shift','Tr(pH)', 'Tr(p)','Nw')
    print(head)
    return


def write_report(iteration, tau, shift, dm, hamil, df=None, stdout=False,
                 ind_row_evals=False):

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
            stdout (optional):
                A boolean to print out the data from the iteration.
            ind_row_evals (optional):
                A boolean which flags individual row energy estimates.
                If this is true a loop is performed over the entire list
                of rows on the density matrix, if they are non-zero an energy
                estimate is performed and stored in df.
        Out:
            stdout (optional):
                Prints the beta, shift, energy_numerator, trace
                and "walkers" for the current density matrix.
            df (optional):
                A dictionary of data that has been updated with the current
                data 
    '''

    if iteration == 0 and stdout:
        write_header()

    energy, energy_numerator, trace = expectation(hamil, dm, 'Energy')
    psips = abs(dm).sum()
    curbeta = round(iteration*tau, abs(int(np.log10(tau))))

    if stdout:
        data  = ' {:> 5}   {:< 1.12E}   {:< 2.12E}   {:< 3.12E}   {:< 4.12E}'
        data = data.format(curbeta,shift,energy_numerator,trace,psips)
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

    '''
    Store data in an array. When the calculation is complete concat it
    and save it as a csv to the specified path. This really should be a
    class object but thats another days problem.

        In:
            df:
                A Pandas DataFrame to store our data from different
                trajectories.
            betaloop:
                The current beta loop we are at.
            beta_loops:
                The number of beta loops we will store.
            csv:
                The name of the csv we want to store.
            path (defaul = current working directory):
                The path where we want to store the data.
        Out:
            Data:
                An array of all the data we have accumulated from beta loops.
                Or if the beta loop cycle is compelte it returns an empty
                array after saving the data to a specified location so we
                can do another cycle with a different set of parameters.
            csv (Saved as a file):
                An array to save as a file.
    '''

    data.append(pd.DataFrame(df))

    if betaloop == beta_loops:
        data = pd.concat(data, ignore_index=True)
        data.to_csv(path + csv + '.csv', index=False)
        data = []
        return data

    return data


def average_betaloops(df):

    '''
    Average the data in a Pandas DataFrame object.

        In:
            df:
                A data frame of beta loops concat'd together
        Out:
            mean:
                means of all the data from the Data Frame, and also
                the average energy from the <Tr(Hp)> / <Tr(p)> estimate
                with the appropriate errors.
    '''

    groupdf = df.groupby('Beta')
    count = groupdf.count()
    mean = groupdf.mean()
    se = groupdf.std()/np.sqrt(count-1)

    cov = groupdf.cov()
    cov = cov['Tr(p)'].iloc[cov.index.get_level_values(1) == 'Tr(Hp)']
    cov = cov.reset_index().set_index('Beta')['Tr(p)']
    mean_energy = mean['Tr(Hp)']/mean['Tr(p)']
    coverr  = (se['Tr(p)']/mean['Tr(p)'])**2
    coverr += (se['Tr(Hp)']/mean['Tr(Hp)'])**2
    coverr -= 2*cov/(count['Tr(Hp)']*mean['Tr(Hp)']*mean['Tr(p)'])
    coverr  = abs(mean_energy*np.sqrt(coverr))

    mean['Nw SE'] = se['Nw']
    mean['<E> SE'] = se['<E>']
    mean['N_rows SE'] = se['N_rows']
    mean['Tr(Hp)/Tr(p)_error'] = coverr
    mean['Tr(Hp)/Tr(p)'] = mean_energy

    return mean


def complex_report(iteration, tau, shift, dm, hamil, df=None, stdout=False,
                   ind_row_evals=False):

    '''
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

    re_energy, re_energy_numerator, re_trace = expectation(hamil, re_dm, 'Energy')
    re_psips = abs(re_dm).sum()

    im_energy, im_energy_numerator, im_trace = expectation(hamil, im_dm, 'Energy')
    im_psips = abs(im_dm).sum()

    curbeta = round(iteration*tau, abs(int(np.log10(tau))))

    if stdout:
        data  = ' {:> 5}   {:< 1.12E}   {:< 1.12E}   {:< 1.12E}   {:< 1.12E}'
        data += '   {:< 1.12E}   {:< 1.12E}   {:< 1.12E}'
        data = data.format(curbeta,shift,re_energy_numerator,
                           im_energy_numerator,re_trace,im_trace,
                           re_psips,im_psips)
        print(data)

    if df != None:
        occ_rows = np.unique(np.nonzero(dm)[0])

        df['Beta'].append(curbeta)
        df['Shift'].append(shift)
        df['Re{Tr(Hp)}'].append(re_energy_numerator)
        df['Im{Tr(Hp)}'].append(im_energy_numerator)
        df['Re{Tr(p)}'].append(re_trace)
        df['Im{Tr(p)}'].append(im_trace)
        df['Re{Nw}'].append(re_psips)
        df['Im{Nw}'].append(im_psips)
        df['Re{<E>}'].append(re_energy)
        df['Im{<E>}'].append(im_energy)
        df['N_rows'].append(len(occ_rows))

        return df

    return
