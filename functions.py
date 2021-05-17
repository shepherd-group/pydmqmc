#!/usr/bin/env python

import numpy as np
import pandas as pd
from numpy import linalg as LA


def fci(hamil):

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


def proj_energy(hamil_matrix, Psi, report=False):

    '''
        In:
            hamil_matrix:
                THe Hamiltonian to use for the estimation
            Psi:
                The wavefunction for energy estimate.
        Out:
            Energy: The energy given as
          sum_i { hamil_matrix[i,row] rho_matrix[row,i] } / rho_matrix[row,row]
    '''

    num = (hamil_matrix[:,0] @ Psi).sum()
    den = Psi[0,0]

    if report == True:
        return num, den
    else:
        return num/den


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


def initialize_psi(HilbertSpace, Particles, Reference=0):

    '''
    Initalize the trial wavefunction assuming the reference is the 1,1
    element of our Hamiltonian.

        In:
            HilbertSpace:
                The size of the determinant space for our Hamiltonian.
            Particles:
                The number of Particles to initalize our trial with.
            Reference:
                The reference state we use as our trial wavefunction.
        Out:
            Psi:
                The trial wavefunction based on our reference.
    '''

    DF = { 'Iteration':[], 'Shift':[], 'Sum H_{0j}*Psi_{j}':[], 'Psi_{0}':[],
           'Nw':[], '<E>':[]}

    Psi = empty_array(HilbertSpace,1)
    Psi[0,Reference] += Particles

    return Psi, DF


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

    f = empty_array(HS,HS)
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


def empty_array(DimOne,DimTwo):

    '''
        In:
            DimOne:
                The row length of our array.
            DimTwo:
                The column length of our array.
        Out:
            zeros:
                A 2D NumPy array with zeros.
    '''

    return np.zeros((DimOne, DimTwo))


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


def stochastic_round(number):

    '''
    See stochastic_round_array for more information.

        In:
            number:
                The value we want to stochastically round.
        Out:
            stoch_number:
                The stochastically rounded number.
    '''

    number_sign = np.sign(number)
    stoch_number = np.abs(number) + np.random.random()
    stoch_number = np.trunc(stoch_number)*number_sign

    return stoch_number


def stochastic_round_array(array):

    '''
    This function performs a stochastic rounding on a NumPy array.
    We can stochastically round a number by generating a uniform
    decimal from 0 to 1 non-inclusive, then add this random number to the
    original number. Then we truncate, if the number was 0.3, there is a 30%
    chance we generate a number from 0.7-1.0 such that the truncated number
    is the next decimal.

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
        return rounded_array

    if round_method == 'trunc':
        rounded_array = np.trunc(array)
        return rounded_array

    if round_method == 'rint':
        rounded_array = np.rint(array)
        return rounded_array

    else:
        print(' Unknown or Unsupplied Rounding Method:', round_method)
        print(' Exiting...')
        return exit()


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
    return print(head)


def write_report(iteration, tau, shift, dm, hamil, df=None, stdout=False):

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
        Out:
            stdout (optional):
                Prints the beta, shift, energy_numerator, trace
                and "walkers" for the current density matrix.
            df (optional):
                A dictionary of data that has been updated with the current
                data 
    '''

    if iteration == 0 and stdout == True:
        write_header()

    energy_numerator = (dm @ hamil).trace()
    trace = dm.trace()
    psips = abs(dm).sum()
    curbeta = round(iteration*tau, abs(int(np.log10(tau))))

    if stdout:
        data  = ' {:<6}   {:< 1.12E}   {:< 2.12E}   {:< 3.12E}   {:< 4.12E}'
        data = data.format(curbeta,shift,energy_numerator,trace,psips)
        print(data)

    if df != None:
        df['Beta'].append(curbeta)
        df['Shift'].append(shift)
        df['Tr(Hp)'].append(energy_numerator)
        df['Tr(p)'].append(trace)
        df['Nw'].append(psips)
        df['<E>'].append(energy_numerator/trace)
        df['N_rows'].append(len(np.unique(np.nonzero(dm)[0])))
        return df


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

    mean['<E> SE'] = se['<E>']
    mean['N_rows SE'] = se['N_rows']
    mean['Tr(Hp)/Tr(p)_error'] = coverr
    mean['Tr(Hp)/Tr(p)'] = mean_energy

    return mean


def fciqmc_header():

    '''
    Writes a header so we know what we are looking at from write report.
        In:
            N/A
        Out:
            N/A
    '''

    head = ' {:<10}    {:<18}    {:<18}    {:<18}    {:<18}'
    head = head.format('Iteration','Shift','Sum H_{0j}*Psi_{j}','Psi_{0}','Nw')
    return print(head)


def fciqmc_report(iteration, tau, shift, Psi, hamil, df=None, stdout=False):

    '''
        In:
            iteration:
                The current iteration for the data.
            tau:
                The time step of the simulation
            shift:
                The current shift for the data simulation.
            Psi:
                The current Psi estimate.
            hamil:
                The systems Hamiltonian we are simulating.
            df (optional):
                A dictionary where we intend to store information.
            stdout (optional):
                A boolean to print out the data from the iteration.
        Out:
            stdout (optional):
                Prints the Iteration, shift, energy_numerator, energy_denom
                and "walkers" for the current density matrix.
            df (optional):
                A dictionary of data that has been updated with the current
                data 
    '''

    if iteration == 0 and stdout == True:
        fciqmc_header()

    num, den = proj_energy(hamil, Psi, report=True) 
    psips = abs(Psi).sum()

    if stdout:
        data  = ' {:<10}   {:< 1.12E}   {:< 2.12E}   {:< 3.12E}   {:< 4.12E}'
        data = data.format(iteration,shift,num,den,psips)
        print(data)

    if df != None:
        df['Iteration'].append(iteration)
        df['Shift'].append(shift)
        df['Sum H_{0j}*Psi_{j}'].append(num)
        df['Psi_{0}'].append(den)
        df['Nw'].append(psips)
        df['<E>'].append(num/den)
        return df


def store_fciqmc(DF, CsvName, Path=''):

    '''
    Store the data collected during an FCIQMC simulation.

        In:
            DF:
                The Pandas DataFrame of data.
            CsvName:
                What would we like to name this file?
            Path:
                Where would we like to store the file?
        Out:
            N/A
    '''

    DF = pd.DataFrame(DF)

    if len(Path) > 0 and Path[-1] == '/':
        DF.to_csv(Path+CsvName+'.csv', index=False)
    elif len(Path) == 0:
        DF.to_csv(Path+CsvName+'.csv', index=False)
    else:
        DF.to_csv(Path+'/'+CsvName+'.csv', index=False)


def simple_reblock(DF, block=5):

    '''
    Performs a simple reblock and generates a mean of the projected
    energy and shift energy for an fciqmc simulation. This is by no means
    an actual reblock but a poor attempt to perform one.

        In:
            DF:
                The Pandas DataFrame we wish to perform the reblocking
                analysis on.
            block (default=5):
                The size of the block to reblock over.
        Out:
            means:
                The average of all the variables.
            errors:
                The standard error of the means.
    '''

    # TODO Add in the covariance for the error estimate.
    means = DF.mean()
    errors = DF.std()/np.sqrt(DF.count()[0])

    return means, errors

