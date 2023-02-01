#!/usr/bin/env python

import numpy as np
import pandas as pd


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


#def average_betaloops_rows(df, rows, tkey, enest=False):
def average_betaloops_rows(df, rows, enest=False):
    groupdf = df.groupby('Beta')
    mean = groupdf.mean()
    se = groupdf.sem()

    results = {'Beta': np.array(mean.index)}

    #if enest:
    #    count = groupdf.count()
    #    gcov = groupdf.cov()

    #    for irow in range(rows):
    #        cov = gcov[f'Tr(Hp)_{irow}'].loc[:,f'Tr(p)_{irow}']
    #        mean_energy = mean[f'Tr(Hp)_{irow}']/mean[f'Tr(p)_{irow}']
    #        coverr  = (se[f'Tr(p)_{irow}']/mean[f'Tr(p)_{irow}'])**2 + (se[f'Tr(Hp)_{irow}']/mean[f'Tr(Hp)_{irow}'])**2
    #        coverr -= 2*cov/(count[f'Tr(Hp)_{irow}']*mean[f'Tr(Hp)_{irow}']*mean[f'Tr(p)_{irow}'])
    #        coverr  = abs(mean_energy*np.sqrt(coverr))

    #        mean[f'Tr(Hp)/Tr(p)_{irow}_error'] = coverr
    #        mean[f'Tr(Hp)/Tr(p)_{irow}'] = mean_energy
    #else:
    #    for irow in range(rows):
    #        if tkey == 'S':
    #            results[f'W(S)_{irow}_error'] = np.array(se[f'W(S)_{irow}'])
    #            results[f'W(S)_{irow}'] = np.array(mean[f'W(S)_{irow}'])
    #            results[f'S_{irow}_error'] = np.array(se[f'S_{irow}'])
    #            results[f'S_{irow}'] = np.array(mean[f'S_{irow}'])
    #        else:
    #            results[f'{tkey}_{irow}_error'] = np.array(se[f'{tkey}_{irow}'])
    #            results[f'{tkey}_{irow}'] = np.array(mean[f'{tkey}_{irow}'])

    for irow in range(rows):
        results[f'Tr(Hp)_{irow}_error'] = np.array(se[f'Tr(Hp)_{irow}'])
        results[f'Tr(Hp)_{irow}'] = np.array(mean[f'Tr(Hp)_{irow}'])
        results[f'Tr(p)_{irow}_error'] = np.array(se[f'Tr(p)_{irow}'])
        results[f'Tr(p)_{irow}'] = np.array(mean[f'Tr(p)_{irow}'])
        results[f'Nw_{irow}_error'] = np.array(se[f'Nw_{irow}'])
        results[f'Nw_{irow}'] = np.array(mean[f'Nw_{irow}'])
        results = pd.DataFrame(results)

    return results
