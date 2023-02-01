#!/usr/bin/env python

import pickle
import numpy as np
import pandas as pd
from average_betaloops import average_betaloops as FTA
from average_betaloops import average_betaloops_rows as FTAR


def depickle(pkl):
    with open(pkl, 'rb') as handle:
        dictionary = pickle.load(handle)
    return dictionary


def get_pkl(betaT, ilvl, flvl, init, rbr, rng):
    pkl = 'stretched-H6-row-data-ipdmqmc'
    pkl += f'-betaT{betaT}'
    pkl += f'-ilvl{ilvl}'
    pkl += f'-flvl{flvl}'
    pkl += f'-initiator{init}'
    pkl += f'-rbr{rbr}'
    pkl += f'-rng{int(rng*26) + betaT + 1000}.pickle'
    return pkl


def collect_data_wrapper(betas, ilvl, flvl, init, rbr, rngs, save):

    betaT = 8
    tdf = []

    for rng in range(rngs[0], rngs[1] + 1):

        dictionary = depickle(get_pkl(betaT, ilvl, flvl, init, rbr, rng))

        tmpd = {'Beta': dictionary['beta']}

        en = np.array(dictionary['pH'])
        tr = np.array(dictionary['trace'])
        nw = np.array(dictionary['nw'])

        for irow in range(200):
            tmpd[f'Tr(Hp)_{irow}'] = en[:,irow]
            tmpd[f'Tr(p)_{irow}'] = tr[:,irow]
            tmpd[f'Nw_{irow}'] = nw[:,irow]

        tdf.append(pd.DataFrame(tmpd))

    df1 = FTAR(pd.concat(tdf), 200).reset_index(drop=False)

    if save:
        ocsv = f'row-analysis-betaT{betaT}-'
        ocsv += f'rbr{rbr}-init{init}-ilvl{ilvl}-flvl{flvl}'
        df1.to_csv(f'csvs/ipdmqmc-{ocsv}-fta.csv', index=False)
        print('\n', f'Saving {ocsv}!\n')
        print(df1)

    return df1


def main():
    '''
      rbr    init    ilvl    flvl         rng
        1       1       0       0        1-10
        1       0       0       0       11-20
        0       1       0       0       21-30
        0       0       0       0       31-40
        0       1       1       0       41-50
        0       1       0       1       51-60
    '''

    betas = np.arange(1, 11)

    df1 = collect_data_wrapper(betas, 0, 0, 1, 1, ( 1,10), True)
    df2 = collect_data_wrapper(betas, 0, 0, 0, 1, (11,20), True)
    df3 = collect_data_wrapper(betas, 0, 0, 1, 0, (21,30), True)
    df4 = collect_data_wrapper(betas, 0, 0, 0, 0, (31,40), True)
    df5 = collect_data_wrapper(betas, 1, 0, 1, 0, (41,50), True)
    df6 = collect_data_wrapper(betas, 0, 1, 1, 0, (51,60), True)


if __name__ == '__main__':
    main()
