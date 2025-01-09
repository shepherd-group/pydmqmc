#!/usr/bin/env python

import os
import numba as nb
import numpy as np
import pandas as pd

from typing import Any, List


class Calculation:
    def __init__(
            self,
            matrix_file: str,
            state_object: Any,
            matrix_constructor: callable = None,
        ) -> None:
        ''' Holds onto information for each calculation because I can not be
        bothered to retype things after doing so many times.
        '''
        self.sym = state_object

        mats = matreader(matrix_file, constructor=matrix_constructor)

        self.D = mats['D']
        self.S = mats['S']
        self.C = mats['C']
        self.H1 = mats['H1E']
        self.dx = mats['MUX']
        self.dy = mats['MUY']
        self.dz = mats['MUZ']

        self.Dmo = (((np.transpose(self.C) @ self.S) @ self.D) @ self.S) @ self.C
        self.H1mo = (self.C.transpose() @ self.H1) @ self.C
        self.dxmo = (self.C.transpose() @ self.dx) @ self.C
        self.dymo = (self.C.transpose() @ self.dy) @ self.C
        self.dzmo = (self.C.transpose() @ self.dz) @ self.C

        self.Sinv = np.linalg.inv(self.S)
        self.Cinv = np.linalg.inv(self.C)
        self.Cdinv = np.linalg.inv(self.C.transpose())

        self.SinvCdinv = self.Sinv @ self.Cdinv
        self.CinvSinv = self.Cinv @ self.Sinv


class State:
    def __init__(
            self,
            index: int,
            data: dict,
            key: tuple,
            norbs: int,
            hamil: str = None,
            h0: dict = None,
            k0: tuple = None,
        ) -> None:

        self.M = norbs
        self.i = index
        self.k = key
        self.fci = data[self.k]
        self.dets = getdets(self.k[1])
        self.idet, self.psis = getwfns(self.k[1], self.fci.shape[0])
        self.Ei = self.fci[self.i]
        self.Dj = self.idet[self.i]
        self.Cj = self.psis[self.i]
        self.j = np.arange(self.Cj.shape[0]) + 1
        self.Xj = self.getbas(self.Cj.shape[0], norbs, self.Dj)
        self.powers = np.power(2, np.arange(norbs))

        if hamil is not None:
            self.ham = getham(hamil)

        if h0 is not None and k0 is not None:
            self.H0 = h0[k0]
            self.D0 = np.power(2, getdets(k0[1]) - 1).sum(axis=1)

    def enum(self) -> zip:
        return zip(self.j, self.Dj, self.Xj, self.Cj)

    def ket_bra(
            self,
            B: Any,
            spatial: bool = False,
        ) -> list[list[float]]:
        A = self
        assert A.M == B.M
        M = A.M

        orbs = np.arange(M)

        if spatial:
            tdm = np.zeros((M//2, M//2))
        else:
            tdm = np.zeros((M, M))

        for a, Da, Xa, Ca in A.enum():
            if abs(Ca) < 1E-12:
                continue

            aOcc = orbs[Xa == 1]

            for b, Db, Xb, Cb in B.enum():
                if abs(Cb) < 1E-12:
                    continue
                elif abs(Ca*Cb) < 1E-12:
                    continue

                bOcc = orbs[Xb == 1]

                nex = aOcc.shape[0] - np.isin(aOcc, bOcc).sum()

                if nex == 0:
                    for iocc in aOcc:
                        if spatial:
                            tdm[iocc//2,iocc//2] += Ca*Cb
                        else:
                            tdm[iocc,iocc] += Ca*Cb
                elif nex == 1:
                    amask = np.isin(aOcc, bOcc)
                    bmask = np.isin(bOcc, aOcc)

                    q = aOcc[np.logical_not(amask)][0]
                    s = bOcc[np.logical_not(bmask)][0]

                    perms = int(2*aOcc.shape[0]) - 2
                    perms -= np.where(aOcc == q)[0][0]
                    perms -= np.where(bOcc == s)[0][0]
                    Gamma_pr = -1 if (perms % 2) else 1

                    if spatial:
                        tdm[s//2,q//2] += Cb*Gamma_pr*Ca
                    else:
                        tdm[s,q] += Cb*Gamma_pr*Ca

        return tdm

    def irep(self, X: list[int]) -> int:
        return self.powers[np.array(X) == 1].sum()

    @staticmethod
    def getbas(ndets: int, norbs: int, D: list[int]) -> list[list[int]]:
        ba = np.zeros((ndets, norbs), dtype=int)
        for j, Dj in enumerate(D):
            # Edge cases when M = 64...
            # Take exact determinant from DET file.
            if Dj == -8070450532247928829:
                a = [0 for _ in range(norbs)]
                a[1-1] = 1
                a[2-1] = 1
                a[61-1] = 1
                a[64-1] = 1
                a = a[::-1]
            elif Dj == -4611686018427387901:
                a = [0 for _ in range(norbs)]
                a[1-1] = 1
                a[2-1] = 1
                a[63-1] = 1
                a[64-1] = 1
                a = a[::-1]
            else:
                a = [int(k) for k in list(bin(Dj))[2:]]
            ba[j,-len(a):] = a
            ba[j,:] = ba[j,::-1]
        return ba


@nb.njit
def njittdm(
        nA: np.array,
        CA: np.array,
        nB: np.array,
        CB: np.array,
        tdm: np.array,
    ) -> np.array:

    N = sum(nA[0])
    M = len(nA[0])

    orbs = np.arange(M)

    for a in range(len(nA)):
        Ca = CA[a]

        if abs(Ca) < 1E-12:
            continue

        na = nA[a]
        aocc = orbs[na == 1]

        for b in range(len(nB)):
            Cb = CB[b]

            if abs(Cb) < 1E-12:
                continue

            nb = nB[b]
            bocc = orbs[nb == 1]

            CaCb = Ca*Cb

            if abs(CaCb) < 1E-12:
                continue

            amask = [aorb in bocc for aorb in aocc]
            nex = N - sum(amask)

            # nex == 0 only matters for | \Psi_A >< \Psi_A |
            # in which case we should be calculating the standard RDM.
            if nex == 0:
                for aorb in aocc:
                    tdm[aorb//2,aorb//2] += CaCb
            if nex != 1:
                continue

            bmask = [borb in aocc for borb in bocc]

            qindex = amask.index(False)
            sindex = bmask.index(False)

            q = aocc[qindex]
            s = bocc[sindex]

            perms = 2*N - 2 - qindex - sindex

            CaCb *= -1 if (perms % 2) else 1

            tdm[s//2,q//2] += CaCb

    return tdm


@nb.njit
def njittdm_cont_orbs(
        nA: np.array,
        CA: np.array,
        nB: np.array,
        CB: np.array,
        tdm: np.array,
        conts: np.array,
    ) -> np.array:

    N = sum(nA[0])
    M = len(nA[0])

    orbs = np.arange(M)

    for a in range(len(nA)):
        Ca = CA[a]

        if abs(Ca) < 1E-12:
            continue

        na = nA[a]
        aocc = orbs[na == 1]

        for b in range(len(nB)):
            Cb = CB[b]

            if abs(Cb) < 1E-12:
                continue

            nb = nB[b]
            bocc = orbs[nb == 1]

            CaCb = Ca*Cb

            if abs(CaCb) < 1E-12:
                continue

            amask = [aorb in bocc for aorb in aocc]
            nex = N - sum(amask)

            # nex == 0 only matters for | \Psi_A >< \Psi_A |
            # in which case we should be calculating the standard RDM.
            if nex == 0:
                for aorb in aocc:
                    tdm[aorb//2,aorb//2] += CaCb
                    conts[aorb//2,aorb//2] = True
            if nex != 1:
                continue

            bmask = [borb in aocc for borb in bocc]

            qindex = amask.index(False)
            sindex = bmask.index(False)

            q = aocc[qindex]
            s = bocc[sindex]

            perms = 2*N - 2 - qindex - sindex

            CaCb *= -1 if (perms % 2) else 1

            tdm[s//2,q//2] += CaCb
            conts[s//2,q//2] = True

    return tdm, conts


@nb.njit
def njittdm_spin_orbs(
        nA: np.array,
        CA: np.array,
        nB: np.array,
        CB: np.array,
        tdm: np.array,
        conts: np.array,
    ) -> np.array:

    N = sum(nA[0])
    M = len(nA[0])

    orbs = np.arange(M)

    for a in range(len(nA)):
        Ca = CA[a]

        if abs(Ca) < 1E-12:
            continue

        na = nA[a]
        aocc = orbs[na == 1]

        for b in range(len(nB)):
            Cb = CB[b]

            if abs(Cb) < 1E-12:
                continue

            nb = nB[b]
            bocc = orbs[nb == 1]

            CaCb = Ca*Cb

            if abs(CaCb) < 1E-12:
                continue

            amask = [aorb in bocc for aorb in aocc]
            nex = N - sum(amask)

            # nex == 0 only matters for | \Psi_A >< \Psi_A |
            # in which case we should be calculating the standard RDM.
            if nex == 0:
                for aorb in aocc:
                    tdm[aorb,aorb] += CaCb
                    conts[aorb,aorb] = True
            if nex != 1:
                continue

            bmask = [borb in aocc for borb in bocc]

            qindex = amask.index(False)
            sindex = bmask.index(False)

            q = aocc[qindex]
            s = bocc[sindex]

            perms = 2*N - 2 - qindex - sindex

            CaCb *= -1 if (perms % 2) else 1

            tdm[s,q] += CaCb
            conts[s,q] = True

    return tdm, conts


@nb.njit
def psisorter(
        refdet: np.array,
        tardet: np.array,
        tarpsi: np.array,
        newpsi: np.array,
    ) -> np.array:

    for i in range(len(tardet)):
        newpsi[refdet == tardet[i]] = tarpsi[i]

    return newpsi


@nb.njit
def spectrapad(
        s0: np.array,
        d0: np.array,
        s1: np.array,
        d1: np.array,
        s2: np.array,
        d2: np.array,
        es1: np.array,
        es2: np.array,
    ) -> tuple:

    iloc1 = 0
    iloc2 = 0

    for i in range(len(s0)):
        det0 = d0[i]

        if det0 in d1:
            es1[i] = s1[iloc1]
            iloc1 += 1

        if det0 in d2:
            es2[i] = s2[iloc2]
            iloc2 += 1

    return es1, es2


def getham(output: str) -> np.array:
    ndets = 0
    data = {}

    with open(output, 'rt') as stream:
        for line in stream:
            i, j, hij = line.split()
            i, j = int(i), int(j)

            if (n := max(i, j)) > ndets:
                ndets = n

            data[(i-1,j-1)] = float(hij)

    h = np.zeros((ndets, ndets), dtype=float)

    for (i, j), hij in data.items():
        h[i,j] = hij
        h[j,i] = hij

    return h


def getdata(output: str) -> dict:

    sym = None
    store = False
    detfile = None
    data = {}

    with open(output, 'rt') as stream:
        for line in stream:
            ld = line.split()

            if '"symmetry":' in line:
                sym = int(ld[-1].replace(',', ''))
            elif '"determinant_file":' in line:
                detfile = ld[-1].replace(',', '').replace('"', '')
            elif not store and 'State     Energy' in line:
                store = True
                if sym is None or detfile is None:
                    raise RuntimeError('Could not find key!')
                key = (sym, detfile)
                data[key] = []
            elif store and len(ld) == 0:
                store = False
                sym = None
                detfile = None
                del key
            elif store:
                i, Ei = ld
                data[key].append(float(Ei))

    for k in data:
        data[k] = np.array(data[k])

    return data


def getdets(output: str) -> list:

    dets = []

    with open(output, 'rt') as stream:
        for line in stream:
            line = line.replace('|', '').replace('>', '')
            occupied = [int(i) for i in line.split()[1:]]
            dets.append(occupied)

    return np.array(dets)


def getwfns(output: str, ndets: int, cache: bool = True) -> tuple[list]:

    idet = []
    psis = []

    wfn_file = output.replace('DET', 'WFN')
    idet_npy = output + '.npy'
    psis_npy = wfn_file + '.npy'

    if os.path.isfile(idet_npy) and os.path.isfile(psis_npy):
        #print(f'Reading cache: {idet_npy}')
        #print(f'Reading cache: {psis_npy}')
        idet = np.load(idet_npy)
        psis = np.load(psis_npy)
    else:
        with open(wfn_file, 'rt') as stream:
            for line in stream:
                ld = line.split()
                idet.append(int(ld[1]))
                psis.append(float(ld[2]))

        assert len(idet) == ndets*ndets
        assert len(psis) == ndets*ndets

        idet = np.array(idet).reshape((ndets, ndets))
        psis = np.array(psis).reshape((ndets, ndets))

        print(f'Saving cache: {idet_npy}')
        print(f'Saving cache: {psis_npy}')

        np.save(idet_npy, idet, allow_pickle=False)
        np.save(psis_npy, psis, allow_pickle=False)

    return idet, psis


def dblock(M1: np.array, M2: np.array, M3: np.array) -> np.array:
    n1 = M1.shape[0]
    n2 = M2.shape[0]
    n3 = M3.shape[0]

    M = np.block([
        [M1, np.zeros((n1, n2))],
        [np.zeros((n2, n1)), M2],
    ])

    M = np.block([
        [ M, np.zeros((n1 + n2, n3))],
        [np.zeros((n3, n1 + n2)), M3],
    ])

    return M


def matreader(mat: str, constructor: callable = None) -> dict:
    mats = {
        'D': [],
        'S': [],
        'C': [],
        'H1E': [],
        'MUX': [],
        'MUY': [],
        'MUZ': [],
    }

    with open(mat, 'rt') as stream:
        for line in stream:
            if 'MATRIX D' in line:
                key = 'D'
            elif 'MATRIX S' in line:
                key = 'S'
            elif 'MATRIX C' in line:
                key = 'C'
            elif 'MATRIX H1E' in line:
                key = 'H1E'
            elif 'MATRIX MUX' in line:
                key = 'MUX'
            elif 'MATRIX MUY' in line:
                key = 'MUY'
            elif 'MATRIX MUZ' in line:
                key = 'MUZ'

            ld = line.split()

            if len(ld) == 0 or 'MATRIX' in line:
                continue

            if 'SYMMETRY BLOCK' in line:
                mats[key].append([])
            else:
                row = [float(v) for v in ld]
                mats[key][-1].append(row)

    if constructor is None:
        for key in mats:
            if key == 'MUX':
                M1, M2 = [np.array(m) for m in mats[key]]
                mats[key] = np.block([
                    [np.zeros((6, 6)), M1, np.zeros((6, 2))],
                    [M2, np.zeros((2, 4))],
                    [np.zeros((2, 10))],
                ])
            elif key == 'MUY':
                M1, M2 = [np.array(m) for m in mats[key]]
                mats[key] = np.block([
                    [np.zeros((6, 8)), M1],
                    [np.zeros((2, 10))],
                    [M2, np.zeros((2, 4))],
                ])
            else:
                M1, M2, M3 = [np.array(m) for m in mats[key]]
                mats[key] = dblock(M1, M2, M3)
    else:
        mats = constructor(mats)

    return mats
