#!/usr/bin/env python

import numpy as np
from integrals_readin import integral_system as init_readin
from utilities import sc0
from utilities import concate_bitarrays_to_label as gen_det_label
from excitations import random_bitarry_symspace as gen_uniform_det

class density_matrix:
    repstr  = ' {:>.6f}  {:> .12E}  {:> .12E}  {:> .12E}  {:> .12E}'
    repstr += '  {:>8}  {:>8}  {:>6}'
    def __init__(
                self,
                initial_particles = None,
                system = None,
            ):

        if any(param is None for param in (initial_particles, system)):
            error  = ' Please specify a valid system class object\n'
            error += ' and initial number of particles.'
            raise ValueError(error)

        self.main = {}
        self.spawns = {}
        self.system = system

        for _ in range(initial_particles):
            ba = gen_uniform_det(self.system)
            label = gen_det_label(ba,ba)
            if label in self.main:
                self.main[label].update(1.0)
            else:
                hij = sc0(ba,self.system)
                T = -(2*hij - 2*self.system.Href)
                self.create_determinant(label,[ba,ba,T,hij,1.0])

    def create_determinant(self,label,data):
        self.main[label] = determinant(
                                        ba1 = data[0],
                                        ba2 = data[1],
                                        T = data[2],
                                        H = data[3],
                                        nw = data[4],
                                    )

    def store_spawns(self,ba1,ba2,nw,nex,hij):
        label = gen_det_label(ba1,ba2)
        if label in self.spawns:
            self.spawns[label][-1] += nw
        else:
            if nex == 0:
                T = -(2*hij - 2*self.system.Href)
            else:
                hii = sc0(ba1,self.system)
                hjj = sc0(ba2,self.system)
                T = -(hii + hjj - 2*self.system.Href)
            self.spawns[label] = [ba1,ba2,T,hij,nw]

    def merge_main_and_spawns(self):
        for label in self.spawns:
            if label in self.main:
                self.main[label].update(self.spawns[label][-1])
            else:
                self.create_determinant(label,self.spawns[label])
        self.spawns = {}

    def update_estimates(self,beta):
        self.numerator = 0.0
        self.denominator = 0.0
        self.particles = 0.0
        for det in self.main.values():
            self.numerator += det.proj
            self.denominator += det.tr
            self.particles += abs(det.nw)
        self.energy = self.numerator / self.denominator
        print(self.repstr.format(beta,0.0,self.numerator,self.denominator,
                                    self.particles,0,0,0))

class determinant:
    def __init__(
                self,
                ba1 = None,
                ba2 = None,
                T = None,
                H = None,
                nw = None,
            ):
        if any(param is None for param in (ba1,ba2,T,H,nw)):
            error = ' Not all the associated values are provided'
            raise ValueError(error)

        self.ba1 = ba1
        self.ba2 = ba2
        self.T = T
        self.H = H
        self.nw = nw
        self.proj = nw * H
        self.istr = int(np.array_equal(ba1,ba2))
        self.tr = self.istr*nw

    def update(self,dnw):
        self.nw += dnw
        self.proj += dnw * self.H
        self.tr += self.istr*dnw

