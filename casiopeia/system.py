#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This file is part of casiopeia.
#
# Copyright 2014-2016 Adrian Bürger, Moritz Diehl
#
# casiopeia is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# casiopeia is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with casiopeia. If not, see <http://www.gnu.org/licenses/>.

from interfaces import casadi_interface as ci
from intro import intro

class System:

    '''The class :class:`System` is used to define non-dynamic, explicit ODE-
    or fully implicit DAE-systems systems within casiopeia.'''

    @property
    def nu(self):

        return self.u.numel()


    @property
    def nq(self):

        return self.q.numel()


    @property
    def np(self):

        return self.p.numel()


    @property
    def nx(self):

        return self.x.numel()


    @property
    def neps_u(self):

        return self.eps_u.numel()


    @property
    def nphi(self):

        return self.phi.numel()


    def __check_all_system_parts_are_casadi_symbolics(self):

        for arg in self.__init__.__code__.co_varnames[1:]:

                if not isinstance(getattr(self, arg), type(ci.mx_sym("a"))):

                    raise TypeError('''
Missing input argument for system definition or wrong variable type for an
input argument. Input arguments must be CasADi symbolic types.''')


    def __print_nondyn_system_information(self):

        print('''
The system is a non-dynamic systems with the general input-output
structure and equality constraints:

y = phi(u, q, p),
g(u, q, p) = 0.

Particularly, the system has:
{0} time-varying controls u
{1} time-constant controls q
{2} parameters p
{3} outputs phi'''.format(self.nu, self.nq, self.np, self.nphi))
        
        print("\nwhere phi is defined by ")
        for i in range(self.phi.numel()):              
            print("y[{0}] = {1}".format(i, self.phi[i]))
                 
        print("\nand where g is defined by ")
        for i in range(self.g.numel()):              
            print("g[{0}] = {1}".format(i, self.g[i]))


    def __print_ode_system_information(self):

        print('''
The system is a dynamic system defined by a set of explicit ODEs xdot
which establish the system state x and by an output function phi which
sets the system measurements:

xdot = f(u, q, x, p, eps_e, eps_u),
y = phi(u, q, x, p).

Particularly, the system has:
{0} time-varying controls u
{1} time-constant controls q
{2} parameters p
{3} states x
{4} outputs phi
{5} input errors eps_u'''.format(self.nu, self.nq, self.np, self.nx, \
    self.nphi, self.neps_u))

        
        print("\nwhere xdot is defined by ")
        for i in range(self.f.numel()):         
            print("xdot[{0}] = {1}".format(i, self.f[i]))

        print("\nand where phi is defined by ")
        for i in range(self.phi.numel()):              
            print("y[{0}] = {1}".format(i, self.phi[i]))

    def print_system_information(self):

        if self.nx == 0:

            self.__print_nondyn_system_information()

        elif self.nx != 0:

            self.__print_ode_system_information()


    def __system_validation(self):

        self.__check_all_system_parts_are_casadi_symbolics()
        self.print_system_information()


    def __init__(self, \
             u = ci.mx_sym("u", 0), \
             q = ci.mx_sym("q", 0), \
             p = None, \
             x = ci.mx_sym("x", 0), \
             eps_u = ci.mx_sym("eps_u", 0), \
             phi = None, \
             f = ci.mx_sym("f", 0), \
             g = ci.mx_sym("g", 0)):


        r'''
        :raises: TypeError, NotImplementedError

        :param u: time-varying controls :math:`u \in \mathbb{R}^{\text{n}_\text{u}}` that are applied piece-wise-constant for each control intervals, and therefor can change from on interval to another, e. g. motor dutycycles, temperatures, massflows (optional)
        :type u: casadi.casadi.MX

        :param q: time-constant controls :math:`q \in \mathbb{R}^{\text{n}_\text{q}}` that are constant over time, e. g. initial mass concentrations of reactants, elevation angles (optional)
        :type q: casadi.casadi.MX

        :param p: unknown parameters :math:`p \in \mathbb{R}^{\text{n}_\text{p}}`
        :type p: casadi.casadi.MX

        :param x: differential states :math:`x \in \mathbb{R}^{\text{n}_\text{x}}` (optional)
        :type x: casadi.casadi.MX

        :param eps_u: input errors :math:`\epsilon_{u} \in \mathbb{R}^{\text{n}_{\epsilon_\text{u}}}` (optional)
        :type eps_u: casadi.casadi.MX

        :param phi: output function :math:`\phi(u, q, x, p) = y \in \mathbb{R}^{\text{n}_\text{y}}`
        :type phi: casadi.casadi.MX

        :param f: explicit system of ODEs :math:`f(u, q, x, p, \epsilon_\text{u}) = \dot{x} \in \mathbb{R}^{\text{n}_\text{x}}` (optional)
        :type f: casadi.casadi.MX

        :param g: equality constraints :math:`g(u, q, p) = 0 \in \mathbb{R}^{\text{n}_\text{g}}` (optional)
        :type g: casadi.casadi.MX


        Depending on the inputs the user provides, the :class:`System`
        is interpreted as follows:


        **Non-dynamic system** (x = None):

        .. math::

            y = \phi(u, q, p)

            0 = g(u, q, p).


        **Explicit ODE system** (x != None):

        .. math::

            y & = & \phi(u, q, x, p) \\

            \dot{x}  & = & f(u, q, x, p, \epsilon_\text{u}).


        '''


        intro()
        
        print('\n' + '# ' + 23 * '-' + \
            ' casiopeia system definition ' + 22 * '-' + ' #')
        print('\nStarting system definition ...')

        self.u = u
        self.q = q
        self.p = p

        self.x = x

        self.eps_u = eps_u

        self.phi = phi
        self.f = f
        self.g = g

        self.__system_validation()
