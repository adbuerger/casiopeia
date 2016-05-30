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

        return self.u.size()


    @property
    def nq(self):

        return self.q.size()


    @property
    def np(self):

        return self.p.size()


    @property
    def nx(self):

        return self.x.size()


    @property
    def nz(self):

        return self.z.size()


    @property
    def neps_e(self):

        return self.eps_e.size()


    @property
    def neps_u(self):

        return self.eps_u.size()


    @property
    def nphi(self):

        return self.phi.size()


    def __check_all_system_parts_are_casadi_symbolics(self):

        for arg in self.__init__.__code__.co_varnames[1:]:

                if not isinstance(getattr(self, arg), type(ci.mx_sym("a"))):

                    raise TypeError('''
Missing input argument for system definition or wrong variable type for an
input argument. Input arguments must be CasADi symbolic types.''')


    def __check_no_explicit_time_dependecy(self):

        if ci.depends_on(self.f, self.t):

            raise NotImplementedError('''
Explicit time dependecies of the ODE right hand side are not yet supported in
casiopeia, but will be in future versions.''')


    def __print_nondyn_system_information(self):

        print('''
The system is a non-dynamic systems with the general input-output
structure and equality constraints:

y = phi(t, u, q, p),
g(t, u, q, p) = 0.

Particularly, the system has:
{0} time-varying controls u
{1} time-constant controls q
{2} parameters p
{3} outputs phi'''.format(self.nu, self.nq, self.np, self.nphi))
        
        print("\nwhere phi is defined by ")
        for i, yi in enumerate(self.phi):         
            print("y[{0}] = {1}".format(i, yi))
                 
        print("\nand where g is defined by ")
        for i, gi in enumerate(self.g):              
            print("g[{0}] = {1}".format(i, gi))


    def __print_ode_system_information(self):

        print('''
The system is a dynamic system defined by a set of explicit ODEs xdot
which establish the system state x and by an output function phi which
sets the system measurements:

xdot = f(t, u, q, x, p, eps_e, eps_u),
y = phi(t, u, q, x, p).

Particularly, the system has:
{0} time-varying controls u
{1} time-constant controls q
{2} parameters p
{3} states x
{4} outputs phi'''.format(self.nu, self.nq, self.np, self.nx, self.nphi))

        
        print("\nwhere xdot is defined by ")
        for i, xi in enumerate(self.f):         
            print("xdot[{0}] = {1}".format(i, xi))
                 
        print("\nand where phi is defined by ")
        for i, yi in enumerate(self.phi):              
            print("y[{0}] = {1}".format(i, yi))


    def print_system_information(self):

        if self.nx == 0 and self.nz == 0:

            self.__print_nondyn_system_information()

        elif self.nx != 0 and self.nz == 0:

            self.__print_ode_system_information()


    def __system_validation(self):

        self.__check_all_system_parts_are_casadi_symbolics()
        self.__check_no_explicit_time_dependecy()

        if self.nx != 0 and self.nz != 0:

            raise NotImplementedError('''
Support of implicit DAEs is not implemented yet,
but will be in future versions.
''')

        if self.nx == 0 and self.nz != 0:

            raise NotImplementedError('''
The system definition provided by the user is invalid.
See the documentation for a list of valid definitions.
''')

        self.print_system_information()


    def __init__(self, \
             t = ci.mx_sym("t", 1),
             u = ci.mx_sym("u", 0), \
             q = ci.mx_sym("q", 0), \
             p = None, \
             x = ci.mx_sym("x", 0), \
             z = ci.mx_sym("z", 0),
             eps_e = ci.mx_sym("eps_e", 0), \
             eps_u = ci.mx_sym("eps_u", 0), \
             phi = None, \
             f = ci.mx_sym("f", 0), \
             g = ci.mx_sym("g", 0)):


        r'''
        :raises: TypeError, NotImplementedError

        :param t: time :math:`t \in \mathbb{R}` *(not yet supported)*
        :type t: casadi.casadi.MX

        :param u: time-varying controls :math:`u \in \mathbb{R}^{\text{n}_\text{u}}` that are applied piece-wise-constant for each control intervals, and therefor can change from on interval to another, e. g. motor dutycycles, temperatures, massflows (optional)
        :type u: casadi.casadi.MX

        :param q: time-constant controls :math:`q \in \mathbb{R}^{\text{n}_\text{q}}` that are constant over time, e. g. initial mass concentrations of reactants, elevation angles (optional)
        :type q: casadi.casadi.MX

        :param p: unknown parameters :math:`p \in \mathbb{R}^{\text{n}_\text{p}}`
        :type p: casadi.casadi.MX

        :param x: differential states :math:`x \in \mathbb{R}^{\text{n}_\text{x}}` (optional)
        :type x: casadi.casadi.MX

        :param z: algebraic states :math:`x \in \mathbb{R}^{\text{n}_\text{z}}` (optional)
        :type z: casadi.casadi.MX

        :param eps_e: equation errors :math:`\epsilon_{e} \in \mathbb{R}^{\text{n}_{\epsilon_\text{e}}}` (optional)
        :type eps_e: casadi.casadi.MX

        :param eps_u: input errors :math:`\epsilon_{u} \in \mathbb{R}^{\text{n}_{\epsilon_\text{u}}}` (optional)
        :type eps_u: casadi.casadi.MX

        :param phi: output function :math:`\phi(t, u, q, x, p) = y \in \mathbb{R}^{\text{n}_\text{y}}`
        :type phi: casadi.casadi.MX

        :param f: explicit system of ODEs :math:`f(t, u, q, x, z, p, \epsilon_\text{e}, \epsilon_\text{u}) = \dot{x} \in \mathbb{R}^{\text{n}_\text{x}}` (optional)
        :type f: casadi.casadi.MX

        :param g: equality constraints :math:`g(t, u, q, x, z, p) = 0 \in \mathbb{R}^{\text{n}_\text{g}}` (optional)
                  (optional)
        :type g: casadi.casadi.MX


        Depending on the inputs the user provides, the :class:`System`
        is interpreted as follows:


        **Non-dynamic system** (x = None, z = None):

        .. math::

            y = \phi(t, u, q, p)

            0 = g(t, u, q, p).


        **Explicit ODE system** (x != None, z = None):

        .. math::

            y & = & \phi(t, u, q, x, p) \\

            \dot{x}  & = & f(t, u, q, x, p, \epsilon_\text{e}, \epsilon_\text{u}).


        **Fully implicit DAE system** *(not yet supported)*:

        .. math::

            y & = & \phi(t, u, q, x, p) \\

            0 & = & f(t, u, q, x, \dot{x}, z, p, \epsilon_\text{e}, \epsilon_\text{u}).

            0 = g(t, u, q, x, z, p)

        '''


        intro()
        
        print('\n' + '# ' + 23 * '-' + \
            ' casiopeia system definition ' + 22 * '-' + ' #')
        print('\nStarting system definition ...')

        self.t = t
        self.u = u
        self.q = q
        self.p = p

        self.x = x
        self.z = z

        self.eps_e = eps_e
        self.eps_u = eps_u

        self.phi = phi
        self.f = f
        self.g = g

        self.__system_validation()
