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
# but WITHOUT ANY WARRANTY; without even the implied warrantime_points of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with casiopeia. If not, see <http://www.gnu.org/licenses/>.

import casiopeia.interfaces.casadi_interface as ci
import casiopeia

import unittest

class System(unittest.TestCase):

    def setUp(self):

        self.t = ci.mx_sym("t", 1)
        self.u = ci.mx_sym("u", 2)
        self.q = ci.mx_sym("q", 11)
        self.p = ci.mx_sym("p", 3)
        self.x = ci.mx_sym("x", 4)
        self.z = ci.mx_sym("z", 5)
        self.eps_e = ci.mx_sym("eps_e", 6)
        self.eps_u = ci.mx_sym("eps_u", 7)
        self.phi = ci.mx_sym("phi", 8)
        self.f = ci.mx_sym("f", 9)
        self.g = ci.mx_sym("g", 10)


    def test_init_p_phi(self):

        sys = casiopeia.system.System(p = self.p, phi = self.phi)
        sys.print_system_information()


    def test_all_nondynamic_inputs(self):

        sys = casiopeia.system.System(t = self.t, u = self.u, p = self.p, \
            phi = self.phi, g = self.g)

    def test_all_ode_inputs(self):

        sys = casiopeia.system.System(t = self.t, u = self.u, q = self.q, \
            p = self.p, x = self.x, eps_e = self.eps_e, eps_u = self.eps_u, \
            phi = self.phi, f = self.f)
        sys.print_system_information()


    def test_init_not_args(self):

        self.assertRaises(TypeError,  casiopeia.system.System)


    def test_init_not_p(self):

        self.assertRaises(TypeError,  casiopeia.system.System, phi = self.phi)


    def test_init_not_phi(self):

        self.assertRaises(TypeError,  casiopeia.system.System, p = self.p)


    def test_assure_no_explicit_time_dependecy(self):

        # Assure as long as explicit time dependecy is not allowed

        self.assertRaises(NotImplementedError, casiopeia.system.System, \
            t = self.t, u = self.u, x = self.x, \
            p = self.p, eps_e = self.eps_e, phi = self.phi, f = self.t)


    def test_assure_not_dae(self):

        self.assertRaises(NotImplementedError, casiopeia.system.System, \
            p = self.p, phi = self.phi, x = self.x, z = self.z)


    def test_assure_not_algebraic_states_without_dynamic_states(self):

        self.assertRaises(NotImplementedError, casiopeia.system.System, \
            p = self.p, phi = self.phi, z = self.z)
    

    def test_sizes_attributes(self):

        sys = casiopeia.system.System(t = self.t, u = self.u, q = self.q, \
            p = self.p, x = self.x, eps_e = self.eps_e, eps_u = self.eps_u, \
            phi = self.phi, f = self.f, g = self.g)
        
        self.assertEqual(sys.nu, self.u.size())
        self.assertEqual(sys.nq, self.q.size())
        self.assertEqual(sys.np, self.p.size())
        self.assertEqual(sys.nx, self.x.size())
        self.assertEqual(sys.nz, 0)
        self.assertEqual(sys.neps_e, self.eps_e.size())
        self.assertEqual(sys.neps_u, self.eps_u.size())
        self.assertEqual(sys.nphi, self.phi.size())
