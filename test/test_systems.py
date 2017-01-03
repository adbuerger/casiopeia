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

        self.u = ci.mx_sym("u", 2)
        self.q = ci.mx_sym("q", 11)
        self.p = ci.mx_sym("p", 3)
        self.x = ci.mx_sym("x", 4)
        self.eps_u = ci.mx_sym("eps_u", 7)
        self.phi = ci.mx_sym("phi", 8)
        self.f = ci.mx_sym("f", 9)
        self.g = ci.mx_sym("g", 10)


    def test_init_p_phi(self):

        sys = casiopeia.system.System(p = self.p, phi = self.phi)
        sys.print_system_information()


    def test_all_nondynamic_inputs(self):

        sys = casiopeia.system.System(u = self.u, p = self.p, \
            phi = self.phi, g = self.g)

    def test_all_ode_inputs(self):

        sys = casiopeia.system.System(u = self.u, q = self.q, \
            p = self.p, x = self.x, eps_u = self.eps_u, \
            phi = self.phi, f = self.f)
        sys.print_system_information()


    def test_init_not_args(self):

        self.assertRaises(TypeError,  casiopeia.system.System)


    def test_init_not_p(self):

        self.assertRaises(TypeError,  casiopeia.system.System, phi = self.phi)


    def test_init_not_phi(self):

        self.assertRaises(TypeError,  casiopeia.system.System, p = self.p)


    def test_sizes_attributes(self):

        sys = casiopeia.system.System(u = self.u, q = self.q, \
            p = self.p, x = self.x, eps_u = self.eps_u, \
            phi = self.phi, f = self.f, g = self.g)
        
        self.assertEqual(sys.nu, self.u.numel())
        self.assertEqual(sys.nq, self.q.numel())
        self.assertEqual(sys.np, self.p.numel())
        self.assertEqual(sys.nx, self.x.numel())
        self.assertEqual(sys.neps_u, self.eps_u.numel())
        self.assertEqual(sys.nphi, self.phi.numel())
