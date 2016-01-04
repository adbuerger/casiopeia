#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2014-2015 Adrian Bürger
#
# This file is part of PECas.
#
# PECas is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PECas is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with PECas. If not, see <http://www.gnu.org/licenses/>.

import casadi as ca
import numpy as np
import casiopeia

from numpy.testing import assert_array_almost_equal

import unittest

class IntegrationTestODE2(unittest.TestCase):

#     # Model and data taken and adapted from Verschueren, Robin: Design and
#     # implementation of a time-optimal controller for model race cars, 
#     # KU Leuven, 2014

    def setUp(self):

        self.x = ca.MX.sym("x", 4)
        self.p = ca.MX.sym("p", 6)
        self.u = ca.MX.sym("u", 2)

        self.f = ca.vertcat( \

            [self.x[3] * np.cos(self.x[2] + self.p[0] * self.u[0]),

            self.x[3] * np.sin(self.x[2] + self.p[0] * self.u[0]),

            self.x[3] * self.u[0] * self.p[1],

            self.p[2] * self.u[1] \
                - self.p[3] * self.u[1] * self.x[3] \
                - self.p[4] * self.x[3]**2 \
                - self.p[5] \
                - (self.x[3] * self.u[0])**2 * self.p[1] * self.p[0]])

        self.phi = self.x

        data = np.array(np.loadtxt("test/data_2d_vehicle_pe.dat", \
            delimiter = ", ", skiprows = 1))

        self.time_points = data[100:250, 1]

        self.ydata = data[100:250, [2, 4, 6, 8]]
        self.udata = data[100:249, [9, 10]]

        self.pinit =[0.5, 17.06, 12.0, 2.17, 0.1, 0.6]

        self.xinit = self.ydata
 
        self.phat_collocation = np.atleast_2d( \
            [0.200652, 11.6528, -26.2501, -74.1967, 16.8705, -1.80125]).T
        self.phat_multiple_shooting = np.atleast_2d( \
            [0.200652, 11.6528, -26.2502, -74.1968, 16.8705, -1.80125]).T


    def test_integration_test_pe_collocation(self):

        odesys = casiopeia.system.System(x = self.x, p = self.p, \
            f = self.f, phi = self.phi, u = self.u)

        pe = casiopeia.pe.LSq(system = odesys, time_points = self.time_points, \
            xinit = self.xinit, ydata = self.ydata, pinit = self.pinit, \
            udata = self.udata)

        self.assertRaises(AttributeError, pe.print_estimation_results)
        pe.run_parameter_estimation()
        pe.print_estimation_results()

        assert_array_almost_equal(pe.estimated_parameters, \
            self.phat_collocation, decimal = 4)


    def test_integration_test_pe_multiple_shooting(self):

        odesys = casiopeia.system.System(x = self.x, p = self.p, \
            f = self.f, phi = self.phi, u = self.u)

        pe = casiopeia.pe.LSq(system = odesys, time_points = self.time_points, \
            xinit = self.xinit, ydata = self.ydata, pinit = self.pinit, \
            udata = self.udata, discretization_method = "multiple_shooting")

        self.assertRaises(AttributeError, pe.print_estimation_results)
        pe.run_parameter_estimation()
        pe.print_estimation_results()

        assert_array_almost_equal(pe.estimated_parameters, \
            self.phat_multiple_shooting, decimal = 4)


    def test_integration_test_sim(self):

        odesys = casiopeia.system.System(x = self.x, p = self.p, \
            f = self.f, phi = self.phi, u = self.u)

        sim = casiopeia.sim.Simulation(odesys, self.phat_collocation)
        sim.run_system_simulation(time_points = self.time_points, \
            x0 = self.ydata[0,:], udata = self.udata)

        simdata = np.array(np.loadtxt("test/data_2d_vehicle_sim.txt")).T
        assert_array_almost_equal(sim.simulation_results, simdata, \
            decimal = 4)
