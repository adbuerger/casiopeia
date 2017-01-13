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

import casadi as ca
import numpy as np
import casiopeia

from numpy.testing import assert_array_almost_equal

import unittest

class IntegrationTestODE1(unittest.TestCase):

    # Model and data taken from Bock, Sager et al.: Uebungen Numerische
    # Mathematik II, Blatt 9, IWR, Universitaet Heidelberg, 2006

    # ODE, no controls

    def setUp(self):

        self.x = ca.MX.sym("x", 2)
        self.p = ca.MX.sym("p", 2)

        self.f = ca.vertcat( \
            -1.0 * self.x[0] + self.p[0] * self.x[0] * self.x[1], 
            1.0 * self.x[1] - self.p[1] * self.x[0] * self.x[1])

        self.phi = self.x

        data = np.array(np.loadtxt("test/data_lotka_volterra_pe.txt"))

        self.time_points = data[:, 0]

        self.ydata = data[:, 1::2]
        self.wv = 1.0 / data[:, 2::2]**2

        self.xinit = self.ydata

        self.phat_collocation = np.atleast_2d([0.693379029, 0.341128482]).T
        self.phat_multiple_shooting = np.atleast_2d([0.693971, 0.340921]).T

        self.covariance_matrix_collocation = np.array( \
            [[0.000628377, 8.09817e-06],
            [8.09817e-06, 1.45677e-05]])

        self.covariance_matrix_multiple_shooting = np.array( \
        [[0.000631059, 8.10761e-06], 
         [8.10761e-06, 1.44984e-05]])


    def test_integration_test_pe_collocation(self):

        odesys = casiopeia.system.System(x = self.x, p = self.p, \
            f = self.f, phi = self.phi)

        pe = casiopeia.pe.LSq(system = odesys, time_points = self.time_points, \
            xinit = self.xinit, ydata = self.ydata, wv = self.wv)

        self.assertRaises(AttributeError, pe.print_estimation_results)
        pe.run_parameter_estimation()
        pe.print_estimation_results()

        assert_array_almost_equal(pe.estimated_parameters, \
            self.phat_collocation, decimal = 8)

        pe.compute_covariance_matrix()

        assert_array_almost_equal(pe.covariance_matrix, \
            self.covariance_matrix_collocation, decimal = 8)


    def test_integration_test_pe_multiple_shooting(self):

        odesys = casiopeia.system.System(x = self.x, p = self.p, \
            f = self.f, phi = self.phi)

        pe = casiopeia.pe.LSq(system = odesys, time_points = self.time_points, \
            xinit = self.xinit, ydata = self.ydata, wv = self.wv, \
            discretization_method = "multiple_shooting")

        self.assertRaises(AttributeError, pe.print_estimation_results)
        pe.run_parameter_estimation()
        pe.print_estimation_results()

        assert_array_almost_equal(pe.estimated_parameters, \
            self.phat_multiple_shooting, decimal = 5)

        pe.compute_covariance_matrix()

        assert_array_almost_equal(pe.covariance_matrix, \
            self.covariance_matrix_multiple_shooting, decimal = 8)


    def test_integration_test_sim(self):

        odesys = casiopeia.system.System(x = self.x, p = self.p, \
            f = self.f, phi = self.phi)
        time_points_sim = np.linspace(0, 10, 101)

        sim = casiopeia.sim.Simulation(odesys, self.phat_collocation)
        sim.run_system_simulation(time_points = time_points_sim, \
            x0 = self.ydata[0,:])

        simdata = np.array(np.loadtxt("test/data_lotka_volterra_sim.txt")).T
        assert_array_almost_equal(sim.simulation_results, simdata, decimal = 3)
