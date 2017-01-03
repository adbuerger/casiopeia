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

class IntegrationTestODE2(unittest.TestCase):

    # Model and data taken and adapted from Verschueren, Robin: Design and
    # implementation of a time-optimal controller for model race cars, 
    # KU Leuven, 2014

    def setUp(self):

        self.x = ca.MX.sym("x", 4)
        self.p = ca.MX.sym("p", 6)
        self.u = ca.MX.sym("u", 2)

        self.f = ca.vertcat( \

            self.x[3] * np.cos(self.x[2] + self.p[0] * self.u[0]),

            self.x[3] * np.sin(self.x[2] + self.p[0] * self.u[0]),

            self.x[3] * self.u[0] * self.p[1],

            self.p[2] * self.u[1] \
                - self.p[3] * self.u[1] * self.x[3] \
                - self.p[4] * self.x[3]**2 \
                - self.p[5] \
                - (self.x[3] * self.u[0])**2 * self.p[1] * self.p[0])

        self.phi = self.x

        data = np.array(np.loadtxt("test/data_2d_vehicle_pe.dat", \
            delimiter = ", ", skiprows = 1))

        self.time_points = data[100:250, 1]

        self.ydata = data[100:250, [2, 4, 6, 8]]
        self.udata = data[100:249, [9, 10]]

        self.pinit =[0.5, 17.06, 12.0, 2.17, 0.1, 0.6]

        self.xinit = self.ydata
 
        self.phat = np.atleast_2d( \
            [0.200652, 11.6528, -26.2501, -74.1967, 16.8705, -1.80125]).T

        self.covariance_matrix = np.loadtxt( \
            "test/covariance_matrix_2d_vehicle_pe.txt", delimiter=",")

        self.time_points_doe = data[200:205, 1]

        self.ydata_doe = data[200:205, [2, 4, 6, 8]]
        
        self.uinit_doe = data[200:205, [9, 10]][:-1, :]

        self.pdata_doe = [0.273408, 11.5602, 2.45652, 7.90959, -0.44353, -0.249098]

        self.umin_doe = [-0.436332, -0.3216]
        self.umax_doe = [0.436332, 1.0]

        self.xmin_doe = [-0.787, -1.531, -12.614, 0.0]
        self.xmax_doe = [1.2390, 0.014, 0.013, 0.7102]

        self.design_results = np.atleast_2d( \
            np.loadtxt("test/optimized_controls_2d_vehicle_doe.txt")).T


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
            self.phat, decimal = 3)

        pe.compute_covariance_matrix()

        assert_array_almost_equal(pe.covariance_matrix, \
            self.covariance_matrix, decimal = 2)


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
            self.phat, decimal = 3)
        
        pe.compute_covariance_matrix()

        assert_array_almost_equal(pe.covariance_matrix, \
            self.covariance_matrix, decimal = 2)


    def test_integration_test_sim(self):

        odesys = casiopeia.system.System(x = self.x, p = self.p, \
            f = self.f, phi = self.phi, u = self.u)

        sim = casiopeia.sim.Simulation(odesys, self.phat)
        sim.run_system_simulation(time_points = self.time_points, \
            x0 = self.ydata[0,:], udata = self.udata)

        simdata = np.array(np.loadtxt("test/data_2d_vehicle_sim.txt")).T
        assert_array_almost_equal(sim.simulation_results, simdata, \
            decimal = 4)


    def test_integration_test_doe_collocation(self):

        odesys = casiopeia.system.System(x = self.x, p = self.p, \
            f = self.f, phi = self.phi, u = self.u)

        doe = casiopeia.doe.DoE(system = odesys, \
            time_points = self.time_points_doe, \
            uinit = self.uinit_doe, pdata = self.pdata_doe, \
            x0 = self.ydata_doe[0,:], \
            umin = self.umin_doe, umax = self.umax_doe, \
            xmin = self.xmin_doe, xmax = self.xmax_doe)

        doe.print_initial_experimental_properties()

        self.assertRaises(AttributeError, \
            doe.print_optimized_experimental_properties)

        # assertRaises only accepts callables, see e. g.:
        # http://stackoverflow.com/questions/1274047/why-isnt-assertraises-
        # catching-my-attribute-error-using-python-unittest

        def no_doe_results_test(doe):

            return doe.design_results

        self.assertRaises(AttributeError, no_doe_results_test, doe)

        doe.run_experimental_design()

        doe.print_optimized_experimental_properties()

        assert_array_almost_equal(doe.optimized_controls, \
            self.design_results, decimal = 4)
