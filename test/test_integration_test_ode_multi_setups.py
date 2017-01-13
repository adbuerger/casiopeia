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

class IntegrationTestODEMultiSetups(unittest.TestCase):

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
            "test/covariance_matrix_2d_vehicle_multi_pe.txt", delimiter=",")


    def test_integration_test_multi_pe(self):

        odesys = casiopeia.system.System(x = self.x, p = self.p, \
            f = self.f, phi = self.phi, u = self.u)

        pe_setups = []

        for k in range(2):

            pe_setups.append(casiopeia.pe.LSq(system = odesys, time_points = self.time_points, \
                xinit = self.xinit, ydata = self.ydata, pinit = self.pinit, \
                udata = self.udata))

        mpe = casiopeia.pe.MultiLSq(pe_setups)
        
        self.assertRaises(AttributeError, mpe.print_estimation_results)
        mpe.run_parameter_estimation()
        mpe.print_estimation_results()

        assert_array_almost_equal(mpe.estimated_parameters, \
            self.phat, decimal = 3)

        mpe.compute_covariance_matrix()

        assert_array_almost_equal(mpe.covariance_matrix, \
            self.covariance_matrix, decimal = 2)
