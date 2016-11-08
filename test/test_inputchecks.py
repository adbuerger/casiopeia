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

import numpy as np
from numpy.testing import assert_array_equal

from casiopeia.interfaces import casadi_interface as ci
from casiopeia import inputchecks

import unittest
import mock

class SetSystem(unittest.TestCase):

    def setUp(self):

        self.system = "system"


    def test_set_system(self):

        system = "system"

        # A better test would be needed here also to make sure that system is
        # subclass of systems.SystemBaseClass, but this is yet problematic

        system = inputchecks.set_system(system)
        self.assertEqual(self.system, system)


class CheckTimepointsInput(unittest.TestCase):

    def setUp(self):

        self.tp_ref = np.linspace(0, 49, 50)


    def test_input_list(self):

        tp = [k for k in range(50)]

        tp = inputchecks.check_time_points_input(tp)
        assert_array_equal(tp, self.tp_ref)


    def test_input_onedim_time_vector(self):

        tp = np.linspace(0, 49, 50)

        tp = inputchecks.check_time_points_input(tp)
        assert_array_equal(tp, self.tp_ref)


    def test_input_row_time_vector(self):

        tp = np.atleast_2d(np.linspace(0, 49, 50))

        tp = inputchecks.check_time_points_input(tp)
        assert_array_equal(tp, self.tp_ref)


    def test_input_column_time_vector(self):

        tp = np.atleast_2d(np.linspace(0, 49, 50)).T

        tp = inputchecks.check_time_points_input(tp)
        assert_array_equal(tp, self.tp_ref)


    def test_input_invalid_time_vector(self):

        tp = np.random.randn(2,2)

        self.assertRaises(ValueError, \
            inputchecks.check_time_points_input, tp)


class CheckControlsData(unittest.TestCase):

    def setUp(self):

        self.number_of_controls = 20

    def test_input_rows(self):

        nu = 3
        udata_ref = np.random.rand(nu, self.number_of_controls)

        udata = inputchecks.check_controls_data(udata_ref, \
            nu, self.number_of_controls)
        assert_array_equal(udata, udata_ref)


    def test_input_columns(self):

        nu = 3
        udata_ref = np.random.rand(nu, self.number_of_controls)

        udata = inputchecks.check_controls_data(udata_ref.T, \
            nu, self.number_of_controls)
        assert_array_equal(udata, udata_ref)


    def test_input_none(self):

        nu = 3
        udata_ref = np.zeros((nu, self.number_of_controls))

        udata = inputchecks.check_controls_data(None, \
            nu, self.number_of_controls)
        assert_array_equal(udata, udata_ref)


    def test_zero_controls(self):

        # In this case, the input value is not used by the function, and
        # therefor irrelevant at this point

        nu = 0
        udata_ref = ci.dmatrix(0, self.number_of_controls)

        udata = inputchecks.check_controls_data(None, \
            nu, self.number_of_controls)
        assert_array_equal(udata, udata_ref)


    def test_input_invalid(self):

        nu = 3
        udata_ref = np.zeros((nu + 1, self.number_of_controls))

        self.assertRaises(ValueError, \
            inputchecks.check_controls_data, udata_ref, \
            nu, self.number_of_controls)


class CheckConstantControlsData(unittest.TestCase):

    def setUp(self):

        pass


    def test_input_rows(self):

        nq = 3
        qdata_ref =  np.atleast_2d(np.random.rand(nq, 1))

        qdata = inputchecks.check_constant_controls_data(qdata_ref, nq)
        assert_array_equal(qdata, qdata_ref)


    def test_input_columns(self):

        nq = 3
        qdata_ref =  np.atleast_2d(np.random.rand(nq, 1))

        qdata = inputchecks.check_constant_controls_data(qdata_ref.T, nq)
        assert_array_equal(qdata, qdata_ref)


    def test_zero_controls(self):

        # In this case, the input value is not used by the function, and
        # therefor irrelevant at this point

        nq = 0
        qdata_ref = ci.dmatrix(0, 1)

        qdata = inputchecks.check_constant_controls_data(None, nq)
        assert_array_equal(qdata, qdata_ref)


    def test_input_none(self):

        nq = 3
        qdata_ref = np.atleast_2d(np.zeros((nq, 1)))

        qdata = inputchecks.check_constant_controls_data(None, nq)
        assert_array_equal(qdata, qdata_ref)


    def test_input_invalid_onedim(self):

        nq = 3
        qdata_ref = np.atleast_2d(np.linspace(0, nq - 2, nq - 1))

        self.assertRaises(ValueError, \
            inputchecks.check_constant_controls_data, qdata_ref, nq)


    def test_input_invalid_twodim(self):

        nq = 3
        qdata_ref = np.random.rand(nq, 2)

        self.assertRaises(ValueError, \
            inputchecks.check_constant_controls_data, qdata_ref, nq)


class CheckStatesData(unittest.TestCase):

    def setUp(self):

        self.number_of_intervals = 20


    def test_input_rows(self):

        nx = 4
        xdata_ref = np.random.rand(nx, self.number_of_intervals + 1)

        xdata = inputchecks.check_states_data(xdata_ref, nx, \
            self.number_of_intervals)
        assert_array_equal(xdata, xdata_ref)


    def test_input_columns(self):

        nx = 4
        xdata_ref = np.random.rand(nx, self.number_of_intervals + 1)

        xdata = inputchecks.check_states_data(xdata_ref.T, nx, \
            self.number_of_intervals)
        assert_array_equal(xdata, xdata_ref)


    def test_input_none(self):

        nx = 4
        xdata_ref = np.zeros((nx, self.number_of_intervals + 1))

        xdata = inputchecks.check_states_data(None, nx, \
            self.number_of_intervals)
        assert_array_equal(xdata, xdata_ref)


    def test_zero_states(self):

        nx = 0
        xdata_ref = ci.dmatrix(0, 0)

        # In this case, the input value is not used by the function, and
        # therefor irrelevant at this point

        xdata = inputchecks.check_states_data(None, nx, \
            self.number_of_intervals)
        assert_array_equal(xdata, xdata_ref)


    def test_input_invalid(self):

        nx = 4
        xdata_ref = np.random.rand(nx, self.number_of_intervals)

        self.assertRaises(ValueError, \
            inputchecks.check_states_data, xdata_ref, nx, \
            self.number_of_intervals + 1)


class CheckParameterData(unittest.TestCase):

    def setUp(self):

        pass


    def test_input_rows(self):

        n_p = 3
        pdata_ref = np.atleast_2d(np.linspace(0, n_p - 1, n_p))

        pdata = inputchecks.check_parameter_data(pdata_ref, n_p)
        assert_array_equal(pdata, np.squeeze(pdata_ref))


    def test_input_columns(self):

        n_p = 3
        pdata_ref = np.atleast_2d(np.linspace(0, n_p - 1, n_p))

        pdata = inputchecks.check_parameter_data(pdata_ref.T, n_p)
        assert_array_equal(pdata, np.squeeze(pdata_ref))


    def test_input_none(self):

        n_p = 3
        pdata_ref = np.zeros(n_p)

        pdata = inputchecks.check_parameter_data(None, n_p)
        assert_array_equal(pdata, np.squeeze(pdata_ref))


    def test_input_invalid_onedim(self):

        n_p = 3
        pdata_ref = np.atleast_2d(np.linspace(0, n_p - 2, n_p - 1))

        self.assertRaises(ValueError, \
            inputchecks.check_parameter_data, pdata_ref, n_p)


    def test_input_invalid_twodim(self):

        n_p = 3
        pdata_ref = np.random.rand(n_p, 2)

        self.assertRaises(ValueError, \
            inputchecks.check_parameter_data, pdata_ref, n_p) 


class CheckMeasurementData(unittest.TestCase):

    def setUp(self):

        self.number_of_measurements = 20
        self.nphi = 2


    def test_input_rows(self):

        ydata_ref = np.random.rand(self.nphi, self.number_of_measurements)

        ydata = inputchecks.check_measurement_data(ydata_ref, \
            self.nphi, self.number_of_measurements)
        assert_array_equal(ydata, ydata_ref)


    def test_input_columns(self):

        ydata_ref = np.random.rand(self.nphi, self.number_of_measurements)

        ydata = inputchecks.check_measurement_data(ydata_ref.T, \
            self.nphi, self.number_of_measurements)
        assert_array_equal(ydata, ydata_ref)


    def test_input_none(self):

        ydata_ref = np.zeros((self.nphi, self.number_of_measurements))

        ydata = inputchecks.check_measurement_data(None, \
            self.nphi, self.number_of_measurements)
        assert_array_equal(ydata, ydata_ref)


    def test_input_invalid(self):

        ydata_ref = np.zeros((self.nphi + 1, self.number_of_measurements))

        self.assertRaises(ValueError, \
            inputchecks.check_measurement_data, ydata_ref, self.nphi, \
            self.number_of_measurements)


class CheckMeasurementWeightings(unittest.TestCase):

    def setUp(self):

        self.number_of_measurements = 20
        self.nphi = 2


    def test_input_rows(self):

        wv_ref = np.random.rand(self.nphi, self.number_of_measurements)

        wv = inputchecks.check_measurement_weightings(wv_ref, \
            self.nphi, self.number_of_measurements)
        assert_array_equal(wv, wv_ref)


    def test_input_columns(self):

        wv_ref = np.random.rand(self.nphi, self.number_of_measurements)

        wv = inputchecks.check_measurement_weightings(wv_ref.T, \
            self.nphi, self.number_of_measurements)
        assert_array_equal(wv, wv_ref)


    def test_input_none(self):

        wv_ref = np.ones((self.nphi, self.number_of_measurements))

        wv = inputchecks.check_measurement_weightings(None, \
            self.nphi, self.number_of_measurements)
        assert_array_equal(wv, wv_ref)


    def test_input_invalid(self):

        wv_ref = np.random.rand(self.nphi + 1, self.number_of_measurements)


        self.assertRaises(ValueError, \
            inputchecks.check_measurement_weightings, wv_ref, \
            self.nphi, self.number_of_measurements)


class CheckInputErrorWeightings(unittest.TestCase):

    def setUp(self):

        self.neps_u = 3
        self.number_of_intervals = 20


    def test_input_rows(self):

        weps_u_ref = 2 * np.ones((self.neps_u, self.number_of_intervals))

        weps_u = inputchecks.check_input_error_weightings(weps_u_ref, \
            self.neps_u, self.number_of_intervals)
        assert_array_equal(weps_u, np.squeeze(weps_u_ref))


    def test_input_columns(self):

        weps_u_ref = 2 * np.ones((self.neps_u, self.number_of_intervals))

        weps_u = inputchecks.check_input_error_weightings(weps_u_ref.T, \
            self.neps_u, self.number_of_intervals)
        assert_array_equal(weps_u, np.squeeze(weps_u_ref))


    def test_input_none(self):

        weps_u_ref = np.ones((self.neps_u, self.number_of_intervals))

        weps_u = inputchecks.check_input_error_weightings(None, \
            self.neps_u, self.number_of_intervals)
        assert_array_equal(weps_u, weps_u_ref)


    def test_zero_equation_errors(self):

        neps_u = 0

        weps_u_ref = ci.dmatrix(0, 0)

        # In this case, the input value is not used by the function, and
        # therefor irrelevant at this point

        weps_u = inputchecks.check_input_error_weightings(None, \
            neps_u, self.number_of_intervals)
        assert_array_equal(weps_u, weps_u_ref)


    def test_input_invalid_onedim(self):

        weps_u_ref = \
            np.atleast_2d(np.linspace(0, self.neps_u - 2, self.neps_u - 1))

        self.assertRaises(ValueError, \
            inputchecks.check_input_error_weightings, weps_u_ref, \
            self.neps_u, self.number_of_intervals)


    def test_input_invalid_twodim(self):

        weps_u_ref = np.random.rand(self.neps_u, 2)

        self.assertRaises(ValueError, \
            inputchecks.check_input_error_weightings, weps_u_ref, \
            self.neps_u, self.number_of_intervals)


class CheckMultiDoeInput(unittest.TestCase):

    def setUp(self):

        pass


    def test_input_not_a_list(self):

        self.assertRaises(TypeError, \
            inputchecks.check_multi_doe_input, \
            "NotAList")


    def test_input_list_of_lenght_one(self):

        self.assertRaises(ValueError, \
            inputchecks.check_multi_doe_input, \
            ["ListWithOnlyOneEntry"])


    def test_input_list_entries_not_of_type_doe(self):

        self.assertRaises(TypeError, \
            inputchecks.check_multi_doe_input, \
            ["FirstListEntry", "SecondListEntry"])


    def test_input_list_entries_of_type_doe(self):

        doe_object = mock.MagicMock()
        type(doe_object).__name__ = "DoE"

        inputchecks.check_multi_doe_input([doe_object, doe_object])


class CheckMultiLsqInput(unittest.TestCase):

    def setUp(self):

        pass


    def test_input_not_a_list(self):

        self.assertRaises(TypeError, \
            inputchecks.check_multi_lsq_input, \
            "NotAList")


    def test_input_list_of_lenght_one(self):

        self.assertRaises(ValueError, \
            inputchecks.check_multi_lsq_input, \
            ["ListWithOnlyOneEntry"])


    def test_input_list_entries_not_of_type_lsq(self):

        self.assertRaises(TypeError, \
            inputchecks.check_multi_lsq_input, \
            ["FirstListEntry", "SecondListEntry"])


    def test_input_list_entries_of_type_lsw(self):

        doe_object = mock.MagicMock()
        type(doe_object).__name__ = "LSq"

        inputchecks.check_multi_lsq_input([doe_object, doe_object])
