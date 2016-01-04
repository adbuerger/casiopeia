#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2014-2016 Adrian Bürger
#
# This file is part of casiopeia.
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

import numpy as np

from interfaces import casadi_interface as ci

def set_system(system):

    return system


def check_time_points_input(tp):

    if np.atleast_2d(tp).shape[0] == 1:

        tp = np.squeeze(np.asarray(tp))

    elif np.atleast_2d(tp).shape[1] == 1:

        tp = np.squeeze(np.atleast_2d(tp).T)

    else:

        raise ValueError("Invalid dimension for tp.")

    return tp   
    

def check_controls_data(udata, nu, number_of_controls):

    if not nu == 0:

        if udata is None:
            udata = np.zeros((nu, number_of_controls))

        udata = np.atleast_2d(udata)

        if udata.shape == (number_of_controls, nu):
            udata = udata.T

        if not udata.shape == (nu, number_of_controls):

            raise ValueError( \
                "Control values provided by user have wrong dimension.")

        return udata

    else:

        return ci.dmatrix(0, number_of_controls)


def check_states_data(xdata, nx, number_of_intervals):

    if not nx == 0:

        if xdata is None:
            xdata = np.zeros((nx, number_of_intervals + 1))

        xdata = np.atleast_2d(xdata)

        if xdata.shape == (number_of_intervals + 1, nx):
            xdata = xdata.T

        if not xdata.shape == (nx, number_of_intervals + 1):

            raise ValueError( \
                "State values provided by user have wrong dimension.")

        return xdata

    else:

        return ci.dmatrix(0,0)


def check_parameter_data(pdata, n_p):

    # Using "np" will overwrite the import of numpy!

    if pdata is None:
        pdata = np.zeros(n_p)

    pdata = np.atleast_1d(np.squeeze(pdata))

    if not pdata.shape == (n_p,):

        raise ValueError( \
            "Parameter values provided by user have wrong dimension.")

    return pdata


def check_measurement_data(ydata, nphi, number_of_measurements):

    if ydata is None:
        ydata = np.zeros((nphi, number_of_measurements))

    ydata = np.atleast_2d(ydata)

    if ydata.shape == (number_of_measurements, nphi):
        ydata = ydata.T

    if not ydata.shape == (nphi, number_of_measurements):

        raise ValueError( \
            "Measurement data provided by user has wrong dimension.")

    return ydata


def check_measurement_weightings(wv, nphi, number_of_measurements):

    if wv is None:
        wv = np.ones((nphi, number_of_measurements))

    wv = np.atleast_2d(wv)

    if wv.shape == (number_of_measurements, nphi):
        wv = wv.T

    if not wv.shape == (nphi, number_of_measurements):

        raise ValueError( \
            "Measurement weightings provided by user have wrong dimension.")

    return wv


def check_equation_error_weightings(weps_e, neps_e):

    if not neps_e == 0:

        if weps_e is None:
            weps_e = np.ones(neps_e)

        weps_e = np.atleast_1d(np.squeeze(weps_e))

        if not weps_e.shape == (neps_e,):

            raise ValueError( \
                "Equation error weightings provided by user have wrong dimension.")

        return weps_e

    else:

        return ci.dmatrix(0, 0)


def check_input_error_weightings(weps_u, neps_u):

    if not neps_u == 0:

        if weps_u is None:
            weps_u = np.ones(neps_u)

        weps_u = np.atleast_1d(np.squeeze(weps_u))

        if not weps_u.shape == (neps_u,):

            raise ValueError( \
                "Input error weightings provided by user have wrong dimension.")

        return weps_u

    else:

        return ci.dmatrix(0, 0)
