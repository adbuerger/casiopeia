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
                "Time-varying control values provided by user have wrong dimension.")

        return udata

    else:

        return ci.dmatrix(0, number_of_controls)


def check_constant_controls_data(qdata, nq):

    if not nq == 0:

        if qdata is None:
            qdata = np.zeros((nq, 1))

        qdata = np.atleast_2d(qdata)

        if qdata.shape == (1, nq):
            qdata = qdata.T

        if not qdata.shape == (nq, 1):

            raise ValueError( \
                "Time-constant control values provided by user have wrong dimension.")

        return qdata

    else:

        return ci.dmatrix(0, 1)


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


def check_input_error_weightings(weps_u, neps_u, number_of_intervals):

    if not neps_u == 0:

        if weps_u is None:
            weps_u = np.ones((neps_u, number_of_intervals))

        weps_u = np.atleast_2d(weps_u)

        if weps_u.shape == (number_of_intervals,neps_u):
            weps_u = weps_u.T

        if not weps_u.shape == (neps_u, number_of_intervals):

            raise ValueError( \
                "Input error weightings provided by user have wrong dimension.")

        return weps_u

    else:

        return ci.dmatrix(0, 0)


def check_multi_doe_input(doe_setups):

    if not type(doe_setups) is list:

        raise TypeError('''
Input for MultiDoE setup has to be a list of objects of type DoE.
''')

    if len(doe_setups) <= 1:

            raise ValueError('''
You must instatiate the multi design method passing at least two
experimental design problems (of type DoE).''')

    for doe_setup in doe_setups:

        # Check for name, check for type not doable here since
        # modules depend on each other

        if not type(doe_setup).__name__ is "DoE":

            raise TypeError('''
Input for MultiDoE setup has to be a list of objects of type DoE.
''')


def check_multi_lsq_input(pe_setups):

    if not type(pe_setups) is list:

        raise TypeError('''
Input for MultiLSq setup has to be a list of objects of type LSq.
''')

    if len(pe_setups) <= 1:

            raise ValueError('''
You must instatiate the multi experiment method passing at least two
parameter estimation problems (of type LSq).''')

    for pe_setup in pe_setups:

        # Check for name, check for type not doable here since
        # modules depend on each other

        if not type(pe_setup).__name__ is "LSq":

            raise TypeError('''
Input for MultiLSq setup has to be a list of objects of type LSq.
''')
