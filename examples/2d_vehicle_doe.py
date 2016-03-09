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
# but WITHOUT ANY WARRANTY; without even the implied warrantime_points of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with casiopeia. If not, see <http://www.gnu.org/licenses/>.

# Model and data taken from: Verschueren, Robin: Design and implementation of a 
# time-optimal controller for model race cars, Master’s thesis, KU Leuven, 2014.

import casadi as ca
import pylab as pl
import casiopeia as cp

# System

x = ca.MX.sym("x", 4)
p = ca.MX.sym("p", 6)
u = ca.MX.sym("u", 2)

f = ca.vertcat( \

    [x[3] * pl.cos(x[2] + p[0] * u[0]),

    x[3] * pl.sin(x[2] + p[0] * u[0]),

    x[3] * u[0] * p[1],

    p[2] * u[1] \
        - p[3] * u[1] * x[3] \
        - p[4] * x[3]**2 \
        - p[5] \
        - (x[3] * u[0])**2 * p[1]* p[0]])

phi = x

system = cp.system.System(x = x, u = u, p = p, f = f, phi = phi)

data = pl.array(pl.loadtxt("data_2d_vehicle.dat", \
    delimiter = ", ", skiprows = 1))

time_points = data[200:400:5, 1]

ydata = data[200:400:5, [2, 4, 6, 8]]

uinit = data[200:400:5, [9, 10]][:-1, :]

pdata = [0.273408, 11.5602, 2.45652, 7.90959, -0.44353, -0.249098]

umin = [-0.436332, -0.3216]
umax = [0.436332, 1.0]

xmin = [-0.787, -1.531, -12.614, 0.0]
xmax = [1.2390, 0.014, 0.013, 0.7102]

doe = cp.doe.DoE(system = system, time_points = time_points, \
    uinit = uinit, pdata = pdata, x0 = ydata[0,:], \
    umin = umin, umax = umax, \
    xmin = xmin, xmax = xmax)

doe.run_experimental_design(solver_options = {"linear_solver": "ma86"})

# pl.savetxt("results_2d_vehicle_doe_coll.txt", doe.design_results["x"])
