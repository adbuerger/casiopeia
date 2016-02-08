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

# Model and data taken from: Diehl, Moritz: Course on System Identification, 
# exercise 7, SYSCOP, IMTEK, University of Freiburg, 2014/2015

import casadi as ca
import pylab as pl
import casiopeia as cp

# Defining constant problem parameters: 
#
#     - m: representing the ball of the mass in kg
#     - L: the length of the pendulum bar in meters
#     - g: the gravity constant in m/s^2
#     - psi: the actime_pointsation angle of the manuver in radians, which stays
#            constant for this problem

m = 1.0
L = 3.0
g = 9.81
psi = pl.pi / 2.0

# System

x = ca.MX.sym("x", 2)
p = ca.MX.sym("p", 1)
u = ca.MX.sym("u", 1)

f = ca.vertcat([x[1], p[0]/(m*(L**2))*(u-x[0]) - g/L * pl.sin(x[0])])

phi = x

system = cp.system.System(x = x, u = u, p = p, f = f, phi = phi)

# Loading data

data = pl.loadtxt('data_pendulum.txt')
time_points = data[:500, 0]
numeas = data[:500, 1]
wmeas = data[:500, 2]
N = time_points.size
ydata = pl.array([numeas,wmeas])
udata = [psi] * (N-1)

# Definition of the weightings for each of the measurements.

# Since no data is provided beforehand, it is assumed that both, angular
# speed and rotational angle have i.i.d. noise, thus the measurement have
# identic standard deviations that can be calculated.

sigmanu = pl.std(numeas, ddof=1)
sigmaw = pl.std(wmeas, ddof=1)

# The weightings for the measurements errors given to casiopeia are calculated
# from the standard deviations of the measurements, so that the least squares
# estimator ist the maximum likelihood estimator for the estimation problem.

wnu = 1.0 / (pl.ones(time_points.size)*sigmanu**2)
ww = 1.0 / (pl.ones(time_points.size)*sigmaw**2)

wv = pl.array([wnu, ww])

# Run parameter estimation and assure that the results is correct

pe = cp.pe.LSq( \
    system = system, time_points = time_points, \
    x0 = ydata[:,0], \
    udata = udata, \
    pinit = 1, \
    xinit = ydata, 
    ydata = ydata, wv = wv, \
    discretization_method = "collocation")

pe.run_parameter_estimation({"linear_solver": "ma27"})
pe.print_estimation_results()

pe.compute_covariance_matrix()
pe.print_estimation_results()

sim = cp.sim.Simulation(system, pe.estimated_parameters)
sim.run_system_simulation(time_points = time_points, \
    x0 = ydata[:,0], udata = udata)

nusim = sim.simulation_results[0,:].T
wsim = sim.simulation_results[1,:].T

pl.close("all")

pl.figure()
pl.subplot2grid((2, 2), (0, 0))
pl.scatter(time_points[::2], numeas[::2], \
    s = 10.0, color = 'k', marker = "x", label = r"$\nu_{meas}$")
pl.plot(time_points, nusim, label = r"$\nu_{sim}$")

pl.xlabel("$t$")
pl.ylabel(r"$\nu$", rotation = 0)
pl.xlim(0.0, 4.2)

pl.legend(loc = "lower left")

pl.subplot2grid((2, 2), (1, 0))
pl.scatter(time_points[::2], wmeas[::2], \
    s = 10.0, color = 'k', marker = "x", label = "$\omega_{meas}$")
pl.plot(time_points, wsim, label = "$\omega_{sim}$")

pl.xlabel("$t$")
pl.ylabel("$\omega$", rotation = 0)
pl.xlim(0.0, 4.2)

pl.legend(loc = "lower right")

pl.subplot2grid((2, 2), (0, 1), rowspan = 2)
pl.scatter(numeas[::2], wmeas[::2], \
    s = 10.0, color = 'k', marker = "x", \
    label = r"$(\nu_{meas},\,\omega_{meas})$")
pl.plot(nusim, wsim, label = r"$(\nu_{sim},\,\omega_{sim})$")

pl.xlabel(r"$\nu$")
pl.ylabel("$\omega$", rotation = 0)
pl.xlim(-2.5, 3.0)
pl.ylim(-5.0, 5.0)

pl.legend()

pl.show()
