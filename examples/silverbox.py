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

# This example is an adapted version of the system identification example
# included in CasADi, for the original file see:
# https://github.com/casadi/casadi/blob/master/docs/examples/python/sysid.py

import pylab as pl

import casadi as ca
import casiopeia as cp

N = 10000
fs = 610.1

p_true = ca.DMatrix([5.625e-6,2.3e-4,1,4.69])
p_guess = ca.DMatrix([5,3,1,5])
scale = ca.vertcat([1e-6,1e-4,1,1])

x = ca.MX.sym("x", 2)
u = ca.MX.sym("u", 1)
p = ca.MX.sym("p", 4)

f = ca.vertcat([
        x[1], \
        (u - scale[3] * p[3] * x[0]**3 - scale[2] * p[2] * x[0] - \
            scale[1] * p[1] * x[1]) / (scale[0] * p[0]), \
    ])

phi = x

odesys = cp.system.System( \
    x = x, u = u, p = p, f = f, phi = phi)

dt = 1.0 / fs
tN = pl.linspace(0, N, N+1) * dt

udata = ca.DMatrix(0.1*pl.random(N))

simulation_true_parameters = cp.sim.Simulation( \
    system = odesys, pdata = p_true / scale)
simulation_true_parameters.run_system_simulation( \
    x0 = [0.0, 0.0], time_points = tN, udata = udata)

ydata = simulation_true_parameters.simulation_results
ydata += 1e-3 * pl.random((x.shape[0], N+1))

wv = pl.ones(ydata.shape)

pe = cp.pe.LSq(system = odesys, \
    time_points = tN, xinit = ydata, \
    ydata = ydata, wv = wv, udata = udata, pinit = p_guess)
    
pe.run_parameter_estimation()

simulation_estimated_parameters = cp.sim.Simulation( \
    system = odesys, pdata = pe.estimated_parameters)
simulation_estimated_parameters.run_system_simulation( \
    x0 = [0.0, 0.0], time_points = tN, udata = udata)

pl.close("all")
pl.figure()

pl.scatter(tN, pl.squeeze(ydata[0,:]))
pl.plot(tN, simulation_estimated_parameters.simulation_results[0,:].T)

pl.scatter(tN, pl.squeeze(ydata[1,:]))
pl.plot(tN, simulation_estimated_parameters.simulation_results[1,:].T)

pl.show()
