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

data_opt = pl.loadtxt("2d_vehicle_doe_coll.txt")
uopt = data_opt[:78].reshape(-1, 2)

pdata = [0.273408, 11.5602, 2.45652, 7.90959, -0.44353, -0.249098]


sim_init = cp.sim.Simulation(system, pdata)
sim_init.run_system_simulation(time_points = time_points, \
    x0 = ydata[0,:], udata = uinit)

ydata_sim_init = sim_init.simulation_results


sim_opt = cp.sim.Simulation(system, pdata)
sim_opt.run_system_simulation(time_points = time_points, \
    x0 = ydata[0,:], udata = uopt)

ydata_sim_opt = sim_opt.simulation_results


delta_init = uinit[:, 0]
D_init = uinit[:, 1]

delta_opt = uopt[:, 0]
D_opt = uopt[:, 1]

umin = [-0.436332, -0.3216]
umax = [0.436332, 1.0]

xmin = [-0.787, -1.531, -12.614, 0.0]
xmax = [1.2390, 0.014, 0.013, 0.7102]


pl.close("all")

pl.figure()

pl.subplot2grid((2, 1), (0, 0))

pl.step(time_points[:-1], delta_init, label = "$\delta_{init}$")
pl.step(time_points[:-1], D_init, label = "$D_{init}$")

pl.plot([time_points[0], time_points[-2]], [umin[0], umin[0]], \
    color = "b", linestyle = "dashed", label = "$\delta_{min}$")
pl.plot([time_points[0], time_points[-2]], [umax[0], umax[0]], \
    color = "b", linestyle = "dotted", label = "$\delta_{max}$")

pl.plot([time_points[0], time_points[-2]], [umin[1], umin[1]], \
    color = "g", linestyle = "dashed", label = "$D_{min}$")
pl.plot([time_points[0], time_points[-2]], [umax[1], umax[1]], \
    color = "g", linestyle = "dotted", label = "$D_{max}$")

pl.xlabel("$t$")
pl.ylabel("$\delta,\,D$", rotation = 0)
pl.ylim(-0.6, 1.1)
pl.legend(loc = "upper right")

pl.subplot2grid((2, 1), (1, 0))

pl.step(time_points[:-1], delta_opt, label = "$\delta_{opt,coll}$")
pl.step(time_points[:-1], D_opt, label = "$D_{opt,coll}$")

pl.plot([time_points[0], time_points[-2]], [umin[0], umin[0]], \
    color = "b", linestyle = "dashed", label = "$\delta_{min}$")
pl.plot([time_points[0], time_points[-2]], [umax[0], umax[0]], \
    color = "b", linestyle = "dotted", label = "$\delta_{max}$")

pl.plot([time_points[0], time_points[-2]], [umin[1], umin[1]], \
    color = "g", linestyle = "dashed", label = "$D_{min}$")
pl.plot([time_points[0], time_points[-2]], [umax[1], umax[1]], \
    color = "g", linestyle = "dotted", label = "$D_{max}$")

pl.xlabel("$t$")
pl.ylabel("$\delta,\,D$", rotation = 0)
pl.ylim(-0.6, 1.1)
pl.legend(loc = "upper right")

pl.show()


x_sim_init = ydata_sim_init[0,:].T
y_sim_init = ydata_sim_init[1,:].T
psi_sim_init = ydata_sim_init[2,:].T
v_sim_init = ydata_sim_init[3,:].T

x_sim_opt = ydata_sim_opt[0,:].T
y_sim_opt = ydata_sim_opt[1,:].T
psi_sim_opt = ydata_sim_opt[2,:].T
v_sim_opt = ydata_sim_opt[3,:].T

pl.figure()

pl.subplot2grid((4, 2), (0, 0))
pl.plot(time_points, x_sim_init, label = "$X_{sim,init}$")
pl.plot(time_points, x_sim_opt, label = "$X_{sim,coll}$")
pl.xlabel("$t$")
pl.ylabel("$X$", rotation = 0)
pl.legend(loc = "lower left")

pl.subplot2grid((4, 2), (1, 0))
pl.plot(time_points, y_sim_init, label = "$Y_{sim,init}$")
pl.plot(time_points, y_sim_opt, label = "$Y_{sim,coll}$")
pl.xlabel("$t$")
pl.ylabel("$Y$", rotation = 0)
pl.legend(loc = "lower left")

pl.subplot2grid((4, 2), (2, 0))
pl.plot(time_points, psi_sim_init, label = "$\psi_{sim,init}$")
pl.plot(time_points, psi_sim_opt, label = "$\psi_{sim,coll}$")
pl.xlabel("$t$")
pl.ylabel("$\psi$", rotation = 0)
pl.legend(loc = "lower left")

pl.subplot2grid((4, 2), (3, 0))
pl.plot(time_points, v_sim_init, label = "$v_{sim,init}$")
pl.plot(time_points, v_sim_opt, label = "$v_{sim,coll}$")
pl.xlabel("$t$")
pl.ylabel("$v$", rotation = 0)
pl.legend(loc = "upper left")

pl.subplot2grid((4, 2), (0, 1), rowspan = 4)
pl.plot(x_sim_init, y_sim_init, label = "$(X_{sim,init},\,Y_{sim,init})$")
pl.plot(x_sim_opt, y_sim_opt, label = "$(X_{sim,coll},\,Y_{sim,coll})$")
pl.xlabel("$X$")
pl.ylabel("$Y$", rotation = 0)
pl.legend(loc = "lower left")

pl.show()
