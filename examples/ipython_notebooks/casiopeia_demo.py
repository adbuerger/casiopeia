#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pylab as pl

def plot_measurements(time_points, ydata):

    pl.rc("text", usetex = True)
    pl.rc("font", family="serif")

    pl.subplot2grid((4, 2), (0, 0))
    pl.plot(time_points, ydata[:,0])
    pl.title("Considered measurement data")
    pl.xlabel("t")
    pl.ylabel("X", rotation = 0, labelpad = 20)

    pl.subplot2grid((4, 2), (1, 0))
    pl.plot(time_points, ydata[:,1])
    pl.xlabel("t")
    pl.ylabel("Y", rotation = 0, labelpad = 15)

    pl.subplot2grid((4, 2), (2, 0))
    pl.plot(time_points, ydata[:,2])
    pl.xlabel("t")
    pl.ylabel(r"\phi", rotation = 0, labelpad = 15)

    pl.subplot2grid((4, 2), (3, 0))
    pl.plot(time_points, ydata[:,3])
    pl.xlabel("t")
    pl.ylabel("v", rotation = 0, labelpad = 20)

    pl.subplot2grid((4, 2), (0, 1), rowspan = 4)
    pl.plot(ydata[:,0], ydata[:, 1])
    pl.title("Considered racecar track (measured)")
    pl.xlabel("X")
    pl.ylabel("Y", rotation = 0, labelpad = 20)
    pl.show()


def plot_measurements_and_simulation_results(time_points, ydata, y_sim):
    
    pl.subplot2grid((4, 2), (0, 0))
    pl.plot(time_points, ydata[:,0], label = "measurements")
    pl.plot(time_points, y_sim[:,0], label = "simulation")
    pl.title("Measurement data compared to simulation results")
    pl.xlabel("t")
    pl.ylabel("X", rotation = 0, labelpad = 20)
    pl.legend(loc = "lower left")

    pl.subplot2grid((4, 2), (1, 0))
    pl.plot(time_points, ydata[:,1], label = "measurements")
    pl.plot(time_points, y_sim[:,1], label = "simulation")
    pl.xlabel("t")
    pl.ylabel("Y", rotation = 0, labelpad = 15)
    pl.legend("lower right")

    pl.subplot2grid((4, 2), (2, 0))
    pl.plot(time_points, ydata[:,2], label = "measurements")
    pl.plot(time_points, y_sim[:,2], label = "simulation")
    pl.xlabel("t")
    pl.ylabel(r"\phi", rotation = 0, labelpad = 15)
    pl.legend("lower left")

    pl.subplot2grid((4, 2), (3, 0))
    pl.plot(time_points, ydata[:,3], label = "measurements")
    pl.plot(time_points, y_sim[:,3], label = "simulation")
    pl.xlabel("t")
    pl.ylabel("v", rotation = 0, labelpad = 20)
    pl.legend("upperleft")

    pl.subplot2grid((4, 2), (0, 1), rowspan = 4)
    pl.plot(ydata[:,0], ydata[:, 1], label = "measurements")
    pl.plot(y_sim[:,0], y_sim[:,1], label = "estimations")
    pl.title("Measured race track compared to simulated results")
    pl.xlabel("X")
    pl.ylabel("Y", rotation = 0, labelpad = 20)
    pl.legend(loc = "upper left")
    pl.show()


def plot_simulation_results_initial_controls(time_points, y_sim):

    pl.rc("text", usetex = True)
    pl.rc("font", family="serif")

    pl.subplot2grid((4, 2), (0, 0))
    pl.plot(time_points, y_sim[:,0])
    pl.title("Simulation results for initial controls")
    pl.xlabel("t")
    pl.ylabel("X", rotation = 0, labelpad = 20)

    pl.subplot2grid((4, 2), (1, 0))
    pl.plot(time_points, y_sim[:,1])
    pl.xlabel("t")
    pl.ylabel("Y", rotation = 0, labelpad = 15)

    pl.subplot2grid((4, 2), (2, 0))
    pl.plot(time_points, y_sim[:,2])
    pl.xlabel("t")
    pl.ylabel(r"\phi", rotation = 0, labelpad = 15)

    pl.subplot2grid((4, 2), (3, 0))
    pl.plot(time_points, y_sim[:,3])
    pl.xlabel("t")
    pl.ylabel("v", rotation = 0, labelpad = 20)

    pl.subplot2grid((4, 2), (0, 1), rowspan = 4)
    pl.plot(y_sim[:,0], y_sim[:, 1])
    pl.title("Simulated race car path for initial controls")
    pl.xlabel("X")
    pl.ylabel("Y", rotation = 0, labelpad = 20)
    pl.show()

def plot_simulation_results_initial_and_optimized_controls(time_points, \
    y_sim_init, y_sim_opt):

    pl.rc("text", usetex = True)
    pl.rc("font", family="serif")

    pl.subplot2grid((4, 2), (0, 0))
    pl.plot(time_points, y_sim_init[:,0], label = "initial")
    pl.plot(time_points, y_sim_opt[:,0], label = "optimized")
    pl.title("Simulation results for initial and optimized control")
    pl.xlabel("$t$")
    pl.ylabel("$X$", rotation = 0)
    pl.legend(loc = "lower left")

    pl.subplot2grid((4, 2), (1, 0))
    pl.plot(time_points, y_sim_init[:,1], label = "initial")
    pl.plot(time_points, y_sim_opt[:,1], label = "optimized")
    pl.xlabel("$t$")
    pl.ylabel("$Y$", rotation = 0)
    pl.legend(loc = "lower left")

    pl.subplot2grid((4, 2), (2, 0))
    pl.plot(time_points, y_sim_init[:,2], label = "initial")
    pl.plot(time_points, y_sim_opt[:,2], label = "optimized")
    pl.xlabel("$t$")
    pl.ylabel("$\psi$", rotation = 0)
    pl.legend(loc = "lower left")

    pl.subplot2grid((4, 2), (3, 0))
    pl.plot(time_points, y_sim_init[:,3], label = "initial")
    pl.plot(time_points, y_sim_opt[:,3], label = "optimized")
    pl.xlabel("$t$")
    pl.ylabel("$v$", rotation = 0)
    pl.legend(loc = "upper left")

    pl.subplot2grid((4, 2), (0, 1), rowspan = 4)
    pl.plot(y_sim_init[:,0], y_sim_init[:,1], label = "initial")
    pl.plot(y_sim_opt[:,0], y_sim_opt[:,1], label = "optimized")
    pl.title("Simulated race car path for initial and optimized controls")
    pl.xlabel("$X$")
    pl.ylabel("$Y$", rotation = 0)
    pl.legend(loc = "lower left")

    pl.show()

def plot_initial_and_optimized_controls(time_points, \
    udata_init, udata_opt, umin, umax):

    pl.rc("text", usetex = True)
    pl.rc("font", family="serif")

    pl.subplot2grid((2, 1), (0, 0))
    pl.step(time_points[:-1], udata_init[:,0], label = "$\delta_{init}$")
    pl.step(time_points[:-1], udata_init[:,1], label = "$D_{init}$")

    pl.plot([time_points[0], time_points[-2]], [umin[0], umin[0]], \
        color = "b", linestyle = "dashed", label = "$\delta_{min}$")
    pl.plot([time_points[0], time_points[-2]], [umax[0], umax[0]], \
        color = "b", linestyle = "dotted", label = "$\delta_{max}$")

    pl.plot([time_points[0], time_points[-2]], [umin[1], umin[1]], \
        color = "g", linestyle = "dashed", label = "$D_{min}$")
    pl.plot([time_points[0], time_points[-2]], [umax[1], umax[1]], \
        color = "g", linestyle = "dotted", label = "$D_{max}$")

    pl.ylabel("$\delta,\,D$", rotation = 0)
    pl.ylim(-0.6, 1.1)
    pl.title("Initial and optimized controls")
    pl.legend(loc = "upper right")

    pl.subplot2grid((2, 1), (1, 0))
    pl.step(time_points[:-1], udata_opt[:,0], label = "$\delta_{opt,coll}$")
    pl.step(time_points[:-1], udata_opt[:,1], label = "$D_{opt,coll}$")

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