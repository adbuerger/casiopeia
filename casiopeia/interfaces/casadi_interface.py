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

import casadi as ca

def sx_sym(name, dim1 = 1, dim2 = 1):

    return ca.SX.sym(name, dim1, dim2)


def sx_function(name, inputs, outputs):

    return ca.Function(name, inputs, outputs)


def mx_sym(name, dim1 = 1, dim2 = 1):

    return ca.MX.sym(name, dim1, dim2)


def mx_function(name, inputs, outputs, options = {}):

    return ca.Function(name, inputs, outputs, options)


def dmatrix(dim1, dim2 = 1):

    return ca.DM(dim1, dim2)


def depends_on(b, a):

    return ca.depends_on(b, a)


def collocation_points(order, scheme):

    # "collocationPoints -> collocation_points. Also the returned list is
    # now one element less than before. To get the old behaviour, prepend
    # the result with zero.", see
    # https://github.com/casadi/casadi/wiki/Changes-v2.4.0-to-v3.0.0

    return [0.0] + ca.collocation_points(order, scheme)


def vertcat(inputlist):

    return ca.vertcat(*inputlist)


def veccat(inputlist):

    return ca.veccat(*inputlist)


def horzcat(inputlist):

    return ca.horzcat(*inputlist)


def repmat(inputobj, dim1, dim2):

    return ca.repmat(inputobj, dim1, dim2)
    

def vec(inputobj):

    return ca.vec(inputobj)


def sqrt(inputobj):

    return ca.sqrt(inputobj)


# def nlpIn(x = None):

#     # return ca.nlpIn(x = x)
#     return {"x": x}


# def nlpOut(f = None, g = None):

#     # return ca.nlpOut(f = f, g = g)
#     return {"f": f, "g": g}


def mul(inputobj):

    return ca.mtimes(inputobj)


def NlpSolver(name, solver, nlp, options):

    return ca.nlpsol(name, solver, nlp, options)


def daeIn(t = None, x = None, p = None):

    if t is None:

        # return ca.daeIn(x = x, p = p)
        return {"x": x, "p": p}

    else:

        # return ca.daeIn(t = t, x = x, p = p)
        return {"t": t, "x": x, "p": p}


def daeOut(ode = None, alg = None):

    if alg is None:

        # return ca.daeOut(ode = ode)
        return {"ode": ode}

    else:

        # return ca.daeOut(ode = ode, alg = alg)
        return {"ode": ode, "alg": alg}


def Integrator(name, method, dae, options = {}):

    return ca.integrator(name, method, dae, options)


def diag(inputobj):

    return ca.diag(inputobj)


def mx(dim1, dim2):

    return ca.MX(dim1, dim2)


def blockcat(a, b, c, d):

    return ca.blockcat(a, b, c, d)


def jacobian(a, b):

    return ca.jacobian(a, b)


def solve(a, b, solver):

    return ca.solve(a, b, solver)


def mx_eye(dim1):

    return ca.MX.eye(dim1)


def trace(inputobj):

    return ca.trace(inputobj)


def det(inputobj):

    return ca.det(inputobj)


def diagcat(inputobj):

    return ca.diagcat(*inputobj)
