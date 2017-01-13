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
import unittest

from numpy.testing import assert_array_equal

import casiopeia.interfaces.casadi_interface as ci

class MXSymbolic(unittest.TestCase):

    def setUp(self):

        self.name = "varname"
        self.dim1 = 3
        self.dim2 = 2


    def test_mx_sym_is_mx_sym_instance(self):

        mx_sym = ci.mx_sym(self.name, self.dim1)
        
        self.assertTrue(isinstance(mx_sym, ca.casadi.MX))


    def test_mx_sym_nodim(self):

        mx_sym = ci.mx_sym(self.name)
        
        self.assertEqual(mx_sym.shape, (1, 1))


    def test_mx_sym_1dim(self):

        mx_sym = ci.mx_sym(self.name, self.dim1)
        
        self.assertEqual(mx_sym.shape, (self.dim1, 1))


    def test_mx_sym_2dim(self):

        mx_sym = ci.mx_sym(self.name, self.dim1, self.dim2)
        
        self.assertEqual(mx_sym.shape, (self.dim1, self.dim2))


class SXSymbolic(unittest.TestCase):

    def setUp(self):

        self.name = "varname"
        self.dim1 = 3
        self.dim2 = 2


    def test_sx_sym_is_sx_sym_instance(self):

        sx_sym = ci.sx_sym(self.name, self.dim1)
        
        self.assertTrue(isinstance(sx_sym, ca.casadi.SX))


    def test_sx_sym_nodim(self):

        sx_sym = ci.sx_sym(self.name)
        
        self.assertEqual(sx_sym.shape, (1, 1))


    def test_sx_sym_1dim(self):

        sx_sym = ci.sx_sym(self.name, self.dim1)
        
        self.assertEqual(sx_sym.shape, (self.dim1, 1))


    def test_sx_sym_2dim(self):

        sx_sym = ci.sx_sym(self.name, self.dim1, self.dim2)
        
        self.assertEqual(sx_sym.shape, (self.dim1, self.dim2))


class SXFunction(unittest.TestCase):

    def setUp(self):

        self.name = "funcname"
        a = ca.SX.sym("a", 1)
        self.input = [a]
        self.output = [a**2]


    def test_sx_function_is_function_instance(self):

        sx_function = ci.sx_function(self.name, self.input, self.output)
        
        self.assertTrue(isinstance(sx_function, ca.casadi.Function))


    def test_sx_function_call(self):

        sx_function = ci.sx_function(self.name, self.input, self.output)
        b = 2
        c = 4

        self.assertTrue(sx_function([b]), c)


class MXFunction(unittest.TestCase):

    def setUp(self):

        self.name = "funcname"
        a = ca.MX.sym("a", 1)
        self.input = [a]
        self.output = [a**2]


    def test_mx_function_is_function_instance(self):

        mx_function = ci.mx_function(self.name, self.input, self.output)
        
        self.assertTrue(isinstance(mx_function, ca.casadi.Function))


    def test_mx_function_call(self):

        mx_function = ci.mx_function(self.name, self.input, self.output)
        b = 2
        c = 4

        self.assertTrue(mx_function([b]), c)


class DMatrix(unittest.TestCase):

    def setUp(self):

        self.dim1 = 3
        self.dim2 = 2


    def test_dmatrix_1dim(self):

        dmatrix = ci.dmatrix(self.dim1)
        
        self.assertEqual(dmatrix.shape, (self.dim1, 1))


    def test_dmatrix_2dim(self):

        dmatrix = ci.dmatrix(self.dim1, self.dim2)
        
        self.assertEqual(dmatrix.shape, (self.dim1, self.dim2))


class DependsOn(unittest.TestCase):

    def setUp(self):

        self.a = ca.MX.sym("a")


    def test_expression_does_depend(self):

        b = 2 * self.a
        self.assertTrue(ci.depends_on(b, self.a))


    def test_expression_does_not_depend(self):

        b = ca.MX.sym("b")
        self.assertFalse(ci.depends_on(b, self.a))


class CollocationPoints(unittest.TestCase):

    def setUp(self):

        self.order = 3
        self.scheme = "radau"
        # self.collocation_points = casadi.collocationPoints(3, "radau")
        self.collocation_points = \
            [0.0, 0.15505102572168222, 0.6449489742783179, 1.0]

    def test_collocation_points(self):

        self.assertEqual(ci.collocation_points(self.order, self.scheme), \
            self.collocation_points)

class Vertcat(unittest.TestCase):

    def setUp(self):

        self.lenlist = 3
        self.inputlist = [k for k in range(self.lenlist)]


    def test_assure_vertcat_returns_column_vector(self):

        v = ci.vertcat(self.inputlist)

        self.assertEqual(v.shape[0], self.lenlist)


class Veccat(unittest.TestCase):

    def setUp(self):

        self.inputlist = [ca.MX.sym("a", 4, 3), ca.MX.sym("b", 2, 4)]


    def test_assure_veccat_returns_column_vector(self):

        v = ci.veccat(self.inputlist)

        self.assertEqual(v.shape[0], 20)


class Horzcat(unittest.TestCase):

    def setUp(self):

        self.lenlist = 3
        self.inputlist = [k for k in range(self.lenlist)]


    def test_assure_horzcat_returns_column_vector(self):

        v = ci.horzcat(self.inputlist)

        self.assertEqual(v.shape[1], self.lenlist)


class Repmat(unittest.TestCase):

    def setUp(self):

        self.repetition = 2
        self.input = np.array([[1, 2]])


    def test_repeat_vestors_horizontally(self):

        est_results = np.array([[1, 2, 1, 2]])

        repeated = ci.repmat(self.input, 1, self.repetition)

        assert_array_equal(repeated, est_results)


    def test_repeat_vestors_vertically(self):

        est_results = np.array([[1, 2], [1, 2]])

        repeated = ci.repmat(self.input, self.repetition, 1)

        assert_array_equal(repeated, est_results)


class Vec(unittest.TestCase):

    def setUp(self):

        self.inputlist = ca.MX.sym("a", 4, 3)


    def test_assure_vec_returns_column_vector(self):

        v = ci.vec(self.inputlist)

        self.assertEqual(v.shape[0], 12)


class Sqrt(unittest.TestCase):

    def setUp(self):

        self.a = 9


    def test_assure_vec_returns_column_vector(self):

        b = ci.sqrt(self.a)

        self.assertEqual(b, 3)


# class NlpIn(unittest.TestCase):

#     def setUp(self):

#         self.x = ca.MX.sym("x")
#         self.ref_str = "{'x': MX(x)}"


#     def test_nlp_in_return_value(self):

#         ret_tuple = ci.nlpIn(x = self.x)
#         self.assertEqual(str(ret_tuple), self.ref_str)


# class NlpOut(unittest.TestCase):

#     def setUp(self):

#         self.f = ca.MX.sym("f")
#         self.g = ca.MX.sym("g")
#         self.ref_str = "{'g': MX(g), 'f': MX(f)}"


#     def test_nlp_out_return_value(self):

#         ret_tuple = ci.nlpOut(f = self.f, g = self.g)
#         self.assertEqual(str(ret_tuple), self.ref_str)


class Mul(unittest.TestCase):

    def setUp(self):

        self.a = np.array([[1, 2, 3]])
        self.b = self.a.T
        self.ref_val = 14


    def test_vector_multiplication(self):

        ret_val = ci.mul([self.a, self.b])
        self.assertEqual(ret_val, self.ref_val)


class NLpSolver(unittest.TestCase):

    # It's not really useful to test NlPSolver completely, so just test if
    # the syntax for calling it is still the same and the class exists
    
    def setUp(self):

        self.x = ca.MX.sym("x")
        self.f = 2 * self.x
        self.g = self.x + 3

        self.nlp = {"x": self.x, "f": self.f, "g": self.g}


    def test_nlp_solver_class_exists(self):

        ci.NlpSolver("nlp", "ipopt", self.nlp, {})


class DaeIn(unittest.TestCase):

    def setUp(self):

        self.t = ca.MX.sym("t")
        self.x = ca.MX.sym("x")
        self.p = ca.MX.sym("p")


    def test_dae_in_return_value_no_t(self):

        ref_str = "{'x': MX(x), 'p': MX(p)}"
        ret_tuple = ci.daeIn(x = self.x, p = self.p)
        self.assertEqual(str(ret_tuple), ref_str)


    def test_dae_in_return_value(self):

        ref_str = "{'x': MX(x), 't': MX(t), 'p': MX(p)}"
        ret_tuple = ci.daeIn(t = self.t, x = self.x, p = self.p)
        self.assertEqual(str(ret_tuple), ref_str)


class DaeOut(unittest.TestCase):

    def setUp(self):

        self.f = ca.MX.sym("f")
        self.g = ca.MX.sym("g")


    def test_dae_out_return_value_no_alg(self):

        ref_str = "{'ode': MX(f)}"
        ret_tuple = ci.daeOut(ode = self.f)
        self.assertEqual(str(ret_tuple), ref_str)


    def test_dae_out_return_value(self):

        ref_str = "{'alg': MX(g), 'ode': MX(f)}"
        ret_tuple = ci.daeOut(ode = self.f, alg = self.g)
        self.assertEqual(str(ret_tuple), ref_str)


class Integrator(unittest.TestCase):

    # It's not really useful to test Integrator completely, so just test if
    # the syntax for calling it is still the same and the class exists
    
    def setUp(self):

        self.x = ca.MX.sym("x")
        self.f = self.x

        self.dae = {"x": self.x, "ode": self.f}


    def test_nlp_solver_class_exists(self):

        ci.Integrator("dae", "rk", self.dae, {})


class Diag(unittest.TestCase):

    def setUp(self):

        self.diag_entires = np.array([2, 2, 2])
        self.diag_matrix = np.diag(self.diag_entires)


    def test_diag_returns_diagonal_entries(self):

        diag_entries_ret = ci.diag(self.diag_matrix)
        assert_array_equal(self.diag_entires, np.squeeze(diag_entries_ret))
        