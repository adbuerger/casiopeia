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

from interfaces import casadi_interface as ci

'''

The naming of the matrix blocks within this class matches the following scheme:

             n_a
           -------------
       m_a |  A  | B^T |
           |-----|-----|
       m_b |  B  |  C  |
           -------------

'''

class KKTMatrix(object):

    @property
    def kkt_matrix(self):

        return self._kkt_matrix


    def _setup_langrangian_hessian(self, gauss_newton_lagrangian_hessian):

        # Get the Hessian matrix of the Lagrangian of the Gauss Newton (!)
        # least squares parameter estimation problem

        self.hess_lag = gauss_newton_lagrangian_hessian


    def _setup_kkt_matrix(self, equality_constraints, optimization_variables):

        # Construct the KKT matrix from the Hessian of the Langrangian and the
        # Jacobian of the equality constraints

        kkt_matrix_A = self.hess_lag

        kkt_matrix_B = ci.jacobian(equality_constraints, \
            optimization_variables)

        kkt_matrix_C = ci.mx(equality_constraints.shape[0], \
            equality_constraints.shape[0])

        self._kkt_matrix = ci.blockcat( \
            kkt_matrix_A, kkt_matrix_B.T, \
            kkt_matrix_B, kkt_matrix_C)


    def __init__(self, gauss_newton_lagrangian_hessian, equality_constraints, \
        optimization_variables):

        r'''

        This class is designed only for internal use within casiopeia!

        This class is used to construct the Gauss Newton KKT matrix for a
        parameter estimation problem, for further use for covariance
        computation with ``fisher_matrix.py`` and ``covariance_matrix.py``

        '''

        self._setup_langrangian_hessian(gauss_newton_lagrangian_hessian)

        self._setup_kkt_matrix(equality_constraints, optimization_variables)


class FisherMatrix(object):

    @property
    def fisher_matrix(self):

        return self._fisher_matrix


    def _split_up_kkt_matrix(self, kkt_matrix, number_of_unknown_parameters):

        self._fisher_matrix_A = kkt_matrix[ \
            : number_of_unknown_parameters, \
            : number_of_unknown_parameters]

        self._fisher_matrix_B = kkt_matrix[ \
            number_of_unknown_parameters :, \
            : number_of_unknown_parameters]

        self._fisher_matrix_C = kkt_matrix[ \
        number_of_unknown_parameters :, \
            number_of_unknown_parameters :]


    def _setup_fisher_matrix(self):

        self._fisher_matrix = self._fisher_matrix_A - ci.mul([ \

                self._fisher_matrix_B.T, \

                ci.solve(self._fisher_matrix_C, \
                        
                        self._fisher_matrix_B, \
                        
                        "csparse")])


    def __init__(self, kkt_matrix, number_of_unknown_parameters):

        r'''

        This class is designed only for internal use within casiopeia!

        This class is used to construct the Fisher information matrix for a
        parameter estimation problem from a KKT matrix constructed using
        the methods from ``kkt_matrix.py``, for further use for covariance
        computation with ``covariance_matrix.py``

        '''

        self._split_up_kkt_matrix(kkt_matrix, number_of_unknown_parameters)

        self._setup_fisher_matrix()

        # self._fisher_matrix = kkt_matrix


class CovarianceMatrix(object):

    @property
    def covariance_matrix(self):

        return self._covariance_matrix


    def _setup_covariance_matrix(self, fisher_matrix):

        self._covariance_matrix = ci.solve( \
            
            fisher_matrix, \
            
            ci.mx_eye(fisher_matrix.shape[0]), \
            
            "csparse")


    def __init__(self, fisher_matrix):

        r'''

        This class is designed only for internal use within casiopeia!

        This class is used to construct the covariance matrix for a
        parameter estimation problem from a Fisher information matrix
        constructed using the methods from ``fisher_matrix.py``.

        '''

        self._setup_covariance_matrix(fisher_matrix)


class DirectFactorizationCovarianceMatrix(object):

    @property
    def covariance_matrix(self):

        return self._covariance_matrix


    def _setup_covariance_matrix(self, kkt_matrix, \
            number_of_unknown_parameters):


        I = ci.mx_eye(number_of_unknown_parameters)
        O = ci.mx(kkt_matrix.shape[0] - number_of_unknown_parameters, \
            number_of_unknown_parameters)

        Z_p = ci.vertcat([I,O])

        self._covariance_matrix = ci.solve(kkt_matrix, Z_p, "csparse")[ \
            :number_of_unknown_parameters, :number_of_unknown_parameters]


    def __init__(self, kkt_matrix, number_of_unknown_parameters):

        r'''

        This class is designed only for comparsion of other covariance matrix 
        implementations in casiopeia!

        '''

        self._setup_covariance_matrix(kkt_matrix, \
            number_of_unknown_parameters)


def setup_covariance_matrix_scaling_factor_beta(equality_constraints, \
    optimization_variables, residuals):

    beta = ci.mul([residuals.T, residuals]) / (residuals.numel() + \
        equality_constraints.numel() - optimization_variables.numel())

    return beta


def setup_a_criterion(covariance_matrix):

    return (1.0 / covariance_matrix.shape[0]) * ci.trace(covariance_matrix)


def setup_d_criterion(covariance_matrix):

    return pow(ci.det(covariance_matrix), (1.0 / covariance_matrix.shape[0]))
