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

from interfaces import casadi_interface as ci

def setup_covariance_matrix(optimization_variables, weightings, \
    equality_constraints, number_of_unknown_parameters):

    '''
          n_a
        -------------
    m_a |  A  | B^T |
        |-----|-----|
    m_b |  B  |  C  |
        -------------

    '''

    # Construct the Hessian matrix of the Lagrangian of the least squares
    # parameter estimation problem

    hess_lag_m_a = optimization_variables.shape[0] - weightings.shape[0]
    hess_lag_n_a = hess_lag_m_a
    hess_lag_A = ci.mx(hess_lag_m_a, hess_lag_n_a)

    hess_lag_m_b = weightings.shape[0]
    hess_lag_B = ci.mx(hess_lag_m_b, hess_lag_n_a)

    hess_lag_C = ci.diag(weightings)

    hess_lag = ci.blockcat( \
        hess_lag_A, hess_lag_B.T, \
        hess_lag_B, hess_lag_C)

    # Construct the KKT matrix from the Hessian of the Langrangian and the
    # Jacobian of the equality constraints

    kkt_matrix_A = hess_lag

    kkt_matrix_B = ci.jacobian(equality_constraints, optimization_variables)

    kkt_matrix_C = ci.mx(equality_constraints.shape[0], \
        equality_constraints.shape[0])

    kkt_matrix = ci.blockcat( \
        kkt_matrix_A, kkt_matrix_B.T, \
        kkt_matrix_B, kkt_matrix_C)

    # Construct the covariance matrix from the inverse of the KKT matrix;
    # using the Schur complement, only the relevant part of the covariance
    # matrix that contains the information about the unknown parameters
    # (cov_mat_A) is constructed

    cov_mat_inv_A = kkt_matrix[: number_of_unknown_parameters, \
        : number_of_unknown_parameters]

    cov_mat_inv_B = kkt_matrix[number_of_unknown_parameters :, \
        : number_of_unknown_parameters]

    cov_mat_inv_C = kkt_matrix[number_of_unknown_parameters :, \
        number_of_unknown_parameters :]

    cov_mat_A = ci.solve( \

            cov_mat_inv_A - ci.mul([ \

                cov_mat_inv_B.T, \
                ci.solve(cov_mat_inv_C, cov_mat_inv_B, "csparse")]), \

            ci.mx_eye(number_of_unknown_parameters), \

            "csparse"

        )

    return cov_mat_A


def setup_beta(residuals, measurements, equality_constraints, \
    optimization_variables):

    beta = \
        ci.mul([residuals.T, residuals]) / (measurements.size() + \
        equality_constraints.size1() - optimization_variables.size())

    return beta


def setup_a_criterion(covariance_matrix):

    return (1.0 / covariance_matrix.shape[0]) * ci.trace(covariance_matrix)


def setup_d_criterion(covariance_matrix):

    return pow(ci.det(covariance_matrix), (1.0 / covariance_matrix.shape[0]))
