"""
Created on Tue Nov  1 17:24:19 2016

@author: Ki Wai Chau

This is the main file for the stochastic grid bundling method for XVA demonstrator with Numba
CPU precompiler. It contains two main functions and some minor supporting functions. 

The numba_sgbm_xva function is the main body of the whole algorithm, while cuda_sgbminnerblock 
is the main part of the backward process. This program works by calling the preset examples and 
model dependent function (especially the forward process generating function), performing the SGBM 
algorithm and returing the solution, error measurment and execution time.

The regressioninbundle and leastsquaresregression functions are used to calculate linear regression
for all the required function in the numerical steps and give an extra function of raising an error
when the L2 norm of the regression coefficients are too big. The success_adjusted_std calculate the 
standard deviation while taking into account of the exceeding bound error.
"""
import time
import logging
from progress_bar import ProgressBar

import numpy as np

# Bringing in an example class in which it holds the details of the stock model,
# the financial product and the regression setting
from example_xva_v2 import Example
from numba_blackscholes import numba_monte_carlo

logging.getLogger()

class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class ExceedBoundError(Error):
    """Self defined error for the linear regression when the coefficients is too big"""
    def __init__(self):
        pass

# Example related parameters are imported here as golbal variables for the consistency
# with the CUDA function.
EXAMPLE = Example
# The following parameters are related to the financial product.
TERMINAL_TIME = EXAMPLE.terminal_time
# The following parameters are related to the linear regression.
NUM_REGRESSION = EXAMPLE.no_of_regression
BASIS_ORDER = EXAMPLE.basis_order
regression_variable = EXAMPLE.regression_variable
# The following function is related to the market dynamic/ regulartory rules.
numerical_scheme = EXAMPLE.numerical_scheme
# The reference solution (if exists) is called here.
REFERENCE_RISKFREE = EXAMPLE.reference_riskfree
REFERENCE_ADJUST = EXAMPLE.refererce_adjust
if REFERENCE_RISKFREE:
    REFERENCE_RISKFREE_PRICE = EXAMPLE.reference_riskfree_price

def numba_sgbm_xva(monte_carlo, no_of_paths, no_time_steps, no_bundles, test_number=10):
    """Perform the SGBM algorithm.

    Under given parameters (total simulation samples, number of time step in partition
    and number of bundels in each time step), we run the sgbm algorithm for a number
    of time given by test_number. Other inputs including the (model dependent) forward
    simulation functuion. Each run consist of three parts, a forward simulation scheme
    to gather all necessary data, a partition procedure to separate data into "bundles"
    and a typical linear regression function with QR decomposition within each
    partition. It prints out the simulation result and the running time of each test.
    There are extra code in this program to work with regression function that throw an 
    error when the L2 norm of the regression coefficients is too big. This script returns 
    not a number if no successful run is completed. 

    :param monte_carlo: Forward simulation function
    :param no_of_paths: Number of stocks simulations in the Monte Carlo run
    :param no_time_steps: Numder of time step in time discretization
    :param no_bundles: Number of bundles in each time step.
    :param test_number: The number of test run for each set of parameters.
    :type montecarlo: function
    :type no_of_paths: int
    :type no_time_steps: int
    :type no_bundles: int
    :type test_number: int
    :returns: (risk_free_price, error_risk_free, std_risk_free,
               risk_adjusted_price, error_risk_adjusted, std_risk_adjusted,
               average_time)
    :rtype: (scaler, scaler, scaler, scaler, scaler, scaler, scaler)
    """
    # Arrays are allocated for storing test result for each run
    result_risk_free = np.empty(test_number)
    result_risk_adjusted = np.empty(test_number)
    test_time = np.empty(test_number)
    success = 0.0
    # Time partition for discretization
    timespan = np.arange(0, TERMINAL_TIME + TERMINAL_TIME/no_time_steps,\
                             TERMINAL_TIME/no_time_steps)

    for test in range(int(test_number)):
        logging.debug("Forward Simulations")
        start_time = time.time()

        # Forward simulation for all necessary information
        sorting_value, basis, expect_basis, expect_brownian_basis,\
                riskfree_price, riskfree_delta, adjusted_price, adjusted_delta =\
                monte_carlo(timespan, no_of_paths)

        middle_time = time.time()

        logging.debug("Backward Progression")
        try:
            # Function is called for the backward step, partition and regression
            result1, result2 = \
            sgbminnerblock(sorting_value, basis, expect_basis, expect_brownian_basis,
                           riskfree_price, riskfree_delta, adjusted_price, 
                           adjusted_delta, timespan, bundles=no_bundles)
        except ExceedBoundError:
            pass
        else:
            # Here we print out the riskfree price, risk adjusted price and full run time
            print("Risk-free option price", result1)
            print("Risk-adjusted option price:", result2)
            result_risk_free[test] = result1
            result_risk_adjusted[test] = result2
            success += 1.0

        end_time = time.time()
        print('Forward time:', middle_time - start_time, ', Backward time:', end_time - middle_time)
        test_time[test] = end_time - start_time

    print("Success rate = ", success/test_number)
    if success == 0.0:
        risk_free_price = error_risk_free = std_risk_free = \
        risk_adjusted_price = error_risk_adjusted = std_risk_adjusted = \
        average_time = np.nan
    else:
        # Calculate the average riskfree price, risk adjusted price and full run time over all runs
        risk_free_price = np.sum(result_risk_free)/success
        risk_adjusted_price = np.sum(result_risk_adjusted)/success
        average_time = np.average(test_time)

        # Calculate the error and the standard deviation of our approximation
        if REFERENCE_RISKFREE:
            error_risk_free = abs(risk_free_price - REFERENCE_RISKFREE_PRICE)
            std_risk_free = \
            success_adjusted_std(result_risk_free, REFERENCE_RISKFREE_PRICE, test_number, success)
        else:
            error_risk_free = np.nan
            std_risk_free = \
            success_adjusted_std(result_risk_free, risk_free_price, test_number, success)

        if not REFERENCE_ADJUST:
            error_risk_adjusted = np.nan
            std_risk_adjusted = \
            success_adjusted_std(result_risk_adjusted, risk_adjusted_price, test_number, success)

    return risk_free_price, error_risk_free, std_risk_free, \
           risk_adjusted_price, error_risk_adjusted, std_risk_adjusted, \
           average_time

def sgbminnerblock(sorting_value, basis, expect_basis, expect_brownian_basis, \
                   riskfree_price, riskfree_delta, adjusted_price, adjusted_delta,\
                   timespan, bundles):
    """Perform the SGBM backward algorithm with given forward simulations

    This algorithm collects all the necessary data from the forward simulation separately and
    then perform the backward step sequentially with respect to the time discretization, starting
    from the last time period. Within each time period, first all samples are sorted according to
    the sorting value and partition them into bundles containing simular sorting value paths.
    In each bundle, servaral regressions are completed and the resulting coefficients are used to
    calculate the target value (option prices) within each bundle. Finally, all the results are
    collected and a new cycle begins, untill we ratch time zero.

    :param sorting_value: Array of the values of sorting function
    :param basis: Array of the regression basis for each path at every time step
    :param expect_basis: Array of the expectation of the regression basis for each path at
                         every time step
    :param expect_brownian_basis: Array of the expectation of the regression basis multipled by a 
                                  Brownian motion at each dimension for each path at every time step
    :param riskfree_price: Array for the riskfree option price for each paths at expiry
    :param riskfree_delta: Array for the riskfree option hedging for each paths at expiry
    :param adjusted_price: Array for the risk-adjusted option price for each paths at expiry
    :param adjusted_delta: Array for the risk-adjusted option hedging for each paths at expiry
    :param timespan: Array of time partition points
    :param bundles: Number of bundles in all time step (except the starting one)
    :type sorting value: N x (M+1) array
    :type basis: N x Q x M array
    :type expect_basis: N x Q x M array
    :type expect_brownian_basis: N x Q x d x M array
    :type riskfree_price: N x 1 array
    :type riskfree_delta: N x d array
    :type adjusted_price: N x 1 array
    :type adjusted_delta: N x d array
    :type timespan: M+1 array
    :type bundles: int
    :returns: (average riskfree option price, average risk adjusted option price)
    :rtype: (scaler, scaler)

    .. todo:: Allowing variation of bundle size between time steps
    """
    num_of_time_points = np.size(timespan) # M+1
    num_of_samples = basis.shape[0] # N
    adaptive_bundles = bundles * np.ones(num_of_time_points, dtype=int)
    adaptive_bundles[0] = 1
    # Calculate the number of bundle for each time step.
    # (preparation for adaptive number of bundles)
    num_of_bundles = np.amax(adaptive_bundles)
    bundle_size = np.zeros((num_of_time_points, num_of_bundles), dtype=int)

    p_bar = ProgressBar(num_of_time_points-1, prefix='backward process',
                        suffix='', decimals=2, barLength=100)

    # The followings are the main backward in time algorithm.
    for j in range(num_of_time_points - 1):
        p_bar.update(j)

        # Calculate the number of paths in each bundle.
        c_bundles_num = adaptive_bundles[-j-2]
        bundle_size[-j-2, 0:c_bundles_num] = int(num_of_samples / c_bundles_num)
        bundle_size[-j-2, 0:num_of_samples - c_bundles_num*int(num_of_samples / c_bundles_num)] += 1

        # Sort the prices and deltas based on the order of sorting value
        order_p = np.argsort(sorting_value[:, -j-2]) 

        # Collect all the values that we have to perform regression on
        regression_coeff = np.zeros((c_bundles_num, NUM_REGRESSION, BASIS_ORDER))
        regression_unknown = regression_variable(basis.shape[0], riskfree_price, \
                                                 riskfree_delta, adjusted_price, adjusted_delta)

        # Collect all the basis related values in this time step and put them in order
        sorted_basis = basis[order_p, :, -j-1]
        sorted_expect_basis = expect_basis[order_p, :, -j-1]
        sorted_expect_brownian_basis = expect_brownian_basis[order_p, :, :, -j-1]
        sorted_regression_unknown = regression_unknown[order_p, :]

        try:
            # Distribute the relavent information into different bundle and run the regression 
            bundle_range = np.empty((c_bundles_num, 2), dtype=int)
            for i in range(c_bundles_num):
                bundle_range[i] = \
                np.array([np.sum(bundle_size[-j-2, 0:i]), np.sum(bundle_size[-j-2, 0:i+1])])
            for i in range(c_bundles_num):
                bundle_regression_unknown = \
                sorted_regression_unknown[bundle_range[i, 0]: bundle_range[i, 1]]
                bundle_basis = sorted_basis[bundle_range[i, 0]: bundle_range[i, 1]]
                regression_coeff[i] = regressioninbundle(bundle_basis, \
                                   bundle_regression_unknown)
        except ExceedBoundError:
            raise ExceedBoundError

        # Calculate the previous value based on problem specific numerical scheme
        theta_parameters = np.array([0, 1]) # Adjustment parameters for the integration approximation 
        delta_t = timespan[-j-1]-timespan[-j-2]
        for i in range(c_bundles_num):
            bundle_expect_basis = sorted_expect_basis[bundle_range[i, 0]: bundle_range[i, 1]]
            bundle_expect_brownian_basis = \
            sorted_expect_brownian_basis[bundle_range[i, 0]: bundle_range[i, 1]]
            riskfree_delta[order_p[bundle_range[i, 0]: bundle_range[i, 1]], :], \
            riskfree_price[order_p[bundle_range[i, 0]: bundle_range[i, 1]], :], \
            adjusted_delta[order_p[bundle_range[i, 0]: bundle_range[i, 1]], :], \
            adjusted_price[order_p[bundle_range[i, 0]: bundle_range[i, 1]], :] = \
            numerical_scheme(theta_parameters, delta_t, regression_coeff[i], \
                                             bundle_expect_basis, bundle_expect_brownian_basis)


    p_bar.update(num_of_time_points-1)

    return (np.mean(riskfree_price[:, :]), np.mean(adjusted_price[:, :]))

def regressioninbundle(basis, regression_unknown):
    """ Loop over all regression unknown and call linear regression function

    For each regression target, this function calls a linear regression algorithm 
    built from numpy package, passes on the target vector and basis function matrix and
    collects the regression coefficients vector.

    :param basis: Array of the regression basis for each path
    :param regression_unkwown: Array of all functions that we have to perfrom regression on.
    :type basis: (No. of samples) x Q array
    :type regression_unkwown: (No. of samples) x (No. of total regression) array
    :returns: Regression coeffcients for each tarhet function
    :rtype: (No. of total regression) x Q array
    """
    # Arrays are allocated for storing  regression coefficient
    regression_coeff = np.zeros((NUM_REGRESSION, BASIS_ORDER))

    try:
        # Perform the regressions
        for i in range(NUM_REGRESSION):
            leastsquaresregression(basis, regression_coeff[i], \
                                   regression_unknown[:, i].reshape(-1, 1))
    except ExceedBoundError:
        raise ExceedBoundError()

    return regression_coeff

def leastsquaresregression(basis, coeff, samples):
    """Perform standard least square regression and raise error when norm exceeds bound 

    Find x that solve Ax = y with least-square regression similar to the technique used 
    in Matlab and raise an error if the L2 norm of the regression coefficient exceeds a
    pre-defined bound.

    :param basis: Array of the regression basis for each path
    :param coeff: Array to hold the regression coefficient 
    :param samples: Array of the values of target function at sample points
    :type basis: (No. of samples) x Q array
    :type coeff: Q array
    :type samples:  (No. of samples) x 1 array
    :returns coeff: 
    :rtype: Q array
    """
    basis = np.matrix(basis)
    samples = np.matrix(samples)
    coeff[:] = np.linalg.solve(np.dot(basis.getH(), basis), 
         np.dot(basis.getH(), samples)).reshape(-1)  # x = (A^T y)/(A^T A)
    if np.dot(coeff.reshape(1, -1), coeff) >= 1e+15:
        print('The approximate exceeds pre-set bound.')
        raise ExceedBoundError()
    return coeff

def success_adjusted_std(data, mean, test_number, success_number):
    """Calculate the standard deviation by only taking into account the successful run"""
    std = np.sum((data - mean) ** 2)
    std -= (test_number - success_number) * mean ** 2
    std = np.sqrt(std / success_number)
    return std

if __name__ == '__main__':
    print(numba_sgbm_xva(numba_monte_carlo, 2**15, 20, 2*7, 5))
