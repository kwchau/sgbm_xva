"""
This is the main file for the stochastic grid bundling method for XVA demonstrator making use of
CUDA Python. It contains two functions. The cuda_sgbm_xva function is the main body of the whole 
algorithm, while cuda_sgbminnerblock is the main part of the backward process. This program works
by calling the preset examples and model dependent function (especially the forward process
generating function), performing the SGBM algorithm and returing the solution, error measurment 
and execution time.
"""

import time
import logging
from progress_bar import ProgressBar

import numpy as np
from pyculib.sorting import RadixSort
from cudafunctions import cuda_regression

from example_xva_v2 import Example
from cuda_blackscholes import cuda_jit_montecarlo

logging.getLogger()

# Example related parameters are imported here as golbal variables.
# The usage of golbal variable is due to local variable cannot be used to assign GPU memory.
# The rest of the data is imported in the same way for consistency.
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

def cuda_sgbm_xva(montecarlo, num_of_paths, num_of_steps, bundles, test_number=10):
    """Perform the SGBM algorithm.

    Under given parameters (number of total simulations, number of time step in partition
    and number of bundels in each time step), we run the sgbm algorithm for a number
    of time given by test_number. Other inputs including the (model dependent) forward
    simulation function. Each run consists of three parts, a forward simulation scheme
    to gather all necessary data, a partition procedure to separate data into "bundles"
    and a typical linear regression function with QR decomposition within each
    partition. It prints out the simulation result and the running time of each test.

    :param montecarlo: Forward simulation function
    :param num_of_paths: Number of stocks simulations in the Monte Carlo run
    :param num_of_steps: Numder of time step in time discretization
    :param bundles: Number of bundles in each time step.
    :param test_number: The number of test run for each set of parameters.
    :type montecarlo: function
    :type num_of_paths: int
    :type num_of_steps: int
    :type bundles: int
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
    # Time partition for discretization
    timespan = np.arange(0, TERMINAL_TIME + TERMINAL_TIME/num_of_steps,
                         TERMINAL_TIME/num_of_steps)

    for test in range(int(test_number)):
        logging.debug("Forward Simulation")
        start_time = time.time()

        #Forward simulation for all necessary information
        sorting_value, basis, expect_basis, expect_brownian_basis,\
                riskfree_price, riskfree_delta, adjusted_price, adjusted_delta =\
                montecarlo(timespan, num_of_paths)

        middle_time = time.time()

        # Function is called for the backward step, partition and regression
        logging.debug("Backward Progression")
        result1, result2 = \
        cuda_sgbminnerblock(sorting_value, basis, expect_basis, expect_brownian_basis,
                            riskfree_price, riskfree_delta, adjusted_price,
                            adjusted_delta, timespan, bundles=bundles)

        # Here we print out the riskfree price, risk adjusted price and full run time at each run
        print("Risk-free option price", result1)
        print("Risk-adjusted option price:", result2)
        result_risk_free[test] = result1
        result_risk_adjusted[test] = result2

        end_time = time.time()
        print('Forward time:', middle_time - start_time, ', Backward time:', end_time - middle_time)
        test_time[test] = end_time - start_time

    # Calculate the average riskfree price, risk adjusted price and full run time over all runs
    risk_free_price = np.average(result_risk_free)
    risk_adjusted_price = np.average(result_risk_adjusted)
    average_time = np.average(test_time)

    # Calculate the error and the standard deviation of our approximation
    if REFERENCE_RISKFREE:
        error_risk_free = abs(np.average(result_risk_free) - REFERENCE_RISKFREE_PRICE)
        std_risk_free = \
        np.sqrt(np.average((result_risk_free - REFERENCE_RISKFREE_PRICE) ** 2))
    else:
        error_risk_free = np.nan
        std_risk_free = np.std(result_risk_free)

    if not REFERENCE_ADJUST:
        error_risk_adjusted = np.nan
        std_risk_adjusted = np.std(result_risk_adjusted)

    return risk_free_price, error_risk_free, std_risk_free, \
           risk_adjusted_price, error_risk_adjusted, std_risk_adjusted, \
           average_time

def cuda_sgbminnerblock(sorting_value, basis, expect_basis, expect_brownian_basis,
                   riskfree_price, riskfree_delta, adjusted_price, adjusted_delta,
                   timespan, bundles):
    """Perform the SGBM backward algorithm with given forward simulations

    This algorithm collects all the necessary data from the forward simulation function and
    then perform the backward step sequentially with respect to the time discretization, starting
    from the last time period. Within each time period, first all samples are sorted according to
    the sorting value and partition them into bundles containing simular sorting value paths.
    In each bundle, several regressions are completed and the resulting coefficients are used to
    calculate the target value (option prices) within each bundle. Finally, all the results are
    collected and a new cycle begins, untill we ratch time zero.

    :param sorting_value: Array of the values of sorting function
    :param basis: Array of the regression basis for each path at every time step
    :param expect_basis: Array of the expectation of the regression basis for each path at
                         every time step
    :param expect_brownian_basis: Array of the expectation of the regression basis multipled by
                                  a Brownian motion at each dimension for each path at every time step
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
        bundle_size[-j-2, 0:num_of_samples - \
                    c_bundles_num*int(num_of_samples / c_bundles_num)] += 1

        # Sort the prices and deltas based on the order of sorting value
        sort = RadixSort(num_of_samples, dtype=np.double)
        current_sorting_value = np.ascontiguousarray(sorting_value[:, -j-2])
        order_p = sort.argsort(current_sorting_value)

        # Collect all the values that we have to perform regression on
        regression_coeff = \
        np.zeros((c_bundles_num, NUM_REGRESSION, BASIS_ORDER))
        regression_unknown = regression_variable(basis.shape[0], riskfree_price, riskfree_delta,
                                                 adjusted_price, adjusted_delta)

        # Collect all the basis related values in this time step and put them in order
        sorted_basis = basis[order_p, :, -j-1]
        sorted_expect_basis = expect_basis[order_p, :, -j-1]
        sorted_expect_brownian_basis = expect_brownian_basis[order_p, :, :, -j-1]
        sorted_regression_unknown = regression_unknown[order_p, :]

        # Distribute the relavent information into different bundle and run the regression 
        # in parallel
        bundle_range = np.empty((c_bundles_num, 2), dtype=int)
        for i in range(c_bundles_num):
            bundle_range[i] = \
            np.array([np.sum(bundle_size[-j-2, 0:i]), \
                      np.sum(bundle_size[-j-2, 0:i+1])])
        cuda_regression(BASIS_ORDER, NUM_REGRESSION, c_bundles_num, bundle_range, \
                        sorted_regression_unknown, sorted_basis, regression_coeff)

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

if __name__ == '__main__':
    print(cuda_sgbm_xva(cuda_jit_montecarlo, 2**15, 20, 2 ** 7))
