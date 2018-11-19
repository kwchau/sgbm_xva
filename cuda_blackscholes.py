""" Contains all models specific functions for CUDA Python XVA solver

This file acts as the container for all the question specific function for CUDA
Python SGBM program, namely the forward simulation for the stock models,
thespecific sorting function for bundling and regression basis with its expectation.

.. todo:: Adding the geometric basket payoff and basis
"""
import time
import math
import numpy as np
from numba import cuda, int32, float64
from pyculib import rand

from mathfunctions import star_and_bin_array
from cudafunctions import cuda_jit_step_euler

from example_xva_v2 import Example

# Example related parameters are imported here as golbal variables.
# The usage of golbal variable is due to local variable cannot be used to assign GPU memory.
# The rest of the data is imported in the same way for consistency.
example = Example
# The following parameters are related to stock model.
INITIAL_VALUE = example.initial_value
NUM_OF_ASSETS = example.num_of_assets
NUM_OF_BROWNIAN_MOTION = example.num_of_brownian_motion
MU_BAR = example.mu_bar
SIGMA_BAR = example.sigma_bar
CHOLESKY_DECOMPOSITION = example.cholesky_decomposition
# The following parameters are related to the financial product.
WEIGHT = example.weight
STRIKE = example.strike
BUY_SELL = example.buy_sell
# The following parameters are related to the regression basis.
BASIS_ORDER = example.basis_order

# The followings are stock-related function.
@cuda.jit
def cuda_jit_black_scholes_drift(d_stock, mu_bar, d_drift):
    """Calculate Black-Scholes drift in parallel with GPU"""
    i = cuda.grid(1)
    if i < d_drift.shape[0]:
        for j in range(d_drift.shape[1]):
            d_drift[i, j] = d_stock[i, j] * mu_bar[j]

@cuda.jit
def cuda_jit_black_scholes_volatility(d_stock, sigma_bar, cholesky_decomposition, d_volatility):
    """Calculate Black-Scholes volatility in parallel with GPU"""
    i = cuda.grid(1)
    if i < d_volatility.shape[0]:
        for j in range(d_volatility.shape[1]):
            for k in range(d_volatility.shape[2]):
                d_volatility[i, j, k] = sigma_bar[j]*d_stock[i, j]*cholesky_decomposition[j, k]

# The following are basis-related functions.
@cuda.jit
def cuda_intrinsicvalue(weight, x, res):
    """Calculate the arithmetic weight average of stock prices in parallel with GPU"""
    i = cuda.grid(1)
    if i < x.shape[0]:
        res[i] = 0
        for j in range(x.shape[1]):
            res[i] += weight[j] * x[i, j]

@cuda.jit
def cuda_jit_intrisic_value_basis(weight_stock, basis_partition, degree, basis):
    """Calculate the regression basis based on arithmetic weighted average in parallel with GPU
    
    This function calculates the arithmetic weighted average of stocks to power 1
    or 2. The resulting values are used as the basis for linear regression.
    
    :param weight_stock: Array of stock value multipied by weight.
    :param basis_partition: Array of all possible distribution of a given degree in over
    the stock dimension
    :param degree: The power to raise the weighted sum to
    :param basis: Array to hold the resulting value
    :type weight_stock: N x q array
    :type basis_partition: (q + degree -1 C degree) x q array
    :type degree: int
    :type basis: N array
    :returns: basis
    :rtype: N array
    """
    i = cuda.grid(1)
    if i < weight_stock.shape[0]:
        temp = cuda.local.array(1, dtype=float64)
        basis[i] = 0
        for j in range(basis_partition.shape[0]):
            temp[0] = math.gamma(degree + 1.)
            for k in range(basis_partition.shape[1]):
                temp[0] *= weight_stock[i, k] ** basis_partition[j, k] \
                /math.gamma(basis_partition[j, k] +1.)
            basis[i] += temp[0]

@cuda.jit
def cuda_weighted_stock(weight, x, res):
    """Calculate the stock multiplied by weight in parallel with GPU"""
    i = cuda.grid(1)
    if i < x.shape[0]:
        for j in range(x.shape[1]):
            res[i, j] = weight[j] * x[i, j]

@cuda.jit
def cuda_intrisic_value_basis_expect(WEIGHT, x, time, mu, sigma, delta_t, order,
                                     combination, basis_expectation):
    """Calculate the expectation of arithmetic weight average basis in parallel with GPU"""
    i = cuda.grid(1)
    if i < x.shape[0]:
        basis_expectation[i] = arithemticbasketmoment(WEIGHT, x[i], time, mu[i], sigma[i],
                         delta_t, order, combination, basis_expectation[i])

@cuda.jit
def cuda_intrisic_value_basis_brownian_expect(
        weight, x, time, mu, sigma, delta_t, order, dimension, combination,
        basis_brownian_expectation):
    """Calculate the expectation of arithmetic weight average multiplied by brownian motion"""
    i = cuda.grid(1)
    if i < x.shape[0]:
        temp = cuda.local.array(2, dtype=float64)
        temp[0] = 0
        for j in range(weight.shape[0]):
            temp[0] += weight[j] * sigma[i, order, j] # From the derivative and Ito of the basis
        temp[1] = arithemticbasketmoment(weight, x[i], time, mu[i], sigma[i], \
            delta_t, order-1, combination, temp[1])
        basis_brownian_expectation[i] = order * temp[1] * temp[0]

@cuda.jit(device=True)
def arithemticbasketmoment(WEIGHT, x, time, mu, sigma, delta_t, order, combination, rec):
    """Calculate the expectation of arithmetic weight average using the geometric mean
    
    This functions calculates the expectation of the arithmetrtic weight average by using 
    the multinomial theorem. For each possible distribution of degree in dimensions, the 
    code calculates the cooresponding multinomial coefficient and calling the geometric 
    mean function. Finaaly the result is aggregated.
    """
    if order == 0:
        return 0
    rec = 0.
    coeff = cuda.local.array(1, dtype=float64)
    for i in range(combination.shape[0]):
        coeff[0] = math.gamma(order + 1.)
        for j in range(combination.shape[1]):
            coeff[0] *= WEIGHT[j] ** combination[i, j] /math.gamma(combination[i, j] + 1.) 
            # multinomial coefficients times weight
        temp = cuda.local.array(1, dtype=float64)
        temp = cuda_diffusion_product(mu, sigma, order, combination[i], x, time, delta_t, temp)
        rec += coeff[0] * temp[0]
    return rec

# This function may be moved to cudafunctions
@cuda.jit(device=True)
def cuda_diffusion_product(mu, sigma, order, d, x, time, delta_t, temp):
    """ Calculate the expectations of product of a diffusion process 
    
    This function calculates the expectations of product (up to second order) of 
    the Euler approximation of a diffusion process in one time step.
    """
    temp[0] = 1
    if order == 1:
        # Result for first order
        for i in range(d.shape[0]):
            temp[0] *= (x[i] + mu[i] * delta_t) ** d[i]
        return temp
    elif order == 2:
        iterator = cuda.local.array(1, dtype=int32)
        iterator[0] = 0
        data = cuda.local.array(2, dtype=int32)
        # Here the active dimensions are determinated.
        for i in range(d.shape[0]):
            if d[i] > 0:
                for j in range(d[i]):
                    data[iterator[0]] = i
                    iterator[0] += 1
                if iterator[0] > 1:
                    break
        # Formula for second order 
        temp[0] = x[data[0]] * x[data[1]] + x[data[1]] * mu[data[0]] * delta_t\
        +  x[data[0]] * mu[data[1]] * delta_t + mu[data[0]]  * mu[data[1]] * delta_t **2
        for k in range(sigma.shape[1]):
            temp[0] += sigma[data[0], k] * sigma[data[1], k] * delta_t
        return temp

# Financial product/Payoff-related functions
@cuda.jit
def cuda_put_payoff(x, weight, strike, res):
    """ Calculate the payoff of an arithmetic basket put option"""
    i = cuda.grid(1)
    if i < x.shape[0]:
        temp = cuda.local.array(1, dtype=float64)
        temp[0] = 0.
        for j in range(weight.shape[0]):
            temp[0] += weight[j] * x[i, j]
        res[i, 0] = max(strike - temp[0], 0.0)

@cuda.jit
def cuda_put_terminal_delta(x, weight, strike, volatility, res):
    """ Calculate the terminal Z of the payoff of an arithmetic basket put option"""
    i = cuda.grid(1)
    if i < x.shape[0]:
        temp = cuda.local.array(2, dtype=float64)
        temp[0] = 0.
        temp[1] = 0.
        for k in range(weight.shape[0]):
            temp[1] += weight[k] * x[i, k]
        for j in range(volatility.shape[2]):
            res[i, j] = 0
            for l in range(volatility.shape[1]):
                res[i, j] -= weight[l] * (max((strike - temp[1], 0.0)) > 0) * volatility[i, l, j]

def cuda_jit_montecarlo(timespan, n): 
    """Compute the Monte Carlo scenarios

    :param timespan: Array of the time discretization
    :param n: Number of Monte Carlo paths
    :type timespan: M+1 array
    :type n: int
    :returns: (sorting_value, basis, expect_basis, expect_brownian_basis, riskfree_price,
              riskfree_delta, adjusted_price, adjusted_delta)
    :rtypes: (N x (M+1) array, N x Q x M array, N x Q x M array, N x Q x q x M array,
    N x 1 array, N x q array, Nx 1 array, N x q array)
    """
    no_of_timesteps = timespan.shape[0] # M+1
    stock_data = np.ones((n, 1)) * INITIAL_VALUE
    sorting_value = np.empty((n, no_of_timesteps))
    basis = np.empty((n, BASIS_ORDER, no_of_timesteps-1))
    expect_basis = np.empty((n, BASIS_ORDER, no_of_timesteps-1))
    expect_brownian_basis = np.empty((n, BASIS_ORDER, NUM_OF_BROWNIAN_MOTION, no_of_timesteps-1))
    riskfree_price = np.zeros((n, 1))
    riskfree_delta = np.zeros((n, NUM_OF_BROWNIAN_MOTION))
    adjusted_price = np.zeros((n, 1))
    adjusted_delta = np.zeros((n, NUM_OF_BROWNIAN_MOTION))

    blksz = 256
    gridsz = int(math.ceil(float(n) / blksz))

    # instantiate a CUDA stream for queueing async CUDA cmds
    stream = cuda.stream()

    # instantiate a cuRAND PRNG
    prng = rand.PRNG(rndtype=rand.PRNG.MRG32K3A, seed=int(time.time()), stream=stream)

    # Allocate device side array
    d_normdist = cuda.device_array(n * NUM_OF_BROWNIAN_MOTION, dtype=np.double, stream=stream)
    d_drift = cuda.device_array((n, NUM_OF_ASSETS), dtype=np.double, stream=stream)
    d_volatility = \
    cuda.device_array((n, NUM_OF_ASSETS, NUM_OF_BROWNIAN_MOTION), dtype=np.double, stream=stream)
    d_sorting_value = cuda.device_array_like(sorting_value[:, 0], stream=stream)
    d_weighted_stock = cuda.device_array_like(stock_data, stream=stream)
    d_basis = cuda.device_array(n, dtype=np.double, stream=stream)
    d_basis_expectation = cuda.device_array(n, dtype=np.double, stream=stream)
    d_basis_brownian_expectation = cuda.device_array(n, dtype=np.double, stream=stream)
    d_price = cuda.device_array((n, 1), dtype=np.double, stream=stream)
    d_delta = cuda.device_array((n, NUM_OF_BROWNIAN_MOTION), dtype=np.double, stream=stream)

    step_cfg = cuda_jit_step_euler[gridsz, blksz, stream] # The forward simulation scheme

    for k in range(no_of_timesteps):
        # Calculate the time interval
        current_time = timespan[k]
        if k != 0:
            previous_time = timespan[k-1]
            dt = current_time - previous_time
        if k != no_of_timesteps - 1:
            next_time = timespan[k+1]
            forward_time_interval = next_time - current_time

        if k == 0:
            # transfer the initial prices
            d_last = cuda.to_device(stock_data, stream=stream)
        else:
            # call cuRAND to populate d_normdist with gaussian noises
            prng.normal(d_normdist, mean=0, sigma=1)
            # invoke step kernel asynchronously
            step_cfg(d_last, dt, d_drift, d_volatility, d_normdist)

        # Calculate the stock model parameters.
        cuda_jit_drift[gridsz, blksz, stream](d_last, MU_BAR, d_drift)
        cuda_jit_volatility[gridsz, blksz, stream](d_last, SIGMA_BAR,
                           CHOLESKY_DECOMPOSITION, d_volatility)

        # Calculate the sorting value
        cuda_sorting_method[gridsz, blksz, stream](WEIGHT, d_last, d_sorting_value)
        sorting_value[:, k] = d_sorting_value.copy_to_host()

        # Calculate the basis function and its two types of expectations
        if k != 0:
            cuda_weighted_stock[gridsz, blksz, stream](WEIGHT, d_last, d_weighted_stock)
            for j in range(BASIS_ORDER):
                basis_partition = star_and_bin_array(j, NUM_OF_ASSETS)
                cuda_jit_basis[gridsz, blksz, stream](d_weighted_stock, basis_partition, j, d_basis)
                basis[:, j, k-1] = d_basis.copy_to_host()
        if k != no_of_timesteps-1:
            expect_basis[:, 0, k] = 1.
            expect_brownian_basis[:, 0, :, k] = 0.
            for j in range(1, BASIS_ORDER):
                basis_partition = star_and_bin_array(j, NUM_OF_ASSETS)
                basis_gradient_partition = star_and_bin_array(j-1, NUM_OF_ASSETS)
                cuda_basis_expect[gridsz, blksz, stream](WEIGHT, d_last, current_time, d_drift, d_volatility, forward_time_interval, j, basis_partition, d_basis_expectation)
                expect_basis[:, j, k] = d_basis_expectation.copy_to_host()
                for l in range(d_volatility.shape[2]):
                    cuda_basis_brownian_expect[gridsz, blksz, stream](WEIGHT, d_last, current_time, d_drift, d_volatility, forward_time_interval, j, l, basis_gradient_partition, d_basis_brownian_expectation)
                    expect_brownian_basis[:, j, l, k] = d_basis_brownian_expectation.copy_to_host()

        stream.synchronize()   # This ensure all GPU work is completed

    #  Calculate the termianl conditions
    cuda_payoff[gridsz, blksz, stream](d_last, WEIGHT, STRIKE, d_price)
    cuda_terminal_delta[gridsz, blksz, stream](d_last, WEIGHT, STRIKE, d_volatility, d_delta)
    riskfree_price[:, :] = d_price.copy_to_host()
    riskfree_price *= BUY_SELL
    adjusted_price[:,:] = d_price.copy_to_host()
    adjusted_price *= BUY_SELL
    riskfree_delta[:, :] = d_delta.copy_to_host()
    riskfree_delta *= BUY_SELL
    adjusted_delta[:,:] = d_delta.copy_to_host()
    adjusted_delta *= BUY_SELL

    return sorting_value, basis, expect_basis, expect_brownian_basis, riskfree_price, riskfree_delta, adjusted_price, adjusted_delta

# The following is the functions currently under development for the geometric basket option.
    
# The following functions are related to the regression basis.
@cuda.jit
def cuda_geometric_value(weight, x, res):
    """Calculate the weighted geometric average of stock prices in parallel with GPU"""
    i = cuda.grid(1)
    if i < x.shape[0]:
        res[i] = 1.
        for j in range(x.shape[1]):
            res[i] *= x[i, j] ** weight[j]
            
@cuda.jit
def cuda_geometric_value_basis(stock, basis_partition, degree, basis):
    """Calculate the geometric average of stock prices to a given order in parallel with GPU"""
    i = cuda.grid(1)
    if i < stock.shape[0]:
        basis[i] = 1
        for j in range(stock.shape[1]):
            basis[i] *= stock[j] ** (degree)
            
@cuda.jit
def cuda_geometric_value_basis_expect(WEIGHT, x, time, mu, sigma, delta_t, order,
                                     combination, basis_expectation):
    """Calculate the expectation of geometric average basis in parallel with GPU"""
    i = cuda.grid(1)
    if i < x.shape[0]:
        #  Calculate the new drift and volitility of geometric average based on Euler
        #  approximation and Ito formula
        product = cuda.local.array(1, dtype=float64)
        new_mu = cuda.local.array(1, dtype=float64)
        new_sigma = cuda.local.array(NUM_OF_BROWNIAN_MOTION, dtype=float64)
        temp = cuda.local.array(1, dtype=float64)
        product[0] = 1.
        new_mu[0] = 0.
        for j in range(x.shape[1]):
            product *= x[i, j]
        for j in range(x.shape[1]):
            new_mu[0] += mu[i]/x[i, j]
            temp[0] = 0
            for k in range(sigma.shape[2]):
                temp[0] += sigma[i, j, k] * sigma[i, k, j]
            new_mu[0] -= 0.5 * temp/ (x[i, j] ** 2)
        temp[0] = 0.
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                for l in range(sigma.shape[2]):
                    temp[0] += sigma[i, j, l] * sigma[i, l, k] / (x[i, j] * x[i, k])
        new_mu[0] += temp[0]
        new_mu[0] *= product[0]
        for j in range(sigma.shape[2]):
            new_sigma[j] = 0.
            for k in range(x.shape[1]):
                new_sigma[j]+= sigma[i, k, j] / x[i, k]
            new_sigma[j] *= product[0]
        #  Call the function to calculate the product of the new process (prod X)^k
        basis_expectation[i] = cuda_diffusion_product(new_mu, new_sigma, order,
                         np.array([order]), product, time, delta_t, basis_expectation[i])
        
@cuda.jit
def cuda_geometric_value_basis_brownian_expect(
        weight, x, time, mu, sigma, delta_t, order, dimension, combination,
        basis_brownian_expectation):
    """Calculate the expectation of the geometric average basis multiplied by a Brownian motion"""
    i = cuda.grid(1)
    if i < x.shape[0]:
        base_value = cuda.local.array(1, dtype=float64)
        temp = cuda.local.array(2, dtype=float64)
        gradient = cuda.local.array(1, dtype=float64)
        basis_brownian_expectation[i] = 0.
        for j in range(x.shaspe[1]):
            # Calculate the current geometric average to the given power 
            base_value[0] = 1.
            for k in range(x.shape[1]):
                base_value[0] *= x[i, k] ** order
            base_value[0] /= x[i, j]
            # Calulation of the derivate of the basis wrt to dimension j
            gradient[0] = 1.
            temp[0] = 0
            for k in range(x.shape[1]):
                temp[0] += mu[i, k]/x[i, k]
                for l in range(sigma.shape[2]):
                    temp[0] -= 0.5 * sigma[i, k, l] * sigma[i, l, j] / x[i, k] **2
            for k in range(x.shape[1]):
                for l in range(x.shape[2]):
                    for m in range(sigma.shape[2]):
                        temp[0] += 0.5 * sigma[i, k, m] * sigma[i, m, l] / (x[i, k] * x[i, l])
            gradient[0] += order * temp[0] * delta_t
            gradient[0] -= mu[i,j] * delta_t/ x[i,j]
            temp[0] = 0.
            for k in range(sigma.shape[1]):
                temp[1] = 0.
                for l in range(sigma.shape[2]):
                    temp[1] += sigma[i, l, k]/x[i, l]
                temp[0] += temp[1]**2
            gradient[0] += order * (order - 1) * 0.5 * temp[0] * delta_t
            temp[0] = 0.
            for k in range(sigma.shape[1]):
                for l in range(sigma.shape[2]):
                    temp[0] += sigma[k,l]*sigma[j, l] / (x[i, k]*x[i, l])
            gradient[0] -= k * temp[0] * delta_t
            temp[0] = 0.
            for k in range(sigma.shape[2]):
                temp[0] += sigma[i, j, k]**2 / x[i, j]
            gradient[0] += temp[0] * delta_t
            gradient[0] *= base_value[0]
            basis_brownian_expectation[i] += gradient[0] * sigma[i, j, dimension]

# Switch for selecting functions for different models
if example.stock_model == "BS":
    cuda_jit_drift = cuda_jit_black_scholes_drift
    cuda_jit_volatility = cuda_jit_black_scholes_volatility
if example.sorting_method == "Intrinsic Value":
    cuda_sorting_method = cuda_intrinsicvalue
elif example.sorting_method == "Geometric Intrinsic Value":
    cuda_sorting_method = cuda_geometric_value
if example.basis == "Intrinsic Value":
    cuda_jit_basis = cuda_jit_intrisic_value_basis
    cuda_basis_expect = cuda_intrisic_value_basis_expect
    cuda_basis_brownian_expect = cuda_intrisic_value_basis_brownian_expect
elif example.basis == "Geometric Intrinsic Value":
    cuda_jit_basis =  cuda_geometric_value_basis
    cuda_basis_expect = cuda_geometric_value_basis_expect
    cuda_basis_brownian_expect = cuda_geometric_value_basis_brownian_expect
if example.put_call == "Put":
    cuda_payoff = cuda_put_payoff
    cuda_terminal_delta = cuda_put_terminal_delta
