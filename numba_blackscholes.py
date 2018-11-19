"""
The forward process models, forward simulations, terminal condition for the backward system, bundling and regression basis can be customized 
by user for various BSDE problems. Here are the options than have been implemented in this version.
"""
import numpy as np
from  itertools import accumulate
from scipy.special import binom
from numba import float64, guvectorize, jit

from mathfunctions import partitionfunc, diffusionproduct

from example_xva_v2 import Example

#  Example related parameters are imported here as golbal variables.
#  The data is imported in this way for the consistency with the CUDA Python
#  program.
example = Example
# The following parameters are related to stock model.
INITIAL_VALUE =  example.initial_value
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
@jit
def numba_black_scholes_drift(x, time):
    """Calculate Black-Scholes drift"""
    return MU_BAR * x

@jit
def numba_black_scholes_volatility(x, time):
    """Calculate Black-Scholes volatility"""
    return np.dot(np.diag(SIGMA_BAR * x), CHOLESKY_DECOMPOSITION)

#  The Euler scheme for stochastic differential equation.
#  This should be moved to mathfunctions.
@guvectorize([(float64[:], float64, float64[:], float64[:, :], float64[:], float64[:])], '(n), (), (n), (n, m), (m)->(n)')
def numba_step_euler(last, dt, drift, volatility, noise, res):
    """u-funcs for Approximating SDE in one time step with Euler scheme"""
    random_part = np.dot(volatility, noise) * np.sqrt(dt)
    for i in range(last.shape[0]):
        res[i] = last[i] + drift[i] * dt + random_part[i]

# The following are basis-related functions.
@guvectorize([(float64[:, :],  float64[:])], '(n, m)->(n)')
def numba_intrinsicvalue(x, res):
    """Calculate the arithmetic weight average of stock prices"""
    for i in range(x.shape[0]):
        res[i] =  np.dot(WEIGHT, x[i])

@jit
def numba_intrisic_value_basis(x, order_of_basis):
    """Calculate the regression basis based on arithmetic weight average
    
    This function calculates the arithmetic weighted average of stocks to power 0, 1
    or 2. The resulting values are used as the basis for linear regression.
    
    :param x: Array of stock
    :param order_of_basis: The order of basis with first order being 0 power
    :type x: N x d array
    :type order_of_basis: int
    :returns: basisvalue
    :rtype: N x order_of_basis array
    """
    basisvalue = np.empty((x.shape[0], order_of_basis))
    basisvalue[:, 0] = 1.
    weighted_x = x * WEIGHT
    if order_of_basis > 1:
        for k in range(1, order_of_basis):
            term = np.zeros(x.shape[0])
            combination =  partitionfunc(k, x.shape[1])
            for l in combination:
                coeff = 1.0
                for zipped in zip(accumulate(l.values()), l.values()):
                    coeff *= binom(*zipped)
                value = np.ones(x.shape[0])
                for key in l:
                    value *= weighted_x[:, key-1] ** l[key]
                term += coeff * value
            basisvalue[:, k] = term
    return basisvalue

@jit
def arithemtic_basket_moment(x, time, delta_t, order):
    """Calculate the expectation of arithmetic weighted average using the geometric mean
    
    This functions calculates the expectation of the arithmetrtic weighted average by using 
    the multinomial theorem. For each possible distribution of degree among all available dimensions,
    the code calculates the cooresponding multinomial coefficient and calling the geometric 
    mean function. Finaly the result is aggregated.
    """
    if order == 0:
        return np.ones(x.shape[0])
    combination = partitionfunc(order, NUM_OF_ASSETS)
    rec = np.zeros(x.shape[0])
    for d in combination:
        coeff = 1.0
        for key in d:
            coeff *= WEIGHT[key-1] ** d[key]
        for zipped in zip(accumulate(d.values()), d.values()):
            coeff *= binom(*zipped)
        rec += coeff * diffusionproduct(numba_black_scholes_drift, numba_black_scholes_volatility, order, d, x, time, delta_t)
    return rec

@jit
def numba_basis_expect(x, time, delta_t, order_of_basis):
    """Calculate the expectation of arithmetic weight average basis"""
    basisvalue = np.zeros((x.shape[0], order_of_basis))
    basisvalue[:, 0] = 1.
    if order_of_basis > 1:
        for k in range(1, order_of_basis):
            basisvalue[:, k] = arithemtic_basket_moment(x, time, delta_t, k)
    return basisvalue

@jit
def numba_cuda_basis_brownian_expect(x, time, delta_t, order_of_basis):
    """Calculate the expectation of arithmetic weight average multiplied by brownian motion"""
    basisvalue = np.zeros((x.shape[0], order_of_basis, NUM_OF_BROWNIAN_MOTION))
    basisvalue[:, 0, :] = 0.
    if order_of_basis > 1:
        for k in range(1, order_of_basis):
            temp = np.empty((x.shape[0], NUM_OF_BROWNIAN_MOTION))
            for i in range(x.shape[0]):
                temp[i] = np.dot(WEIGHT, numba_black_scholes_volatility(x[i], time))
            basisvalue[:, k, :] = k * np.einsum('i, ij -> ij', arithemtic_basket_moment(x, time, delta_t, k-1), temp)
    return basisvalue

@guvectorize([(float64[:, :], float64[:])], '(n, m) ->(n)')
def numba_payoff(x, res):
    """ Calculate the payoff of an arithmetic basket put option"""
    for i in range(x.shape[0]):
        res[i] =  np.maximum(STRIKE - np.dot(WEIGHT, x[i]), 0.0)

@jit
def numba_gradient(x):
    """ Calculate the gradiate of the payoff of an arithmetic basket put option"""
    temp = np.maximum(STRIKE - np.dot(WEIGHT, x), 0.0)
    return  WEIGHT * (temp > 0)

@guvectorize([(float64[:, :], float64[:], float64[:, :])], '(n, m), ()->(n, m)')
def numba_terminal_delta(x, time, res):
    """ Calculate the terminal Z of the payoff of an arithmetic basket put option"""
    for i in range(x.shape[0]):
        res[i] =  -np.dot(numba_gradient(x[i]) , numba_black_scholes_volatility(x[i], time))

def numba_monte_carlo(timespan, n, paths=False):
    """Compute the Monte Carlo scenarios with Euler approximation
    """
    no_of_timesteps = timespan.shape[0]
    stock_data = np.ones((n, 1)) * INITIAL_VALUE
    if paths == True:
        stock = np.empty((n, NUM_OF_ASSETS, no_of_timesteps))
    sorting_value = np.empty((n, no_of_timesteps))
    basis = np.empty((n, BASIS_ORDER, no_of_timesteps-1))
    expect_basis = np.empty((n, BASIS_ORDER, no_of_timesteps-1))
    expect_brownian_basis = \
    np.empty((n, BASIS_ORDER, NUM_OF_BROWNIAN_MOTION , no_of_timesteps-1))

    for k in range(no_of_timesteps):
        noises = np.random.normal(0., 1., (n, NUM_OF_BROWNIAN_MOTION))
        current_time = timespan[k]
        if k != 0:
            previous_time = timespan[k-1]
            dt = current_time - previous_time
        if k != no_of_timesteps - 1:
            next_time = timespan[k+1]
            forward_time_interval = next_time - current_time
        if k != 0:
            for i in range(n):
                stock_data[i] = \
                numba_step_euler(stock_data[i], dt, \
                                 numba_black_scholes_drift(stock_data[i], previous_time), \
                              numba_black_scholes_volatility(stock_data[i], previous_time), noises[i])
            basis[:, :, k-1] = numba_intrisic_value_basis(stock_data, BASIS_ORDER)
        if k != no_of_timesteps-1:
            expect_basis[:, :, k] = \
            numba_basis_expect(stock_data, current_time, \
                                                  forward_time_interval, BASIS_ORDER)
            expect_brownian_basis[:, :, :, k] = \
            numba_cuda_basis_brownian_expect(stock_data, \
                                                                current_time, \
                                                                forward_time_interval, \
                                                                BASIS_ORDER)
        if paths == True:
            stock[:, :, k] = stock_data
        sorting_value[:, k] = numba_intrinsicvalue(stock_data)

    # Construct array to hold the backward process and initialization
    riskfree_price = np.zeros((n, 1))
    riskfree_delta = np.zeros((n, NUM_OF_BROWNIAN_MOTION))
    adjusted_price = np.zeros((n, 1))
    adjusted_delta = np.zeros((n, NUM_OF_BROWNIAN_MOTION))
    riskfree_price[:, :] = BUY_SELL * numba_payoff(stock_data).reshape(-1, 1)
    riskfree_delta[:, :] = BUY_SELL * numba_terminal_delta(stock_data, timespan[-1])
    adjusted_price[:, :] = BUY_SELL * numba_payoff(stock_data).reshape(-1, 1)
    adjusted_delta[:, :] = BUY_SELL * numba_terminal_delta(stock_data, timespan[-1])

    if paths == True:
        return stock, sorting_value, basis, expect_basis, expect_brownian_basis, \
               riskfree_price, riskfree_delta, adjusted_price, adjusted_delta
    return sorting_value, basis, expect_basis, expect_brownian_basis, \
           riskfree_price, riskfree_delta, adjusted_price, adjusted_delta
