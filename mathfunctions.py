"""
Created on Thu Nov  2 12:27:48 2017

@author: Ki Wai Chau
"""
import numpy as np
import scipy.stats as ss
from scipy.special import binom

def diffusionproduct(mu, sigma, order, d, x, time, delta_t):
    """Calculate the expectations of product of a diffusion process
    
    This function calculates the expectations of product (up to second order) 
    of the Euler approximation of a diffusion process in one time step

    :param mu: The drift function for the difussion
    :sigma sigma: The volitility function of the difussion
    :param order: The multiplication order
    :param d: The active dimension in the product.
    :param x: Currect state of the diffusion
    :param time: current time
    :param delta_t: The length of the time step under consideration
    :type mu: function
    :type sigma: function
    :type order: int
    :type d: dict
    :type x: Nxd array
    :type time: float
    :type delta_t: float
    :returns: Array of expectations for each starting position in x_0
    :rtype: N array
    """
    if order == 1:
        # Result for first order
        for key in d:
            rec = np.empty(x.shape[0])
            for i in range(x.shape[0]):
               rec[i] = x[i, key-1] + mu(x[i], time)[key-1] * delta_t
        return rec
    elif order == 2:
        factors = ()
        for key in d:
            for i in range(d[key]):
                factors = factors + (key-1,)
        rec = np.empty(x.shape[0])
        for i in range(x.shape[0]):
            temp = x[i, factors[0]] * mu(x[i], time)[factors[1]] \
            + x[i, factors[1]] * mu(x[i], time)[factors[0]] \
            + np.einsum('ij, kj->ik', sigma(x[i], time), sigma(x[i], time))[factors]\
            +  mu(x[i], time)[factors[0]] * mu(x[i], time)[factors[1]] * delta_t 
            rec[i] = delta_t * temp + x[i, factors[0]] * x[i, factors[1]] 
        return rec # Result for second order

def partitionfunc(degree, k, lowest_degree=0, key=1):
    '''  Calculate all possible distribution of multiplication power

    Multiplication power degree is distributed among the prescripted dimensions.
    The key input is use to track the iterations of applying the function as
    this function will be used recursively. This function returns a generator for
    dictionary.

    :param degree: The multiplication degree to be distributed
    :param k: The number of dimension for the distribution
    :param lowest_degree: The lowest possible degree for a dimension
    :param key: The current dimension under condsideration
    :type degree: int
    :type k: int
    :type l: int
    :type key: int
    :rtype: generator
    '''
    if k < 1:
        raise IndexError('The number of factors must be positive.')
    if k == 1:
        if (degree >= lowest_degree) and (degree != 0):
            yield {key:degree}
        elif degree == 0:
            yield {}
    if k > 1: 
        for i in range(lowest_degree, degree+1):
            for result in partitionfunc(degree-i, k-1, lowest_degree, key+1):
                if i != 0:
                    temp = {key:i, }
                else:
                    temp = {}
                temp.update(result)
                yield temp
                
def star_and_bin_array(star, bins):
    """  Calculate all possible distribution of multiplication power

    This function calls partitionfunc and returns the results in array instead.
    """
    i = 0
    temp = np.zeros((int(binom(star+bins-1, star)), bins), dtype=int)
    for combination in partitionfunc(star, bins):
        for key in combination:
            temp[i, key-1] = combination[key]
        i += 1
    return temp
                
def step_euler(last, dt, drift, volatility, noise):
    """Approximate SDE in one time step with Euler scheme"""
    return last + drift * dt + np.dot(volatility, noise)
        
def euler(x0, mu, sigma, TIMESPAN, random = None):
    """Approximate SDE in one time step with Euler scheme directly with model function"""
    N = len(TIMESPAN)
    q = len(x0)
    if random is None:
        random = np.random.normal(scale=1, size=(N, q))
    process = np.empty((q, N))
    process[:, 0] = x0
    for n in range(0, N-1):
        tn = TIMESPAN[n]
        xn = process[:, n]
        delta_n = TIMESPAN[n+1] - TIMESPAN[n]
        process[:, n+1] = step_euler(xn, delta_n, mu(xn, tn), sigma(xn, tn), np.sqrt(delta_n) * random[n])
    return process
            
#Black and Scholes
def d1(S0, K, r, sigma, T):
    return (np.log(S0/K) + (r + sigma**2 / 2) * T)/(sigma * np.sqrt(T))
 
def d2(S0, K, r, sigma, T):
    return (np.log(S0 / K) + (r - sigma**2 / 2) * T) / (sigma * np.sqrt(T))
 
def black_scholes(type,S0, K, r, sigma, T):
    """Calculate the 1-dimension Black-Scholes put/call price"""
    if type=="C":
        return S0 * ss.norm.cdf(d1(S0, K, r, sigma, T)) - K * np.exp(-r * T) * ss.norm.cdf(d2(S0, K, r, sigma, T))
    else:
       return K * np.exp(-r * T) * ss.norm.cdf(-d2(S0, K, r, sigma, T)) - S0 * ss.norm.cdf(-d1(S0, K, r, sigma, T))