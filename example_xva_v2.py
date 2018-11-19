"""
Created on Mon May 14 17:21:26 2018

@author: Ki Wai Chau

This file holds the parameters for all specific example and the numerical scheme
for the BSDE.

.. todo:: Moving the numerical scheme to the function file
"""
import numpy as np

class GermenIndexPut():
    # Example Information
    name = "Put option for a 5 stocks German index model"
    dimension = 5
    # Stock parameters
    stock_model = "BS"
    initial_value =  np.array([0.01, 0.01, 0.01, 0.01, 0.01])
    num_of_assets = 5
    num_of_brownian_motion = 5
    mu_bar = 0.05 * np.ones(num_of_assets, dtype=float)
    sigma_bar = np.array([0.518, 0.648, 0.623, 0.570, 0.530])
    cholesky_decomposition = np.array([[1., 0., 0., 0., 0.],\
                                   [0.79, 0.613107, 0., 0., 0.],\
                                   [0.82, 0.134071, 0.556439, 0., 0.],\
                                   [0.91, 0.132277, 0.0109005, 0.39279, 0.],\
                                   [0.84, 0.157232, 0.0181865, 0.291768, 0.429207]])
    cholesky_inverse = np.linalg.inv(cholesky_decomposition)
    # Market parameters and functions
    riskless_rate = 0.05
    divident_yield = np.zeros(num_of_assets, dtype=float)
    bank_bond_yield = 0
    counterparty_bond_yield = 0
    counterparty_bond_repo_rate = 0
    variation_margin_interest_rate = 0.1
    stock_repo_rate =  0.07 * np.ones(num_of_assets, dtype=float)
    def riskfree_scheme_price(self, theta, delta_t, expect_basis, rate, price_coefficient, delta_coefficient, delta_process):
        temp = \
        (1 - (1-theta[0]) * rate * delta_t) * np.einsum('j, ij-> i',price_coefficient, expect_basis)\
        - delta_t * theta[0] * np.einsum('i, ji, kj -> k', (self.mu_bar + self.divident_yield - self.stock_repo_rate)/self.sigma_bar, self.cholesky_inverse, delta_process)\
        - delta_t * (1 - theta[0]) * np.einsum('i, ji, jk, lk -> l', 
               (self.mu_bar + self.divident_yield - self.stock_repo_rate)/self.sigma_bar, self.cholesky_inverse, delta_coefficient, expect_basis)
        return temp / (1 + rate * delta_t * theta[0])
    def riskfree_scheme_delta(self, theta, delta_t, expect_basis, expect_brownian_basis, rate, price_coefficient, delta_coefficient):
        riskfree_delta = \
        - (1-theta[1]) * (1/theta[1]) * np.einsum('ij, kj ->ik', expect_basis, delta_coefficient)\
        + (1/theta[1]) * (1-(1-theta[1]) * rate * delta_t) * np.einsum('j, ijk-> ik', price_coefficient, expect_brownian_basis)\
        - delta_t * (1-theta[1]) * (1/theta[1]) * np.einsum('i, ji, jk, lkm -> lm', \
               (self.mu_bar + self.divident_yield - self.stock_repo_rate)/self.sigma_bar, self.cholesky_inverse, delta_coefficient, expect_brownian_basis)
        return riskfree_delta
    def numerical_scheme(self, theta, delta_t, regression_coeff, expect_basis, expect_brownian_basis):
        riskfree_delta = self.riskfree_scheme_delta(theta, delta_t, expect_basis, expect_brownian_basis, self.riskless_rate, regression_coeff[0, :], regression_coeff[1:6, :])
        riskfree_price = self.riskfree_scheme_price(theta, delta_t, expect_basis, self.riskless_rate , regression_coeff[0, :], regression_coeff[1:6, :], riskfree_delta)
        
        adjusted_delta =  \
        self.riskfree_scheme_delta(theta, delta_t, expect_basis, expect_brownian_basis, self.bank_bond_yield + self.counterparty_bond_yield - self.counterparty_bond_repo_rate , regression_coeff[6, :], regression_coeff[7:12, :])\
        + (1/theta[1]) * (1-theta[1]) * (self.bank_bond_yield + self.counterparty_bond_yield - self.counterparty_bond_repo_rate + self.variation_margin_interest_rate) * delta_t * np.einsum('j, ijk-> ik', regression_coeff[0, :], expect_brownian_basis)
        adjusted_price = self.riskfree_scheme_price(theta, delta_t, expect_basis, self.bank_bond_yield + self.counterparty_bond_yield - self.counterparty_bond_repo_rate , regression_coeff[6, :], regression_coeff[7:12, :], adjusted_delta)\
                         + delta_t * theta[0] * (self.bank_bond_yield + self.counterparty_bond_yield - self.counterparty_bond_repo_rate + self.variation_margin_interest_rate) *\
                         riskfree_price/  (1 +  (self.bank_bond_yield + self.counterparty_bond_yield - self.counterparty_bond_repo_rate) * delta_t * theta[0])\
                         + delta_t * (1 - theta[0]) * (self.bank_bond_yield + self.counterparty_bond_yield - self.counterparty_bond_repo_rate + self.variation_margin_interest_rate) *\
                         np.einsum('j, ij-> i',  regression_coeff[0, :], expect_basis) /  (1 +  (self.bank_bond_yield + self.counterparty_bond_yield - self.counterparty_bond_repo_rate) * delta_t * theta[0])
        return riskfree_delta, riskfree_price.reshape((-1, 1)), adjusted_delta, adjusted_price.reshape((-1, 1))
    # Product parameters
    weight = np.array([38.1, 6.5, 5.7, 27.0, 22.7], dtype=np.double)
    strike = 1.
    terminal_time = 1.
    buy_sell = -1.
    put_call = "Put"
    # Regression functions and parameters
    sorting_method = "Intrinsic Value"
    basis = "Intrinsic Value"
    basis_order = 3
    no_of_regression = 1 + 5 + 1 + 5
    def regression_variable(self, no_of_samples, derivative, delta, adjusted_derivative, adjusted_delta):
        rec = np.empty((no_of_samples, self.no_of_regression))
        rec[:,0] = derivative.reshape(-1)
        for i in range(1,6):
            rec[:, i] = delta[:, i-1]
        rec[:, 6] = adjusted_derivative.reshape(-1)
        for i in range(7,12):
            rec[:, i] = adjusted_delta[:, i-7]
        return rec
    # Reference Solution
    reference_riskfree = False
    reference_riskfree_price = -0.175866
    refererce_adjust = False
    
class ArithmeticBasketPut():
    def __init__(self, dimension):
        # Example Information
        self.name = "Arithmetic basket put option for BS model"
        self.dimension = dimension
        # Market parameters and functions
        self.riskless_rate = 0.06
        self.bank_bond_yield = 0.
        self.counterparty_bond_yield = 0.
        self.counterparty_bond_repo_rate = 0.
        self.variation_margin_interest_rate = 0.1
        self.stock_repo_rate =  0.06 * np.ones(dimension, dtype=float)
        # Stock parameters
        self.stock_model = "BS"
        self.initial_value = 40. * np.ones(dimension)
        self.num_of_assets = dimension
        self.num_of_brownian_motion = dimension
        self.divident_yield = np.zeros(dimension, dtype=float)
        self.mu_bar = self.riskless_rate - self.divident_yield
        self.sigma_bar = 0.2 * np.ones(dimension)
        self.correlation_matrix = 0.75 * np.identity(dimension) + 0.25 * np.ones((dimension, dimension))
        self.cholesky_decomposition = np.linalg.cholesky(self.correlation_matrix)
        self.cholesky_inverse = np.linalg.inv(self.cholesky_decomposition)
        # Product parameters
        self.weight = 1/dimension * np.ones(dimension, dtype=np.single)
        self.strike = 40.
        self.terminal_time = 1.
        self.buy_sell = 1.
        self.put_call = "Put"
        # Regression functions and parameters
        self.sorting_method = "Intrinsic Value"
        self.basis = "Intrinsic Value"
        self.basis_order = 3
        self.no_of_regression = 1 + self.num_of_brownian_motion + 1 + self.num_of_brownian_motion
    def riskfree_scheme_price(self, theta, delta_t, expect_basis, rate, price_coefficient, delta_coefficient, delta_process):
        temp = \
        (1 - (1-theta[0]) * rate * delta_t) * np.einsum('j, ij-> i',price_coefficient, expect_basis)\
        - delta_t * theta[0] * np.einsum('i, ji, kj -> k', (self.mu_bar + self.divident_yield - self.stock_repo_rate)/self.sigma_bar, self.cholesky_inverse, delta_process)\
        - delta_t * (1 - theta[0]) * np.einsum('i, ji, jk, lk -> l', 
               (self.mu_bar + self.divident_yield - self.stock_repo_rate)/self.sigma_bar, self.cholesky_inverse, delta_coefficient, expect_basis)
        return temp / (1 + rate * delta_t * theta[0])
    def riskfree_scheme_delta(self, theta, delta_t, expect_basis, expect_brownian_basis, rate, price_coefficient, delta_coefficient):
        riskfree_delta = \
        - (1-theta[1]) * (1/theta[1]) * np.einsum('ij, kj ->ik', expect_basis, delta_coefficient)\
        + (1/theta[1]) * (1-(1-theta[1]) * rate * delta_t) * np.einsum('j, ijk-> ik', price_coefficient, expect_brownian_basis)\
        - delta_t * (1-theta[1]) * (1/theta[1]) * np.einsum('i, ji, jk, lkm -> lm', \
               (self.mu_bar + self.divident_yield - self.stock_repo_rate)/self.sigma_bar, self.cholesky_inverse, delta_coefficient, expect_brownian_basis)
        return riskfree_delta
    def numerical_scheme(self, theta, delta_t, regression_coeff, expect_basis, expect_brownian_basis):
        riskfree_delta = self.riskfree_scheme_delta(theta, delta_t, expect_basis, expect_brownian_basis, self.riskless_rate, regression_coeff[0, :], regression_coeff[1:1+self.num_of_brownian_motion, :])
        riskfree_price = self.riskfree_scheme_price(theta, delta_t, expect_basis, self.riskless_rate , regression_coeff[0, :], regression_coeff[1:1+self.num_of_brownian_motion, :], riskfree_delta)
        
        adjusted_delta =  \
        self.riskfree_scheme_delta(theta, delta_t, expect_basis, expect_brownian_basis, self.bank_bond_yield + self.counterparty_bond_yield - self.counterparty_bond_repo_rate , regression_coeff[1+self.num_of_brownian_motion,:], 
                                   regression_coeff[2 + self.num_of_brownian_motion: 2+2* self.num_of_brownian_motion, :])\
        + (1/theta[1]) * (1-theta[1]) * (self.bank_bond_yield + self.counterparty_bond_yield - self.counterparty_bond_repo_rate + self.variation_margin_interest_rate) * delta_t * np.einsum('j, ijk-> ik', regression_coeff[0, :], expect_brownian_basis)
        adjusted_price = self.riskfree_scheme_price(theta, delta_t, expect_basis, self.bank_bond_yield + self.counterparty_bond_yield - self.counterparty_bond_repo_rate , regression_coeff[1+self.num_of_brownian_motion, :], 
                                                    regression_coeff[2+self.num_of_brownian_motion:2+2*self.num_of_brownian_motion, :], adjusted_delta)\
                         + delta_t * theta[0] * (self.bank_bond_yield + self.counterparty_bond_yield - self.counterparty_bond_repo_rate + self.variation_margin_interest_rate) *\
                         riskfree_price/  (1 +  (self.bank_bond_yield + self.counterparty_bond_yield - self.counterparty_bond_repo_rate) * delta_t * theta[0])\
                         + delta_t * (1 - theta[0]) * (self.bank_bond_yield + self.counterparty_bond_yield - self.counterparty_bond_repo_rate + self.variation_margin_interest_rate) *\
                         np.einsum('j, ij-> i',  regression_coeff[0, :], expect_basis) /  (1 +  (self.bank_bond_yield + self.counterparty_bond_yield - self.counterparty_bond_repo_rate) * delta_t * theta[0])
        return riskfree_delta, riskfree_price.reshape((-1, 1)), adjusted_delta, adjusted_price.reshape((-1, 1))
    def regression_variable(self, no_of_samples, derivative, delta, adjusted_derivative, adjusted_delta):
        rec = np.empty((no_of_samples, self.no_of_regression))
        rec[:,0] = derivative.reshape(-1)
        for i in range(self.num_of_brownian_motion):
            rec[:, i+1] = delta[:, i]
        rec[:, 1+self.num_of_brownian_motion] = adjusted_derivative.reshape(-1)
        for i in range(self.num_of_brownian_motion):
            rec[:, i + 2 +self.num_of_brownian_motion] = adjusted_delta[:, i]
        return rec
    # Reference Solution
    reference_riskfree = False
    refererce_adjust = False

#  Under development
class GeometicBasketPut():
    def __init__(self, dimension):
        # Market parameters and functions
        self.riskless_rate = 0.06
        self.bank_bond_yield = 0.
        self.counterparty_bond_yield = 0.
        self.counterparty_bond_repo_rate = 0.
        self.variation_margin_interest_rate = 0.1
        self.stock_repo_rate =  0.07 * np.ones(dimension, dtype=float)
        # Stock parameters
        self.stock_model = "BS"
        self.initial_value = 40. * np.ones(dimension)
        self.num_of_assets = dimension
        self.num_of_brownian_motion = dimension
        self.divident_yield = np.zeros(dimension, dtype=float)
        self.mu_bar = self.riskless_rate - self.divident_yield
        self.sigma_bar = 0.2 * np.ones(dimension)
        self.correlation_matrix = 0.75 * np.identity(dimension) + 0.25 * np.ones((dimension, dimension))
        self.cholesky_decomposition = np.linalg.cholesky(self.correlation_matrix)
        self.cholesky_inverse = np.linalg.inv(self.cholesky_decomposition)
        # Product parameters
        self.weight = 1/dimension * np.ones(dimension, dtype=np.single)
        self.strike = 40.
        self.terminal_time = 1.
        self.buy_sell = 1.
        self.put_call = "Put"
    def riskfree_scheme_price(self, theta, delta_t, expect_basis, rate, price_coefficient, delta_coefficient, delta_process):
        temp = \
        (1 - (1-theta[0]) * rate * delta_t) * np.einsum('j, ij-> i',price_coefficient, expect_basis)\
        - delta_t * theta[0] * np.einsum('i, ji, kj -> k', (self.mu_bar + self.divident_yield - self.stock_repo_rate)/self.sigma_bar, self.cholesky_inverse, delta_process)\
        - delta_t * (1 - theta[0]) * np.einsum('i, ji, jk, lk -> l', 
               (self.mu_bar + self.divident_yield - self.stock_repo_rate)/self.sigma_bar, self.cholesky_inverse, delta_coefficient, expect_basis)
        return temp / (1 + rate * delta_t * theta[0])
    def riskfree_scheme_delta(self, theta, delta_t, expect_basis, expect_brownian_basis, rate, price_coefficient, delta_coefficient):
        riskfree_delta = \
        - (1-theta[1]) * (1/theta[1]) * np.einsum('ij, kj ->ik', expect_basis, delta_coefficient)\
        + (1/theta[1]) * (1-(1-theta[1]) * rate * delta_t) * np.einsum('j, ijk-> ik', price_coefficient, expect_brownian_basis)\
        - delta_t * (1-theta[1]) * (1/theta[1]) * np.einsum('i, ji, jk, lkm -> lm', \
               (self.mu_bar + self.divident_yield - self.stock_repo_rate)/self.sigma_bar, self.cholesky_inverse, delta_coefficient, expect_brownian_basis)
        return riskfree_delta
    def numerical_scheme(self, theta, delta_t, regression_coeff, expect_basis, expect_brownian_basis):
        riskfree_delta = self.riskfree_scheme_delta(theta, delta_t, expect_basis, expect_brownian_basis, self.riskless_rate, regression_coeff[0, :], regression_coeff[1:6, :])
        riskfree_price = self.riskfree_scheme_price(theta, delta_t, expect_basis, self.riskless_rate , regression_coeff[0, :], regression_coeff[1:6, :], riskfree_delta)
        
        adjusted_delta =  \
        self.riskfree_scheme_delta(theta, delta_t, expect_basis, expect_brownian_basis, self.bank_bond_yield + self.counterparty_bond_yield - self.counterparty_bond_repo_rate , regression_coeff[6, :], regression_coeff[7:12, :])\
        + (1/theta[1]) * (1-theta[1]) * (self.bank_bond_yield + self.counterparty_bond_yield - self.counterparty_bond_repo_rate + self.variation_margin_interest_rate) * delta_t * np.einsum('j, ijk-> ik', regression_coeff[0, :], expect_brownian_basis)
        adjusted_price = self.riskfree_scheme_price(theta, delta_t, expect_basis, self.bank_bond_yield + self.counterparty_bond_yield - self.counterparty_bond_repo_rate , regression_coeff[6, :], regression_coeff[7:12, :], adjusted_delta)\
                         + delta_t * theta[0] * (self.bank_bond_yield + self.counterparty_bond_yield - self.counterparty_bond_repo_rate + self.variation_margin_interest_rate) *\
                         riskfree_price/  (1 +  (self.bank_bond_yield + self.counterparty_bond_yield - self.counterparty_bond_repo_rate) * delta_t * theta[0])\
                         + delta_t * (1 - theta[0]) * (self.bank_bond_yield + self.counterparty_bond_yield - self.counterparty_bond_repo_rate + self.variation_margin_interest_rate) *\
                         np.einsum('j, ij-> i',  regression_coeff[0, :], expect_basis) /  (1 +  (self.bank_bond_yield + self.counterparty_bond_yield - self.counterparty_bond_repo_rate) * delta_t * theta[0])
        return riskfree_delta, riskfree_price.reshape((-1, 1)), adjusted_delta, adjusted_price.reshape((-1, 1))
    # Regression functions and parameters
    sorting_method = "Geometric Intrinsic Value"
    basis = "Geometric Intrinsic Value"
    basis_order = 2
    no_of_regression = 1 + 5 + 1 + 5
    def regression_variable(self, no_of_samples, derivative, delta, adjusted_derivative, adjusted_delta):
        rec = np.empty((no_of_samples, self.no_of_regression))
        rec[:,0] = derivative.reshape(-1)
        for i in range(1,6):
            rec[:, i] = delta[:, i-1]
        rec[:, 6] = adjusted_derivative.reshape(-1)
        for i in range(7,12):
            rec[:, i] = adjusted_delta[:, i-7]
        return rec
    # Reference Solution
    reference_riskfree = False
    refererce_adjust = False

Example = ArithmeticBasketPut(5)
