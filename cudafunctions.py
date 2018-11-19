#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 17:16:33 2018

@author: Ki Wai Chau
"""
import math
import numpy as np
from numba import float64, guvectorize, cuda

@guvectorize([(float64[:], float64, float64[:], float64[:, :], float64[:], float64[:])], '(n), (), (n), (n, m), (m)->(n)', target='cuda')
def cuda_step_euler(last, dt, drift, volatility, noise, res):
    """   Approximate SDE in one time step with Euler scheme with cuda u-funcs"""
    for i in range(last.shape[0]):
        res[i] = last[i]
        res[i] += drift[i] * dt
        for j in range(volatility.shape[1]):
            res[i] += volatility[i, j] * noise[j] * math.sqrt(dt)

@cuda.jit            
def cuda_jit_step_euler(last, dt, drift, volatility, normdist):
    """   Approximate SDE in one time step with Euler scheme with cuda jit"""
    i = cuda.grid(1)
    noise = normdist[i * volatility.shape[2]: (i + 1) * volatility.shape[2]]
    if i < last.shape[0]:
        for k in range(last.shape[1]):
            last[i, k] += drift[i, k] * dt
            for j in range(volatility.shape[2]):
                last[i, k] += volatility[i, k, j] * math.sqrt(dt) * noise[j]
                
@cuda.jit(device=True)
def device_rsolv(a, n, d, b):
    """ Solves Rx = b, based upon QR decomposition 
    
    For a linear equations system with upper tranagular matrix R, and constant term b, 
    this function calculate x = R^-1 * b inside GPU and return it in place of b. Notice
    that the matrix R is seprate into 2 parts and store in the upper trangular part of a
    and array d. This setting is used to save the memory usage of the whole QR algorithm. 
    
    :param a: This matrix contains the upper triangular matrix R minus the diagonal
              Only the upper half of the matrix is used
    :param n: The dimension of the matrix
    :param d: The diagonal array for the upper trangular matrix R 
    :param b: The constant term for the linear system
    :type a: numpy array
    :type n: int
    :type d: numpy array
    :type b: numpy array
    """
    temp = cuda.local.array(1, dtype=float64)
    
    b[-1] /= d[-1]
    for i in range(n-2, -1, -1):
        temp[0] = 0
        for j in range(i+1, n):
            temp[0] += a[i, j] * b[j]
        b[i] = (b[i] -temp[0])/d[i]
    return b

@cuda.jit(device=True)
def device_qrsolv(a, m, n, c, d, b):
    """ Solves Ax=b, based upon QR decomposition
    
    For a linear equations system with QR decomposable matrix A, and constant term b,
    this function calculate x = A^-1 * b inside GPU and return in in place of b. Notice
    that an alternative matrix a, with the upper trangular part being the R part of A 
    (minus the diagonal) and the lower trangular part contains the orthogonal basis of Q.
    
    :param a: This matrix contains the QR decomposition of A in condensed form
    :param m: The row dimension of the matrix
    :param n: The column dimension of the matrix
    :param c: This vector contains v^Tv/2 for all v in Q.
    :param d: The diagonal array for the upper trangular matrix R times scalling factor
    :param b:The constant term for the linear system
    :type a: numpy array
    :type m: int
    :type n: int
    :type c: numpy array
    :type d: numpy array
    :type b: numpy array
    """
    # Creat temporaory local storage
    temp = cuda.local.array(2, dtype=float64) #(sum, type)
    for j in range(n):
        # Calculate sum = v_j^T*b
        temp[0] = 0.
        for i in range(j, m):
            temp[0] += a[i, j] * b[i]
        # Calculate beta * sum = (2/v_j^T*v_j) (v_j^T*b)
        temp[1] = temp[0]/c[j]
        # b = (I + beta v_j* v_j^T) * b
        for i in range(j, m):
            b[i] -= temp[1] * a[i,j]
    device_rsolv(a, n, d, b)
    return b

@cuda.jit(device=True)
def device_qrdcmp(a, m, n, c, d):
    """ Proform QR decoposition
    
    This function takes in a matrix a, and uses Householder reflections to perform QR
    decomposition. It returns in place of a a mixed matrix with the upper half being 
    the R portion of the decoposition and the lower half the Q portion. The array d is
    the diagonal element of R and c is a array of scaling factor.
    
    :param a: This matrix for decomposition
    :param m: The row dimension of the matrix
    :param n: The column dimension of the matrix
    :param c: This vector contains v^Tv/2 for all v in Q.
    :param d: The diagonal array for the upper trangular matrix R times scalling factor
    :type a: numpy array
    :type m: int
    :type n: int
    :type c: numpy array
    :type d: numpy array
    """ 
    temp = cuda.local.array(4, dtype=float64) #(scale, sum, sigma, tau)
    for k in range(n):
        temp[0] = 0.
        for i in range(k, m):
            temp[0] = max(temp[0], math.fabs(a[i, k]))
        if temp[0] == 0:
            c[k] = d[k] = 0.0
        else:
            for i in range(k, m):
                a[i, k] /= temp[0]
                temp[1] = 0.0
            for i in range(k, m):
                temp[1] += a[i, k] ** 2
            
            temp[2] = math.copysign(math.sqrt(temp[1]), a[k, k])
            a[k, k] += temp[2]
            c[k] = temp[2] * a[k, k]
            d[k] = -temp[0] * temp[2]
            for j in range(k+1, n):
                temp[1] = 0.0
                for i in range(k, m):
                    temp[1] += a[i, k] * a[i, j]
                temp[3] = temp[1]/c[k]
                for i in range(k, m):
                    a[i, j] -= temp[3] * a[i, k]
                    
    return a, c, d
                    
@cuda.jit
def cuda_qrdcmp(bundles_num, a, m, n, c, d):
    """ Proform QR decoposition in parallel with GPU"""
    i = cuda.grid(1)
    if i < bundles_num:
        device_qrdcmp(a[i], m, n, c[i], d[i])
        
@cuda.jit
def cuda_qrsolv(bundle_num, index_num, a, m, n, c, d, b):
    """ Solves Ax=b, based upon QR decomposition in parallel with GPU"""
    i = cuda.grid(1)
    if i < bundle_num*index_num:
        device_qrsolv(a[int(i/index_num)], m, n, c[int(i/index_num)], d[int(i/index_num)], b[int(i/index_num), i%index_num])

def cuda_regression(basis_order, no_of_regression, c_bundles_num, bundle_range, sorted_regression_unknown, sorted_basis, regression_coeff):
    """Perform Regression in parallel for different bundle and regression target"""

    blksz = 256
    gridsz = int(math.ceil( c_bundles_num * no_of_regression / blksz))
    
    regression_matrix = np.empty((c_bundles_num, basis_order, basis_order), dtype=np.double)
    regression_vector = np.empty((c_bundles_num, no_of_regression, basis_order), dtype=np.double)

    for b in range(c_bundles_num):
        regression_matrix[b] = np.dot(np.transpose(sorted_basis[bundle_range[b, 0]: bundle_range[b, 1]]), \
                         sorted_basis[bundle_range[b, 0]: bundle_range[b, 1]])
        for i in range(no_of_regression):
            regression_vector[b, i] = np.dot(np.transpose(sorted_basis[bundle_range[b, 0]: bundle_range[b, 1]]),\
                             sorted_regression_unknown[bundle_range[b, 0]: bundle_range[b, 1], i])
                
    c = cuda.device_array((c_bundles_num, basis_order), dtype=np.double)
    d = cuda.device_array((c_bundles_num, basis_order), dtype=np.double)
    d_regression_matrix = cuda.to_device(regression_matrix)
    d_regression_vector = cuda.to_device(regression_vector)
    cuda_qrdcmp[gridsz, blksz](c_bundles_num, d_regression_matrix, basis_order, basis_order, c, d)

    cuda_qrsolv[gridsz, blksz](c_bundles_num, no_of_regression, d_regression_matrix, basis_order, basis_order, c, d, d_regression_vector)
        
    regression_coeff[:, :, :] = d_regression_vector.copy_to_host()
    
    return regression_coeff
    
def test():
    """Test the CUDA QR decomposition and solver"""
    a = np.array([[[3., 5., 2.],[-1., 4., 2.],[1., 0., 1.]]])
    c = cuda.device_array((2, a.shape[1]), dtype=np.double)
    d = cuda.device_array((2, a.shape[1]), dtype=np.double)
    b = np.array([[[5, 12., 1.], [-1., 3., 0.]]])
        
    d_a = cuda.to_device(a)
    d_b = cuda.to_device(b)
    
    blksz = 256
    gridsz = int(1024 / blksz)
    
    cuda_qrdcmp[gridsz, blksz](1, d_a, a.shape[1], a.shape[2], c, d)
    cuda_qrsolv[gridsz, blksz](1, 2, d_a, a.shape[1], a.shape[2], c, d, d_b)
    
    b = d_b.copy_to_host()
    
    print(b)

if __name__ == '__main__':
    test()