"""
Module for Rulkov map simulation

Created on Thu Jun 15 12:51:47 2023

@author: Edmilson Roque dos Santos
"""

import numpy as np
import sympy as spy



def rulkov_map(x_state, alpha = 4.4, sigma = 0.001, beta = 0.001):
    '''
    Isolated map for the Rulkov map.

    Parameters
    ----------
    x_state : numpy array
        Network state.
    alpha : float, optional
        Rulkov parameter. The default is 4.4.
    sigma : float, optional
        Rulkov parameter. The default is 0.001.
    beta : float, optional
        Rulkov parameter. The default is 0.001.

    Returns
    -------
    x_state : numpy array
        Iteration of map.

    '''
    x_ = x_state.copy()
    
    x = x_[0::2]
    y = x_[1::2]
    
    f_x_iteration = alpha/(1.0 + x**2) + y
    f_y_iteration = y - sigma*x - beta
    
    x_state = np.zeros(x_.shape)
    
    x_state[0::2] = f_x_iteration.copy()
    x_state[1::2] = f_y_iteration.copy()
    
    return x_state

def diff_coupling_x(x_state, A):
    '''
    Diffusive coupling for the coupled Rulkov maps.

    Parameters
    ----------
    x_state : numpy array
        Network state.
    A : numpy array
        Adjacency matrix.

    Returns
    -------
    cplg_x : numpy array
        Coupling via fast variables.

    '''
    x_ = x_state.copy()
    
    cplg_x = np.zeros(x_state.shape)
    
    x = x_[0::2]
    
    cplg_x[0::2] = A @ x
    
    return cplg_x

def spy_rulkov_map(x_t, alpha = 4.4, sigma = 0.001, beta = 0.001):
    '''
    Symbolic isolated map of Rulkov map.

    Parameters
    ----------
    x_t : list
        Symbolic variables for network state.
    alpha : float, optional
        Rulkov parameter. The default is 4.4.
    sigma : float, optional
        Rulkov parameter. The default is 0.001.
    beta : float, optional
        Rulkov parameter. The default is 0.001.

    Returns
    -------
    f_isolated : sympy Matrix
        Symbolic isolated map.

    '''    
    x = x_t[0::2]
    y = x_t[1::2]
    
    N = len(x)
    
    f_isolated = spy.zeros(2*N, 1)
    
    for i in range(N):
        f_isolated[2*i] = alpha/(1 + x[i]**2) + y[i]
        f_isolated[2*i + 1] = y[i] - sigma*x[i] - beta
        
    return f_isolated 
            
def spy_diff_coupling_x(x_t, A):
    '''
    Symbolic coupling for coupled Rulkov maps.

    Parameters
    ----------
    x_t : list
        Symbolic variables for network state.
    A : numpy array
        Adjacency matrix.

    Returns
    -------
    cplg_x : sympy Matrix
        Symbolic Coupling via fast variables.

    '''
    N_2 = len(x_t)
    
    spy_A = spy.Matrix(A)
    
        
    cplg_x = spy.zeros(N_2, 1)
    
    x = spy.Matrix(x_t[0::2])
    
    a = spy_A @ x
    
    for i in range(int(N_2/2)):
        cplg_x[2*i] = a[i]#spy.expand(a[i] + a[i]*x[i]**2)
        
    return cplg_x
