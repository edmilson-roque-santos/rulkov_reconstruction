"""
Collection of methods for solving $\ell_1$ minimization.

Created on --- 2020

@author: Edmilson Roque dos Santos
"""

import cvxpy as cp
import numpy as np

VERBOSE = False

def l_1_optimization(b, PHI, noisy_measurement, params, solver_default = cp.ECOS):
    '''
    

    Parameters
    ----------
    b : TYPE
        DESCRIPTION.
    PHI : TYPE
        DESCRIPTION.
    noisy_measurement : TYPE
        DESCRIPTION.
    params : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    M = params['length_of_time_series']
    
    if not noisy_measurement:
        number_of_coefficients = len(PHI[0, :])
        # The threshold value below which we consider an element to be zero.
        delta = params['threshold_connect']
    
        # Create variable.
        c = cp.Variable(shape = number_of_coefficients)
    
        # Create constraint.
        constraints = [(PHI @ c) == b]
        
        # Form objective.
        obj = cp.Minimize(cp.norm(c, 1))
        
        # Form and solve problem.
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=solver_default,verbose=False)
       
        if VERBOSE:            
            print("status: {}".format(prob.status))
        
        if(prob.status == 'infeasible'):
            print("status: {}".format(prob.status))
            return np.ones(number_of_coefficients)*1e-15, 'infeasible'
        
        if not (prob.status == 'infeasible'):
            if(params['normalize_cols']):
                sparse_vector = c.value/(params['norm_column'][: number_of_coefficients]*np.sqrt(M))  
            else:
                sparse_vector = c.value/(np.sqrt(M))
            nnz_l1 = (np.absolute(sparse_vector) > delta).sum()
           
            if VERBOSE:
                # Number of nonzero elements in the solution (its cardinality or diversity).
                print('Found a feasible x in R^{} that has {} nonzeros.'.format(number_of_coefficients, nnz_l1))
                print("optimal objective value: {}".format(obj.value))
            
            return sparse_vector, nnz_l1
            
    if noisy_measurement:
        number_of_coefficients = len(PHI[0, :])
        # The threshold value below which we consider an element to be zero.
        delta = params['threshold_connect']
        
        # Create variable.
        c = cp.Variable(shape = number_of_coefficients)
            
        # Form objective.
        obj = cp.Minimize(cp.norm(c, 1))
        
        epsilon = cp.Parameter(nonneg=True)
        
        epsilon.value = params['noise_magnitude']
        # Create constraint.
        constraints = [cp.norm2(PHI @ c - b) <= epsilon]    
        
        # Form and solve problem.
        prob = cp.Problem(obj, constraints)
        
        prob.solve(solver=solver_default, verbose=False)
        #CVXOPT
        if(prob.status == 'infeasible'):
            print("status: {}".format(prob.status))
            return np.ones(number_of_coefficients)*1e-15, 'infeasible'                

        if not (prob.status == 'infeasible'):
            if(params['normalize_cols']):
                sparse_vector = c.value/(params['norm_column'][: number_of_coefficients]*np.sqrt(M))  
            else:
                sparse_vector = c.value/(np.sqrt(M))
                
                
            # Number of nonzero elements in the solution (its cardinality or diversity).
            nnz_l1 = (np.absolute(sparse_vector) > delta).sum()
            #print('Found a feasible x in R^{} that has {} nonzeros.'.format(number_of_coefficients, nnz_l1))
            #print("optimal objective value: {}".format(obj.value))
        
            return sparse_vector, nnz_l1 
            
def iteratively_soft_thresholding(b, PHI, params):
    '''
    

    Parameters
    ----------
    b : TYPE
        DESCRIPTION.
    PHI : TYPE
        DESCRIPTION.
    noisy_measurement : TYPE
        DESCRIPTION.
    params : TYPE
        DESCRIPTION.

    Returns
    -------
    sparse_vector : TYPE
        DESCRIPTION.
    nnz_l1 : TYPE
        DESCRIPTION.

    '''
    
    M = params['length_of_time_series'] 
    number_of_coefficients = len(PHI[0, :])
    
    #Iteratively least square solution #
    X_t_i = np.linalg.lstsq(PHI, b, rcond=-1)[0]
    
    
    threshold_connect = params['threshold_connect']
    
    iterations_IST = params.get('iterations_IST', 5)
    
    for k in range(iterations_IST):
        X_t_i[np.absolute(X_t_i) < threshold_connect] = 0.0
        
        X_t_i_C = np.copy(X_t_i)
        pos_inds = np.argwhere(np.absolute(X_t_i_C) >= threshold_connect)
        pos_inds = pos_inds.astype('int')
        pos_inds = pos_inds[:, 0]
    
        X_t_i[pos_inds] = np.linalg.lstsq(PHI[:, pos_inds], b, rcond=-1)[0]
    
        X_t_i_D = np.copy(X_t_i)
    
    if params['normalize_cols']:
        sparse_vector = X_t_i_D/(params['norm_column'][: number_of_coefficients]*np.sqrt(M))
    else:
        sparse_vector = X_t_i_D/(np.sqrt(M))  
    nnz_l1 = (np.absolute(X_t_i_D) > threshold_connect).sum()
    
    return sparse_vector, nnz_l1
