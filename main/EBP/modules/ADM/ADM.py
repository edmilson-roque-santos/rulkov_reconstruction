"""
Implement Alternating Direction Method designed by DOI: 10.1109/TIT.2016.2601599
And adapted by 10.1109/TMBMC.2016.2633265.

Created on Mon Jun 26 10:38:37 2023

@author: Edmilson Roque dos Santos
"""

import numpy as np
from numpy import linalg as LA
from scipy.linalg import null_space

##===========================================================================##
def soft_thresholding(x, d):
    '''
  
    Soft-threshold operator
  
    Parameters
    ----------
    x : numpy array
        Vector to be soft-thresholded.
    d : float
        Threshold magnitude. Any entry of x below than tau is set to zero.
  
    Returns
    -------
    x : numpy array
        Soft-thresholded vector.

    '''
    return np.sign(x)*np.maximum(np.absolute(x) - d, np.zeros(x.shape[0]))

#======================================#======================================#

def ADM(Y, q_init, lambda_parameter, MaxIter = 10000, tol=1e-8):
    '''
    Alternating Direction Method

    Parameters
    ----------
    Y : numpy array
        Orthonormal basis of the null space.
    q_init : numpy array
        Normalized row of the Y matrix.
    lambda_parameter : float
        regularizer parameter.
    MaxIter : int
        Maximum number of iterations of the ADM algorithm. The default is 10000.
    tol : float
        Tolarance for stopping criterion
        ||q_ - q||_2 < tol stops algorithm. Tolerance for convergence of the adm algorithm. The default is 1e-8. 

    Returns
    -------
    q : numpy array
        The recovered sparse vector such that x0 = Y q. 

    '''
    
    #MaxIter = 10000
    #MaxIter = np.abs(len(PHI[:, 0]) - number_of_coefficients*2)
    #MaxIter = int(np.log(MaxIter)*(MaxIter)**4); # Max iteration before stop
    
    q = q_init

    for k in range(MaxIter):
        q_old = np.copy(q)

        x = soft_thresholding(Y.dot(q), lambda_parameter) # update y by soft thresholding

        if(LA.norm(Y.T @ x) <= 0.0):
            break
        
        q = Y.T @ x/LA.norm(Y.T @ x) # update q by projection to the sphere
        res_q = LA.norm(q_old - q)
        
        if (res_q <= tol):
            return q
            break
    #print(res_q)    
    
    return q   
 
#======================================#======================================#

def ADM_initial_guess_variation(kernel_matrix, lambda_parameter, 
                                sparse_vector_dimension,
                                params,
                                percentage_of_rows = 0.1,
                                tol_ADM = 1e-8):
    '''
    

    Parameters
    ----------
    kernel_matrix : numpy array
        Orthonormal basis of the null space.
    lambda_parameter : float
        regularizer parameter.
    sparse_vector_dimension : int
        Dimension of the ambient space that sparse vector lies on.
    params : dict 
        
    percentage_of_rows : float, optional
        Percentage of rows to be used in the initialization of the ADM method. The default is 0.1.
    tol_ADM : float
        Tolarance for stopping criterion
        ||q_ - q||_2 < tol stops algorithm. The default is 1e-8.
    
    Returns
    -------
    sparse_vector : numpy array
        Sparse solution found for the particular regularizer parameter.
    number_of_terms : int
        Number of terms which are different than zero.
    ind_nonzero_coefficients : numpy array
        Indices in the support of the sparse solution.

    '''    
           
    M = params['length_of_time_series']
    
    #Select uniformly 10 percent of the row for searching the sparse vector
    percentage_rows_kernel_matrix = np.random.randint(0, len(kernel_matrix[:, 0]), int(percentage_of_rows*len(kernel_matrix[:, 0])))
   
    number_of_row_kernel_matrix = len(percentage_rows_kernel_matrix)
    sparse_vector_comparison = np.zeros((number_of_row_kernel_matrix, sparse_vector_dimension))
    
    number_of_zeros_sparse_candidates = np.zeros(number_of_row_kernel_matrix)
    
    for j in range(number_of_row_kernel_matrix):
        q_init = kernel_matrix[percentage_rows_kernel_matrix[j], :]/(LA.norm(kernel_matrix[percentage_rows_kernel_matrix[j], :]))
            
        sparse_vector_output = ADM(kernel_matrix, q_init, lambda_parameter)
               
        c = kernel_matrix @ sparse_vector_output
        
        if(params['normalize_cols']):
            sparse_vector_comparison[j, :] = c/(params['norm_column'][: sparse_vector_dimension]*np.sqrt(M))  
        else:
            sparse_vector_comparison[j, :] = c/(np.sqrt(M))
        
        number_of_zeros_sparse_candidates[j] = len(np.argwhere(np.absolute(sparse_vector_comparison[j, :]) < lambda_parameter)[:, 0])
        
            
    #Find the vector with the largest number of zeros    
    indices_sparse_vectos = np.argwhere(number_of_zeros_sparse_candidates == np.amax(number_of_zeros_sparse_candidates))[:, 0]    
    
    #Save the indices of non zero coefficients
    ind_nonzero_coefficients = np.argwhere(np.absolute(sparse_vector_comparison[indices_sparse_vectos[0], :]) >= lambda_parameter)[:, 0]

    sparse_vector = sparse_vector_comparison[indices_sparse_vectos[0], :]
    
    #Set thresholded coefficients to zero    
    sparse_vector[np.absolute(sparse_vector) < lambda_parameter] = 0.0 
     
    #Check that the solution found by ADM is unique. 
    if(len(indices_sparse_vectos) > 1 and len(ind_nonzero_coefficients) > 0):
        
        difference_verification = sparse_vector_comparison[indices_sparse_vectos[0], ind_nonzero_coefficients] - sparse_vector_comparison[indices_sparse_vectos[1], ind_nonzero_coefficients]
        if(np.amax(difference_verification) > tol_ADM):
            print('ADM has discovered two different sparsest vectors')
        
    number_of_terms = len(ind_nonzero_coefficients)
    
    return sparse_vector, number_of_terms, ind_nonzero_coefficients
    
def ADM_pareto(PHI_implicit_method, params, number_of_points = 30, lambda_0 = 1e-8):
    '''
    Pareto front used in 10.1109/TMBMC.2016.2633265 

    Parameters
    ----------
    PHI : numpy array 
        Library matrix evaluated along the trajectory.
    params : dict
    
    number_of_points : int
        Number of points to be evaluated along the Parato front.
    lambda_0 : float
        Initial regularizer parameter for the Pareto front.
        
    Returns
    -------
    sparsity_of_vector : numpy array
        lambda regularizer vs number of terms.
    pareto_front : numpy array
        number of terms vs |PSI sparse vector|.
    matrix_sparse_vectors : numpy array
        Sparse vector found for each regularizer parameter.

    '''
    
    number_of_coefficients = len(PHI_implicit_method[0, :])
    
    sparsity_of_vector = np.zeros((number_of_points, 2))
    pareto_front = np.zeros((number_of_points,2))
    matrix_sparse_vectors = np.zeros((number_of_points, number_of_coefficients))

    sparse_vector_dimension = number_of_coefficients

    ker_orthonormal_basis_PHI = null_space(PHI_implicit_method)
    print(ker_orthonormal_basis_PHI.shape)
    # Vary lambda by factor of 2 until we hit the point where all coefficients are forced to zero.
    lambda_parameter = lambda_0
    for counter in range(number_of_points):
            
        sparse_vector, number_of_terms, indices_non_zero_coefficients_sparse_vector = \
            ADM_initial_guess_variation(ker_orthonormal_basis_PHI, 
                                        lambda_parameter, 
                                        sparse_vector_dimension, 
                                        params)
        
        if(number_of_terms <= 0):
            break
            
        sparsity_of_vector[counter, 0] = lambda_parameter
        sparsity_of_vector[counter, 1] = number_of_terms
        pareto_front[counter, 0] = number_of_terms
        pareto_front[counter, 1] = np.sum(PHI_implicit_method.dot(sparse_vector))
        matrix_sparse_vectors[counter, :] = sparse_vector

        
        lambda_parameter = 2*lambda_parameter
    
    return sparsity_of_vector, pareto_front, matrix_sparse_vectors

def pareto_test(sparsity_of_vector, pareto_front, matrix_sparse_vectors):
    '''
    

    Parameters
    ----------
    sparsity_of_vector : numpy array
        lambda regularizer vs number of terms.
    pareto_front : numpy array
        number of terms vs |PSI sparse vector|.
    matrix_sparse_vectors : numpy array
        Sparse vector found for each regularizer parameter.

    Returns
    -------
    sparse_vector : numpy array
        Selected sparse vector after Pareto front.

    '''
    pos_different_zero = np.argwhere(np.absolute(pareto_front[:, 1]) > 1e-14)[:, 0]
    
    pos_minimum_error = np.argmin(np.absolute(pareto_front[pos_different_zero, 1]))    
    
    #sparse vector chosen from the pareto test
    sparse_vector = matrix_sparse_vectors[pos_minimum_error, :]
    
    return sparse_vector
