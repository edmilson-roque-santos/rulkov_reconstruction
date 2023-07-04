'''
Script to reconstruct a network using both adapted and non-adapted basis.
'''
import cvxpy as cp
import networkx as nx 
import numpy as np
import os
import sympy as spy
from tqdm import tqdm

from EBP import tools, net_dyn, optimizer
from EBP.base_polynomial import pre_settings as pre_set 
from EBP.base_polynomial import poly_library as polb
from EBP.base_polynomial import triage as trg
from EBP import greedy_algorithms as gnr_alg
from EBP.modules.ADM import ADM
from EBP.modules.rulkov import rulkov

solver_default = cp.ECOS
VERBOSE = True
completing_on = False     #Boolean: to complete the coefficient matrix at each node


tqdm_par = {
"unit": "it",
"unit_scale": 1,
"ncols": 80,
"bar_format": "[Reconstructing progress] {percentage:2.0f}% |{bar}| {n:.1f}/{total:.1f}  [{rate_fmt}, {remaining_s:.1f}s rem]",
"smoothing": 0
}


def symb_net_dyn(params):
    '''
    Express the network dynamics in symbolic language.

    Parameters
    ----------
    params : dict

    Returns
    -------
    net_dyn_exp : sympy expression
        Network dynamics in symbolic language.

    '''
    net_dynamics_dict = dict()
    net_dynamics_dict['adj_matrix'] = params['adj_matrix']
    degree = np.sum(params['adj_matrix'], axis=0)
    
    net_dynamics_dict['f'] = rulkov.spy_rulkov_map
    net_dynamics_dict['h'] = rulkov.spy_diff_coupling_x
    net_dynamics_dict['max_degree'] = np.max(degree)
    net_dynamics_dict['coupling'] = params['coupling']
    
    net_dyn_exp = net_dyn.spy_gen_net_dyn(net_dynamics_dict)
    
    return net_dyn_exp

def retrieve_dyn_sym(x_eps, params, indep_term = True):
    '''
    Build the reconstruction model from the coefficient vector.

    Parameters
    ----------
    x_eps : numpy array
        Coefficient vector reconstruction from minimization problem.
    params : dict
        
    indep_term : boolean, optional
        To include the independent term or nor in the coefficient vector.
        The default is True.

    Returns
    -------
    symb_node_dyn : sympy expression
        Reconstructed model in symbolic language.

    '''
    
    L = params['L']
    symbolic_PHI = params['symbolic_PHI']
    spy_PHI = spy.Matrix(symbolic_PHI)
    
    sv = x_eps.copy()
    roud = 8
    sv = np.around(sv, roud)
    threshold = 10**(-roud)

    sv[np.absolute(sv) < threshold] = 0
    
    if indep_term:
        c_num_x = sv[:L]
        c_den_x = np.zeros(L)
        c_den_x[0] = 1
        c_den_x[1:] = sv[L+1:]
        
    else:               
        c_num_x = sv[:L]
        c_den_x = np.zeros(L)
        c_den_x = -1*sv[L:]
    
    c_num_spy_x = spy.Matrix(c_num_x).n(roud)
    c_den_spy_x = spy.Matrix(c_den_x).n(roud)

    #calculate the numerator and denominator using symbolic representation
    num_x = spy_PHI.dot(c_num_spy_x)
    den_x = spy_PHI.dot(c_den_spy_x)

    symb_node_dyn = spy.simplify(num_x/den_x)

    return symb_node_dyn

def reconstr(X_t_, params, solver_optimization = solver_default):
    '''
    Network reconstruction at once strategy.
    
    Given the multivariate time series the algorithm returns a graph.

    Parameters
    ----------
    X_t_ : numpy array - size (length_of_time_series, number_of_vertices)
        Multivariate time series.
    params : dict
    
    solver_optimization : cp solver, optional
        Solver to be used in the convex minimization problem. 
        The default is solver_default.

    Returns
    -------
    net_dict : dict
        Dictionary encoding Adjacency matrix, Graph structure and Library matrix.
        Keys: 
            'G' : networkx Graph
                Reconstructed graph structure 
            'A' : numpy array
                Reconstructed adjacency matrix
            'PHI' : numpy array
                Library matrix
            'PHI.T PHI' : numpy array
                Library matrix multiplied by its transpose
            'params' : dict
                Core dictionary of the code to track the relevant information of the reconstruction.
            'info_x_eps' : dict
                Trace the reconstruction for each node. Save the information of the coefficient vector of each node.
            'x_eps_matrix' : numpy array
                Final coefficient matrix after the reconstruction.
    '''
    params_ = params.copy()
    params_['nodelist'] = np.arange(0, X_t_.shape[1], 1, dtype = int)
    N = params_['nodelist'].shape[0]
    
    threshold = params_['threshold_connect']
    
    net_dict = dict()          #create greedy algorithm dictionary to save info
    
    net_dict['info_x_eps'] = dict()   #info is dictionary with several info saving along the process
    net_dict['sym_node_dyn'] = dict()
    
    params_['number_of_vertices'] = params_['nodelist'].shape[0]
    if params_['use_canonical']:
        params_ = trg.triage_params(params_)
     
    if params_['use_orthonormal'] and params_['build_from_reduced_basis']:
        params_ = pre_set.set_orthnormfunc(params_['orthnorm_func_filename'], params_)
    
    if params_['use_orthonormal']:
        params_ = trg.triage_params(params_)
    
    X_t = X_t_[:-1, :]
    B = X_t_[1:, :]
    
    params_['length_of_time_series'] = X_t.shape[0]
    
    PHI, params_ = polb.library_matrix(X_t, params_)
    net_dict['PHI'] = PHI
    net_dict['PHI.T PHI'] = PHI.T @ PHI 
    net_dict['params'] = params_.copy()   #for reference we save the params used
    
    L = int(2*params_['L'] - 1)
    params_['indices_cluster'] = np.arange(0, L, dtype = int)
    
    params_['power_indices'] = np.vstack((params_['power_indices'], params_['power_indices'][1:,:]))
    
    id_trial = params_['id_trial']
    
    #Coefficient matrix to be used along the process
    x_eps_matrix = np.zeros((L, N))
    
    x_eps_dict = dict()     #x_eps is created to save info about the reconstruction of nodes
    x_eps_dict['params'] = params_.copy()
    
           
    B_ = B.copy()
    for id_node in tqdm(id_trial, **tqdm_par):
        b = B_[:, id_node]
        try:
            THETA = np.hstack((PHI, -1*np.diag(b) @ PHI[:, 1:]))
            
            x_eps, num_nonzeros_vec = optimizer.l_1_optimization(b, THETA, 
                                                                 params_['noisy_measurement'], 
                                                                 params_,
                                                                 solver_default)            
            
        except:
            x_eps = np.zeros(L)
            if VERBOSE:
                print('Solver failed: node = ', id_node)

        x_eps_can = x_eps.copy()                            
        x_eps_dict[id_node] = x_eps_can
        
        x_eps_matrix[:, id_node] = x_eps_can
        
        net_dict['sym_node_dyn'][id_node] = retrieve_dyn_sym(x_eps_can, params_, 
                                                             indep_term = True)
    
    net_dict['info_x_eps'] = x_eps_dict.copy()
    net_dict['x_eps_matrix'] = x_eps_matrix
    net_dict['x_eps_matrix'][np.absolute(net_dict['x_eps_matrix']) < threshold] = 0.0
                        
    return  net_dict      


def ADM_reconstr(X_t_, params):
    '''
    Network reconstruction at once strategy.
    
    Given the multivariate time series the algorithm returns a graph.

    Parameters
    ----------
    X_t_ : numpy array - size (length_of_time_series, number_of_vertices)
        Multivariate time series.
    params : dict
    
    solver_optimization : cp solver, optional
        Solver to be used in the convex minimization problem. 
        The default is solver_default.

    Returns
    -------
    net_dict : dict
        Dictionary encoding Adjacency matrix, Graph structure and Library matrix.
        Keys: 
            'G' : networkx Graph
                Reconstructed graph structure 
            'A' : numpy array
                Reconstructed adjacency matrix
            'PHI' : numpy array
                Library matrix
            'PHI.T PHI' : numpy array
                Library matrix multiplied by its transpose
            'params' : dict
                Core dictionary of the code to track the relevant information of the reconstruction.
            'info_x_eps' : dict
                Trace the reconstruction for each node. Save the information of the coefficient vector of each node.
            'x_eps_matrix' : numpy array
                Final coefficient matrix after the reconstruction.
    '''
    params_ = params.copy()
    params_['nodelist'] = np.arange(0, X_t_.shape[1], 1, dtype = int)
    N = params_['nodelist'].shape[0]
    
    threshold = params_['threshold_connect']
    
    net_dict = dict()          #create greedy algorithm dictionary to save info
    
    net_dict['info_x_eps'] = dict()   #info is dictionary with several info saving along the process
    net_dict['sym_node_dyn'] = dict()

    params_['number_of_vertices'] = params_['nodelist'].shape[0]
    if params_['use_canonical']:
        params_ = trg.triage_params(params_)
     
    if params_['use_orthonormal'] and params_['build_from_reduced_basis']:
        params_ = pre_set.set_orthnormfunc(params_['orthnorm_func_filename'], params_)
    
    if params_['use_orthonormal']:
        params_ = trg.triage_params(params_)
    
    X_t = X_t_[:-1, :]
    B = X_t_[1:, :]
    
    params_['length_of_time_series'] = X_t.shape[0]
    
    PHI, params_ = polb.library_matrix(X_t, params_)
    net_dict['PHI'] = PHI
    net_dict['PHI.T PHI'] = PHI.T @ PHI 
    net_dict['params'] = params_.copy()   #for reference we save the params used
    
    L = int(2*params_['L'])
    params_['indices_cluster'] = np.arange(0, L, dtype = int)
    
    params_['power_indices'] = np.vstack((params_['power_indices'], params_['power_indices'][1:,:]))
    
    id_trial = params_['id_trial']
    
    #Coefficient matrix to be used along the process
    x_eps_matrix = np.zeros((L, N))
    
    x_eps_dict = dict()     #x_eps is created to save info about the reconstruction of nodes
    x_eps_dict['params'] = params_.copy()
    
               
    B_ = B.copy()
    for id_node in tqdm(id_trial, **tqdm_par):
        b = B_[:, id_node]
        
        if np.mod(id_node, 2):
            x_eps = np.linalg.lstsq(PHI, b, rcond=-1)[0]/(np.sqrt(params_['length_of_time_series']))
                                            
            x_eps_matrix[:params_['L'], id_node] = x_eps
            x_eps_can = x_eps_matrix[:, id_node].copy()    
            net_dict['sym_node_dyn'][id_node] = retrieve_dyn_sym(x_eps_can, params_, 
                                                                 indep_term = True)
        else:
            THETA = np.hstack((PHI, np.diag(b) @ PHI))
            sparsity_of_vector, pareto_front, matrix_sparse_vectors = ADM.ADM_pareto(THETA, params_)
            x_eps_dict[id_node] = matrix_sparse_vectors
            x_eps = ADM.pareto_test(sparsity_of_vector, pareto_front, matrix_sparse_vectors)
            x_eps_can = x_eps.copy()                                    
            x_eps_matrix[:, id_node] = x_eps_can
            net_dict['sym_node_dyn'][id_node] = retrieve_dyn_sym(x_eps_can, params_, 
                                                                 indep_term = False)
        '''
        try:
            
        except:
            x_eps = np.zeros(L)
            if VERBOSE:
                print('Solver failed: node = ', id_node)
        '''  
        
        
    
    net_dict['info_x_eps'] = x_eps_dict.copy()
    net_dict['x_eps_matrix'] = x_eps_matrix
    net_dict['x_eps_matrix'][np.absolute(net_dict['x_eps_matrix']) < threshold] = 0.0
                        
    return  net_dict      


def uniform_error(net_dict, num_samples = 50, time_eval = 1):
    '''
    Calculate the error in the reconstructed model. The error is expressed
    as an average over equally spaced points in the trajectory of the network
    dynamics.

    Parameters
    ----------
    net_dict : dict
        recontruction dictionary data.
    num_samples : int, optional
        Number of points to evaluate the error. The default is 50.
    time_eval : int, optional
        Number of iterations ahead to iterate the reconstructed model. The default is 1.

    Returns
    -------
    error_matrix : numpy array
        Error for each node in the network.

    '''
    Y_t = net_dict['Y_t']
    t_test = Y_t.shape[0]
    N = Y_t.shape[1]
    
    
    test_indices = np.linspace(0, t_test-time_eval-1, num_samples, dtype = int)
    
    error_matrix = np.zeros(N)
    for id_test in test_indices:
        y_0 = Y_t[id_test, :]
        Z = net_dyn.generate_net_dyn_model(y_0, time_eval, net_dict)

        for id_node in range(N):
            y_true = Y_t[id_test:id_test + time_eval + 1, id_node]
            error = tools.RSME(y_true, Z[:, id_node])
            
            error_matrix[id_node] = error_matrix[id_node]\
                + error**2/num_samples   
    
    return np.sqrt(error_matrix)
