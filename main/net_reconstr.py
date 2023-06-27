'''
Script to reconstruct a network using both adapted and non-adapted basis.
'''
import cvxpy as cp
import networkx as nx 
import numpy as np
import os
from tqdm import tqdm

from EBP import tools, net_dyn, optimizer
from EBP.base_polynomial import pre_settings as pre_set 
from EBP.base_polynomial import poly_library as polb
from EBP.base_polynomial import triage as trg
from EBP import greedy_algorithms as gnr_alg
from EBP.modules.ADM import ADM

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
    net_dict['G'] = nx.Graph() #create the empty graph
    net_dict['G'].add_nodes_from(params_['nodelist']) 
    net_dict['A'] = nx.to_numpy_array(net_dict['G'])
    
    net_dict['info_x_eps'] = dict()   #info is dictionary with several info saving along the process
    
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
    
    if params_['use_orthonormal']:
        from scipy.linalg import block_diag
        
        R = block_diag(params_['R'], params_['R'][1:, 1:])
    
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
            if params_['use_orthonormal']:
                THETA = polb.implicit_PHI(id_node, B, PHI, params_)
                
            if params_['use_canonical']:
                THETA = np.hstack((PHI, np.diag(b) @ PHI[:, 1:]))
            
            x_eps, num_nonzeros_vec = optimizer.l_1_optimization(b, THETA, 
                                                                 params_['noisy_measurement'], 
                                                                 params_,
                                                                 solver_default)
            '''
            x_eps = np.linalg.lstsq(THETA, b, rcond=-1)[0]/(np.sqrt(params_['length_of_time_series'])) 
            '''
        except:
            x_eps = np.zeros(L)
            if VERBOSE:
                print('Solver failed: node = ', id_node)
                
        if params_['use_canonical']:
            x_eps_can = x_eps.copy()                    
        if params_['use_orthonormal']:
            '''
            x_eps_temp = np.zeros(x_eps.shape[0] + 1)
            x_eps_temp[:params_['L']] = x_eps[:params_['L']]
            x_eps_temp[params_['L']] = 1
            x_eps_temp[params_['L'] + 1:] = x_eps[params_['L']:]
            x_eps_can_ = R @ x_eps_temp
            x_eps_can = np.delete(x_eps_can_, params_['L'])
            '''
            x_eps_can = R @ x_eps
        x_eps_dict[id_node] = x_eps_can
        if completing_on:
            x_eps_matrix = gnr_alg.completing_coeff_matrix(id_node, x_eps_can, 
                                                           x_eps_matrix, params_, 
                                                           params_)
        else:
            x_eps_matrix[:, id_node] = x_eps_can
        adj_matrix = net_dyn.get_adj_from_coeff_matrix(x_eps_matrix, params_, 
                                                       threshold, False)
        
        net_dict['A'] = adj_matrix
        
        #Update the graph structure using the links reconstructed when probing id_node
        G = nx.from_numpy_array(net_dict['A'], create_using=nx.Graph)
        edgelist = list(G.edges(data=True))
        
        net_dict['G'].add_edges_from(edgelist)
    
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
    net_dict['G'] = nx.Graph() #create the empty graph
    net_dict['G'].add_nodes_from(params_['nodelist']) 
    net_dict['A'] = nx.to_numpy_array(net_dict['G'])
    
    net_dict['info_x_eps'] = dict()   #info is dictionary with several info saving along the process
    
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
    
    if params_['use_orthonormal']:
        from scipy.linalg import block_diag
        
        R = block_diag(params_['R'], params_['R'][1:, 1:])
    
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
        try:
            if params_['use_canonical']:
                sparsity_of_vector, pareto_front, matrix_sparse_vectors = ADM.ADM_pareto(PHI, b, params_)
                x_eps_dict[id_node] = matrix_sparse_vectors
                x_eps = ADM.pareto_test(sparsity_of_vector, pareto_front, matrix_sparse_vectors)
                
        except:
            x_eps = np.zeros(L)
            if VERBOSE:
                print('Solver failed: node = ', id_node)
                
        if params_['use_canonical']:
            x_eps_can = x_eps.copy()                    
        if params_['use_orthonormal']:
            '''
            x_eps_temp = np.zeros(x_eps.shape[0] + 1)
            x_eps_temp[:params_['L']] = x_eps[:params_['L']]
            x_eps_temp[params_['L']] = 1
            x_eps_temp[params_['L'] + 1:] = x_eps[params_['L']:]
            x_eps_can_ = R @ x_eps_temp
            x_eps_can = np.delete(x_eps_can_, params_['L'])
            '''
            x_eps_can = R @ x_eps
        #x_eps_dict[id_node] = x_eps_can
        if completing_on:
            x_eps_matrix = gnr_alg.completing_coeff_matrix(id_node, x_eps_can, 
                                                           x_eps_matrix, params_, 
                                                           params_)
        else:
            x_eps_matrix[:, id_node] = x_eps_can
        adj_matrix = net_dyn.get_adj_from_coeff_matrix(x_eps_matrix, params_, 
                                                       threshold, False)
        
        net_dict['A'] = adj_matrix
        
        #Update the graph structure using the links reconstructed when probing id_node
        G = nx.from_numpy_array(net_dict['A'], create_using=nx.Graph)
        edgelist = list(G.edges(data=True))
        
        net_dict['G'].add_edges_from(edgelist)
    
    net_dict['info_x_eps'] = x_eps_dict.copy()
    net_dict['x_eps_matrix'] = x_eps_matrix
    net_dict['x_eps_matrix'][np.absolute(net_dict['x_eps_matrix']) < threshold] = 0.0
                        
    return  net_dict      
