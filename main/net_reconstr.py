'''
Script to reconstruct a network using both adapted and non-adapted basis.
'''
import cvxpy as cp
import networkx as nx 
import numpy as np
import os
from scipy.linalg import null_space, svd, block_diag
import sympy as spy
from tqdm import tqdm
import time

from EBP import tools, net_dyn, optimizer
from EBP.base_polynomial import pre_settings as pre_set 
from EBP.base_polynomial import poly_library as polb
from EBP.base_polynomial import triage as trg
from EBP import greedy_algorithms as gnr_alg
from EBP.modules.ADM import ADM
from EBP.modules.rulkov import rulkov
import lab_rulkov as lr

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
    degree = -1*np.diag(params['adj_matrix'])
    
    net_dynamics_dict['f'] = rulkov.spy_rulkov_map
    net_dynamics_dict['h'] = rulkov.spy_diff_coupling_x
    net_dynamics_dict['max_degree'] = np.max(degree)
    net_dynamics_dict['coupling'] = params['coupling']
    
    net_dyn_exp = net_dyn.spy_gen_net_dyn(net_dynamics_dict)
    
    return net_dyn_exp

def get_true_coeff_net_dyn(params):
    '''
    Obtain the true coefficient matrix for the network dynamics splitting
    into numerator and denominator.

    Parameters
    ----------
    params : dict
        
    Returns
    -------
    c_matrix_num : numpy array
        Coefficient matrix corresponding to the numerator.
    c_matrix_den : numpy array
        Coefficient matrix corresponding to the denominator.
    '''
    
    F_num, F_den = symb_net_dyn(params)
    
    dict_can_bf = polb.dict_canonical_basis(params)

    L, N = params['L'], params['number_of_vertices']
    
    c_matrix_num = np.zeros((L, N))        
    c_matrix_den = np.zeros((L, N))        
    
    for id_node in range(N):
        c_matrix_num[:, id_node] = polb.get_coeff_matrix_wrt_basis(F_num[id_node].expand(), 
                                                       dict_can_bf)
        
        c_matrix_den[:, id_node] = polb.get_coeff_matrix_wrt_basis(F_den[id_node].expand(), 
                                                       dict_can_bf)
    
    return c_matrix_num, c_matrix_den
    
    
def retrieve_dyn_sym(x_eps, params, indep_term = True, threshold = 1e-8):
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
    roud = int(-1*np.log10(threshold)) - 1
    sv[np.absolute(sv) < threshold] = 0
    sv = np.around(sv, roud)
    
    if indep_term:
        c_num_x = sv[:L]
        c_den_x = np.zeros(L)
        c_den_x[0] = 1
        c_den_x[1:] = -1*sv[L:]
        
    else:               
        c_num_x = sv[:L]
        c_den_x = np.zeros(L)
        c_den_x = -1*sv[L:]
    
    c_num_spy_x = spy.Matrix(c_num_x).n(roud)
    c_den_spy_x = spy.Matrix(c_den_x).n(roud)

    #calculate the numerator and denominator using symbolic representation
    num_x = spy_PHI.dot(c_num_spy_x)
    den_x = spy_PHI.dot(c_den_spy_x)

    symb_node_dyn = spy.cancel(num_x/den_x) #spy.ratsimp(num_x/den_x) 

    expr_n, expr_d = symb_node_dyn.as_numer_denom()
    dict_can_bf = polb.dict_canonical_basis(params)

    c_vec_num = polb.get_coeff_matrix_wrt_basis(expr_n.expand(), 
                                                dict_can_bf)        
    c_vec_den = polb.get_coeff_matrix_wrt_basis(expr_d.expand(), 
                                                dict_can_bf)           
    
    if indep_term: 
        coeff = np.concatenate((c_vec_num, c_vec_den[1:]))
    else:       
        coeff = np.concatenate((c_vec_num, c_vec_den))
    
    return symb_node_dyn, coeff

def reconstr(X_t_, params, solver_optimization = solver_default, sym_net_dyn = False):
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
    
    sym_net_dyn: boolean, optional 
        To express the network dynamics using symbolic language, 
        and consequently, calculate an error function. The default is False.
    
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
    net_dict['Y_t'] = params_['Y_t'] #x_time_series[-test_time:, :]

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
    
    #True coefficient matrix
    cn, cd = get_true_coeff_net_dyn(net_dict['params'])
    net_dict['c_true'] = np.vstack((cn, cd[1:, :]))
    
    #Coefficient matrix to be used along the process
    x_eps_matrix = np.zeros((L, N))
    
    x_eps_dict = dict()     #x_eps is created to save info about the reconstruction of nodes
    x_eps_dict['params'] = params_.copy()
    
           
    B_ = B.copy()
    for id_node in tqdm(id_trial, **tqdm_par):
        b = B_[:, id_node]
        
        x_eps_dict[id_node] = dict()
        if np.mod(id_node, 2):
            start = time.time()
            x_eps = np.linalg.lstsq(PHI, b, rcond=-1)[0]#/(np.sqrt(params_['length_of_time_series']))
            end = time.time()            
            
            x_eps_dict[id_node]['time'] = end - start
            
            x_eps_matrix[:params_['L'], id_node] = x_eps
                        
            if sym_net_dyn:
                x_eps_can = x_eps_matrix[:, id_node].copy()
                net_dict['sym_node_dyn'][id_node], c = retrieve_dyn_sym(x_eps_can, params_, 
                                                                        indep_term = False)
                #x_eps_matrix[:, id_node] = c.copy()
        else:
            start = time.time()
            try:
                if sym_net_dyn:
                    THETA = np.hstack((PHI, np.diag(b) @ PHI[:, 1:]))
                else:
                    THETA = np.hstack((PHI, -1*np.diag(b) @ PHI[:, 1:]))
                '''
                x_eps, num_nonzeros_vec = optimizer.l_1_optimization(b, THETA, 
                                                                     params_['noisy_measurement'], 
                                                                     params_,
                                                                     solver_default)            
                '''
                x_eps = np.linalg.lstsq(THETA, b, rcond=-1)[0]#/(np.sqrt(params_['length_of_time_series']))
                
            except:
                x_eps = np.zeros(L)
                if VERBOSE:
                    print('Solver failed: node = ', id_node)
                    
            end = time.time()
                
            if params_['use_orthonormal']:        
                R = params_['R']
                R_ = block_diag(R, R)
                c_num_x = x_eps.copy()[:params_['L']]
                c_den_x = np.zeros(params_['L'])
                c_den_x[0] = 1
                c_den_x[1:] = x_eps.copy()[params_['L']:]
                x = np.concatenate((c_num_x, c_den_x))
                x_eps_can_temp = R_ @ x.copy()                            
                x_eps_can = np.concatenate((x_eps_can_temp[:params_['L']],
                                            x_eps_can_temp[params_['L']+1:]))
                
            else:   
                x_eps_can = x_eps.copy()           
           
            x_eps_dict[id_node]['x_eps_can'] = x_eps_can
            x_eps_dict[id_node]['time'] = end - start
            
            x_eps_matrix[:, id_node] = x_eps_can
            if sym_net_dyn:
                net_dict['sym_node_dyn'][id_node], c = retrieve_dyn_sym(x_eps_can, params_, 
                                                                    indep_term = True)
        
                #x_eps_matrix[:, id_node] = c
            
    net_dict['info_x_eps'] = x_eps_dict.copy()
    net_dict['x_eps_matrix'] = x_eps_matrix
    net_dict['x_eps_matrix'][np.absolute(net_dict['x_eps_matrix']) < threshold] = 0.0
    
    if sym_net_dyn:
        net_dict['error'] = uniform_error(net_dict, num_samples = 50, time_eval = 5)         
                        
    return net_dict      


def ADM_reconstr(X_t_, params, plot_pareto = False, sym_net_dyn = True):
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
    sym_net_dyn: boolean, optional 
        To express the network dynamics using symbolic language, 
        and consequently, calculate an error function. The default is False.
    
    
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
    net_dict['Y_t'] = params['Y_t'] #x_time_series[-test_time:, :]
    
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
    
    #True coefficient matrix
    cn, cd = get_true_coeff_net_dyn(net_dict['params'])
    net_dict['c_true'] = np.vstack((cn, cd))
    
    #Coefficient matrix to be used along the process
    x_eps_matrix = np.zeros((L, N))
    
    x_eps_dict = dict()     #x_eps is created to save info about the reconstruction of nodes
    x_eps_dict['params'] = params_.copy()
    
               
    B_ = B.copy()
    for id_node in tqdm(id_trial, **tqdm_par):
        b = B_[:, id_node]
        
        x_eps_dict[id_node] = dict()
        
        if np.mod(id_node, 2):
            start = time.time()
            x_eps = np.linalg.lstsq(PHI, b, rcond=-1)[0]#/(np.sqrt(params_['length_of_time_series']))
            end = time.time()            
            
            x_eps_dict[id_node]['time'] = end - start
            
            x_eps_matrix[:params_['L'], id_node] = x_eps
            
            if sym_net_dyn:
                x_eps_can = x_eps_matrix[:, id_node].copy()
                net_dict['sym_node_dyn'][id_node], c = retrieve_dyn_sym(x_eps_can, params_, 
                                                                     indep_term = False)
                #x_eps_matrix[:, id_node] = c.copy()
        else:
            start = time.time()
            THETA = np.hstack((PHI, np.diag(b) @ PHI))
            sparsity_of_vector, pareto_front, matrix_sparse_vectors = ADM.ADM_pareto(THETA, params_)
            if plot_pareto:
                lr.plot_pareto_front(sparsity_of_vector, pareto_front)
            
            x_eps_dict[id_node]['matrix_sparse_vectors'] = matrix_sparse_vectors
            
            x_eps, threshold_ADM = ADM.pareto_test(sparsity_of_vector, pareto_front, matrix_sparse_vectors)
            end = time.time()
            x_eps_dict[id_node]['time'] = end - start
            
            x_eps_can = x_eps.copy()                                    
            
            print('Calculating symbolic expression')
            
            net_dict['sym_node_dyn'][id_node], c = retrieve_dyn_sym(x_eps_can, params_, 
                                                                    indep_term = False,
                                                                    threshold = threshold)
            x_eps_matrix[:, id_node] = c    
            
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
    
    if sym_net_dyn:
        net_dict['error'] = uniform_error(net_dict, num_samples = 50, time_eval = 1)         
    
    return net_dict      

def kernel_calculation(X_t_, params, if_spectral = False, if_kernel = False):
    '''
    Calculates the dimension of the kernel of the library matrix
    
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
            'PHI' : numpy array
                Library matrix
            'PHI.T PHI' : numpy array
                Library matrix multiplied by its transpose
            'params' : dict
                Core dictionary of the code to track the relevant information of the reconstruction.
            'info_x_eps' : dict
                Trace the reconstruction for each node. Save the information of the coefficient vector of each node.
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
    M = 2*PHI.shape[1]
    params_['power_indices'] = np.vstack((params_['power_indices'], params_['power_indices'][1:,:]))
    
    id_trial = params_['id_trial']
    
    x_eps_dict = dict()     #x_eps is created to save info about the reconstruction of nodes
    x_eps_dict['params'] = params_.copy()
               
    B_ = B.copy()
    for id_node in tqdm(id_trial, **tqdm_par):
        b = B_[:, id_node]
        x_eps_dict[id_node] = dict()
        
        THETA = np.hstack((PHI, np.diag(b) @ PHI))
        
        if if_spectral:
            s = svd(THETA, compute_uv=False)
            x_eps_dict[id_node]['spectral'] = s
            
        ker_THETA = null_space(THETA, rcond=np.finfo(float).eps *M)
        x_eps_dict[id_node]['dim_ker'] = ker_THETA.shape[1]
        if if_kernel:
            x_eps_dict[id_node]['ker'] = ker_THETA        
        
    net_dict['info_x_eps'] = x_eps_dict.copy()
                        
    return net_dict      


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
            error = tools.RSME(y_true[1:], Z[1:, id_node])
            
            error_matrix[id_node] = error_matrix[id_node]\
                + error**2/num_samples   
        
    return np.sqrt(error_matrix)
