"""
Network reconstruction for fourier basis 

Created on Mon feb  21 11:01:11 2022

@author: Edmilson Roque dos Santos
"""
import cvxpy as cp
import networkx as nx
import numpy as np

from . import fourier_library as fourier_lb
from . import triage as trg

from .. import net_dyn, optimizer

threshold_connect = 1e-8    #default threshold for Network Selection
tolp = 1e-5    #default tol determines the Model Selection parameter
default_args_alg = dict()
default_args_alg['noise_step'] = 10
default_args_alg['noise_min'] = 1e-2
default_args_alg['noise_max'] = 1e-0
default_args_alg['search_array'] = np.linspace(default_args_alg['noise_min'],
                                               default_args_alg['noise_max'],
                                               default_args_alg['noise_step'])

MS_dict_default = dict()
MS_dict_default['predict_num_iters'] = 5

VERBOSE = True

def get_mapping_indices_nodes(params):
    '''
    Identify the indices relative to the basis functions of nodes in the
    'cluster_list'.

    Parameters
    ----------
    params : dict
        
    Returns
    -------
    indices_cluster : numpy array
        indices vector to select columns in the library matrix.

    '''    
    indices = params['power_indices'].copy()
    index_set = np.arange(0, indices.shape[0], 1, dtype = int)
    cluster_list = params['cluster_list'].copy()
    
    indices_cluster = np.array([0])     #independent term is always present
    
    for l in index_set:
        if np.any(indices[l, cluster_list] > 0):
            indices_cluster = np.concatenate((indices_cluster, [l]),
                                         axis = None)
    return indices_cluster

def completing_coeff_matrix(id_node, x_eps_can, x_eps_matrix, params_node, params,
                            threshold = threshold_connect):
    '''
    Complete coefficient matrix from a coefficient vector. It satisfies 
    pairwise symmetry. 

    Parameters
    ----------
    id_node : int
        Probed node which induces the connections to its neighbors.
    x_eps_can : numpy array
        Coefficient vector to corresponding to probed node id_node.
    x_eps_matrix : numpy array
        Coefficient matrix to be completed.
    params_node : dict
        parameteres dictionary containing info about the cluster that id_node
        belongs to.
    params : dict
        
    Returns
    -------
    x_eps_matrix : numpy array
        Completed coefficient matrix.

    '''
    
    x_eps_matrix[params_node['indices_cluster'], id_node] = x_eps_can
    x_eps_matrix[np.absolute(x_eps_matrix) < threshold] = 0
    
    
    N = params['number_of_vertices']
    
    for start in range(2):
        id_vec = []
        group_indices = np.arange(0+start, 2*params['max_deg_harmonics'], 2, 
                                  dtype = int)
        
        for deg in range(params['max_deg_harmonics']):
            id_vec = np.append(id_vec, np.arange(1 + N*group_indices[deg], 
                                                 N + 1 + N*group_indices[deg], 
                                                 dtype = int))
        id_vec = np.array(id_vec, dtype = int)
        y_eps_matrix = x_eps_matrix[id_vec, :]
        id_nonzero_basis = np.where(np.absolute(y_eps_matrix[:, id_node]) >= threshold)[0]
        
        power_indices = params['power_indices'][id_vec, :]
        power_indices = power_indices[id_nonzero_basis, :]
        for l in range(power_indices.shape[0]):
            exp_vector = power_indices[l, :]
            entries_ones = np.where( exp_vector > 0)[0]
            mask_nodes_identifier = ~(entries_ones == id_node)
            
            for id_conn in entries_ones[mask_nodes_identifier]:
                exp_vector_temp = exp_vector.copy()
                #Permutation of exponent vector support
                exp_vector_temp[id_node] = exp_vector[id_conn]
                exp_vector_temp[id_conn] = exp_vector[id_node]
                mask_index_identifier = params['power_indices'][id_vec, :] == exp_vector_temp
                mask_index_identifier = np.all(mask_index_identifier, axis = 1)
                
                #The coefficient matrix is completed when the permutation is satisfied
                #The term in the expansion of id_conn should correspond to the exp_vector 
                
                y_eps_matrix[mask_index_identifier, id_conn] = y_eps_matrix[id_nonzero_basis[l], id_node] #x_eps_can
        x_eps_matrix[id_vec, :] = y_eps_matrix
            
    return x_eps_matrix


def model_selection(node, epsilon, x_eps, params, 
                    tol = tolp, threshold = threshold_connect,
                    select_criterion = 'crit_3', 
                    model_selection_dict = MS_dict_default):
    '''
    To select the coefficient vector of a node in the B_eps relaxing path algorithm.

    Parameters
    ----------
    node : int
        Probed node at the current step epsilon for the B_eps relaxing path algorithm.
    epsilon : numpy float64
        Relaxing parameter.
    x_eps : numpy array - size (num_of_basis_functions, num_of_nodes)
        DESCRIPTION.
    B_dot : numpy array - size (predict_num_iters, number_of_vertices)
        Matrix of multivariate time series for prediction.
    params : dict
    tol : float, optional
        Tolerance. The default is 1e-5.
    select_criterion : TYPE, optional
        Select criterion for Model Selection. The default is 'crit_2'.
        Criterion 3 based on notes updated online. 
    model_selection_dict: dict, optional.
        Model selection dictionary .The default is the dictionary MS_dict_default.
    Returns
    -------
    bool
        Model selection result: if the criterion is satisfied or not.

    '''
    params_ = params.copy()
    
    if select_criterion == 'crit_2':
        B_dot = model_selection_dict['B_dot']
        PHI_dot = model_selection_dict['PHI_dot']
        params_dot = model_selection_dict['params_dot']
        
        if params_['normalize_cols']:
            const = np.sqrt(PHI_dot.shape[0])*params_dot['norm_column'][params_['indices_cluster']]
            zeta = np.linalg.norm(const*PHI_dot[:, params_['indices_cluster']] @ x_eps - B_dot[1:, node])
        else: 
            zeta = np.linalg.norm(np.sqrt(PHI_dot.shape[0])*PHI_dot[:, params_['indices_cluster']] @ x_eps - B_dot[1:, node])
        
        if VERBOSE:
            print(zeta, epsilon, np.abs(zeta - epsilon))
        if np.abs(zeta - epsilon) < tol:
            return True
        else:
            return False
    
    if select_criterion == 'crit_3':
        if params_['eps_counter'] >= 1:
            supp_t_1 = np.where(np.absolute(params_['x_eps_path'][:, params_['eps_counter']]) >= threshold)[0]
            supp_t = np.where(np.absolute(params_['x_eps_path'][:, params_['eps_counter'] - 1]) >= threshold)[0] 
            
            if supp_t.shape[0] > 0:
                mask_t_isin_t_1 = np.isin(supp_t, supp_t_1)
                mask_t_1_isin_t = np.isin(supp_t_1, supp_t)
                
                support_change = np.append(supp_t[~mask_t_isin_t_1], supp_t_1[~mask_t_1_isin_t])
                
                rel_change = np.abs(support_change.shape[0]) #/supp_t.shape[0])
                
                if rel_change < tol:
                    return True            
            else:
                return False
        else:
            return False            
            
    
def B_eps_algorithm(B, PHI_incluster, params, 
                    gr_alg, tol = tolp, threshold = threshold_connect,
                    fixed_search_set = False,
                    relaxing_path = default_args_alg['search_array'],
                    select_criterion = 'crit_3',
                    solver_default = cp.ECOS, 
                    model_selection_dict = MS_dict_default):   
    '''
    B_eps relaxing algorithm solves the intra-connections for a given cluster 
    of nodes listed in params['cluster_list']

    Parameters
    ----------
    B : numpy array
        DESCRIPTION.
    PHI_incluster: numpy array 
        DESCRIPTION.
    params : TYPE
        DESCRIPTION.
    gr_alg : TYPE
        DESCRIPTION.
    tol : float, optional
        Tolerance for Model selection method. The default is tolp = 1e-5.
    threshold : float, optional
        Value that defines the support of a vector. Any entry that has absolute
        value smaller than threshold is equal to zero. 
        The default is threshold_connect = 1e-8.
    fixed_search_set : boolean, optional
        Define if the relaxing path is used in terms of a pre-defined array or given
        by the user. 
        The default is False which considers two values determined by 
        [norm_min, 1.1*norm_min]. 
    relaxing_path: TYPE, optional
        Array of epsilon values to search. The default is default_args_alg['search_space'].
    select_criterion: str, optional
        Criterion in Model Selection
    solver_default:    
        Solver to be used in the convex minimization problem
    model_selection_dict:
        Model selection dictionary
    Returns
    -------
    b_eps_alg : TYPE
        DESCRIPTION.

    '''
    
    params_ = params.copy()
    N = B.shape[1]
    
    #Boolean to determine if the relaxing path is broken as soon as a solution 
    #is found.
    break_relaxing_path = False
    
    #Cluster list identifies set of nodes.
    cluster_list = np.array(params['cluster_list'], dtype = int)
    size_id_trial = cluster_list.shape[0]
    
    #id_trial determines the list of nodes be probed 
    id_trial = np.array(params['id_trial'], dtype = int)  
    
    #Generate an array of indices corresponding to nodes inside the cluster
    #to be used in select specific terms in library matrix and power indices.
    params_['indices_cluster'] = get_mapping_indices_nodes(params_)
    
    #Selecting specific entries in power indices
    params_['power_indices'] = params['power_indices'][params_['indices_cluster'], :]
    params_['power_indices'] = params_['power_indices'][:, cluster_list]
     
    params_['number_of_vertices'] = len(params['cluster_list'])
    params_['cluster_list'] = cluster_list
    
    PHI = PHI_incluster.copy()
    PHI_incluster = PHI[:, params_['indices_cluster']]
    
    if params_['use_orthonormal']:
        L = params_['R'].shape[0]
    if params_['use_canonical']:
        L = params_['L']
    
    b_eps_alg = dict()      #Create b-eps algorithm dictionary
    b_eps_alg['G'] = nx.Graph()
    b_eps_alg['G'].add_nodes_from(cluster_list)
    b_eps_alg['A'] = nx.to_numpy_array(b_eps_alg['G'])
    b_eps_alg['feasible_set'] = gr_alg.get('feasible_set', np.array([False]*B.shape[1]))    
    b_eps_alg['comp_feasible_set'] = gr_alg.get('comp_feasible_set', np.array([True]*B.shape[1]))
    b_eps_alg['tol'] = gr_alg.get('tol', tol)
    
    
    #Coefficient matrix to be used along the process
    #The greedy part of the algorithm is reserved in this matrix
    x_eps_matrix = gr_alg.get('x_eps_matrix', np.zeros((L, N)))
    l2_coeff_matrix = np.zeros((L, size_id_trial))
    
    x_eps_dict = dict()     #x_eps is created to save info about the reconstruction of nodes
    x_eps_dict['params'] = params_.copy()
               
    node_counter = 0
    params_node = params_.copy()
    B_ = B.copy()
    for id_node in id_trial:
        b = B_[:, id_node]
        
        x_eps_dict[id_node] = dict()
        
        if params_['normalize_cols']:
            const = np.sqrt(PHI_incluster.shape[0])*params['norm_column'][params_['indices_cluster']]
            min_l2_sol = (np.linalg.pinv(const*PHI_incluster) @ b)
            noise_min = np.linalg.norm(const*PHI_incluster @ min_l2_sol - b)
        else:
            min_l2_sol = (np.linalg.pinv(np.sqrt(PHI_incluster.shape[0])*PHI_incluster) @ b)
            noise_min = np.linalg.norm(np.sqrt(PHI_incluster.shape[0])*PHI_incluster @ min_l2_sol - b)
        
        if fixed_search_set:
            noise_vector = relaxing_path
            x_eps_dict['noise_vector'] = noise_vector
            
        else:
            #10% of the minimum noise
            noise_vector = np.array([noise_min, 1.1*noise_min], 
                                    dtype = np.float64)
        if VERBOSE:   
            print(id_node, noise_min)
        x_eps_dict[id_node]['noise_min'] = noise_min
        
        if select_criterion == 'crit_3':
            params_node['x_eps_path'] = np.zeros((min_l2_sol.shape[0], 
                                                  noise_vector.shape[0]))
            
            x_eps_dict[id_node]['x_eps_path'] = np.zeros((min_l2_sol.shape[0], 
                                                  noise_vector.shape[0]))
            
        l2_coeff_matrix[params_node['indices_cluster'], node_counter] = min_l2_sol        
        if params_['use_canonical']:
            x_eps_dict[id_node]['min_l2_sol'] = min_l2_sol 

        if params_['use_orthonormal']:
            R = params_['R']
            full_vec = R @ l2_coeff_matrix
            x_eps_dict[id_node]['min_l2_sol'] = full_vec[params_node['indices_cluster']]
            x_eps_dict['params']['R'] = R
            
        eps_counter = 0
        
        #Flag to stop MS step if MS is satisfied once.
        MS_flag = True
        for epsilon in noise_vector:
            #From Candes: the irrelevant entries are less than \epsilon/sqrt(N)
            threshold_noise = epsilon/np.sqrt(L)
            params_node['noise_magnitude'] = epsilon
            
            try:
                x_eps, num_nonzeros_vec = optimizer.l_1_optimization(b, PHI_incluster, 
                                                                     params_node['noisy_measurement'], 
                                                                     params_node,
                                                                     solver_default)
            except:
                x_eps = np.zeros(min_l2_sol.shape[0])
                if not VERBOSE:
                    print('Solver failed: node = ', id_node)
                
            if select_criterion == 'crit_3':
                params_node['x_eps_path'][:, eps_counter] = x_eps
                params_node['eps_counter'] = eps_counter
                
            #Model selection is performed
            MS = model_selection(id_node, epsilon, x_eps, params_node, tol,
                                 threshold_noise, select_criterion, model_selection_dict)
            
            if params_['use_canonical']:
                x_eps_can = x_eps.copy()                    
            if params_['use_orthonormal']:
                c_vec = np.zeros(L)
                c_vec[params_node['indices_cluster']] = x_eps
                x_eps_can = R @ c_vec
                x_eps_can = x_eps_can[params_node['indices_cluster']]
            
            x_eps_dict[id_node][epsilon] = x_eps_can
            if select_criterion == 'crit_3':
                x_eps_dict[id_node]['x_eps_path'][:, eps_counter] = x_eps_can
                
            if MS and not isinstance(num_nonzeros_vec, str) and MS_flag:
                x_eps_matrix = completing_coeff_matrix(id_node, x_eps_can, x_eps_matrix, params_node, params)
                c_matrix = x_eps_matrix[params_node['indices_cluster'], :]
                adj_matrix = net_dyn.get_adj_from_coeff_matrix(c_matrix[:, params_node['cluster_list']], 
                                                               params_node, threshold_noise, True)
                x_eps_dict['x_eps_matrix'] = x_eps_matrix
                b_eps_alg['A'] = adj_matrix
                
                #Update the graph structure using the links reconstructed when probing id_node
                G = nx.from_numpy_array(b_eps_alg['A'], create_using=nx.Graph)
                mapping = dict(zip(G, params_node['cluster_list']))
                G = nx.relabel_nodes(G, mapping)
                edgelist = list(G.edges(data=True))
                
                b_eps_alg['G'].add_edges_from(edgelist)
                b_eps_alg['feasible_set'][id_node] = True
                b_eps_alg['comp_feasible_set'][id_node] = False
                b_eps_alg['node {}'.format(id_node)] = []
                
                
                if select_criterion == 'crit_3' and MS_flag:
                    x_eps_dict[id_node]['eps_flag'] = dict()
                    x_eps_dict[id_node]['eps_flag']['eps'] = epsilon
                    x_eps_dict[id_node]['eps_flag']['eps_counter'] = eps_counter                
                    MS_flag = False
                    
                if break_relaxing_path:                
                    break       
            
            eps_counter = eps_counter + 1
        if not MS:
            neigh_node = np.array(list(b_eps_alg['G'][id_node]), dtype = int)
            mask_neigh = np.isin(cluster_list, neigh_node)
            mask_node = np.isin(cluster_list, id_node)
            b_eps_alg['node {}'.format(id_node)] = cluster_list[~mask_neigh & ~mask_node]
        
        node_counter = node_counter + 1    
    
    b_eps_alg['info_x_eps'] = x_eps_dict.copy()
    b_eps_alg['x_eps_matrix'] = x_eps_matrix
    b_eps_alg['x_eps_matrix'][np.absolute(b_eps_alg['x_eps_matrix']) < threshold] = 0.0
    
    return b_eps_alg

def MS_dict(X_t_, model_selection_dict, params, library_method = fourier_lb.library_matrix):
    '''
    Generate model selection dictionary in case criterion is 'crit_2'.

    Parameters
    ----------
    X_t_ : numpy array
        Multivariate time series.
    model_selection_dict : dict
        Model selection dictionary.
    params : dict
        
    Returns
    -------
    X_t : numpy array
        Domain return map.
    B : numpy array 
        Image return map.
    model_selection_dict : dict
        Model selection dictionary.

    '''
    X_t = X_t_[:-1 - model_selection_dict['predict_num_iters'], :]
    B = X_t_[1: - model_selection_dict['predict_num_iters'], :]
    model_selection_dict['B_dot'] = X_t_[- model_selection_dict['predict_num_iters']:, :]
        
    params_dot = params.copy()
    params_dot['length_of_time_series'] = model_selection_dict['B_dot'][:-1, :].shape[0]
            
    PHI_dot, params_dot = library_method(model_selection_dict['B_dot'][:-1, :], params_dot)    
    model_selection_dict['PHI_dot'] = PHI_dot
    model_selection_dict['params_dot'] = params_dot
    
    return X_t, B, model_selection_dict

def GR_algorithm(X_t_, initial_partition, params, tol = tolp,
                 threshold = threshold_connect, 
                 fixed_search_set = False,
                 relaxing_path = default_args_alg['search_array'],
                 select_criterion = 'crit_3', 
                 solver_default = cp.ECOS, 
                 model_selection_dict = MS_dict_default,
                 library_method = fourier_lb.library_matrix):
    '''
    Greedy network reconstruction algorithm
    
    Given the multivariate time series the algorithm returns a graph.

    Parameters
    ----------
    X_t_ : numpy array - size (length_of_time_series, number_of_vertices)
        Multivariate time series.
    
    initial_partition : list
        Initial graph partition.
    params : dict
    tol : float, optional
        Tolerance for Model selection method. The default is tolp = 1e-5.
    threshold : float, optional
        Value that defines the support of a vector. Any entry that has absolute
        value smaller than threshold is equal to zero. 
        The default is threshold_connect = 1e-8.
    fixed_search_set : boolean, optional
        Define if the relaxing path is used in terms of a pre-defined array or given
        by the user. The default is False which considers two values determined by 
        [norm_min, 1.1*norm_min] 
    relaxing_path: TYPE, optional
        Array of epsilon values to search. The default is default_args_alg['search_space'].
    select_criterion: str, optional
        Criterion in Model Selection
    solver_default: cp solver   
        Solver to be used in the convex minimization problem
    model_selection_dict:
        Model selection dictionary
    Returns
    -------
    gr_alg : dict
        Dictionary encoding Adjacency matrix, Graph structure and Library matrix.

    '''
    params['nodelist'] = np.arange(0, X_t_.shape[1], 1, dtype = int)
    params_ = params.copy()
    
    gr_alg = dict()          #create greedy algorithm dictionary to save info
    gr_alg['G'] = nx.Graph() #create the empty graph
    gr_alg['G'].add_nodes_from(params['nodelist']) 
    gr_alg['A'] = nx.to_numpy_array(gr_alg['G'])
    
    gr_alg['feasible_set'] = np.array([False]*X_t_.shape[1])        #feasible set is the list of nodes which has been successfully reconstructed
    gr_alg['comp_feasible_set'] = np.array([True]*X_t_.shape[1])    #com_feasible set is the complement of the feasible_set
    gr_alg['info_x_eps'] = dict()   #info is dictionary with several info saving along the process
    
    gr_alg['inter_cluster_connections'] = False     #Boolean variable to identify if we are solving the inter-cluster links
    
   
    params_['number_of_vertices'] = params['nodelist'].shape[0]
    params_ = trg.triage_params(params_)
    
    ##Select the data to be fed into library matrix
    if select_criterion == 'crit_2':
        
        X_t, B, model_selection_dict = MS_dict(X_t_, model_selection_dict, 
                                               params_,
                                               library_method)
        
    else: 
        X_t = X_t_[:-1, :]
        B = X_t_[1:, :]
        
    params_['length_of_time_series'] = X_t.shape[0]
    
    PHI, params_ = library_method(X_t, params_)
    gr_alg['PHI'] = PHI
    gr_alg['PHI.T PHI'] = PHI.T @ PHI 
    gr_alg['params'] = params_.copy()   #for reference we save the params used
    if params_['use_orthonormal']:
        R = params_['R']
        inv_R = np.linalg.inv(R)
    
    #For each cluster pre-processed in the network, 
    #the algorithm reconstructs the intra-cluster connnections
    
    id_cluster = 0
    for cluster_list in initial_partition:
        params_['cluster_list'] = cluster_list
        params_['id_trial'] = cluster_list.copy()
        
        PHI_incluster = PHI.copy()
                
        #It calls the B_eps relaxing path algorithm
        b_eps_alg = B_eps_algorithm(B, PHI_incluster, params_, gr_alg, tol, 
                                    threshold, fixed_search_set, relaxing_path, 
                                    select_criterion, solver_default,
                                    model_selection_dict)
        
        edgelist = list(b_eps_alg['G'].edges(data=True))
        gr_alg['G'].add_edges_from(edgelist)
        gr_alg['feasible_set'] = b_eps_alg.get('feasible_set')
        gr_alg['comp_feasible_set'] = b_eps_alg.get('comp_feasible_set')
        gr_alg['x_eps_matrix'] = b_eps_alg.get('x_eps_matrix')
        gr_alg['info_x_eps'][id_cluster] = dict()
        gr_alg['info_x_eps'][id_cluster] = b_eps_alg['info_x_eps'].copy()
        for id_node in cluster_list:
            gr_alg['node {}'.format(id_node)] = b_eps_alg['node {}'.format(id_node)]
        id_cluster = id_cluster + 1
        if VERBOSE:
            print('feasible_set: ', params['nodelist'][gr_alg['feasible_set']])
            print('edgelist: ', edgelist) 
    
    #The second stage starts for those nodes, whose reconstruction failed 
    #in the first stage, i.e., nodes which share inter-cluster connections. 
    
    gr_alg['inter_cluster_connections'] = True
    
    card_comp_set = params['nodelist'][gr_alg['feasible_set']].shape[0]       #update the list of feasible nodes
    gr_alg['tol'] = tol 
    
    
    if not card_comp_set == params['nodelist'].shape[0] and gr_alg['inter_cluster_connections']:
        comp_feasible_nodes = params['nodelist'][gr_alg['comp_feasible_set']]
        params_['cluster_list'] = np.array(comp_feasible_nodes)
        params_['id_trial'] = params_['cluster_list'].copy()
            
        #The matrix of data must be updated. The update is to carry the information
        #learned in the previous steps. 
        
        if gr_alg['params']['use_orthonormal']:
            B_ = B - np.sqrt(PHI.shape[0])*PHI @ (inv_R @ gr_alg['x_eps_matrix'])
            
        if gr_alg['params']['use_canonical']:
            if gr_alg['params']['normalize_cols']:
                const = np.sqrt(PHI.shape[0])*gr_alg['params']['norm_column']
                B_ = B - (const*PHI) @ gr_alg['x_eps_matrix']
            else: 
                B_ = B - (np.sqrt(PHI.shape[0])*PHI) @ gr_alg['x_eps_matrix']
                
        PHI_incluster = PHI.copy()
        
        b_eps_alg = B_eps_algorithm(B_, PHI_incluster, 
                                    params_, gr_alg, tol, threshold,
                                    fixed_search_set, relaxing_path,
                                    select_criterion, solver_default,
                                    model_selection_dict)
        
        edgelist = list(b_eps_alg['G'].edges(data=True))
        gr_alg['G'].add_edges_from(edgelist)
        gr_alg['feasible_set'] = b_eps_alg.get('feasible_set')
        gr_alg['comp_feasible_set'] = b_eps_alg.get('comp_feasible_set')
        gr_alg['x_eps_matrix'] = b_eps_alg.get('x_eps_matrix')
        gr_alg['info_x_eps'][-1] = dict()
        gr_alg['info_x_eps'][-1] = b_eps_alg['info_x_eps'].copy()
       
    
    gr_alg['A'] = nx.to_numpy_array(gr_alg['G'])       
    return gr_alg






