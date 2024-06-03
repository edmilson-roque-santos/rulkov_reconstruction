"""
Collection of methods about Network dynamics.

Created on --- 2020

@author: Edmilson Roque dos Santos
"""

import numpy as np
from numpy.random import default_rng
import sympy as spy

def net_dynamics(x, args):
    '''
    Iterates the network dynamics at one time step.    

    Parameters
    ----------
    x : numpy array - shape (N,)
        State at time step t.
    args : dict
    Dictionary with network dynamics information content.
    Keys: 
        'coupling' : float
            coupling strength
        'max_degree' : int
            maximum degree of the network
        'f' : function
            isolated map
        'h' : function
            coupling function
    Returns
    -------
    numpy array
        Next state at time step t + 1.

    '''
    Lambda = args['coupling']
    max_degree = args['max_degree']
    f_isolated = args['f']
    h_coupling = args['h']
    
    return f_isolated(x) + h_coupling(x, args['adj_matrix'])*Lambda/(max_degree)
        
def gen_net_dynamics(number_of_iterations, args, use_noise = False):
    '''
    It generates a trajectory of the network dynamics with length given by
    "number_of_iterations". The initial condition is given by a uniform
    random distribution in the half-open interval [0.0, 1.0).

    Parameters
    ----------
    number_of_iterations : TYPE
        DESCRIPTION.
    args : dict
        Dictionary with network dynamics information content.
        Keys: 
            'random_seed' : int
                Seed for the pseudo random generator.
            'adj_matrix' : numpy array
                Adjacency matrix 
            'eps' : float
                noise magnitude
            'coupling' : float
                coupling strength
            'max_degree' : int
                maximum degree of the network
            'f' : function
                isolated map
            'h' : function
                coupling function
    use_noise : boolean
        Add dynamical noise to the network dynamics.
    Returns
    -------
    time_series : numpy array
        Multivariate time series of the network dynamics.

    '''
    random_seed = args.get('random_seed', 1) 
    rng = default_rng(random_seed)  #Initializes an instance of the pseudo random generator;
    
    A = args['adj_matrix']
    
    N = 2*A.shape[0] #Number of vertices
    
    number_of_iterations = args['transient_time'] + number_of_iterations
    
    time_series  = np.zeros((int(number_of_iterations), N))
    x_state = np.zeros(N)
    initial_condition = rng.random(N) #Random initial condition
    initial_condition = np.asarray(initial_condition, dtype=np.float64)
    
    iterator = range(1, int(number_of_iterations))

    time_series[0, :] = initial_condition 
    x_state = initial_condition.copy()
    
    for i in iterator: 
        x_state = net_dynamics(x_state, args)
        #To add dynamical noise, it is given as
        if use_noise:
            x_state = x_state + args['eps']*rng.random(N)
        time_series[i, :] = x_state.copy()
        
    return time_series[args['transient_time']:, :]

def spy_gen_net_dyn(args):
    '''
    Symbolic network dynamics map.

    Parameters
    ----------
    args : dict
        Dictionary with network dynamics information content.
        Keys: 
            'adj_matrix' : numpy array
                Adjacency matrix 
            'coupling' : float
                coupling strength
            'max_degree' : int
                maximum degree of the network
            'f' : sympy Matrix
                symbolic isolated map
            'h' : sympy Matrix
                symbolic coupling function

    Returns
    -------
    F : sympy Matrix
        Symbolic network dynamics.

    '''
    A = args['adj_matrix']
    
    N = 2*A.shape[0] #Number of vertices
    
    x_t = [spy.symbols('x_{}'.format(j)) for j in range(0, N)]
    
    Lambda = args['coupling']
    max_degree = args['max_degree']
    f_isolated = args['f']
    h_coupling = args['h']
    
    f_dict = f_isolated(x_t)
    cplg_dict = h_coupling(x_t, args['adj_matrix'])
    F_num = f_dict['num'] + cplg_dict['num']*Lambda/(max_degree)
    F_den = f_isolated(x_t)['den']
    return F_num, F_den
    

def get_adj_row_from_coeff_vec(id_node, coefficient_vector, parameters, 
                               threshold_connect, add_weight = False):
    '''
    

    Parameters
    ----------
    id_node : TYPE
        DESCRIPTION.
    coefficient_vector : TYPE
        DESCRIPTION.
    parameters : TYPE
        DESCRIPTION.
    threshold_connect : TYPE
        DESCRIPTION.
    add_weight : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    A_id_node : TYPE
        DESCRIPTION.

    '''
    
    N = parameters['number_of_vertices']
    
    A_id_node = np.zeros(N)
    nodelist = np.arange(N)
    
    id_nonzero_basis = np.where(np.absolute(coefficient_vector) >= threshold_connect)[0]
    
    power_indices = parameters['power_indices'][id_nonzero_basis, :]

    if not add_weight:              
        nodes_to_connect = []
    
        for l in range(power_indices.shape[0]):
            exp_vector = power_indices[l, :]
            
            entries_ones = np.where( exp_vector > 0)[0]
            nodes_to_connect = np.append(nodes_to_connect, entries_ones)
            
        nodes_to_connect = np.unique(np.array(nodes_to_connect, dtype = int))
        
        mask_contains_link = np.isin(nodelist, nodes_to_connect)
        A_id_node[mask_contains_link] = np.ones(N)[mask_contains_link]
        A_id_node[id_node] = 0    
        
        return A_id_node


    if add_weight:
        
        weights_edges = np.zeros((nodelist.shape[0], id_nonzero_basis.shape[0]))
        for l in range(power_indices.shape[0]):
            exp_vector = power_indices[l, :]
            
            entries_ones = np.where( exp_vector > 0)[0]
            
            weight = np.absolute(coefficient_vector[id_nonzero_basis[l]]) 
            weights_edges[entries_ones, l] = np.ones(entries_ones.shape[0])*weight 
        
        if len(id_nonzero_basis) > 0:
            weighted_edges = np.max(weights_edges, axis = 1)    
            A_id_node = weighted_edges.copy()
            A_id_node[id_node] = 0
            
        return A_id_node

def get_adj_from_coeff_matrix(coefficient_matrix, parameters, threshold_connect,
                              add_weight = False):
    '''
    

    Parameters
    ----------
    coefficient_matrix : TYPE
        DESCRIPTION.
    parameters : TYPE
        DESCRIPTION.
    threshold_connect : TYPE
        DESCRIPTION.

    Returns
    -------
    A : TYPE
        DESCRIPTION.

    '''
    #N = coefficient_matrix.shape[1]
    N = parameters['number_of_vertices']
    A = np.zeros((N, N))
        
    for i in range(N):
        '''
        This definition of the adjacency matrix respects networkx definition.
        "For directed graphs, explicitly mention create_using=nx.DiGraph, and 
        entry i,j of A corresponds to an edge from i to j."
        '''
        A[:, i] = get_adj_row_from_coeff_vec(i, coefficient_matrix[:, i], 
                                            parameters, 
                                            threshold_connect, add_weight) 
       
    return A  

def generate_net_dyn_model(y_0, time_length, net_dict):
    '''
    Generate trajectory using the model caracterized by coefficient_matrix

    Parameters
    ----------
    y_0 : numpy array
        Initial condition on phase space.
    coefficient_matrix : numpy array - size: (number_of_basis_functions, number_of_vertices)
        Coefficient matrix relative to params['symbolic_PHI'].
        In case this correspondence is not matched, the trajectory is false.
    time_length : float
        Length of time to generate trajectory.
    params : dict
        

    Returns
    -------
    Z: numpy array - size: (time_length, number_of_vertices)
        In case the trajectory does not satisfies the following:
        Z should consists of a trajectory of dynamical system. So, 
        Z should lie on the phase space and not go to infinite.
        returns Z (which is shorter) that satisfies.

    '''
    params = net_dict['params']
    
    N = params['number_of_vertices']
    x_t = [spy.symbols('x_{}'.format(j)) for j in range(0, N)]
    
    Z = np.zeros((time_length+1, N))             
    
    Z[0, :] = y_0
    
    for j in range(1, time_length + 1):
        for i in range(N):
            sym_expr = spy.lambdify([x_t], net_dict['sym_node_dyn'][i], 'numpy')
            Z[j, i] = sym_expr(Z[j - 1, :].T)
            
            mask_bounds = (np.any(np.isnan(Z)))#(Z < params['lower_bound']) | (Z > params['upper_bound'])\| 
            
            if np.any(mask_bounds):
                print('Warning: Trajectory reached infinity!')
                break
        
    return Z

def gen_isolated_map_model(net_dict):
    '''
    Generate reconstructed return map for each node

    Parameters
    ----------
    net_dict : dict

    Returns
    -------
    Z: numpy array - size: (time_length, number_of_vertices)
        Reconstructed return map

    '''
    params = net_dict['params']
    
    N = params['number_of_vertices']
    x_t = [spy.symbols('x_{}'.format(j)) for j in range(0, N)]
    
    Z = dict()
    
    for i in range(N):
        sym_expr = spy.lambdify([x_t], net_dict['sym_node_dyn'][i], 'numpy')
        eval_vec = np.zeros(N)
        Y_t = net_dict['Y_t']
        interv = np.arange(np.min(Y_t[:, i]), np.max(Y_t[:, i]), 0.001)
        Z[i] = np.zeros(interv.shape[0])
        for j in range(interv.shape[0]):
            eval_vec[i] = interv[j]
            Z[i][j] = sym_expr(eval_vec.T)
            
            mask_bounds = (np.any(np.isnan(Z[i])))#(Z < params['lower_bound']) | (Z > params['upper_bound'])\| 
            
            if np.any(mask_bounds):
                print('Warning: Trajectory reached infinity!')
                break
        
    return Z