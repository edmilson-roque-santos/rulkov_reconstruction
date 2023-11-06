"""
Script for reconstruction of Rulkov map via Pade representation

Created on Thu Jun 15 10:44:10 2023

@author: Edmilson Roque dos Santos
"""

import cvxpy as cp
import h5dict
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec    
import networkx as nx 
import numpy as np
import os
import scipy.special


from EBP import net_dyn, tools
from EBP.base_polynomial import pre_settings as pre_set 
from EBP.base_polynomial import poly_library as polb
from EBP.modules.rulkov import rulkov

import lab_rulkov as lr
import net_reconstr 

colors = ['darkgrey', 'orange', 'darkviolet', 'darkslategrey', 'silver']
folder_name = 'results'
ortho_folder_name = 'ortho_func_folder'


def out_dir_ortho(net_name, exp_name, params):
    '''
    Create the folder name for save orthonormal functions 
    locally inside results folder.

    Parameters
    ----------
    net_name : str
        Network structure filename.
    exp_name : str
        Filename.
    params : dict
        

    Returns
    -------
    out_results_direc : str
        Out results directory.

    '''
        
    out_results_direc = os.path.join(folder_name, ortho_folder_name)
    out_results_direc = os.path.join(out_results_direc, net_name)
    out_results_direc = os.path.join(out_results_direc, exp_name)
    out_results_direc = os.path.join(out_results_direc, '')
    if os.path.isdir(out_results_direc ) == False:
        
        try:
            os.makedirs(out_results_direc)
        except:
            'Folder has already been created'
    #For coupling analysis it is necessary to save each orthonormal function 
    #with respect to this coupling.
    filename = 'onf_deg_{}_lgth_ts_{}_coupling_{}_crossed_{}_seed_{}'.format(params['max_deg_monomials'],
                                                              params['length_of_time_series'], 
                                                              params['coupling'],
                                                              params['expansion_crossed_terms'],
                                                              params['random_seed'])
    out_results_direc = os.path.join(out_results_direc, filename)
    
    return out_results_direc



def compare_script(script_dict):
    '''
    Script for basis choice comparison. 

    Parameters
    ----------
    script_dict : dict
    Dictionary with specifier of the comparison script
    Keys:
        opt_list : list of boolean
            Each entry determines which basis is selected. 
            Order: #canonical, normalize_cols, orthonormal
        lgth_time_series : float
            Length of time series.
        exp_name : str
            Filename.
        net_name: str
            Network structure filename.
        id_trial: numpy array 
            Set of nodes to be reconstructed
            
    Returns
    -------
    dictionary result from net reconstruction algorithm.

    '''
    ############# Construct the parameters dictionary ##############
    parameters = dict()
    
    parameters['exp_name'] = script_dict['exp_name']
    parameters['Nseeds'] = 1
    parameters['random_seed'] = script_dict.get('random_seed', 1)
    parameters['network_name'] = script_dict['net_name']
    parameters['max_deg_monomials'] = 3
    parameters['expansion_crossed_terms'] = True#
    
    parameters['use_kernel'] = True
    parameters['noisy_measurement'] = False
    parameters['use_canonical'] = script_dict['opt_list'][0]
    parameters['normalize_cols'] = script_dict['opt_list'][1]
    parameters['use_orthonormal'] = script_dict['opt_list'][2]
    parameters['single_density'] = False
    
    G = script_dict['G']
    
    N = len(nx.nodes(G))
    A = nx.to_numpy_array(G, nodelist = list(range(N)))
    A = np.asarray(A)
    degree = np.sum(A, axis=0)
    parameters['adj_matrix'] = A
    parameters['coupling'] = 0.01
    #==========================================================#
    net_dynamics_dict = dict()
    net_dynamics_dict['adj_matrix'] = parameters['adj_matrix'] - degree*np.identity(A.shape[0])
    
    transient_time = 5000
    test_time = 2001
    parameters['length_of_time_series'] = script_dict['lgth_time_series']
    
    net_dynamics_dict['f'] = rulkov.rulkov_map
    net_dynamics_dict['h'] = rulkov.diff_coupling_x
    net_dynamics_dict['max_degree'] = np.max(degree)
    net_dynamics_dict['coupling'] = parameters['coupling']
    net_dynamics_dict['random_seed'] = parameters['random_seed']
    net_dynamics_dict['transient_time'] = transient_time
    x_time_series = net_dyn.gen_net_dynamics(script_dict['lgth_time_series']+test_time, net_dynamics_dict)  
    
    X_time_series = x_time_series[:-test_time, :]
    
    #==========================================================#    
    net_dict = dict()
    
    
    mask_bounds = (X_time_series < -1e5) | (X_time_series > 1e5) | (np.any(np.isnan(X_time_series)))
    if np.any(mask_bounds):
        raise ValueError("Network dynamics does not live in a compact set ")
        
    if not np.any(mask_bounds):

        X_t = X_time_series[:script_dict['lgth_time_series'],:]
        
        parameters['lower_bound'] = np.min(X_t)
        parameters['upper_bound'] = np.max(X_t)
        
        parameters['number_of_vertices'] = X_t.shape[1]
        
        parameters['X_time_series_data'] = X_t
        
        params = parameters.copy()
        
        if params['use_orthonormal']:
            out_dir_ortho_folder = out_dir_ortho(script_dict['net_name'], 
                                                 script_dict['exp_name'], params)
            
            output_orthnormfunc_filename = out_dir_ortho_folder
        
            params['cluster_list'] = script_dict['cluster_list']
            
            if not os.path.isfile(output_orthnormfunc_filename):
                params['orthnorm_func_filename'] = output_orthnormfunc_filename
                params['orthnormfunc'] = pre_set.create_orthnormfunc_clusters_kde(script_dict['cluster_list'], params)    
    
            if os.path.isfile(output_orthnormfunc_filename):
                params['orthnorm_func_filename'] = output_orthnormfunc_filename
                      
            params['build_from_reduced_basis'] = False
        
        params['threshold_connect'] = 1e-8
        
        if script_dict['id_trial'] is not None:
            params['id_trial'] = script_dict['id_trial']
        
        params['Y_t'] = x_time_series[-test_time:, :]
        
        net_dict = script_dict['exp'](X_t, params)
    
    return net_dict
    
def save_dict(dictionary, out_dict):
    '''
    Save dictionary in the output dictionary and avoids some keys that are not
    allowed in hdf5.

    Parameters
    ----------
    dictionary : dict
    out_dict : dict

    Returns
    -------
    None.

    '''
    keys = dictionary.keys()
    for key in keys:
        try:
            out_dict[key] = dictionary[key]
        except:
            print("Error: not possible to save", key)

def out_dir(net_name, exp_name): 
    '''
    Create the folder name for save comparison  
    locally inside results folder.

    Parameters
    ----------
    net_name : str
        Network structure filename.
    exp_name : str
        Filename.
    
    Returns
    -------
    out_results_direc : str
        Out results directory.

    '''       
    out_results_direc = os.path.join(folder_name, net_name)
    out_results_direc = os.path.join(out_results_direc, exp_name)
    out_results_direc = os.path.join(out_results_direc, '')
    
    if os.path.isdir(out_results_direc) == False:
        try:
            os.makedirs(out_results_direc)
        except:
            'Folder has already been created'
    return out_results_direc

def compare_setup(exp_name, net_name, G, lgth_endpoints, random_seed = 1, 
                  save_full_info = False):
    '''
    
    Parameters
    ----------
    exp_name : str
        filename.
    net_name : str
        Network structure filename.
    lgth_endpoints : list
        Start, end and space for length time vector.
    random_seed : int
        Seed for the random pseudo-generator.
    save_full_info : dict, optional
        To save the library matrix. The default is False.

    Returns
    -------
    exp_dictionary : TYPE
        DESCRIPTION.

    '''
    exp_params = dict()
    #canonical
    exp_params[0] = [True, False, False]
    #normalize_cols
    #exp_params[0] = [True, True, False]
    #orthonormal
    #exp_params[1] = [False, False, True]
    
    length_time_series_vector = np.arange(lgth_endpoints[0], lgth_endpoints[1],
                                          lgth_endpoints[2], dtype = int)
    
    #Filename for output results
    out_results_direc = out_dir(net_name, exp_name)
    filename = "lgth_endpoints_{}_{}_{}_seed_{}".format(lgth_endpoints[0], lgth_endpoints[1],
                                                lgth_endpoints[2], random_seed) 
    
    if os.path.isfile(out_results_direc+filename+".hdf5"):
        out_results_hdf5 = h5dict.File(out_results_direc+filename+".hdf5", 'r')
        exp_dictionary = out_results_hdf5.to_dict()  
        out_results_hdf5.close()      
        return exp_dictionary
    
    else:
        out_results_hdf5 = h5dict.File(out_results_direc+filename+".hdf5", 'a')    
        out_results_hdf5['lgth_endpoints'] = lgth_endpoints
        out_results_hdf5['exp_params'] = dict() 
        out_results_hdf5['exp_params'] = exp_params
        
        for key in exp_params.keys():    
            out_results_hdf5[key] = dict()
            for lgth_time_series in length_time_series_vector:
                print('exp:', key, 'n = ', lgth_time_series)
                
                script_dict = dict()
                script_dict['opt_list'] = exp_params[key]
                script_dict['lgth_time_series'] = lgth_time_series
                script_dict['exp_name'] = exp_name
                script_dict['net_name'] = net_name
                script_dict['G'] = G
                
                if exp_params[key][2]:
                    N = len(G)
                    script_dict['cluster_list'] = [np.arange(0, 2*N, 2), np.arange(1, 2*N, 2)]
                
                script_dict['id_trial'] = None
                script_dict['random_seed'] = random_seed
                script_dict['exp'] = net_reconstr.ADM_reconstr#
                
                net_dict = compare_script(script_dict)
                out_results_hdf5[key][lgth_time_series] = dict()
                N = net_dict['params']['number_of_vertices']
                for id_node in range(N):
                    out_results_hdf5[key][lgth_time_series]['x_eps_matrix'] = net_dict['x_eps_matrix']
                    out_results_hdf5[key][lgth_time_series]['error'] = net_dict['error']
                    
                if save_full_info:
                    out_results_hdf5[key][lgth_time_series]['PHI.T PHI'] = net_dict['PHI.T PHI']
                    out_results_hdf5[key][lgth_time_series]['params'] = dict()
                    save_dict(net_dict['params'], out_results_hdf5[key][lgth_time_series]['params'])            
                
                
        exp_dictionary = out_results_hdf5.to_dict()        
        out_results_hdf5.close()
        return exp_dictionary

def ker_compare_setup(exp_name, net_name, G, lgth_endpoints, random_seed = 1, 
                  save_full_info = False):
    '''
    Compare the dimension of the kernel for different library matrices.
    
    Parameters
    ----------
    exp_name : str
        filename.
    net_name : str
        Network structure filename.
    lgth_endpoints : list
        Start, end and space for length time vector.
    random_seed : int
        Seed for the random pseudo-generator.
    save_full_info : dict, optional
        To save the library matrix. The default is False.

    Returns
    -------
    exp_dictionary : TYPE
        DESCRIPTION.

    '''
    exp_params = dict()
    #canonical
    exp_params[0] = [True, False, False]
    #normalize_cols
    #exp_params[0] = [True, True, False]
    #orthonormal
    #exp_params[1] = [False, False, True]
    
    length_time_series_vector = np.arange(lgth_endpoints[0], lgth_endpoints[1],
                                          lgth_endpoints[2], dtype = int)
    
    #Filename for output results
    out_results_direc = out_dir(net_name, exp_name)
    filename = "lgth_endpoints_{}_{}_{}_seed_{}".format(lgth_endpoints[0], lgth_endpoints[1],
                                                lgth_endpoints[2], random_seed) 
    
    if os.path.isfile(out_results_direc+filename+".hdf5"):
        out_results_hdf5 = h5dict.File(out_results_direc+filename+".hdf5", 'r')
        exp_dictionary = out_results_hdf5.to_dict()  
        out_results_hdf5.close()      
        return exp_dictionary
    
    else:
        out_results_hdf5 = h5dict.File(out_results_direc+filename+".hdf5", 'a')    
        out_results_hdf5['lgth_endpoints'] = lgth_endpoints
        out_results_hdf5['exp_params'] = dict() 
        out_results_hdf5['exp_params'] = exp_params
        
        for key in exp_params.keys():    
            out_results_hdf5[key] = dict()
            for lgth_time_series in length_time_series_vector:
                print('exp:', key, 'n = ', lgth_time_series)
                
                script_dict = dict()
                script_dict['opt_list'] = exp_params[key]
                script_dict['lgth_time_series'] = lgth_time_series
                script_dict['exp_name'] = exp_name
                script_dict['net_name'] = net_name
                script_dict['G'] = G
                
                if exp_params[key][2]:
                    N = len(G)
                    script_dict['cluster_list'] = [np.arange(0, 2*N, 2), np.arange(1, 2*N, 2)]
                
                script_dict['id_trial'] = None
                script_dict['random_seed'] = random_seed
                script_dict['exp'] = net_reconstr.kernel_calculation
                
                net_dict = compare_script(script_dict)
                out_results_hdf5[key][lgth_time_series] = dict()
                N = net_dict['params']['number_of_vertices']
                for id_node in range(N):
                    out_results_hdf5[key][lgth_time_series][id_node] = net_dict['info_x_eps'][id_node]
                    print('ker', net_dict['info_x_eps'][id_node])
                    
                if save_full_info:
                    out_results_hdf5[key][lgth_time_series]['PHI.T PHI'] = net_dict['PHI.T PHI']
                    out_results_hdf5[key][lgth_time_series]['params'] = dict()
                    save_dict(net_dict['params'], out_results_hdf5[key][lgth_time_series]['params'])            
                
                
        exp_dictionary = out_results_hdf5.to_dict()        
        out_results_hdf5.close()
        return exp_dictionary

def access_n_c(net_dict):
    N = int(net_dict['params']['number_of_vertices']/2)
    
    defect_PHI = np.zeros(N)
    
    for i, id_node in enumerate(net_dict['params']['id_trial']):
        defect_PHI[i] = net_dict['info_x_eps'][id_node]
    
    mask = defect_PHI == 1
    
    return np.all(mask)
    
def determine_critical_n(exp_param, size, exp_name, net_info, id_trial = None, 
                         random_seed = 1, r = 3):
    '''
    Determine the minimum length of time series for a successfull reconstruction.

    Parameters
    ----------
    exp_param : list
        Set the optlist for compare_script.
    size : float
        Network size.
    exp_name : str
        Filename.
    net_class : str
        Common network structure filename.
    id_trial : numpy array
        Set of nodes to be reconstructed.
    random_seed : int
        Seed for the random pseudo-generator.

    Returns
    -------
    n_critical : float
        minimum length of time series.

    '''
    net_name = net_info['net_class']+"_{}".format(size)
    
    if not os.path.isfile('network_structure/'+net_name):
        try:
            true_graph = net_info['gen'](size, 'network_structure/'+net_name)
        except:
            print("There is already a net!")
    
    m_N_ = scipy.special.comb(2*size, 2, exact = True)*scipy.special.comb(r, 2, exact = True) + 2*size*r + 1
    m_N = 2*m_N_
    
    #Based on a fit from points between 3 - 14, we identify a exponential growth. 
    #We employ this fitting to calculate larger values of N
    
    m_N_0 = int(np.round(np.exp(4.4)*np.exp(0.3*size)))
    
    size_step = int(np.round(m_N_0/10))
        
    
    lgth_time_series_vector = np.arange(m_N_0, m_N**2, size_step, dtype = int)
    id_, max_iterations = 0, 100
    
    find_critical = True
    while (find_critical) and (id_ < max_iterations):
        lgth_time_series = lgth_time_series_vector[id_]
        print('lgth:', lgth_time_series)
        
        script_dict = dict()
        script_dict['opt_list'] = exp_param
        script_dict['lgth_time_series'] = lgth_time_series
        script_dict['exp_name'] = exp_name
        script_dict['net_name'] = net_name
        script_dict['random_seed'] = random_seed
        script_dict['G'] = true_graph
        script_dict['id_trial'] = np.arange(0, 2*(len(true_graph)), 2)
        script_dict['exp'] = net_reconstr.kernel_calculation
        
        net_dict = compare_script(script_dict)
    
        if access_n_c(net_dict):
            find_critical = False
            print('Defect THETA = 1!')
        
        id_ = id_ + 1
    
    n_critical = lgth_time_series
    return n_critical

def compare_setup_critical_n(exp_name, net_info, size_endpoints, id_trial,
                             random_seed = 1, save_full_info = False):
    '''
    Comparison script to growing the net size and evaluate the critical length of 
    time series for a successful reconstruction.
    
    Parameters
    ----------
    exp_name : str
        Filename.
    net_name : str
        Network structure filename.
    size_endpoints : list
        Start, end and space for size vector.
    random_seed : int
        Seed for the random pseudo-generator.
    save_full_info : dict, optional
        To save the library matrix. The default is False.

    Returns
    -------
    exp_dictionary : dict
        Output results dictionary.

    '''
    exp_params = dict()
    #canonical
    exp_params[0] = [True, False, False]
    #normalize_cols
    #exp_params[0] = [True, True, False]
    #orthonormal
    #exp_params[1] = [False, False, True]
    
    size_vector = np.arange(size_endpoints[0], size_endpoints[1],
                                          size_endpoints[2], dtype = int)
    
    #Filename for output results
    out_results_direc = out_dir(net_info['net_class'], exp_name)
        
    filename = "size_endpoints_{}_{}_{}_seed_{}".format(size_endpoints[0], 
                                                        size_endpoints[1],
                                                        size_endpoints[2], 
                                                        random_seed) 
    
    if os.path.isfile(out_results_direc+filename+".hdf5"):
        out_results_hdf5 = h5dict.File(out_results_direc+filename+".hdf5", 'r')
        exp_dictionary = out_results_hdf5.to_dict()  
        out_results_hdf5.close()      
        return exp_dictionary
    
    else:
        out_results_hdf5 = h5dict.File(out_results_direc+filename+".hdf5", 'a')    
        out_results_hdf5['size_endpoints'] = size_endpoints
        out_results_hdf5['exp_params'] = dict() 
        out_results_hdf5['exp_params'] = exp_params
        
        for key in exp_params.keys():    
            out_results_hdf5[key] = dict()
            for size in size_vector:
                print('exp:', key, 'N = ', size)
                
                n_critical = determine_critical_n(exp_params[key], size, exp_name, 
                                                  net_info, id_trial, random_seed)
                
                out_results_hdf5[key][size] = dict()
                out_results_hdf5[key][size]['n_critical'] = n_critical
                
        exp_dictionary = out_results_hdf5.to_dict()        
        out_results_hdf5.close()
        return exp_dictionary

def net_seed(G, rs, method):
    exp_name = 'n_vary_trs_5000'
    net_name = 'star_graph_N=5'
    lgth_endpoints = [350, 2001, 50]
    random_seed = rs
    save_full_info = False
    exp_ = method(exp_name, net_name, G, lgth_endpoints, random_seed, 
                      save_full_info)
    return exp_

def ker_net_seed(G, rs, method):
    exp_name = 'ker_n_vary_trs_5000'
    net_name = 'star_graph_N=9'
    lgth_endpoints = [100, 2101, 100]
    random_seed = rs
    save_full_info = False
    exp_ = method(exp_name, net_name, G, lgth_endpoints, random_seed, 
                      save_full_info)
    return exp_

def star_n_c_script(rs):
    '''
    Script to generate an experiment of determining the critical length 
    of time series as the size of the network is increased.

    Parameters
    ----------
    rs : int
        Int for the seed of the random pseudo-generator.

    Returns
    -------
    None.

    '''
    exp_name = 'star_graph_nc'
    
    net_info = dict()
    net_info['net_class'] = 'star_graph'
    net_info['gen'] = tools.star_graph
    size_endpoints = [15, 21, 2]
    id_trial = None
    compare_setup_critical_n(exp_name, net_info, size_endpoints, id_trial, 
                             random_seed = rs, save_full_info = False)

def MC_script(main, net_name = 'star_graphs_n_4_hub_coupled'):
    
    G = nx.read_edgelist("network_structure/{}.txt".format(net_name),
                        nodetype = int, create_using = nx.Graph)
    ##### Randomness
    Nseeds = 6
    MonteCarlo_seeds = np.arange(5, Nseeds + 5)     # Seed for random number generator
    
    exp_ = dict()
    for rs in MonteCarlo_seeds:
    
        #exp_[rs] = ker_net_seed(G, rs, method = ker_compare_setup)
        exp_[rs] = main(G, rs, method = compare_setup)
        
    return exp_

def MC_script_nc():
    
    ##### Randomness
    Nseeds = 1
    MonteCarlo_seeds = np.arange(1, 1 + Nseeds)     # Seed for random number generator
    
    exp_ = dict()
    for rs in MonteCarlo_seeds:
    
        exp_[rs] = star_n_c_script(rs)
        
    return exp_

def exp_setup(lgths_endpoints, exps_name, 
              net_name = 'star_graphs_n_4_hub_coupled',
              Nseeds = 10):
    '''
    Setting the experiment of increasing the length of time series.

    Parameters
    ----------
    lgths_endpoints : list
        List with the endpoints of the array to create the lgth vector
    exps_name : list
        List with the filename of the experiment.
    net_name : str, optional
        Name of the network to be evaluated. The default is 'ring_graph_N=16'.
    Nseeds : int, optional
        Total number of seeds. The default is 10.

    Returns
    -------
    exps_dictionary : dict
        Experiment dictionary with information gathered from the hdf5 file.
    title : TYPE
        DESCRIPTION.

    '''
    
    exps_dictionary = dict()
    
    for id_exp in range(len(exps_name)):
        exps_dictionary[id_exp] = dict()
        lgth_endpoints = lgths_endpoints[id_exp]
        exp_name = exps_name[id_exp]
        out_results_direc = os.path.join(folder_name, net_name)
        out_results_direc = os.path.join(out_results_direc, exp_name)
        out_results_direc = os.path.join(out_results_direc, '')
        
        if os.path.isdir(out_results_direc) == False:
            print("Failed to find the desired result folder !")
        
        for seed in range(1, Nseeds + 1):
            exps_dictionary[id_exp][seed] = dict()
         
            filename = "lgth_endpoints_{}_{}_{}_seed_{}".format(lgth_endpoints[0], lgth_endpoints[1],
                                                        lgth_endpoints[2], seed) 
            
            if os.path.isfile(out_results_direc+filename+".hdf5"):
                out_results_hdf5 = h5dict.File(out_results_direc+filename+".hdf5", 'r')
                exp_dictionary = out_results_hdf5.to_dict()  
                out_results_hdf5.close()
            
            exps_dictionary[id_exp][seed] = exp_dictionary
    
    return exps_dictionary

def exp_setting_n_c(exps_name, sizes_endpoints, net_class = 'ring_graph', 
                    Nseeds = 10):
    '''
    Setting the experiment of determing the critical length of time series for
    a successfull reconstruction.

    Parameters
    ----------
    exps_name : list
        Name of experiments to be read from the file.
    sizes_endpoints : list
        Endpoints of the arrays to determine the numpy arrays.
    net_class : str, optional
        Class of graph to be increased in order to perform the experiment. The default is 'ring_graph'.
    Nseeds : int, optional
        Total number of seeds. The default is 10.

    Returns
    -------
    exps_dictionary : dict
        Experiment dictionary with information gathered from the hdf5 file.

    '''
    exps_dictionary = dict()
    
    for id_exp in range(len(exps_name)):
        exps_dictionary[id_exp] = dict()
        size_endpoints = sizes_endpoints[id_exp]

        exp_name = exps_name[id_exp]
        out_results_direc = os.path.join(folder_name, net_class)
        out_results_direc = os.path.join(out_results_direc, exp_name)
        out_results_direc = os.path.join(out_results_direc, '')
        
        if os.path.isdir(out_results_direc ) == False:
            print("Failed to find the desired result folder !")
        
        for seed in range(1, Nseeds + 1):    
            exps_dictionary[id_exp][seed] = dict()
            
            filename = "size_endpoints_{}_{}_{}_seed_{}".format(size_endpoints[0], size_endpoints[1],
                                                        size_endpoints[2], seed) 
            
            if os.path.isfile(out_results_direc+filename+".hdf5"):
                try:
                    out_results_hdf5 = h5dict.File(out_results_direc+filename+".hdf5", 'r')
                    exp_dictionary = out_results_hdf5.to_dict()  
                    out_results_hdf5.close()
                    exps_dictionary[id_exp][seed] = exp_dictionary
                except:
                    print('Failed to open the desired file!')    
                    #exp_dictionary = dict()
                    del exps_dictionary[id_exp][seed]
            else:
                print('Failed to find the desired file!')
                
                print(out_results_direc+filename+".hdf5")

    return exps_dictionary

def stars_coupled_plot_script(Nseeds = 10):
    
    exps_dictionary =  exp_setup(lgths_endpoints = [[100, 2101, 100]],
                                       exps_name = ['ker_n_vary_trs_5000'],
                                       net_name = 'star_graphs_n_4_hub_coupled',
                                       Nseeds = Nseeds)
    title = [r'Dependence on $n$']
    lr.plot_lgth_dependence('star_graphs_n_4_hub_coupled', 
                            exps_dictionary, 
                            title, 
                            filename = 'Figures/ker_n_vary_star_coupled')    


def star_plot_script(Nseeds = 10):
    exps_dictionary = exp_setup(lgths_endpoints = [[350, 2001, 50], [350, 2001, 50]],
                                       exps_name = ['l2_n_vary_trs_5000', 'n_vary_trs_5000'],
                                        net_name = 'star_graph_N=5',
                                        Nseeds = Nseeds)
    lgth_vector = np.arange(350, 2001, 50, dtype = int)
    
    for seed in range(1, Nseeds + 1):
        for lgth in lgth_vector:
            exps_dictionary[0][seed][0][lgth]['error'] = np.sqrt(2)*exps_dictionary[0][seed][0][lgth]['error']
    title = [r'$\ell_2$', r'Implicit SINDy']
    lr.plot_lgth_dependence('star_graph_N=5', 
                            exps_dictionary, 
                            title, 
                            method = lr.error_compare,
                            plot_ycoord= False,
                            plot_def = False,
                            filename = 'error_compare')
    
def n_c_plot_script(Nseeds = 10):
    '''
    Script to plot an experiment of determining the critical length 
    of time series as the size of the network is increased.

    Parameters
    ----------
    Nseeds : int, optional
        Total number of seeds of the random pseudo-generator. The default is 10.

    Returns
    -------
    exps_dictionary : dict
        Experiment dictionary with information gathered from the hdf5 file.

    '''
    title = []
    exps_name = ['star_graph_nc']
    size_endpoints = [[1, 11, 1]]#[[3, 51, 5]]
    exps_dictionary = exp_setting_n_c(exps_name, size_endpoints, 
                                             net_class = 'star_graph',
                                             Nseeds = Nseeds)
    #
    lr.plot_n_c_size(exps_dictionary, title, filename = 'nc_vs_N_star')
    return exps_dictionary 

def test():    
    script_dict = dict()
    script_dict['opt_list'] = [True, False, False]
    script_dict['lgth_time_series'] = 200
    script_dict['exp_name'] = 'test_reconstr'
    script_dict['net_name'] = 'two_nodes'
    script_dict['G'] = nx.read_edgelist("network_structure/{}.txt".format(script_dict['net_name']),
                                        nodetype = int, create_using = nx.Graph)
    script_dict['cluster_list'] = [np.arange(0, 20, 2), np.arange(1, 20, 2)]#[np.array([0, 2]), np.array([1, 3])]#
    script_dict['id_trial'] = None#np.arange(0, 20, 2)
    script_dict['random_seed'] = 1
    
    #net_dict = net_reconstr.ADM_reconstr(X_t, params)
    #solver_optimization = cp.ECOS#CVXOPT
    #net_dict = net_reconstr.reconstr(X_t, params, solver_optimization)
    #net_reconstr.kernel_calculation(X_t, params)
    script_dict['exp'] = net_reconstr.kernel_calculation
    net_dict = compare_script(script_dict)
    return net_dict        
    '''
    error_matrix = net_reconstr.uniform_error(net_dict, num_samples = 50, time_eval = 1)

    #lr.fig_return_map(net_dict['Y_t'], filename=None)
    #lr.fig_time_series(net_dict['Y_t'])

    folder = 'Figures/'
    filename = None#folder+'Fig_1_v0'
    lr.Fig_1(net_dict, script_dict['net_name'], id_node = 0, filename = filename )
    
    return net_dict, error_matrix
    '''