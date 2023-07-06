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
    
    transient_time = 2000
    test_time = 2000
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
        
        net_dict = script_dict['exp'](X_t, params)
        
    net_dict['Y_t'] = x_time_series[-test_time:, :]
    
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

def compare_setup(exp_name, net_name, lgth_endpoints, random_seed = 1, 
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
    #exp_params[0] = [True, False, False]
    #normalize_cols
    exp_params[0] = [True, True, False]
    #orthonormal
    exp_params[1] = [False, False, True]
    
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
                script_dict['id_trial'] = None
                script_dict['random_seed'] = random_seed
                
                net_dict = compare_script(script_dict)
                out_results_hdf5[key][lgth_time_series] = dict()
                out_results_hdf5[key][lgth_time_series]['A'] = net_dict['A']
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
                script_dict['id_trial'] = None
                script_dict['random_seed'] = random_seed
                script_dict['exp'] = net_reconstr.kernel_calculation
                
                net_dict = compare_script(script_dict)
                out_results_hdf5[key][lgth_time_series] = dict()
                N = net_dict['params']['number_of_vertices']
                for id_node in range(N):
                    out_results_hdf5[key][lgth_time_series][id_node] = net_dict['info_x_eps'][id_node]
                if save_full_info:
                    out_results_hdf5[key][lgth_time_series]['PHI.T PHI'] = net_dict['PHI.T PHI']
                    out_results_hdf5[key][lgth_time_series]['params'] = dict()
                    save_dict(net_dict['params'], out_results_hdf5[key][lgth_time_series]['params'])            
                
                
        exp_dictionary = out_results_hdf5.to_dict()        
        out_results_hdf5.close()
        return exp_dictionary

def net_seed(G, rs):
    exp_name = 'ker_n_vary'
    net_name = 'star_graph_N=9'
    lgth_endpoints = [100, 2101, 100]
    random_seed = rs
    save_full_info = False
    exp_ = ker_compare_setup(exp_name, net_name, G, lgth_endpoints, random_seed, 
                      save_full_info)
    return exp_

def MC_script(net_name = 'star_graphs_n_4_hub_coupled'):
    
    G = nx.read_edgelist("network_structure/{}.txt".format(net_name),
                        nodetype = int, create_using = nx.Graph)
    ##### Randomness
    Nseeds = 10
    MonteCarlo_seeds = np.arange(1, Nseeds + 1)     # Seed for random number generator
    
    exp_ = dict()
    for rs in MonteCarlo_seeds:
    
        exp_[rs] = net_seed(G, rs)
    
    return exp_

def star_coupled_setup(net_name = 'star_graphs_n_4_hub_coupled', Nseeds = 10):
    '''
    Setting the experiment of increasing the length of time series.

    Parameters
    ----------
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
    lgths_endpoints = [[2000+100, 2000+2000, 100]]
    exps_name = ['reconstr_n_varying']
    title = [r'Dependence on $n$']
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
    
    return exps_dictionary, title

def star_plot_script(Nseeds = 10):
    exps_dictionary, title = star_coupled_setup(net_name = 'star_graphs_n_4_hub_coupled', Nseeds = Nseeds)
    lr.plot_lgth_dependence('star_graphs_n_4_hub_coupled', exps_dictionary, title, filename = None)    
    
    
def test():    
    script_dict = dict()
    script_dict['opt_list'] = [False, False, True]
    script_dict['lgth_time_series'] = 2202
    script_dict['exp_name'] = 'test_orth'
    script_dict['net_name'] = 'two_nodes'
    script_dict['G'] = nx.read_edgelist("network_structure/{}.txt".format(script_dict['net_name']),
                                        nodetype = int, create_using = nx.Graph)
    script_dict['cluster_list'] = [np.array([0, 2]), np.array([1, 3])]#[np.arange(0, 20, 2), np.arange(1, 20, 2)]#
    script_dict['id_trial'] = None#np.arange(0, 20, 2)
    script_dict['random_seed'] = 3
    
    #net_dict = net_reconstr.ADM_reconstr(X_t, params)
    #solver_optimization = cp.ECOS#CVXOPT
    #net_dict = net_reconstr.reconstr(X_t, params, solver_optimization)
    #net_reconstr.kernel_calculation(X_t, params)
    script_dict['exp'] = net_reconstr.reconstr
    net_dict = compare_script(script_dict)
    return net_dict

'''
error_matrix = net_reconstr.uniform_error(net_dict, num_samples = 50, time_eval = 1)

#lr.fig_return_map(net_dict['Y_t'], filename=None)
#lr.fig_time_series(net_dict['Y_t'])

folder = 'Figures/'
filename = None#folder+'Fig_1_v0'
lr.Fig_1(net_dict, script_dict['net_name'], id_node = 0, filename = filename )
'''