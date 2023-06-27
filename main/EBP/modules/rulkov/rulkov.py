"""
Module for Rulkov map simulation

Created on Thu Jun 15 12:51:47 2023

@author: Edmilson Roque dos Santos
"""

import numpy as np
import sympy as spy
from scipy import stats

def rulkov_map(x_state, alpha = 4.4, sigma = 0.001, beta = 0.001):
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
    x_ = x_state.copy()
    
    cplg_x = np.zeros(x_state.shape)
    
    x = x_[0::2]
    
    cplg_x[0::2] = A @ x
    
    return cplg_x

def cluster_moment_est(cluster_list, params):
    '''
    The assumption to estimate the density function for each cluster is that
    all nodes in a given cluster behaves similarly, hence we use all trajectories
    of those nodes to estimate the same density function.
    
    Parameters
    ----------
    cluster_list : list
        List of graph partition containing node list for each cluster.
    params : dict

    Returns
    -------
    parameters : dict
        Updated parameters dictionary to be used throughout the simulation.
        The updated arguments are: the calculation of density functions per cluster
        given by the cluster_list variable.

    '''
    
    parameters = dict()
    
    #If kernel density estimation is used, a data point must be given before hand
    if(params['use_kernel'] and params['use_integral_1d']):
        
        parameters['type_density'] = params.get('type_density', '1d_Kernel')
        parameters['density'] = params.get('density', None)
        if parameters['density'] == None:
            #Gather data points to be used on the kernel density estimator
            parameters['X_time_series_data'] = params.get('X_time_series_data', np.array([]))
            if len(parameters['X_time_series_data'] > 0):
                
                x_t = [spy.symbols('x_{}'.format(j)) for j in range(0, params['number_of_vertices'])]
                num_clusters = len(cluster_list)
                id_vec = np.arange(0, params['number_of_vertices'], dtype = int)
                for id_cluster in range(num_clusters):
                    id_vec_cluster =  np.asarray(cluster_list[id_cluster], dtype = int)
                    mask_cluster = np.isin(id_vec, id_vec_cluster)
                    
                    X_t_cluster = params['X_time_series_data'][:, mask_cluster]
                    data_cluster = X_t_cluster.T.flatten()
                    kernel_cluster = stats.gaussian_kde(data_cluster, bw_method = 5e-2)
                    
                    for id_node in id_vec_cluster:
                        parameters[x_t[id_node]] = dict()
                        parameters[x_t[id_node]]['type_density'] = params.get('type_density', '1d_Kernel')
                        parameters[x_t[id_node]]['density'] = params.get('density', None)
        
                        #Lower and upper bound of the phase space of the isolated dynamics
                        parameters[x_t[id_node]]['lower_bound'] = params.get('lower_bound', np.min(data_cluster))
                        parameters[x_t[id_node]]['upper_bound'] = params.get('upper_bound', np.max(data_cluster))
                        parameters[x_t[id_node]]['density'] =  kernel_cluster#/kernel.integrate_box_1d(parameters[x_t[id_node]]['lower_bound'], parameters[x_t[id_node]]['upper_bound'])
                        parameters[x_t[id_node]]['density_normalization'] = kernel_cluster.integrate_box_1d(parameters[x_t[id_node]]['lower_bound'], parameters[x_t[id_node]]['upper_bound'])

    return parameters

def params_cluster(cluster_list, params):
    '''
    Update symbolic representation of the moments calculation accordingly 
    to cluster_list.

    Parameters
    ----------
    cluster_list : list
        List of graph partition containing node list for each cluster.
    params : dict

    Returns
    -------
    parameters : Updated
        Updated parameters dictionary to be used throughout the simulation.
        
    '''
    parameters = params.copy()
    params_cluster = cluster_moment_est(cluster_list, params)
    x_t = [spy.symbols('x_{}'.format(j)) for j in range(0, params['number_of_vertices'])]
    for id_node in range(params['number_of_vertices']):
        parameters[x_t[id_node]] = params_cluster[x_t[id_node]]
    
    return parameters