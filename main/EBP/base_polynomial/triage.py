import numpy as np
import os 
import scipy.special
import sympy as spy
from scipy import stats

from . import poly_library as polb
from .. import tools


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
    
    parameters = params.copy()
    
    #If kernel density estimation is used, a data point must be given before hand
    if(params['use_kernel'] and params['use_integral_1d']):
        parameters['type_density'] = params.get('type_density', '1d_Kernel')
        parameters['density'] = params.get('density', None)
        if parameters['density'] == None:
            #Gather data points to be used on the kernel density estimator
            if len(params['X_time_series_data'] > 0):
                x_t = [spy.symbols('x_{}'.format(j)) for j in range(0, params['number_of_vertices'])]
                num_clusters = len(cluster_list)
                id_vec = np.arange(0, params['number_of_vertices'], dtype = int)
                for id_cluster in range(num_clusters):
                    id_vec_cluster =  np.asarray(cluster_list[id_cluster], dtype = int)
                    mask_cluster = np.isin(id_vec, id_vec_cluster)
                    
                    X_t_cluster = params['X_time_series_data'][:, mask_cluster]
                    data_cluster = X_t_cluster.T.flatten()
                    kernel_cluster = stats.gaussian_kde(data_cluster, bw_method = 1e-1)
                    
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
    #parameters = params.copy()
    params_cluster = cluster_moment_est(cluster_list, params)
    '''
    x_t = [spy.symbols('x_{}'.format(j)) for j in range(0, params['number_of_vertices'])]
    for id_node in range(params['number_of_vertices']):
        parameters[x_t[id_node]] = params_cluster[x_t[id_node]]
    '''
    return params_cluster

def triage_params(params):
    '''
        Update the parameters dictionary to be used along the simulation. For any argument
        not expliclitly given, this method fill the empty arguments.

    Parameters
    ----------
    params : dict        

    Returns
    -------
    parameters : dict
        Updated parameters dictionary to be used throughout the simulation.

    '''
    #Construct the parameters dictionary
    parameters = dict()
    
    parameters['exp_name'] = params.get('exp_name')     #Experiment name

    parameters['Nseeds'] = params.get('Nseeds', 1)      #Number of seeds 
    
    #Identifier of seed
    parameters['random_seed'] = params.get('random_seed', 1)        
    
    #Number of vertices in the graph
    parameters['number_of_vertices'] = params.get('number_of_vertices', False)
    #Index array for those nodes which are included in the libray matrix
    #for reconstruction
    parameters['id_trial'] = params.get('id_trial', np.arange(parameters['number_of_vertices'], dtype = int))
    parameters['cluster_list'] = params.get('cluster_list')
    
    #Node to be estimated in the reconstruction
    parameters['node_test'] = params.get('node_test')
    
    #Length of time series
    parameters['length_of_time_series'] = params.get('length_of_time_series', 100)
    
    #Time series
    parameters['X_time_series_data'] = params.get('X_time_series_data', np.array([]))
    
    #Adjacency matrix of the graph
    parameters['adj_matrix'] = params.get('adj_matrix')
    
    #Coupling strength of the network dynamics
    parameters['coupling'] = params.get('coupling', 1e-3)
    
    #To determine the threshold for the support set of coefficients
    if not np.any(parameters['adj_matrix'] == None):
        
        max_degree = np.max(np.sum(parameters['adj_matrix'], axis=0))
        if (max_degree < 1):
            max_degree = 1                            
    
    if np.any(parameters['adj_matrix'] == None):
         max_degree = 1
    parameters['threshold_connect'] =  params.get('threshold_connect',  parameters['coupling']/(max_degree)*10)
    
    
    #Max order of the exponent vector
    parameters['max_deg_monomials'] = params.get('max_deg_monomials', 2)
    
    #Determine if there is crossed terms in the basis expansion
    parameters['expansion_crossed_terms'] =  params.get('expansion_crossed_terms', False)
    
    #Calculation of the number of elemens in the library set 
    if not parameters['expansion_crossed_terms']:
        parameters['L'] = parameters['number_of_vertices']*parameters['max_deg_monomials'] + 1
    else:
        L = scipy.special.comb(parameters['number_of_vertices'], 2, exact = True)*scipy.special.comb(parameters['max_deg_monomials'], 2, exact = True) + parameters['number_of_vertices']*parameters['max_deg_monomials'] + 1
        parameters['L'] = int(round(L))
        
    #Determine if the columns of the library matrix will be normalized    
    parameters['normalize_cols'] =   params.get('normalize_cols', False)
    
    #Choose the nonadapted basis functions 
    parameters['use_canonical'] =  params.get('use_canonical', False)
    
    #Choose to adapt the basis functions wrt invariant density 
    parameters['use_orthonormal'] =  params.get('use_orthonormal', True)
    
    #Use the soft thresholding method for model selection
    parameters['use_soft_thresholding'] =  params.get('use_soft_thresholding', False)
    
    #Solve the quadratically constrained basis pursuit
    parameters['noisy_measurement'] =  params.get('noisy_measurement', True)
    parameters['noise_magnitude'] = params.get('noise_magnitude')
    
    #Lower and upper bound of the phase space of the isolated dynamics
    parameters['lower_bound'] = params.get('lower_bound', 0)
    parameters['upper_bound'] = params.get('upper_bound', 1)
    
    #To use Lebesgue integration
    parameters['use_lebesgue'] = params.get('use_lebesgue', False)
    
    #To estime the kernel density function
    parameters['use_kernel'] = params.get('use_kernel', True) 
    
    #To use one-dimensional integration
    parameters['use_integral_1d'] = params.get('use_integral_1d', True)
    
    #To use clustered density function
    parameters['single_density'] = params.get('single_density', False)
    
    #Type of measures
    if(parameters['use_lebesgue']):
        parameters['density'] = params.get('density', lambda x : 1.0)   #Lebesgue measure:
        parameters['type_density'] = params.get('type_density', 'Leb')
        
    
    if(parameters['use_orthonormal']):
        params_ = parameters.copy()
        
        parameters = params_cluster(params['cluster_list'], params_) 
        
        #To use orthonormal functions saved in a file before hand
        parameters['use_orthnorm_func_data'] = params.get('use_orthnorm_func_data', True)
        
        #To save the orthonormal functions in a file
        parameters['save_orthnormfunc'] = params.get('save_orthnormfunc', False)
        
        #To construct the orthonormal functions from a pre defined reduced basis
        parameters['build_from_reduced_basis'] = params.get('build_from_reduced_basis', False)
        
        #In case of using the file to construct the orthonormal functions
        if(parameters['use_orthnorm_func_data']):
            parameters['N_inputdata'] = params.get('N_inputdata', 2)
            parameters['max_deg_inputdata'] =  params.get('max_deg_inputdata', 10)           
            
            #Generate an optional name for file name to save basis functions
            orthnorm_func_filename = "orthnorm_func_N_{}_max_deg_{}".format(parameters['N_inputdata'], parameters['max_deg_inputdata'])
            
            #If the file name is given, it is saved in the parameters dictionary
            parameters['orthnorm_func_filename'] = params.get('orthnorm_func_filename', orthnorm_func_filename)
            
            if os.path.isfile(parameters['orthnorm_func_filename']):
                #Load basis functions using class SympyDict from module tools
                parameters['orthnormfunc'] = tools.SympyDict.load(parameters['orthnorm_func_filename'])
            if not os.path.isfile(parameters['orthnorm_func_filename']):
                parameters['orthnormfunc'] = dict()
                
        #To determine a mapping between two distinct basis functions. 
        #They are determined by number of vertices (N) and max_deg_monomials
        parameters['orthnorm_func_map'] = params.get('orthnorm_func_map', False)
            
        if(parameters['orthnorm_func_map']):
            #Generate the indexes of the basis functions for the given values
            #of N and max_deg_monomials
            power_indices = polb.index_iteration(parameters)
            
            #Construct a power indexes (temporary) to map from the saved 
            #basis function on the file and the new basis functions
            temp_params = dict()
            temp_params['number_of_vertices'] = parameters['N_inputdata']
            temp_params['max_deg_monomials'] = parameters['max_deg_inputdata']
            temp_params['expansion_crossed_terms'] = parameters['expansion_crossed_terms']
            temp_power_indices = polb.index_iteration(temp_params)
            
            #Function map_power_indices maps bijectively 
            #temp_power_indices to power_indices
            parameters['orthnormfunc'], pi_transf = tools.map_power_indices(parameters, temp_power_indices, power_indices)
        
            #To adjust minor differences
            new_power_indices_ = tools.delete_equal_rows(pi_transf, power_indices)
            new_power_indices_ = np.vstack((pi_transf, new_power_indices_))
            
            #Pos processed power indices is saved in the parameters dictionary
            parameters['power_indices'] = new_power_indices_
        
        #Name of experiment is composed with the type of technique we use.
        parameters['exp_name'] = parameters['exp_name'] + '_' + 'ADAPT'
        parameters['exp_name'] = parameters['exp_name'] + '_' + parameters['type_density']
    else:
        parameters['exp_name'] = parameters['exp_name'] + '_' + 'NONADAPT'
    
    if(parameters['normalize_cols']):
        parameters['exp_name'] = parameters['exp_name'] + '_NORMED'
    
    parameters['network_name'] =  params.get('network_name')
    ##### Identification for output
    out_direc = os.path.join(parameters['network_name'], parameters['exp_name'])
    #if os.path.isdir(out_direc) == False:
        #os.makedirs(out_direc)
    scenario_output = os.path.join(out_direc, 'n_{}_order_{}'.format(parameters['number_of_vertices'], parameters['max_deg_monomials']))
    
    #Filename for output results is saved as well.
    parameters['output_filename'] = params.get('output_filename', scenario_output)
    
    return parameters
