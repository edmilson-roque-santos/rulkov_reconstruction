"""
Scripts for visualizing and analysing the coupled Rulkov dynamics

Created on Thu Jun 15 11:24:57 2023

@author: Edmilson Roque dos Santos
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec    
import networkx as nx
import numpy as np
import os
from scipy import stats

import h5dict

from EBP.base_polynomial import triage as trg
from EBP.base_polynomial import poly_library as polb

from EBP import tools, net_dyn
from EBP.modules.rulkov import rulkov

# Set plotting parameters
params_plot = {'axes.labelsize': 15,
              'axes.titlesize': 15,
              'axes.linewidth': 1.0,
              'axes.xmargin':0, 
              'axes.ymargin': 0,
              'legend.fontsize': 18,
              'xtick.labelsize': 16,
              'ytick.labelsize': 16,
              'figure.figsize': (8, 6),
              'figure.titlesize': 18,
              'font.serif': 'Computer Modern Serif',
              'mathtext.fontset': 'cm',
              'axes.linewidth': 1.0
             }

plt.rcParams.update(params_plot)
plt.rc('text', usetex=True)

#=============================================================================#
#Generate the time series of coupled Rulkov maps
#=============================================================================#

def gen_X_time_series_sample(lgth_time_series, net_name = 'star_graphs_n_4_hub_coupled'):
        
    parameters = dict()
    parameters['lgth_time_series'] = lgth_time_series
    parameters['network_name'] = net_name
    parameters['random_seed'] = 1
    
    G = nx.read_edgelist("network_structure/{}.txt".format(parameters['network_name']),
                         nodetype = int, create_using = nx.Graph)
        
    parameters['number_of_vertices'] = len(nx.nodes(G))
    A = nx.to_numpy_array(G, nodelist = list(range(parameters['number_of_vertices'])))
    A = np.asarray(A)
    degree = np.sum(A, axis=0)
    parameters['adj_matrix'] = A - degree*np.identity(A.shape[0])
    parameters['coupling'] = 1e-1
    #==========================================================#
    net_dynamics_dict = dict()
    net_dynamics_dict['adj_matrix'] = parameters['adj_matrix']
    
    alpha, sigma, beta = 4.4, 0.001, 0.001
    
    net_dynamics_dict['f'] = lambda x: rulkov.rulkov_map(x, alpha, sigma, beta)
    net_dynamics_dict['h'] = lambda x: rulkov.diff_coupling_x(x, parameters['adj_matrix'])
    net_dynamics_dict['max_degree'] = np.max(degree)
    net_dynamics_dict['coupling'] = parameters['coupling']
    net_dynamics_dict['random_seed'] = parameters['random_seed']
    X_time_series = net_dyn.gen_net_dynamics(lgth_time_series, net_dynamics_dict)  
    
    return X_time_series

#=============================================================================#
#Generate orthonormal functions
#=============================================================================#

def generate_orthonorm_funct(X_time_series,
                             cluster_list,
                             exp_name = 'gen_orthf_cluster', 
                             max_deg_monomials = 2,
                             use_single = False,
                             use_crossed_terms = False):
    """
    Routine to calculate for each coupling strength the orthonormal functions 
    relative to the data which lies in the subset of the phase space.
    
    """
    
    ############# Construct the parameters dictionary ##############
    parameters = dict()
    
    parameters['exp_name'] = exp_name
    parameters['Nseeds'] = 1
    
    parameters['network_name'] = "rulkov"
    parameters['max_deg_monomials'] = max_deg_monomials
    parameters['expansion_crossed_terms'] = use_crossed_terms
    parameters['single_density'] = use_single 
    
    parameters['use_kernel'] = True
    parameters['normalize_coupling_function'] = False
    parameters['use_orthonormal'] = True
    parameters['use_canonical'] = False
   
    parameters['cluster_list'] = cluster_list
    
    ##### Identification for output
    folder='orth_data'
    outfilename = os.path.join(folder, '')
    outfile_functions = os.path.join(outfilename, exp_name)
    outfilename = os.path.join(outfilename, "")
    
    if os.path.isdir(outfile_functions) == False:
        os.makedirs(outfile_functions)
    outfile_functions = os.path.join(outfile_functions, "")
    
    X_time_series = X_time_series[:, :]
    lgth_time_series = X_time_series.shape[0]
    
    parameters['lower_bound'] = np.min(X_time_series)
    parameters['upper_bound'] = np.max(X_time_series)
    
    parameters['number_of_vertices'] = X_time_series.shape[1]
    parameters['length_of_time_series'] = X_time_series.shape[0] - 1
    
    parameters['X_time_series_data'] = X_time_series
    
    parameters['coupling'] = 1e-1
    parameters['threshold_connect'] = 1e-8
    
    for seed in range(1, parameters['Nseeds'] + 1):
        #Extract the time series for the state and map
        X_t = X_time_series[:-1, :]
        params = parameters.copy()
        
        params['random_seed'] = seed
        
        if params['use_orthonormal']:
            params['orthnorm_func_filename'] = outfile_functions+\
            "orthnorm_sig_{:.6f}_deg_{}_lgth_{}".format(parameters['coupling'], 
                                                        params['max_deg_monomials'],
                                                        lgth_time_series)
            
            #For the opto electronic data, we can not use build from reduced basis
            #The trick to reduce the number of basis functions does not work.
            params['build_from_reduced_basis'] = False
            params['save_orthnormfunc'] = True
            params = trg.triage_params(params)

            if not use_single:
                params = rulkov.params_cluster(parameters['cluster_list'], params)   
        
        PHI, params = polb.library_matrix(X_t, params)    
    

#=============================================================================#
#Plotting scripts
#=============================================================================#

def plot_return_map(ax, X_time_series):
    '''
    Plot return map for each node from multivariate time series.

    Parameters
    ----------
    ax : Matplotlib Axes object
    
    X_time_series : numpy array - size (length_of_time_series, number_of_vertices)
       Multivariate time series.
    
    Returns
    -------
    None.
    '''
    number_of_iterations = X_time_series.shape[0]
    N = X_time_series.shape[1]
    
    x_nodelist = np.arange(0, N, 2, dtype = int)
    y_nodelist = np.arange(1, N, 2, dtype = int)
    
    nodelists = [x_nodelist, y_nodelist]
    id_col = 0
    
    for nodelist in nodelists: 
        lower_bound = np.min(X_time_series[:, nodelist])
        upper_bound = np.max(X_time_series[:, nodelist])
        col = mpl.color_sequences['tab20c']
        id_color = 0
        for index in nodelist:
            ax[id_col].plot(X_time_series[:number_of_iterations-1, index], 
                       X_time_series[1:number_of_iterations, index], 
                       'o', 
                       color = col[id_color],
                       markersize=2)
            id_color = id_color + 1
            
        if id_col == 0:
            ax[id_col].set_title(r'a)', loc='left')
            ax[id_col].set_ylabel(r'$x(t + 1)$')
            ax[id_col].set_xlabel(r'$x(t)$')
        else:
            ax[id_col].set_title(r'b)', loc='left')
            ax[id_col].set_ylabel(r'$y(t + 1)$')
            ax[id_col].set_xlabel(r'$y(t)$')
        ax[id_col].set_xlim(lower_bound, upper_bound)

        id_col = id_col + 1
        

def plot_time_series(ax, X_time_series):
    '''
    Plot time series for each node from multivariate time series.

    Parameters
    ----------
    ax : Matplotlib Axes object
    
    X_time_series : numpy array - size (length_of_time_series, number_of_vertices)
       Multivariate time series.
    
    Returns
    -------
    None.
    '''
    N = X_time_series.shape[1]
    n_iterations = np.arange(X_time_series.shape[0])

    x_nodelist = np.arange(0, N, 2, dtype = int)
    y_nodelist = np.arange(1, N, 2, dtype = int)
    
    nodelists = [x_nodelist, y_nodelist]
    id_col = 0
    
    for nodelist in nodelists: 
    
        col = mpl.color_sequences['tab20c']
        id_color = 0
        for index in nodelist:
            ax[id_col].plot(n_iterations, X_time_series[:, index], 
                       '-', 
                       color = col[id_color])
            
            ax[id_col].set_xlim(0.80*n_iterations[-1], n_iterations[-1])
            
            id_color = id_color + 1
            
        if id_col == 0:
            ax[id_col].set_ylabel(r'$x(t)$')
            ax[id_col].set_xlabel(r'$t$')
        else:
            ax[id_col].set_ylabel(r'$y(t)$')
            ax[id_col].set_xlabel(r'$t$')

        id_col = id_col + 1


def plot_kernel_density(ax, X_time_series):
    '''
    Plot density function for each node from multivariate time series.

    Parameters
    ----------
    ax : Matplotlib Axes object
    
    X_time_series : numpy array - size (length_of_time_series, number_of_vertices)
       Multivariate time series.
    
    Returns
    -------
    None.
    '''
    number_of_iterations = X_time_series.shape[0]
    N = X_time_series.shape[1]
    x_nodelist = np.arange(0, N, 2, dtype = int)
    y_nodelist = np.arange(1, N, 2, dtype = int)
    
    nodelists = [x_nodelist, y_nodelist]
    id_col = 0
    
    for nodelist in nodelists: 
        
        lower_bound = np.min(X_time_series[:, nodelist])
        upper_bound = np.max(X_time_series[:, nodelist])
        interval = np.arange(lower_bound, upper_bound, 0.001)
           
        col = mpl.color_sequences['tab20c']
        id_color = 0
        for index in nodelist:
            Opto_orbit = X_time_series[: number_of_iterations, index]
            kernel = stats.gaussian_kde(Opto_orbit, bw_method = 0.05)
            ax[id_col].plot(interval, 
                      kernel(interval)/kernel.integrate_box_1d(lower_bound, upper_bound), 
                      label="{}".format(int(index/2)+1),
                      color = col[id_color])
            
            id_color = id_color + 1
        
        if id_col == 0:
            ax[id_col].set_title(r'c)', loc='left')
            ax[id_col].set_ylabel(r'$\rho(x)$')  
            ax[id_col].set_xlabel(r'$x$')
        else:
            ax[id_col].set_title(r'd)', loc='left')
            ax[id_col].set_ylabel(r'$\rho(y)$')        
            ax[id_col].set_xlabel(r'$y$')
            ax[id_col].legend(loc=0, ncol=3, fontsize=8)

        ax[id_col].set_xlim(lower_bound, upper_bound)
        id_col = id_col + 1
        
    return ax
    
def fig_return_map(X_time_series, filename=None):
   
    fig, ax = plt.subplots(2, 2, figsize = (8, 5), dpi=300)
    
    plot_return_map(ax[0, :], X_time_series)
    
    
    plot_kernel_density(ax[1, :], X_time_series)
    
    if filename == None:
        plt.tight_layout()

        plt.show()
    else:
        plt.tight_layout()
        plt.savefig(filename+".pdf", format = 'pdf')
    
        
def fig_time_series(X_time_series, filename=None):
    fig, ax = plt.subplots(1, 2, figsize = (10, 3), dpi=300)
 
    plot_time_series(ax, X_time_series)
    if filename == None:
        plt.tight_layout()

        plt.show()
    else:
        plt.tight_layout()
        plt.savefig(filename+".pdf", format = 'pdf')


#=============================================================================#
#Figure scripts
#=============================================================================#
def coupled_rulkov_figs_supp():
    X_t = gen_X_time_series_sample(7001)
    folder='Fig_supp'
    filename = folder+'/'+'return_map'
    fig_return_map(X_t[2000:, :],filename)
    
    filename = folder+'/'+'time_series'
    fig_time_series(X_t[2000:, :],filename)        
