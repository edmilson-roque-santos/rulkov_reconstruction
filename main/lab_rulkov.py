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
from EBP.modules.rulkov import rulkov

from EBP import tools, net_dyn

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
    parameters['adj_matrix'] = A 
    parameters['coupling'] = 0.0
    #==========================================================#
    net_dynamics_dict = dict()
    net_dynamics_dict['adj_matrix'] = parameters['adj_matrix'] - degree*np.identity(A.shape[0])
    
    transient_time = 2000
    
    net_dynamics_dict['f'] = rulkov.rulkov_map
    net_dynamics_dict['h'] = rulkov.diff_coupling_x
    net_dynamics_dict['max_degree'] = np.max(degree)
    net_dynamics_dict['coupling'] = parameters['coupling']
    net_dynamics_dict['random_seed'] = parameters['random_seed']
    net_dynamics_dict['transient_time'] = transient_time

    X_time_series = net_dyn.gen_net_dynamics(lgth_time_series, net_dynamics_dict)  
    
    return X_time_series

    

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
        

def plot_time_series(ax, X_time_series, perc_view = 0.8):
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
            
            ax[id_col].set_xlim((1 - perc_view)*n_iterations[-1], n_iterations[-1])
            
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

'''
def fig_comp_ts(net_dict, filename = None):
    #INCORRECT! One cannot expect to generate a long time series from the 
    reconstructed model.
        
    Y_t = net_dict['Y_t']
    t_test = Y_t.shape[0]
    
    y_0 = Y_t[0, :]
    Z = net_dyn.generate_net_dyn_model(y_0, t_test, net_dict)
    
    fig, ax = plt.subplots(1, 2, figsize = (10, 3), dpi=300)
 
    plot_time_series(ax, Z)
    if filename == None:
        plt.tight_layout()

        plt.show()
    else:
        plt.tight_layout()
        plt.savefig(filename+".pdf", format = 'pdf')
'''
def fig_comp_rm(net_dict, filename = None):
    
    params = net_dict['params']
    Y_t = net_dict['Y_t']
    
    Z = net_dyn.gen_return_map_model(net_dict)
    
    fig, ax = plt.subplots(1, 2, figsize = (10, 3), dpi=300)
 
    plot_return_map(ax, Y_t)
    
    id_node = 0
    
    interv = np.arange(np.min(Y_t[:, id_node]), np.max(Y_t[:, id_node]), 0.001)
    ax[0].plot(interv, Z[id_node]-2.84, 'ro', markersize=2)
    
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
