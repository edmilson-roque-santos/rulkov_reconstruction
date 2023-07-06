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
params_plot = {'axes.labelsize': 16,
              'axes.titlesize': 16,
              'axes.linewidth': 1.0,
              'axes.xmargin':0, 
              'axes.ymargin': 0,
              'legend.fontsize': 16,
              'xtick.labelsize': 15,
              'ytick.labelsize': 15,
              'figure.figsize': (8, 6),
              'figure.titlesize': 20,
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
        

def plot_time_series(ax, X_time_series, perc_view = 0.8, sharex = True):
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
            
            if not sharex:
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

def plot_isolated_comp(ax, Y_t, z_t, id_node, id_x = 0):
    colors = ['silver', 'midnightblue', 'gray']
    
    if id_x == 0:
        ax.set_ylabel(r'$f^{x}(u, v)$')
        ax.set_xlabel(r'$u$')
    else:
        ax.set_ylabel(r'$f^{y}(u, v)$')
        ax.set_xlabel(r'$v$')   
    
    interv = np.arange(np.min(Y_t[:, id_node]), np.max(Y_t[:, id_node]), 0.001)
    r_t = np.zeros(2*interv.shape[0])
    r_t[id_x::2] = interv
    r_t_ = rulkov.rulkov_map(r_t)
    
    ax.plot(interv, r_t_[id_x::2], '-', label = r'True',
            linewidth = 5.2, 
            color = colors[0])
    ax.plot(interv, z_t, '--', label = r'Reconstructed',
            color = colors[1])
    ax.set_xlim(np.min(np.min(Y_t[:, id_node])), np.max(Y_t[:, id_node]))
    ax.set_ylim(np.min(r_t_[id_x::2]), np.max(r_t_[id_x::2]))

def ax_plot_graph(ax, net_name, plot_net_alone=False):
    '''
    Plot the ring graph

    Parameters
    ----------
    ax : Matplotlib Axes object
        Draw the graph in the specified Matplotlib axes.
    plot_net_alone : boolean, optional
        To plot the network itself outside an environment. The default is False.

    Returns
    -------
    None.

    '''
    colors = ['darkgrey', 'orange', 'darkviolet', 'darkslategrey', 'silver']

    G_true = nx.read_edgelist("network_structure/{}.txt".format(net_name),
                        nodetype = int, create_using = nx.Graph)
    #pos_true = nx.circular_layout(G_true)
    pos_true = nx.bipartite_layout(G_true, nodes = [0])
    nx.draw_networkx_nodes(G_true, pos = pos_true,
                           ax = ax, node_color = colors[3], 
                           linewidths= 2.0,
                           node_size = 550,
                           alpha = 1.0)
    nx.draw_networkx_nodes(G_true, pos = pos_true,
                           node_color = colors[0], 
                           node_size = 500,
                           ax = ax,
                           alpha = 1.0)
    
    nx.draw_networkx_edges(G_true,pos = pos_true, 
                           ax = ax,
                           edgelist = list(G_true.edges()), 
                           edge_color = colors[4],
                           arrows = True,
                           arrowsize = 7,
                           width = 1.0,
                           alpha = 1.0)
    ax.margins(0.6)
    ax.axis("off")
    if plot_net_alone:
        ax.set_title('{}'.format('Original Network'))


def compare_basis(exp_dictionary, net_name):
    '''
    Given a experiment dict, it calculates the performance of the reconstruction.

    Parameters
    ----------
    exp_dictionary : dict
        Output results dictionary.
    net_name : str
        Filename.

    Returns
    -------
    lgth_vector : numpy array 
        Array with length of time series vector.
    FP_comparison : numpy array
        False positive proportion for each length of time series.
    FN_comparison : numpy array
        False negative proportion for each length of time series.
    d_matrix : TYPE
        DESCRIPTION.

    '''
    
    exp_vec = list(exp_dictionary['exp_params'].keys())
    lgth_endpoints = exp_dictionary['lgth_endpoints']
    G = nx.read_edgelist("network_structure/{}.txt".format(net_name),
                        nodetype = int, create_using = nx.Graph)
    
    N = 2*len(G)
    
    lgth_vector = np.arange(lgth_endpoints[0], lgth_endpoints[1],
                                      lgth_endpoints[2], dtype = int)
    
    dim_comparison = np.zeros((len(exp_vec), lgth_vector.shape[0], N))

    for id_exp in range(len(exp_vec)):
        for id_key in range(len(lgth_vector)):
            key = lgth_vector[id_key]

            for id_node in range(N):
                dim_comparison[id_exp, id_key, id_node] = exp_dictionary[exp_vec[id_exp]][key][id_node]
                
    return lgth_vector, dim_comparison

def plot_hist_ker(ax, lgth_vector, dim_comparison):
    col = mpl.color_sequences['tab20c']
    nseeds, num_exp, num_lgth_vec, N = dim_comparison.shape
    id_vec = np.arange(0, N, 2, dtype=int)
    data_coord = np.zeros((nseeds*id_vec.shape[0], num_lgth_vec))

    for id_exp in range(num_exp):
        data = dim_comparison[:, id_exp, :, :]
        for counter in range(num_lgth_vec):
            data_coord[:, counter] = data[:, counter, id_vec].flatten()
        '''
        ax.violinplot(data_coord, 
                      positions = lgth_vector,
                      showmeans = True, 
                      showmedians = True)
        '''
        ax.plot(lgth_vector, data_coord.mean(axis=0), '-o',
                color = col[id_exp])
        ax.fill_between(lgth_vector, 
                        data_coord.mean(axis=0)-data_coord.std(axis=0), 
                        data_coord.mean(axis=0)+data_coord.std(axis=0), 
                        color = col[id_exp],
                        alpha=0.2)

        id_vec = id_vec + 1
        data = dim_comparison[:, id_exp, :, :]
        for counter in range(num_lgth_vec):
            data_coord[:, counter] = data[:, counter, id_vec].flatten()
        '''
        ax.violinplot(data_coord, 
                      positions = lgth_vector,
                      showmeans = True, 
                      showmedians = True)
        '''
        ax.plot(lgth_vector, data_coord.mean(axis=0), '-o',
                color = col[id_exp+1])
        ax.fill_between(lgth_vector, 
                        data_coord.mean(axis=0)-data_coord.std(axis=0), 
                        data_coord.mean(axis=0)+data_coord.std(axis=0), 
                        color = col[id_exp],
                        alpha=0.2)

def plot_comparison_analysis(ax, exp_dictionary, net_name, plot_legend):    
    '''
    To plot a comparison between EBP and BP for increasing the length of time series.

    Parameters
    ----------
    ax : Matplotlib Axes object
        Draw the graph in the specified Matplotlib axes.
    exp_dictionary : dict
        Dictionary carrying the information about the experiments to be plotted.
    net_name : str
        Network filename.
    plot_legend : boolean
        To plot the legend inside the ax panel.

    Returns
    -------
    None.

    '''
    seeds = list(exp_dictionary.keys())
    Nseeds = int(len(seeds))
    G = nx.read_edgelist("network_structure/{}.txt".format(net_name),
                        nodetype = int, create_using = nx.Graph)
    
    N = 2*len(G)
    
    lgth_endpoints = exp_dictionary[seeds[0]]['lgth_endpoints']
    lgth_vector = np.arange(lgth_endpoints[0], lgth_endpoints[1],
                                      lgth_endpoints[2], dtype = int)
    
    num_exp_vec = len(list(exp_dictionary[seeds[0]]['exp_params'].keys()))
    dim_comparison = np.zeros((Nseeds, num_exp_vec, lgth_vector.shape[0], N))
    
    
    for id_seed in range(Nseeds):
        lgth_vector, dim_comparison[id_seed, :, :, :]  = \
                                                        compare_basis(exp_dictionary[seeds[id_seed]], 
                                                                      net_name)
    plot_hist_ker(ax, lgth_vector-2000, dim_comparison)
    ax.set_xlabel(r'length of time series $n$')
    ax.set_ylabel(r'dim($\ker{\Psi}(\bar{x})$)')
    #ax.set_yscale('log')
    
    
def plot_lgth_dependence(net_name, exps_dictionary, title, filename = None):    
    '''
    Plot the reconstruction performance vs length of time series.


    Parameters
    ----------
    net_name : str
        Network filename.
    exps_dictionary : dict
        Dictionary carrying the information about the experiments to be plotted.
    title : str
        Title to be plotted.
    filename : str, optional
        Saving pdf filename. The default is None.

    Returns
    -------
    None.

    '''
    keys = list(exps_dictionary.keys())
    n_cols = int(len(keys))
    
    fig = plt.figure(figsize = (6, 2), dpi = 300)
    
    plot_legend = True
    for id_col in range(n_cols):
        gs1 = GridSpec(nrows=1, ncols=1, figure=fig)
        
        exp_dictionary = exps_dictionary[keys[id_col]]
        
        ax1 = fig.add_subplot(gs1[0])
        
        plot_comparison_analysis(ax1, exp_dictionary, net_name, plot_legend)
        if plot_legend:
            plot_legend = False
        fig.suptitle(title[id_col])
    
    
    if filename == None:
        fig.suptitle('dimension of kernel')
        plt.show()
    else:
        plt.savefig(filename+".pdf", format='pdf', bbox_inches='tight')
        
    return     

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

def Fig_1(net_dict, net_name, id_node = 0, filename = None):

    fig_ = plt.figure(layout='constrained', 
                      figsize = (10, 6), 
                      dpi = 300)
    subfigs = fig_.subfigures(1, 2)
    
    fig = subfigs[0]
    
    subfigsnest = fig.subfigures(2, 1, hspace = 0.01, height_ratios=[1, 2.5])
    
    ax = subfigsnest[0].subplots(1, 1)
    ax_plot_graph(ax, net_name)
    subfigsnest[0].suptitle(r'a)', x=0.0, ha='left')

    ax1 = subfigsnest[1].subplots(2, 1, sharex=True)
    plot_time_series(ax1, net_dict['Y_t'], perc_view=1.0)
    subfigsnest[1].suptitle(r'b)', x=0.0, ha='left')
    
    Y_t = net_dict['Y_t']
        
    Z = net_dyn.gen_isolated_map_model(net_dict)
    
    fig = subfigs[1]
    (ax3, ax4) = fig.subplots(2, 1)
    fig.suptitle(r'c)', x=0.0, ha='left')
    
    z_t = Z[id_node]
    plot_isolated_comp(ax3, Y_t, z_t, id_node, id_x = 0)
    
    z_t = Z[id_node+1]
    plot_isolated_comp(ax4, Y_t, z_t, id_node+1, id_x = 1)
    ax4.legend(loc=0)
    
    
    if filename == None:
        #plt.tight_layout()

        plt.show()
    else:
        #plt.tight_layout()
        plt.savefig(filename+".pdf", format = 'pdf')    

def fig_loc_eval(net_dict, id_node = 0, filename = None):

    Y_t = net_dict['Y_t']
    
    fig, ax = plt.subplots(figsize = (10, 3), dpi=300)
    
    ax.plot(Y_t[:, id_node+1], Y_t[:, id_node], '-')
    
    if filename == None:
        plt.tight_layout()

        plt.show()
    else:
        plt.tight_layout()
        plt.savefig(filename+".pdf", format = 'pdf') 
    
