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
import scipy.special
from scipy import optimize        
import sympy as spy


import h5dict

from EBP.base_polynomial import triage as trg
from EBP.base_polynomial import poly_library as polb
from EBP.modules.rulkov import rulkov

from EBP import tools, net_dyn

# Set plotting parameters
params_plot = {'axes.labelsize': 16,
              'axes.titlesize': 16,
              'axes.linewidth': 1.0,
              'axes.xmargin':0.1, 
              'axes.ymargin': 0.1,
              'legend.fontsize': 12,
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

def gen_X_time_series_sample(lgth_time_series, net_name = 'star_graph_15'):
        
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
    parameters['coupling'] = 0.01
    #==========================================================#
    net_dynamics_dict = dict()
    net_dynamics_dict['adj_matrix'] = parameters['adj_matrix'] - degree*np.identity(A.shape[0])
    
    transient_time = 5000
    
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
        

def plot_time_series(ax, X_time_series, perc_view = 0.05, sharex = True):
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
        ax.set_ylabel(r'$f(u, v)$')
        ax.set_xlabel(r'$u$')
    else:
        ax.set_ylabel(r'$g(u, v)$')
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

def ax_plot_graph(ax, G_true, pos_true, ns = 500, plot_net_alone=False):
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

    
    nx.draw_networkx_nodes(G_true, pos = pos_true,
                           ax = ax, node_color = colors[3], 
                           linewidths= 2.0,
                           node_size = ns+50,
                           alpha = 1.0)
    nx.draw_networkx_nodes(G_true, pos = pos_true,
                           node_color = colors[0], 
                           node_size = ns,
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
    ax.margins(0.4)
    ax.axis("off")
    if plot_net_alone:
        ax.set_title('{}'.format('Original Network'))


def ker_dim_compare(exp_dictionary, net_name):
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
                try:
                    dim_comparison[id_exp, id_key, id_node] = exp_dictionary[exp_vec[id_exp]][key][id_node]
                except:
                    dim_comparison[id_exp, id_key, id_node] = 0
                    print('key', key, 'is not in the file')
    return lgth_vector, dim_comparison

def error_compare(exp_dictionary, net_name):
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
    
    error_comparison = np.zeros((len(exp_vec), lgth_vector.shape[0], N))

    for id_exp in range(len(exp_vec)):
        for id_key in range(len(lgth_vector)):
            key = lgth_vector[id_key]
            try:
                error_comparison[id_exp, id_key, :] = exp_dictionary[exp_vec[id_exp]][key]['error']
            except:
                error_comparison[id_exp, id_key, :] = np.zeros(N)
                print('key', key, 'is not in the file')
    
    return lgth_vector, error_comparison


def compare_basis_net_size(exp_dictionary):
    '''
    Given a experiment dict, it calculates the performance of the reconstruction.

    Parameters
    ----------
    exp_dictionary : dict
        Output results dictionary.
    
    Returns
    -------
    size_vector : numpy array 
        Array with length of time series vector.
    n_critical_comparison : numpy array
        n_critical for each length of time series.
    
    '''
    exp_vec = list(exp_dictionary['exp_params'].keys())
    size_endpoints = exp_dictionary['size_endpoints']
    
    size_vector = np.arange(size_endpoints[0], size_endpoints[1],
                                      size_endpoints[2], dtype = int)
    
    n_critical_comparison = np.zeros((len(exp_vec), size_vector.shape[0]))
    
    for id_exp in range(len(exp_vec)):
        for id_key in range(len(size_vector)):
            key = size_vector[id_key]
            n_critical = exp_dictionary[exp_vec[id_exp]][key]['n_critical']
            n_critical_comparison[id_exp, id_key] = n_critical
            
            
    return size_vector, n_critical_comparison

def plot_hist_ker(ax, lgth_vector, dim_comparison, 
                  plot_ycoord = False, col = mpl.color_sequences['tab20c']):
    
    nseeds, num_exp, num_lgth_vec, N = dim_comparison.shape
    id_vec = np.arange(0, N, 2, dtype=int)
    data_coord = np.zeros((nseeds*id_vec.shape[0], num_lgth_vec))

    for id_exp in range(num_exp):
        data = dim_comparison[:, id_exp, :, :]
        for counter in range(num_lgth_vec):
            data_coord[:, counter] = data[:, counter, id_vec].flatten()
        
       
        ax.plot(lgth_vector, data_coord.mean(axis=0), 
                '-o',
                color = col[id_exp],
                label=r'fast variable')
        ax.fill_between(lgth_vector, 
                        data_coord.mean(axis=0)-data_coord.std(axis=0), 
                        data_coord.mean(axis=0)+data_coord.std(axis=0), 
                        color = col[id_exp],
                        alpha=0.2)

        id_vec = id_vec + 1
        data = dim_comparison[:, id_exp, :, :]
        for counter in range(num_lgth_vec):
            data_coord[:, counter] = data[:, counter, id_vec].flatten()
        if plot_ycoord:
            ax.plot(lgth_vector, data_coord.mean(axis=0), 
                    '-o',
                    color = col[id_exp+1],
                    label=r'slow variable')
            ax.fill_between(lgth_vector, 
                            data_coord.mean(axis=0)-data_coord.std(axis=0), 
                            data_coord.mean(axis=0)+data_coord.std(axis=0), 
                            color = col[id_exp+1],
                            alpha=0.2)
        
        #ax.legend(loc=0,fontsize=12)
        
def plot_error_comparison(ax, lgth_vector, dim_comparison,
                          title, col = mpl.color_sequences['tab20c']):
    
    nseeds, num_exp, num_lgth_vec, N = dim_comparison.shape
    id_vec = np.arange(0, N, 2, dtype=int)
    data_coord = np.zeros((nseeds*id_vec.shape[0], num_lgth_vec))

    for id_exp in range(num_exp):
        data = dim_comparison[:, id_exp, :, :]
        for counter in range(num_lgth_vec):
            data_coord[:, counter] = data[:, counter, id_vec].flatten()
        
        ax.plot(lgth_vector, data_coord.mean(axis=0), 
                '-o',
                color = col[id_exp],
                label=title)
        ax.fill_between(lgth_vector, 
                        data_coord.mean(axis=0)-data_coord.std(axis=0), 
                        data_coord.mean(axis=0)+data_coord.std(axis=0), 
                        color = col[id_exp],
                        alpha=0.2)

        id_vec = id_vec + 1
        

def plot_comparison_analysis(ax, exp_dictionary, net_name, method, title,
                             plot_ycoord, plot_def, plot_legend, 
                             color = None):    
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
                                method(exp_dictionary[seeds[id_seed]], 
                                                net_name)
    lgth_vector = lgth_vector
    
    if plot_def:
        plot_hist_ker(ax, lgth_vector, dim_comparison, plot_ycoord, col = mpl.color_sequences['tab20c'])
        ax.hlines(1, lgth_vector[0], lgth_vector[-1],
                  colors='k',
                  linestyles='dashed')
        ax.set_ylabel(r'def($\Psi(\bar{x})$)')
    
    else:
        plot_error_comparison(ax, lgth_vector, dim_comparison, title,col = [color])
        ax.set_ylabel(r'$E_i$')
        ax.set_xlim(300, 2050)
        if plot_legend:
            ax.legend(loc=0)
    ax.set_xlabel(r'length of time series $n$')
    
    #ax.set_ylim(0, 1300)
    #ax.set_title(r'Kernel')
    ax.set_yscale('log')
    
def plot_comparison_n_critical(ax, exp_dictionary, plot_legend):    
    '''
    To plot the comparison between EBP and BP in the experiment: n_c vs N

    Parameters
    ----------
    ax : Matplotlib Axes object
        Draw the graph in the specified Matplotlib axes.
    exp_dictionary : dict
        Dictionary carrying the information about the experiments to be plotted.
    plot_legend : boolean
        To plot the legend inside the ax panel.

    Returns
    -------
    None.

    '''
    seeds = list(exp_dictionary.keys())
    Nseeds = int(len(seeds))
    
    size_endpoints = exp_dictionary[seeds[0]]['size_endpoints']
    size_vector = np.arange(size_endpoints[0], size_endpoints[1],
                                      size_endpoints[2], dtype = int)
    
    n_c_comparison = np.zeros((Nseeds, 2, size_vector.shape[0]))
    
    
    for id_seed in range(Nseeds):
        size_vector, n_c_comparison[id_seed, :, :] = compare_basis_net_size(exp_dictionary[seeds[id_seed]])
    
    avge_nc_comparison = n_c_comparison.mean(axis = 0)    
    std_nc_comparison = n_c_comparison.std(axis = 0)     
    x_v = size_vector+1
    
    col = mpl.color_sequences['tab20b']
    ax.fill_between(x_v, 
                    avge_nc_comparison[0, :], 
                    3000, 
                    color = col[3],
                    alpha=1.0)
    
    N_vector = np.arange(2, 12, 1, dtype = int)
    m_vec = np.zeros(N_vector.shape[0])
    r = 3
    for i, N in enumerate(N_vector):
        m_N_ = scipy.special.comb(2*N, 2, exact = True)*scipy.special.comb(r, 2, exact = True) + 2*N*r + 1
        m_N = 2*m_N_
        m_vec[i] = m_N
    ax.fill_between(x_v, 
                    m_vec, 
                    avge_nc_comparison[0, :], 
                    color = col[2],
                    alpha=1.0)
    
    ax.plot(x_v, avge_nc_comparison[0, :], 'o-', 
            label=r'$n_0$', 
            color='k')
    ax.fill_between(x_v, 
                    avge_nc_comparison[0, :]-std_nc_comparison[0, :], 
                    avge_nc_comparison[0, :]+std_nc_comparison[0, :], 
                    color = 'k',
                    alpha=0.5)
    
    from scipy import optimize

    ##########
    # Fitting the data -- Least Squares Method
    ##########

    # Power-law fitting is best done by first converting
    # to a linear equation and then fitting to a straight line.
    # Note that the `logyerr` term here is ignoring a constant prefactor.
    #
    #  y = a * x^b
    #  log(y) = log(a) + b*log(x)
    #

    # define our (line) fitting function
    fitfunc = lambda p, x: p[0] + p[1] * x
    errfunc = lambda p, x, y, err: (y - fitfunc(p, x)) / err
    y_data = avge_nc_comparison[0, 2:]
    logy =y_data
    x = (size_vector[2:]+1)
    yerr = std_nc_comparison[0, 2:]
    logyerr = 1#yerr / y_data
    pinit = [1.0, -1.0]
    out = optimize.leastsq(errfunc, pinit,
                           args=(x**2, logy, logyerr), full_output=1)

    pfinal = out[0]
    covar = out[1]
    print('pfinal-n_0', pfinal)
    print('covar-n_0', covar)

    index = pfinal[1]
    amp = pfinal[0]

    indexErr = np.sqrt( covar[1][1] )
    ampErr = np.sqrt( covar[0][0] ) * amp
    leastsq_regression = amp + (index)*x**2 #np.power(10, amp + x*(index))
    amp_ = '{:.2f}'.format(amp)
    a = '{:.2f}'.format(index)
    print('exp-n_0', r'${} + {} N^2$'.format(amp_, a))
    ax.plot(x, leastsq_regression, ls = 'dashed', color='tab:orange',
            label = r'${} + {} N^2$'.format(amp_, a),
            alpha=0.9)
    
    ax.legend(loc=0)
    ax.set_ylabel(r'$n$')
    #ax.set_yscale('log')
    plt.setp(ax.get_xticklabels(), visible=True)
    ax.set_xlabel(r'$N$')
    
def plot_lgth_dependence(net_name, exps_dictionary, title, 
                         method = ker_dim_compare,
                         plot_ycoord= False,
                         plot_def = True,
                         filename = None):    
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
    
    if n_cols > 1:
        color = ['darkcyan', 'midnightblue']
    else:
        color = [None]
    
    fig_ = plt.figure(figsize = (7, 2), dpi = 300)
    
    subfigs = fig_.subfigures(1, 2)
    
    fig1 = subfigs[0]
    
    ax = fig1.subplots(1, 1)
    G_true = nx.read_edgelist("network_structure/{}.txt".format(net_name),
                        nodetype = int, create_using = nx.Graph)
    
    pos_true = nx.spring_layout(G_true)
    
    ax_plot_graph(ax, G_true, pos_true, ns = 100)
    ax.set_title(r'a)', loc='left')
    
    fig = subfigs[1]
    gs1 = GridSpec(nrows=1, ncols=1, figure=fig)
    ax1 = fig.add_subplot(gs1[0])
    plot_legend = True
    for id_col in range(n_cols):
        exp_dictionary = exps_dictionary[keys[id_col]]
        
        plot_comparison_analysis(ax1, exp_dictionary, net_name, method, title[id_col], 
                                 plot_ycoord, plot_def, plot_legend, 
                                 color[id_col])
        
        if plot_legend:
            plot_legend = True
        
        ax1.set_title(r'b)', loc='left')
    #fig.suptitle('b)')
    if filename == None:
        
        plt.show()
    else:
        plt.savefig(filename+".pdf", format='pdf', bbox_inches='tight')
        
    return     

def plot_n_c_size(exps_dictionary, title, filename = None,
                  plot_legend_global = True):     
    '''
    Plot the n_c vs N.

    Parameters
    ----------
    exp_dictionary : dict
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
    
    fig = plt.figure(figsize = (6, 3), dpi = 300)
    
    if plot_legend_global:
        plot_legend = True
    else:
        plot_legend = False
    for id_col in range(n_cols):
        gs1 = GridSpec(nrows=1, ncols=1, figure=fig)
        exp_dictionary = exps_dictionary[keys[id_col]]
        ax1 = fig.add_subplot(gs1[0])
        #ax2 = fig1.add_subplot(gs1[1])
        
        plot_comparison_n_critical(ax1, exp_dictionary, plot_legend)
        n_1_vs_N(ax1, r = 3)
        if plot_legend:
            plot_legend = False
        #fig1.suptitle(title[1], x = 0.05)
    
    #fig.suptitle(r'Performance diagram')

    if filename == None:
        plt.show()
    else:
        plt.savefig(filename+".pdf", format='pdf', bbox_inches='tight')
    
def num_basis(r=3):
    
    N_vector = np.arange(2, 50, 1, dtype = int)
    m_N = np.zeros(N_vector.shape[0])
    for i, N in enumerate(N_vector):
        m_N[i] = scipy.special.comb(2*N, 2, exact = True)*scipy.special.comb(r, 2, exact = True) + 2*N*r + 1
    
    fig, ax = plt.subplots()
    ax.plot(N_vector, m_N, 'k-')
    plt.show()

def f(x, m_N):
    delta = m_N - x
    min_fun = delta*np.log(delta)-m_N
    return min_fun

def min_length_time_series(N = 3, r = 3):
    
    m_N_ = scipy.special.comb(2*N, 2, exact = True)*scipy.special.comb(r, 2, exact = True) + 2*N*r + 1
    m_N = 2*m_N_
    
    root = optimize.newton(f, m_N-10, args=(m_N, ))
    
    diff = m_N - root
    print(root, diff)
    n = np.arange(1, m_N, 1, dtype = int)
    
    delta = m_N - n
    min_fun = delta*np.log(delta)
    min_fun_planted = np.log(delta)*delta**2
    min_fun_linear = delta
    
    fig, ax = plt.subplots()
    ax.plot(n, min_fun, '-')
    ax.plot(n, min_fun_planted, '--')
    ax.plot(n, min_fun_linear, 'r--')
    ax.hlines(m_N, n[0], n[-1], colors = 'black')
    ax.set_ylim(1, m_N+100)
    ax.vlines(root, 1, m_N)
    plt.show()    

def n_1_vs_N(ax = None, r = 3):
    
    N_vector = np.arange(2, 12, 1, dtype = int)
    m_vec = np.zeros(N_vector.shape[0])
    diff_vec = np.zeros(N_vector.shape[0])
    root_vec = np.zeros(N_vector.shape[0])
    for i, N in enumerate(N_vector):
        m_N_ = scipy.special.comb(2*N, 2, exact = True)*scipy.special.comb(r, 2, exact = True) + 2*N*r + 1
        m_N = 2*m_N_
        m_vec[i] = m_N
        
        root_vec[i] = optimize.newton(f, m_N-10, args=(m_N, ))
        
        diff_vec[i] = m_N - root_vec[i]
    if ax is None:
        fig, ax = plt.subplots(2,1, sharex=True)
        ax[0].plot(N_vector, m_vec, '-')
        ax[0].plot(N_vector, root_vec, '-')
        ax[0].set_yscale('log')
    
        ax[1].plot(N_vector, diff_vec, '-')
        ax[1].set_yscale('log')
        plt.show()   
    
    else:
        ax.plot(N_vector, root_vec, '--', color='black', label=r'$n_1 \propto N^2$')
        ax.plot(N_vector, m_vec, '-.', color='black', label=r'$2 m(N, r)$')
        
        col = mpl.color_sequences['tab20b']
        ax.fill_between(N_vector, 
                        root_vec, 
                        m_vec, 
                        color = col[1],
                        alpha=1.0)
        
        ax.fill_between(N_vector, 
                        0, 
                        root_vec, 
                        color = col[0],
                        alpha=1.0)
        
        ax.text(9, 2500, r'IV', fontsize = 15)
        ax.text(10.5, 1700, r'III', fontsize = 15)
        ax.text(10.5, 1200, r'II', fontsize = 15)
        ax.text(10, 500, r'I', fontsize = 15)
        ##########
        # Fitting the data -- Least Squares Method
        ##########

        # Power-law fitting is best done by first converting
        # to a linear equation and then fitting to a straight line.
        # Note that the `logyerr` term here is ignoring a constant prefactor.
        #
        #  y = a * x^b
        #  log(y) = log(a) + b*log(x)
        #

        # define our (line) fitting function
        fitfunc = lambda p, x: p[0] + p[1] * x
        errfunc = lambda p, x, y, err: (y - fitfunc(p, x)) / err
        y_data = root_vec
        logy = y_data
        x = N_vector
        yerr = 0.0001
        logyerr = yerr / y_data
        pinit = [1.0, -1.0]
        out = optimize.leastsq(errfunc, pinit,
                               args=(x**2, logy, logyerr), full_output=1)

        pfinal = out[0]
        covar = out[1]
        print('pfinal-n_1', pfinal)
        print('covar-n_1', covar)

        index = pfinal[1]
        amp = pfinal[0]

        indexErr = np.sqrt( covar[1][1] )
        ampErr = np.sqrt( covar[0][0] ) * amp
        leastsq_regression = amp + (index)*x**2
        amp_ = '{:.2f}'.format(amp)
        a = '{:.2f}'.format(index)
        print('exp-n_1', r'$n_1 = {} + {} N^2$'.format(amp_, a))
        
        '''
        ax.plot(x, leastsq_regression, '--', color='k',
                label = r'$n_1 = {} + {} N^2$'.format(amp_, a),
                alpha=0.8)
        '''
        ax.legend(loc=0)
    return diff_vec, root_vec
    
def latex_model(net_dict, N = 2):
    
    x_t = [spy.symbols('x_{}'.format(j)) for j in range(0, 2*N)]
    
    x_t_ = [spy.symbols('x_{}'.format(j)) for j in range(1, N+1)]
    y_t_ = [spy.symbols('y_{}'.format(j)) for j in range(1, N+1)]
    
    subs_ = dict()
    for i in range(1, 2*N):
        if np.mod(i, 2):
            subs_[x_t[i]] = y_t_[int(i/2)]
        else:
            subs_[x_t[i]] = x_t_[int(i/2)]
    print(subs_)
    symbs = net_dict['sym_node_dyn']
    for keys in symbs.keys():
        f = symbs[keys]
        g = f.subs(subs_)
        h = g.subs({x_t[0]:x_t_[0]})
        print(keys, spy.latex(spy.simplify(h).n(8)))
    
def plot_pareto_front(sparsity_of_vector, pareto_front):

    pos_different_zero = np.argwhere(np.absolute(pareto_front[:, 1]) > 1e-14)[:, 0]
    
    pos_minimum_error = np.argmin(np.absolute(pareto_front[pos_different_zero, 1]))       
    
    fig, ax = plt.subplots(1, 2, figsize = (6, 3), dpi=300)
    
    ax[0].set_title(r'a)', loc = 'left')
    ax[0].semilogx(sparsity_of_vector[:, 0], sparsity_of_vector[:, 1], 'o', color = 'dimgray')

    ax[0].scatter(sparsity_of_vector[pos_minimum_error, 0], 
                  sparsity_of_vector[pos_minimum_error, 1], 
                  s=80, 
                  facecolors='none', 
                  edgecolors='r')

    ax[0].set_xlabel(r"Threshold $\gamma$")
    ax[0].set_ylabel("Number of terms")
    ax[1].set_title(r'b)', loc = 'left')
    ax[1].semilogy(pareto_front[:, 0], pareto_front[:, 1], 'o', color = 'dimgray')
    ax[1].scatter(pareto_front[pos_minimum_error, 0], 
                  pareto_front[pos_minimum_error, 1], 
                  s=80, 
                  facecolors='none', 
                  edgecolors='r')
    
    ax[1].set_xlabel("Number of terms")
    ax[1].set_ylabel(r"Error $\|\Psi(\bar{\mathbf{x}}) u\|_1$")
    plt.tight_layout()
    
    filename='pareto_font'
    plt.savefig(filename+".pdf", format='pdf', bbox_inches='tight')
    plt.show()
    
def plot_corr(lgth_time_series, size, index):
    
    X_t = gen_X_time_series_sample(lgth_time_series, net_name = 'star_graph_{}'.format(size))
    #X_t = np.random.normal(size = (lgth_time_series, size))
    
    for id_ in index:
        lags, ax = tools.x_corr(X_t[:, id_], X_t[:, id_])
        lgn = int(lags.shape[0]/2)
        plt.semilogy(lags[lgn:], np.absolute(ax)[lgn:])
    
    #lags, cx = tools.x_corr(X_t[:, index[0]], X_t[:, index[1]])
    #plt.semilogy(lags[lgn:], np.absolute(cx)[lgn:])
    
    return ax 
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
    G_true = nx.read_edgelist("network_structure/{}.txt".format(net_name),
                        nodetype = int, create_using = nx.Graph)
    
    pos_true = nx.bipartite_layout(G_true, nodes = [0])
    
    ax_plot_graph(ax, G_true, pos_true)
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
    
