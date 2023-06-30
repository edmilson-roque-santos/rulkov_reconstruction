"""
Collection of auxiliary methods and classes.

Created on --- 2020

@author: Edmilson Roque dos Santos
"""


import matplotlib.pyplot as plt
import networkx as nx 
import numpy as np
from numpy.random import default_rng
import os 
import re
from scipy import stats
from sklearn.metrics import mean_squared_error
import sympy as spy
from sympy.parsing.sympy_parser import parse_expr

import h5py

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

#========================================================#
#Network structure
#========================================================#
def ring_graph(N, filename = None):
    G = nx.cycle_graph(N, create_using=nx.Graph())
    
    if filename != None:
        nx.write_edgelist(G, filename+".txt", data=False)
    
    return G 

def star_graph(N, filename = None):
    G = nx.star_graph(N, create_using=nx.Graph())
    
    if filename != None:
        nx.write_edgelist(G, filename+".txt", data=False)
    
    return G 

def adjacent_edges(nodes, halfk): 
    N = len(nodes) 
    for i, u in enumerate(nodes): 
        for j in range(i+1, i+halfk+1): 
            v = nodes[j % N]
            yield u, v
            
def make_ring_lattice(N, filename = None, k = 3): 
    G = nx.Graph() 
    nodes = range(N) 
    G.add_nodes_from(nodes) 
    G.add_edges_from(adjacent_edges(nodes, k)) 
    
    if filename != None:
        nx.write_edgelist(G, filename+".txt", data=False)

    return G

def random_toy_net_model(random_seed = 1, 
                    num_of_cluster = 5,
                    N_nodes_cluster = 10,
                    mean_degree_in_cluster = 3,
                    mean_degree_intercluster = 4,
                    filename = None,
                    save_net = False, 
                    plot_net = False):
    '''
    

    Parameters
    ----------
    random_seed : TYPE, optional
        DESCRIPTION. The default is 1.
    num_of_cluster : TYPE, optional
        DESCRIPTION. The default is 5.
    N_nodes_cluster : TYPE, optional
        DESCRIPTION. The default is 10.
    mean_degree_in_cluster : TYPE, optional
        DESCRIPTION. The default is 3.
    mean_degree_intercluster : TYPE, optional
        DESCRIPTION. The default is 4.
    filename : TYPE, optional
        DESCRIPTION. The default is None.
    save_net : TYPE, optional
        DESCRIPTION. The default is False.
    plot_net : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    '''
    
    args = dict()
    args['random_seed'] = random_seed,
    args['num_of_cluster'] = num_of_cluster
    args['N_nodes_cluster'] =  N_nodes_cluster 
    args['mean_degree_in_cluster'] =  mean_degree_in_cluster
    args['mean_degree_intercluster'] = mean_degree_intercluster
    args['filename'] = filename
   
    np.random.seed(args['random_seed'])
    N = args['num_of_cluster']*args['N_nodes_cluster']
    
    nodelist = np.arange(0, N, 1, dtype = int)
    headnodes_list = nodelist[ : :args['N_nodes_cluster']]
    G = nx.Graph()
    G.add_nodes_from(nodelist)
    for id_cluster in range(args['num_of_cluster']):
        head_node = headnodes_list[id_cluster]
        cluster_list = np.arange(head_node, head_node + args['N_nodes_cluster'], 1, dtype = int)
        p = args['mean_degree_in_cluster']/args['N_nodes_cluster']
        H = nx.erdos_renyi_graph(args['N_nodes_cluster'], p, seed=np.random, directed=False)
        mapping = dict(zip(H, cluster_list))
        H = nx.relabel_nodes(H, mapping)
        edgelist = list(H.edges())
        G.add_edges_from(edgelist)
    
    size_intercluster = headnodes_list.shape[0]
    p = args['mean_degree_intercluster']/size_intercluster
    H = nx.erdos_renyi_graph(size_intercluster, p, seed=np.random, directed=False)
    mapping = dict(zip(H, headnodes_list))
    H = nx.relabel_nodes(H, mapping)
    edgelist = list(H.edges())
    G.add_edges_from(edgelist)
    A = nx.to_numpy_array(G)
    
    if plot_net:
        fig, ax = plt.subplots(dpi = 300) 
        ax.matshow(A, cmap=plt.cm.Blues)
        ax.set_ylabel(r'node index')
        ax.set_xlabel(r'node index')
    
    if save_net:
        if args['filename'] == None:
            filename = 'cluster_ncluster_N={}_ncluster={}'.format(N, args['num_of_cluster'])
        
        nx.write_edgelist(G, filename+".txt", data = False)
        plt.tight_layout()
        plt.savefig(filename+".pdf", format='pdf')
    
    return A    
    
def ER_net(random_seed = 1, N_nodes = 16,
           mean_degree = 4,
           filename = None,
           save_net = False, 
           plot_net = False):
    
    args = dict()
    args['random_seed'] = random_seed
    args['N_nodes'] =  N_nodes
    args['mean_degree'] =  mean_degree
    args['filename'] = filename
    
    np.random.seed(args['random_seed'])
    N = args['N_nodes']
    p = args['mean_degree']/N
    G = nx.erdos_renyi_graph(N, p, seed=np.random, directed=False)
    
    A = nx.to_numpy_array(G)
    
    if plot_net:
        fig, ax = plt.subplots(dpi = 300) 
        ax.matshow(A, cmap=plt.cm.Blues)
        ax.set_ylabel(r'node index')
        ax.set_xlabel(r'node index')
    
    if save_net:
        if args['filename'] == None:
            filename = 'ER_N={}_mean_degree={}_seed_{}'.format(N, 
                                                               args['mean_degree'],
                                                               args['random_seed'])
        
        nx.write_edgelist(G, filename+".txt", data = False)
        plt.tight_layout()
        plt.savefig(filename+".pdf", format='pdf')
         
def rich_club_net(random_seed, filename):
    rng = default_rng(random_seed)
    seed_vec = rng.choice(100, 2, replace = False)
    
    num_of_clusters = 30                   #Number of clusters
    N_nodes_cluster = 2                    #Number of nodes in each cluster
    N_nodes_integr_cluster = 30            #Number of nodes in the integrating cluster
    mean_degree = 3
    
    N = N_nodes_cluster*(num_of_clusters - 1) + N_nodes_integr_cluster                #Total number of nodes
    
    #probability_vector_connections = rng.random(num_of_clusters)
    probability_vector_connections = np.ones(num_of_clusters)*mean_degree/N_nodes_cluster
    probability_vector_connections = -np.sort(-probability_vector_connections)   

    G = nx.Graph()

    for i in range(num_of_clusters):
        p_i_connections = probability_vector_connections[i]
    
        if(i != 0):
            np.random.seed(seed_vec[0])
            G_i = nx.erdos_renyi_graph(N_nodes_cluster, p_i_connections, seed=np.random, directed=False)
            G = nx.disjoint_union(G, G_i)
        else:
            np.random.seed(seed_vec[1])
            G_i = nx.erdos_renyi_graph(N_nodes_integr_cluster, p_i_connections, seed=np.random, directed=False)
            nodes_integrating_cluster = list(G_i.nodes())
            G = nx.disjoint_union(G, G_i)
     
    p_connect_other_node = 0.10             # Probability of connection between node in the integrating cluster and other cluster
      
    for i in range(N_nodes_integr_cluster):        
        p_i_connections = rng.random(N - N_nodes_integr_cluster)
        counter = 0
        for j in range(N_nodes_integr_cluster, N):
            if(p_i_connections[counter] < p_connect_other_node):
                G.add_edge(i,j)             
            counter = counter + 1
    
    nx.write_edgelist(G, filename+"N_{}_p_{}.txt".format(N, p_connect_other_node), data=False)
    
def plot_net(G):
    
    fig =  plt.subplots(figsize = (10, 5))    
    pos = nx.spring_layout(G, iterations = 50)        
    nx.draw_networkx(G, pos = pos, node_size = 50, with_labels=False, font_weight='bold')    
    limits = plt.axis('off')
    plt.show()

def map_power_indices(params, pi_old, pi_new):
    
    N = pi_old.shape[1]
    N_new = pi_new.shape[1]
    
    if(N == N_new):
        deg_vector = pi_old.sum(axis = 1)
        mask = deg_vector > params['max_deg_monomials']
        if np.any(mask):
            index_array = np.arange(0, pi_old.shape[0], 1, dtype = int)
            pi_old = np.delete(pi_old, index_array[mask][0], axis = 0)
                    
        return params['orthnormfunc'], pi_old
    else:    
        deg_vector = pi_old.sum(axis = 1)
        mask = deg_vector > params['max_deg_monomials']
        index_array = np.arange(0, pi_old.shape[0], 1)
        pi_old = np.delete(pi_old, index_array[mask], axis = 0)
        
        pi_transf = np.zeros((pi_old.shape[0], pi_new.shape[1]), dtype = int)
        pi_transf[: , :pi_old.shape[1]] = pi_old
    
        params_orthnormfunc_new = dict()
        
        for l in range(pi_old.shape[0]):
                
            params_orthnormfunc_new[tuple(pi_transf[l, :])] = params['orthnormfunc'][tuple(pi_old[l, :])]
        
        return params_orthnormfunc_new, pi_transf

def delete_equal_rows(M_1, M_2):
        
    m_1 = M_1.shape[0]
    
    M_2_copy = np.copy(M_2)
    for m in range(m_1):
        m_2 = M_2_copy.shape[0]    
        index_array = np.arange(0, m_2, 1)
        
        mask = np.all(M_1[m, :] == M_2_copy, axis = 1)
        
        if(index_array[mask].shape[0] == 0):
            return M_2_copy
    
        M_2_copy = np.delete(M_2_copy, index_array[mask][0], axis = 0)
    
    return M_2_copy

def kernel_data(time_series_data):
    
    kernel = stats.gaussian_kde(time_series_data, bw_method = 5e-2)

    return kernel

def FP_FN_rel_error(node, Adj_row, hat_Adj_row, add_weight = False):
    id_vec = np.arange(Adj_row.shape[0], dtype = int)
    mask_node = np.isin(id_vec, [node])
    
    mask_true = Adj_row > 0    
    rel_error = dict()
    
    rel_error['FN'] = 1.0 - np.sum(hat_Adj_row[mask_true])/np.sum(mask_true)
    rel_error['FP'] = np.sum(hat_Adj_row[(~mask_true) & (~mask_node)])/np.sum((~mask_true) & (~mask_node))
    
    if add_weight:
        mask_est = hat_Adj_row > 0
        
        rel_error['FN'] = np.sum((~mask_est) & (mask_true))/np.sum(mask_true)
        
        mask_FP = (~mask_true) & (~mask_node)
        
        
        total = np.sum(hat_Adj_row[mask_FP]) + np.sum(~mask_est)
        
        rel_error['FP'] = np.sum(hat_Adj_row[mask_FP])/total
        
    
    return rel_error

def RSME(y_true, y_pred):
    
    return mean_squared_error(y_true, y_pred, squared=False)

#========================================================#
#Functions saving data
#========================================================#

def save_hdf5(filename, parameters, to_store=[], try_append=False):
    if try_append and os.path.isfile(filename):
        mode = "a"
        append = True
    else:
        mode = "w"
        append = False
    open = False
    while not open:
        try:
            out = h5py.File(filename, mode)
            open = True
        except OSError:
            pass
    if not append:
        par = out.create_group("parameters")
        for parameter in parameters:
            try:
                par.create_dataset(parameter, data=parameters[parameter])
            except:
                try:
                    print("Couldn't save parameter: "+parameter)
                except:
                    print("Couldn't save parameter: "+spy.srepr(parameter))

        realizations = out.create_group("realizations")
    else:
        realizations = out["realizations"]

    rlz = realizations.create_group("seed_{:}".format(parameters['random_seed']))
    for result in to_store:
        try:
            rlz.create_dataset(result, data=to_store[result])
        except:
            print("Couldn't save parameter: "+result)
    out.close()    
    
def load_hdf5(filename):
    data = dict()
    try:
        results = h5py.File(filename, "r")
        data["parameters"] = dict()
        for key in results["parameters"].keys():
            data["parameters"][key] = results["parameters"][key][()]
        for group in results["realizations"].keys():
            data[group] = dict()
            for key in results["realizations"][group].keys():
                data[group][key] = results["realizations"][group][key][()]
        results.close()
    except:
        print("Couldn't open file: "+filename)
            
    return data

def info_sim(filename, parameters, try_append = True):
    if try_append and os.path.isfile(filename):
        mode = "a"
        append = True
    else:
        mode = "w"
        append = False
    try:
        file = open(filename, mode)
        file.write("Error:_coupling_{:.2e}_seed_{}".format(parameters['alpha'], parameters['random_seed']))
        file.write("\n")
    except OSError:
        pass
    file.close()
 
class SympyDict(dict):
    """
    This saves a dictonary of sympy expressions to a file
    in human readable form.
    >>> a, b = sympy.symbols('a, b')
    >>> d = SympyDict({'a':a, 'b':b})
    >>> d.save('name.sympy')
    >>> del d
    >>> d2 = SympyDictd.load('name.sympy')
    """

    def __init__(self, *args, **kwargs):
        super(SympyDict, self).__init__(*args, **kwargs)

    def __repr__(self):
        d = dict(self)
        for key in d.keys():
            d[key] = spy.srepr(d[key])
        # regex is just used here to insert a new line after
        # each dict key, value pair to make it more readable
        return re.sub('(: \"[^"]*\",)', r'\1\n',  d.__repr__())

    def save(self, file):
        with open(file, 'w') as savefile:
            savefile.write(self.__repr__())

    @classmethod
    def load(cls, file_path):
        with open(file_path, 'r') as loadfile:
            d = loadfile.read()
            d = parse_expr(d)
        d = locals()['d']
        for key in d.keys():
            d[key] = spy.sympify(d[key])
        return d

        
