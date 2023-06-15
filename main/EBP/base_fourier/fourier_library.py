import os
import numpy as np 
from scipy.integrate import quad
import sympy as spy
import time

from ..tools import SympyDict


def sin_poly(order):
    '''
    Sin function to be used in the moment calculation

    Parameters
    ----------
    order : int
        degree of the trigonometric polynomial

    Returns
    -------
    sin_order : function

    '''
    sin_order = lambda x: np.sin(order*x)
    return sin_order
    

def cos_poly(order):
    '''
    Cos function to be used in the moment calculation

    Parameters
    ----------
    order : int
        degree of the trigonometric polynomial

    Returns
    -------
    cos_order : function

    '''
    cos_order = lambda x: np.cos(order*x)
    return cos_order
    

def spy_sin(x_t, exp_vector):
    '''
    
    Parameters
    ----------
    x_t : list
        DESCRIPTION.
    exp_vector : tuple
        DESCRIPTION.

    Returns
    -------
    homo_monomial : TYPE
        DESCRIPTION.

    '''
    
    inner_prod = x_t[0]*exp_vector[0]
    for id_deg in range(1, len(exp_vector)):
        inner_prod = inner_prod + x_t[id_deg]*exp_vector[id_deg]
        
    return spy.sin(inner_prod)
  
def spy_cos(x_t, exp_vector):
    '''
    
    Parameters
    ----------
    x_t : list
        DESCRIPTION.
    exp_vector : tuple
        DESCRIPTION.

    Returns
    -------
    homo_monomial : TYPE
        DESCRIPTION.

    '''
    
    inner_prod = x_t[0]*exp_vector[0]
    for id_deg in range(1, len(exp_vector)):
        inner_prod = inner_prod + x_t[id_deg]*exp_vector[id_deg]
        
    return spy.cos(inner_prod)
    

def l2_inner_product(func1, func2, params):
    """
    Calculate inner product between two functions func1 and func2;
    
    \langle f_1, f_2 \rangle = \int f_1 f_2 d\mu. 

    Parameters
    ----------
    func1 : function
        DESCRIPTION.
    func2 : function
        DESCRIPTION.
    params : dictionary
        DESCRIPTION.

    Returns
    -------
    inner product calculation
        DESCRIPTION.

    """
    density = params['density']
    a = params['lower_bound']
    b = params['upper_bound']
    norm_factor = params['density_normalization']
    integrand = lambda x: func1(x)*func2(x)*density(x)/norm_factor
    proj, err = quad(integrand, a, b, epsabs = 1e-8, epsrel = 1e-8)
        
    return proj

def generate_moments(params):
    """
    Calculate moments of the invariant density and saves into a dictionary.
    The maximum degree of the moment is given by a key, max_deg_generating,
    of params.

    Parameters
    ----------
    params : TYPE
        DESCRIPTION.

    Returns
    -------
    parameters : TYPE
        DESCRIPTION.

    """
    deg_array = params['deg_array']
    
    params['moments'] = dict()
        
    for deg_i in deg_array:
        params['moments']['sin_{}x'.format(deg_i)] = l2_inner_product(sin_poly(deg_i), 
                                                                     lambda x: 1, 
                                                                     params)       
        params['moments']['cos_{}x'.format(deg_i)] = l2_inner_product(cos_poly(deg_i), 
                                                                     lambda x: 1, 
                                                                     params)       
            
    
    for deg_i in deg_array:
        for deg_j in deg_array:
            params['moments']['sin_{}x_sin_{}x'.format(deg_i, deg_j)] = l2_inner_product(sin_poly(deg_i),
                                                                                       sin_poly(deg_j), 
                                                                                       params)       
            params['moments']['cos_{}x_cos_{}x'.format(deg_i, deg_j)] = l2_inner_product(cos_poly(deg_i), 
                                                                                       cos_poly(deg_j), 
                                                                                       params)       
            params['moments']['sin_{}x_cos_{}x'.format(deg_i, deg_j)] = l2_inner_product(sin_poly(deg_i), 
                                                                                       cos_poly(deg_j), 
                                                                                       params)       
    
    return params    


def fourier_dict(deg_array, include_sin = True, include_cos = True):
    '''
    

    Parameters
    ----------
    deg_array : TYPE
        DESCRIPTION.

    Returns
    -------
    fourier_bases_dict : TYPE
        DESCRIPTION.

    '''
    fourier_bases_dict = dict()
    
    l = 0
    fourier_bases_dict[l] = lambda x: 1
    
    l = 1
    for deg in deg_array:
        if include_sin:
            fourier_bases_dict[l] = sin_poly(deg) 
            l = l + 1
        if include_cos:   
            fourier_bases_dict[l + 1] = cos_poly(deg)
            l = l + 1
        if not (include_sin or include_cos):
            raise ValueError("include_sin and include_cos cannot both be False")    
        l = l + 2
        
    return fourier_bases_dict

def spy_fourier_dict(deg_array, include_sin = True, include_cos = True):
    '''
    Symboiic fourier basis. The independent term, e.g. 1, is always included. 

    Parameters
    ----------
    
    deg_array : numpy array
        Sequence of degree indices to be present in the Fourier expansion. 
        Example #1, sin x, cos x, sin 2x, cos 2x, sin 3x, cos 3x#
        deg_array = np.arange(1, 3, 1, dtype = int)
    
    include_sin : boolean 
        Boolean to determine if sine functions are included        
    include_cos : boolean 
        Boolean to determine if cosine functions are included
    Returns
    -------
    fourier_bases_dict : dict
        symbolic fourier basis.

    '''
    x = spy.symbols('x')
    fourier_bases_dict = dict()
    
    l = 0
    fourier_bases_dict[l] = 1
    
    l = 1
    for deg in deg_array:
        
        if include_sin:
            fourier_bases_dict[l] = spy_sin([x], [deg])
            l = l + 1
        if include_cos:
            fourier_bases_dict[l] = spy_cos([x], [deg])
            l = l + 1
        if not (include_sin or include_cos):
            raise ValueError("include_sin and include_cos cannot both be False")
        
    return fourier_bases_dict


def multiply(coefficient, func):
    return lambda x: coefficient*func(x)

def difference(func1, func2):
    return lambda x: func1(x) - func2(x)


def gramschmt_fourier(power_index, fourier_bases_dict, params):
    '''
    

    Parameters
    ----------
    power_index : int
        DESCRIPTION.
    fourier_bases_dict : dictionary which is kept fourier basis
        DESCRIPTION.
    params : dictionary 
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''   
    try:
        return params['orthnormfunc'][power_index]
    except:
        
        if power_index == 0:
            norm = np.sqrt(l2_inner_product(fourier_bases_dict[power_index], 
                                            fourier_bases_dict[power_index], 
                                            params))
            
            return lambda x: fourier_bases_dict[power_index](x)/norm
        
        else:
            temp_func = fourier_bases_dict[power_index]
            
            for pindex in range(1, power_index + 1):
                ref_func = gramschmt_fourier(pindex - 1, fourier_bases_dict, 
                                               params)
                
                proj_deg = l2_inner_product(fourier_bases_dict[power_index], 
                                            ref_func, 
                                            params)
                
                temp_func = difference(temp_func, multiply(proj_deg, ref_func))
                
            norm = np.sqrt(l2_inner_product(temp_func, temp_func, params))
    
            normed_func = lambda x: temp_func(x)/norm
                
            return normed_func

def multiply_spy(coefficient, func):
    '''
    

    Parameters
    ----------
    coefficient : TYPE
        DESCRIPTION.
    func : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    return coefficient*func

def difference_spy(func1, func2):
    '''
    

    Parameters
    ----------
    func1 : TYPE
        DESCRIPTION.
    func2 : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    return func1 - func2

def spy_gramschmt_fourier(power_index, fourier_bases_dict, params):
    '''
    Recursive method to orthonormalize a function corresponding to the 
    index power_index.

    Parameters
    ----------
    power_index : int
        Integer that identifies the basis function on the library to be orthonormalized.
    fourier_bases_dict : dict
        Symbolic fourier basis.
    params : dict
        
    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    
    x = spy.symbols('x')

    try:        #First attempt to access the orthonormal function calculated at
                #previous iterations of the recursive function spy_gram_schmidt.
        #print(params['orthnormfunc'])
        return params['orthnormfunc'][power_index]

    except:     
        
        #Calculate the first orthonormal function to be normalizing 
        #the constant function.
        if power_index == 0: 
            
            fb = spy.lambdify([x], fourier_bases_dict[power_index],
                     'numpy')
    
            norm = np.sqrt(l2_inner_product(fb, 
                                            fb, 
                                            params))
            
            return fourier_bases_dict[power_index]/norm
        
        #On the opposite side, calculate other orthonormal function 
        #identified by exp_array.    
        else:
            temp_func = fourier_bases_dict[power_index]
           
            for id_key in range(1, power_index + 1):
                try:
                    ref_func = params['orthnormfunc'][id_key - 1]
                except:
                    ref_func = spy_gramschmt_fourier(id_key - 1, fourier_bases_dict, params)
                
                fb_expr = spy.lambdify([x], fourier_bases_dict[power_index], 'numpy')
                
                fb_ref_func = spy.lambdify([x], ref_func, 'numpy')
                
                I = l2_inner_product(fb_expr, fb_ref_func, params)
                
                temp_func = difference_spy(temp_func, multiply_spy(I, ref_func))
    
            fb_temp_func = spy.lambdify([x], temp_func, 'numpy')
                
            I = l2_inner_product(fb_temp_func, fb_temp_func, params)
            norm = spy.sqrt(I)
    
            normed_func = temp_func/norm
            
            return normed_func

def fourier_orthonormal(params):
    '''
    

    Parameters
    ----------
    params : TYPE
        DESCRIPTION.

    Returns
    -------
    fourier_orthnorm_dict : TYPE
        DESCRIPTION.

    '''
    deg_array = params['deg_array']
    fourier_bases_dict = fourier_dict(deg_array)
    
    fourier_orthnorm_dict = dict()
    
    params['orthnormfunc'] = dict()
    
    for keys in fourier_bases_dict.keys():
        start_time = time.time()
        fourier_orthnorm_dict[keys] = gramschmt_fourier(keys, 
                                                        fourier_bases_dict, 
                                                        params)
           
        params['orthnormfunc'][keys] = fourier_orthnorm_dict[keys]
        
        end_time = time.time()

        print(keys, (end_time - start_time)/60, '\n')
        
    return fourier_orthnorm_dict


def spy_fourier_orthonormal(params):
    '''
    Construct the symbolic orthornomal basis functions

    Parameters
    ----------
    params : dict

    Returns
    -------
    fourier_orthnorm_dict : TYPE
        DESCRIPTION.

    '''
    deg_array = params['deg_array']
    fourier_bases_dict = spy_fourier_dict(deg_array)
    
    fourier_orthnorm_dict = dict()
    
    params['orthnormfunc'] = dict()
    
    for keys in fourier_bases_dict.keys():
        start_time = time.time()
        fourier_orthnorm_dict[keys] = spy_gramschmt_fourier(keys, 
                                                        fourier_bases_dict, 
                                                        params)
           
        params['orthnormfunc'][keys] = fourier_orthnorm_dict[keys]
        
        end_time = time.time()

        print(keys, (end_time - start_time)/60, '\n')
        
    return fourier_orthnorm_dict

def get_coeff_matrix_wrt_basis(sym_expr, dict_basis_functions):
    '''
    To obtain the coefficients with respect to the given basis in the dictionary.

    Parameters
    ----------
    sym_expr : sympy expression
        expression to be identified in the dict_basis_functions.
    dict_basis_functions : dict
        basis functions to span the sym_expr.

    Returns
    -------
    numpy array
        coefficient vector relative to the basis functions

    '''
    #Create an empty list to include all coefficient entries
    coefficient_vector = []

    #In case the sympy expression is the very first element of the GS process
    if isinstance(sym_expr, float):
        return sym_expr
    
    else:
        #Obtain the independent term of the sympy expression    
        coefficient_vector.append(sym_expr.as_coeff_Add()[0])
        
        #Run over the basis functions to obtain the coefficients
        for keys in list(dict_basis_functions.keys())[1 :]:
            
            if sym_expr.has(dict_basis_functions[keys]):
                poly_expr = spy.Poly(sym_expr)
                
                coefficient_vector.append(poly_expr.coeff_monomial(dict_basis_functions[keys]))
            else:
                coefficient_vector.append(0)
                
        return np.array(coefficient_vector, dtype = float)


def get_matrix_change_of_basis(orthnorm_base, base):
    '''
    Construct the matrix of change of basis.

    Parameters
    ----------
    orthnorm_base : dict
        orthonormal basis functions .
    base : dict
        fourier basis functions.

    Returns
    -------
    R : numpy array
        Matrix of change of basis.

    '''
    m = len(orthnorm_base.keys())
    R = np.zeros((m, m))
    
    for key in orthnorm_base.keys():
        if key == 0:
            R[key, key] = get_coeff_matrix_wrt_basis(orthnorm_base[key], base) 
        else:
            R[:, key] = get_coeff_matrix_wrt_basis(orthnorm_base[key], base)

    return R

def get_net_coeff(N_orth_base, N_base):
    '''
    Construct the matrix of change of basis taking the number of vertices into
    account.

    Parameters
    ----------
    N_orth_base : dict
        Orthonormal basis functions taking number of vertices into account.
    N_base : dict
        Fourier basis functions taking number of vertices into account.

    Returns
    -------
    R : TYPE
        DESCRIPTION.

    '''
    R = get_matrix_change_of_basis(N_orth_base, N_base)
    
    return R


def get_N_base(orth_base, base, params):
    '''
    Construct the libraries taking the number of vertices into account.

    Parameters
    ----------
    orth_base : dict
        orthonormal basis functions only depending on degree indices.
    base : dict
        fourier basis functions only depending on degree indices.
    params : dict
        
    Returns
    -------
    N_base : dict
        fourier basis functions taking degree indices and number of vertices
        into account.
    N_orth_base : dict
        orthonormal basis function taking degree indices and number of vertices
        into account.

    '''
    N = params['number_of_vertices']
   
    x = spy.symbols('x')
    x_t = [spy.symbols('x_{}'.format(j)) for j in range(0, N)]

    N_base = dict()
    N_orth_base = dict()
    
    N_base[0] = float(base[0])
    N_orth_base[0] = float(orth_base[0])
    l = 1
    
    for key in range(1, len(base.keys())):
        fb_spy = spy.lambdify([x], base[key], 'sympy')
        orth_fb_spy = spy.lambdify([x], orth_base[key], 'sympy')
        
        for id_node in range(N):
            N_base[l] = fb_spy(x_t[id_node])
            N_orth_base[l] = orth_fb_spy(x_t[id_node])
            
            l = l + 1            
        
    return N_base, N_orth_base        

def power_indices(params):

    params_ = params.copy()
    
    N = params_['number_of_vertices']
    L = params_['L']

    params_['power_indices'] = np.zeros((L, N))
    params_['power_indices'][0, :] = np.zeros(N)
    
    id_index = np.arange(1, N + 1, dtype = int)
    
    for l in range(1, params['max_deg_harmonics'] + 1):
        
        params_['power_indices'][int(2*(l - 1)*N) + id_index, :] = np.identity(N)*l
        params_['power_indices'][int((2*l - 1)*N) + id_index, :] = np.identity(N)*l
        
    return params_
            
def library_matrix(X_t, params):
    '''
    Transform data X_t using the basis functions in basis_dict

    Parameters
    ----------
    X_t : numpy array
        Multivariate time series.
    basis_dict : dict
        Fourier basis functions to be transformed into the library matrix.
    params : dict
        
    Returns
    -------
    PHI : numpy array
        Library matrix relative to the fourier basis function in basis_dict.

    '''
    N = params['number_of_vertices']
    M = params['length_of_time_series']
    L = params.get('L')
    
    params = power_indices(params)
    
    canonical_basis = params.get('use_canonical', False)
    orthonormal_basis = params.get('use_orthonormal', True)
    
    if (canonical_basis) & (orthonormal_basis):
        raise ValueError("Incompatible choice: Choose a single set of basis functions.")    
        return
    
    params = power_indices(params)
    
    if params['use_canonical']:
        basis_dict = spy_fourier_dict(params['deg_array'])
    
    if params['use_orthonormal']:
        if not os.path.isfile(params['orthnorm_func_filename']):
            params['orthnormfunc'] = spy_fourier_orthonormal(params)
        
        #In case of saving the orthornormal functions
        if params['save_orthnormfunc'] and not os.path.isfile(params['orthnorm_func_filename']):
            d = SympyDict(params['orthnormfunc'])
            d.save(params['orthnorm_func_filename']) 
        
        spy_fourier_base = spy_fourier_dict(params['deg_array'])
        N_base, N_orth_base = get_N_base(params['orthnormfunc'], spy_fourier_base, params)
        params['R'] = get_net_coeff(N_orth_base, N_base)
        basis_dict = params['orthnormfunc']
    PHI = np.zeros((M, L))
    
    for key in basis_dict.keys():
        start_time = time.time()
        
        x = spy.symbols('x')
    
        f = spy.lambdify([x], basis_dict[key],
                         'numpy')
        
        id_index = np.arange(1, N + 1, dtype = int)
        
        if key == 0:
            PHI[:, key] = f(X_t)/np.sqrt(M)
        else:
            PHI[:, int((key - 1)*N) + id_index] = f(X_t)/np.sqrt(M)
        
        end_time = time.time()
        print(key, (end_time - start_time)/60, '\n')
  
    return PHI, params


























