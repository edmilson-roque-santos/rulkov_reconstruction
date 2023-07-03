import itertools

import numpy as np
from numpy import linalg as LA

import scipy.special
from scipy.integrate import quad
import sympy as spy
#import time

from tqdm import tqdm

from ..tools import SympyDict

tqdm_par = {
"unit": "it",
"unit_scale": 1,
"ncols": 80,
"bar_format": "[Orthonormalizing progress] {percentage:2.0f}% |{bar}| {n:.1f}/{total:.1f}  [{rate_fmt}, {remaining_s:.1f}s rem]",
"smoothing": 0
}

tqdm_par_red = tqdm_par.copy()
tqdm_par_red['bar_format'] = "[Orthonormalizing from reduced progress] {percentage:2.0f}% |{bar}| {n:.1f}/{total:.1f}  [{rate_fmt}, {remaining_s:.1f}s rem]"
    
def poly(order):
    '''
    Polynomial function to be used in the moment calculation

    Parameters
    ----------
    order : int
        degree of the polynomial

    Returns
    -------
    poly_order : function

    '''
    poly_order = lambda x: x**order
    return poly_order
    
def polynomial_exp(X_t, exp_vector):
    '''
    
    Parameters
    ----------
    X_t : numpy array
        Multivariate time series.
    exp_vector : numpy array - size: (X_t.shape[0],)
        Exponent vector to be applied the multivariate monomial.

    Returns
    -------
    multivariate monomial: X_t**exp_vector.

    '''
    homo_monomial = X_t[:, 0]**(exp_vector[0])
    for id_deg in range(1, len(exp_vector)):
        homo_monomial = homo_monomial*(X_t[:, id_deg]**(exp_vector[id_deg]))
        
    return homo_monomial 

def sympy_polynomial_exp(x_t, exp_vector):
    '''
    Monomial term with corresponding exponent vector given by exp_vector.

    Parameters
    ----------
    x_t : list
        list of sympy exprssions denoting variables of the multivariate 
        polynomial.
    exp_vector : tuple
        Exponent vector of the monomial in the multivariate polynomial.

    Returns
    -------
    homo_monomial : sympy expression
        Correspoding monomial with form  x_t**exp_vec.

    '''
    homo_monomial = x_t[0]**(exp_vector[0])
    for id_deg in range(1, len(exp_vector)):
        homo_monomial = homo_monomial*(x_t[id_deg]**(exp_vector[id_deg]))
        
    return homo_monomial 


#=======================================##=======================================#          

def logistic_inv_measure(x):
    """
    Invariant measure for Logistic map f(x) = 4 x(1 - x).

    Parameters
    ----------
    x : ndarray
        The points where invariant measure is evaluated.
        
    Returns
    -------
    logistic_inv_measure : ndarray
        The values at each point.
        
    """

    logistic_inv_measure = 1.0/(np.sqrt(x*(1 - x))*np.pi)
    
    return logistic_inv_measure

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
       
    proj, err = quad(integrand, a, b, epsabs = 1e-12, epsrel = 1e-12)
    
    return np.around(proj, 8)

def generate_moments(params, single_density = True):
    '''
    Calculate moments of the invariant density and saves into a dictionary.
    The maximum degree of the moment is given by a key, max_deg_generating,
    of params.

    Parameters
    ----------
    params : dict
                
    single_density: boolean
     The default is True.
    
    Returns
    -------
    parameters : dict
        Updated parameters dictionary to be used throughout the simulation.
        Update: Each variable has the calculated moment to be used.
        
    '''
    
    N = params['number_of_vertices']
    order = params['max_deg_generating']
    x_t = [spy.symbols('x_{}'.format(j)) for j in range(0, N)]
    
    if single_density:
        stop = False
        for id_node in range(N):
            params[x_t[id_node]]['moments'] = dict()
            if not stop:
                for deg in range(order + 1):
                    params[x_t[id_node]]['moments']['m_{}'.format(deg)] = l2_inner_product(poly(deg),
                                                                             lambda x: 1,
                                                                             params[x_t[id_node]])       
                stop = True
            else:
                for deg in range(order + 1):
                    params[x_t[id_node]]['moments']['m_{}'.format(deg)] = params[x_t[0]]['moments']['m_{}'.format(deg)]
    
    if not single_density:
        for id_node in range(N):
            params[x_t[id_node]]['moments'] = dict()
            for deg in range(order + 1):
                params[x_t[id_node]]['moments']['m_{}'.format(deg)] = l2_inner_product(poly(deg),
                                                                         lambda x: 1,
                                                                         params[x_t[id_node]])       
                
    return params    



def multiply(coefficient, func):
    return lambda x: coefficient*func(x)

def difference(func1, func2):
    return lambda x: func1(x) - func2(x)

def orthogonal(order, parameters):
    
    density, a, b = parameters['density'], parameters['lower_bound'], parameters['upper_bound']
    
    if(order == 0):
        return poly(0)
    
    else:
        temp_func = lambda x: poly(order)(x)
        
        for deg in range(1, order + 1):
            ref_func = orthogonal(deg - 1, density, a, b)
            proj_deg = l2_inner_product(poly(order), ref_func, parameters)
            temp_func = difference(temp_func, multiply(proj_deg, ref_func))
            
        return temp_func

def orthonormal(order, parameters):
       
    #density, a, b = parameters['density'], parameters['lower_bound'], parameters['upper_bound']
    
    if(order == 0):
        norm = np.sqrt(l2_inner_product(poly(0), poly(0), parameters))
        
        return lambda x: poly(0)(x)/norm
    
    else:
        temp_func = lambda x: poly(order)(x)
        
        for deg in range(1, order + 1):
            ref_func = orthonormal(deg - 1, parameters)
            proj_deg = l2_inner_product(poly(order), ref_func, parameters)
            temp_func = difference(temp_func, multiply(proj_deg, ref_func))
            
        norm = np.sqrt(l2_inner_product(temp_func, temp_func, parameters))

        normed_func = lambda x: temp_func(x)/norm
            
        return normed_func

#=======================================##=======================================#

#=======================================##=======================================#
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


def term_integration(expr, params):
    
    integr = float(0.0)       #Start the integral value as zero.
        
    #If expr is simply a constant, simply sum to the integral's value.
    if(expr.is_constant()):
        integr = expr
        return float(integr)
    
    #Otherwise, break expression in terms to calculate separately the contribution
    #for each of them in the product.        
    else:
        set_symb_expr = list(expr.free_symbols)
        num_of_symb = len(set_symb_expr)
        tuple_a_x_b = expr.leadterm(set_symb_expr[0])
        
        expr_temp = params[set_symb_expr[0]]['moments']['m_{}'.format(tuple_a_x_b[1])]*tuple_a_x_b[0]

        for symb_expr in range(1, num_of_symb + 1):
            expr_temp = expr_temp.expand()
            if not expr_temp.is_constant():
                tuple_a_x_b = expr_temp.leadterm(set_symb_expr[symb_expr])
                expr_temp = params[set_symb_expr[symb_expr]]['moments']['m_{}'.format(tuple_a_x_b[1])]*tuple_a_x_b[0]
            if expr_temp.is_constant():
                integr = expr_temp
        return float(integr)
            

def fubini_integration_BF(expr_GS, num_of_args, params):
    '''
    Brute-force Fubini integrate an expression expr_GS_exp with num_of_args

    Parameters
    ----------
    expr_GS_exp : TYPE
        DESCRIPTION.
    num_of_args : TYPE
        DESCRIPTION.
    params : TYPE
        DESCRIPTION.

    Returns
    -------
    I : TYPE
        DESCRIPTION.

    '''
    #If expr_GS_args has a unique term.
    if num_of_args < 2:
        return term_integration(expr_GS, params)
        
    #If expr_GS_args has more than one term.
    else:
        expr_args = expr_GS.args
        integral = 0
        
        for id_args in range(len(expr_args)):
            integral = integral + term_integration(expr_args[id_args], params)
        
        return float(integral)
    
def spy_l2_inner_product(expr_1, expr_2, params):
    '''
    

    Parameters
    ----------
    expr_1 : TYPE
        DESCRIPTION.
    expr_2 : TYPE
        DESCRIPTION.
    params : TYPE
        DESCRIPTION.

    Returns
    -------
    I : TYPE
        DESCRIPTION.

    '''
    expr = expr_1*expr_2
    expr = expr.expand()
    poly_expr = spy.poly(expr)
    num_of_args = len(poly_expr.monoms())#len(list(expr.args))
    
    integral = fubini_integration_BF(expr, num_of_args, params)
    
    return float(integral)

def spy_gram_schmidt(exp_array, params_, power_indices):
    '''
    Recursive method to orthonormalize a function corresponding to the exponent
    vector exp_array.

    Parameters
    ----------
    exp_array : tuple
        Exponential vector corresponding to the current step function to be 
        orthonormalized.
    params : dict

    power_indices : numpy array
        Exponent vectors correspondingly to each basis function in the library.

    Returns
    -------
    sympy expression 
        Orthonormal function with respect to the previous functions in the basis.

    '''
    
    N = params_['number_of_vertices']
    threshold_proj = 1e-8       #Determine the threshold under which the 
                                #inner product is set to zero.
    params = params_.copy()
                            
    x_t = [spy.symbols('x_{}'.format(j)) for j in range(0, N)]

    try:        #First attempt to access the orthonormal function calculated at
                #previous iterations of the recursive function spy_gram_schmidt.
        return params['orthnormfunc'][exp_array], params

    except:     
        
        #For the one-dimensional case
        if params['use_integral_1d']:
            #If there is no key "orthonormfunc", it is created one.
            if not 'orthnormfunc' in params:
                params['orthnormfunc'] = dict()
            
            use_path = params.get('use_path', True)
            zero_tuple = tuple(np.zeros(N, dtype = int))
            
            #Calculate the first orthonormal function to be normalizing 
            #the constant function.
            if(exp_array == zero_tuple): 
                prod_moments = 1
                for id_node in range(N):
                    prod_moments = prod_moments*params[x_t[id_node]]['moments']['m_0']
                norm = np.sqrt(prod_moments)
                phi_0 = 1/norm 
                params['orthnormfunc'][zero_tuple] = phi_0
                
                if not use_path:    
                    return phi_0
                
                if use_path:    
                    return phi_0, params
                
            #On the opposite side, calculate other orthonormal function 
            #identified by exp_array.    
            else:
                temp_func = sympy_polynomial_exp(x_t, exp_array)
                
                #power_indices has an intrinsic ordering. Hence, we create mask 
                # to identify which index exp_array corresponds to.
                mask_id_vec = power_indices == exp_array
                mask_id = np.all(mask_id_vec, axis = 1)
                id_keys = np.argwhere(mask_id)[0][0]
                
                for id_key in range(1, id_keys + 1):
                    try:
                        ref_func = params['orthnormfunc'][tuple(power_indices[id_key - 1, :])]
                    except:
                        if not use_path:    
                            ref_func = spy_gram_schmidt(tuple(power_indices[id_key - 1, :]), params, power_indices)
                        if use_path:    
                            ref_func, params = spy_gram_schmidt(tuple(power_indices[id_key - 1, :]), params, power_indices)
                            
                    expr = sympy_polynomial_exp(x_t, exp_array)
                    
                    integral = spy_l2_inner_product(expr, ref_func, params)
                    
                    #Cut-off any inner product below threshold_proj
                    if(np.abs(integral) < threshold_proj):
                        integral = float(0.0)
                        
                    temp_func = difference_spy(temp_func, multiply_spy(integral, ref_func))
                    
                integral = spy_l2_inner_product(temp_func, temp_func, params)
                
                norm = spy.sqrt(integral)
                norm = float(norm)
                normed_func = temp_func/norm
                params['orthnormfunc'][exp_array] = normed_func
                if not use_path:    
                    return normed_func   
                if use_path:
                    return normed_func, params 

#=======================================##=======================================#

# Function to find out all combinations of positive numbers  
# that add upto given number. It uses findCombinationsUtil()  
def findCombinations(n): 
    seq_yields_sum = []
    # array to store the combinations 
    # It can contain max n elements 
    arr = [0] * n; 
  
    # find all combinations 
    findCombinationsUtil(arr, 0, n, n, seq_yields_sum);
    return seq_yields_sum

def findCombinationsUtil(arr, index, num, 
                              reducedNum, seq_yields_sum): 
    '''
    # arr - array to store the combination 
    # index - next location in array 
    # num - given number 
    # reducedNum - reduced number  
    '''
    # Base condition 
    if (reducedNum < 0): 
        return; 
  
    # If combination is  
    # found, print it 
    if (reducedNum == 0): 
        seq = []
        for i in range(index): 
            seq.append(arr[i])
        seq_yields_sum.append(seq)
        
        return; 
  
    # Find the previous number stored in arr[].  
    # It helps in maintaining increasing order 
    prev = 1 if(index == 0) else arr[index - 1]; 
  
    # note loop starts from previous  
    # number i.e. at array location 
    # index - 1 
    for k in range(prev, num + 1): 
          
        # next element of array is k 
        arr[index] = k; 
  
        # call recursively with 
        # reduced number 
        findCombinationsUtil(arr, index + 1, num,  
                                 reducedNum - k, seq_yields_sum); 

def lexico_permute(s):
    ''' Generate all permutations in lexicographic order of string `s`

        This algorithm, due to Narayana Pandita, is from
        https://en.wikipedia.org/wiki/Permutation#Generation_in_lexicographic_order
        
        To produce the next permutation in lexicographic order of sequence `a`

        1. Find the largest index j such that a[j] < a[j + 1]. If no such index exists, 
        the permutation is the last permutation.
        2. Find the largest index k greater than j such that a[j] < a[k].
        3. Swap the value of a[j] with that of a[k].
        4. Reverse the sequence from a[j + 1] up to and including the final element a[n].
        
        In particular, this version is adapted from PM 2Ring
        https://stackoverflow.com/questions/8306654/finding-all-possible-permutations-of-a-given-string-in-python
    '''

    a = sorted(s)
    n = len(a) - 1
    while True:
        yield list(a)

        #1. Find the largest index j such that a[j] < a[j + 1]
        for j in range(n-1, -1, -1):
            if a[j] < a[j + 1]:
                break
        else:
            return

        #2. Find the largest index k greater than j such that a[j] < a[k]
        v = a[j]
        for k in range(n, j, -1):
            if v < a[k]:
                break

        #3. Swap the value of a[j] with that of a[k].
        a[j], a[k] = a[k], a[j]

        #4. Reverse the tail of the sequence
        a[j+1:] = a[j+1:][::-1]


def f_exp_vector(degree, num_var = 2):
    '''
    Create list of exponents vectors with size equals degree which the sum equals
    degree as well.

    Parameters
    ----------
    num_var : int
        Number of variables in the multivariate polynomial basis functions.
    degree : int
        Maximum degree of the multivariate polynomial basis functions.

    Returns
    -------
    exp_initial_ : list
        List of exponents vectors (each array is a list).

    '''
    
    #List of all combinations of exponents that add up to degree deg
    seq_exp_sum_d = findCombinations(degree)
    
    #Filter exponent vectors which have more than num_var entries        
    exp_initial = list(filter(lambda x: len(x) == num_var, seq_exp_sum_d))
    
    exp_initial_ = []
    for id_exp_initial in range(len(exp_initial)):
        #Generate exponent vectors using permutations of exp_initial
        #But it removes duplicates.
        for exp_vec in lexico_permute(exp_initial[id_exp_initial]):
            exp_initial_.append(exp_vec)
    exp_initial_.sort()
    
    return exp_initial_

def reduced_index_iteration(params):
    '''
    Generates array with size (L, number_of_vertices) containing 
    some exponent vectors with respect to the basis functions,
    
    where  
    if 'expansion_crossed_terms' == True:
        L = number of combinations of exponents that add up to degree deg

        reduced_index_iteration considers only a subset of possible permutations 
        (excluding duplicates) when compared to index_iteration. 
    
    if 'expansion_crossed_terms' == False:
        L = max_deg_monomials + 1
    
    Parameters
    ----------
    params : dict
        Core dictionary in the package. In this method it uses the 'number_of_vertices'
        and 'max_deg_monomials'

    Returns
    -------
    numpy array - size: (num_of_basis_functions, number_of_vertices); dtype - int
        Power_indices - the exponent vectors correspondingly to each
        basis function in the library.

    '''
    N = params['number_of_vertices']
    order = params['max_deg_monomials']

    power_indices = []
    power_indices.append(np.asarray(tuple(np.zeros(N, dtype = int))))
    
    #If the expansion has no crossed terms    
    if not params['expansion_crossed_terms']:
        for deg in range(1, order + 1):
            exp_vec = np.zeros(N, dtype = int)
            exp_vec[0] = deg
            power_indices.append(exp_vec)
        
    #Otherwise, permutations must be taken into account
    if params['expansion_crossed_terms']:            
        
        for num_var in range(1, 3):
            for deg in range(1, order + 1):     
                exp_final = f_exp_vector(deg, num_var)
            
                for id_exp in range(len(exp_final)):
                    exp_final_ = np.asarray(exp_final[id_exp]+[0]*(N - len(exp_final[id_exp])))
                    power_indices.append(exp_final_)
  
    return np.array(power_indices, dtype = int)


def index_iteration(params):
    '''
    Generates array with size (L, number_of_vertices) containing 
    all exponent vectors with respect to the basis functions,
    
    where 
    if 'expansion_crossed_terms' == True:
        L = comb(number_of_vertices + max_deg_monomials, max_deg_monomials). 
        index_iteration considers all possible permutations (excluding duplicates) 
        of these exponent vectors. 
    
    if 'expansion_crossed_terms' == False:
        L = number_of_vertices*max_deg_monomials + 1
    
    
    Parameters
    ----------
    params : dict
        Core dictionary in the package. In this method it uses the 'number_of_vertices'
        and 'max_deg_monomials'

    Returns
    -------
    numpy array - size: (num_of_basis_functions, number_of_vertices); dtype - int
        Power_indices - the exponent vectors correspondingly to each
        basis function in the library.

    '''
    N = params['number_of_vertices']
    order = params['max_deg_monomials']
    index_vec = np.arange(0, N, 1, dtype = int)
    
    power_indices = []
    power_indices.append(np.asarray(tuple(np.zeros(N, dtype = int))))
    
    #If the expansion has no crossed terms    
    if not params['expansion_crossed_terms']:
        l = 1
        
        for id_node in range(N):
            for deg in range(1, order + 1):
                exp_vec = np.zeros(N, dtype = int)
                exp_vec[id_node] = deg
                
                power_indices.append(exp_vec)
                l = l + 1   

    #Otherwise, permutations must be taken into account
    if params['expansion_crossed_terms']:
        l = 1
        #Generate power indices. 
        
        #Isolated functions
        for id_node in range(N):
            for deg in range(1, order + 1):
                exp_vec = np.zeros(N, dtype = int)
                exp_vec[id_node] = deg
                
                power_indices.append(exp_vec)
                l = l + 1 
        #Pairwise functions
        #For each pair of nodes organized in lexicographic order
        num_var = 2
        for exp_vec in itertools.combinations(index_vec, num_var):
        
            #Fix a degree of the monomial
            for deg in range(2, order + 1):     
                #Take all combinations of indices that summed yield the degree deg
                exp_final = f_exp_vector(deg, num_var)
             
                for id_exp in range(len(exp_final)):
                               
                    exp_array = np.zeros(N, dtype = int)
                    loc_exp = np.asarray(exp_vec, dtype = int)
                    
                    #Localize the index and put corresponded degrees
                    exp_array[loc_exp] = exp_final[id_exp]
                    #Power indices is constructed.
                    power_indices.append(np.asarray(exp_array))
                    l = l + 1
              
    return np.array(power_indices, dtype = int)     


def dict_canonical_basis(params):
    '''
    Create a dictionary containing the multivariate monomial functions
    according to the 'power indices' in params dictionary.
    
    Parameters
    ----------
    params : dict

    Returns
    -------
    dict_basis_functions : dict
        For each tuple identifying the exponent vector there is a corresponding
        sympy expression representing the multivariate monomial function.

    '''
    
    N = params['number_of_vertices']
    power_indices = params['power_indices']
    x_t = [spy.symbols('x_{}'.format(j)) for j in range(0, N)]

    dict_basis_functions = dict()
    
    for j in range(power_indices.shape[0]):
        dict_basis_functions[tuple(power_indices[j, :])] = sympy_polynomial_exp(x_t, power_indices[j, :])
        
    return dict_basis_functions

def symbolic_canonical(params):
    
    N = params['number_of_vertices']
    power_indices = params['power_indices']
    x_t = [spy.symbols('x_{}'.format(j)) for j in range(0, N)]

    PHI = []
    
    for j in range(power_indices.shape[0]):
        PHI.append(sympy_polynomial_exp(x_t, power_indices[j, :]))
        
    params['symbolic_PHI'] = PHI 
    
    return params

def get_coeff_matrix_wrt_basis(sym_expr, dict_basis_functions):
    '''
    To obtain the coefficients with respect to the given basis in the dictionary

    Parameters
    ----------
    sym_expr : Sympy expression
        
    dict_basis_functions : dictionary

    Returns
    -------
    numpy array - size: (num_of_basis_functions, )
        Coefficient vector which represents the sympy expression with respect to 
        the basis in the dictionary.

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

def poly_orthonormal_PHI(X_t, params, power_indices):
    '''
    Create the library matrix PHI for the multivariate time series X_t

    Parameters
    ----------
    X_t : numpy array - size: (lenght of time series, number of vertices)
        Multivariate time series to be plugged into the library matrix.
    params : dict
        
    power_indices : numpy array - size: (num_of_basis_functions, number_of_vertices); dtype - int
        The exponent vectors correspondingly to each
        basis function in the library.

    Returns
    -------
    PHI : numpy array - size: (length of time series, num_of_basis_functions)
        Library matrix
    params : dict
        Update dictionary to be used along the simulation.
        The update arguments are a symbolic representation of the functions, symbolic_PHI,
        and matrix of change of basis, R. 

    '''
    N = params['number_of_vertices']
    M = params['length_of_time_series']
    L = params['L']
    
    PHI = np.zeros((M, L))
    R = np.zeros((L, L))

    x_t = [spy.symbols('x_{}'.format(j)) for j in range(0, N)]
    
    #Calculate the first term of orthonormal function
    func, params = spy_gram_schmidt(tuple(power_indices[0, :]), params, 
                                   power_indices)
    
    #Create dictionary with the polynomial basis
    dict_canonical_basis_functions = dict_canonical_basis(params)
    
    #Calculate the first element of the matrix R (from QR decomposition)
    R[0, 0] = get_coeff_matrix_wrt_basis(float(func), dict_canonical_basis_functions)
    
    #Create a expression to be evaluated at data X_t
    f = spy.lambdify([x_t], params['orthnormfunc'][tuple(power_indices[0, :])],
                     'numpy')
    
    #Evaluate expression f at X_t as first column of library matrix
    PHI[:, 0] = f(X_t.T)/np.sqrt(M)
    
    #Create list of symbolic version of the orthonormal basis
    symbolic_PHI = []
    sym_f = spy.lambdify([tuple(x_t)], 
                         params['orthnormfunc'][tuple(power_indices[0, :])],
                         'sympy')
    
    symbolic_PHI.append(sym_f(tuple(x_t)))
    
    for l in tqdm(range(1, power_indices.shape[0]), **tqdm_par):
        #start_time = time.time()

        func, params = spy_gram_schmidt(tuple(power_indices[l, :]), params,
                                       power_indices)
        
        f = spy.lambdify([x_t], params['orthnormfunc'][tuple(power_indices[l, :])],
                         'numpy')
        
        PHI[:, l] = f(X_t.T)/np.sqrt(M)


        sym_f = spy.lambdify([tuple(x_t)], 
                         params['orthnormfunc'][tuple(power_indices[l, :])],
                         'sympy')
        
        symbolic_PHI.append(sym_f(tuple(x_t)))
        R[:, l] = get_coeff_matrix_wrt_basis(sym_f(tuple(x_t)), 
                                                       dict_canonical_basis_functions)
        
        #end_time = time.time()
        #print(l, (end_time - start_time)/60, '\n')
    #Adapt params dictionary to include the symbolic version of orthonormal functions
    params['symbolic_PHI'] = symbolic_PHI
    
    #Adapt params dictionary to include matrix R
    params['R'] = R
    
    return PHI, params

def reduced_poly_orthonormal_PHI(X_t, params, reduced_pindex):
    '''
    
    Parameters
    ----------
    X_t : numpy array
        Multivariate time series.
    params : dict
        
    reduced_pindex : numpy array
        Reduced number of exponent vectors to be used to construct the orthonormal
        functions. The result from the method: reduced_index_iteration(params).

    Returns
    -------
    PHI : numpy array: size (length of time series, L)
        Library matrix
    params : dict
        Updated parameters dictionary to be used throughout the simulation.
        Update: matrix of change of basis R and symbolic representation of the 

    '''
    N = params['number_of_vertices']
    M = params['length_of_time_series']
    L = params['L']
    order = params['max_deg_monomials']
    
    #Create a index vector of N entries
    index_vec = np.arange(N, dtype = int)

    
    PHI = np.zeros((M, L))
    R = np.zeros((L, L))
    
    x_t = [spy.symbols('x_{}'.format(j)) for j in range(0, N)]
    
    func, params = spy_gram_schmidt(tuple(reduced_pindex[0, :]), params,
                                    reduced_pindex)
    
    #Create dictionary with the polynomial basis
    dict_canonical_basis_functions = dict_canonical_basis(params)
    
    #Calculate the first element of the matrix R (from QR decomposition)
    R[0, 0] = get_coeff_matrix_wrt_basis(float(func), dict_canonical_basis_functions)
    
    f = spy.lambdify([x_t], params['orthnormfunc'][tuple(reduced_pindex[0, :])],
                     'numpy')
    
    PHI[:, 0] = f(X_t.T)/np.sqrt(M)
    
    #Create list of symbolic version of the orthonormal basis
    symbolic_PHI = []
    sym_f = spy.lambdify([tuple(x_t)], 
                         params['orthnormfunc'][tuple(reduced_pindex[0, :])],
                         'sympy')
    
    symbolic_PHI.append(sym_f(tuple(x_t)))
    
    l = 1
    
    #start_time = time.time()
    for num_var in tqdm(range(1, 3),**tqdm_par_red):    
        one_enter = True
       
        #Create a list of all permutations in the exp_vec_initial
        exp_vec = list(itertools.combinations(index_vec, num_var))
        #For each index of variables that receives some degree of exp_final[id_exp]
        for id_exp_vec in range(len(exp_vec)):
            
            #Identify which basis functions to be evaluated
            #Isolated
            if (num_var == 1):
                start = 1
                num_basis = order + 1
                
            #Pairwise functions
            if (num_var > 1):
                if (params['expansion_crossed_terms']): 
                    if (one_enter):
                        start = num_basis
                        num_basis_ = scipy.special.comb(2, 2, exact = True)*scipy.special.comb(order, 2, exact = True)
                        num_basis = start + int(round(num_basis_))
                        one_enter = False
                else: 
                    start = 1
                    num_basis = 1
                    
            for idex in range(start, num_basis):
                
                #Select a exponent array
                exp_vec_temp = reduced_pindex[idex, :]
                
                func, params = spy_gram_schmidt(tuple(exp_vec_temp), params,
                                                reduced_pindex)
                
                #Create a reduced number of variables which a dummy variables.
                #Dummy variables are auxiliar variables to run over permutations in the 
                #actual variables.
                dummy_t = [spy.symbols('x_{}'.format(j)) for j in range(0, num_var)]
            
                f = spy.lambdify([dummy_t], 
                                 params['orthnormfunc'][tuple(exp_vec_temp)],
                                 'numpy')
                
                #In parallel, we create a symbolic version of the above 
                #orthornormal function
                sym_f = spy.lambdify([tuple(dummy_t)], 
                                     params['orthnormfunc'][tuple(exp_vec_temp)],
                                     'sympy')
                
                #print(exp_vec)        
                
                loc_exp = np.asarray(exp_vec[id_exp_vec], dtype = int)
                
                #Create a array to receive columns of the data X_t as we run over
                #permutations of the exponent array
                #eval_array = np.zeros((M, num_var))    
                
                sym_eval_array = []
                
                eval_array = np.copy(X_t[:, loc_exp])
                
                sym_list = []
                for id_loc in range(loc_exp.shape[0]):
                    sym_list.append(x_t[loc_exp[id_loc]])
                
                sym_eval_array = sym_list
                
                #Evaluate at this columns from the data X_t and plug it into library matrix
                PHI[:, l] = f(eval_array.T)/np.sqrt(M)
                    
                #At the same time, build matrix R of the QR decomposition
                symbolic_PHI.append(sym_f(tuple(sym_eval_array)))
                R[:, l] = get_coeff_matrix_wrt_basis(sym_f(tuple(sym_eval_array)), 
                                                           dict_canonical_basis_functions)
            
                l = l + 1
                #end_time = time.time()
                #print(l, (end_time - start_time)/60, '\n')
    
    #Update params dictionary to include the symbolic version of orthonormal functions
    params['symbolic_PHI'] = symbolic_PHI
    
    #Update params dictionary to include matrix R
    params['R'] = R
    
    return PHI, params


def implicit_PHI(node, B, PHI, params):
    
    N = params['number_of_vertices']
    
    x_t = [spy.symbols('x_{}'.format(j)) for j in range(0, N)]
    
    exp_vec = np.zeros(N)
    exp_vec[node] = 1
        
    #Create a expression to be evaluated at data X_t
    f = spy.lambdify([x_t], params['orthnormfunc'][tuple(exp_vec)],
                     'numpy')
    
    #Evaluate expression f at X_t as first column of library matrix
    b = B[:, node]#f(B.T)
    
    THETA = np.hstack((PHI, -1*np.diag(b) @ PHI[:, 1:])) 
    
    return THETA, b



def library_matrix(X_t, params): 
    '''
    Parameters
    ----------
    X_t : numpy array - size: (lenght of time series, number of vertices)
        Time series to be plugged into the library matrix.
    params : dictionary
         
    Returns
    -------
    PHI : numpy array
    dictionary_matrix : dictionary
        The library matrix built from time series X_t and dictionary_matrix contains
        all info about the reconstruction such as number of elements in the basis and normalization
        of columns factor.

    '''
    
    N = params.get('number_of_vertices', X_t.shape[1])      #N - number of vertices
    M = params.get('length_of_time_series', X_t.shape[0])   #M - Number of measuments in time
    order = params.get('max_deg_monomials', 2)              #order - Maximum degree in the polynimal expansion
    normalization = params.get('normalize_cols', False)      #normalization - condition on the columns to be normalized
    crossed_terms_condition = params.get('expansion_crossed_terms', False)
    canonical_basis = params.get('use_canonical', False)
    orthonormal_basis = params.get('use_orthonormal', True)
    id_trial = params.get('id_trial', np.arange(0, N, 1, dtype = int))
    
    L = params.get('L')
    
    if (canonical_basis) & (orthonormal_basis):
        print("Incompatible choice: Choose a single set of basis functions.")
        return
    
    #In case of using the canonical basis functions
    if canonical_basis:
        power_indices = index_iteration(params)
        params['power_indices'] = power_indices
        params = symbolic_canonical(params)
        PHI = np.zeros((M, L))
        
        PHI[:, 0] = np.ones(M)/np.sqrt(M)
        
        if not crossed_terms_condition:
            for deg in range(1, order + 1):
                index_vec = np.arange(1, L, order) + (deg - 1)
                #PHI[:, 1 + (deg - 1)*N : 1 + deg*N] = poly(deg)(X_t)/np.sqrt(M)
                PHI[:, index_vec] = poly(deg)(X_t)/np.sqrt(M)
                
            if(normalization):
                norm_column = LA.norm(PHI, axis = 0)
                
                params['norm_column'] = norm_column
                return PHI/norm_column, params
            else:    
                return PHI, params
        
        if crossed_terms_condition:
            for l in range(power_indices.shape[0]):
                PHI[:, l] = polynomial_exp(X_t, power_indices[l, :])/np.sqrt(M)
            
            if(normalization):
                norm_column = LA.norm(PHI, axis = 0)
                
                params['norm_column'] = norm_column
                return PHI/norm_column, params
            else:
                return PHI, params
                                
    if(orthonormal_basis):
        
        PHI = np.zeros((M, L))

        #In case to use a pre-defined orthonormal basis 
        '''
        Warning. Removing this part introduces a single representation for
        the power indices.
        '''
        #if(params['use_orthnorm_func_data'] and params['orthnorm_func_map']):
        #   power_indices  = params.get('power_indices')
           
        #Otherwise, the indices are constructed for the chosen params
        #else:   
        power_indices = index_iteration(params)
        params['power_indices'] = power_indices
        
        if not crossed_terms_condition:
            #
            params['max_deg_generating'] = (np.max(np.sum(params['power_indices'], axis = 1))*2)**2
            params = generate_moments(params, params['single_density'])
            
            if params['build_from_reduced_basis']:
                reduced_power_indices = reduced_index_iteration(params)
                PHI, params = reduced_poly_orthonormal_PHI(X_t, params, 
                                                           reduced_power_indices)
            
            if not params['build_from_reduced_basis']:
                PHI, params = poly_orthonormal_PHI(X_t, params, power_indices)
            
            #In case of saving the orthornormal functions
            if(params['save_orthnormfunc']):
                d = SympyDict(params['orthnormfunc'])
                d.save(params['orthnorm_func_filename'])    
        
        if crossed_terms_condition:
            #Estimate the maximum degree appearing the calculation of norm
            params['max_deg_generating'] = (np.max(np.sum(power_indices, axis = 1))*2)**2
            
            #Calculate all moments up to order max_deg_generating
            params = generate_moments(params, params['single_density'])
            if not params['build_from_reduced_basis']:
                #Calculate library matrix for polynomial basis
                PHI, params = poly_orthonormal_PHI(X_t, params, power_indices)
                #In case of saving the orthornormal functions
                if(params['save_orthnormfunc']):
                    d = SympyDict(params['orthnormfunc'])
                    d.save(params['orthnorm_func_filename'])    
            
            #In case of using an orthonormal basis from pre defined calculation
            if params['build_from_reduced_basis']:
                reduced_power_indices = reduced_index_iteration(params)
                PHI, params = reduced_poly_orthonormal_PHI(X_t, params, reduced_power_indices)
                
        #In case of normalizing columns of library matrix
        if(normalization):
            norm_column = LA.norm(PHI, axis = 0)
            
            params['norm_column'] = norm_column
            return PHI/norm_column, params
        else:
            return PHI, params     
     