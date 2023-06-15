"""
Set of methods to aid fourier_library module

Created on Thu Feb 17 08:32:25 2022

@author: Edmilson Roque dos Santos
"""

from .triage import triage_params
import os

def create_orthnormfunc_filename(params):
    '''
    To create the orthonormal filename using parameters for a simulation.

    Parameters
    ----------
    params : dict
    
    Returns
    -------
    scenario_output : str
        path for orthonormal functions file.

    '''
    parameters = params.copy()
    
    parameters = triage_params(parameters)
    
    ##### Identification for output
    out_direc = os.path.join(params['exp_name'], params['network_name'])
    if os.path.isdir(out_direc) == False:
        os.makedirs(out_direc)
    
    #For coupling analysis it is necessary to save each orthonormal function 
    #with respect to this coupling.
    filename = 'fourier_onf_deg_{}_seed_{}_lgth_ts_{}_coupling_{}'.format(parameters['max_deg_harmonics'], 
                                                              parameters['random_seed'], 
                                                              parameters['length_of_time_series'], 
                                                              parameters['coupling'])
    scenario_output = os.path.join(out_direc, filename)
    
    return scenario_output