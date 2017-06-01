'''

General utility functions 

'''
import os
import numpy as np
from scipy import interpolate


def code_dir(): 
    ''' Directory where all the code is located (the directory that this file is in!)
    '''
    return os.path.dirname(os.path.realpath(__file__))


def dat_dir(): 
    ''' dat directory is symlinked to a local path where the data files are located
    '''
    return os.path.dirname(os.path.realpath(__file__)).split('code')[0]+'dat/'


def fig_dir(): 
    ''' dat directory is symlinked to a local path where the data files are located
    '''
    return os.path.dirname(os.path.realpath(__file__)).split('code')[0]+'fig/'


#def GrabLocalFile(string, machine='harmattan'): 
#    ''' scp dat_dir()+string from machine (harmattan)
#    '''
#    # parse subdirectory
#    sub_dir = '/'.join(string.split('/')[:-1] + [''])
#
#    if machine == 'harmattan': 
#        scp_cmd = "scp harmattan:/data1/hahn/centralMS/"+string+" "+dat_dir()+sub_dir
#        print scp_cmd
#        os.system(scp_cmd)
#    else: 
#        raise NotImplementedError
#
#    return None
