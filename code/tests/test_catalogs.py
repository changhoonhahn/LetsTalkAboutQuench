'''
'''
import numpy as np 

# --- Local ---
import env 
from catalogs import Catalog as Cat
 
import matplotlib.pyplot as plt 
from ChangTools.plotting import prettyplot
from ChangTools.plotting import prettycolors


def catRead(name): 
    ''' *** TESTED *** 
    Test reading in catalog 
    '''
    cat = Cat()
    logM, logSFR, w = cat.Read(name)
    plt.scatter(logM, logSFR, s=2)
    plt.xlim([8., 12.])
    plt.ylim([-4., 2.])
    plt.show() 
    return None 


if __name__=='__main__': 
    pass 
