'''

General utility functions 

'''
import os
import numpy as np
import scipy.stats as Stat
from scipy.stats import nbinom as nbinom
from scipy import interpolate

def flatten(x): 
    ''' deals with those pesky cases when you have an array or list with only one element!
    '''
    if isinstance(x, float): 
        return x
    elif isinstance(x, list): 
        if len(x) == 1: 
            return x[0] 
        else: 
            return x
    elif isinstance(x, np.ndarray): 
        if len(x) == 1: 
            return x[0]
        else: 
            return x 
    else: 
        return x


def check_env(): 
    if os.environ.get('IQUENCH_DIR') is None: 
        raise ValueError("set $IQUENCH_DIR environment varaible!") 
    return None


def dat_dir(): 
    ''' directory that contains all the data files, defined by environment 
    variable $IQUENCH_DIR
    '''
    return os.environ.get('IQUENCH_DIR') 


def fig_dir(): 
    ''' directory to dump all the figure files 
    '''
    if os.environ.get('IQUENCH_FIGDIR') is None: 
        if os.path.isdir(dat_dir()+'/figs/'):
            return dat_dir()+'/figs/'
        else: 
            raise ValueError("create figs/ folder in $IQUENCH_DIR directory for figures; or specify $IQUENCH_FIGDIR")
    else: 
        return os.environ.get('IQUENCH_FIGDIR')


def doc_dir(): 
    ''' directory for paper related stuff 
    '''
    return fig_dir().split('fig')[0]+'doc/'


def gaussianKDE_contour(x, y, xmin=None, xmax=None, ymin=None, ymax=None):
    ''' Returns [xx, yy, f]. To plot filled contour contourf(xx, yy, f, cmap='Blues'). To plot 
    contour lines contour(xx, yy, f, colors='k')
    '''
    if len(x) != len(y): 
        raise ValueError("x and y do not have the same dimensions")
    assert xmin
    assert xmax
    assert ymin
    assert ymax
    # import x and y range on x and y 
    lims = np.where((x >= xmin) & (x < xmax) & (y >= ymin) & (y < ymax))
    x_in = x[lims] 
    y_in = y[lims]

    # Peform the kernel density estimate
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])

    values = np.vstack([x_in, y_in])
    kernel = Stat.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    return [xx, yy, f]


def bar_plot(bin_edges, values): 
    ''' Take outputs from numpy histogram and return pretty bar plot
    '''
    xx = [] 
    yy = [] 

    for i_val, val in enumerate(values): 
        xx.append(bin_edges[i_val]) 
        yy.append(val)
        xx.append(bin_edges[i_val+1]) 
        yy.append(val)

    return [np.array(xx), np.array(yy)]


def NB_pdf_logx(k, mu, theta, loc=0, dk=1.): 
    ''' PDF of Negative binomial distribution 
    '''
    pdeff = np.zeros(len(k))
    big = np.where(k > 1.)
    p = theta / (theta+mu)
    cdf1 = nbinom.cdf(k[big]-dk, theta, p, loc)
    cdf2 = nbinom.cdf(k[big]+dk, theta, p, loc)
                                        
    pdeff[big] = (cdf2 - cdf1)/(np.log10(k[big]+dk) - np.log10(k[big]-dk))
    return pdeff
