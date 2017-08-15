'''

General utility functions 

'''
import os
import numpy as np
import scipy.stats as Stat
from scipy.stats import nbinom as nbinom

from scipy import interpolate
from astropy.cosmology import FlatLambdaCDM 

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
    return os.path.dirname(os.path.realpath(__file__)).split('code')[0]+'figs/'


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


def f_zoft(H0=70., Om0=0.274): 
    ''' return function z(t_cosmic) 
    '''
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om0) # cosmo 
    z_arr = np.arange(0., 10.1, 0.1)
    t_arr = cosmo.age(z_arr).value 

    zoft = interpolate.interp1d(list(reversed(t_arr)), list(reversed(z_arr)), kind='cubic') 
    return zoft


def f_tofz(H0=70., Om0=0.274): 
    ''' return function z(t_cosmic) 
    '''
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om0) # cosmo 
    z_arr = np.arange(0., 10.1, 0.1)
    t_arr = cosmo.age(z_arr).value 

    zoft = interpolate.interp1d(list(reversed(z_arr)), list(reversed(t_arr)), kind='cubic') 
    return zoft
