'''

General utility functions 

'''
import os
import numpy as np
import scipy.stats as Stat
from scipy.stats import nbinom as nbinom
from scipy import interpolate
# -- astropy --
from astropy import units as U
from astropy import constants as Const


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


def HAsfr(zdist, ha_flux): 
    ''' calculate the Halpha SFR for the NSA data using 
    the ZDIST and HAFLUX columns
    '''
    ha_flux = ha_flux * 1e-17 * U.erg/U.s/U.cm**2 # get the units right 
    H0 = 70. * U.km/U.s/U.Mpc
    ha_flux *= 4. * np.pi * (zdist * Const.c / H0)**2 

    sfr = ha_flux.to(U.erg/U.s) /(10.**41.28)
    return sfr.value


def jansky(flux, kcorrect):
    '''Getting fluxes in Janskies from Nanomaggies:
    Inputs: Choose Petrosian/Sersic Nmgy and the relevant Kcorrection
    '''
    flux_in_Jy = flux*3631*(10.0**(-9.0))*(10**(kcorrect/(-2.5)))
    return flux_in_Jy


def UVsfr(zdist, fmag, nmag, rmag, f_flux):
    ''' Calculate UV star formation rates.
    Inputs: nsa.field('ZDIST'), F-band magnitude, N-band magnitude, r-band magnitude, F-band flux in Janskies
    '''
    fn = fmag - nmag
    opt = nmag - rmag   # N-r

    #Luminosity Distance
    #dist = WMAP7.comoving_distance(z)
    #ldist = (1+z)*dist.value
    H0 = 70. * U.km/U.s/U.Mpc
    ldist = (zdist * Const.c/H0).to(U.Mpc).value

    #calculating Attenuation 'atten'
    atten = np.repeat(-999., len(fmag))

    case1 = np.where((opt > 4.) & (fn < 0.95))
    atten[case1] = 3.32*fn[case1] + 0.22
    case2 = np.where((opt > 4.) & (fn >= 0.95))
    atten[case2] = 3.37
    case3 = np.where((opt <= 4.) & (fn < 0.9))
    atten[case3] = 2.99*fn[case3] + 0.27
    case4 = np.where((opt <= 4.) & (fn >= 0.9))
    atten[case4] = 2.96

    lum = 4.*np.pi*(ldist**2.0)*(3.087**2.0)*(10**(25.0 +(atten/2.5)))*f_flux  #Luminosity
    sfr = 1.08*(10**(-28.0))*np.abs(lum)
    return sfr


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
