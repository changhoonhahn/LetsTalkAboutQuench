'''
'''
import numpy as np 

# --- Local ---
import env 
from catalogs import Catalog as Cat
import util as UT
 
import matplotlib.pyplot as plt 
import corner as DFM 
from ChangTools.plotting import prettyplot
from ChangTools.plotting import prettycolors


def SFR_Mstar_SSFRcut(name, xrange=None, yrange=None): 
    ''' Test that the logSFR-logM* relation is reasonable and that
    read works.
    '''
    cat = Cat() 
    logM, logSFR, w = cat.Read(name)
    
    prettyplot() 
    fig = plt.figure() 
    sub = fig.add_subplot(111)
    if xrange is None:
        xrange = [7., 12.]
    if yrange is None:
        yrange = [-4., 2.]
    
    DFM.hist2d(logM, logSFR, color='#1F77B4', 
                levels=[0.68, 0.95], range=[xrange, yrange], 
                plot_datapoints=True, fill_contours=False, plot_density=True, ax=sub) 
    f_SSFRcut = cat._default_fSSFRcut()
    sub.plot(np.linspace(xrange[0], xrange[1], 10), [f_SSFRcut(m)+m for m in np.linspace(xrange[0], xrange[1], 10)], c='r', ls='--', lw=2)

    sub.set_xlim(xrange)
    sub.set_ylim(yrange)
    sub.set_xlabel(r'$\mathtt{log \; M_* \;\;[M_\odot]}$', labelpad=10, fontsize=25) 
    sub.set_ylabel(r'$\mathtt{log \; SFR \;\;[M_\odot \, yr^{-1}]}$', labelpad=10, fontsize=25) 
    fig.savefig(''.join([UT.fig_dir(), name, '.sfr_mstar.ssfrcut.png']), bbox_inches='tight') 
    plt.close() 
    return None 


def SFR_Mstar(name, xrange=None, yrange=None): 
    ''' Test that the logSFR-logM* relation is reasonable and that
    read works.
    '''
    cat = Cat() 
    logM, logSFR, w = cat.Read(name)
    
    prettyplot() 
    fig = plt.figure() 
    sub = fig.add_subplot(111)
    if xrange is None:
        xrange = [7., 12.]
    if yrange is None:
        yrange = [-4., 2.]
    
    DFM.hist2d(logM, logSFR, color='#1F77B4', 
                levels=[0.68, 0.95], range=[xrange, yrange], 
                plot_datapoints=True, fill_contours=False, plot_density=True, ax=sub) 
    #plt.scatter(logM, logSFR, s=2)
    sub.set_xlim(xrange)
    sub.set_ylim(yrange)
    sub.set_xlabel(r'$\mathtt{log \; M_* \;\;[M_\odot]}$', labelpad=10, fontsize=25) 
    sub.set_ylabel(r'$\mathtt{log \; SFR \;\;[M_\odot \, yr^{-1}]}$', labelpad=10, fontsize=25) 
    fig.savefig(''.join([UT.fig_dir(), name, '.sfr_mstar.png']), bbox_inches='tight') 
    plt.close() 
    return None 


def pssfr(name, Mrange=[10.,10.5], xrange=None): 
    '''
    '''
    cat = Cat() 
    logM, logSFR, w = cat.Read(name)
    
    # mass bins
    inmbin = np.where((logM >= Mrange[0]) & (logM < Mrange[1]))
    
    prettyplot() 
    fig = plt.figure() 
    sub = fig.add_subplot(111)
    
    if xrange is None: 
        xrange = [-2.5, 2]
    yy, xx_edges = np.histogram(logSFR[inmbin], bins=20, range=[-2.5, 2], normed=True)
    x_plot, y_plot = UT.bar_plot(xx_edges, yy)
    sub.plot(x_plot, y_plot, lw=3)
    sub.set_xlim(xrange)
    sub.set_xlabel(r'$\mathtt{log \; SFR \;\;[M_\odot \, yr^{-1}]}$', labelpad=10, fontsize=25) 
    sub.set_ylabel(r'$\mathtt{P(log \; SFR)\;}$', labelpad=10, fontsize=25) 
    f_fig = ''.join([UT.fig_dir(), name, '.Psfr.mstar', str(Mrange[0]), '_', str(Mrange[1]), '.png'])
    fig.savefig(f_fig, bbox_inches='tight') 
    plt.close() 
    return None



if __name__=='__main__': 
    SFR_Mstar_SSFRcut('illustris_10myr', xrange=[7.5, 12.])
