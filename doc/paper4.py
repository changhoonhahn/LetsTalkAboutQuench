'''

IQ paper 4 --- High z galaxies


'''
import numpy as np 
import scipy as sp
import corner as DFM 
from scipy import linalg
from scipy.stats import multivariate_normal as MNorm
from sklearn.mixture import GaussianMixture as GMix
# -- letstalkaboutquench --
from letstalkaboutquench import util as UT
from letstalkaboutquench import catalogs as Cats
from letstalkaboutquench import galprop as Gprop
from letstalkaboutquench.fstarforms import fstarforms
from letstalkaboutquench.fstarforms import sfr_mstar_gmm
# -- plotting --
import matplotlib as mpl
import matplotlib.pyplot as plt 
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['axes.xmargin'] = 1
mpl.rcParams['xtick.labelsize'] = 'x-large'
mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.labelsize'] = 'x-large'
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['legend.frameon'] = False


def candels(): 
    ''' SFR -- M* relation of CANDLES galaxies in the 6 redshift bins
    '''
    zlo = [0.5, 1., 1.4, 1.8, 2.2, 2.6]
    zhi = [1., 1.4, 1.8, 2.2, 2.6, 3.0]

    datas = [] 
    for i in range(1,7): 
        f_candels = ''.join([UT.dat_dir(), 'highz/CANDELS/CANDELS_z', str(i), '.txt']) 
        data = np.loadtxt(f_candels, skiprows=2, unpack=True) # z, M*, SFR
        datas.append(data)
    
    fig = plt.figure(figsize=(12,8))
    bkgd = fig.add_subplot(111, frameon=False)
    for i_z, data in enumerate(datas): 
        sub = fig.add_subplot(2,3,i_z+1) 
        DFM.hist2d(np.log10(data[1]), np.log10(data[2]), color='C0', 
                levels=[0.68, 0.95], range=[[7.8, 12.], [-4., 4.]], 
                plot_datapoints=True, fill_contours=False, plot_density=True, 
                ax=sub) 
        sub.set_xlim([8.5, 12.]) 
        sub.set_ylim([-3., 4.]) 
        sub.text(0.95, 0.05, '$'+str(zlo[i_z])+'< z <'+str(zhi[i_z])+'$', 
                ha='right', va='bottom', transform=sub.transAxes, fontsize=20)
        if i_z == 0: 
            sub.text(0.05, 0.95, 'CANDELS', 
                    ha='left', va='top', transform=sub.transAxes, fontsize=25)
    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', labelpad=15, fontsize=25) 
    bkgd.set_ylabel(r'log ( SFR $[M_\odot \, yr^{-1}]$ )', labelpad=15, fontsize=25) 
    fig.subplots_adjust(wspace=0.2, hspace=0.15)
    fig_name = ''.join([UT.doc_dir(), 'highz/figs/candels_sfrM.pdf'])
    fig.savefig(fig_name, bbox_inches='tight')
   

def candels_sfms():
    ''' SFMS fits to the SFR -- M* relation of CANDLES galaxies in 
    the 6 redshift bins
    '''
    zlo = [0.5, 1., 1.4, 1.8, 2.2, 2.6]
    zhi = [1., 1.4, 1.8, 2.2, 2.6, 3.0]

    fSFMS = fstarforms()
    datas = [] 
    sfms_fits = [] 
    for i in range(1,7): 
        f_candels = ''.join([UT.dat_dir(), 'highz/CANDELS/CANDELS_z', str(i), '.txt']) 
        data = np.loadtxt(f_candels, skiprows=2, unpack=True) # z, M*, SFR
        datas.append(data)
        # fit the SFMSes
        sfms_fit = fSFMS.fit(np.log10(data[1]), np.log10(data[2]), 
                method='gaussmix',      # Gaussian Mixture Model fitting 
                fit_range=[8.5, 12.0],  # stellar mass range
                dlogm = 0.4,            # stellar mass bins of 0.4 dex
                Nbin_thresh=100,        # at least 100 galaxies in bin 
                fit_error='bootstrap',  # uncertainty estimate method 
                n_bootstrap=100)        # number of bootstrap bins
        sfms_fits.append(sfms_fit) 
    
    # SFMS overplotted ontop of SFR--M* relation 
    fig = plt.figure(figsize=(12,8))
    bkgd = fig.add_subplot(111, frameon=False)
    for i_z, data in enumerate(datas): 
        sub = fig.add_subplot(2,3,i_z+1) 
        DFM.hist2d(np.log10(data[1]), np.log10(data[2]), color='C0', 
                levels=[0.68, 0.95], range=[[7.8, 12.], [-4., 4.]], 
                plot_datapoints=True, fill_contours=False, plot_density=True, 
                ax=sub) 
        # plot SFMS fit
        sub.errorbar(sfms_fits[i_z][0], sfms_fits[i_z][1], sfms_fits[i_z][2], fmt='.k')
        sub.set_xlim([8.5, 12.]) 
        sub.set_ylim([-3., 4.]) 
        sub.text(0.95, 0.05, '$'+str(zlo[i_z])+'< z <'+str(zhi[i_z])+'$', 
                ha='right', va='bottom', transform=sub.transAxes, fontsize=20)
        if i_z == 0: 
            sub.text(0.05, 0.95, 'CANDELS', 
                    ha='left', va='top', transform=sub.transAxes, fontsize=25)

    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', labelpad=15, fontsize=25) 
    bkgd.set_ylabel(r'log ( SFR $[M_\odot \, yr^{-1}]$ )', labelpad=15, fontsize=25) 
    fig.subplots_adjust(wspace=0.2, hspace=0.15)
    fig_name = ''.join([UT.doc_dir(), 'highz/figs/candels_sfms.pdf'])
    fig.savefig(fig_name, bbox_inches='tight')
    
    # SFMS fit redshift evolution 
    fig = plt.figure(figsize=(6,6))
    sub = fig.add_subplot(111)
    for i_z, sfms_fit in enumerate(sfms_fits):  
        sub.fill_between(sfms_fit[0], sfms_fit[1]-sfms_fit[2], sfms_fit[1]+sfms_fit[2], 
                color='C'+str(i_z), linewidth=0.5, alpha=0.75, 
                label='$'+str(zlo[i_z])+'< z <'+str(zhi[i_z])+'$') 
    sub.legend(loc='upper left', prop={'size':15}) 
    sub.set_xlim([8.5, 12.]) 
    sub.set_ylim([-0.5, 4.]) 
    fig_name = ''.join([UT.doc_dir(), 'highz/figs/candels_sfmsz.pdf'])
    fig.savefig(fig_name, bbox_inches='tight')
    return None 


if __name__=="__main__": 
    #candels() 
    candels_sfms()
