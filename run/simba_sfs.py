'''

quick script to identify SFS of simba 


'''
import os
import pickle 
import numpy as np 
import corner as DFM 
# --- lets talk about quench --- 
from letstalkaboutquench import util as UT
from letstalkaboutquench import catalogs as Cats  
# -- starFS -- 
from starfs.starfs import starFS as sFS
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


mufasa_mmin = 9.2

# read in SIMBA z=0 data 
for simba in ['_simba_inst', '_simba_100myr']: 
    cats = Cats.Catalog()
    logm, logsfr, _, cs = cats.Read(simba, keepzeros=True) 

    cut = (cs == 1) & (~cats.zero_sfr) & (logm > mufasa_mmin) 

    # fit the SFMS
    fSFS = sFS()#fit_range=[mufasa_mmin, 12.0]) # stellar mass range
    sfs_fit = fSFS.fit(
            logm[cut], logsfr[cut], 
            method='gaussmix',      # Gaussian Mixture Model fitting 
            dlogm = 0.2,          # stellar mass bins of 0.4 dex
            slope_prior = [0.0, 2.0], 
            Nbin_thresh=100,        # at least 100 galaxies in bin 
            error_method='bootstrap',  # uncertainty estimate method 
            n_bootstrap=100)        # number of bootstrap bins
    
    f_sfs = os.path.join(UT.dat_dir(), 'paper1', 'gmmSFSfit.%s.gfcentral.mlim.p' % simba)
    pickle.dump(fSFS, open(f_sfs, 'wb'))
    
    _ = fSFS.powerlaw(logMfid=10.5) 
    print('power law slope = %.2f' % fSFS._powerlaw_m) 
    print('power law offset = %.2f' % fSFS._powerlaw_c) 
    
    fig = plt.figure(figsize=(5,5))
    sub = fig.add_subplot(111)
    DFM.hist2d(logm[cut], logsfr[cut], color='k', 
            levels=[0.68, 0.95], range=[[7.8, 12.], [-4., 4.]], 
            plot_datapoints=True, fill_contours=False, plot_density=True, 
            ax=sub) 
    sub.errorbar(fSFS._fit_logm, fSFS._fit_logsfr, yerr=fSFS._fit_err_logssfr, fmt='.C0') 
    mx = np.linspace(8., 12., 10)
    sub.plot(mx, fSFS._powerlaw_m * (mx - 10.5) + fSFS._powerlaw_c, c='C1', ls='--') 
    sub.set_xlim([8.5, 12.]) 
    sub.set_ylim([-3., 2.]) 
    sub.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', fontsize=25) 
    sub.set_ylabel(r'log ( SFR $[M_\odot \, yr^{-1}]$ )', fontsize=25) 
    ffig = os.path.join(UT.dat_dir(), 'paper1', 'gmmSFSfit.%s.gfcentral.mlim.png' % simba)
    fig.savefig(ffig, bbox_inches='tight') 
    plt.close() 
