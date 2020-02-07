import os
import pickle 
import numpy as np 
import pandas as pd 
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


def get_sim(sim=None,sample='cent'):
    dname="dat/"
    fname={'Illustris':'Illustris_with_hih2.dat',
            'Eagle':'EAGLE_RefL0100HashPhotfix_MHIH2HIIRhalfmassSFRT1e4_allabove1.8e8Msun.txt',
            'Mufasa':'halos_m50n512_z0.0.dat',
            'TNG':'TNG_with_hih2.dat',
            'Simba':'halos_m100n1024_z0.0.dat',
            'SC-SAM':'SCSAMgalprop_updatedVersion.dat'}
    cols={'Illustris':[1,2,3,4,5], #cent, sfr, Mstar, MHI, MH2
            'Eagle':[2,6,10], #logMstar, logMHI, logMH2
            'Mufasa':[1,2,3,4,5],#cent, sfr, Mstar, MHI, MH2, #old Mstar, MHI, MH2, sfr, cent
            'TNG':[1,2,3,4,5], #cent, sfr, Mstar, MHI, MH2
            'Simba':[1,2,3,4,5], #Mstar, MHI, MH2, sfr, cent
            'SC-SAM':[3,20,7,15,16]} #cent, Mstar, MHI, MH2, sfr
    
    if sim=='Eagle': #eagle has sfr and cent in a different file so need to add those
        Mstar,mHI,mH2=np.loadtxt(dname+fname[sim],usecols=cols[sim],unpack=True)
        fname2='EAGLE_RefL0100_MstarMcoldgasSFR_allabove1.8e8Msun.txt'
        sfr,cent=np.loadtxt(dname+fname2,usecols=(4,5),unpack=True) #check that instant
        c=cent.astype(bool)
        mgas=np.log10(10**(mHI)+10**(mH2))
        data=pd.DataFrame({'central':c,'logMstar':Mstar,'log_SFR':np.log10(sfr)
        ,'logMHI':mHI,'logMH2':mH2,'logMgas':mgas})   
    elif sim=='SC-SAM':
        cent,sfr,Mstar,mHI,mH2=np.loadtxt(dname+fname[sim],usecols=cols[sim],unpack=True)
        c=np.invert(cent.astype(bool))
        data=pd.DataFrame({'central':c,'logMstar':np.log10(Mstar*1.e9),'log_SFR':np.log10(sfr)
        ,'logMHI':np.log10(mHI*1.e9),'logMH2':np.log10(mH2*1.e9)
        ,'logMgas':np.log10((mHI+mH2)*1.e9)})
    else:
        cent,sfr,Mstar,mHI,mH2=np.loadtxt(dname+fname[sim],usecols=cols[sim],unpack=True)
        c=cent.astype(bool)
        data=pd.DataFrame({'central':c,'logMstar':np.log10(Mstar),'log_SFR':np.log10(sfr)
        ,'logMHI':np.log10(mHI),'logMH2':np.log10(mH2),'logMgas':np.log10(mHI+mH2)})

    data=data[data['logMstar'] > 8.0]
    data.reset_index
    if sample=='cent':
        keep=(data['central']==True)
        print(f"{sim} central galaxies {keep.sum()}")
        data=data[keep]
        data.reset_index
    elif sample=='sat':
        keep=(data['central']==False)
        print(f"{sim} satellite galaxies {keep.sum()}") 
        data=data[keep]
        data.reset_index
    else:
        print(f"{sim} total galaxies {data.shape[0]}")
    return data



# read in SIMBA z=0 data 
for sim in ['Illustris', "Eagle", 'Mufasa', 'TNG', 'Simba', 'SC-SAM']: 
    data = get_sim(sim=sim,sample='cent')
    logm = np.array(data.logMstar).flatten()
    logsfr = np.array(data.log_SFR).flatten() 
    censat = data.central

    cut = censat & np.isfinite(logsfr) 

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
    
    f_sfs = os.path.join(UT.dat_dir(), 'ari', 'gmmSFSfit.%s.gfcentral.mlim.p' % sim)
    pickle.dump(fSFS, open(f_sfs, 'wb'))
    
    _ = fSFS.powerlaw(logMfid=10.5) 
    print(' --- %s --- ' % sim) 
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
    sub.text(0.05, 0.05, 'slope = %.2f, offset = %.2f' % (fSFS._powerlaw_m, fSFS._powerlaw_c) , ha='right', va='bottom', 
                transform=sub.transAxes, fontsize=20)
    sub.set_xlim([8.5, 12.]) 
    sub.set_ylim([-3., 2.]) 
    sub.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', fontsize=25) 
    sub.set_ylabel(r'log ( SFR $[M_\odot \, yr^{-1}]$ )', fontsize=25) 
    ffig = os.path.join(UT.dat_dir(), 'ari', 'gmmSFSfit.%s.gfcentral.mlim.png' % sim)
    fig.savefig(ffig, bbox_inches='tight') 
    plt.close() 
