'''
Plots for Paper II of quenched galaxies series 

author(s): ChangHoon Hahn 

'''
import h5py
import numpy as np 
import scipy as sp
import corner as DFM 
# -- LetsTalkAboutQuench --
from letstalkaboutquench import util as UT
from letstalkaboutquench import catalogs as Cats
from letstalkaboutquench import galprop as Gprop
from letstalkaboutquench.fstarforms import fstarforms
from letstalkaboutquench.fstarforms import sfr_mstar_gmm
# -- plotting -- 
import matplotlib as mpl
import matplotlib.pyplot as plt 
from matplotlib.font_manager import FontProperties
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


def HAUV_SFR_Mstar(): 
    ''' SFR-M* relation of observations using Halpha and UV derived SFRs
    '''
    # read in Claire's NSA catalog with value added
    f_nsa = h5py.File(''.join([UT.dat_dir(), 'dickey_NSA_iso_lowmass_gals.valueadd.hdf5']), 'r') 
    # read in SDSS catalog
    f_sdss = h5py.File(''.join([UT.dat_dir(), 'tinker_SDSS_centrals_M9.7.valueadd.hdf5']), 'r') 
    
    # comparison of Halpha to UV SFR-M* 
    fig = plt.figure(figsize=(12,6))
    bkgd = fig.add_subplot(111, frameon=False) 
    sub1 = fig.add_subplot(121) # NSA catalog
    sub1.scatter(f_nsa['MASS_claire'].value, f_nsa['HALPHA_SFR_claire'], c='k', s=0.5, 
            label=r'H$\alpha$ SFR')
    sub1.scatter(f_nsa['MASS_claire'].value, np.log10(f_nsa['UVSFR']), c='C1', s=0.1, 
            label='UV SFR')
    sub1.set_xlim([8., 12.]) 
    sub1.set_ylim([-4., 2.]) 
    sub1.text(0.9, 0.1, 'NSA Centrals', ha='right', va='center', 
        transform=sub1.transAxes, fontsize=20)
    sub1.legend(loc='upper left', frameon=False, 
            handletextpad=0.5, markerscale=20, prop={'size': 15}) 

    sub2 = fig.add_subplot(122) # SDSS
    sub2.scatter(np.log10(f_sdss['ms_tinker'].value), np.log10(f_sdss['HASFR']), c='k', s=0.5) 
    sub2.scatter(np.log10(f_sdss['ms_tinker'].value), np.log10(f_sdss['UVSFR']), c='C1', s=0.1)
    sub2.set_xlim([8., 12.]) 
    sub2.set_ylim([-4., 2.]) 
    sub2.text(0.1, 0.9, 'SDSS Centrals', ha='left', va='center', 
        transform=sub2.transAxes, fontsize=20)

    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', labelpad=15, fontsize=25) 
    bkgd.set_ylabel(r'log ( SFR $[M_\odot \, yr^{-1}]$ )', labelpad=15, fontsize=25) 

    fig.subplots_adjust(wspace=0.2, hspace=0.15)
    fig_name = ''.join([UT.doc_dir(), 'figs/HAUV_SFR_Mstar1.png']) #pdf'])
    fig.savefig(fig_name, bbox_inches='tight')

    fig = plt.figure(figsize=(12,6))
    bkgd = fig.add_subplot(111, frameon=False) 
    sub1 = fig.add_subplot(121) # Halpha 
    DFM.hist2d(f_nsa['MASS_claire'].value, f_nsa['HALPHA_SFR_claire'].value, 
            color='k', levels=[0.68, 0.95], range=[[7.8, 12.], [-4., 2.]], 
            plot_datapoints=False, fill_contours=False, plot_density=False, 
            ax=sub1) 
    DFM.hist2d(f_nsa['MASS_claire'].value, np.log10(f_nsa['HASFR'].value), #f_nsa['HALPHA_SFR_claire'].value, 
            color='C1', levels=[0.68, 0.95], range=[[7.8, 12.], [-4., 2.]], 
            plot_datapoints=True, fill_contours=False, plot_density=True, 
            ax=sub1) 
    DFM.hist2d(np.log10(f_sdss['ms_tinker'].value), np.log10(f_sdss['HASFR'].value), color='C0' ,
            levels=[0.68, 0.95], range=[[7.8, 12.], [-4., 2.]], 
            plot_datapoints=True, fill_contours=False, plot_density=True, 
            ax=sub1) 
    sub1.set_xlim([8., 12.]) 
    sub1.set_ylim([-4., 2.]) 
    sub1.text(0.1, 0.9, r'H$\alpha$ SFR', ha='left', va='center', 
        transform=sub1.transAxes, fontsize=20)

    sub2 = fig.add_subplot(122)
    DFM.hist2d(f_nsa['MASS_claire'].value, np.log10(f_nsa['UVSFR'].value), color='C1', 
            levels=[0.68, 0.95], range=[[7.8, 12.], [-4., 2.]],
            plot_datapoints=True, fill_contours=False, plot_density=True, 
            ax=sub2) 
    DFM.hist2d(np.log10(f_sdss['ms_tinker'].value), np.log10(f_sdss['UVSFR'].value), color='C0' ,
            levels=[0.68, 0.95], range=[[7.8, 12.], [-4., 2.]], 
            plot_datapoints=True, fill_contours=False, plot_density=True, 
            ax=sub2) 
    sub2.set_xlim([8., 12.]) 
    sub2.set_ylim([-4., 2.]) 
    sub2.text(0.1, 0.9, r'UV SFR', ha='left', va='center', 
        transform=sub2.transAxes, fontsize=20)
    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', labelpad=15, fontsize=25) 
    bkgd.set_ylabel(r'log ( SFR $[M_\odot \, yr^{-1}]$ )', labelpad=15, fontsize=25) 

    fig.subplots_adjust(wspace=0.2, hspace=0.15)
    fig_name = ''.join([UT.doc_dir(), 'figs/HAUV_SFR_Mstar2.png']) #pdf'])
    fig.savefig(fig_name, bbox_inches='tight')
    return None 
   

def SSFR_Dn4000(): 
    ''' SSFR vs Dn4000 relation of observations
    '''
    # read in Claire's NSA catalog with value added
    f_nsa = h5py.File(''.join([UT.dat_dir(), 'dickey_NSA_iso_lowmass_gals.valueadd.hdf5']), 'r') 
    # read in SDSS catalog
    f_sdss = h5py.File(''.join([UT.dat_dir(), 'tinker_SDSS_centrals_M9.7.valueadd.hdf5']), 'r') 

    # calculate median SSFR in Dn4000 bins 
    d4000_bin = np.linspace(f_nsa['D4000_claire'].value.min(), 
            f_nsa['D4000_claire'].value.max(), 20)
    nsa_d4000_mid, nsa_ssfr_med = [], []
    for i in range(len(d4000_bin)-1): 
        inbin = ((f_nsa['D4000_claire'].value > d4000_bin[i]) & 
                (f_nsa['D4000_claire'].value < d4000_bin[i+1]) & 
                np.isfinite(f_nsa['HALPHA_SSFR_claire'].value))
        if np.sum(inbin) < 5.: 
            continue 
        nsa_d4000_mid.append(0.5 * (d4000_bin[i] + d4000_bin[i+1]))
        nsa_ssfr_med.append(np.median(f_nsa['HALPHA_SSFR_claire'].value[inbin]))

    d4000_bin = np.linspace(0., f_sdss['D4000'].value.max(), 20)
    sdss_d4000_mid, sdss_ssfr_med = [], []
    sdss_hassfr = np.log10(f_sdss['HASFR'].value) - np.log10(f_sdss['ms_tinker'].value)
    for i in range(len(d4000_bin)-1): 
        inbin = ((f_sdss['D4000'].value > d4000_bin[i]) & 
                (f_sdss['D4000'].value < d4000_bin[i+1]) & 
                np.isfinite(sdss_hassfr) & 
                (f_sdss['HASFR'].value != -999.))
        if np.sum(inbin) < 5.: 
            continue 
        sdss_d4000_mid.append(0.5 * (d4000_bin[i] + d4000_bin[i+1]))
        sdss_ssfr_med.append(np.median(sdss_hassfr[inbin]))
    
    # comparison of Halpha to UV SFR-M* 
    fig = plt.figure(figsize=(8,6))
    sub1 = fig.add_subplot(111) # NSA catalog
    sub1.scatter(f_nsa['D4000_claire'].value, f_nsa['HALPHA_SSFR_claire'].value, c='C1', s=0.5) 
    sub1.plot(nsa_d4000_mid, nsa_ssfr_med, c='k', lw=3, ls='--') 
    sub1.scatter(f_sdss['D4000'].value, sdss_hassfr, c='C0', s=0.5) 
    sub1.plot(sdss_d4000_mid, sdss_ssfr_med, c='k', lw=3, ls=':') 
    print sdss_d4000_mid, sdss_ssfr_med
    sub1.set_xlim([0.9, 2.25]) 
    sub1.set_ylim([-13., -8.]) 
    sub1.text(0.05, 0.1, 'NSA Centrals', ha='left', va='center', 
        transform=sub1.transAxes, fontsize=20)
    sub1.set_xlabel(r'$D_n 4000$', fontsize=25) 
    sub1.set_ylabel(r'log ( H$\alpha$ SSFR $[yr^{-1}]$ )', fontsize=25) 
    fig.subplots_adjust(wspace=0.2, hspace=0.15)
    fig_name = ''.join([UT.doc_dir(), 'figs/HASSFR_Dn4000.pdf'])
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close() 
    return None 


def P_Dn4000(): 
    ''' distribuiton of Dn4000
    '''
    # read in Claire's NSA catalog with value added
    f_nsa = h5py.File(''.join([UT.dat_dir(), 'dickey_NSA_iso_lowmass_gals.valueadd.hdf5']), 'r') 
    # read in SDSS catalog
    f_sdss = h5py.File(''.join([UT.dat_dir(), 'tinker_SDSS_centrals_M9.7.valueadd.hdf5']), 'r') 

    # calculate median SSFR in Dn4000 bins 
    fig = plt.figure(figsize=(8,8))
    sub1 = fig.add_subplot(211) # NSA catalog
    _ = sub1.hist(f_nsa['D4000'].value, range=[0.9, 2.25], bins=20, density=True) 
    sub1.set_xlim([0.9, 2.25]) 
    #sub1.set_ylim([-13., -8.]) 
    sub1.text(0.05, 0.1, 'NSA Centrals', ha='left', va='center', 
        transform=sub1.transAxes, fontsize=20)
    sub1.set_ylabel(r'$P(D_n 4000)$', fontsize=25) 
    sub1 = fig.add_subplot(212) # NSA catalog
    _ = sub1.hist(f_sdss['D4000'].value, range=[0.9, 2.25], bins=20, density=True) 

    # comparison of Halpha to UV SFR-M* 
    #sub1.scatter(f_nsa['D4000'].value, f_nsa['HALPHA_SSFR'].value, c='k', s=0.5) 
    #sub1.plot(d4000_mid, ssfr_med, c='C1', lw=3, ls='--') 
    sub1.set_xlim([0.9, 2.25]) 
    #sub1.set_ylim([-13., -8.]) 
    sub1.text(0.05, 0.1, 'SDSS Centrals', ha='left', va='center', 
        transform=sub1.transAxes, fontsize=20)
    sub1.set_xlabel(r'$D_n 4000$', fontsize=25) 
    sub1.set_ylabel(r'$P(D_n 4000)$', fontsize=25) 
    fig.subplots_adjust(wspace=0.2, hspace=0.15)
    fig_name = ''.join([UT.doc_dir(), 'figs/P_Dn4000.pdf'])
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close() 
    return None 


if __name__=="__main__": 
    #P_Dn4000()
    #SSFR_Dn4000()
    HAUV_SFR_Mstar()
