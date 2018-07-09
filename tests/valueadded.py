'''
'''
import h5py
import numpy as np
from astropy.io import fits
from astropy import units as U
from astropy import constants as Const
from pydl.pydlutils.spheregroup import spherematch
# -- iQuench -- 
from letstalkaboutquench import util as UT
# -- plotting--
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


def valueadd(): 
    '''
    '''
    fnsa = h5py.File(''.join([UT.dat_dir(), 'dickey_NSA_iso_lowmass_gals.valueadd.hdf5']), 'r') 
    fsdss = h5py.File(''.join([UT.dat_dir(), 'tinker_SDSS_centrals_M9.7.valueadd.hdf5']), 'r') 
    
    fig = plt.figure(figsize=(12,6))
    sub = fig.add_subplot(121)
    sub.scatter(np.log10(fsdss['MASS'].value), np.log10(fsdss['UVSFR'].value), c='C0', s=1)
    sub.scatter(np.log10(fnsa['MASS'].value), np.log10(fnsa['UVSFR'].value), c='C1', s=1)
    #sub.scatter(np.log10(fsdss['ms_tinker'].value), np.log10(fsdss['UVSFR'].value), c='C0', s=1)
    #sub.scatter(fnsa['MASS_claire'].value, np.log10(fnsa['UVSFR'].value), c='C1', s=1)
    sub.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', fontsize=25) 
    sub.set_xlim([8., 12.]) 
    sub.set_ylabel(r'log ( SFR $[M_\odot \, yr^{-1}]$ )', fontsize=25) 
    sub.set_ylim([-4., 2.]) 
    sub.text(0.075, 0.975, 'UV SFR',  ha='left', va='top', 
            transform=sub.transAxes, fontsize=25)

    sub = fig.add_subplot(122)
    #sub.scatter(np.log10(fsdss['ms_tinker'].value), np.log10(fsdss['HASFR'].value), c='C0', s=1)
    sub.scatter(np.log10(fsdss['MASS'].value), np.log10(fsdss['HASFR'].value), c='C0', s=1)
    #sub.scatter(fnsa['MASS_claire'].value, np.log10(fnsa['HASFR'].value), c='C1', s=1)
    sub.scatter(np.log10(fnsa['MASS'].value), np.log10(fnsa['HASFR'].value), c='C1', s=1)
    sub.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', fontsize=25) 
    sub.set_xlim([8., 12.]) 
    sub.set_ylim([-4., 2.]) 
    sub.text(0.075, 0.975, r'H$\alpha$ SFR',  ha='left', va='top', 
            transform=sub.transAxes, fontsize=25)
    fig.savefig(''.join([UT.fig_dir(), 'valueaddSDSScentral.png']), bbox_inches='tight') 
    return None


def NSA_UVHAsfr(): 
    ''' plot UV/HA SFR-M* relation of NSA galaxies
    '''
    # read in NSA data 
    nsa = fits.open(''.join([UT.dat_dir(), 'nsa_v0_1_2.fits']))
    nsa_data = nsa[1].data
    # NSA HA SFR
    nsa_ha_sfr = UT.HAsfr(nsa_data.field('ZDIST'), nsa_data.field('HAFLUX'))
    # NSA UV SFR
    nsa_fuv_jansky = UT.jansky(nsa_data.field('NMGY')[:,0],  # nanomaggies
            nsa_data.field('KCORRECT')[:,0]) # kcorrect
    nsa_uv_sfr = UT.UVsfr(nsa_data.field('z'),
                   nsa_data.field('ABSMAG')[:,0],
                   nsa_data.field('ABSMAG')[:,1],
                   nsa_data.field('ABSMAG')[:,4],
                   nsa_fuv_jansky)
    
    # check the UV/HA SFR -- Mstar relation 
    fig = plt.figure(figsize=(6,6))
    sub = fig.add_subplot(111)
    sub.scatter(np.log10(nsa_data.field('mass')), np.log10(nsa_uv_sfr), 
            c='k', s=0.5, label='UV SFR')
    sub.scatter(np.log10(nsa_data.field('mass')), np.log10(nsa_ha_sfr), 
            c='C1', s=0.1, label=r'H$\alpha$ SFR')
    sub.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', fontsize=25) 
    sub.set_xlim([7., 12.]) 
    sub.set_ylabel(r'log ( SFR $[M_\odot \, yr^{-1}]$ )', fontsize=25) 
    sub.set_ylim([-4., 2.]) 
    fig.savefig(''.join([UT.fig_dir(), 'NSA_HAUVsfr.png']), bbox_inches='tight') 
    return None 



def NSA_tinker_mass(): 
    fnsa = h5py.File(''.join([UT.dat_dir(), 'dickey_NSA_iso_lowmass_gals.valueadd.hdf5']), 'r') 
    fsdss = h5py.File(''.join([UT.dat_dir(), 'tinker_SDSS_centrals_M9.7.valueadd.hdf5']), 'r') 
    
    fig = plt.figure(figsize=(6,6))
    sub = fig.add_subplot(111)
    sub.scatter(np.log10(fsdss['MASS'].value), np.log10(fsdss['ms_tinker'].value), c='k', s=1)
    #sub.scatter(np.log10(fnsa['MASS'].value), fnsa['MASS_claire'].value, c='C1', s=1)
    sub.plot([6., 12.], [6., 12.], c='C1', lw=1.5, ls='--') 
    sub.set_xlabel(r'log ( $M_*^\mathrm{NSA} \;\;[M_\odot]$ )', fontsize=25) 
    sub.set_xlim([7., 12.]) 
    sub.set_ylabel(r'log ( $M_*^\mathrm{Tinker} \;\;[M_\odot]$ )', fontsize=25) 
    sub.set_ylim([7., 12.]) 
    fig.savefig(''.join([UT.fig_dir(), 'Mstar_nsa_tinker_comparison.png']), bbox_inches='tight') 
    return None 


if __name__=='__main__': 
    #NSA_UVHAsfr()
    #valueadd()
    NSA_tinker_mass()
