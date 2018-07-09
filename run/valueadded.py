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


def valueaddNSAcentrals(): 
    ''' add NSA columns to Claire's NSA central dwarf galaxy catalog
    '''
    # Claire's original central catalog
    claire = np.loadtxt(''.join([UT.dat_dir(), 'dickey_NSA_iso_lowmass_gals.txt']), skiprows=1)
    # data columns in the file 
    cols_claire = ['NSAID_claire', 'MASS_claire', 'DHOST_claire', 'D4000_claire', 
        'HAEW_claire', 'HALPHA_SFR_claire', 'HALPHA_SSFR_claire']
    assert claire.shape[1] == len(cols_claire)

    # read in NSA data 
    nsa = fits.open(''.join([UT.dat_dir(), 'nsa_v0_1_2.fits']))
    nsa_data = nsa[1].data
    n_nsa = len(nsa_data.field('RA'))
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
    
    # match up based on NSA ID 
    claire_nsaid = claire[:,0].astype(int)
    i_nsa = []
    for i in range(claire.shape[0]):
        match = (claire_nsaid[i] == nsa_data.field("NSAID"))
        if np.sum(match) > 1:
            raise ValueError
        elif np.sum(match) == 0: 
            raise ValueError
        i_nsa.append(np.arange(n_nsa)[match])
    i_nsa = np.array(i_nsa).flatten()
    
    # save to hdf5 file 
    f = h5py.File(''.join([UT.dat_dir(), 'dickey_NSA_iso_lowmass_gals.valueadd.hdf5']), 'w') 
    for i_col, col in enumerate(cols_claire): # add in Claire's data first
        f.create_dataset(col, data=claire[:,i_col])

    for name in nsa_data.names: # add in the NSA data columns  
        nsa_col_data = nsa_data.field(name)
        f.create_dataset(name, data=nsa_col_data[i_nsa])

    # add in UV SFR column 
    f.create_dataset('HASFR', data=nsa_ha_sfr[i_nsa]) 
    f.create_dataset('UVSFR', data=nsa_uv_sfr[i_nsa])
    f.close()
    return None 


def valueaddSDSScentrals(): 
    ''' add NSA columns to Jeremy's central galaxy catalog
    '''
    # Jeremy's original central catalog
    tinker = np.loadtxt(''.join([UT.dat_dir(), 'tinker_SDSS_centrals_M9.7.dat']), skiprows=2)
    # data columns in the file 
    cols_tinker = ['ms_tinker', 'mr_tinker', 'mg_tinker', 'mh_tinker', 
            'd_tinker', 'psat_tinker', 'dn4k_tinker', 'sfr_tinker', 
            'hdelta_tinker', 'rexp_tinker', 'nsersic_tinker', 
            'con_tinker', 'sigv1_tinker', 'ka_tinker']
    assert tinker.shape[1] == len(cols_tinker)

    # read in RA and Dec that Jeremy provided
    tinker_ra, tinker_dec = np.loadtxt(''.join([UT.dat_dir(), 'central_positions_tinker_SDSS.dat']), 
            skiprows=2, unpack=True, usecols=[0,1])
    
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
    
    # match Jeremy's SDSS catalog to NSA 
    i_nsa = matchNSA(tinker_ra, tinker_dec, nsa_data) 
    hasmatch = (i_nsa != -999)

    f = h5py.File(''.join([UT.dat_dir(), 'tinker_SDSS_centrals_M9.7.valueadd.hdf5']), 'w') 
    for i_col, col in enumerate(cols_tinker): # add in Tinker's data first
        f.create_dataset(col, data=tinker[:,i_col])

    for name in nsa_data.names:
        nsa_col_data = nsa_data.field(name)
        if 'S' in str(nsa_col_data.dtype): 
            blank = np.array(['-999'])
            blank.astype(nsa_col_data.dtype)
        else: 
            blank = np.array([-999.]) 
            blank.astype(nsa_col_data.dtype)

        empty_shape = list(nsa_col_data.shape)
        empty_shape[0] = tinker.shape[0]
        empty = np.tile(blank[0], empty_shape)
        empty[hasmatch] = nsa_col_data[i_nsa[hasmatch]]
        f.create_dataset(name, data=empty)

    # add in UV SFR column 
    empty = np.tile(-999., tinker.shape[0]) 
    empty[hasmatch] = nsa_ha_sfr[i_nsa[hasmatch]]
    f.create_dataset('HASFR', data=empty) 
    empty = np.tile(-999., tinker.shape[0]) 
    empty[hasmatch] = nsa_uv_sfr[i_nsa[hasmatch]]
    f.create_dataset('UVSFR', data=empty)
    f.close()
    return None 


def matchNSA(ra, dec, nsa_data): 
    ''' match target RA and Dec to RA and DEC from the NSA 
    catalog. Return the matching indices of the NSA catalog. 
    If there's no match -999 is returned. If there's more 
    than one the script will just crash. 
    '''
    match_length = (3 * U.arcsec).to(U.degree) # 3'' matching length
    # now lets spherematch the two
    m_nsa, m_target, dmatch = spherematch(
            nsa_data.field('ra'), nsa_data.field('dec'), 
            ra, dec, 
            match_length.value, maxmatch=0) 

    i_nsa = np.zeros(len(ra)).astype(int)
    for i in range(len(ra)):
        targ = (m_target == i) 
        if np.sum(targ) > 1: 
            raise ValueError
        elif np.sum(targ) == 0: 
            i_nsa[i] = -999 
        else: 
            i_nsa[i] = m_nsa[targ] 
    return i_nsa 


if __name__=="__main__": 
    #valueaddNSAcentrals()
    valueaddSDSScentrals()
