import os 
import h5py
import numpy as np 
import dill as pickle 

# --- Local ---
import util as UT
    
    
class Catalog: 
    def __init__(self): 
        ''' class object for reading in log(M*) and log(SFR)s of various galaxy 
        catalogs.
        '''
        self.catalog_dict = {
                'santacruz1': 'sc_sam_sfr_mstar_correctedweights.txt', 
                'santacruz2': 'sc_sam_sfr_mstar_correctedweights.txt', 
                'tinkergroup': 'tinker_SDSS_centrals_M9.7.dat',
                'illustris1': 'Illustris1_SFR_M_values.csv', 
                'illustris2': 'Illustris1_SFR_M_values.csv', 
                'nsa_dickey': 'dickey_NSA_iso_lowmass_gals.txt', 
                'mufasa': 'Mufasa_m50n512_z0.cat'
                }
        self.catalog_list = self.catalog_dict.keys()

    def Read(self, name): 
        ''' Here we deal with the disparate outputs of the different catalogs and output
        log(M*), log(SFR), weight 

        name : (string) 
            catalog name e.g. 'santacruz1'
        '''
        f_name = ''.join([UT.dat_dir(), self._File(name)]) 
        if name == 'santacruz1': # santa cruz log(SFR) [10^5 yr]
            sc_logM, sc_logSFR1, sc_logw = np.loadtxt(f_name, unpack=True, skiprows=1, usecols=[0,1,3])  
            logM = sc_logM 
            logSFR = sc_logSFR1
            w = 10**sc_logw 
        elif name == 'santacruz2': # santa cruz log(SFR) [10^8 yr]
            sc_logM, sc_logSFR2, sc_logw = np.loadtxt(f_name, unpack=True, skiprows=1, usecols=[0,2,3])  
            logM = sc_logM 
            logSFR = sc_logSFR2
            w = 10**sc_logw 
        elif name == 'tinkergroup': # Tinker group catalog 
            tink_Mstar, tink_logSSFR = np.loadtxt(f_name, unpack=True, skiprows=2, usecols=[0, 7])
            logM = np.log10(tink_Mstar)
            logSFR = tink_logSSFR + logM
            w = np.ones(len(logM))
        elif name == 'illustris1': # Illustris SFR timescale = 2 x 10^7 yr
            ill_logMstar, ill_ssfr1 = np.loadtxt(f_name, unpack=True, skiprows=1, delimiter=',', usecols=[0,1])
            logM = ill_logMstar
            logSFR = np.log10(ill_ssfr1) + ill_logMstar
            w = np.ones(len(logM))
        elif name == 'illustris2': # Illustris SFR timescale = 10^9 yr
            ill_logMstar, ill_ssfr2 = np.loadtxt(f_name, unpack=True, skiprows=1, delimiter=',', usecols=[0,2])
            logM = ill_logMstar
            logSFR = np.log10(ill_ssfr2) + ill_logMstar
            w = np.ones(len(logM))
        elif name == 'nsa_dickey': 
            dic_logM, dic_logSFR =  np.loadtxt(f_name, unpack=True, skiprows=1, usecols=[1,5]) 
            logM = dic_logM            
            logSFR = dic_logSFR 
            w = np.ones(len(logM))
        elif name == 'mufasa':
            muf_M, muf_SSFR =  np.loadtxt(f_name, unpack=True, skiprows=1, usecols=[0, 1]) 
            logM = np.log10(muf_M)
            logSFR = np.log10(muf_SSFR) + logM
            w = np.ones(len(logM))
        else: 
            raise ValueError('')

        # deal with sfr = 0 or other non finite numbers 
        sfr_zero = np.where(np.isfinite(logSFR) == False)
        print len(sfr_zero[0]), ' of ', len(logM), ' galaxies have 0/non-finite SFRs'
        print 'logSFR of these galaxies will be -999.'
        logSFR[sfr_zero] = -999.

        return [logM, logSFR, w]

    def _File(self, name): 
        ''' catalog file names
        '''
        if name not in self.catalog_dict.keys(): 
            raise ValueError('catalog not yet included')
        else: 
            return self.catalog_dict[name]

    def CatalogLabel(self, name):
        ''' Label of catalogs. Given the catalog name, return label 
        '''
        label_dict = {
                'santacruz1': 'Santa Cruz [$10^5$ yr]', 
                'santacruz2': 'Santa Cruz [$10^8$ yr]', 
                'tinkergroup': 'Tinker Group',
                'illustris1': r'Illustris [$2 \times 10^7$ yr]', 
                'illustris2': r'Illustris [$10^9$ yr]',
                'nsa_dickey': 'NSA Dickey', 
                'mufasa': 'MUFASA'
                }

        return label_dict[name]

    def CatalogColors(self, name): 
        ''' Colors for different catalogs. Defined so that all the plot colors are 
        consistent.
        '''
        color_dict = {
                'santacruz1': 'blue', 
                'santacruz2': 'darkblue', 
                'tinkergroup': 'red',
                'illustris1': 'green', 
                'illustris2': 'darkgreen',
                'nsa_dickey': 'purple', 
                'mufasa': 'yellow'
                }
        return color_dict[name]


def Build_Illustris_SFH(): 
    ''' Build catalog from Illustris SFH data (binsv2all1e8Msunh_z0.hdf5)
    '''
    # read in illustris data 
    f = h5py.File(UT.dat_dir()+'binsv2all1e8Msunh_z0.hdf5', 'r')
    # load data into a dict
    galpop = {} 
    galpop['t'] = []
    galpop['z'] = []
    galpop['m_star0'] = f['CurrentStellarMass'].value.flatten() * 1e10 # [Msun]

    t_bins = np.array([0.0, 0.005, 0.015, 0.025, 0.035, 0.045, 0.055, 0.065, 0.075, 0.085, 0.095, 0.125,0.175,0.225,0.275,0.325,0.375,0.425,0.475,0.55,0.65,0.75,0.85,0.95,1.125,1.375,1.625,1.875,2.125,2.375,2.625,2.875,3.125,3.375,3.625,3.875,4.25,4.75,5.25,5.75,6.25,6.75,7.25,7.75,8.25,8.75,9.25,9.75,10.25,10.75,11.25,11.75,12.25,12.75,13.25,13.75])

    dm_grid = f['FormedStellarMass'].value # grid of d M* in bins of cosmic time and metallicities 
    dm_t = np.sum(dm_grid, axis=1) # summed

    galpop['sfr0'] = (1.e10 * (dm_t[:,0] + dm_t[:,1])/(0.015 * 1e9)).flatten() # "current" SFR 
    galpop['t'].append(0.005)
    
    zoft = UT.f_zoft()
    tofz = UT.f_tofz() 
    galpop['z'].append(zoft(tofz(0.) - galpop['t'][0]))

    for i in range(len(t_bins)-5): 
        # calculate M* and SFRs at previous time bins 
        dt = t_bins[3+i] - t_bins[2+i]
        t_i = 0.5 * (t_bins[3+i] + t_bins[2+i])
        galpop['t'].append(t_i)
        galpop['z'].append(zoft(tofz(0.) - t_i))

        galpop['sfr'+str(i+1)] = (10. * dm_t[:,2+i] / dt).flatten() 
        galpop['m_star'+str(i+1)] = galpop['m_star0'] - 1e10 * np.sum(dm_t[:,:i+2], axis=1)
    
    galpop['t'] = np.array(galpop['t'])
    galpop['z'] = np.array(galpop['z'])

    # save galpop dict to hdf5 file 
    g = h5py.File(UT.dat_dir()+'illustris_sfh.hdf5', 'w') 
    for k in galpop.keys(): 
        g.create_dataset(k, data=galpop[k])
    f.close() 
    g.close() 
    return None  


if __name__=='__main__': 
    Build_Illustris_SFH()
    '''
    Cat = Catalog()
    
    for cata in ['santacruz1', 'santacruz2', 'tinkergroup', 'illustris1', 'illustris2', 'nsa_dickey', 'mufasa']: 
        logM, logSFR, w = Cat.Read(cata)
        print cata
        print Cat.CatalogLabel(cata)
        print logM[:10], logSFR[:10], w[:10]
        print '------------------------------------------------------------'
    '''
