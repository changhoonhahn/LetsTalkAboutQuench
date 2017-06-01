import os 
import numpy as np 
import dill as pickle 

# --- Local ---
import util as UT


    
    
    
class Catalog: 
    def __init__(self): 
        ''' class object for reading in log(M*) and log(SFR)s of various catalogs
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
        # deal with sfr = 0 
        sfr_zero = np.where(logSFR == -np.inf)
        print len(sfr_zero[0]), ' of ', len(logM), ' galaxies have 0 SFR; logSFR of these galaxies will be -999.'
        logSFR[sfr_zero] = -999.

        return [logM, logSFR, w]

    def _File(self, name): 
        ''' catalog file names
        '''
        if name not in self.catalog_dict.keys(): 
            raise ValueError('catalog not yet included')
        else: 
            return self.catalog_dict[name]

if __name__=='__main__': 
    Cat = Catalog()
    
    for cata in ['santacruz1', 'santacruz2', 'tinkergroup', 'illustris1', 'illustris2', 'nsa_dickey', 'mufasa']: 
        logM, logSFR, w = Cat.Read(cata)
        print cata
        print logM[:10], logSFR[:10], w[:10]
        print '------------------------------------------------------------'
