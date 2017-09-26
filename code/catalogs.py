import os 
import h5py
import numpy as np 

# --- Local ---
import util as UT
    
    
class Catalog: 
    def __init__(self): 
        ''' class object for reading in log(M*) and log(SFR)s of various galaxy 
        catalogs.
        '''
        self.catalog_dict = { 
                'illustris_1gyr': 'Illustris1_extended_individual_galaxy_values_all1e8Msunh_z0.csv', 
                'illustris_10myr': 'Illustris1_extended_individual_galaxy_values_all1e8Msunh_z0.csv', 
                #'illustris1': 'Illustris1_SFR_M_values.csv', 
                #'illustris2': 'Illustris1_SFR_M_values.csv', 
                'eagle_1gyr': 'EAGLE_RefL0100_MstarSFR_allabove1.8e8Msun.txt',
                'eagle_10myr': 'EAGLE_RefL0100_MstarSFR_allabove1.8e8Msun.txt',
                'mufasa_1gyr': 'MUFASA_GALAXY.txt',
                'mufasa_10myr': 'MUFASA_GALAXY.txt',
                #'santacruz1': 'sc_sam_sfr_mstar_correctedweights.txt',
                #'santacruz2': 'sc_sam_sfr_mstar_correctedweights.txt',
                'tinkergroup': 'tinker_SDSS_centrals_M9.7.dat',
                'nsa_dickey': 'dickey_NSA_iso_lowmass_gals.txt', 
                }
        self.catalog_list = self.catalog_dict.keys()

    def Read(self, name): 
        ''' Here we deal with the disparate outputs of the different catalogs and output
        log(M*), log(SFR), weight 

        name : (string) 
            catalog name e.g. 'santacruz1'
        '''
        f_name = ''.join([UT.dat_dir(), self._File(name)]) 
        if 'illustris' in name:  
            if name == 'illustris_1gyr': # Illustris SFR on 1 Gyr timescales
                logM, ill_ssfr = np.loadtxt(f_name, unpack=True, skiprows=1, delimiter=',', usecols=[0,3])
            elif name == 'illustris_10myr': # Illustris SFR on 10 Myr timescales
                logM, ill_ssfr = np.loadtxt(f_name, unpack=True, skiprows=1, delimiter=',', usecols=[0,1])
            elif name == 'illustris1': # Illustris SFR timescale = 2 x 10^7 yr
                logM, ill_ssfr = np.loadtxt(f_name, unpack=True, skiprows=1, delimiter=',', usecols=[0,1])
            elif name == 'illustris2': # Illustris SFR timescale = 10^9 yr
                logM, ill_ssfr = np.loadtxt(f_name, unpack=True, skiprows=1, delimiter=',', usecols=[0,2])
            logSFR = np.log10(ill_ssfr) + logM
            w = np.ones(len(logM))
        elif 'eagle' in name: # EAGLE simulation 
            if name == 'eagle_1gyr': # SFR on 1 Gyr timescales
                logM, eag_SFR = np.loadtxt(f_name, unpack=True, skiprows=1, usecols=[2,4])
            elif name == 'eagle_10myr': # SFR on 1 Myr timescales
                logM, eag_SFR = np.loadtxt(f_name, unpack=True, skiprows=1, usecols=[2,3])
            logSFR = np.log10(eag_SFR) 
            w = np.ones(len(logM))
        elif 'mufasa' in name: # MUFASA simulation 
            if name == 'mufasa_1gyr': 
                logM, logSFR = np.loadtxt(f_name, unpack=True, skiprows=13, usecols=[6,8])
            elif name == 'mufasa_10myr': 
                logM, logSFR = np.loadtxt(f_name, unpack=True, skiprows=13, usecols=[6,7])
            w = np.ones(len(logM))
        elif name == 'santacruz1': # santa cruz log(SFR) [10^5 yr]
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
        sfr_zero = np.where((np.isfinite(logSFR) == False) | (logSFR < -5))
        print '------ ', name, ' ------'
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
                'illustris_1gyr': r'Illustris [$1$ Gyr]',
                'illustris_10myr': r'Illustris [$10$ Myr]', 
                'illustris1': r'Illustris [$2 \times 10^7$ yr]', 
                'illustris2': r'Illustris [$10^9$ yr]',
                'eagle_1gyr': r'EAGLE [$1$ Gyr]',
                'eagle_10myr': r'EAGLE [$10$ Myr]',
                'mufasa_1gyr': r'MUFASA [$1$ Gyr]', 
                'mufasa_10myr': r'MUFASA [$10$ Myr]', 
                'santacruz1': 'Santa Cruz [$10^5$ yr]', 
                'santacruz2': 'Santa Cruz [$10^8$ yr]', 
                'tinkergroup': 'Tinker Group',
                'nsa_dickey': 'NSA Dickey' 
                }
        if name not in label_dict.keys():
            print name 
            raise ValueError
        return label_dict[name]

    def CatalogColors(self, name): 
        ''' Colors for different catalogs. Defined so that all the plot colors are 
        consistent.
        '''
        color_dict = {
                'illustris_1gyr': 'green',
                'illustris_10myr': 'darkgreen', 
                'illustris1': 'green', 
                'illustris2': 'darkgreen',
                'eagle_1gyr': 'darkred',
                'eagle_10myr': 'red',
                'mufasa_1gyr': 'yellow',
                'mufasa_10myr': 'darkyellow',
                'santacruz1': 'blue', 
                'santacruz2': 'darkblue', 
                'tinkergroup': 'red',
                'nsa_dickey': 'purple'
                }
        if name not in color_dict.keys():
            print name 
            raise ValueError
        return color_dict[name]

    def _default_Mstar_range(self, name): 
        ''' stellar mass range of catalog where the SFR-M* relation
        is well defined. This is determined by eye and *conservatively*
        selected.
        '''
        if name == 'illustris_1gyr': 
            fit_Mrange = [8.5, 10.5]
        elif name == 'illustris_10myr': 
            fit_Mrange = [9.75, 10.75]
        elif name == 'eagle_1gyr': 
            fit_Mrange = [8.5, 11.]
        elif name == 'eagle_10myr': 
            fit_Mrange = [8.5, 10.5]
        elif name == 'mufasa_1gyr': 
            fit_Mrange = [8.75, 10.5]
        elif name == 'mufasa_10myr': 
            fit_Mrange = [8.75, 10.5]
        elif name == 'tinkergroup': 
            fit_Mrange = [10., 11.5]
        elif name == 'nsa_dickey': 
            fit_Mrange = [7.5, 10.]
        return fit_Mrange
    
    def _default_fSSFRcut(self, name): 
        ''' default SSFRcut 
        '''
        f_SSFRcut = lambda mm: -11.
        return f_SSFRcut


def Build_Illustris_SFH(): 
    ''' Build central galaxy catalog with SFHs from Illustris data and 
    Jeremey Tinker's group finder
    '''
    # read in illustris SFH data 
    f = h5py.File(UT.dat_dir()+'binsv2all1e8Msunh_z0.hdf5', 'r')
    # load data into a dict
    galpop = {} 
    galpop['t'] = []
    galpop['z'] = []
    galpop['m_star0'] = f['CurrentStellarMass'].value.flatten() * 1e10 # [Msun]
    
    # time binning of SFHs 
    t_bins = np.array([0.0, 0.005, 0.015, 0.025, 0.035, 0.045, 0.055, 0.065, 0.075, 0.085, 0.095, 0.125,0.175,0.225,0.275,0.325,0.375,0.425,0.475,0.55,0.65,0.75,0.85,0.95,1.125,1.375,1.625,1.875,2.125,2.375,2.625,2.875,3.125,3.375,3.625,3.875,4.25,4.75,5.25,5.75,6.25,6.75,7.25,7.75,8.25,8.75,9.25,9.75,10.25,10.75,11.25,11.75,12.25,12.75,13.25,13.75])

    dm_grid = f['FormedStellarMass'].value # grid of d M* in bins of cosmic time and metallicities 
    dm_t = np.sum(dm_grid, axis=1) # summed over metallicities 

    galpop['sfr0'] = (1.e10 * (dm_t[:,0] + dm_t[:,1])/(0.015 * 1e9)).flatten() # "current" SFR averaged over 0.015 Gyr
    galpop['t'].append(0.5*0.015) 
    
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
    f.close() 

    # group finder data 
    grp = np.loadtxt(UT.dat_dir()+"Illustris1_extended_individual_galaxy_values_all1e8Msunh_z0.csv", 
            unpack=True, skiprows=1, delimiter=',', usecols=[-1]) 
    galpop['central'] = grp 

    # save galpop dict to hdf5 file 
    g = h5py.File(UT.dat_dir()+'illustris_sfh.hdf5', 'w') 
    for k in galpop.keys(): 
        g.create_dataset(k, data=galpop[k])
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
