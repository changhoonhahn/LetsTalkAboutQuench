import os 
import h5py
import numpy as np 
import warnings 

# --- Local ---
import util as UT
    
    
class Catalog: 
    def __init__(self): 
        ''' class object for reading in log(M*) and log(SFR)s of various galaxy 
        catalogs.
        '''
        # dictionary of all the file names
        # this needs to be cleaned up before public release
        self.catalog_dict = { 
                'illustris_inst': 'Illustris1_extended_individual_galaxy_values_all1e8Msunh_z0.csv', 
                'illustris_10myr': 'Illustris1_extended_individual_galaxy_values_all1e8Msunh_z0.csv', 
                'illustris_100myr': 'Illustris1_extended_individual_galaxy_values_all1e8Msunh_z0.csv', 
                'illustris_1gyr': 'Illustris1_extended_individual_galaxy_values_all1e8Msunh_z0.csv', 
                'eagle_inst': 'EAGLE_RefL0100_MstarMcoldgasSFR_allabove1.8e8Msun.txt', 
                'eagle_10myr': 'EAGLE_RefL0100_MstarSFR_allabove1.8e8Msun.txt',
                'eagle_100myr': 'EAGLE_RefL0100_MstarSFR100Myr_allabove1.8e8Msun.txt', 
                'eagle_1gyr': 'EAGLE_RefL0100_MstarSFR_allabove1.8e8Msun.txt',
                'mufasa_inst': 'MUFASA_GALAXY.txt',
                'mufasa_1gyr': 'MUFASA_GALAXY.txt',
                'scsam_inst': 'SCSAMgalprop.dat', 
                'scsam_10myr': 'SCSAMgalprop.dat', 
                'scsam_100myr': 'SCSAMgalprop.dat', 
                'scsam_1gyr': 'SCSAMgalprop.dat', 
                'nsa_combined': 'NSA_complete_SFRs.cat', 
                'nsa_combined_uv': 'NSA_complete_SFRs.cat', 
                'tinkergroup': 'tinker_SDSS_centrals_M9.7.dat',
                'nsa_dickey': 'dickey_NSA_iso_lowmass_gals.txt', 
                }
        self.catalog_list = self.catalog_dict.keys()

    def Read(self, name, silent=False): 
        ''' Here we deal with the disparate outputs of the different catalogs and output
        log(M*), log(SFR), weight, censat

        name : (string) 
            catalog name e.g. 'santacruz1'
        
        Notes
        -----
        * central/satellite : 1/0
        '''
        f_name = ''.join([UT.dat_dir(), self._File(name)]) 

        if 'illustris' in name: # illustris simulation 
            # header for file 
            # log10(M_*[Msun]),sSFR[1/yr](0Myr),sSFR[1/yr](10Myr),sSFR[1/yr](20Myr),sSFR[1/yr](1Gyr),log10(M_HI[Msun]),sigma_*(<0.8kpc)[km/s],log10(SFing M_HI[Msun]),Z(SFing gas)[metal mass fraction],log10(M_BH[Msun]),iscentral_SUBFIND
            if name == 'illustris_inst': # instantaneous SFRs
                logM, _ssfr, censat = np.loadtxt(f_name, unpack=True, 
                        skiprows=1, delimiter=',', usecols=[0,1,-1])
            elif name == 'illustris_10myr': # 10 Myr SFRs 
                logM, _ssfr, censat = np.loadtxt(f_name, unpack=True, 
                        skiprows=1, delimiter=',', usecols=[0,2,-1])
            elif name == 'illustris_100myr': # 100 Myr SFRs
                logM, _ssfr, censat = np.loadtxt(f_name, unpack=True, 
                        skiprows=1, delimiter=',', usecols=[0,4,-1])
            elif name == 'illustris_1gyr': # 1 Gyr SFRs
                logM, _ssfr, censat = np.loadtxt(f_name, unpack=True, 
                        skiprows=1, delimiter=',', usecols=[0,5,-1])
            else: 
                raise ValueError(name+" not found")
            logSFR = np.log10(_ssfr) + logM # calculate log SFR from sSFR
            w = np.ones(len(logM)) # uniform weights 

        elif 'eagle' in name: # EAGLE simulation 
            if name == 'eagle_inst': # instantaneous SFRs
                # header: GroupNr, SubGroupNr, log10(StellarMass)[Msun], log10(ColdGasMass)[Msun], InstantSFR[Msun/yr], Central_SUBFIND
                logM, _SFR, censat = np.loadtxt(f_name, unpack=True, skiprows=1, usecols=[2,4,5])
            elif name == 'eagle_10myr': # 10 Myr SFRs 
                # header: GroupNr, SubGroupNr, log10(StellarMass)[Msun], SFR10Myr[Msun/yr], SFR1Gyr[Msun/yr], Central_SUBFIND
                logM, _SFR, censat = np.loadtxt(f_name, unpack=True, skiprows=1, usecols=[2,3,5])
            elif name == 'eagle_100myr': # 100 Myr SFRs
                # header: GroupNr, SubGroupNr, log10(StellarMass)[Msun], SFR10Myr[Msun/yr], SFR100Myr[Msun/yr], Central_SUBFIND
                logM, _SFR, censat = np.loadtxt(f_name, unpack=True, skiprows=1, usecols=[2,4,5])
            elif name == 'eagle_1gyr': # SFR on 1 Gyr timescales
                # header: GroupNr, SubGroupNr, log10(StellarMass)[Msun], SFR10Myr[Msun/yr], SFR1Gyr[Msun/yr], Central_SUBFIND
                logM, _SFR, censat = np.loadtxt(f_name, unpack=True, skiprows=1, usecols=[2,4,5])
            else: 
                raise ValueError(name+" not found")
            logSFR = np.log10(_SFR) # log SFRs
            w = np.ones(len(logM)) # uniform weights 

        elif 'mufasa' in name: # MUFASA simulation 
            # header: positionx[kpc](0), positiony[kpc](1), positionz[kpc](2), velocityx[km/s](3), velocityy[km/s](4), velocityz[km/s](5), 
            # log10(Mstar_gal/Msun)(6), log10(SFR_10M[Msun/yr](7), log10(SFR_1G[Msun/yr])(8), log10(coldgasmass[Msun])(9), log10(Z/sfr[yr/Msun])(10), cen(1)/sat(0)(11)
            if name == 'mufasa_inst': 
                # 10 Myr is actually instantaneous (ask Romeel for details) 
                logM, logSFR, censat = np.loadtxt(f_name, unpack=True, skiprows=13, usecols=[6,7,11])
            elif name == 'mufasa_1gyr': 
                logM, logSFR, censat = np.loadtxt(f_name, unpack=True, skiprows=13, usecols=[6,8,11])
            else: 
                raise ValueError(name+" not found")
            w = np.ones(len(logM)) # uniform weights

        elif 'scsam' in name: # Santa Cruz Semi-Analytic model
            # 0 hosthaloid (long long) 1 birthhaloid (long long) 2 redshift 3 sat_type 0= central 4 mhalo total halo mass [1.0E09 Msun] 5 m_strip stripped mass [1.0E09 Msun] 
            # 6 rhalo halo virial radius [Mpc)] 7 mstar stellar mass [1.0E09 Msun] 8 mbulge stellar mass of bulge [1.0E09 Msun] 9 mstar_merge stars entering via mergers] [1.0E09 Msun] 
            # 10 v_disk rotation velocity of disk [km/s] 11 sigma_bulge velocity dispersion of bulge [km/s] 12 r_disk exponential scale radius of stars+gas disk [kpc] 
            # 13 r_bulge 3D effective radius of bulge [kpc] 14 mcold cold gas mass [1.0E09 Msun] 15 Metal_star metal mass in stars [Zsun*Msun] 16 Metal_cold metal mass in cold gas [Zsun*Msun] 
            # 17 sfr instantaneous SFR [Msun/yr] 18 sfrave10myr SFR averaged over 10 Myr [Msun/yr] 19 sfrave20myr SFR averaged over 20 Myr [Msun/yr] 
            # 20 sfrave100myr SFR averaged over 100 Myr [Msun/yr] 21 sfrave1gyr SFR averaged over 1 Gyr [Msun/yr] 22 mass_outflow_rate [Msun/yr] 23 mBH black hole mass [1.0E09 Msun] 
            # 24 maccdot accretion rate onto BH [Msun/yr] 25 maccdot_radio accretion rate in radio mode [Msun/yr] 26 tmerge time since last merger [Gyr] 
            # 27 tmajmerge time since last major merger [Gyr] 28 mu_merge mass ratio of last merger [] 29 t_sat time since galaxy became a satellite in this halo [Gyr] 
            # 30 r_fric distance from halo center [Mpc] 31 x_position x coordinate [cMpc] 32 y_position y coordinate [cMpc] 33 z_position z coordinate [cMpc] 34 vx x component of velocity [km/s] 
            # 35 vy y component of velocity [km/s] 36 vz z component of velocity [km/s]
            if name == 'scsam_inst': 
                _M, _SFR, _censat = np.loadtxt(f_name, unpack=True, skiprows=47, usecols=[7,17,3]) 
            elif name == 'scsam_10myr': 
                _M, _SFR, _censat = np.loadtxt(f_name, unpack=True, skiprows=47, usecols=[7,18,3]) 
            elif name == 'scsam_100myr': 
                _M, _SFR, _censat = np.loadtxt(f_name, unpack=True, skiprows=47, usecols=[7,20,3]) 
            elif name == 'scsam_1gyr': 
                _M, _SFR, _censat = np.loadtxt(f_name, unpack=True, skiprows=47, usecols=[7,21,3]) 
            logM = np.log10(_M) + 9.
            logSFR = np.log10(_SFR)
            w = np.ones(len(logM))
            censat = np.ones(len(logM))
            censat[_censat != 0] = 0. # cen/sat is reversed where centrals = 1

        elif name == 'nsa_combined': # NSA + SDSS combined, run through a group catalog 
            logM, logSFR = np.loadtxt(f_name, unpack=True, skiprows=2, usecols=[0,1])
            w = np.ones(len(logM)) 
            censat = np.ones(len(logM)) 
        
        elif name == 'nsa_combined_uv': # NSA + SDSS combined, run through a group catalog
            # but with UV photometry star formation rate 
            logM, logSFR = np.loadtxt(f_name, unpack=True, skiprows=2, usecols=[0,2])
            w = np.ones(len(logM)) 
            censat = np.ones(len(logM)) 

        elif name == 'tinkergroup': # SDSS DR7 Tinker group catalog 
            tink_Mstar, tink_logSSFR = np.loadtxt(f_name, unpack=True, skiprows=2, usecols=[0, 7])
            logM = np.log10(tink_Mstar)
            logSFR = tink_logSSFR + logM
            w = np.ones(len(logM))
            censat = np.ones(len(logM)) # all centrals

        elif name == 'nsa_dickey': 
            dic_logM, dic_logSFR =  np.loadtxt(f_name, unpack=True, skiprows=1, usecols=[1,5]) 
            logM = dic_logM            
            logSFR = dic_logSFR 
            w = np.ones(len(logM))
            censat = np.ones(len(logM)) # all centrals (since isolation criteria is more stringent)  
        else: 
            raise ValueError(name+' not found')

        # deal with sfr = 0 or other non finite numbers 
        sfr_zero = np.where((np.isfinite(logSFR) == False) | (logSFR < -5))
        msg_warn = '\n'.join(['', 
            ''.join(['------ ', name, ' ------']), 
            ''.join([str(len(sfr_zero[0])), ' of ', str(len(logM)), 
                ' galaxies have 0/non-finite SFRs']), 
            'logSFR of these galaxies will be -999.']) 

        if not silent: 
            warnings.warn(msg_warn) 

        logSFR[sfr_zero] = -999.

        return [logM, logSFR, w, censat]

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
        if 'illustris' in name: 
            name_cat = 'Illustris'
        elif 'eagle' in name: 
            name_cat = 'EAGLE'
        elif 'mufasa' in name: 
            name_cat = 'MUFASA'
        elif 'scsam' in name: 
            name_cat = 'Santa Cruz SAM'
        elif 'tinkergroup' in name: 
            return 'SDSS DR7 Group Catalog'
        elif 'nsa_dickey' in name: 
            return 'NSA'
        else: 
            raise ValueError

        if 'inst' in name: 
            name_tscale = 'instant.'
        elif '10myr' in name: 
            name_tscale = '$10$ Myr'
        elif '100myr' in name: 
            name_tscale = '$100$ Myr'
        elif '1gyr' in name: 
            name_tscale = '$1$ Gyr'
        else: 
            raise ValueError("specify SFR timescale") 

        return ''.join([name_cat, ' ', r'[', name_tscale, ']']) 
    
    def _SimulationVolume(self, name): 
        ''' Simulation volumes in units of Mpc^3
        '''
        if 'illustris' in name: # illustris 106.5^3 Mpc^3
            return 106.5**3
        elif 'eagle' in name: # eagle 100^3 Mpc^3
            return 100.**3
        elif 'mufasa' in name: # mufasa (50 Mpc/h)^3
            return 50.**3
    
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
