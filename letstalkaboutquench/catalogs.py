import os 
import h5py
import numpy as np 
import warnings 
from astropy import units as U
# --- Local ---
from . import util as UT
    
    
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
                'mufasa_inst': 'MUFASA_combined.dat',
                'mufasa_10myr': 'MUFASA_combined.dat',
                'mufasa_100myr': 'MUFASA_GALAXY_extra.txt', 
                'mufasa_1gyr': 'MUFASA_combined.dat',
                'scsam_inst': 'SCSAMgalprop_updatedVersion.dat', #'SCSAMgalprop.dat',  
                'scsam_10myr': 'SCSAMgalprop_updatedVersion.dat', #'SCSAMgalprop.dat', 
                'scsam_100myr': 'SCSAMgalprop_updatedVersion.dat', #'SCSAMgalprop.dat', 
                'scsam_1gyr': 'SCSAMgalprop_updatedVersion.dat', #'SCSAMgalprop.dat', 
                'nsa_combined': 'NSA_complete_SFRs.cat', 
                'nsa_combined_uv': 'NSA_complete_SFRs.cat', 
                'tinkergroup': 'tinker_SDSS_centrals_M9.7.dat',
                'nsa_dickey': 'dickey_NSA_iso_lowmass_gals.txt', 
                }
        self.catalog_list = self.catalog_dict.keys()
        
        # dictionary group catalog files 
        self.groupfind_dict = { 
                'illustris': 'illustris_groups_Mall.prob', 
                'eagle': 'EAGLE_groups_allabove1.8e8.prob', 
                'mufasa': 'MUFASA_groups.prob10.prob',
                'scsam': 'SCSAM_groups3.prob', # 'SCSAM_updated_groups.prob'
                }

    def Read(self, name, keepzeros=False, silent=False): 
        ''' Here we deal with the disparate outputs of the different catalogs and output
        log(M*), log(SFR), weight, censat

        name : (string) 
            catalog name e.g. 'santacruz1'

        keepzeros : (bool) 
            If True, keeps galaxies with zero SFR in the catalog. If False
            authomatically removes them.
        
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
            zerosfr = (_ssfr == 0.) # galaxies with zero SFRs

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
            zerosfr = (_SFR == 0.) # galaxies with zero SFRs

        elif 'mufasa' in name: # MUFASA simulation 
            # header: logM*, logSFR (isnt), logSFR (10Myr), logSFR (100Myr), logSFR (1Gyr), central/satellite
            if name == 'mufasa_inst': 
                logM, logSFR, censat = np.loadtxt(f_name, unpack=True, skiprows=1, usecols=[0,1,-1])
            elif name == 'mufasa_10myr': 
                logM, logSFR, censat = np.loadtxt(f_name, unpack=True, skiprows=1, usecols=[0,2,-1])
            elif name == 'mufasa_100myr': 
                #logM, logSFR, censat = np.loadtxt(f_name, unpack=True, skiprows=1, usecols=[0,3,-1])
                logM, logSFR, censat = np.loadtxt(f_name, unpack=True, skiprows=17, usecols=[6,8,12])
            elif name == 'mufasa_1gyr': 
                logM, logSFR, censat = np.loadtxt(f_name, unpack=True, skiprows=1, usecols=[0,4,-1])
            else: 
                raise ValueError(name+" not found")
            w = np.ones(len(logM)) # uniform weights
            zerosfr = (np.invert(np.isfinite(logSFR)) | (logSFR <= -99.))

        elif 'scsam' in name: # Santa Cruz Semi-Analytic model
            # 0 hosthaloid (long long)
            # 1 birthhaloid (long long)
            # 2 redshift
            # 3 sat_type 0= central
            # 4 mhalo total halo mass [1.0E09 Msun]
            # 5 m_strip stripped mass [1.0E09 Msun]
            # 6 rhalo halo virial radius [Mpc)]
            # 7 mstar stellar mass [1.0E09 Msun]
            # 8 mbulge stellar mass of bulge [1.0E09 Msun]
            # 9 mstar_merge stars entering via mergers] [1.0E09 Msun]
            # 10 v_disk rotation velocity of disk [km/s]
            # 11 sigma_bulge velocity dispersion of bulge [km/s]
            # 12 r_disk exponential scale radius of stars+gas disk [kpc]
            # 13 r_bulge 3D effective radius of bulge [kpc]
            # 14 mcold cold gas mass [1.0E09 Msun]
            # 15 mHI cold gas mass [1.0E09 Msun]
            # 16 mH2 cold gas mass [1.0E09 Msun]
            # 17 mHII cold gas mass [1.0E09 Msun]
            # 18 Metal_star metal mass in stars [Zsun*Msun]
            # 19 Metal_cold metal mass in cold gas [Zsun*Msun]
            # 20 sfr instantaneous SFR [Msun/yr]
            # 21 sfrave20myr SFR averaged over 20 Myr [Msun/yr]
            # 22 sfrave100myr SFR averaged over 10 Myr [Msun/yr]
            # 23 sfrave1gyr SFR averaged over 1 Gyr [Msun/yr]
            # 24 mass_outflow_rate [Msun/yr]
            # 25 mBH black hole mass [1.0E09 Msun]
            # 26 maccdot accretion rate onto BH [Msun/yr]
            # 27 maccdot_radio accretion rate in radio mode [Msun/yr]
            # 28 tmerge time since last merger [Gyr]
            # 29 tmajmerge time since last major merger [Gyr]
            # 30 mu_merge mass ratio of last merger []
            # 31 t_sat time since galaxy became a satellite in this halo [Gyr]
            # 32 r_fric distance from halo center [Mpc]
            # 33 x_position x coordinate [cMpc]
            # 34 y_position y coordinate [cMpc]
            # 35 z_position z coordinate [cMpc]
            # 36 vx x component of velocity [km/s]
            # 37 vy y component of velocity [km/s]
            # 38 vz z component of velocity [km/s]
            if name == 'scsam_inst': 
                #_M, _SFR, _censat = np.loadtxt(f_name, unpack=True, skiprows=47, usecols=[7,17,3])
                _M, _SFR, _censat = np.loadtxt(f_name, unpack=True, skiprows=39, usecols=[7,20,3]) 
            elif name == 'scsam_100myr': 
                #_M, _SFR, _censat = np.loadtxt(f_name, unpack=True, skiprows=47, usecols=[7,20,3])
                _M, _SFR, _censat = np.loadtxt(f_name, unpack=True, skiprows=39, usecols=[7,22,3]) 
            elif name == 'scsam_1gyr': 
                #_M, _SFR, _censat = np.loadtxt(f_name, unpack=True, skiprows=47, usecols=[7,21,3])
                _M, _SFR, _censat = np.loadtxt(f_name, unpack=True, skiprows=39, usecols=[7,23,3]) 
            else: 
                raise ValueError
            logM = np.log10(_M) + 9.
            logSFR = np.log10(_SFR)
            w = np.ones(len(logM))
            censat = np.ones(len(logM))
            censat[_censat != 0] = 0. # cen/sat is reversed where centrals = 1
            zerosfr = (_SFR == 0.) 

        elif name == 'nsa_combined': # NSA + SDSS combined, run through a group catalog 
            logM, logSFR = np.loadtxt(f_name, unpack=True, skiprows=2, usecols=[0,1])
            w = np.ones(len(logM)) 
            censat = np.ones(len(logM)) 
            zerosfr = np.zeros(len(logM), dtype=bool)  

        elif name == 'nsa_combined_uv': # NSA + SDSS combined, run through a group catalog
            # but with UV photometry star formation rate 
            logM, logSFR = np.loadtxt(f_name, unpack=True, skiprows=2, usecols=[0,2])
            w = np.ones(len(logM)) 
            censat = np.ones(len(logM)) 
            zerosfr = np.zeros(len(logM), dtype=bool)  

        elif name == 'tinkergroup': # SDSS DR7 Tinker group catalog 
            tink_Mstar, tink_logSSFR = np.loadtxt(f_name, unpack=True, skiprows=2, usecols=[0, 7])
            logM = np.log10(tink_Mstar)
            logSFR = tink_logSSFR + logM
            w = np.ones(len(logM))
            censat = np.ones(len(logM)) # all centrals
            zerosfr = np.zeros(len(logM), dtype=bool)  
            zerosfr[tink_logSSFR == -99.] = True
            #zerosfr[np.invert(np.isfinite(logSFR))] = True

        elif name == 'nsa_dickey': 
            dic_logM, dic_logSFR =  np.loadtxt(f_name, unpack=True, skiprows=1, usecols=[1,5]) 
            logM = dic_logM            
            logSFR = dic_logSFR 
            w = np.ones(len(logM))
            censat = np.ones(len(logM)) # all centrals (since isolation criteria is more stringent)  
            zerosfr = np.zeros(len(logM), dtype=bool)  
            zerosfr[np.invert(np.isfinite(logSFR))] = True
        else: 
            raise ValueError(name+' not found')

        self.zero_sfr = zerosfr

        if not silent: 
            # deal with sfr = 0 or other non finite numbers 
            #sfr_zero = np.where((np.isfinite(logSFR) == False) | (logSFR < -5))
            msg_warn = '\n'.join(['', 
                ''.join(['------ ', name, ' ------']), 
                ''.join([str(np.sum(zerosfr)), ' of ', str(len(logM)), 
                    ' galaxies have 0/non-finite SFRs'])]) 
            warnings.warn(msg_warn) 

        if keepzeros: 
            return [logM, logSFR, w, censat]
        else: 
            logSFR[zerosfr] = -999.
            return [logM, logSFR, w, censat]

    def GroupFinder(self, name): 
        ''' read in satellite probabilities from Jeremy's group finder
        '''
        cat_name = name.split('_')[0] # name of catalog
        if cat_name not in ['illustris', 'eagle', 'mufasa', 'scsam']: 
            raise NotImplementedError("Group finder values not yet available for catalog") 
            
        # group finder file name
        f_name = ''.join([UT.dat_dir(), 'group_finder/', self.groupfind_dict[cat_name]]) 
        psat = np.loadtxt(f_name, unpack=True, usecols=[5]) 
        assert psat.min() >= 0.
        assert psat.max() <= 1. 
        return psat
    
    def Mhalo_GroupFinder(self, name): 
        ''' read in halo mass from Jeremy's group finder
        '''
        cat_name = name.split('_')[0] # name of catalog
        if cat_name not in ['illustris', 'eagle', 'mufasa', 'scsam']: 
            raise NotImplementedError("Group finder values not yet available for catalog") 
            
        # group finder file name
        f_name = ''.join([UT.dat_dir(), 'group_finder/', self.groupfind_dict[cat_name]]) 
        mhalo = np.loadtxt(f_name, unpack=True, usecols=[6]) 
        return mhalo 
    
    def noGFSplashbacks(self, name, cut='3vir', silent=True, overwrite=False, test=False): 
        ''' Using the output from the group finder remove pure central galaxies (psat < 0.01) 
        that may potentially splashback galaxies
        '''
        cat_name = name.split('_')[0] # name of catalog
        if cat_name not in ['illustris', 'eagle', 'mufasa', 'scsam']: 
            raise NotImplementedError("Group finder values not yet available for catalog") 
        if cut == '3vir': 
            f_gfsplback = ''.join([UT.dat_dir(), cat_name, '.GF_splashback.3vir.dat']) 
        elif cut == 'geha':
            f_gfsplback = ''.join([UT.dat_dir(), cat_name, '.GF_splashback.geha.dat']) 
        
        if not os.path.isfile(f_gfsplback) or overwrite: 
            if not silent: print('constructing %s ...' % f_gfsplback)

            from astropy.cosmology import WMAP7
            from scipy.spatial import KDTree
            # read in pos 
            if cat_name == 'illustris':
                x, y, z = np.loadtxt(''.join([UT.dat_dir(), 'Illustris_pos_vel_M_all1e8Msunh_z0_for_group_finder.csv']), 
                        unpack=True, delimiter=',', usecols=[0,1,2]) # in units of kpc/h
                h_illustris = 0.704
                x /= h_illustris
                y /= h_illustris
                z /= h_illustris
            elif cat_name == 'eagle': 
                x, y, z = np.loadtxt(''.join([UT.dat_dir(), 'EAGLE_RefL0100_PosVelMstar_allabove1.8e8Msun.txt']), skiprows=1, 
                        unpack=True, usecols=[0,1,2,]) # in Mpc 
                x *= 1000. 
                y *= 1000. 
                z *= 1000. 
            elif cat_name == 'mufasa': 
                x, y, z = np.loadtxt(''.join([UT.dat_dir(), 'MUFASA_GALAXY_extra.txt']), skiprows=17,
                        unpack=True, usecols=[0,1,2]) 
            elif cat_name == 'scsam': 
                x, y, z = np.loadtxt(''.join([UT.dat_dir(), 'SCSAMgalprop_updatedVersion.dat']), skiprows=39,
                        unpack=True, usecols=[33,34,35]) 
                x *= 1000. 
                y *= 1000. 
                z *= 1000. 
            else: 
                raise NotImplementedError
            xyz = np.array([x, y, z]).T
            if not silent: 
                print('x, y, z range of %s galaxies' % cat_name) 
                print('%f < x < %f' % (x.min(), x.max()))
                print('%f < y < %f' % (y.min(), y.max()))
                print('%f < z < %f' % (z.min(), z.max()))

            # group finder file name
            f_name = ''.join([UT.dat_dir(), 'group_finder/', self.groupfind_dict[cat_name]]) 
            psat, mhalo = np.loadtxt(f_name, unpack=True, usecols=[5,6]) 
            assert psat.min() >= 0.
            assert psat.max() <= 1. 
            iscen = (psat < 0.01) 
            isnotsplash = np.ones(xyz.shape[0]).astype(bool) 
            isnotsplash[~iscen] = False 

            mhalo *= U.M_sun 

            delta_c = 200. 
            rho_c = WMAP7.critical_density(0) # critical density at z=0
            r_vir = (((3. * mhalo) / (4. * np.pi * delta_c * rho_c))**(1./3)).to(U.kpc) 

            if cut == '3vir': # 3 virial radii
                mhsort = np.argsort(mhalo.value[iscen])[::-1]

                kdt = KDTree(xyz[iscen,:])
                for i in (np.arange(xyz.shape[0])[iscen])[mhsort]: 
                    if not isnotsplash[i]:  
                        continue 
                    i_neigh = kdt.query_ball_point(xyz[i], 3*r_vir[i].value)
                    i_neigh = np.array(i_neigh) 
                    ii_neigh = np.arange(xyz.shape[0])[iscen][i_neigh]
                    assert mhalo[i] == mhalo[ii_neigh][isnotsplash[ii_neigh]].max() 
                    isnotsplash[ii_neigh] = False
                    isnotsplash[i] = True
            elif cut == 'geha': 
                logm, _, _, _ = self.Read(name)
                luminous = (logm > np.log10(2.5) + 10.) # luminous sample
                i_lum = np.arange(xyz.shape[0])[luminous]
        
                # calculate d_host 
                kdt = KDTree(xyz[luminous,:]) 
                d_host, _ = kdt.query(xyz[iscen,:], k=2) 
                iszero = (d_host[:,0] == 0)
                d_host[iszero,0] = d_host[iszero,1]
                d_host = d_host[:,0]

                rvir = np.tile(-999., xyz.shape[0])
                rvir[iscen] = d_host * U.kpc
                
                gehacut = (d_host > 1500.) 
                isnotsplash[np.arange(xyz.shape[0])[iscen][gehacut]] = False
            if not silent: 
                print('of %i group finder centrals, %i are not splashbacks' % (np.sum(iscen), np.sum(isnotsplash)))
            r_vir = r_vir.value
            datas = np.vstack([isnotsplash.astype(int), iscen.astype(int), xyz[:,0], xyz[:,1], xyz[:,2], r_vir]).T
            if cut == '3vir': 
                hdr = '\n'.join(['1 = not splashback; 0 = splashback', 'splashback, gf cen, x, y, z, r_vir']) 
            elif cut == 'geha': 
                hdr = '\n'.join(['1 = not splashback; 0 = splashback', 'splashback, gf cen, x, y, z, d_host']) 
            np.savetxt(f_gfsplback, datas, header=hdr, delimiter='\t', fmt="%i %i %f %f %f %f") 
        else: 
            datas = np.loadtxt(f_gfsplback, skiprows=2, unpack=True) 
            isnotsplash = datas[0].astype(int)
            isnotsplash = isnotsplash.astype(bool)
            iscen = datas[1].astype(int) 
            iscen = iscen.astype(bool)
            x = datas[2]
            y = datas[3]
            z = datas[4]
            xyz = np.array([x, y, z]).T
            r_vir = datas[5] 

        if test: 
            return isnotsplash, xyz, r_vir
        return isnotsplash 

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
            name_cat = 'SC-SAM'
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
   
    def _SFR_resolution(self, name): 
        ''' SFR resolution for simulations 
        '''
        if name not in ['illustris_100myr', 'eagle_100myr', 'mufasa_100myr']: 
            raise ValueError("other catalogs do not have well defined SFR resolutions")  
        elif name == 'illustris_100myr': 
            dsfr = 0.0126
        elif name == 'eagle_100myr': 
            dsfr = 0.018
        elif name == 'mufasa_100myr': 
            dsfr = 0.182
        return dsfr

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


def _mufasa_groupfinder_pre(): 
    ''' pre-process MUFASA groupfinder data  
    '''
    # read in group finder data for MUFASA
    f_name = ''.join([UT.dat_dir(), 'group_finder/', 'MUFASA_groups_all.prob']) 
    prob_str = np.loadtxt(f_name, unpack=True, usecols=[0], dtype='S') 
    data = np.loadtxt(f_name, unpack=True, usecols=range(1,14)) 
    
    # only keep ones with first column == PROB10  
    prob10 = (prob_str == 'PROB10') 
    data_out = [np.zeros(np.sum(prob10))] 
    for i in range(len(data)): 
        data_out.append(data[i][prob10]) 

    # write it out 
    out_name = ''.join([UT.dat_dir(), 'group_finder/', 'MUFASA_groups.prob10.prob']) 
    np.savetxt(out_name, np.array(data_out).T) 
    return None


if __name__=='__main__': 
    _mufasa_groupfinder_pre()
    #Build_Illustris_SFH()
    '''
    Cat = Catalog()
    
    for cata in ['santacruz1', 'santacruz2', 'tinkergroup', 'illustris1', 'illustris2', 'nsa_dickey', 'mufasa']: 
        logM, logSFR, w = Cat.Read(cata)
        print cata
        print Cat.CatalogLabel(cata)
        print logM[:10], logSFR[:10], w[:10]
        print '------------------------------------------------------------'
    '''
