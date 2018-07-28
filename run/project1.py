'''

all the calculations for project 1 paper 1. 


'''
import pickle
import numpy as np 
import scipy as sp
# -- Local --
from letstalkaboutquench import util as UT
from letstalkaboutquench import catalogs as Cats
from letstalkaboutquench import galprop as Gprop
from letstalkaboutquench.fstarforms import fstarforms


illustris_mmin = 8.4
eagle_mmin = 8.4
mufasa_mmin = 9.2
scsam_mmin = 8.8
tinker_mmin = 9.7
dickey_mmax = 9.7


def dSFS(name): 
    ''' calculate dSFS for specified catalog and SF timescale and 
    save M*, SFR, and dSFS of group finder centrals to file 
    ''' 
    # read in catalog
    Cat = Cats.Catalog()
    logMstar, logSFR, weight, censat = Cat.Read(name)
    
    # group finder definition of centrals 
    if name not in ['nsa_dickey', 'tinkergroup']: 
        psat = Cat.GroupFinder(name) 
        iscen = (psat < 0.01) 
    else: 
        iscen = (censat == 1) 
    nonzero = (~Cat.zero_sfr) 
    
    # stellar mass limit due to resolution effects
    _, mlim = _mlim_fit(name, logMstar, iscen) 
    
    # read in GMM SFS fits 
    f_gmm = ''.join([UT.dat_dir(), 'paper1/', 'gmmSFSfit.', name, '.gfcentral.mlim.p'])
    fSFMS = pickle.load(open(f_gmm, 'rb')) 
    
    fit_cut = (iscen & nonzero & mlim)  # sample cut for SFS fitting
    
    # dSFS from interpolating and extrapolating the well defined 
    # GMM SFS fits  
    dsfs0 = fSFMS.d_SFS(
            logMstar[fit_cut], logSFR[fit_cut],
            method='interpexterp',
            err_thresh=0.2,  # this is the uncertainty threshold but for z=1 is irrelevant
            silent=False) 
    dsfs_all0 = np.tile(-999., len(logMstar)) 
    dsfs_all0[fit_cut] = dsfs0
    
    # dSFS from fitting a power-law to the GMM SFS fits 
    _ = fSFMS.powerlaw(logMfid=10.5) 
    dsfs1 = fSFMS.d_SFS(
            logMstar[fit_cut], logSFR[fit_cut],
            method='powerlaw',
            silent=False) 
    dsfs_all1 = np.tile(-999., len(logMstar)) 
    dsfs_all1[fit_cut] = dsfs1
    
    # compile data
    sample_cut = (iscen & mlim) 
    sample = np.array([logMstar[sample_cut], logSFR[sample_cut], 
        dsfs_all1[sample_cut], dsfs_all0[sample_cut]]).T

    # save to file 
    f_out = ''.join([UT.dat_dir(), 'paper1/dsfs.', name, '.gfcentrals.mlim.dat']) 

    hdr = '\n'.join(['distance to the SF sequence (d_SFS=-999. for zero SFR galaxies)', 
        'SFMS fit choices: dlogm=0.2, SSFR_cut=-11., Nbin_thresh=100', 
        'dSFS choices: logM_fid=10.5, interp=True, extrap=True, err_thresh=0.2', 
        'logM* logSFR, dSFS_powerlaw, dSFS_interp']) 
    np.savetxt(f_out, sample, header=hdr, fmt='%f, %f, %f, %f') 
    return None 


def gmmSFSfits(name): 
    ''' GMM SFS fits to the specified simulation+SFR timescale. Instead of 
    re-running the GMM fitting every time I generate a figure, the output
    from this function serves as the final result. 
    '''
    # read in catalog
    Cat = Cats.Catalog()
    logMstar, logSFR, weight, censat = Cat.Read(name)
    
    # group finder definition of centrals 
    if name not in ['nsa_dickey', 'tinkergroup']: 
        psat = Cat.GroupFinder(name) 
        iscen = (psat < 0.01) 
    else: 
        iscen = (censat == 1) 
    nonzero = (~Cat.zero_sfr) 
    
    # stellar mass range for SFMS fit and stellar mass limit  
    fitrange, mlim = _mlim_fit(name, logMstar, (iscen & nonzero)) 

    # sample cut for SFS fitting
    # group finder centrals & non zero SFRs & stellar mass limit 
    fit_cut = (iscen & nonzero & mlim)      
    fSFMS = fstarforms()
    fit_logm, fit_logsfr, fit_sig_logsfr = fSFMS.fit(
            logMstar[fit_cut], logSFR[fit_cut],
            method='gaussmix', 
            fit_range=fitrange, 
            dlogm=0.2,              # stellar mass bins of 0.2 dex
            SSFR_cut=-11., 
            Nbin_thresh=100,        # at least 100 galaxies in bin 
            fit_error='bootstrap',  # uncertainty estimate method 
            n_bootstrap=100)        # number of bootstrap bins
    
    f_out = ''.join([UT.dat_dir(), 'paper1/', 'gmmSFSfit.', name, '.gfcentral.mlim.p'])
    pickle.dump(fSFMS, open(f_out, 'wb'))
    return None 


def gmmSFSfits_lowthresh(name): 
    ''' GMM SFS fits to the specified simulation+SFR timescale with
    a particularly lower Nbin_thresh. This is specifically for the
    GMM component figures 
    '''
    # read in catalog
    Cat = Cats.Catalog()
    logMstar, logSFR, weight, censat = Cat.Read(name)
    
    # group finder definition of centrals 
    if name not in ['nsa_dickey', 'tinkergroup']: 
        psat = Cat.GroupFinder(name) 
        iscen = (psat < 0.01) 
    else: 
        iscen = (censat == 1) 
    nonzero = (~Cat.zero_sfr) 
    
    # stellar mass range for SFMS fit and stellar mass limit  
    fitrange, mlim = _mlim_fit(name, logMstar, (iscen & nonzero)) 

    # sample cut for SFS fitting
    # group finder centrals & non zero SFRs & stellar mass limit 
    fit_cut = (iscen & nonzero & mlim)      
    fSFMS = fstarforms()
    fit_logm, fit_logsfr, fit_sig_logsfr = fSFMS.fit(
            logMstar[fit_cut], logSFR[fit_cut],
            method='gaussmix', 
            fit_range=fitrange, 
            dlogm=0.2,              # stellar mass bins of 0.2 dex
            SSFR_cut=-11., 
            Nbin_thresh=10,         # only require 10 galaxies in bin 
            fit_error='bootstrap',  # uncertainty estimate method 
            n_bootstrap=100)        # number of bootstrap bins
    
    f_out = ''.join([UT.dat_dir(), 'paper1/', 'gmmSFSfit.', name, '.gfcentral.lowNbinthresh.mlim.p'])
    pickle.dump(fSFMS, open(f_out, 'wb'))
    return None 


def gmmSFSfits_nosplashbacks(name, cut='3vir'): 
    ''' same as above but only for group finder cenrals galaxies that are not 
    splashbacks. 
    '''
    if name in ['nsa_dickey', 'tinkergroup']: 
        # only for simulations 
        raise ValueError
    # read in catalog
    Cat = Cats.Catalog()
    logMstar, logSFR, weight, censat = Cat.Read(name)
    nonzero = (~Cat.zero_sfr) 
    
    # group finder definition of centrals with no splashbacks
    iscen = Cat.noGFSplashbacks(name, cut=cut) 
    
    # stellar mass range for SFMS fit and stellar mass limit  
    fitrange, mlim = _mlim_fit(name, logMstar, (iscen & nonzero)) 

    # sample cut for SFS fitting
    # group finder centrals & non zero SFRs & stellar mass limit 
    fit_cut = (iscen & nonzero & mlim)      
    fSFMS = fstarforms()
    fit_logm, fit_logsfr, fit_sig_logsfr = fSFMS.fit(
            logMstar[fit_cut], logSFR[fit_cut],
            method='gaussmix', 
            fit_range=fitrange, 
            dlogm=0.2,              # stellar mass bins of 0.2 dex
            SSFR_cut=-11., 
            Nbin_thresh=100,        # at least 100 galaxies in bin 
            fit_error='bootstrap',  # uncertainty estimate method 
            n_bootstrap=100)        # number of bootstrap bins
    
    f_out = ''.join([UT.dat_dir(), 'paper1/', 
        'gmmSFSfit.', name, '.gfcentral.nosplbacks.', cut, '.mlim.p'])
    pickle.dump(fSFMS, open(f_out, 'wb'))
    return None 


def gmmSFSfits_morecomp(name): 
    ''' same as above but the SFS fitting is no longer restricted to 3 
    components!
    '''
    # read in catalog
    Cat = Cats.Catalog()
    logMstar, logSFR, weight, censat = Cat.Read(name)
    
    # group finder definition of centrals 
    if name not in ['nsa_dickey', 'tinkergroup']: 
        psat = Cat.GroupFinder(name) 
        iscen = (psat < 0.01) 
    else: 
        iscen = (censat == 1) 
    nonzero = (~Cat.zero_sfr) 
    
    # stellar mass range for SFMS fit and stellar mass limit  
    fitrange, mlim = _mlim_fit(name, logMstar, (iscen & nonzero)) 

    # sample cut for SFS fitting
    # group finder centrals & non zero SFRs & stellar mass limit 
    fit_cut = (iscen & nonzero & mlim)      
    fSFMS = fstarforms()
    fit_logm, fit_logsfr, fit_sig_logsfr = fSFMS.fit(
            logMstar[fit_cut], logSFR[fit_cut],
            method='gaussmix', 
            fit_range=fitrange, 
            dlogm=0.2,              # stellar mass bins of 0.2 dex
            SSFR_cut=-11., 
            max_comp=6,             # ** max is 6 components now ** 
            Nbin_thresh=100,        # at least 100 galaxies in bin 
            fit_error='bootstrap',  # uncertainty estimate method 
            n_bootstrap=100)        # number of bootstrap bins
    
    f_out = ''.join([UT.dat_dir(), 'paper1/', 
        'gmmSFSfit.', name, '.gfcentral.morecomp.mlim.p'])
    pickle.dump(fSFMS, open(f_out, 'wb'))
    return None 


def _gmmSFSfit_frankenSDSS(): 
    ''' GMM SFS fit to the unholy combination of Claire's NSA catalog
    with Jeremy's SDSS catalog. This is for reference only! 
    '''
    # read in catalog
    Cat = Cats.Catalog()
    # nsa catalogs 
    logM_nsa, logSFR_nsa, _, censat_nsa = Cat.Read('nsa_dickey')
    iscen_nsa = (censat_nsa == 1) 
    nonzero_nsa = ~Cat.zero_sfr
    # stellar mass range for SFMS fit and stellar mass limit  
    _, mlim_nsa = _mlim_fit('nsa_dickey', logM_nsa, (iscen_nsa & nonzero_nsa)) 

    logM_sdss, logSFR_sdss, _, censat_sdss = Cat.Read('tinkergroup')
    iscen_sdss = (censat_sdss == 1) 
    nonzero_sdss = ~Cat.zero_sfr
    _, mlim_sdss = _mlim_fit('tinkergroup', logM_sdss, (iscen_sdss & nonzero_sdss)) 

    logM_all = np.concatenate([
        logM_nsa[iscen_nsa & nonzero_nsa & mlim_nsa], 
        logM_sdss[iscen_sdss & nonzero_sdss & mlim_sdss]]) 
    logSFR_all = np.concatenate([
        logSFR_nsa[iscen_nsa & nonzero_nsa & mlim_nsa], 
        logSFR_sdss[iscen_sdss & nonzero_sdss & mlim_sdss]]) 

    fSFMS = fstarforms() # fit the SFMS  
    _ = fSFMS.fit(logM_all, logSFR_all, 
            method='gaussmix', 
            fit_range=None, 
            dlogm=0.2,              # stellar mass bins of 0.2 dex
            SSFR_cut=-11., 
            Nbin_thresh=100,        # at least 100 galaxies in bin 
            fit_error='bootstrap',  # uncertainty estimate method 
            n_bootstrap=100)        # number of bootstrap bins
    
    f_out = ''.join([UT.dat_dir(), 'paper1/', 'gmmSFSfit.franken_sdss.gfcentral.mlim.p'])
    pickle.dump(fSFMS, open(f_out, 'wb'))
    return None 


def gmmSFSpowerlaw(logMfid=10.5): 
    ''' power-law fits to the GMM SFS fits to the specified catalog + SFR timescale
    '''
    tscales = ['inst', '100myr'] # tscales 
    sims_list = ['illustris', 'eagle', 'mufasa', 'scsam'] # simulations 

    f_table = open(''.join([UT.dat_dir(), 'paper1/', 'SFMS_powerlawfit.txt']), 'w') 
    f_table.write("# best-fit (maximum likelihood) paremters for power-law fits to SFMS \n")
    f_table.write("# log SFR_sfs = m x (log M* - "+str(logMfid)+") + b \n")
    for i_t, tscale in enumerate(tscales): 
        for i_c, cc in enumerate(sims_list): 
            name = '_'.join([cc, tscale]) 
            f_gmm = ''.join([UT.dat_dir(), 'paper1/', 'gmmSFSfit.', name, '.gfcentral.mlim.p'])
            fSFMS = pickle.load(open(f_gmm, 'rb')) 
            # power-law fit of the SFMS fit 
            _ = fSFMS.powerlaw(logMfid=logMfid)
            f_table.write('--- %s --- \n' % name) 
            f_table.write('power-law m: %f \n' % fSFMS._powerlaw_m) 
            f_table.write('power-law b: %f \n' % fSFMS._powerlaw_c) 
            if 'mufasa' in name: 
                _ = fSFMS.powerlaw(logMfid=logMfid, mlim=[8., 10.5])
                f_table.write('--- %s logM* < 10.5--- \n' % name) 
                f_table.write('power-law m: %f \n' % fSFMS._powerlaw_m) 
                f_table.write('power-law b: %f \n' % fSFMS._powerlaw_c) 
    
    # **for reference** fit the franken-SDSS sample 
    f_gmm = ''.join([UT.dat_dir(), 'paper1/', 'gmmSFSfit.franken_sdss.gfcentral.mlim.p'])
    fSFMS = pickle.load(open(f_gmm, 'rb')) 
    # power-law fit of the SFMS fit 
    _ = fSFMS.powerlaw(logMfid=10.5) 
    f_table.write('--- SDSS + NSA --- \n') 
    f_table.write('power-law m: %f \n' % fSFMS._powerlaw_m) 
    f_table.write('power-law b: %f \n' % fSFMS._powerlaw_c) 
    f_table.close() 
    return None 


def _mlim_fit(name, logMstar, cut): 
    ''' return fitting range and stellar mass limit
    '''
    # stellar mass limit due to resolution effects
    if 'scsam' in name: 
        fitrange = [scsam_mmin, np.ceil(logMstar[cut].max()/0.2)*0.2]
        mlim = (logMstar > scsam_mmin)
    elif name == 'illustris_100myr': 
        fitrange = [illustris_mmin, np.ceil(logMstar[cut].max()/0.2)*0.2]
        mlim = (logMstar > illustris_mmin)
    elif name == 'eagle_100myr': 
        fitrange = [eagle_mmin, np.ceil(logMstar[cut].max()/0.2)*0.2]
        mlim = (logMstar > eagle_mmin)
    elif name == 'mufasa_100myr': 
        fitrange = [mufasa_mmin, np.ceil(logMstar[cut].max()/0.2)*0.2]
        mlim = (logMstar > mufasa_mmin)
    elif name == 'nsa_dickey': 
        fitrange = [np.floor(logMstar[cut].min()/0.2)*0.2, dickey_mmax]
        mlim = (logMstar < dickey_mmax)
    elif name == 'tinkergroup': 
        fitrange = [tinker_mmin, np.ceil(logMstar[cut].max()/0.2)*0.2]
        mlim = (logMstar > tinker_mmin)
    else: 
        fitrange = None
        mlim = np.ones(len(logMstar)).astype(bool) 
    return fitrange, mlim 


if __name__=="__main__": 
    for t in ['inst', '100myr']: 
        for name in ['illustris', 'eagle', 'mufasa', 'scsam']:
            #gmmSFSfits(name+'_'+t)
            #dSFS(name+'_'+t) 
            #gmmSFSfits_lowthresh(name+'_'+t)
            #gmmSFSfits_nosplashbacks(name+'_'+t, cut='geha')
            gmmSFSfits_morecomp(name+'_'+t)
    for name in ['nsa_dickey', 'tinkergroup']: 
        #gmmSFSfits(name)
        #gmmSFSfits_lowthresh(name)
        gmmSFSfits_morecomp(name)
        #dSFS(name) 
    #_gmmSFSfit_frankenSDSS()
    #gmmSFSpowerlaw()
