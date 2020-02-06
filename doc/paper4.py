'''

IQ paper 4 --- High z galaxies


'''
import os
import pickle
import numpy as np 
import corner as DFM 
from scipy.stats import multivariate_normal as MNorm
# -- letstalkaboutquench --
from letstalkaboutquench import util as UT
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

# some build in values 
zlo = [0.5, 1., 1.4, 1.8, 2.2, 2.6]
zhi = [1., 1.4, 1.8, 2.2, 2.6, 3.0]
dir_fig = os.path.join(UT.doc_dir(), 'highz', 'figs') 

def highzSFSfit(name, i_z, censat='all', noise=False, seed=1, dlogM=0.4, slope_prior=[0., 2.], overwrite=False): 
    """ fit SFS to the SFR-M* relation of `name` catalog `i_z` redshift bin 
    
    :param name: 
        string specifying catalog name 

    :param i_z: 
        int specifying redshift bin 

    :param censat:
        If 'all', include both centrals and satellites. 
        If 'centrals', include centrals only 
        If 'satellites', include satellites only 
    
    :param noise: 
        If False, return original M* and SFRs
        If True, return M* and SFR with CANDELS level noise added by 
        add_uncertainty function 
        
    :param seed: 
        int specifying the random seed of noise 
    """
    f_dat = fHighz(name, i_z, censat=censat, noise=noise, seed=seed)
    f_sfs =  os.path.join(os.path.dirname(f_dat), 
                          'SFSfit_%s.dlogM%.1f.slope_prior_%.1f_%.1f.p' % 
                          (os.path.basename(f_dat).strip('.txt'), dlogM, slope_prior[0], slope_prior[1]))
    
    if os.path.isfile(f_sfs) and not overwrite: 
        print(f_sfs)
        fSFS = pickle.load(open(f_sfs, 'rb'))
    else: 
        logm, logsfr, cs, notzero = readHighz(name, i_z, censat=censat, noise=noise, seed=seed)
        cut = (cs & notzero) 

        # fit the SFMS
        fSFS = sFS(fit_range=[8.5, 12.0]) # stellar mass range
        sfs_fit = fSFS.fit(logm[cut], logsfr[cut], 
                method='gaussmix',      # Gaussian Mixture Model fitting 
                dlogm = dlogM,          # stellar mass bins of 0.4 dex
                slope_prior = slope_prior, 
                Nbin_thresh=100,        # at least 100 galaxies in bin 
                error_method='bootstrap',  # uncertainty estimate method 
                n_bootstrap=100)        # number of bootstrap bins
        pickle.dump(fSFS, open(f_sfs, 'wb'))
    return fSFS


def readHighz(name, i_z, censat='all', noise=False, seed=1): 
    """ write M* and SFR data from the iz redshift bin of `name` catalog for 
    convenient access 
    
    :param name: 
        string specifying catalog name 

    :param i_z: 
        int specifying redshift bin 

    :param censat:
        If 'all', include both centrals and satellites. 
        If 'centrals', include centrals only 
        If 'satellites', include satellites only 
    
    :param noise: 
        If False, return original M* and SFRs
        If True, return M* and SFR with CANDELS level noise added by 
        add_uncertainty function 
        
    :param seed: 
        int specifying the random seed of noise 

    :returns logms, logsfr, cs_cut, notzero: 
        - logms: log M* 
        - logsfr: log SFR 
        - cs_cut: boolean array specifying central or satellites 
        - notzero: boolean array specifying SFR != 0 
    """
    name = name.lower() 
    if not noise: 
        f_data = fHighz(name, i_z, censat=None, noise=False) # data file name 

        if 'candels' in name: 
            if censat != 'all': 
                raise ValueError('no central/satellite classification for CANDELS data') 
            _, logms, logsfr = np.loadtxt(f_data, skiprows=2, unpack=True) # z, M*, SFR
            if name != 'candels_cami': 
                notzero = np.isfinite(logsfr)
            else: 
                notzero = np.isfinite(logsfr) & (logsfr > -3.) 

        elif 'illustris' in name: 
            ill_ind = {'illustris_10myr': 1, 'illustris_100myr': 3, 'illustris_1gyr': 2} 
            ms, sfr, cs = np.loadtxt(f_data, skiprows=2, unpack=True, 
                                     usecols=[0, ill_ind[name], 4]) # M*, SFR 10Myr, SFR 1Gyr, SFR 100Myr, cen/sat
            logms = np.log10(ms) 
            logsfr = np.log10(sfr) 
            notzero = (sfr != 0.)

        elif name == 'tng':
            logms, logsfr, cs = np.loadtxt(f_data, skiprows=2, unpack=True) # log M*, log SFR, cen/sat
            notzero = np.isfinite(logsfr)

        elif name == 'eagle': 
            ms, sfr, cs = np.loadtxt(f_data, skiprows=2, unpack=True) # M*, SFR instantaneous
            logms = ms
            logsfr = np.log10(sfr) 
            notzero = (sfr != 0.)

        elif 'sam-light' in name: 
            _, ms, sfr, cs = np.loadtxt(f_data, skiprows=2, unpack=True) 
            logms = np.log10(ms)
            logsfr = np.log10(sfr)
            notzero = (sfr != 0.)  

        elif name == 'simba': 
            ms, sfr, cs = np.loadtxt(f_data, skiprows=2, unpack=True) # M*, SFR instantaneous
            logms = np.log10(ms)
            logsfr = np.log10(sfr) 
            notzero = (sfr != 0.)
        else: 
            raise NotImplementedError('%s not implemented' % name) 
    else: 
        f_data = fHighz(name, i_z, censat=None, noise=True, seed=seed) 
        if not os.path.isfile(f_data): add_uncertainty(name, i_z, seed=seed) # write files 

        logms, logsfr, cs = np.loadtxt(f_data, unpack=True, usecols=[0,1,2], skiprows=2) 
        notzero = np.isfinite(logsfr)
    
    # central/satellite cut 
    if censat == 'centrals': cs_cut = (cs == 1) 
    elif censat == 'satellites': cs_cut = (cs == 0) 
    elif censat == 'all': cs_cut = np.ones(len(logms)).astype(bool) 
    else: raise ValueError("censat = ", censat)  
    return logms, logsfr, cs_cut, notzero


def add_uncertainty(name, i_z, seed=1): 
    ''' add in measurement uncertainties for SFR and M*. sigma_logSFR = 0.33 dex and 
    sigma_logM* = 0.07 dex. This is done in the simplest way possible --- i.e. add 
    gaussian noise 
    '''
    assert name.lower() != 'candels', "don't add uncertainties to candels" 
    sig_logsfr = 0.33
    sig_logms = 0.07

    np.random.seed(seed) # random seed  

    # read in log M*, log SFR, central/satellite, SFR != 0 
    logms, logsfr, cs, notzero = readHighz(name, i_z, censat='centrals')
    censat = np.zeros(len(logms))
    censat[cs] = 1

    dlogsfr = sig_logsfr * np.random.randn(len(logsfr))
    dlogms = sig_logms * np.random.randn(len(logms))
    if np.sum(~notzero) == 0:
        logsfr_new = logsfr + dlogsfr
        logms_new = logms + dlogms
    else: 
        # for now, add noise to non-zero SFRs but leaver zero SFRs alone. 
        logsfr_new = np.zeros(len(logsfr))
        logsfr_new[notzero] = logsfr[notzero] + dlogsfr[notzero]
        logsfr_new[~notzero] = logsfr[~notzero]
        logms_new = logms + dlogms

    # save to file 
    fnew = fHighz(name, i_z, censat=None, noise=True, seed=seed) 
    hdr = 'Gaussian noise with sigma_logSFR = 0.33 dex and sigma_M* = 0.07 dex added to data\n logM*, logSFR' 
    np.savetxt(fnew, np.array([logms_new, logsfr_new, censat]).T, delimiter='\t', fmt='%.5f %.5f %i', header=hdr) 
    return None 


def fHighz(name, i_z, censat='all', noise=False, seed=1): 
    """ High z project file names
    """
    dat_dir = ''.join([UT.dat_dir(), 'highz/'])
    if name == 'candels': 
        f_data = ''.join([dat_dir, 'CANDELS/CANDELS_Iyer_z', str(i_z), '.txt']) 
    elif name == 'candels_goods': 
        f_data = ''.join([dat_dir, 'CANDELS/CANDELS_GOODS_Iyer_z', str(i_z), '.txt']) 
    elif name ==  'candels_cami': 
        f_data = ''.join([dat_dir, 'CANDELS/CANDELS_cami_z%s.txt' % i_z]) 
    elif 'illustris' in name: 
        f_data = ''.join([dat_dir, 'Illustris/Illustris_z', str(i_z), '.txt']) 
    elif name == 'tng': 
        f_data = ''.join([dat_dir, 'Illustris/IllustrisTNG_z', str(i_z), '.txt']) 
    elif name == 'eagle': 
        f_data = ''.join([dat_dir, 'EAGLE/EAGLE_z', str(i_z), '.txt']) 
    elif name == 'sam-light-full': # SAM light cone full 
        f_data = ''.join([dat_dir, 'SAM_lightcone/SAMfull_z', str(i_z), '.txt'])
    elif name == 'sam-light-slice': # SAM light cone dz=0.01 slice around the median redshift 
        f_data = ''.join([dat_dir, 'SAM_lightcone/SAMslice_z', str(i_z), '.txt'])
    elif name == 'simba': 
        f_data = ''.join([dat_dir, 'SIMBA/SIMBA_z', str(i_z), '.txt'])
    else: 
        raise NotImplementedError
    if censat is not None: f_data = f_data.replace('.txt', '.%s.txt' % censat) 
    if noise: 
        f_data = f_data.replace('.txt', '.wnoise.seed%i.txt' % seed)
    return f_data


def dSFS(name, censat='all', noise=False, seed=1, dlogM=0.4, slope_prior=[0., 2.], method='interpexterp'): 
    ''' calculate dSFS for specified high z catalog 
    ''' 
    zlo = [0.5, 1., 1.4, 1.8, 2.2, 2.6]
    zhi = [1., 1.4, 1.8, 2.2, 2.6, 3.0]
    logms, logsfrs = [], [] 
    sfs_fits, dsfss = [], [] 

    for i_z in range(len(zlo)): 
    
        # read in logM and logSFR along with bestfits
        if name != 'candels': 
            logm, logsfr, cs, notzero = readHighz(name, i_z+1, censat=censat, noise=noise, seed=seed)
            fSFS = highzSFSfit(name, i_z+1, censat=censat, noise=noise, seed=seed, dlogM=dlogM, slope_prior=slope_prior) # fit the SFSs
        else: 
            logm, logsfr, cs, notzero = readHighz(name, i_z+1, censat='all', noise=False)
            fSFS = highzSFSfit(name, i_z+1, censat='all', noise=False, dlogM=dlogM, slope_prior=slope_prior) # fit the SFSs
        cut = (cs & notzero) 

        logms.append(logm[cut])
        logsfrs.append(logsfr[cut])
        sfs_fits.append([fSFS._fit_logm, fSFS._fit_logsfr, fSFS._fit_err_logssfr]) 

        # calculate dSFS 
        if method == 'powerlaw': 
            _ = fSFS.powerlaw(logMfid=10.5) 
            dsfs = fSFS.d_SFS(logm[cut], logsfr[cut], method=method, silent=False) 
        elif method == 'broken_powerlaw': 
            _ = fSFS.broken_powerlaw(logMturnover=10.) 
            dsfs = fSFS.d_SFS(logm[cut], logsfr[cut], method=method, silent=False) 
        elif method == 'interpexterp': 
            dsfs = fSFS.d_SFS(logm[cut], logsfr[cut], method=method, err_thresh=0.2, silent=False) 
        dsfss.append(dsfs) 

    # SFMS overplotted ontop of SFR--M* relation 
    fig = plt.figure(figsize=(12,8))
    bkgd = fig.add_subplot(111, frameon=False)
    for i_z in range(len(zlo)): 
        sub = fig.add_subplot(2,3,i_z+1) 
        sub.scatter(logms[i_z], logsfrs[i_z], color='k', s=1) 
        sub.errorbar(sfs_fits[i_z][0], sfs_fits[i_z][1], sfs_fits[i_z][2], fmt='.C0')
        is_quiescent = ((dsfss[i_z] < -1.) & (dsfss[i_z] != -999.)) 
        sub.scatter(logms[i_z][is_quiescent], logsfrs[i_z][is_quiescent], color='C1', s=1) 
        sub.set_xlim([8.5, 12.]) 
        sub.set_ylim([-3., 4.]) 
        sub.text(0.95, 0.05, '$'+str(zlo[i_z])+'< z <'+str(zhi[i_z])+'$', 
                ha='right', va='bottom', transform=sub.transAxes, fontsize=20)
        if i_z == 0: 
            sub.text(0.05, 0.95, ' '.join(name.upper().split('_')),
                    ha='left', va='top', transform=sub.transAxes, fontsize=20)

    bkgd.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', labelpad=15, fontsize=25) 
    bkgd.set_ylabel(r'log ( SFR $[M_\odot \, yr^{-1}]$ )', labelpad=15, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    fig.subplots_adjust(wspace=0.2, hspace=0.15)

    ffig = os.path.join(dir_fig, 'dsfs.%s_%s.%s.dlogM%.1f.slope_prior%.1f_%.1f.png' % (name.lower(), censat, method, dlogM, slope_prior[0], slope_prior[1]))
    if noise: ffig = ffig.replace('.png', '_wnoise.png') 
    fig.savefig(ffig, bbox_inches='tight')
    return None 


def fQ_dSFS(name, censat='all', noise=False, seed=1, dlogM=0.4, slope_prior=[0., 2.], method='interpexterp', dSFS_limit=1.): 
    ''' calculate quiescent fraction based on dSFS for specified high z catalog 
    ''' 
    zlo = [0.5, 1., 1.4, 1.8, 2.2, 2.6]
    zhi = [1., 1.4, 1.8, 2.2, 2.6, 3.0]

    fqs = []
    for i_z in range(len(zlo)): 
        # read in logM and logSFR along with bestfits
        if name != 'candels': 
            logm, logsfr, cs, notzero = readHighz(name, i_z+1, censat=censat, noise=noise, seed=seed)
            fSFS = highzSFSfit(name, i_z+1, censat=censat, noise=noise, seed=seed, dlogM=dlogM, slope_prior=slope_prior) # fit the SFSs
        else: 
            logm, logsfr, cs, notzero = readHighz(name, i_z+1, censat='all', noise=False)
            fSFS = highzSFSfit(name, i_z+1, censat='all', noise=False, dlogM=dlogM, slope_prior=slope_prior) # fit the SFSs
        cut = (cs & notzero) 

        # calculate dSFS 
        if method == 'powerlaw': 
            _ = fSFS.powerlaw(logMfid=10.5) 
            dsfs = fSFS.d_SFS(logm[cut], logsfr[cut], method=method, silent=False) 
        elif method == 'broken_powerlaw': 
            _ = fSFS.broken_powerlaw(logMturnover=10., mlim=fSFS._has_nbinthresh) 
            dsfs = fSFS.d_SFS(logm[cut], logsfr[cut], method=method, silent=False) 
        elif method == 'interpexterp': 
            dsfs = fSFS.d_SFS(logm[cut], logsfr[cut], method=method, err_thresh=0.2, silent=False) 

        # calculate fQ 
        quenched = (dsfs < -1.*dSFS_limit)

        mbins = np.linspace(8., 12., 17) 
        m_fq, fq = [], [] 
        for i_m in range(len(mbins)-1):  
            inmbin = (logm[cut] > mbins[i_m]) & (logm[cut] <= mbins[i_m+1]) 
            if np.sum(inmbin) < 20.: continue 

            m_fq.append(0.5 * (mbins[i_m] + mbins[i_m+1])) 
            fq.append(float(np.sum(inmbin & quenched))/float(np.sum(inmbin)))
        print(m_fq)
        print(fq)
        fqs.append(np.array([m_fq, fq]))

    # SFMS overplotted ontop of SFR--M* relation 
    fig = plt.figure(figsize=(5,5))
    sub = fig.add_subplot(111) 
    for i_z in range(len(zlo)): 
        sub.plot(fqs[i_z][0], fqs[i_z][1], c='C%i' % i_z)
    sub.set_xlim(8.5, 12.) 
    sub.set_ylim(0., 1.) 
    sub.text(0.05, 0.95, ' '.join(name.upper().split('_')),
            ha='left', va='top', transform=sub.transAxes, fontsize=20)

    sub.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', fontsize=25) 
    sub.set_ylabel(r'$f_Q$ defined using $d_{\rm SFS}$', fontsize=25) 

    ffig = os.path.join(dir_fig, 
            'fQ_dsfs.%s_%s.%s.dlogM%.1f.slope_prior%.1f_%.1f.dSFSlim_%.1f.png' % (name.lower(), censat, method, dlogM, slope_prior[0], slope_prior[1], dSFS_limit))
    if noise: ffig = ffig.replace('.png', '_wnoise.png') 
    fig.savefig(ffig, bbox_inches='tight')
    return None 


def fQ_dSFS_comparison(censat='all', noise=False, seed=1, dlogM=0.4, slope_prior=[0., 2.], method='interpexterp', dSFS_limit=1.): 
    ''' calculate quiescent fraction based on dSFS for specified high z catalog 
    ''' 
    names = ['sam-light-slice', 'eagle', 'illustris_100myr', 'tng', 'simba', 'candels']
    lbls = ['SC-SAM', 'EAGLE', 'Illustris', 'Illustris TNG', 'SIMBA', 'CANDELS'] 
    
    fig = plt.figure(figsize=(12,8))
    bkgd = fig.add_subplot(111, frameon=False)
    for i_z in range(len(zlo)): 
        sub = fig.add_subplot(2,3,i_z+1) 

        plts = []
        for i_n, name in enumerate(names):  # plot fQ fits
            if name != 'candels': 
                logm, logsfr, cs, notzero = readHighz(name, i_z+1, censat=censat, noise=noise, seed=seed)
                fSFS = highzSFSfit(name, i_z+1, censat=censat, noise=noise, seed=seed, dlogM=dlogM, slope_prior=slope_prior) # fit the SFSs
                colour = 'C'+str(i_n) 
            else: 
                logm, logsfr, cs, notzero = readHighz(name, i_z+1, censat='all', noise=False)
                fSFS = highzSFSfit(name, i_z+1, censat='all', noise=False, dlogM=dlogM, slope_prior=slope_prior) # fit the SFSs
                colour = 'k' 
            cut = (cs & notzero) 

            # calculate dSFS 
            if method == 'powerlaw': 
                _ = fSFS.powerlaw(logMfid=10.5) 
                dsfs = fSFS.d_SFS(logm[cut], logsfr[cut], method=method, silent=False) 
            elif method == 'broken_powerlaw': 
                _ = fSFS.broken_powerlaw(logMturnover=10., mlim=fSFS._has_nbinthresh) 
                dsfs = fSFS.d_SFS(logm[cut], logsfr[cut], method=method, silent=False) 
            elif method == 'interpexterp': 
                dsfs = fSFS.d_SFS(logm[cut], logsfr[cut], method=method, err_thresh=0.2, silent=False) 

            # calculate fQ 
            quenched = (dsfs < -1.*dSFS_limit)

            mbin0 = fSFS._mbins[fSFS._has_nbinthresh,0]
            mbin1 = fSFS._mbins[fSFS._has_nbinthresh,1]
            m_fq, fq, fq_err = [], [], [] 
            for i_m in range(np.sum(fSFS._has_nbinthresh)):  
                _inmbin = (fSFS._fit_logm >  mbin0[i_m]) & (fSFS._fit_logm <= mbin1[i_m]) 
                if np.sum(_inmbin) != 1: 
                    continue 
                inmbin = (logm[cut] > mbin0[i_m]) & (logm[cut] <= mbin1[i_m]) 

                m_fq.append(0.5 * (mbin0[i_m] + mbin1[i_m])) 
                fq.append(float(np.sum(inmbin & quenched))/float(np.sum(inmbin)))
                fq_err.append(fSFS._fit_err_logssfr[_inmbin][0]) 

            _plt, = sub.plot(m_fq, fq, color=colour)
            sub.fill_between(m_fq, np.array(fq) - np.array(fq_err), np.array(fq) + np.array(fq_err),
                    color=colour, alpha=0.5, linewidth=0)
            plts.append(_plt) 
        sub.set_xlim([9., 11.2]) 
        sub.set_ylim([0., 1.]) 
        if i_z < 3: sub.set_xticklabels([]) 
        if i_z not in [0, 3]: sub.set_yticklabels([]) 
        sub.text(0.95, 0.95, '$'+str(zlo[i_z])+'< z <'+str(zhi[i_z])+'$', 
                ha='right', va='top', transform=sub.transAxes, fontsize=15)
        if i_z == 0: 
            sub.legend(plts[:3], lbls[:3], loc='upper left', handletextpad=0.5, prop={'size': 17}) 
        elif i_z == 1: 
            sub.legend(plts[3:], lbls[3:], loc='upper left', handletextpad=0.5, prop={'size': 17}) 
            #sub.text(0.05, 0.95, ' '.join(name.upper().split('_')),
            #        ha='left', va='top', transform=sub.transAxes, fontsize=20)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', labelpad=15, fontsize=25) 
    bkgd.set_ylabel(r'Quiescent Fraction ($f_{\rm Q}$)', labelpad=15, fontsize=25) 
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    
    ffig = os.path.join(dir_fig, 
            'fQ_dsfs.%s.%s.dlogM%.1f.slope_prior%.1f_%.1f.dSFSlim_%.1f.png' % (censat, method, dlogM, slope_prior[0], slope_prior[1], dSFS_limit))
    if noise: ffig = ffig.replace('.png', '_wnoise.png') 
    fig.savefig(ffig, bbox_inches='tight')
    return None 


def fQ_dSFS_zevo_comparison(censat='centrals', noise=False, seed=1, dlogM=0.4, slope_prior=[0., 2.],  method='interpexterp', dSFS_limit=1.): 
    ''' Compare the QF derived from GMMs among the data and simulation  
    '''
    names = ['sam-light-slice', 'eagle', 'illustris_100myr', 'tng', 'simba', 'candels_goods']
    lbls = ['SC-SAM', 'EAGLE', 'Illustris', 'Illustris TNG', 'SIMBA', 'CANDELS (GOODS)'] 
    zlo = [0.5, 1., 1.4, 1.8, 2.2, 2.6]
    zhi = [1., 1.4, 1.8, 2.2, 2.6, 3.0]
    zlbls = ['$0.5 < z < 1.0$', '$1.0 < z < 1.4$', '$1.4 < z < 1.8$', '$1.8 < z < 2.2$', '$2.2 < z < 2.6$', '$2.6 < z < 3.0$']
    
    fig = plt.figure(figsize=(12,8))
    bkgd = fig.add_subplot(111, frameon=False)
    for i_n, name in enumerate(names):  # plot fQ fits
        sub = fig.add_subplot(2,3,i_n+1) 
        print('--- %s ---' % name) 

        plts = []
        for i_z in range(len(zlo)): 
            # read in logM and logSFR along with bestfits
            if 'candels' not in name: 
                logm, logsfr, cs, notzero = readHighz(name, i_z+1, censat=censat, noise=noise, seed=seed)
                fSFS = highzSFSfit(name, i_z+1, censat=censat, noise=noise, seed=seed, dlogM=dlogM, slope_prior=slope_prior) # fit the SFSs
            else: 
                logm, logsfr, cs, notzero = readHighz(name, i_z+1, censat='all', noise=False)
                fSFS = highzSFSfit(name, i_z+1, censat='all', noise=False, dlogM=dlogM, slope_prior=slope_prior) # fit the SFSs
            cut = (cs & notzero) 

            # calculate dSFS 
            if method == 'powerlaw': 
                _ = fSFS.powerlaw(logMfid=10.5) 
                dsfs = fSFS.d_SFS(logm[cut], logsfr[cut], method=method, silent=False) 
            elif method == 'broken_powerlaw': 
                _ = fSFS.broken_powerlaw(logMturnover=10., mlim=fSFS._has_nbinthresh) 
                dsfs = fSFS.d_SFS(logm[cut], logsfr[cut], method=method, silent=False) 
            elif method == 'interpexterp': 
                dsfs = fSFS.d_SFS(logm[cut], logsfr[cut], method=method, err_thresh=0.2, silent=False) 

            # calculate fQ 
            quenched = (dsfs < -1.*dSFS_limit)
            
            mbin0 = fSFS._mbins[fSFS._has_nbinthresh,0]
            mbin1 = fSFS._mbins[fSFS._has_nbinthresh,1]
            m_fq, fq, fq_err = [], [], [] 
            for i_m in range(np.sum(fSFS._has_nbinthresh)):  
                _inmbin = (fSFS._fit_logm >  mbin0[i_m]) & (fSFS._fit_logm <= mbin1[i_m]) 
                if np.sum(_inmbin) != 1: 
                    continue 
                inmbin = (logm[cut] > mbin0[i_m]) & (logm[cut] <= mbin1[i_m]) 

                m_fq.append(0.5 * (mbin0[i_m] + mbin1[i_m])) 
                fq.append(float(np.sum(inmbin & quenched))/float(np.sum(inmbin)))
                fq_err.append(fSFS._fit_err_logssfr[_inmbin][0]) 

            #sub.plot(fq_dict[name][i_z][0], fq_dict[name][i_z][1], c='C%i' % i_z) 
            _plt = sub.fill_between([0], [0], [0], 
                    color=lighten_color('C0', 0.2 + float(i_z) * 0.25), linewidth=0)
            if 'candels' not in name: 
                sub.fill_between(m_fq, np.array(fq) - np.array(fq_err), np.array(fq) + np.array(fq_err),
                        color=lighten_color('C%i' % i_n, 0.2 + float(i_z) * 0.25), linewidth=0)
            else: 
                sub.fill_between(m_fq, np.array(fq) - np.array(fq_err), np.array(fq) + np.array(fq_err),
                        color=lighten_color('k', 0.1 + float(i_z) * 0.15), linewidth=0)
            plts.append(_plt) 

        sub.set_xlim([9., 11.2]) 
        sub.set_ylim([0., 1.]) 
        if i_n < 3: sub.set_xticklabels([]) 
        if i_n not in [0, 3]: sub.set_yticklabels([]) 
        sub.text(0.05, 0.95, lbls[i_n], ha='left', va='top', transform=sub.transAxes, fontsize=20)
        if i_n == 2: 
            sub.legend(plts, zlbls, loc='upper right', handletextpad=0.3, prop={'size': 12}) 
        #    sub.legend(plts[:3], zlbls[:3], loc='center left', handletextpad=0.3, prop={'size': 15}) 
        #elif i_n == 4: 
        #    sub.legend(plts[3:], zlbls[3:], loc='center left', handletextpad=0.3, prop={'size': 15}) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', labelpad=15, fontsize=25) 
    bkgd.set_ylabel(r'Quiescent Fraction ($f_{\rm Q}$)', labelpad=15, fontsize=25) 
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    
    ffig = os.path.join(dir_fig, 
            'fQ_zevo_dsfs.%s.%s.dlogM%.1f.slope_prior%.1f_%.1f.dSFSlim_%.1f.png' % (censat, method, dlogM, slope_prior[0], slope_prior[1], dSFS_limit))
    if noise: ffig = ffig.replace('.png', '_wnoise.png') 
    fig.savefig(ffig, bbox_inches='tight')
    return None


def probSFS(name, censat='all', noise=False, seed=1, dlogM=0.4, slope_prior=[0., 2.]): 
    ''' calculate dSFS for specified high z catalog 
    ''' 
    zlo = [0.5, 1., 1.4, 1.8, 2.2, 2.6]
    zhi = [1., 1.4, 1.8, 2.2, 2.6, 3.0]
    logms, logsfrs = [], [] 
    sfs_fits, probsfss = [], [] 

    for i_z in range(len(zlo)): 
    
        # read in logM and logSFR along with bestfits
        if name != 'candels': 
            logm, logsfr, cs, notzero = readHighz(name, i_z+1, censat=censat, noise=noise, seed=seed)
            fSFS = highzSFSfit(name, i_z+1, censat=censat, noise=noise, seed=seed, dlogM=dlogM, slope_prior=slope_prior) # fit the SFSs
        else: 
            logm, logsfr, cs, notzero = readHighz(name, i_z+1, censat='all', noise=False)
            fSFS = highzSFSfit(name, i_z+1, censat='all', noise=False, dlogM=dlogM, slope_prior=slope_prior) # fit the SFSs
        cut = (cs & notzero) 

        logms.append(logm[cut])
        logsfrs.append(logsfr[cut])
        sfs_fits.append([fSFS._fit_logm, fSFS._fit_logsfr, fSFS._fit_err_logssfr]) 

        # calculate probSFS 
        probsfs = fSFS.probSFS(logm[cut], logsfr[cut]) 
        probsfss.append(probsfs) 

    # SFMS overplotted ontop of SFR--M* relation 
    fig = plt.figure(figsize=(12,8))
    bkgd = fig.add_subplot(111, frameon=False)
    for i_z in range(len(zlo)): 
        sub = fig.add_subplot(2,3,i_z+1) 
        
        hasprob = (probsfss[i_z] != -999.)
        sub.scatter(logms[i_z][hasprob], logsfrs[i_z][hasprob], c=probsfss[i_z][hasprob], s=1, linewidth=0) 
        sub.errorbar(sfs_fits[i_z][0], sfs_fits[i_z][1], sfs_fits[i_z][2], fmt='.C0')
        sub.set_xlim([8.5, 12.]) 
        sub.set_ylim([-3., 4.]) 
        sub.text(0.95, 0.05, '$'+str(zlo[i_z])+'< z <'+str(zhi[i_z])+'$', 
                ha='right', va='bottom', transform=sub.transAxes, fontsize=20)
        if i_z == 0: 
            sub.text(0.05, 0.95, ' '.join(name.upper().split('_')),
                    ha='left', va='top', transform=sub.transAxes, fontsize=20)

    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', labelpad=15, fontsize=25) 
    bkgd.set_ylabel(r'log ( SFR $[M_\odot \, yr^{-1}]$ )', labelpad=15, fontsize=25) 
    fig.subplots_adjust(wspace=0.2, hspace=0.15)

    ffig = os.path.join(dir_fig, 'probsfs.%s_%s.dlogM%.1f.slope_prior%.1f_%.1f.png' % (name.lower(), censat, dlogM, slope_prior[0], slope_prior[1]))
    if noise: ffig = ffig.replace('.png', '_wnoise.png') 
    fig.savefig(ffig, bbox_inches='tight')
    return None 


def candels(): 
    ''' SFR -- M* relation of CANDLES galaxies in the 6 redshift bins
    '''
    zlo = [0.5, 1., 1.4, 1.8, 2.2, 2.6]
    zhi = [1., 1.4, 1.8, 2.2, 2.6, 3.0]

    logms, logsfrs = [], [] 
    for i in range(1,len(zlo)+1): 
        logm, logsfr = readHighz('candels', i, keepzeros=False)
        logms.append(logm)
        logsfrs.append(logsfr)
    
    fig = plt.figure(figsize=(12,8))
    bkgd = fig.add_subplot(111, frameon=False)
    for i_z in range(len(zlo)): 
        sub = fig.add_subplot(2,3,i_z+1) 
        DFM.hist2d(logms[i_z], logsfrs[i_z], color='C0', 
                levels=[0.68, 0.95], range=[[7.8, 12.], [-4., 4.]], 
                plot_datapoints=True, fill_contours=False, plot_density=True, 
                ax=sub) 
        sub.set_xlim([8.5, 12.]) 
        sub.set_ylim([-3., 4.]) 
        sub.text(0.95, 0.05, '$'+str(zlo[i_z])+'< z <'+str(zhi[i_z])+'$', 
                ha='right', va='bottom', transform=sub.transAxes, fontsize=20)
        if i_z == 0: 
            sub.text(0.05, 0.95, 'CANDELS', 
                    ha='left', va='top', transform=sub.transAxes, fontsize=25)
    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', labelpad=15, fontsize=25) 
    bkgd.set_ylabel(r'log ( SFR $[M_\odot \, yr^{-1}]$ )', labelpad=15, fontsize=25) 
    fig.subplots_adjust(wspace=0.2, hspace=0.15)
    fig_name = ''.join([UT.doc_dir(), 'highz/figs/candels_sfrM.pdf'])
    fig.savefig(fig_name, bbox_inches='tight')


def pssfr(name, i_z, censat='all', noise=False, dlogM=0.4, slope_prior=[0., 2.], seed=1): 
    ''' p(log SSFR) distribution 
    '''
    logm, logsfr, cs, notzero = readHighz(name, i_z, censat=censat, noise=noise, seed=seed)
    logssfr = logsfr - logm 
    cut = (cs & notzero) 

    # fit the SFS
    fSFS = highzSFSfit(name, i_z, censat=censat, noise=noise, seed=seed, dlogM=dlogM, slope_prior=slope_prior)
    mbins = fSFS._mbins#[fSFS._mbins_sfs]
    nmbin = len(mbins)#np.sum(fSFS._mbins_sfs)
    nrow, ncol = 2, int(np.ceil(0.5*nmbin))
    ii = 0 

    icomps = fSFS._GMM_compID(fSFS._gbests, fSFS._mbins_median, slope_prior=slope_prior)

    fig = plt.figure(figsize=(5*ncol,5*nrow))
    bkgd = fig.add_subplot(1,1,1, frameon=False)
    for imbin in range(nmbin): 
        inmbin = ((logm > mbins[imbin][0]) & (logm <= mbins[imbin][1]) & cut)
        
        sub = fig.add_subplot(nrow, ncol, imbin+1)
        _ = sub.hist(logssfr[inmbin], bins=40, 
                range=[-14., -8.], density=True, histtype='stepfilled', 
                color='k', alpha=0.25, linewidth=1.75)
        
        if fSFS._has_nbinthresh[imbin]: 
            i_mbin = np.where((fSFS._mbins_median > mbins[imbin][0]) & (fSFS._mbins_median < mbins[imbin][1]))[0][0]
            gmm_ws = fSFS._gbests[i_mbin].weights_.flatten()
            gmm_mus = fSFS._gbests[i_mbin].means_.flatten()
            gmm_vars = fSFS._gbests[i_mbin].covariances_.flatten()

            isfs = icomps[0][ii]

            x_ssfr = np.linspace(-14., -8, 100)
            for icomp in range(len(gmm_mus)):  
                sub.plot(x_ssfr, gmm_ws[icomp] * MNorm.pdf(x_ssfr, gmm_mus[icomp], gmm_vars[icomp]), c='k', lw=0.75, ls=':') 
            if isfs is not None: 
                sub.plot(x_ssfr, gmm_ws[isfs] * MNorm.pdf(x_ssfr, gmm_mus[isfs], gmm_vars[isfs]), c='b', lw=1, ls='-') 
            ii += 1 
        #sub.legend(loc='upper left', prop={'size':15}) 
        sub.set_xlim([-13.6, -8.]) 
        
        if imbin == 0:
            sub.text(0.05, 0.95, 
                    '%s\n $%.1f < z < %.1f$' % (' '.join(name.upper().split('_')), zlo[i_z-1], zhi[i_z-1]),
                    ha='left', va='top', transform=sub.transAxes, fontsize=15)
        sub.text(0.05, 0.05, r'$N_{\rm gal} = %i$' % (np.sum(inmbin)), 
                ha='left', va='bottom', transform=sub.transAxes, fontsize=15)
        sub.set_title('%.1f $<$ log $M_* <$ %.1f' % (mbins[imbin][0], mbins[imbin][1]), fontsize=15)

    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel('log$(\; \mathrm{SSFR}\; [\mathrm{yr}^{-1}]\;)$', labelpad=5, fontsize=25) 
    bkgd.set_ylabel('$p\,(\;\log\; \mathrm{SSFR}\; [\mathrm{yr}^{-1}]\;)$', 
            labelpad=5, fontsize=25)
    #fig.subplots_adjust(wspace=0.1, hspace=0.075)

    ffig = os.path.join(dir_fig, '%s_z%i_%s_pssfr.dlogM%.1f.slope_prior%.1f_%.1f.pdf' % 
            (name.lower(), i_z, censat, dlogM, slope_prior[0], slope_prior[1]))
    if noise: ffig = ffig.replace('.pdf', '_wnoise.pdf') 
    fig.savefig(ffig, bbox_inches='tight')
    return None 

################################################
# figures: SFS 
################################################
def SFR_Mstar_comparison(censat='all', noise=False, seed=1, dlogM=0.4, slope_prior=[0., 2.]):
    ''' Compare the SFS fits among the data and simulation  
    '''
    names = ['sam-light-slice', 'eagle', 'illustris_100myr', 'tng', 'simba', 'candels_goods']
    lbls = ['SC-SAM', 'EAGLE', 'Illustris', 'Illustris TNG', 'SIMBA', 'CANDELS'] 
    
    # SFMS overplotted ontop of SFR--M* relation 
    fig = plt.figure(figsize=(18,18))
    bkgd = fig.add_subplot(111, frameon=False)
    for i_z in range(len(zlo)): 
        for i_n, name in enumerate(names):  # plot SFMS fits
            # fit SFR-M* 
            if 'candels' not in name: 
                logm, logsfr, cs, notzero = readHighz(name, i_z+1, censat=censat, noise=noise, seed=seed)
                fSFS = highzSFSfit(name, i_z+1, censat=censat, noise=noise, seed=seed, dlogM=dlogM, slope_prior=slope_prior) # fit the SFMSes
            else: 
                logm, logsfr, cs, notzero = readHighz(name, i_z+1, censat='all', noise=False)
                fSFS = highzSFSfit(name, i_z+1, censat='all', noise=False, dlogM=dlogM, slope_prior=slope_prior) # fit the SFMSes
            cut = (cs & notzero) 

            sfs_fit = [fSFS._fit_logm, fSFS._fit_logsfr, fSFS._fit_err_logssfr]

            sub = fig.add_subplot(6,6,i_z*6+i_n+1) 
            DFM.hist2d(logm[cut], logsfr[cut], color='C0', 
                    levels=[0.68, 0.95], range=[[7.8, 12.], [-4., 4.]], 
                    plot_datapoints=True, fill_contours=False, plot_density=True, 
                    ax=sub) 
            sub.errorbar(sfs_fit[0], sfs_fit[1], sfs_fit[2], fmt='.k') # plot SFS fit
            sub.set_xlim([8.5, 12.]) 
            sub.set_ylim([-3., 4.]) 
            if i_z < 5: sub.set_xticklabels([]) 
            if i_n != 0: sub.set_yticklabels([]) 
            if i_n == 0: sub.text(0.025, 0.95, '$%.1f < z < %.1f$' % (zlo[i_z], zhi[i_z]), 
                                  ha='left', va='top', transform=sub.transAxes, fontsize=20)
            if i_z == 0: sub.set_title(lbls[i_n], fontsize=25) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', labelpad=15, fontsize=25) 
    bkgd.set_ylabel(r'log ( SFR $[M_\odot \, yr^{-1}]$ )', labelpad=15, fontsize=25) 
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    ffig = os.path.join(dir_fig, 
            'sfr_mstar_comparison_%s_dlogM%.1f.slope_prior_%.1f_%.1f.png' % (censat, dlogM, slope_prior[0], slope_prior[1])) 
    if noise: ffig = ffig.replace('.png', '_wnoise_seed%i.png' % seed)
    fig.savefig(ffig, bbox_inches='tight')
    fpdf = UT.fig_tex(ffig, pdf=True) 
    fig.savefig(fpdf, bbox_inches='tight')
    return None


def SFS_comparison(censat='all', noise=False, seed=1, dlogM=0.4, slope_prior=[0., 2.]): 
    ''' Compare the SFS fits among the data and simulation  
    '''
    names = ['sam-light-slice', 'eagle', 'illustris_100myr', 'tng', 'simba', 'candels_goods']
    lbls = ['SC-SAM', 'EAGLE', 'Illustris', 'Illustris TNG', 'SIMBA', 'CANDELS (GOODS)'] 
    
    sfs_dict = {} 
    for name in names:  
        sfs_fits = [] 
        for i in range(1,len(zlo)+1): 
            if 'candels' not in name: 
                fSFS = highzSFSfit(name, i, censat=censat, noise=noise, seed=seed, dlogM=dlogM, slope_prior=slope_prior) # fit the SFSs
            else: 
                fSFS = highzSFSfit(name, i, censat='all', noise=False, dlogM=dlogM, slope_prior=slope_prior) # fit the SFSs
            sfs_fit = [fSFS._fit_logm, fSFS._fit_logsfr, fSFS._fit_err_logssfr]
            sfs_fits.append(sfs_fit) 
        sfs_dict[name] = sfs_fits
    
    # SFMS overplotted ontop of SFR--M* relation 
    fig = plt.figure(figsize=(12,8))
    bkgd = fig.add_subplot(111, frameon=False)
    for i_z in range(len(zlo)): 
        sub = fig.add_subplot(2,3,i_z+1) 

        plts = []
        for i_n, name in enumerate(names):  # plot SFMS fits
            if 'candels' in name: 
                _plt = sub.errorbar(
                    sfs_dict[name][i_z][0], 
                    sfs_dict[name][i_z][1], 
                    yerr=sfs_dict[name][i_z][2], fmt='.k')
            else: 
                colour = 'C'+str(i_n) 
                _plt = sub.fill_between(
                    sfs_dict[name][i_z][0], 
                    sfs_dict[name][i_z][1] - sfs_dict[name][i_z][2], 
                    sfs_dict[name][i_z][1] + sfs_dict[name][i_z][2], 
                    color='C%i' % i_n, alpha=0.75, linewidth=0.)
            plts.append(_plt) 
        sub.set_xlim(9., 11.5) 
        sub.set_ylim(-1., 3.75) 
        if i_z < 3: sub.set_xticklabels([]) 
        if i_z not in [0, 3]: sub.set_yticklabels([]) 
        sub.text(0.95, 0.05, '$%.1f < z < %.1f$' % (zlo[i_z], zhi[i_z]), 
                 ha='right', va='bottom', transform=sub.transAxes, fontsize=20)
        if i_z == 0: 
            sub.legend(plts[:3], lbls[:3], loc='upper left', handletextpad=0.5, prop={'size': 17}) 
        elif i_z == 1: 
            sub.legend(plts[3:], lbls[3:], loc='upper left', handletextpad=0.5, prop={'size': 17}) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', labelpad=15, fontsize=25) 
    bkgd.set_ylabel(r'log ( SFR $[M_\odot \, yr^{-1}]$ )', labelpad=15, fontsize=25) 
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    ffig = os.path.join(dir_fig, 'sfs_comparison_%s_dlogM%.1f.slope_prior_%.1f_%.1f.png' % 
            (censat, dlogM, slope_prior[0], slope_prior[1]))
    if noise: ffig = ffig.replace('.png', '_wnoise_seed%i.png' % seed)
    fig.savefig(ffig, bbox_inches='tight')
    fpdf = UT.fig_tex(ffig, pdf=True) 
    fig.savefig(fpdf, bbox_inches='tight')
    return None


def SFS_zevo_comparison(censat='all', noise=False, seed=1, dlogM=0.4, slope_prior=[0., 2.]): 
    ''' Compare the SFMS fits among the data and simulation  
    '''
    names = ['sam-light-slice', 'eagle', 'illustris_100myr', 'tng', 'simba', 'candels_goods']
    lbls = ['SC-SAM', 'EAGLE', 'Illustris', 'Illustris TNG', 'SIMBA', 'CANDELS'] 
    zlbls = ['$0.5 < z < 1.0$', '$1.0 < z < 1.4$', '$1.4 < z < 1.8$', '$1.8 < z < 2.2$', '$2.2 < z < 2.6$', '$2.6 < z < 3.0$']

    sfs_dict = {} 
    for name in names:  
        sfs_fits = [] 
        for i in range(1,len(zlo)+1): 
            if 'candels' not in name: 
                fSFS = highzSFSfit(name, i, censat=censat, noise=noise, seed=seed, dlogM=dlogM, slope_prior=slope_prior) # fit the SFSs
            else: 
                fSFS = highzSFSfit(name, i, censat='all', noise=False, dlogM=dlogM, slope_prior=slope_prior) # fit the SFSs
            sfs_fit = [fSFS._fit_logm, fSFS._fit_logsfr, fSFS._fit_err_logssfr]
            sfs_fits.append(sfs_fit) 
        sfs_dict[name] = sfs_fits
    
    fig = plt.figure(figsize=(12,8))
    bkgd = fig.add_subplot(111, frameon=False)

    for i_n, name in enumerate(names): 
        sub = fig.add_subplot(2,3,i_n+1) 
        plts = []
        for i_z in range(len(zlo)): 
            _plt = sub.fill_between([0], [0], [0], 
                    color=lighten_color('C0', 0.2 + float(i_z) * 0.25), linewidth=0)
            if name != 'candels': 
                sub.fill_between(sfs_dict[name][i_z][0], 
                        sfs_dict[name][i_z][1] - sfs_dict[name][i_z][2], 
                        sfs_dict[name][i_z][1] + sfs_dict[name][i_z][2], 
                        color=lighten_color('C%i' % i_n, 0.2 + float(i_z) * 0.25), linewidth=0)
            else: 
                sub.fill_between(sfs_dict[name][i_z][0], 
                        sfs_dict[name][i_z][1] - sfs_dict[name][i_z][2], 
                        sfs_dict[name][i_z][1] + sfs_dict[name][i_z][2], 
                        color=lighten_color('k', 0.1 + float(i_z) * 0.15), linewidth=0)
            #        color='C%i' % i_z, alpha=0.75, linewidth=0.)
            plts.append(_plt) 
        sub.set_xlim(9., 11.5) 
        sub.set_ylim(-1., 3.75) 
        if i_n < 3: sub.set_xticklabels([]) 
        if i_n not in [0, 3]: sub.set_yticklabels([]) 
        sub.text(0.95, 0.05, lbls[i_n], ha='right', va='bottom', transform=sub.transAxes, fontsize=20)
        if i_n == 0: 
            sub.legend(plts[:3], zlbls[:3], loc='upper left', handletextpad=0.5, prop={'size': 17}) 
        elif i_n == 1: 
            sub.legend(plts[3:], zlbls[3:], loc='upper left', handletextpad=0.5, prop={'size': 17}) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', labelpad=15, fontsize=25) 
    bkgd.set_ylabel(r'log ( SFR $[M_\odot \, yr^{-1}]$ )', labelpad=15, fontsize=25) 
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    ffig = os.path.join(dir_fig, 'sfs_zevo_comparison_%s_dlogM%.1f.slope_prior_%.1f_%.1f.png' % 
            (censat, dlogM, slope_prior[0], slope_prior[1]))
    if noise: ffig = ffig.replace('.png', '_wnoise_seed%i.png' % seed)
    fig.savefig(ffig, bbox_inches='tight')
    fpdf = UT.fig_tex(ffig, pdf=True) 
    fig.savefig(fpdf, bbox_inches='tight')
    return None

################################################
# figures: QF 
################################################
def fcomp(name, i_z, censat='centrals', noise=False, seed=1, dlogM=0.4, slope_prior=[0., 2.]):
    ''' get the component weights from GMM best-fit. quiescent fraction defined 
    as all components below SFS 
    '''
    logm, logsfr, cs, nonzero = readHighz(name, i_z, censat=censat, noise=noise, seed=seed)
    fSFS = highzSFSfit(name, i_z, censat=censat, noise=noise, seed=seed, dlogM=dlogM, slope_prior=slope_prior)
    
    # M* bins where SFS is reasonably fit 
    mbin0 = fSFS._mbins[fSFS._has_nbinthresh,0]
    mbin1 = fSFS._mbins[fSFS._has_nbinthresh,1]
    nmbin = len(fSFS._mbins_median)
    assert np.sum(fSFS._has_nbinthresh) == nmbin 

    try: 
        logm_sfs, _, _, w_sfs = fSFS._theta_sfs.T
        _, _, werr_sfs = fSFS._err_sfs.T
    except ValueError: 
        logm_sfs = np.array([]) 
    try: 
        logm_q, _, _, w_q = fSFS._theta_q.T
        _, _, werr_q = fSFS._err_q.T
    except ValueError: 
        logm_q = np.array([]) 
    try: 
        logm_int, _, _, w_int = fSFS._theta_int.T
        _, _, werr_int = fSFS._err_int.T
    except ValueError: 
        logm_int = np.array([]) 
    try: 
        logm_sbs, _, _, w_sbs = fSFS._theta_sbs.T
        _, _, werr_sbs = fSFS._err_sbs.T
    except ValueError: 
        logm_sbs = np.array([]) 
    try: 
        logm_int1, _, _, w_int1 = fSFS._theta_int1.T
        _, _, werr_int1 = fSFS._err_int1.T
    except ValueError: 
        logm_int1 = np.array([]) 
    try: 
        logm_int2, _, _, w_int2 = fSFS._theta_int2.T
        _, _, werr_int2 = fSFS._err_int2.T
    except ValueError: 
        logm_int2 = np.array([]) 
    try:
        logm_sbs1, _, _, w_sbs1 = fSFS._theta_sbs1.T
        _, _, werr_sbs1 = fSFS._err_sbs1.T
    except ValueError: 
        logm_sbs1 = np.array([]) 
    try: 
        logm_sbs2, _, _, w_sbs2 = fSFS._theta_sbs2.T
        _, _, werr_sbs2 = fSFS._err_sbs2.T
    except ValueError: 
        logm_sbs2 = np.array([]) 

    f_comps = np.zeros((5, nmbin)) # zero, sfs, q, other0, other1
    err_f_comps = np.zeros((5, nmbin)) # zero sfs, q, other0, other1 
    for i_m in range(nmbin): 
        # calculate the fraction of galaxies have that zero SFR
        inmbin      = cs & (logm > mbin0[i_m]) & (logm < mbin1[i_m]) # within bin 
        inmbin_z    = inmbin & ~nonzero # has SFR = 0 
        f_comps[0, i_m] = float(np.sum(inmbin_z))/float(np.sum(inmbin))

        f_nz = 1. - f_comps[0, i_m]  # multiply by non-zero fraction

        mbin_sfs = (mbin0[i_m] < logm_sfs) & (logm_sfs < mbin1[i_m]) 
        if np.sum(mbin_sfs) > 0: 
            f_comps[1, i_m] = f_nz * w_sfs[mbin_sfs]
            err_f_comps[1, i_m] = werr_sfs[mbin_sfs]

        mbin_q = (mbin0[i_m] < logm_q) & (logm_q < mbin1[i_m]) 
        if np.sum(mbin_q) > 0: 
            f_comps[2, i_m] = f_nz * w_q[mbin_q] 
            err_f_comps[2, i_m] = werr_q[mbin_q]

        mbin_int = (mbin0[i_m] < logm_int) & (logm_int < mbin1[i_m]) 
        mbin_int1 = (mbin0[i_m] < logm_int1) & (logm_int1 < mbin1[i_m]) 
        mbin_int2 = (mbin0[i_m] < logm_int2) & (logm_int2 < mbin1[i_m]) 
        if np.sum(mbin_int) > 0: 
            f_comps[3, i_m] += f_nz * w_int[mbin_int]
            err_f_comps[3, i_m] += werr_int[mbin_int]**2
        if np.sum(mbin_int1) > 0: 
            f_comps[3, i_m] += f_nz * w_int1[mbin_int1]
            err_f_comps[3, i_m] += werr_int1[mbin_int1]**2
        if np.sum(mbin_int2) > 0: 
            f_comps[3, i_m] += f_nz * w_int2[mbin_int2]
            err_f_comps[3, i_m] += werr_int2[mbin_int2]**2
        err_f_comps[3, i_m] = np.sqrt(err_f_comps[3, i_m]) 
        
        mbin_sbs = (mbin0[i_m] < logm_sbs) & (logm_sbs < mbin1[i_m]) 
        mbin_sbs1 = (mbin0[i_m] < logm_sbs1) & (logm_sbs1 < mbin1[i_m]) 
        mbin_sbs2 = (mbin0[i_m] < logm_sbs2) & (logm_sbs2 < mbin1[i_m]) 
        if np.sum(mbin_sbs) > 0: 
            f_comps[4, i_m] += f_nz * w_sbs[mbin_sbs]
            err_f_comps[4, i_m] += werr_sbs[mbin_sbs]**2
        if np.sum(mbin_sbs1) > 0: 
            f_comps[4, i_m] += f_nz * w_sbs1[mbin_sbs1]
            err_f_comps[4, i_m] += werr_sbs1[mbin_sbs1]**2
        if np.sum(mbin_sbs2) > 0: 
            f_comps[4, i_m] += f_nz * w_sbs2[mbin_sbs2]
            err_f_comps[4, i_m] += werr_sbs2[mbin_sbs2]**2
        err_f_comps[4, i_m] = np.sqrt(err_f_comps[4, i_m]) 
    
    return 0.5*(mbin0 + mbin1), f_comps, err_f_comps 


def QF(name, i_z, censat='centrals', noise=False, seed=1, dlogM=0.4, slope_prior=[0., 2.]):
    ''' derive quiescent fraction from GMM best-fit. quiescent fraction defined as all components below SFS 
    '''
    mmid, fcomps, err_fcomps = fcomp(name, i_z, censat=censat, noise=noise, seed=seed, dlogM=dlogM, slope_prior=slope_prior) 
    f_Q = fcomps[0,:] + fcomps[2,:] + fcomps[3,:]
    err_f_Q = np.sqrt(err_fcomps[0,:]**2 + err_fcomps[2,:]**2 + err_fcomps[3,:]**2)
    return mmid, f_Q, err_f_Q


def fcomp_comparison(censat='centrals', noise=False, seed=1, dlogM=0.4, slope_prior=[0., 2.]): 
    '''
    '''
    names = ['sam-light-slice', 'eagle', 'illustris_100myr', 'tng', 'simba', 'candels_goods']
    lbls = ['SC-SAM', 'EAGLE', 'Illustris', 'Illustris TNG', 'SIMBA', 'CANDELS (GOODS)'] 
    zlo = [0.5, 1., 1.4, 1.8, 2.2, 2.6]
    zhi = [1., 1.4, 1.8, 2.2, 2.6, 3.0]
    
    # SFMS overplotted ontop of SFR--M* relation 
    fig = plt.figure(figsize=(18,18))
    bkgd = fig.add_subplot(111, frameon=False)
    for i_z in range(len(zlo)): 
        for i_n, name in enumerate(names):  # plot SFMS fits
            sub = fig.add_subplot(6,6,i_z*6+i_n+1) 
            if 'candels' not in name: 
                mmid, f_comps, err_fcomps = fcomp(name, i_z+1, censat=censat, noise=noise, seed=seed, dlogM=dlogM, 
                        slope_prior=slope_prior)
            else: 
                mmid, f_comps, err_fcomps = fcomp(name, i_z+1, censat='all', noise=False, dlogM=dlogM, 
                        slope_prior=slope_prior)
            
            f_zero, f_sfs, f_q, f_other0, f_other1 = list(f_comps)
            
            sub.fill_between(mmid, np.zeros(len(mmid)), f_zero, # SFR = 0 
                    linewidth=0, color='C3') 
            sub.fill_between(mmid, f_zero, f_zero+f_q,              # Quenched
                    linewidth=0, color='C1') 
            sub.fill_between(mmid, f_zero+f_q, f_zero+f_q+f_other0,   # other0
                    linewidth=0, color='C2') 
            sub.fill_between(mmid, f_zero+f_q+f_other0, f_zero+f_q+f_other0+f_sfs, # SFMS 
                    linewidth=0, color='C0') 
            sub.fill_between(mmid, f_zero+f_q+f_other0+f_sfs, f_zero+f_q+f_other0+f_sfs+f_other1, # star-burst 
                    linewidth=0, color='C4') 

            sub.set_xlim([9, 11.5]) 
            sub.set_ylim([0., 1.]) 
            if i_z < 5: sub.set_xticklabels([]) 
            if i_n != 0: sub.set_yticklabels([]) 
            if i_n == 5: sub.text(0.95, 0.05, '$'+str(zlo[i_z])+'< z <'+str(zhi[i_z])+'$', 
                    ha='right', va='bottom', transform=sub.transAxes, fontsize=25)
            if i_z == 0: sub.set_title(lbls[i_n], fontsize=25) 
    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', labelpad=15, fontsize=25) 
    bkgd.set_ylabel(r'GMM component fractions', labelpad=15, fontsize=25) 
    fig.subplots_adjust(wspace=0.1, hspace=0.1)

    ffig = os.path.join(dir_fig, 'fcomp_comparison_%s_dlogM%.1f.slope_prior_%.1f_%.1f.pdf' % 
            (censat, dlogM, slope_prior[0], slope_prior[1]))
    if noise: ffig = ffig.replace('.pdf', '_wnoise_seed%i.pdf' % seed)
    fpdf = UT.fig_tex(ffig, pdf=True) 
    fig.savefig(fpdf, bbox_inches='tight')
    return None


def QF_comparison(censat='centrals', noise=False, seed=1, dlogM=0.4, slope_prior=[0., 2.]): 
    ''' Compare the QF derived from GMMs among the data and simulation  
    '''
    names = ['sam-light-slice', 'eagle', 'illustris_100myr', 'tng', 'simba', 'candels_goods']
    lbls = ['SC-SAM', 'EAGLE', 'Illustris', 'Illustris TNG', 'SIMBA', 'CANDELS (GOODS)'] 
    
    fq_dict = {} 
    for name in names:  
        fqs = [] 
        for i in range(1,len(zlo)+1): 
            if 'candels' not in name: 
                marr, fq, fqerr = QF(name, i, censat=censat, noise=noise, seed=seed, dlogM=dlogM, slope_prior=slope_prior)
            else: 
                marr, fq, fqerr = QF(name, i, censat='all', noise=False, dlogM=dlogM, slope_prior=slope_prior) 
            fqs.append([marr, fq, fqerr]) 
        fq_dict[name] = fqs
    
    fig = plt.figure(figsize=(12,8))
    bkgd = fig.add_subplot(111, frameon=False)
    for i_z in range(len(zlo)): 
        sub = fig.add_subplot(2,3,i_z+1) 

        plts = []
        for i_n, name in enumerate(names):  # plot fQ fits
            if 'candels' in name: 
                colour = 'k'
                _plt = sub.fill_between(fq_dict[name][i_z][0], 
                                        fq_dict[name][i_z][1] - fq_dict[name][i_z][2], 
                                        fq_dict[name][i_z][1] + fq_dict[name][i_z][2],
                                        alpha=0.3, color=colour, linewidth=0, zorder=1)
            else: 
                colour = 'C'+str(i_n) 
                _plt = sub.fill_between(fq_dict[name][i_z][0], 
                                        fq_dict[name][i_z][1] - fq_dict[name][i_z][2], 
                                        fq_dict[name][i_z][1] + fq_dict[name][i_z][2],
                                        alpha=0.5, color=colour, linewidth=0, zorder=10)
            plts.append(_plt) 
        sub.set_xlim([9., 11.2]) 
        sub.set_ylim([0., 1.]) 
        if i_z < 3: sub.set_xticklabels([]) 
        if i_z not in [0, 3]: sub.set_yticklabels([]) 
        sub.text(0.95, 0.05, '$'+str(zlo[i_z])+'< z <'+str(zhi[i_z])+'$', 
                ha='right', va='bottom', transform=sub.transAxes, fontsize=20)
        if i_z == 0: 
            sub.legend(plts[:3], lbls[:3], loc='upper left', handletextpad=0.5, prop={'size': 17}) 
        elif i_z == 1: 
            sub.legend(plts[3:], lbls[3:], loc='upper left', handletextpad=0.5, prop={'size': 17}) 
            #sub.text(0.05, 0.95, ' '.join(name.upper().split('_')),
            #        ha='left', va='top', transform=sub.transAxes, fontsize=20)
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', labelpad=15, fontsize=25) 
    bkgd.set_ylabel(r'Quiescent Fraction ($f_{\rm Q}$)', labelpad=15, fontsize=25) 
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    
    ffig = os.path.join(dir_fig, 'fq_comparison_%s_dlogM%.1f.slope_prior_%.1f_%.1f.pdf' % (censat, dlogM, slope_prior[0], slope_prior[1]))
    if noise: ffig = ffig.replace('.pdf', '_wnoise_seed%i.pdf' % seed)
    fpdf = UT.fig_tex(ffig, pdf=True) 
    fig.savefig(fpdf, bbox_inches='tight')
    return None


def QF_zevo_comparison(censat='centrals', noise=False, seed=1, dlogM=0.4, slope_prior=[0., 2.]): 
    ''' Compare the QF derived from GMMs among the data and simulation  
    '''
    names = ['sam-light-slice', 'eagle', 'illustris_100myr', 'tng', 'simba', 'candels_goods']
    lbls = ['SC-SAM', 'EAGLE', 'Illustris', 'Illustris TNG', 'SIMBA', 'CANDELS (GOODS)'] 
    zlo = [0.5, 1., 1.4, 1.8, 2.2, 2.6]
    zhi = [1., 1.4, 1.8, 2.2, 2.6, 3.0]
    zlbls = ['$0.5 < z < 1.0$', '$1.0 < z < 1.4$', '$1.4 < z < 1.8$', '$1.8 < z < 2.2$', '$2.2 < z < 2.6$', '$2.6 < z < 3.0$']
    
    fq_dict = {} 
    for name in names:  
        print('--- %s ---' % name) 
        fqs = [] 
        for i in range(1,len(zlo)+1): 
            if 'candels' not in name: 
                marr, fq, fqerr = QF(name, i, censat=censat, noise=noise, seed=seed, dlogM=dlogM, slope_prior=slope_prior)
            else: 
                marr, fq, fqerr = QF(name, i, censat='all', noise=False, dlogM=dlogM, slope_prior=slope_prior) 
            fqs.append([marr, fq, fqerr]) 
        fq_dict[name] = fqs
    
    fig = plt.figure(figsize=(12,8))
    bkgd = fig.add_subplot(111, frameon=False)
    for i_n, name in enumerate(names):  # plot fQ fits
        sub = fig.add_subplot(2,3,i_n+1) 
        print('--- %s ---' % name) 

        plts = []
        for i_z in range(len(zlo)): 
            #sub.plot(fq_dict[name][i_z][0], fq_dict[name][i_z][1], c='C%i' % i_z) 
            _plt = sub.fill_between([0], [0], [0], 
                    color=lighten_color('C0', 0.2 + float(i_z) * 0.25), linewidth=0)
            if 'candels' not in name: 
                sub.fill_between(fq_dict[name][i_z][0], 
                                        fq_dict[name][i_z][1] - fq_dict[name][i_z][2], 
                                        fq_dict[name][i_z][1] + fq_dict[name][i_z][2], 
                                        color=lighten_color('C%i' % i_n, 0.2 + float(i_z) * 0.25), linewidth=0)
            else: 
                sub.fill_between(fq_dict[name][i_z][0], 
                                        fq_dict[name][i_z][1] - fq_dict[name][i_z][2], 
                                        fq_dict[name][i_z][1] + fq_dict[name][i_z][2], 
                                        color=lighten_color('k', 0.1 + float(i_z) * 0.15), 
                                        linewidth=0)
            #                        alpha=0.5, color='C'+str(i_z), linewidth=0)
            plts.append(_plt) 

        sub.set_xlim([9., 11.2]) 
        sub.set_ylim([0., 1.]) 
        if i_n < 3: sub.set_xticklabels([]) 
        if i_n not in [0, 3]: sub.set_yticklabels([]) 
        sub.text(0.05, 0.95, lbls[i_n], ha='left', va='top', transform=sub.transAxes, fontsize=20)
        if i_n == 2: 
            sub.legend(plts, zlbls, loc='upper right', handletextpad=0.3, prop={'size': 12}) 
        #    sub.legend(plts[:3], zlbls[:3], loc='center left', handletextpad=0.3, prop={'size': 15}) 
        #elif i_n == 4: 
        #    sub.legend(plts[3:], zlbls[3:], loc='center left', handletextpad=0.3, prop={'size': 15}) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', labelpad=15, fontsize=25) 
    bkgd.set_ylabel(r'Quiescent Fraction ($f_{\rm Q}$)', labelpad=15, fontsize=25) 
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    
    ffig = os.path.join(dir_fig, 'fq_zevo_comparison_%s.dlogM%.1f.slope_prior_%.1f_%.1f.pdf' % 
            (censat, dlogM, slope_prior[0], slope_prior[1]))
    if noise: ffig = ffig.replace('.pdf', '_wnoise_seed%i.pdf' % seed)
    fpdf = UT.fig_tex(ffig, pdf=True) 
    fig.savefig(fpdf, bbox_inches='tight')
    return None


##################################################
# appendix: CANDELS fields 
##################################################
def CANDELS_fields(dlogM=0.4, slope_prior=[0., 2.]):
    ''' Compare the SFS fits among the data and simulation  
    '''
    names = ['candels', 'candels_goods', 'candels_cami']
    lbls = ['CANDELS', 'CANDELS (GOODS)', 'CANDELS (Camilla)'] 
    
    # SFMS overplotted ontop of SFR--M* relation 
    fig = plt.figure(figsize=(12,8))
    bkgd = fig.add_subplot(111, frameon=False)
    for i_z in range(len(zlo)): 
        for i_n, name in enumerate(names):  # plot SFMS fits
            # fit SFR-M* 
            logm, logsfr, cs, notzero = readHighz(name, i_z+1, censat='all', noise=False, seed=1)
            fSFS = highzSFSfit(name, i_z+1, censat='all', noise=False, seed=1, dlogM=dlogM, slope_prior=slope_prior) # fit the SFMSes
            cut = (cs & notzero) 

            sfs_fit = [fSFS._fit_logm, fSFS._fit_logsfr, fSFS._fit_err_logssfr]

            sub = fig.add_subplot(2,3,i_z+1) 
            DFM.hist2d(logm[cut], logsfr[cut], color='C%i' % i_n, 
                    levels=[0.68, 0.95], range=[[7.8, 12.], [-4., 4.]], 
                    plot_datapoints=False, fill_contours=False, plot_density=False, 
                    ax=sub) 
            sub.errorbar(sfs_fit[0], sfs_fit[1], sfs_fit[2], fmt='.C%i' % i_n, label=lbls[i_n]) # plot SFS fit
            sub.set_xlim([8.5, 12.]) 
            sub.set_ylim([-3., 4.]) 
            if i_z < 3: sub.set_xticklabels([]) 
            if i_z not in [0, 3]: sub.set_yticklabels([]) 
            if i_n == 0: sub.text(0.025, 0.95, '$%.1f < z < %.1f$' % (zlo[i_z], zhi[i_z]), 
                                  ha='left', va='top', transform=sub.transAxes, fontsize=20)
        if i_z == 5: sub.legend(loc='lower right', fontsize=15)

    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', labelpad=15, fontsize=25) 
    bkgd.set_ylabel(r'log ( SFR $[M_\odot \, yr^{-1}]$ )', labelpad=15, fontsize=25) 
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    ffig = os.path.join(dir_fig, 'CANDLES_sfr_mstar_comparison_dlogM%.1f.slope_prior_%.1f_%.1f.png' % (dlogM, slope_prior[0], slope_prior[1])) 
    fig.savefig(ffig, bbox_inches='tight')
    
    fig = plt.figure(figsize=(18,3*len(names)))
    bkgd = fig.add_subplot(111, frameon=False)
    for i_z in range(len(zlo)): 
        for i_n, name in enumerate(names):  # plot SFMS fits
            # fit SFR-M* 
            logm, logsfr, cs, notzero = readHighz(name, i_z+1, censat='all', noise=False, seed=1)
            cut = (cs & notzero) 

            sfs_fit = [fSFS._fit_logm, fSFS._fit_logsfr, fSFS._fit_err_logssfr]

            sub = fig.add_subplot(len(names),6,i_n*6+i_z+1) 
            sub.scatter(logm[cut], logsfr[cut], color='k', s=0.01) 
            sub.set_xlim([8.5, 12.]) 
            sub.set_ylim([-3., 4.]) 
            if i_z != 0: sub.set_yticklabels([]) 
            if i_n == 0:
                sub.set_xticklabels([]) 
                sub.set_title( '$%.1f < z < %.1f$' % (zlo[i_z], zhi[i_z]), fontsize=20) 
            if i_z == 0: 
                sub.text(0.025, 0.95, lbls[i_n], ha='left', va='top', transform=sub.transAxes, fontsize=15)

    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', labelpad=15, fontsize=25) 
    bkgd.set_ylabel(r'log ( SFR $[M_\odot \, yr^{-1}]$ )', labelpad=15, fontsize=25) 
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    ffig = os.path.join(dir_fig, 'CANDLES_fields.png') 
    fig.savefig(ffig, bbox_inches='tight')
    return None
    return None


def CANDELS_fields_QF(dlogM=0.4, slope_prior=[0., 2.]): 
    ''' Compare the QF derived from GMMs between CANDELS and CANDELS GOODS fields
    '''
    names = ['candels', 'candels_goods', 'candels_cami']
    lbls = ['CANDELS', 'CANDELS (GOODS)', 'CANDELS (Camilla)'] 
    zlo = [0.5, 1., 1.4, 1.8, 2.2, 2.6]
    zhi = [1., 1.4, 1.8, 2.2, 2.6, 3.0]
    zlbls = ['$0.5 < z < 1.0$', '$1.0 < z < 1.4$', '$1.4 < z < 1.8$', '$1.8 < z < 2.2$', '$2.2 < z < 2.6$', '$2.6 < z < 3.0$']
    
    fq_dict = {} 
    for name in names:  
        print('--- %s ---' % name) 
        fqs = [] 
        for i in range(1,len(zlo)+1): 
            marr, fq, fqerr = QF(name, i, censat='all', noise=False, dlogM=dlogM, slope_prior=slope_prior) 
            fqs.append([marr, fq, fqerr]) 
        fq_dict[name] = fqs
    
    fig = plt.figure(figsize=(5*len(names),5))
    bkgd = fig.add_subplot(111, frameon=False)
    for i_n, name in enumerate(names):  # plot fQ fits
        sub = fig.add_subplot(1,len(names),i_n+1) 
        print('--- %s ---' % name) 

        plts = []
        for i_z in range(len(zlo)): 
            #sub.plot(fq_dict[name][i_z][0], fq_dict[name][i_z][1], c='C%i' % i_z) 
            _plt = sub.fill_between([0], [0], [0], 
                    color=lighten_color('k', 0.1 + float(i_z) * 0.15), linewidth=0)
            sub.fill_between(fq_dict[name][i_z][0], 
                                    fq_dict[name][i_z][1] - fq_dict[name][i_z][2], 
                                    fq_dict[name][i_z][1] + fq_dict[name][i_z][2], 
                                    color=lighten_color('k', 0.1 + float(i_z) * 0.15), 
                                    linewidth=0)
            #                        alpha=0.5, color='C'+str(i_z), linewidth=0)
            plts.append(_plt) 

        sub.set_xlim([9., 11.2]) 
        sub.set_ylim([0., 1.]) 
        if i_n not in [0, 3]: sub.set_yticklabels([]) 
        sub.text(0.05, 0.95, lbls[i_n], ha='left', va='top', transform=sub.transAxes, fontsize=20)
        if i_n == 1: 
            sub.legend(plts, zlbls, loc='upper right', handletextpad=0.3, prop={'size': 12}) 
        #    sub.legend(plts[:3], zlbls[:3], loc='center left', handletextpad=0.3, prop={'size': 15}) 
        #elif i_n == 4: 
        #    sub.legend(plts[3:], zlbls[3:], loc='center left', handletextpad=0.3, prop={'size': 15}) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', labelpad=15, fontsize=25) 
    bkgd.set_ylabel(r'Quiescent Fraction ($f_{\rm Q}$)', labelpad=15, fontsize=25) 
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    
    ffig = os.path.join(dir_fig, 'CANDLES_fields.fq_zevo.dlogM%.1f.slope_prior_%.1f_%.1f.png' % 
            (dlogM, slope_prior[0], slope_prior[1]))
    fig.savefig(ffig, bbox_inches='tight')
    return None


##################################################
# appendix: resolution effects 
##################################################
def Pssfr_res_impact(censat='centrals', n_mc=20, noise=False, seed=1, poisson=False): 
    ''' Plot the impact of SFR resolution on the P(SSFR) distribution. 
    '''
    names = ['eagle', 'illustris_100myr', 'tng', 'simba']
    lbls = ['EAGLE', 'Illustris', 'Illustris TNG', 'SIMBA'] 
    SFRres_dict = {'eagle': 0.018, 'illustris_100myr': 0.0126, 'tng': 0.014, 'simba': 0.182} 
    mbins = [[8.9, 9.4], [8.9, 9.4], [8.9, 9.4], [9.4, 9.9]] 

    fig = plt.figure(figsize=(16,6))
    bkgd = fig.add_subplot(111, frameon=False)
    for i_c, name in enumerate(names):
        for i_m in range(2):  
            if i_m == 0: mbin = mbins[i_c] 
            else: mbin = [10.5, 10.9]

            logm, logsfr, cs, nonzero = readHighz(name, 1, censat=censat, noise=noise, seed=seed)
            #logm, logsfr, nonzero = readHighz(name, 1, keepzeros=True, noise=noise, seed=seed)
            
            inmbin = (logm > mbin[0]) & (logm < mbin[1]) & cs # in M* bin 
            ngal_bin = float(np.sum(inmbin)) 

            dsfr_res = SFRres_dict[name] # SFR resolution 

            hs_uniform, hs_poisson = [], [] 
            rand = np.random.RandomState(seed)
            for i_mc in range(n_mc): # loop through everything n_mc times 
                # sample logSFR' from [logSFR + dlogSFR] 
                logsfr_nz   = np.log10(10**logsfr[nonzero] + dsfr_res * np.random.uniform(size=np.sum(nonzero)))
                logsfr_z    = np.log10(dsfr_res * np.random.uniform(size=np.sum(~nonzero)))
                _logsfr     = np.zeros(len(logsfr))
                _logsfr[nonzero] = logsfr_nz
                _logsfr[~nonzero] = logsfr_z 
                _logssfr = _logsfr - logm 
                
                h0, h1 = np.histogram(_logssfr, bins=40, range=[-16., -8.])
                hs_uniform.append(h0)
                
                logsfr_nz   = np.log10(10**logsfr[nonzero] + dsfr_res * rand.poisson(size=np.sum(nonzero))) 
                logsfr_z    = np.log10(dsfr_res * rand.poisson(size=np.sum(~nonzero))) 
                _logsfr     = np.zeros(len(logsfr))
                _logsfr[nonzero] = logsfr_nz
                _logsfr[~nonzero] = logsfr_z 
                _logssfr = _logsfr - logm 

                h0, h1 = np.histogram(_logssfr, bins=40, range=[-16., -8.])
                hs_poisson.append(h0)

            hs_uniform = np.array(hs_uniform)/ngal_bin
            hs_poisson = np.array(hs_poisson)/ngal_bin
               
            sub = fig.add_subplot(2,4,i_c+1+4*i_m)
            h0, h1 = np.histogram(logsfr[nonzero] - logm[nonzero], bins=40, range=[-16., -8.])
            bar_x, bar_y = UT.bar_plot(h1, h0/ngal_bin)
            sub.plot(bar_x, bar_y, c='k', ls='-', lw=1.5)#, label='w/o SFR $=0$')

            bar_x, bar_y = UT.bar_plot(h1, np.mean(hs_uniform,axis=0))
            sub.plot(bar_x, bar_y, c='C1', lw=1, label=r"$\mathrm{SFR}_i' \in [\mathrm{SFR}, \mathrm{SFR}+\Delta_\mathrm{SFR}]$")
            sub.errorbar(0.5*(h1[1:] + h1[:-1])-0.02, np.mean(hs_uniform, axis=0), yerr=np.std(hs_uniform, axis=0),
                         fmt='.C1', markersize=.5)
            
            if poisson:
                bar_x, bar_y = UT.bar_plot(h1, np.mean(hs_poisson,axis=0))
                sub.plot(bar_x, bar_y, c='C1', ls='--', lw=1)#, label=r"$\mathrm{SFR}_i' \in [\mathrm{SFR}, \mathrm{SFR}+\Delta_\mathrm{SFR}]$")
                sub.errorbar(0.5*(h1[1:] + h1[:-1])-0.02, np.mean(hs_poisson, axis=0), yerr=np.std(hs_poisson, axis=0),
                             fmt='.C1', markersize=.5)
            #sub.errorbar(0.5*(h1[1:] + h1[:-1])+0.02, np.mean(hs_nz, axis=0), 
            #        yerr=np.std(hs_nz, axis=0), fmt='.k', markersize=.5)

            sub.set_xlim([-13.25, -8.6])
            if i_m == 0: sub.set_xticks([])
            else: sub.set_xticks([-13., -11., -9.]) 
            sub.set_ylim(0., None) 
            #if i_m == 0: sub.set_ylim([0., 1.]) 
            #else:  sub.set_ylim([0., 2.]) 
            #if i_c != 0: sub.set_yticks([]) 
            if i_m == 0: sub.set_title(lbls[i_c], fontsize=20) 
            sub.set_ylim(0., 1.3*sub.get_ylim()[1])
            sub.text(0.5, 0.92, '$'+str(mbin[0])+'< \log M_* <'+str(mbin[1])+'$',
                ha='center', va='top', transform=sub.transAxes, fontsize=15)
    
            if (i_c == 3) and (i_m == 1): 
                sub.legend(loc='lower left', handletextpad=0.1, bbox_to_anchor=(0.01, 0.65), frameon=False, prop={'size':13}) 
    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_ylabel(r'P(log SSFR  $[yr^{-1}])$', labelpad=10, fontsize=20)
    bkgd.set_xlabel(r'log SSFR  $[yr^{-1}]$', labelpad=10, fontsize=20)

    fig.subplots_adjust(wspace=0.2, hspace=0.05)
    ffig = os.path.join(dir_fig, 'Pssfr_res_impact_comparison_%s.pdf' % censat)
    if noise: ffig = ffig.replace('.pdf', '_wnoise_seed%i.pdf' % seed) 
    fig.savefig(ffig, bbox_inches='tight')
    plt.close()
    return None


def Mlim_res_impact(censat='centrals', n_mc=20, noise=False, seed=1, dlogM=0.4, slope_prior=[0.,2.], threshold=0.2): 
    ''' determine M_lim stellar mass where the resolution limit of simulations
    impact the SFS fits.
    '''
    names = ['eagle', 'illustris_100myr', 'tng', 'simba']
    lbls = ['EAGLE', 'Illustris', 'Illustris TNG', 'SIMBA'] 
    SFRres_dict = {'eagle': 0.018, 'illustris_100myr': 0.0126, 'tng': 0.014, 'simba': 0.182} 
    
    fig = plt.figure(figsize=(18,3*len(names)))
    bkgd = fig.add_subplot(111, frameon=False)
    for i_n, name in enumerate(names):  # plot SFMS fits
        for i_z in range(len(zlo)): 
            # read in log M* and log SFR of sim (including SFR=0) 
            logm, logsfr, cs, nonzero = readHighz(name, i_z+1, censat=censat, noise=noise, seed=seed)
            isnonzero = (cs & nonzero) 
            iszero = (cs & ~nonzero) 
            # standard SFS fit 
            sfs_std = highzSFSfit(name, i_z+1, censat=censat, noise=noise, seed=seed, 
                    dlogM=dlogM, slope_prior=slope_prior)
            sfs_std_logm    = sfs_std._fit_logm
            sfs_std_sfr     = sfs_std._fit_logsfr
            sfs_std_sfr_err = sfs_std._fit_err_logssfr
            
            dsfr_res = SFRres_dict[name] # SFR resolution 

            sfs_mc = np.zeros((n_mc, len(sfs_std_logm)))
            #sfs_mc_mbins_sfs = [] 
            for i_mc in range(n_mc): # loop through everything n_mc times 
                # sample logSFR' from [logSFR + dlogSFR] 
                logm_nz     = logm[isnonzero]
                logsfr_nz   = np.log10(10**logsfr[isnonzero] + dsfr_res * np.random.uniform(size=np.sum(isnonzero)))
                logm_z      = logm[iszero] 
                logsfr_z    = np.log10(dsfr_res * np.random.uniform(size=np.sum(iszero)))
                _logm       = np.concatenate([logm_nz, logm_z]) 
                _logsfr     = np.concatenate([logsfr_nz, logsfr_z]) 

                fSFS_mc = fstarforms(fit_range=[8.5, 12.0]) # stellar mass range
                sfs_fit = fSFS_mc.fit(_logm, _logsfr,
                        method='gaussmix',      # Gaussian Mixture Model fitting 
                        dlogm = dlogM,          # stellar mass bins of 0.4 dex
                        slope_prior = slope_prior, 
                        Nbin_thresh=100,        # at least 100 galaxies in bin 
                        error_method='bootstrap',  # uncertainty estimate method 
                        n_bootstrap=1)        # number of bootstrap bins
                for ii in range(len(sfs_std_logm)):
                    sfs_mc[i_mc,ii] = fSFS_mc._fit_logsfr[np.argmin(np.abs((fSFS_mc._fit_logm) - sfs_std_logm[ii]))]
                #sfs_mc[i_mc,:] = fSFS_mc._fit_logsfr
                #sfs_mc_mbins_sfs.append(fSFS_mc._fit_logm) 
            mu_sfs_mc = np.average(sfs_mc, axis=0)  # average SFS SFR of n_mc
            #joint_mbins = sfs_std._mbins_sfs & (np.sum(np.array(sfs_mc_mbins_sfs), axis=0) == n_mc)
            #np.isfinite(mu_sfs_mc) # M* bins with SFS 

            dsfs_res = sfs_std_sfr - mu_sfs_mc # change in SFS fit from resolution limit 
            
            # determine logM* limit based on when dsfs_res shifts by > threshold dex 
            if noise: threshold = np.sqrt(threshold**2 + 0.33**2)
            above_thresh = (np.abs(dsfs_res) > threshold) & (sfs_std_logm < 10.) 
            if np.sum(above_thresh) > 0:
                mlim = (sfs_std_logm[(np.abs(dsfs_res) > threshold)]).max() + 0.5 * dlogM 
            else: 
                mlim = None 
            
            # --- plot the comparison --- 
            sub = fig.add_subplot(len(names),6,i_n*6+i_z+1) 
            DFM.hist2d(logm[isnonzero], logsfr[isnonzero], color='C0', 
                    levels=[0.68, 0.95], range=[[7.8, 12.], [-4., 4.]], 
                    plot_datapoints=True, fill_contours=False, plot_density=True, 
                    ax=sub) 
            # plot standard SFS fit
            plt_std = sub.errorbar(sfs_std_logm, sfs_std_sfr, sfs_std_sfr_err, fmt='.k') # plot SFS fit
            plt_res = sub.scatter(sfs_std_logm, mu_sfs_mc, marker='x', color='C1', lw=1, s=40)

            if i_n == 3 and i_z == 5: sub.legend([plt_std, plt_res], 
                    ['w/ SFR res. eff.', r"$\mathrm{SFR}_i' \in [\mathrm{SFR}, \mathrm{SFR}+\Delta_\mathrm{SFR}]$"], 
                    loc='lower right', handletextpad=-0.02, prop={'size': 10}) 
            if i_n < len(names)-1: sub.set_xticklabels([]) 
            if i_z > 0: sub.set_yticklabels([]) 
            else: sub.text(0.05, 0.95, lbls[i_n], ha='left', va='top', transform=sub.transAxes, fontsize=20)
            
            if mlim is not None: 
                sub.text(0.95, 0.05, '$\log\,M_{\lim}='+str(round(mlim,2))+'$', 
                         ha='right', va='bottom', transform=sub.transAxes, fontsize=15)
                sub.vlines(mlim, -4., 4., color='k', linestyle='--', linewidth=0.5)
                sub.fill_between([7.5, mlim], [-3., -3.], [4., 4.], color='k', alpha=0.2)
            sub.set_xlim([8.5, 12.]) 
            sub.set_xticks([9., 10., 11., 12.]) 
            sub.set_ylim([-3., 4.]) 

    bkgd.set_xlabel(r'log $M_* \;\;[M_\odot]$', labelpad=10, fontsize=25) 
    bkgd.set_ylabel(r'log SFR $[M_\odot \, yr^{-1}]$', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    
    ffig = os.path.join(dir_fig, 'Mlim_res_impact_%s.pdf' % (censat))
    if noise: ffig = ffig.replace('.pdf', '_wnoise_seed%i.pdf' % seed) 
    fig.savefig(ffig, bbox_inches='tight')
    plt.close()
    return None

##################################################
# appendix: slice vs full  
##################################################
def SFS_SAM_comparison(noise=False, seed=1, dlogM=0.4, slope_prior=[0., 2.]): 
    ''' Compare the SFMS fits among the data and simulation  
    '''
    names = ['sam-light-slice', 'sam-light-full']
    lbls = ['SAM (slice)', 'SAM (full)']
    zlo = [0.5, 1., 1.4, 1.8, 2.2, 2.6]
    zhi = [1., 1.4, 1.8, 2.2, 2.6, 3.0]
    
    fig = plt.figure(figsize=(12,8))
    for i in range(1,len(zlo)+1): 
        sub = fig.add_subplot(2,3,i) 
        for i_n, name in enumerate(names):  
            if noise and 'sam-light' in name: 
                logm, logsfr, cs, notzero = readHighz(name, i, censat='centrals', noise=noise, seed=seed)
                fSFS = highzSFSfit(name, i, censat='centrals', noise=noise, seed=seed, dlogM=dlogM, slope_prior=slope_prior) 
            else:
                logm, logsfr, cs, notzero = readHighz(name, i, censat='centrals')
                fSFS = highzSFSfit(name, i, censat='centrals',  dlogM=dlogM, slope_prior=slope_prior) 
            
            cut = (cs & notzero) 
            sfs_fit = [fSFS._fit_logm, fSFS._fit_logsfr, fSFS._fit_err_logssfr]

            DFM.hist2d(logm[cut], logsfr[cut], color='C%i' % i_n, 
                    levels=[0.68, 0.95], range=[[7.8, 12.], [-4., 4.]], 
                    plot_datapoints=False, fill_contours=False, plot_density=False, 
                    ax=sub) 
            #sub.errorbar(sfs_fit[0], sfs_fit[1], sfs_fit[2], fmt='.k') # plot SFS fit
            sub.fill_between(sfs_fit[0], sfs_fit[1] - sfs_fit[2], sfs_fit[1] + sfs_fit[2], 
                    color='C%i' % i_n, alpha=0.75, linewidth=0., label=lbls[i_n], zorder=10)
        sub.set_xlim([8.5, 12.]) 
        sub.set_ylim([-1.5, 3.]) 
        sub.text(0.95, 0.05, '$'+str(zlo[i-1])+'< z <'+str(zhi[i-1])+'$', 
                ha='right', va='bottom', transform=sub.transAxes, fontsize=20)
        if i == 1: sub.legend(loc='upper left', handletextpad=0.5, prop={'size': 15}) 

    bkgd = fig.add_subplot(111, frameon=False)
    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', labelpad=15, fontsize=25) 
    bkgd.set_ylabel(r'log ( SFR $[M_\odot \, yr^{-1}]$ )', labelpad=15, fontsize=25) 
    fig.subplots_adjust(wspace=0.2, hspace=0.15)
    ffig = os.path.join(dir_fig, 'sfs_SAM_comparison_centrals_dlogM%.1f_slope_prior_%.1f_%.1f.pdf' % 
            (dlogM, slope_prior[0], slope_prior[1])) 
    if noise: ffig = ffig.replace('.pdf', '_wnoise_seed%i.pdf' % seed)
    fig.savefig(ffig, bbox_inches='tight')
    fpdf = UT.fig_tex(ffig, pdf=True) 
    fig.savefig(fpdf, bbox_inches='tight')
    return None


def QF_SAM_comparison(noise=False, seed=1, dlogM=0.4, slope_prior=[0., 2.]): 
    ''' Compare the SFMS fits among the data and simulation  
    '''
    names = ['sam-light-slice', 'sam-light-full']
    lbls = ['SC-SAM slice', 'SC-SAM full'] 
    zlo = [0.5, 1., 1.4, 1.8, 2.2, 2.6]
    zhi = [1., 1.4, 1.8, 2.2, 2.6, 3.0]
    
    fq_dict = {} 
    for name in names:  
        fqs = [] 
        for i in range(1,len(zlo)+1): 
            marr, fq, fqerr = QF(name, i, censat='centrals', noise=noise, seed=seed, dlogM=dlogM, slope_prior=slope_prior)
            fqs.append([marr, fq, fqerr]) 
        fq_dict[name] = fqs
    
    fig = plt.figure(figsize=(12,8))
    bkgd = fig.add_subplot(111, frameon=False)
    for i_z in range(len(zlo)): 
        sub = fig.add_subplot(2,3,i_z+1) 

        plts = []
        for i_n, name in enumerate(names):  # plot fQ fits
            if name == 'candels': colour = 'k'
            else: colour = 'C'+str(i_n) 
            _plt = sub.fill_between(fq_dict[name][i_z][0], 
                                    fq_dict[name][i_z][1] - fq_dict[name][i_z][2], 
                                    fq_dict[name][i_z][1] + fq_dict[name][i_z][2],
                                    alpha=0.5, color=colour, linewidth=0)
            plts.append(_plt) 
        sub.set_xlim([8.5, 12.]) 
        sub.set_ylim([0., 1.]) 
        if i_z < 3: sub.set_xticklabels([]) 
        if i_z not in [0, 3]: sub.set_yticklabels([]) 
        sub.text(0.95, 0.05, '$'+str(zlo[i_z])+'< z <'+str(zhi[i_z])+'$', 
                ha='right', va='bottom', transform=sub.transAxes, fontsize=20)
        if i_z == 0: 
            sub.legend(plts[:3], lbls[:3], loc='upper left', handletextpad=0.5, prop={'size': 17}) 
        elif i_z == 1: 
            sub.legend(plts[3:], lbls[3:], loc='upper left', handletextpad=0.5, prop={'size': 17}) 
            #sub.text(0.05, 0.95, ' '.join(name.upper().split('_')),
            #        ha='left', va='top', transform=sub.transAxes, fontsize=20)
    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', labelpad=15, fontsize=25) 
    bkgd.set_ylabel(r'Quiescent Fraction ($f_{\rm Q}$)', labelpad=15, fontsize=25) 
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    ffig = os.path.join(dir_fig, 'fq_SAM_comparison_centrals_dlogM%.1f_slope_prior_%.1f_%.1f.pdf' % 
            (dlogM, slope_prior[0], slope_prior[1])) 
    if noise: ffig = ffig.replace('.pdf', '_wnoise_seed%i.pdf' % seed)
    fig.savefig(ffig, bbox_inches='tight')
    fpdf = UT.fig_tex(ffig, pdf=True) 
    fig.savefig(fpdf, bbox_inches='tight')
    return None


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


if __name__=="__main__": 
    # fit SFS for CANDELS
    '''
    for iz in range(1,7): 
        print('--- candels %i of 6 ---' % iz) 
        _ = highzSFSfit('candels', iz, censat='all', dlogM=0.4, slope_prior=[0., 2.], overwrite=True)
        pssfr('candels', iz, censat='all', dlogM=0.4, slope_prior=[0., 2.]) 
        print('--- candels (GOODS only) %i of 6 ---' % iz) 
        _ = highzSFSfit('candels_goods', iz, censat='all', dlogM=0.4, slope_prior=[0., 2.], overwrite=True)
        pssfr('candels_goods', iz, censat='all', dlogM=0.4, slope_prior=[0., 2.]) 
        print('--- candels (CAMILLA SED only) %i of 6 ---' % iz) 
        _ = highzSFSfit('candels_cami', iz, censat='all', dlogM=0.4, slope_prior=[0., 2.], overwrite=True)
        pssfr('candels_cami', iz, censat='all', dlogM=0.4, slope_prior=[0., 2.]) 
    '''
    # fit SFS for sims  
    '''
    for censat in ['all', 'centrals', 'satellites']:
        for name in ['eagle', 'illustris_100myr', 'tng', 'simba', 'sam-light-full', 'sam-light-slice']:
            for iz in range(1,7): 
                print('--- %s %s %i of 6 ---' % (name, censat, iz)) 
                _ = highzSFSfit(name, iz, censat=censat, dlogM=0.4, slope_prior=[0., 2.], overwrite=True)
                pssfr(name, iz, censat=censat, dlogM=0.4, slope_prior=[0., 2.]) 
        SFR_Mstar_comparison(censat=censat, dlogM=0.4, slope_prior=[0., 2.])
    '''
    # fit SFS for sims w/ noise 
    '''
    for censat in ['all', 'centrals', 'satellites']:
        for name in ['eagle', 'illustris_100myr', 'tng', 'simba', 'sam-light-full', 'sam-light-slice']:
            for iz in range(1,7): 
                print('--- %s %s %i of 6 ---' % (name, censat, iz)) 
                _ = highzSFSfit(name, iz, censat=censat, noise=True, dlogM=0.4, slope_prior=[0., 2.], overwrite=True)
                pssfr(name, iz, censat=censat, noise=True, dlogM=0.4, slope_prior=[0., 2.]) 
        SFR_Mstar_comparison(censat=censat, noise=True, seed=1, dlogM=0.4, slope_prior=[0., 2.])
    ''' 
    '''
    for censat in ['all', 'centrals', 'satellites']:
        slope_prior = [0.0, 2.]
        SFS_comparison(censat=censat, dlogM=0.4, slope_prior=slope_prior)
        SFS_comparison(censat=censat, noise=True, seed=1, dlogM=0.4, slope_prior=slope_prior)
    
        SFS_zevo_comparison(censat=censat, dlogM=0.4, slope_prior=slope_prior)
        SFS_zevo_comparison(censat=censat, noise=True, seed=1, dlogM=0.4, slope_prior=slope_prior) 
    
    for censat in ['centrals']:
        #Pssfr_res_impact(censat=censat, n_mc=20, noise=False, seed=1, poisson=False)
        #Pssfr_res_impact(censat=censat, n_mc=20, noise=True, seed=1, poisson=False)
        #Mlim_res_impact(censat=censat, n_mc=20, noise=False, seed=1, threshold=0.2)
        Mlim_res_impact(censat=censat, n_mc=20, noise=True, seed=1, threshold=0.2)

    #Pssfr_res_impact(n_mc=100, noise=True, seed=1, poisson=False)
    # GMM component fraction comparison
    #for censat in ['centrals']:
    #    fcomp_comparison(noise=False, seed=1, dlogM=0.4, slope_prior=[0., 2.])
    #    fcomp_comparison(noise=True, seed=1, dlogM=0.4, slope_prior=[0., 2.])

    fcomp_comparison(noise=False, seed=1, dlogM=0.4, slope_prior=[0., 2.])
    fcomp_comparison(noise=True, seed=1, dlogM=0.4, slope_prior=[0., 2.])
    '''
    # quiescent fraction and 
    '''
    for censat in ['all', 'centrals', 'satellites']:
        slope_prior = [0., 2.]
        QF_comparison(censat=censat, noise=False, seed=1, dlogM=0.4, slope_prior=slope_prior)
        QF_comparison(censat=censat, noise=True, seed=1, dlogM=0.4, slope_prior=slope_prior)

        QF_zevo_comparison(censat=censat, noise=False, seed=1, dlogM=0.4, slope_prior=slope_prior)
        QF_zevo_comparison(censat=censat, noise=True, seed=1, dlogM=0.4, slope_prior=slope_prior)
    '''
    # comparison between SAM slice and full 
    '''
    for prior in [0., 0.2, 0.4]:   
        SFS_SAM_comparison(noise=False, seed=1, dlogM=0.4, slope_prior=[prior, 2.])
        SFS_SAM_comparison(noise=True, seed=1, dlogM=0.4, slope_prior=[prior, 2.])
        QF_SAM_comparison(noise=False, seed=1, dlogM=0.4, dev_thresh=dthresh)
        QF_SAM_comparison(noise=True, seed=1, dlogM=0.4, dev_thresh=dthresh)
    ''' 
    # dSFS 
    for method in ['powerlaw', 'interpexterp', 'broken_powerlaw']: #[]: 
        #for name in ['tng']:#['eagle', 'illustris_100myr', 'tng', 'simba', 'sam-light-full', 'sam-light-slice']:
        #    dSFS(name, censat='centrals', noise=False, seed=1, dlogM=0.4, slope_prior=[0., 2.], method=method)
        #    dSFS(name, censat='centrals', noise=True, seed=1, dlogM=0.4, slope_prior=[0., 2.], method=method)
        #    fQ_dSFS(name, censat='centrals', noise=False, seed=1, dlogM=0.4, slope_prior=[0., 2.], method=method, dSFS_limit=1.)
        #    fQ_dSFS(name, censat='centrals', noise=True, seed=1, dlogM=0.4, slope_prior=[0., 2.], method=method, dSFS_limit=1.)
        fQ_dSFS_comparison(censat='centrals', noise=False, seed=1, dlogM=0.4, slope_prior=[0., 2.], method=method, dSFS_limit=1.)
        fQ_dSFS_comparison(censat='centrals', noise=True, seed=1, dlogM=0.4, slope_prior=[0., 2.], method=method, dSFS_limit=1.)
        fQ_dSFS_zevo_comparison(censat='centrals', noise=False, seed=1, dlogM=0.4, slope_prior=[0., 2.], method=method, dSFS_limit=1.)
        fQ_dSFS_zevo_comparison(censat='centrals', noise=True, seed=1, dlogM=0.4, slope_prior=[0., 2.], method=method, dSFS_limit=1.)
    ##for name in ['eagle']:
    #    probSFS(name, censat='all', noise=False, seed=1, dlogM=0.4, slope_prior=[0., 2.])
    #for method in ['powerlaw', 'interpexterp']: 

    # candels fields comparison 
    '''
    CANDELS_fields(dlogM=0.4, slope_prior=[0., 2.])
    CANDELS_fields_QF(dlogM=0.4, slope_prior=[0., 2.])
    '''
