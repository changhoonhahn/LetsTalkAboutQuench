'''

examining satellites in simulations


'''
import os
import numpy as np 
import corner as DFM 
# -- letstalkaboutquench --
from letstalkaboutquench import util as UT
from letstalkaboutquench import catalogs as Cats
# -- starFS -- 
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


def plot_sfr_mstar(): 
    ''' plot SFR - M* relation for satellites 
    '''
    fig = plt.figure(figsize=(12, 7)) 

    for i_t, tscale in enumerate(['inst', '100myr']): 
        for i_s, sim in enumerate(['illustris', 'eagle', 'scsam']): 

            # read satellites 
            logms, logsfr, weights = satellites('%s_%s' % (sim, tscale), silent=True) 

            # plot 
            sub = fig.add_subplot(2,3,3*i_t+i_s+1)
            if i_s == 0: 
                sub.text(0.05, 0.95, 'SFR [%s]' % tscale, 
                        ha='left', va='top', transform=sub.transAxes, fontsize=20)
            if i_t == 1: 
                sub.text(0.95, 0.05, sim, ha='right', va='bottom', 
                        transform=sub.transAxes, fontsize=20)
            
            DFM.hist2d(logms, logsfr, color='C%i' % (i_s+2), 
                    levels=[0.68, 0.95], range=[[7.8, 12.], [-4., 2.]], 
                    plot_datapoints=True, fill_contours=False, plot_density=True, ax=sub) 
            sub.set_xlim([7.8, 11.8]) 
            sub.set_xticks([8., 9., 10., 11.]) 
            if i_t == 0: sub.set_xticklabels([]) 
            if i_s != 0: sub.set_yticklabels([]) 
            sub.set_ylim([-4., 1.5]) 
            sub.set_yticks([-4., -3, -2., -1., 0., 1]) 

    fig.text(0.5, 0.00, r'log$\; M_* \;\;[M_\odot]$', ha='center', fontsize=25) 
    fig.text(0.07, 0.5, r'log ( SFR $[M_\odot \, yr^{-1}]$ )', rotation='vertical', va='center', fontsize=25) 
    fig.subplots_adjust(wspace=0.05, hspace=0.05)

    ffig = os.path.join(UT.dat_dir(), 'satellites', 'sfr_mstar.png') 
    fig.savefig(ffig, bbox_inches='tight')
    plt.close()
    return None 


def plot_sfs(): 
    ''' plot SFR - M* relation for satellites 
    '''
    fig = plt.figure(figsize=(12, 7)) 

    for i_t, tscale in enumerate(['inst', '100myr']): 
        for i_s, sim in enumerate(['illustris', 'eagle', 'scsam']): 

            # read satellites 
            logms, logsfr, weights = satellites('%s_%s' % (sim, tscale), silent=True) 
            
            # SFS 
            fsfs = sfs_satellites('%s_%s' % (sim, tscale)) 
            if sim == 'mufasa': 
                print(logsfr.min(), logsfr.max()) 
                print(fsfs._fit_logm, fsfs._fit_logsfr) 

            # plot 
            sub = fig.add_subplot(2,3,3*i_t+i_s+1)
            if i_s == 0: 
                sub.text(0.05, 0.95, 'SFR [%s]' % tscale, 
                        ha='left', va='top', transform=sub.transAxes, fontsize=20)
            if i_t == 1: 
                sub.text(0.95, 0.05, sim, ha='right', va='bottom', 
                        transform=sub.transAxes, fontsize=20)
            
            DFM.hist2d(logms, logsfr, color='C%i' % (i_s+2), 
                    levels=[0.68, 0.95], range=[[7.8, 12.], [-4., 2.]], 
                    plot_datapoints=True, fill_contours=False, plot_density=True, ax=sub) 
            sub.errorbar(fsfs._fit_logm, fsfs._fit_logsfr, yerr=fsfs._fit_err_logssfr, fmt='.k') 

            sub.set_xlim([7.8, 11.8]) 
            sub.set_xticks([8., 9., 10., 11.]) 
            if i_t == 0: sub.set_xticklabels([]) 
            if i_s != 0: sub.set_yticklabels([]) 
            sub.set_ylim([-4., 1.5]) 
            sub.set_yticks([-4., -3, -2., -1., 0., 1]) 

    fig.text(0.5, 0.00, r'log$\; M_* \;\;[M_\odot]$', ha='center', fontsize=25) 
    fig.text(0.07, 0.5, r'log ( SFR $[M_\odot \, yr^{-1}]$ )', rotation='vertical', va='center', fontsize=25) 
    fig.subplots_adjust(wspace=0.05, hspace=0.05)

    ffig = os.path.join(UT.dat_dir(), 'satellites', 'sfs.png') 
    fig.savefig(ffig, bbox_inches='tight')
    plt.close()
    return None 


def plot_qf_inst(): 
    ''' plot quiescent fraction for satellites 
    '''
    markers = ['x', 's']
    tscale = 'inst'

    fig = plt.figure(figsize=(4, 4)) 
    sub = fig.add_subplot(111)
    for i_s, sim in enumerate(['illustris', 'eagle']):#, 'scsam']): 
        # calculate quiescent fraction satellites 
        mmid, qf, err_qf = qf_satellites('%s_%s' % (sim, tscale))
        
        sub.fill_between(mmid, qf-err_qf, qf+err_qf, 
                alpha=0.3, color='C%i' % (i_s+2), linewidth=0, label=sim)
        sub.scatter(mmid, qf, marker=markers[i_s], color='white')

    sub.set_xlim([8.3, 10.5]) 
    sub.legend(loc='lower left', frameon=False, handletextpad=0.2, fontsize=20) 
    sub.set_ylim([0., 1.]) 
    sub.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', labelpad=15, fontsize=25) 
    sub.set_ylabel(r'Quiescent Fraction ($f_{\rm Q}$)', labelpad=15, fontsize=25) 

    ffig = os.path.join(UT.dat_dir(), 'satellites', 'qf_inst.png') 
    fig.savefig(ffig, bbox_inches='tight')
    plt.close()
    return None 


def plot_qf(): 
    ''' plot quiescent fraction for satellites 
    '''
    markers = ['x', 's']
    fig = plt.figure(figsize=(8, 4)) 
    bkgd = fig.add_subplot(111, frameon=False)

    for i_t, tscale in enumerate(['inst', '100myr']): 
        sub = fig.add_subplot(1,2,i_t+1)
        for i_s, sim in enumerate(['illustris', 'eagle']):#, 'scsam']): 
            # calculate quiescent fraction satellites 
            mmid, qf, err_qf = qf_satellites('%s_%s' % (sim, tscale))
            
            sub.fill_between(mmid, qf-err_qf, qf+err_qf, 
                    alpha=0.3, color='C%i' % (i_s+2), linewidth=0, label=sim)
            sub.scatter(mmid, qf, marker=markers[i_s], color='white')
        # plot 
        sub.text(0.05, 0.95, 'SFR [%s]' % tscale, ha='left', va='top', transform=sub.transAxes, fontsize=20)

        sub.set_xlim([8., 10.5]) 
        sub.set_xticks([8., 9., 10.,]) 
        if i_t != 0: sub.set_yticklabels([]) 
        else: sub.legend(loc='lower left', frameon=False, handletextpad=0.2, fontsize=20) 
        sub.set_ylim([0., 1.]) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', labelpad=15, fontsize=25) 
    bkgd.set_ylabel(r'Quiescent Fraction ($f_{\rm Q}$)', labelpad=15, fontsize=25) 
    fig.subplots_adjust(wspace=0.05)

    ffig = os.path.join(UT.dat_dir(), 'satellites', 'qf.png') 
    fig.savefig(ffig, bbox_inches='tight')
    plt.close()
    return None 


def fcomp_satellites(name):
    ''' get the component weights from GMM best-fit. quiescent fraction defined 
    as all components below SFS 
    '''
    # read satellites 
    logms, logsfr, weights = satellites(name, silent=True) 
    
    nonzero = (logsfr != -99.) & (logsfr != -999.) & (np.isfinite(logsfr)) 
            
    fSFS = sfs_satellites(name) 
    
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
        inmbin      = (logms > mbin0[i_m]) & (logms < mbin1[i_m]) # within bin 
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


def qf_satellites(name):
    ''' derive quiescent fraction from GMM best-fit. quiescent fraction defined as all components below SFS 
    '''
    mmid, fcomps, err_fcomps = fcomp_satellites(name)
    f_Q = fcomps[0,:] + fcomps[2,:] + fcomps[3,:]
    err_f_Q = np.sqrt(err_fcomps[0,:]**2 + err_fcomps[2,:]**2 + err_fcomps[3,:]**2)
    return mmid, f_Q, err_f_Q


def sfs_satellites(name): 
    ''' sfs fit to the satellite population
    '''
    # read satellites 
    logms, logsfr, weights = satellites(name, silent=True) 
    nonzero = (logsfr != -99.) & (logsfr != -999.) & (np.isfinite(logsfr)) 
    print('%i satellites with SFR > 0 in %s' % (np.sum(nonzero), name)) 

    # fit the SFS
    fSFS = sFS(fit_range=[mass_limit(name), 12.0]) # stellar mass range

    sfs_fit = fSFS.fit(logms[nonzero], logsfr[nonzero], 
                method='gaussmix',      # Gaussian Mixture Model fitting 
                dlogm = 0.2,            # stellar mass bins of 0.2 dex
                slope_prior = [0., 2.], # slope prior 
                Nbin_thresh=100,        # at least 100 galaxies in bin 
                error_method='bootstrap',  # uncertainty estimate method 
                n_bootstrap=100)        # number of bootstrap bins
    return fSFS 


def satellites(name, silent=True): 
    ''' get satellites for some simulation 
    '''
    assert '_' in name
    assert name.split('_')[0] in ['illustris', 'eagle', 'mufasa', 'scsam'] 
    assert name.split('_')[-1] in ['inst', '100myr']

    Cat = Cats.Catalog()

    logms, logsfr, weights, censat = Cat.Read(name, keepzeros=True, silent=silent)

    # is satellite 
    is_sat = (censat  == 0) 
    assert np.sum(is_sat) > 0, 'no satellites in sims' 
    
    # impose stellar mass limit 
    mlim = mass_limit(name) 
    in_mlim = (logms >= mlim) 
    
    # combine all the cuts 
    allcuts = (is_sat & in_mlim) 

    return logms[allcuts], logsfr[allcuts], weights[allcuts]


def mass_limit(name): 
    ''' mass limit of simulation set by resolution limit of the sims
    or mass limit observational samples 
    '''
    mlim_dict = {
            'illustris': 8.4, 
            'eagle': 8.4, 
            'mufasa': 9.2, 
            'scsam': 8.8
            } 
    sim = name.split('_')[0] 
    return mlim_dict[sim] 


if __name__=="__main__": 
    # plot SFR-M* relation of the satellites
    #plot_sfr_mstar()

    # plot SFS of the satellites
    #plot_sfs()

    # plot QF of the satellites
    #plot_qf()
    plot_qf_inst() # instant SFR only 
