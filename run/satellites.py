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


def plot_qf_inst_censat(): 
    ''' plot quiescent fraction for satellites 
    '''
    markers = ['x', 's']
    tscale = 'inst'

    fig = plt.figure(figsize=(8, 4)) 
    bkgd = fig.add_subplot(111, frameon=False)
    sub1 = fig.add_subplot(121)
    sub2 = fig.add_subplot(122)
    for i_s, sim in enumerate(['illustris', 'eagle']):#, 'scsam']): 
        # calculate quiescent fraction satellites 
        mmid, qf, err_qf = qf_satellites('%s_%s' % (sim, tscale))
        
        sub1.fill_between(mmid, qf-err_qf, qf+err_qf, 
                alpha=0.3, color='C%i' % (i_s+2), linewidth=0, label=sim)
        sub1.scatter(mmid, qf, marker=markers[i_s], color='white')
        
        mmid, qf, err_qf = qf_centrals('%s_%s' % (sim, tscale))
        
        sub2.fill_between(mmid, qf-err_qf, qf+err_qf, 
                alpha=0.3, color='C%i' % (i_s+2), linewidth=0, label=sim)
        sub2.scatter(mmid, qf, marker=markers[i_s], color='white')

    sub1.set_xlim([8.3, 10.5]) 
    sub1.legend(loc='lower left', frameon=False, handletextpad=0.2, fontsize=20) 
    sub1.set_ylim([0., 1.]) 
    sub1.text(0.05, 0.95, 'satellites', ha='left', va='top', transform=sub1.transAxes, fontsize=20)

    sub2.set_xlim([8.3, 10.5]) 
    sub2.set_ylim([0., 1.]) 
    sub2.set_yticklabels([]) 
    sub2.text(0.05, 0.95, 'centrals', ha='left', va='top', transform=sub2.transAxes, fontsize=20)
    
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', labelpad=15, fontsize=25) 
    bkgd.set_ylabel(r'Quiescent Fraction ($f_{\rm Q}$)', labelpad=15, fontsize=25) 
    fig.subplots_adjust(wspace=0.05)

    ffig = os.path.join(UT.dat_dir(), 'satellites', 'qf_inst.censat.png') 
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


def plot_qf_mhalo(): 
    ''' plot quiescent fraction for satellites as a function of mhalo
    '''
    fig = plt.figure(figsize=(8, 4))
    for i_n, name, sim in zip(range(2), ['illustris_inst', 'eagle_inst'], ['Illustris', 'EAGLE']): 
        logms, logsfr, weights = satellites(name, silent=True) 
        logmh = np.log10(mhalo_satellites(name)) 
        
        nonzero = (logsfr != -99.) & (logsfr != -999.) & (np.isfinite(logsfr)) 

        sub = fig.add_subplot(1,2,i_n+1)
        sub.scatter(logmh[nonzero], logms[nonzero], s=1) 
        sub.set_xlim(10., 15.) 
        sub.set_ylim(8., 12.) 
        if i_n > 0: sub.set_yticklabels([]) 
        sub.text(0.95, 0.05, sim, ha='right', va='bottom', transform=sub.transAxes, fontsize=20)
    
    ffig = os.path.join(UT.dat_dir(), 'satellites', 'mh_ms.png') 
    fig.savefig(ffig, bbox_inches='tight')
    plt.close()

    mhbin_lo = [10.0, 11.5]
    mhbin_hi = [11.5, 14.5]

    markers = ['x', 's']
    fig = plt.figure(figsize=(8, 4)) 
    bkgd = fig.add_subplot(111, frameon=False)
    for i_m in range(len(mhbin_lo)): 
        for i_s, sim in enumerate(['illustris_inst', 'eagle_inst']):
            sub = fig.add_subplot(1,len(mhbin_lo),i_m+1)

            # calculate quiescent fraction satellites 
            mmid, qf, err_qf = qf_satellites('%s' % sim, Mhalo=[mhbin_lo[i_m], mhbin_hi[i_m]])
            
            sub.fill_between(mmid, qf-err_qf, qf+err_qf, alpha=0.3, color='C%i' % (i_s+2), linewidth=0)
            sub.scatter(mmid, qf, marker=markers[i_s], color='white')

            sub.set_xlim([8., 10.5]) 
            sub.set_xticks([8., 9., 10.,]) 
            sub.set_ylim([0., 1.]) 
            if i_m > 0: sub.set_yticklabels([]) 
        sub.text(0.05, 0.05, r'$M_h \in [10^{%.1f}, 10^{%.1f}]$' % (mhbin_lo[i_m], mhbin_hi[i_m]), 
                ha='left', va='bottom', transform=sub.transAxes, fontsize=20)
    sub.legend(loc='lower left', frameon=False, handletextpad=0.2, fontsize=20) 
    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', labelpad=15, fontsize=25) 
    bkgd.set_ylabel(r'Quiescent Fraction ($f_{\rm Q}$)', labelpad=15, fontsize=25) 
    fig.subplots_adjust(wspace=0.05)

    ffig = os.path.join(UT.dat_dir(), 'satellites', 'qf_mhalo.png') 
    fig.savefig(ffig, bbox_inches='tight')
    plt.close()
    return None 


def fcomp_satellites(name, Mhalo=None):
    ''' get the component weights from GMM best-fit. quiescent fraction defined 
    as all components below SFS 
    '''
    # read satellites 
    logms, logsfr, weights = satellites(name, silent=True) 
    if Mhalo is not None: 
        mhalo = mhalo_satellites(name) 
        assert len(mhalo) == len(logms)
        cut = (np.log10(mhalo) > Mhalo[0]) & (np.log10(mhalo) <= Mhalo[1])  
        print('%i galaxies with %.1f < Mh < %.1f' % (np.sum(cut), Mhalo[0], Mhalo[1]))
    else: 
        cut = np.ones(len(logms)).astype(bool) 
    
    nonzero = (logsfr != -99.) & (logsfr != -999.) & (np.isfinite(logsfr)) 
            
    fSFS = sfs_satellites(name) 
    
    # M* bins where SFS is reasonably fit 
    mbin0 = fSFS._mbins[fSFS._has_nbinthresh,0]
    mbin1 = fSFS._mbins[fSFS._has_nbinthresh,1]
    nmbin = len(fSFS._mbins_median)
    assert np.sum(fSFS._has_nbinthresh) == nmbin 
        
    cut_mbin = np.ones(len(mbin0)).astype(bool) 
    for i_m in range(nmbin): 
        inmbin = (logms > mbin0[i_m]) & (logms < mbin1[i_m]) & cut # within bin 
        if np.sum(inmbin) == 0: cut_mbin[i_m] = False
    
    mbin0 = mbin0[cut_mbin]
    mbin1 = mbin1[cut_mbin]
    nmbin = np.sum(cut_mbin)

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
        inmbin      = (logms > mbin0[i_m]) & (logms < mbin1[i_m]) & cut # within bin 
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


def qf_satellites(name, Mhalo=None):
    ''' derive quiescent fraction from GMM best-fit. quiescent fraction defined as all components below SFS 
    '''
    mmid, fcomps, err_fcomps = fcomp_satellites(name, Mhalo=Mhalo)
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


def mhalo_satellites(name, silent=True): 
    ''' get host halo mass for satellites for some simulation 
    '''
    Cat = Cats.Catalog()
    if '_' in name: 
        assert name.split('_')[0] in ['illustris', 'eagle', 'mufasa', 'scsam'] 
        assert name.split('_')[-1] in ['inst', '100myr']
    
        logms, logsfr, weights, censat = Cat.Read(name, keepzeros=True, silent=silent)
        mhalo = Cat.Mhalo_GroupFinder(name)
    else: 
        raise NotImplementedError

    # is satellite 
    is_sat = (censat  == 0) 
    assert np.sum(is_sat) > 0, 'no satellites in sims' 
    
    # impose stellar mass limit 
    mlim = mass_limit(name) 
    in_mlim = (logms >= mlim) 
    
    # combine all the cuts 
    allcuts = (is_sat & in_mlim) 

    return mhalo[allcuts]


def satellites(name, silent=True): 
    ''' get satellites for some simulation 
    '''
    if '_' in name: 
        assert name.split('_')[0] in ['illustris', 'eagle', 'mufasa', 'scsam'] 
        assert name.split('_')[-1] in ['inst', '100myr']

        Cat = Cats.Catalog()

        logms, logsfr, weights, censat = Cat.Read(name, keepzeros=True, silent=silent)
    else: 
        assert name in ['z1illustris100myr', 'z1tng']

        if name == 'z1illustris100myr': 
            f_data = os.path.join(UT.dat_dir(), 'highz', 'Illustris', 'Illustris_z1.txt')
            # M*, SFR 10Myr, SFR 1Gyr, SFR 100Myr, cen/sat
            ms, sfr, censat = np.loadtxt(f_data, skiprows=2, unpack=True, usecols=[0, 3, 4]) 
            logms   = np.log10(ms) 
            logsfr  = np.log10(sfr) 
        elif name == 'z1tng': 
            f_data = os.path.join(UT.dat_dir(), 'highz', 'Illustris', 'IllustrisTNG_z1.txt')
            logms, logsfr, censat = np.loadtxt(f_data, skiprows=2, unpack=True) # log M*, log SFR, cen/sat
        weights = np.ones(len(logms))
    # is satellite 
    is_sat = (censat  == 0) 
    assert np.sum(is_sat) > 0, 'no satellites in sims' 
    
    # impose stellar mass limit 
    mlim = mass_limit(name) 
    in_mlim = (logms >= mlim) 
    
    # combine all the cuts 
    allcuts = (is_sat & in_mlim) 

    return logms[allcuts], logsfr[allcuts], weights[allcuts]


# -- centrals -- 
def fcomp_centrals(name, Mhalo=None):
    ''' get the component weights from GMM best-fit. quiescent fraction defined 
    as all components below SFS 
    '''
    # read satellites 
    logms, logsfr, weights = centrals(name, silent=True) 
    #if Mhalo is not None: 
    #    mhalo = mhalo_satellites(name) 
    #    assert len(mhalo) == len(logms)
    #    cut = (np.log10(mhalo) > Mhalo[0]) & (np.log10(mhalo) <= Mhalo[1])  
    #    print('%i galaxies with %.1f < Mh < %.1f' % (np.sum(cut), Mhalo[0], Mhalo[1]))
    #else: 
    cut = np.ones(len(logms)).astype(bool) 
    
    nonzero = (logsfr != -99.) & (logsfr != -999.) & (np.isfinite(logsfr)) 
            
    fSFS = sfs_centrals(name) 
    
    # M* bins where SFS is reasonably fit 
    mbin0 = fSFS._mbins[fSFS._has_nbinthresh,0]
    mbin1 = fSFS._mbins[fSFS._has_nbinthresh,1]
    nmbin = len(fSFS._mbins_median)
    assert np.sum(fSFS._has_nbinthresh) == nmbin 
        
    cut_mbin = np.ones(len(mbin0)).astype(bool) 
    for i_m in range(nmbin): 
        inmbin = (logms > mbin0[i_m]) & (logms < mbin1[i_m]) & cut # within bin 
        if np.sum(inmbin) == 0: cut_mbin[i_m] = False
    
    mbin0 = mbin0[cut_mbin]
    mbin1 = mbin1[cut_mbin]
    nmbin = np.sum(cut_mbin)

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
        inmbin      = (logms > mbin0[i_m]) & (logms < mbin1[i_m]) & cut # within bin 
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


def qf_centrals(name, Mhalo=None):
    ''' derive quiescent fraction from GMM best-fit. quiescent fraction defined as all components below SFS 
    '''
    mmid, fcomps, err_fcomps = fcomp_centrals(name, Mhalo=Mhalo)
    f_Q = fcomps[0,:] + fcomps[2,:] + fcomps[3,:]
    err_f_Q = np.sqrt(err_fcomps[0,:]**2 + err_fcomps[2,:]**2 + err_fcomps[3,:]**2)
    return mmid, f_Q, err_f_Q


def sfs_centrals(name): 
    ''' sfs fit to the satellite population
    '''
    # read satellites 
    logms, logsfr, weights = centrals(name, silent=True) 
    nonzero = (logsfr != -99.) & (logsfr != -999.) & (np.isfinite(logsfr)) 
    print('%i centrals with SFR > 0 in %s' % (np.sum(nonzero), name)) 

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


def centrals(name, silent=True): 
    ''' get centrals for some simulation 
    '''
    if '_' in name: 
        assert name.split('_')[0] in ['illustris', 'eagle', 'mufasa', 'scsam'] 
        assert name.split('_')[-1] in ['inst', '100myr']

        Cat = Cats.Catalog()

        logms, logsfr, weights, censat = Cat.Read(name, keepzeros=True, silent=silent)
    else: 
        assert name in ['z1illustris100myr', 'z1tng']

        if name == 'z1illustris100myr': 
            f_data = os.path.join(UT.dat_dir(), 'highz', 'Illustris', 'Illustris_z1.txt')
            # M*, SFR 10Myr, SFR 1Gyr, SFR 100Myr, cen/sat
            ms, sfr, censat = np.loadtxt(f_data, skiprows=2, unpack=True, usecols=[0, 3, 4]) 
            logms   = np.log10(ms) 
            logsfr  = np.log10(sfr) 
        elif name == 'z1tng': 
            f_data = os.path.join(UT.dat_dir(), 'highz', 'Illustris', 'IllustrisTNG_z1.txt')
            logms, logsfr, censat = np.loadtxt(f_data, skiprows=2, unpack=True) # log M*, log SFR, cen/sat
        weights = np.ones(len(logms))
    # is satellite 
    is_cen = (censat == 1) 
    assert np.sum(is_cen) > 0, 'no centrals in sims' 
    
    # impose stellar mass limit 
    mlim = mass_limit(name) 
    in_mlim = (logms >= mlim) 
    
    # combine all the cuts 
    allcuts = (is_cen & in_mlim) 

    return logms[allcuts], logsfr[allcuts], weights[allcuts]


def mass_limit(name): 
    ''' mass limit of simulation set by resolution limit of the sims
    or mass limit observational samples 
    '''
    mlim_dict = {
            'illustris': 8.4, 
            'eagle': 8.4, 
            'mufasa': 9.2, 
            'scsam': 8.8,
            'z1illustris100myr': 8.4, 
            'z1tng': 8.4 
            } 
    sim = name.split('_')[0] 
    return mlim_dict[sim] 


# --- appendix --- 
def plot_sfr_mstar_illustrises(): 
    ''' plot SFR - M* relation for Illustris and Illustris TNG satellites 
    '''
    fig = plt.figure(figsize=(10,5)) 
    
    i_z = 1 # z ~ 0.75

    for i_s, sim in enumerate(['z1illustris100myr', 'z1tng']): 
        logms, logsfr, _ = satellites(sim, silent=False)
        notzero = np.isfinite(logsfr)

        cut     = (notzero) 
        logms   = logms[cut]
        logsfr  = logsfr[cut]

        # plot 
        sub = fig.add_subplot(1,2,i_s+1)
        sub.text(0.95, 0.05, ['Illustris', 'TNG'][i_s], ha='right', va='bottom', transform=sub.transAxes, fontsize=20)
        
        DFM.hist2d(logms, logsfr, color='C%i' % (i_s+2), 
                levels=[0.68, 0.95], range=[[7.8, 12.], [-4., 2.]], 
                plot_datapoints=True, fill_contours=False, plot_density=True, ax=sub) 
        sub.set_xlim([7.8, 11.8]) 
        sub.set_xticks([8., 9., 10., 11.]) 
        if i_s != 0: sub.set_yticklabels([]) 
        sub.set_ylim([-3., 2.]) 
        sub.set_yticks([-3, -2., -1., 0., 1, 2.]) 

    fig.text(0.5, -0.02, r'log$\; M_* \;\;[M_\odot]$', ha='center', fontsize=25) 
    fig.text(0.04, 0.5, r'log ( SFR $[M_\odot \, yr^{-1}]$ )', rotation='vertical', va='center', fontsize=25) 
    fig.subplots_adjust(wspace=0.05, hspace=0.05)

    ffig = os.path.join(UT.dat_dir(), 'satellites', 'sfr_mstar.illustrises.png') 
    fig.savefig(ffig, bbox_inches='tight')
    plt.close()
    return None 


def plot_sfs_illustrises(): 
    ''' plot SFR - M* relation for satellites 
    '''
    fig = plt.figure(figsize=(10,5)) 

    for i_s, sim in enumerate(['z1illustris100myr', 'z1tng']): 
        # read satellites 
        logms, logsfr, weights = satellites(sim, silent=True) 
        
        # SFS 
        fsfs = sfs_satellites(sim)

        # plot 
        sub = fig.add_subplot(1,2,i_s+1)
        sub.text(0.95, 0.05, ['Illustris', 'TNG'][i_s], ha='right', va='bottom', transform=sub.transAxes, fontsize=20)
        
        DFM.hist2d(logms, logsfr, color='C%i' % (i_s+2), 
                levels=[0.68, 0.95], range=[[7.8, 12.], [-4., 2.]], 
                plot_datapoints=True, fill_contours=False, plot_density=True, ax=sub) 
        sub.errorbar(fsfs._fit_logm, fsfs._fit_logsfr, yerr=fsfs._fit_err_logssfr, fmt='.k') 

        sub.set_xlim([7.8, 11.8]) 
        sub.set_xticks([8., 9., 10., 11.]) 
        if i_s != 0: sub.set_yticklabels([]) 
        sub.set_ylim([-3., 2.]) 
        sub.set_yticks([-3, -2., -1., 0., 1, 2.]) 

    fig.text(0.5, -0.02, r'log$\; M_* \;\;[M_\odot]$', ha='center', fontsize=25) 
    fig.text(0.04, 0.5, r'log ( SFR $[M_\odot \, yr^{-1}]$ )', rotation='vertical', va='center', fontsize=25) 
    fig.subplots_adjust(wspace=0.05, hspace=0.05)

    ffig = os.path.join(UT.dat_dir(), 'satellites', 'sfs.illustrises.png') 
    fig.savefig(ffig, bbox_inches='tight')
    plt.close()
    return None 


def plot_qf_illustrises(): 
    ''' plot quiescent fraction for satellites 
    '''
    markers = ['x', 's']
    fig = plt.figure(figsize=(4, 4)) 
    sub = fig.add_subplot(111)
    bkgd = fig.add_subplot(111, frameon=False)

    for i_s, sim in enumerate(['z1illustris100myr', 'z1tng']): 
        # calculate quiescent fraction satellites 
        mmid, qf, err_qf = qf_satellites(sim)
            
        sub.fill_between(mmid, qf-err_qf, qf+err_qf, 
                alpha=0.3, color='C%i' % (i_s+2), linewidth=0, label=['Illustris', 'TNG'][i_s])
        sub.scatter(mmid, qf, marker=markers[i_s], color='white')
    
    sub.legend(loc='upper left', fontsize=15, frameon=False)
    sub.set_xlim([9., 10.75]) 
    sub.set_xticks([9., 10.]) 
    sub.set_ylim(0, 1) 
    sub.text(0.05, 0.05, r'$z\sim0.75$', ha='left', va='bottom', transform=sub.transAxes, fontsize=20)

    bkgd.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    bkgd.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', labelpad=15, fontsize=25) 
    bkgd.set_ylabel(r'Quiescent Fraction ($f_{\rm Q}$)', labelpad=15, fontsize=25) 
    fig.subplots_adjust(wspace=0.05)

    ffig = os.path.join(UT.dat_dir(), 'satellites', 'qf.illustrises.png') 
    fig.savefig(ffig, bbox_inches='tight')
    plt.close()
    return None 


if __name__=="__main__": 
    # plot SFR-M* relation of the satellites
    #plot_sfr_mstar()

    # plot SFS of the satellites
    #plot_sfs()

    # plot QF of the satellites
    #plot_qf()
    #plot_qf_inst() # instant SFR only 
    plot_qf_inst_censat()
    #plot_qf_mhalo()

    #plot_sfr_mstar_illustrises()
    #plot_sfs_illustrises()
    #plot_qf_illustrises()
