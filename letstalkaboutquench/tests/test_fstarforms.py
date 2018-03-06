'''
'''
import numpy as np 
from scipy import linalg
from scipy.stats import multivariate_normal as MNorm
import corner as DFM 

import env
from catalogs import Catalog as Cat
from fstarforms import fstarforms
from fstarforms import xdGMM 
from fstarforms import sfr_mstar_gmm
import util as UT
import corner as DFM 

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


def assess_SFMS_fit(catalog, fit_method):
    ''' Assess the quality of the SFMS fits by comparing 
    to the actual P(SFR) at each of the mass bins. 
    '''
    # read in catalogs
    cat = Cat()
    logm, logsfr, w = cat.Read(catalog)  
    
    # fit the SFMS using whatever method 
    fSFMS = fstarforms()
    fit_logm, fit_logsfr = fSFMS.fit(logm, logsfr, method=fit_method) 
    F_sfms = fSFMS.powerlaw()
    
    # common sense cuts imposed in fSFMS
    logm = logm[fSFMS._sensecut]
    logsfr = logsfr[fSFMS._sensecut]

    n_col = int(np.ceil(float(len(fit_logm)+1)/2))

    fig = plt.figure(1, figsize=(5*n_col, 8))
    # plot log SFR - log M* relation 
    sub = fig.add_subplot(2,n_col,1)
    DFM.hist2d(logm, logsfr, color='black', #color='#1F77B4', 
            ax=sub, levels=[0.68, 0.95], range=[[int(np.floor(logm.min()/0.5))*0.5, int(np.ceil(logm.max()/0.5))*0.5], [-4., 2.]], 
            fill_contours=False) 
    sub.scatter(fit_logm, fit_logsfr, c='b', marker='x', lw=3, s=40)
    sub.text(0.9, 0.1, cat.CatalogLabel(catalog),
            ha='right', va='center', transform=sub.transAxes, fontsize=20)
    sub.set_ylabel(r'log SFR  $[M_\odot / yr]$', fontsize=20) 
    sub.set_xlabel(r'log$\; M_* \;\;[M_\odot]$', fontsize=20) 

    for i_m in range(len(fit_logm)):  
        sub = fig.add_subplot(2, n_col, i_m+2)

        in_mbin = np.where((logm > fit_logm[i_m]-0.5*fSFMS._dlogm) & 
                (logm < fit_logm[i_m]+0.5*fSFMS._dlogm))
        sub.text(0.1, 0.9, 
                str(round(fit_logm[i_m]-0.5*fSFMS._dlogm,1))+'$<$ log$\,M_* <$'+str(round(fit_logm[i_m]+0.5*fSFMS._dlogm,1)), 
                ha='left', va='center', transform=sub.transAxes, fontsize=20)

        _ = sub.hist(logsfr[in_mbin] - logm[in_mbin], bins=32, 
                range=[-14., -8.], normed=True, histtype='step', color='k', linewidth=1.75)

        # plot the fits 
        xx = np.linspace(-14, -8, 50) 
        if 'gaussfit' in fit_method: 
            gaus = lambda xx, aa, x0, sig: aa * np.exp(-(xx - x0)**2/(2*sig**2))
            popt = fSFMS._fit_popt[i_m]
            sub.plot(xx, fSFMS._fit_amp[i_m]*gaus(xx, *popt), c='r', lw=2, ls='--')
        elif 'negbinomfit' in fit_method: # negative binomial distribution 
            NB_fit = lambda xx, aa, mu, theta: UT.NB_pdf_logx(np.power(10., xx+aa), mu, theta)
            popt = fSFMS._fit_popt[i_m]
            sub.plot(xx, fSFMS._fit_amp[i_m]*NB_fit(xx, *popt), c='r', lw=2, ls='--')
            #sub.vlines(np.log10(popt[1])-popt[0], 0., 2., color='r', linewidth=2)
        elif 'gaussmix' in fit_method:
            gmm_weights = fSFMS._gmix_weights[i_m]
            gmm_means = fSFMS._gmix_means[i_m]
            gmm_vars = fSFMS._gmix_covariances[i_m]


            for ii, icomp in enumerate(gmm_means.argsort()[::-1]): 
                if ii == 0: 
                    sub.plot(xx, gmm_weights[icomp]*MNorm.pdf(xx, gmm_means[icomp], gmm_vars[icomp]), lw=2, ls='--')
                    gmm_tot = gmm_weights[icomp]*MNorm.pdf(xx, gmm_means[icomp], gmm_vars[icomp])
                else: 
                    sub.plot(xx, gmm_weights[icomp]*MNorm.pdf(xx, gmm_means[icomp], gmm_vars[icomp]), lw=1.5, ls='--')
                    gmm_tot += gmm_weights[icomp]*MNorm.pdf(xx, gmm_means[icomp], gmm_vars[icomp])
            sub.plot(xx, gmm_tot, c='r', lw=2, ls='-')

        # x-axis
        sub.set_xlim([-13, -8])
        sub.set_xticks([-13, -12, -11, -10, -9, -8])
        sub.set_ylim([0., 1.4])
        sub.set_yticks([0., 0.5, 1.])

        sub.set_ylabel(r'P(log SSFR  $[yr^{-1}])$', fontsize=20) 
        sub.set_xlabel(r'log SSFR  $[yr^{-1}]$', fontsize=20) 

    fig.subplots_adjust(hspace=0.3, wspace=0.4)
    fig_name = ''.join([UT.fig_dir(), 'Pssfr_fit.assess.', catalog, '.', fit_method, '.png'])
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close() 
    return None


def assess_SFMS_fit_dlogM(catalog, fit_method, mbin=[9.7, 9.9]):
    ''' Assess the quality of the SFMS fits by comparing 
    to the actual P(SFR) for a specified mass bins. 
    '''
    # read in catalogs
    cat = Cat()
    _logm, _logsfr, w, censat = cat.Read(catalog)  
    iscen = (censat == 1) 
    logm = _logm[iscen] 
    logsfr = _logsfr[iscen] 
    
    # fit the SFMS using whatever method 
    fSFMS = fstarforms()
    fit_logm, fit_logsfr = fSFMS.fit(logm, logsfr, method=fit_method, fit_range=mbin, dlogm=0.2) 
    print fit_logm, fit_logsfr
    F_sfms = fSFMS.powerlaw()
    
    # common sense cuts imposed in fSFMS
    logm = logm[fSFMS._sensecut]
    logsfr = logsfr[fSFMS._sensecut]

    n_col = int(np.ceil(float(len(fit_logm)+1)/2))

    fig = plt.figure(1, figsize=(10,4))
    # plot log SFR - log M* relation 
    sub = fig.add_subplot(121)
    DFM.hist2d(logm, logsfr, color='#ee6a50',
            ax=sub, levels=[0.68, 0.95], 
            range=[[int(np.floor(logm.min()/0.5))*0.5, int(np.ceil(logm.max()/0.5))*0.5], 
                [-4., 2.]], 
            fill_contours=False) 
    sub.scatter([fit_logm[0]], [fit_logsfr[0]], c='k', marker='x', lw=3, s=40)
    sub.fill_between(mbin, [2.,2.], [-5.,-5], color='k', linewidth=0, alpha=0.25)
    sub.text(0.9, 0.1, cat.CatalogLabel(catalog),
            ha='right', va='center', transform=sub.transAxes, fontsize=20)
    sub.set_ylabel(r'log SFR  $[M_\odot / yr]$', fontsize=20) 
    sub.set_xlabel(r'log$\; M_* \;\;[M_\odot]$', fontsize=20) 

    sub = fig.add_subplot(122)
    i_m = 0 
    in_mbin = np.where((logm > fit_logm[i_m]-0.5*fSFMS._dlogm) & 
            (logm < fit_logm[i_m]+0.5*fSFMS._dlogm))
    sub.text(0.9, 0.9, 
            str(round(fit_logm[i_m]-0.5*fSFMS._dlogm,1))+'$<$ log$\,M_* <$'+str(round(fit_logm[i_m]+0.5*fSFMS._dlogm,1)), 
            ha='right', va='center', transform=sub.transAxes, fontsize=20)

    _ = sub.hist(logsfr[in_mbin] - logm[in_mbin], bins=32, 
            range=[-14., -8.], normed=True, histtype='step', color='k', linewidth=1.75)
    sub.vlines(fit_logsfr[0] - fit_logm[0], 0., 1.5, color='k') 

    # plot the fits 
    xx = np.linspace(-14, -8, 50) 
    if 'gaussfit' in fit_method: 
        gaus = lambda xx, aa, x0, sig: aa * np.exp(-(xx - x0)**2/(2*sig**2))
        popt = fSFMS._fit_popt[i_m]
        sub.plot(xx, fSFMS._fit_amp[i_m]*gaus(xx, *popt), c='r', lw=2, ls='--')
    elif 'negbinomfit' in fit_method: # negative binomial distribution 
        NB_fit = lambda xx, aa, mu, theta: UT.NB_pdf_logx(np.power(10., xx+aa), mu, theta)
        popt = fSFMS._fit_popt[i_m]
        sub.plot(xx, fSFMS._fit_amp[i_m]*NB_fit(xx, *popt), c='r', lw=2, ls='--')
        #sub.vlines(np.log10(popt[1])-popt[0], 0., 2., color='r', linewidth=2)
    elif 'gaussmix' in fit_method:
        gmm_weights = fSFMS._gmix_weights[i_m]
        gmm_means = fSFMS._gmix_means[i_m]
        gmm_vars = fSFMS._gmix_covariances[i_m]
    
        if len(gmm_means) == 3: 
            colours = ['C0', 'C2', 'C1']
        elif len(gmm_means) == 2: 
            colours = ['C0', 'C1']
        else: 
            colours = ['C0']
        for ii, icomp in enumerate(gmm_means.argsort()[::-1]): 
            if ii == 0: 
                sub.plot(xx, gmm_weights[icomp]*MNorm.pdf(xx, gmm_means[icomp], gmm_vars[icomp]), lw=2, ls='--', c=colours[ii])
                gmm_tot = gmm_weights[icomp]*MNorm.pdf(xx, gmm_means[icomp], gmm_vars[icomp])
            else: 
                sub.plot(xx, gmm_weights[icomp]*MNorm.pdf(xx, gmm_means[icomp], gmm_vars[icomp]), lw=1.5, ls='--', c=colours[ii])
                gmm_tot += gmm_weights[icomp]*MNorm.pdf(xx, gmm_means[icomp], gmm_vars[icomp])
        if len(gmm_means) > 1: 
            sub.plot(xx, gmm_tot, c='k', lw=1.5, ls=':')

    # x-axis
    sub.set_xlim([-13, -8])
    sub.set_xticks([-13, -12, -11, -10, -9, -8])
    sub.set_ylim([0., 1.4])
    sub.set_yticks([0., 0.5, 1.])

    sub.set_ylabel(r'P(log SSFR  $[yr^{-1}])$', fontsize=20) 
    sub.set_xlabel(r'log SSFR  $[yr^{-1}]$', fontsize=20) 

    str_mbin = '_'.join([str(mm) for mm in mbin])
    fig.subplots_adjust(wspace=0.3)
    fig_name = ''.join([UT.fig_dir(), 'Pssfr_fit.assess.', catalog, '.', fit_method, 
        '.', str_mbin, '.png'])
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close() 
    return None


def SFR_Mstar_Catalogs(fit_method, contour='dfm'): 
    ''' Compare SFR vs M* relation of various simlations and data
    '''
    Cata = Cat()
    # Read in various data sets
    catalog_list = ['tinkergroup', 
            'nsa_dickey', 
            'illustris_1gyr', #'illustris_10myr', 
            'eagle_1gyr', #'eagle_10myr', 
            'mufasa_1gyr']#, 'mufasa_10myr']

    catalog_labels = [] # Labels 
    logSFRs, logMstars, weights = [], [], [] 
    for cat in catalog_list: 
        catalog_labels.append(Cata.CatalogLabel(cat))
        logMstar, logSFR, weight = Cata.Read(cat)
        logSFRs.append(logSFR)
        logMstars.append(logMstar) 
        weights.append(weight)
    
    n_cols = int(np.ceil(np.float(len(logSFRs))/2.))
    fig = plt.figure(1, figsize=(14,8))
    bkgd = fig.add_subplot(111, frameon=False)

    for i_data in range(len(logSFRs)):
        sub = fig.add_subplot(2, n_cols, i_data+1)
        if contour == 'dfm':  
            colour = 'C'+str(i_data)
            DFM.hist2d(logMstars[i_data], logSFRs[i_data], color=colour, 
                    levels=[0.68, 0.95], range=[[7., 12.], [-4., 2.]], 
                    plot_datapoints=True, fill_contours=False, plot_density=True, ax=sub) 
        elif contour == 'gaussianKDE': 
            xx, yy, f = UT.gaussianKDE_contour(logMstars[i_data], logSFRs[i_data], 
                    xmin=6., xmax=12., ymin=-4., ymax=2.)
            sub.contourf(xx, yy, f, cmap='Blues')
            sub.contour(xx, yy, f, colors='k')
            sub.set_xlim([7., 12.]) 
            sub.set_ylim([-4., 2.]) 
        elif contour == False: 
            sub.scatter(logMstars[i_data], logSFRs[i_data], c='k', s=2, lw=0)
            sub.set_xlim([7., 12.]) 
            sub.set_ylim([-4., 2.]) 
        else: 
            raise ValueError() 
        sub.text(0.9, 0.1, catalog_labels[i_data],
                ha='right', va='center', transform=sub.transAxes, fontsize=20)
        #sub.plot(np.linspace(6., 12., 10), np.linspace(6., 12., 10) - 11., ls='--', c='k') 

        fSFMS = fstarforms()
        fit_logm, fit_logsfr = fSFMS.fit(logMstars[i_data], logSFRs[i_data], method=fit_method) 
        sub.scatter(fit_logm, fit_logsfr, c='k', marker='x', lw=3, s=40)

    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel(r'log $M_* \;\;[M_\odot]$', labelpad=20, fontsize=30) 
    bkgd.set_ylabel(r'log SFR $[M_\odot \, yr^{-1}]$', labelpad=20, fontsize=30) 
    
    fig.subplots_adjust(wspace=0.15, hspace=0.15)
    
    fig_name = ''.join([UT.fig_dir(), 'SFR_Mstar_catalogs.', contour, '.png'])
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close()
    return None 


def assess_SFMS_dMS(catalog, fit_method):
    ''' Assess the quality of the SFMS fits by comparing 
    to the actual P(SFR) at each of the mass bins. 
    '''
    # read in catalogs
    cat = Cat()
    logm, logsfr, w = cat.Read(catalog)  
    
    # fit the SFMS using whatever method 
    fSFMS = fstarforms()
    fit_logm, fit_logsfr = fSFMS.fit(logm, logsfr, method=fit_method, dlogm=0.5) 
    F_sfms = fSFMS.powerlaw()
    dMS = fSFMS.d_MS(logm, logsfr) 
    
    # common sense cuts imposed in fSFMS
    logm = logm[fSFMS._sensecut]
    logsfr = logsfr[fSFMS._sensecut]

    n_col = int(np.ceil(float(len(fit_logm)+1)/2))

    fig = plt.figure(1, figsize=(10, 4))
    # plot log SFR - log M* relation 
    sub = fig.add_subplot(121)
    DFM.hist2d(logm, logsfr, color='#ee6a50',
            ax=sub, levels=[0.68, 0.95], range=[[int(np.floor(logm.min()/0.5))*0.5, int(np.ceil(logm.max()/0.5))*0.5], [-4., 2.]], 
            fill_contours=False) 
    #sub.scatter(fit_logm, fit_logsfr, c='b', marker='x', lw=3, s=40)
    sub.plot(fit_logm, F_sfms(fit_logm), c='k', lw=2, ls='--')
    sub.text(0.9, 0.1, cat.CatalogLabel(catalog),
            ha='right', va='center', transform=sub.transAxes, fontsize=20)
    sub.set_ylabel(r'log SFR  $[M_\odot / yr]$', fontsize=20) 
    sub.set_xlabel(r'log$\; M_* \;\;[M_\odot]$', fontsize=20) 

    sub = fig.add_subplot(122)
    for i_m in range(len(fit_logm)):  
        in_mbin = np.where((logm > fit_logm[i_m]-0.5*fSFMS._dlogm) & 
                (logm < fit_logm[i_m]+0.5*fSFMS._dlogm))
        mbin_lbl = str(round(fit_logm[i_m]-0.5*fSFMS._dlogm,1))+'$<$ log$\,M_* <$'+str(round(fit_logm[i_m]+0.5*fSFMS._dlogm,1)) 
        _ = sub.hist(dMS[in_mbin], bins=32, 
                range=[-3., 1.], normed=True, histtype='step', color='C'+str(i_m), 
                linewidth=2, label=mbin_lbl)
    # x-axis
    sub.set_xlim([-2.75, 1.])
    sub.set_xticks([-2, -1, 0, 1])
    sub.set_ylim([0., 1.4])
    sub.set_yticks([0., 0.5, 1.])
    sub.legend(loc='upper right') 

    sub.set_ylabel(r'P($d_\mathrm{MS}$  [dex])', fontsize=20) 
    sub.set_xlabel(r'$d_\mathrm{MS}$ [dex]', fontsize=20) 

    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    fig_name = ''.join([UT.fig_dir(), 'PdMS_fit.assess.', catalog, '.', fit_method, '.png'])
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close() 
    return None


def fQ_dMS(fit_method): 
    ''' Compare SFR vs M* relation of various simlations and data
    '''
    Cata = Cat()
    # Read in various data sets
    catalog_list = ['tinkergroup', 
            'nsa_dickey', 
            'illustris_1gyr', #'illustris_10myr', 
            'eagle_1gyr', #'eagle_10myr', 
            'mufasa_1gyr']#, 'mufasa_10myr']

    catalog_labels = [] # Labels 
    logSFRs, logMstars, weights = [], [], [] 
    for cat in catalog_list: 
        catalog_labels.append(Cata.CatalogLabel(cat))
        logMstar, logSFR, weight = Cata.Read(cat)
        logSFRs.append(logSFR)
        logMstars.append(logMstar) 
        weights.append(weight)
    
    fig = plt.figure(1, figsize=(6,6))
    sub = fig.add_subplot(111) 

    for i_data in range(len(logSFRs)):
        fSFMS = fstarforms()
        fit_logm, fit_logsfr = fSFMS.fit(logMstars[i_data], logSFRs[i_data], method=fit_method) 
        F_sfms = fSFMS.powerlaw()
        dMS = fSFMS.d_MS(logMstars[i_data], logSFRs[i_data]) 
        
        qed = np.where(dMS < -1)# 0.9) # 3 sig 
        sfq = np.zeros(len(logMstars[i_data])) 
        sfq[qed] = 1. 
        
        mbin = np.arange(logMstars[i_data].min()-0.25, logMstars[i_data].max()+0.25, 0.5)
        mstars, fq_dms = [], [] 
        for im in range(len(mbin)-1): 
            inmbin = np.where((logMstars[i_data] >= mbin[im]) & (logMstars[i_data] < mbin[im+1]))
            mstars.append(0.5*(mbin[im]+mbin[im+1]))
            fq_dms.append(np.sum(sfq[inmbin])/np.float(len(inmbin[0])))
        sub.plot(mstars, fq_dms, c='C'+str(i_data), lw=2, label=catalog_labels[i_data])

    #bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    sub.set_xlim([7., 12.]) 
    sub.set_xlabel(r'log $M_* \;\;[M_\odot]$', fontsize=30) 
    sub.set_ylim([0., 1.]) 
    sub.set_ylabel(r'$f_\mathrm{Q}^{d_\mathrm{MS}}$', fontsize=30) 
    sub.legend(loc='upper left', prop={'size': 15}) 
    
    fig_name = ''.join([UT.fig_dir(), 'fQ_dMS_catalogs.png'])
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close()
    return None 


def sfmsfit_check(catalog, fit_method):
    ''' Tests on whether the fstarforms SFMS fitting works
    ''' 
    cat = Cat()    
    if catalog not in cat.catalog_list: 
        raise ValueError("catalog not in list") 
    
    # read in the catalog
    logm, logsfr, w = cat.Read(catalog)  
    
    # fit the SFMS using whatever method 
    fSFMS = fstarforms()
    fit_logm, fit_logsfr = fSFMS.fit(logm, logsfr, method=fit_method) 
    F_sfms = fSFMS.powerlaw()

    fig = plt.figure(1)
    sub = fig.add_subplot(111)
    DFM.hist2d(logm, logsfr, color='#1F77B4', 
            levels=[0.68, 0.95], range=[[7., 12.], [-4., 2.]], 
            plot_datapoints=True, fill_contours=False, plot_density=True, 
            ax=sub) 
    sub.scatter(fit_logm, fit_logsfr, c='orange', marker='x', lw=3, s=40)
    sub.plot(np.linspace(6., 12., 20), F_sfms(np.linspace(6., 12., 20)), c='k', lw=2, ls='--') 
    sub.set_xlim([7., 12.]) 
    sub.set_ylim([-3., 2.]) 
    sub.set_xlabel(r'log $M_* \;\;[M_\odot]$', fontsize=25) 
    sub.set_ylabel(r'log $\mathrm{SFR}\;\;[M_\odot/yr]$', fontsize=25) 
    fig_name = ''.join([UT.fig_dir(), 'SFMSfit.', fit_method, '.', catalog, '.png'])
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close()
    return None 


def SFMSfrac(catalog):   
    # read in catalogs
    cat = Cat()
    logm, logsfr, w = cat.Read(catalog)  
    
    fig = plt.figure(1)
    sub = fig.add_subplot(111)

    for fit_method in ['gaussfit', 'gaussmix']: 
        # fit the SFMS using whatever method 
        fSFMS = fstarforms()
        _ = fSFMS.fit(logm, logsfr, method=fit_method) 
        fit_logm, frac_sfms = fSFMS.frac_SFMS() 

        sub.plot(fit_logm, frac_sfms) 
    sub.set_xlim([9., 12.]) 
    sub.set_xlabel('log $M_*\;\;[M_\odot]$', fontsize=20)
    sub.set_ylim([0., 1.]) 
    sub.set_ylabel('$f_\mathrm{SFMS}$', fontsize=20)
    plt.show()
    plt.close() 
    return None


def SFRMstar_GMM(catalog, n_comp_max=30):
    ''' Test the 2D GMM fit to the SFR-Mstar plane 
    '''
    cat = Cat()
    _logm, _logsfr, w, censat = cat.Read(catalog)  
    iscen = (censat == 1) 
    logm = _logm[iscen] 
    logsfr = _logsfr[iscen]

    gbest = sfr_mstar_gmm(logm, logsfr, n_comp_max=n_comp_max)
    
    fig = plt.figure()
    sub = fig.add_subplot(111)
    sub.scatter(logm, logsfr, color='k', s=4) 

    for i, (mean, covar) in enumerate(zip(gbest.means_, gbest.covariances_)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color='C'+str(i % 10), linewidth=0)
        ell.set_clip_box(sub.bbox)
        ell.set_alpha(0.5)
        sub.add_artist(ell)

    sub.set_xlim([7., 12.]) 
    sub.set_xlabel(r'log$\; M_* \;\;[M_\odot]$', fontsize=20) 
    sub.set_ylim([-4., 2.]) 
    sub.set_ylabel(r'log SFR  $[M_\odot / yr]$', fontsize=20) 
    fig_name = ''.join([UT.fig_dir(), 'SFRMstar.2D_GMM.', catalog, '.compmax', str(n_comp_max), '.png'])
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close() 
    return None 


def xd_GMM(name, mbin=[10., 10.2]): 
    '''
    '''
    Cata = Cat()
    _logM, _logSFR, w, censat = Cata.Read(name)
    iscen = (censat == 1)
    logM = _logM[iscen]
    logSFR = _logSFR[iscen]
    
    # SFR uncertainty from star particle mass 
    logSFR_err = 0.434*(2.e-2)/(10.**logSFR)

    mlim = (logM > 10.0) & (logM < 10.2) & (np.isfinite(logSFR_err))
    
    ncomps = range(1, 6)
    fig = plt.figure(figsize=(4*len(ncomps), 4))
    xx = np.linspace(-14., -9., 100)
    bics = [] 
    for i in ncomps: 
        xdg = xdGMM(i, n_iter=500) 
        xdg.Fit(logSFR[mlim]-logM[mlim], logSFR_err[mlim]) 
        X, Xerr = xdg._X_check(logSFR[mlim]-logM[mlim], logSFR_err[mlim]) 
        bics.append(xdg.bic(X, Xerr))

        sub = fig.add_subplot(1,len(ncomps),i)
        _ = sub.hist(logSFR[mlim] - logM[mlim], normed=True, 
                histtype='step', range=[-13., -9], bins=32)
        for icomp in range(len(xdg.mu)):
            sub.plot(xx, xdg.alpha[icomp]*MNorm.pdf(xx, xdg.mu[icomp], xdg.V[icomp]))
            if icomp == 0:
                gtot = xdg.alpha[icomp]*MNorm.pdf(xx, xdg.mu[icomp], xdg.V[icomp])
            else:
                gtot += xdg.alpha[icomp]*MNorm.pdf(xx, xdg.mu[icomp], xdg.V[icomp])
        sub.plot(xx, gtot, c='k', ls='--')
        sub.set_xlim([-13., -9])
        if i < np.max(ncomps):  
            sub.set_xticklabels([])
    print ncomps
    print bics 
    str_mbin = '_'.join([str(mm) for mm in mbin])
    fig_name = ''.join([UT.fig_dir(), 'XDGMM.', name, '.', str_mbin, '.png'])
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close() 
    return None 


if __name__=='__main__': 
    #SFR_Mstar_Catalogs('gaussmix', contour='dfm')
    xd_GMM('illustris_100myr') 
    #for c in ['illustris', 'eagle', 'mufasa']:
    #    for tscale in ['inst', '10myr', '100myr', '1gyr']: 
    #        try: 
    #            SFRMstar_GMM(c+'_'+tscale)
    #        except (ValueError, NotImplementedError): 
    #            continue 
    #assess_SFMS_fit_dlogM('eagle_100myr', 'gaussmix', mbin=[9.5, 9.7])
    #assess_SFMS_dMS('nsa_dickey', 'gaussmix')
    #fQ_dMS('gaussmix')

    #assess_SFMS_fit('tinkergroup', 'gaussmix')
    #assess_SFMS_fit('illustris_10myr', 'gaussmix')
    #SFMSfrac('tinkergroup')
