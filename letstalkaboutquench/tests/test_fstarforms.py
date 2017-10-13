'''
'''
import numpy as np 
from scipy.stats import multivariate_normal as MNorm
import corner as DFM 

import env
from catalogs import Catalog as Cat
from fstarforms import fstarforms
import util as UT

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


if __name__=='__main__': 
    assess_SFMS_fit('tinkergroup', 'gaussmix')
    #SFMSfrac('tinkergroup')
