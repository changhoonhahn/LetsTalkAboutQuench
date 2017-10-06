'''
'''
import numpy as np 

import env
from catalogs import Catalog as Cat
import util as UT
import galprop as GP 
import corner as DFM 
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt 
from ChangTools.plotting import prettyplot
from ChangTools.plotting import prettycolors


def build_dMS(catalog, fit_method): 
    ''' Calculate dMS for specified catalog and output to file 
    '''
    cat = Cat()    
    if catalog not in cat.catalog_list: 
        raise ValueError("catalog not in list") 
    
    logm, logsfr, w = cat.Read(catalog)  
    
    # zero SFR
    notzero = np.where(logsfr != -999.)
    dMS = np.repeat(999., len(logm)) 

    fit_Mrange = cat._default_Mstar_range(catalog) 
    f_SSFRcut = cat._default_fSSFRcut(catalog)

    d = GP.dMS(logsfr[notzero], logm[notzero], 
            method=fit_method, # method for fitting the SFMS 
            fit_Mrange=fit_Mrange, # stellar mass range of fit
            f_SSFRcut=f_SSFRcut
            )
    dMS[notzero] = d
    
    if 'yr' in catalog: # specific star formation timescale  
        d_file = ''.join([UT.dat_dir(), 'dMS.', fit_method, '.', 
            catalog.rsplit('_', 1)[-1], '.', 
            cat.catalog_dict[catalog].rsplit('.', 1)[0], '.dat'])
    else: 
        d_file = ''.join([UT.dat_dir(), 'dMS.', fit_method, '.', 
            cat.catalog_dict[catalog].rsplit('.', 1)[0], '.dat'])

    hdr = ''.join([catalog, ' SFMS fit using ', fit_method, 
        ' with M* range ', str(fit_Mrange[0]), ' - ', str(fit_Mrange[1]),
        ' a log SSFR cut of > -11 ']) 
    np.savetxt(d_file, dMS, fmt=['%f'], header=hdr)
    return None 


def sfmsfit_comp(catalog):
    '''
    ''' 
    cat = Cat()    
    if catalog not in cat.catalog_list: 
        raise ValueError("catalog not in list") 
    
    logm, logsfr, w = cat.Read(catalog)  
    
    # zero SFR
    notzero = np.where(logsfr != -999.)

    fit_Mrange = cat._default_Mstar_range(catalog) 
    f_SSFRcut = cat._default_fSSFRcut(catalog)

    _, gauss = GP.SFMS_bestfit(logsfr[notzero], logm[notzero], 
            method='SSFRcut_gaussfit_linearfit', # method for fitting the SFMS 
            fit_Mrange=fit_Mrange, # stellar mass range of fit
            f_SSFRcut=f_SSFRcut, 
            forTest=True)
    
    _, ziNB = GP.SFMS_bestfit(logsfr[notzero], logm[notzero], 
            method='SSFRcut_negbinomfit_linearfit', # method for fitting the SFMS 
            fit_Mrange=fit_Mrange, # stellar mass range of fit
            f_SSFRcut=f_SSFRcut, 
            forTest=True)

    prettyplot()
    fig = plt.figure(1)
    sub = fig.add_subplot(111)
    DFM.hist2d(logm[notzero], logsfr[notzero], color='#1F77B4', 
            levels=[0.68, 0.95], range=[[7., 12.], [-4., 2.]], 
            plot_datapoints=True, fill_contours=False, plot_density=True, 
            ax=sub) 
    sub.scatter(gauss[0], gauss[1]+gauss[0], c='orange', marker='x', lw=3, s=40, label='Gauss.')
    sub.scatter(ziNB[0], ziNB[1]+ziNB[0], c='blue', marker='x', lw=3, s=40, label='Neg. Binom.')
    sub.set_xlim([7., 12.]) 
    sub.set_ylim([-4., 2.]) 
    sub.set_xlabel(r'$\mathtt{log \; M_* \;\;[M_\odot]}$', fontsize=25) 
    sub.set_ylabel(r'$\mathtt{log \; SFR \;\;[M_\odot \, yr^{-1}]}$', fontsize=25) 
    sub.legend(loc='upper left', prop={'size':20})

    fig_name = ''.join([UT.fig_dir(), 'SFMSfit.comparison.', catalog, '.png'])
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close()
    return None 


def sfmsfit_check(catalog, fit_method):
    '''
    ''' 
    cat = Cat()    
    if catalog not in cat.catalog_list: 
        raise ValueError("catalog not in list") 
    
    logm, logsfr, w = cat.Read(catalog)  
    
    # zero SFR
    notzero = np.where(logsfr != -999.)

    fit_Mrange = cat._default_Mstar_range(catalog) 
    f_SSFRcut = cat._default_fSSFRcut(catalog)

    fit, arr = GP.SFMS_bestfit(logsfr[notzero], logm[notzero], 
            method=fit_method, # method for fitting the SFMS 
            fit_Mrange=fit_Mrange, # stellar mass range of fit
            f_SSFRcut=f_SSFRcut, 
            forTest=True)

    prettyplot()
    fig = plt.figure(1)
    sub = fig.add_subplot(111)
    DFM.hist2d(logm[notzero], logsfr[notzero], color='#1F77B4', 
            levels=[0.68, 0.95], range=[[7., 12.], [-4., 2.]], 
            plot_datapoints=True, fill_contours=False, plot_density=True, 
            ax=sub) 
    sub.scatter(arr[0], arr[1]+arr[0], c='orange', marker='x', lw=3, s=40)
    sub.plot(np.linspace(6., 12., 20), fit(np.linspace(6., 12., 20)), c='k', lw=2, ls='--') 
    sub.set_xlim([7., 12.]) 
    sub.set_ylim([-3., 2.]) 
    sub.set_xlabel(r'$\mathtt{log \; M_* \;\;[M_\odot]}$', fontsize=25) 
    sub.set_ylabel(r'$\mathtt{log \; SFR \;\;[M_\odot \, yr^{-1}]}$', fontsize=25) 
    fig_name = ''.join([UT.fig_dir(), 'SFMSfit.', fit_method, '.', catalog, '.png'])
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close()
    return None 



def dMS_Mstar(catalog, fit_method):
    '''
    ''' 
    cat = Cat()    
    if catalog not in cat.catalog_list: 
        raise ValueError("catalog not in list") 
    
    logm, logsfr, w = cat.Read(catalog)  
    
    # zero SFR
    notzero = np.where(logsfr != -999.)

    fit_Mrange = cat._default_Mstar_range(catalog) 
    f_SSFRcut = cat._default_fSSFRcut(catalog)
    
    dMS = GP.dMS(logsfr[notzero], logm[notzero], 
            method=fit_method, # method for fitting the SFMS 
            fit_Mrange=fit_Mrange, # stellar mass range of fit
            f_SSFRcut=f_SSFRcut)

    prettyplot()
    fig = plt.figure(1)
    sub = fig.add_subplot(111)
    DFM.hist2d(logm[notzero], dMS, color='#1F77B4', 
            levels=[0.68, 0.95], range=[[7., 12.], [-5., 1.]], 
            plot_datapoints=True, fill_contours=False, plot_density=True, 
            ax=sub) 
    sub.plot([7., 12.], [-0.9, -0.9], c='r', ls='--', lw=2)
    sub.set_xlim([7., 12.]) 
    sub.set_ylim([-5., 1.]) 
    sub.set_xlabel(r'$\mathtt{log \; M_* \;\;[M_\odot]}$', fontsize=25) 
    sub.set_ylabel(r'$\mathtt{d_{MS} \;\;[dex]}$', fontsize=25) 

    fig_name = ''.join([UT.fig_dir(), 'dMS_Mstar.', catalog, '.png'])
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close()
    return None 



def P_dMS(catalog, fit_method):
    ''' plot the distribution of d_MS for catalog based on fit_method 
    ''' 
    cat = Cat()    
    if catalog not in cat.catalog_list: 
        raise ValueError("catalog not in list") 
    
    logm, logsfr, w = cat.Read(catalog)  
    
    # zero SFR
    notzero = np.where(logsfr != -999.)

    fit_Mrange = cat._default_Mstar_range(catalog) 
    f_SSFRcut = cat._default_fSSFRcut(catalog)
    
    dMS = GP.dMS(logsfr[notzero], logm[notzero], 
            method=fit_method, # method for fitting the SFMS 
            fit_Mrange=fit_Mrange, # stellar mass range of fit
            f_SSFRcut=f_SSFRcut
            )
    
    dlogM = 0.5 # dex
    logM_bins = np.arange(fit_Mrange[0], fit_Mrange[1]+dlogM, dlogM) 
    n_col = int(np.float(len(logM_bins))/2)
    
    prettyplot() 
    fig = plt.figure(1, figsize=(8*n_col, 12))
    bkgd = fig.add_subplot(111, frameon=False)
    for i_m in range(len(logM_bins)-1):  
        sub = fig.add_subplot(2, n_col, i_m + 1)
        inmbin = np.where((logm[notzero] > logM_bins[i_m]) & (logm[notzero] > logM_bins[i_m+1]))

        sub.hist(dMS[inmbin], bins=32, range=[-5., 2.], normed=True)

        sub.text(0.4, 0.9, 
                str(logM_bins[i_m])+' < $\mathtt{log\,M_*}$ < '+str(logM_bins[i_m+1]),  
                ha='center', va='center', transform=sub.transAxes, fontsize=25)
        sub.set_xlim([-5., 1.]) 
    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel(r'$\mathtt{d_{MS}\;\;[dex]}$', labelpad=20, fontsize=30) 
    bkgd.set_ylabel(r'$\mathtt{P(d_{MS})}$', labelpad=20, fontsize=30) 
    fig_name = ''.join([UT.fig_dir(), 'P_dMS.', fit_method, '.', catalog, '.png'])
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close()
    return None 


def fQ_dMS(fit_method):
    ''' plot the distribution of d_MS for catalog based on fit_method 
    ''' 
    catalogs = ['illustris_1gyr', 'illustris_10myr', 'eagle_1gyr', 'eagle_10myr', 
            'mufasa_1gyr', 'mufasa_10myr', 'tinkergroup', 'nsa_dickey']
    qfs, qfs_ssfrcut = [], [] 
    for catalog in catalogs:
        cat = Cat()    
        if catalog not in cat.catalog_list: 
            raise ValueError("catalog not in list") 
        
        logm, logsfr, w = cat.Read(catalog)  
        sfq = np.zeros(len(logm)) # 1 if Q 0 if SF
        # zero SFR
        notzero = np.where(logsfr != -999.)

        fit_Mrange = cat._default_Mstar_range(catalog) 
        f_SSFRcut = cat._default_fSSFRcut(catalog)

        dMS = GP.dMS(logsfr[notzero], logm[notzero], 
                method=fit_method, # method for fitting the SFMS 
                fit_Mrange=fit_Mrange, # stellar mass range of fit
                f_SSFRcut=f_SSFRcut
                )
        qed = np.where(dMS < -0.9) # 3 sig 
        sfq[notzero[0][qed]] = 1 
        sfq[np.where(logsfr == -999.)] = 1 
        
        mbin = np.arange(logm[notzero].min(), logm[notzero].max()+0.5, 0.5)
        ms = [] 
        fq = [] 
        fq_ssfrcut = [] 
        for im in range(len(mbin)-1): 
            inmbin = np.where((logm[notzero] >= mbin[im]) & (logm[notzero] < mbin[im+1]))
            if len(inmbin[0]) > 100:
                ms.append(0.5*(mbin[im] + mbin[im+1]))
                fq.append(np.sum(sfq[inmbin])/float(len(inmbin[0])))
                fq_ssfrcut.append(np.sum(logsfr[notzero[0][inmbin]] < logm[notzero[0][inmbin]] - 11.)/ float(len(inmbin[0])))

        qf = [np.array(ms), np.array(fq)]
        qf_ssfrcut = [np.array(ms), np.array(fq_ssfrcut)]
        qfs.append(qf) 
        qfs_ssfrcut.append(qf_ssfrcut) 
    
    prettyplot()
    fig = plt.figure(1, figsize=(17, 8))
    bkgd = fig.add_subplot(111, frameon=False)
    sub1 = fig.add_subplot(121)
    sub2 = fig.add_subplot(122)
    pretty_colors = prettycolors() 
    for iqf, qf in enumerate(qfs): 
        sub1.plot(qf[0], qf[1], lw=2, c=pretty_colors[iqf])
        sub2.plot(qfs_ssfrcut[iqf][0], qfs_ssfrcut[iqf][1], lw=2, c=pretty_colors[iqf], label=' '.join(catalogs[iqf].split('_')))
    sub1.set_xlim([6., 12.]) 
    sub1.set_ylim([0., 1.])
    sub1.set_ylabel('$\mathtt{f_{Q}^{d_{MS}}(M_*)}$', fontsize=25)
    sub2.set_xlim([6., 12.]) 
    sub2.set_ylim([0., 1.])
    sub2.set_ylabel('$\mathtt{f_{Q}^{SSFR}(M_*)}$', fontsize=25)
    sub2.legend(loc='upper left') 
    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel(r'$\mathtt{log \; M_* \;\;[M_\odot]}$', labelpad=20, fontsize=25) 
    fig_name = ''.join([UT.fig_dir(), 'fQ_dMS.', fit_method, '.png'])
    fig.savefig(fig_name, bbox_inches='tight') 
    plt.close()
    return None 


def assess_SFMS_fit(catalog, fit_method, Mrange=None):
    ''' Assess the quality of the Gaussian / Negative Binomials 
    fits by comparing the best-fit distribution to the P(SFR) at 
    each of the mass bins. 
    '''
    # hardcoded fitting stellar mass range
    dlogM = 0.5 # dex
        
    # read in catalogs
    cat = Cat()
    logMstar, logSFR, weight = cat.Read(catalog) 

    fit_Mrange = cat._default_Mstar_range(catalog)
    f_SSFRcut = cat._default_fSSFRcut(catalog)

    Mass_cut = np.where((logMstar >= fit_Mrange[0]) & (logMstar < fit_Mrange[1])) 
    assert len(Mass_cut[0]) > 0
    
    logMstar_fit = logMstar[Mass_cut]
    logSFR_fit = logSFR[Mass_cut]

    logM_bins = np.arange(fit_Mrange[0], fit_Mrange[1]+dlogM, dlogM) 
    if Mrange is not None: 
        logM_bins = Mrange
    
    n_col = int(np.ceil(np.float(len(logM_bins))/2))

    prettyplot()
    pretty_colors = prettycolors()
    fig = plt.figure(1, figsize=(17, 8))
    sub = fig.add_subplot(1,2,1)

    DFM.hist2d(logMstar, logSFR, color='black', #color='#1F77B4', 
            ax=sub, levels=[0.68, 0.95], range=[[7.5, 10.5], [-4., 2.]], 
            fill_contours=False) 
    if Mrange is not None: 
        sub.fill_between(Mrange, np.repeat(-4., len(Mrange)), 
                np.repeat(2., len(Mrange)), color='red', alpha=0.5)
                
    sfms_fit, med = GP.SFMS_bestfit(logSFR, logMstar, 
            method=fit_method, forTest=True, 
            fit_Mrange=fit_Mrange, f_SSFRcut=f_SSFRcut) 

    sub.scatter(med[0], med[1]+med[0], c='b', marker='x', lw=3, s=40)
    #mm = np.arange(6., 12.5, 0.5)
    #sub.plot(mm, f_SFRcut(mm), c='r', lw=2, ls='-.', label='SSFR cut')
    #sub.plot(mm, sfms_fit(mm)-mm, c='k', lw=2, ls='--', label='Gauss.')
    sub.text(0.4, 0.9, cat.CatalogLabel(catalog),
            ha='right', va='center', transform=sub.transAxes, fontsize=25)
    sub.set_ylabel(r'$\mathtt{log \; SFR \;\;[M_\odot \, yr^{-1}]}$', fontsize=30) 
    sub.set_xlabel(r'$\mathtt{log \; M_* \;\;[M_\odot]}$', fontsize=30) 

    for i_m in range(len(logM_bins)-1):  
        sub = fig.add_subplot(1, 2, i_m + 2)

        SSFR_cut = f_SSFRcut(logMstar_fit) 
        in_mbin = np.where(
                (logMstar_fit >= logM_bins[i_m]) & 
                (logMstar_fit < logM_bins[i_m+1]) & 
                (logSFR_fit-logMstar_fit > SSFR_cut))

        sub.text(0.4, 0.9, 
                str(logM_bins[i_m])+' < $\mathtt{log\,M_*}$ < '+str(logM_bins[i_m+1]),  
                ha='center', va='center', transform=sub.transAxes, fontsize=25)
        
        if len(in_mbin[0]) > 20: 
            yy, xx_edges = np.histogram(logSFR_fit[in_mbin] - logMstar_fit[in_mbin], 
                    bins=20, range=[-14, -8], normed=True)
            x_plot, y_plot = UT.bar_plot(xx_edges, yy)
            sub.plot(x_plot, y_plot, c='k', lw=3, ls='--')

            xx = 0.5 * (xx_edges[1:] + xx_edges[:-1])
    
            if 'gaussfit' in fit_method: # Gaussian fit 
                gaus = lambda xx, aa, x0, sig: aa * np.exp(-(xx - x0)**2/(2*sig**2))

                popt, pcov = curve_fit(gaus, xx, yy, 
                        p0=[1., np.median(logSFR_fit[in_mbin] - logMstar_fit[in_mbin]), 0.3])
                sub.plot(np.linspace(-13, -8, 50), gaus(np.linspace(-13., -8., 50), *popt), 
                        c='r', lw=2)
                #sub.vlines(popt[1], 0., 2., color='b', linewidth=2)
            elif 'negbinomfit' in fit_method: # negative binomial distribution 
                NB_fit = lambda xx, aa, mu, theta: UT.NB_pdf_logx(np.power(10., xx+aa), mu, theta)
                popt, pcov = curve_fit(NB_fit, xx, yy, p0=[12., 100, 1.5])

                sub.plot(np.linspace(-13, -8, 20), NB_fit(np.linspace(-13., -8., 20), *popt), 
                        c='r', lw=2)
                #sub.vlines(np.log10(popt[1])-popt[0], 0., 2., color='r', linewidth=2)

        # x-axis
        sub.set_xlim([-13, -8])
        sub.set_xticks([-13, -12, -11, -10, -9, -8])
        sub.set_ylim([0., 1.4])

        sub.set_ylabel(r'$\mathtt{P(log \; SSFR \;\;[yr^{-1}])}$', fontsize=30) 
        sub.set_xlabel(r'$\mathtt{log \; SSFR \;\;[yr^{-1}]}$', fontsize=30) 
    fig_name = ''.join([UT.fig_dir(), 'Pssfr_fit.assess.', catalog, '.', fit_method, '.png'])
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close() 
    return None


if __name__=="__main__": 
    dMS_Mstar('tinkergroup', 'SSFRcut_gaussfit_linearfit')
