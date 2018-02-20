'''

Plots for Paper I of quenched galaxies series 


Author(s): ChangHoon Hahn 


'''
import numpy as np 
import corner as DFM 
from scipy.stats import multivariate_normal as MNorm

import matplotlib.pyplot as plt 
from matplotlib import lines as mlines

# -- Local --
import util as UT
import catalogs as Cats
import galprop as Gprop
from fstarforms import fstarforms

from ChangTools.plotting import prettycolors
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


def Catalogs_SFR_Mstar(): 
    ''' Compare SFR vs M* relation of central galaxies from various simlations and 
    observations 
    '''
    tscales = ['inst', '100myr']
    
    Cat = Cats.Catalog()
    # Read in various data sets
    sims_list = ['illustris', 'eagle', 'mufasa', 'scsam'] 
    obvs_list = ['tinkergroup', 'nsa_dickey']#, 'nsa_combined', 'nsa_combined_uv'] 

    fig = plt.figure(1, figsize=(20,7.5))
    bkgd = fig.add_subplot(111, frameon=False)
    plot_range = [[7., 12.], [-4., 2.]]

    # plot SFR-M* for the observations 
    sub0 = fig.add_subplot(2,5,5)
    for i_c, cat in enumerate(obvs_list): 
        logMstar, logSFR, weight, censat = Cat.Read(cat)
    
        iscen = (censat == 1)
        DFM.hist2d(logMstar[iscen], logSFR[iscen], color='C'+str(i_c), 
                levels=[0.68, 0.95], range=plot_range, 
                plot_datapoints=True, fill_contours=False, plot_density=True, 
                ax=sub0) 

    sub0.text(0.9, 0.1, 'SDSS Centrals', ha='right', va='center', 
                transform=sub0.transAxes, fontsize=20)

    for i_t, tscale in enumerate(tscales): 
        for i_c, cc in enumerate(sims_list): 
            cat = '_'.join([cc, tscale]) 
            sub = fig.add_subplot(2,5,1+i_c+i_t*5) 

            try: 
                lbl = Cat.CatalogLabel(cat)
                logMstar, logSFR, weight, censat = Cat.Read(cat)
            except (ValueError, NotImplementedError): 
                sub.set_xlim(plot_range[0])
                sub.set_ylim(plot_range[1])
                sub.text(0.5, 0.5, r'$\mathtt{'+cc.upper()+'}$', 
                        ha='center', va='center', transform=sub.transAxes, 
                        rotation=45, color='red', fontsize=40)
                continue 

            if i_c == 0: 
                sub.text(0.1, 0.9, 'SFR ['+(lbl.split('[')[-1]).split(']')[0]+']', 
                        ha='left', va='center', transform=sub.transAxes, fontsize=20)
            if i_t == 1: 
                sub.text(0.9, 0.1, lbl.split('[')[0], ha='right', va='center', 
                        transform=sub.transAxes, fontsize=20)

            iscen = (censat == 1)

            DFM.hist2d(logMstar[iscen], logSFR[iscen], color='C'+str(i_c+2), 
                    levels=[0.68, 0.95], range=plot_range, 
                    plot_datapoints=True, fill_contours=False, plot_density=True, ax=sub) 

    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', labelpad=15, fontsize=25) 
    bkgd.set_ylabel(r'log ( SFR $[M_\odot \, yr^{-1}]$ )', labelpad=15, fontsize=25) 
    
    fig.subplots_adjust(wspace=0.15, hspace=0.15)
    
    fig_name = ''.join([UT.fig_dir(), 'Catalogs_SFR_Mstar_SFR.pdf'])
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close()
    return None 


def Catalog_SFMS_fit(tscale): 
    ''' Compare the GMM fits to the SFMS 
    '''
    if tscale not in ['inst', '10myr', '100myr', '1gyr']: 
        raise ValueError
    
    Cat = Cats.Catalog()
    # Read in various data sets
    sims_list = ['illustris', 'eagle', 'mufasa', 'scsam'] 
    obvs_list = ['tinkergroup', 'nsa_dickey'] 

    fig = plt.figure(1, figsize=(16,8))
    bkgd = fig.add_subplot(111, frameon=False)
    plot_range = [[7., 12.], [-4., 2.]]

    # plot SFR-M* for the observations 
    sub0 = fig.add_subplot(231)
    for i_c, cat in enumerate(obvs_list): 
        logMstar, logSFR, weight, censat = Cat.Read(cat)
        iscen = (censat == 1)

        # fit the SFMS  
        fSFMS = fstarforms()
        fit_logm, fit_logsfr = fSFMS.fit(logMstar[iscen], logSFR[iscen], 
                method='gaussmix', forTest=True) 
        F_sfms = fSFMS.powerlaw()

        DFM.hist2d(logMstar[iscen], logSFR[iscen], color='C'+str(i_c), 
                levels=[0.68, 0.95], range=plot_range, 
                plot_datapoints=True, fill_contours=False, plot_density=True, alpha=0.5, 
                ax=sub0) 
        sub0.scatter(fit_logm, fit_logsfr, c='k', marker='x', lw=3, s=40)

    sub0.text(0.9, 0.1, 'SDSS Centrals', ha='right', va='center', 
                transform=sub0.transAxes, fontsize=20)

    _i = 0 
    fit_logms = [None for i in sims_list]
    fit_logsfrs = [None for i in sims_list]
    for i_c, cc in enumerate(sims_list): 
        cat = '_'.join([cc, tscale]) 
        sub = fig.add_subplot(2,3,2+i_c) 

        try: 
            lbl = Cat.CatalogLabel(cat)
            logMstar, logSFR, weight, censat = Cat.Read(cat)
        except (ValueError, NotImplementedError): 
            sub.set_xlim(plot_range[0])
            sub.set_ylim(plot_range[1])
            sub.text(0.5, 0.5, r'$\mathtt{'+cc.upper()+'}$', 
                    ha='center', va='center', transform=sub.transAxes, 
                    rotation=45, color='red', fontsize=40)
            continue 

        if _i == 0: 
            sub0.text(0.1, 0.9, 'SFR ['+(lbl.split('[')[-1]).split(']')[0]+']', 
                    ha='left', va='center', transform=sub0.transAxes, fontsize=20)
            _i += 1 

        iscen = (censat == 1)

        # fit the SFMS  
        fSFMS = fstarforms()
        fit_logm, fit_logsfr = fSFMS.fit(logMstar[iscen], logSFR[iscen], 
                method='gaussmix', forTest=True) 
        fit_logms[i_c] = fit_logm 
        fit_logsfrs[i_c] = fit_logsfr
        F_sfms = fSFMS.powerlaw()

        DFM.hist2d(logMstar[iscen], logSFR[iscen], color='C'+str(i_c+2), 
                levels=[0.68, 0.95], range=plot_range, 
                plot_datapoints=True, fill_contours=False, plot_density=True, ax=sub) 
        sub.text(0.9, 0.1, lbl.split('[')[0], ha='right', va='center', 
                transform=sub.transAxes, fontsize=20)
        sub.scatter(fit_logm, fit_logsfr, c='k', marker='x', lw=3, s=40)
    
    sub = fig.add_subplot(2,3,len(sims_list)+2) 
    for i in range(len(fit_logms)):   
        if fit_logms[i] is not None: 
            sub.scatter(fit_logms[i], fit_logsfrs[i], c='C'+str(i+2), marker='x', lw=3, s=40) 
    sub.set_xlim(plot_range[0]) 
    sub.set_ylim(plot_range[1]) 

    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', labelpad=15, fontsize=25) 
    bkgd.set_ylabel(r'log ( SFR $[M_\odot \, yr^{-1}]$ )', labelpad=15, fontsize=25) 
    
    fig.subplots_adjust(wspace=0.15, hspace=0.15)
    
    fig_name = ''.join([UT.fig_dir(), 'Catalogs_SFMSfit_SFR', tscale, '.pdf'])
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close()
    return None 


def SFMSfit_example(): 
    ''' Pedagogical example of how the GMM SFMS fitting works. Some P(SSFR) distribution 
    with the GMM components overplotted on them.
    '''
    sim = 'illustris_inst' 
    mranges = [[10.4, 10.6], [11.0, 11.2]]
    cols = ['C0', 'C4']
    panels = ['a)', 'b)']

    Cat = Cats.Catalog()
    logMstar, logSFR, weight, censat = Cat.Read(sim)
    iscen = (censat == 1)
        
    # fit the SFMS  
    fSFMS = fstarforms() 
    _fit_logm, _fit_logsfr = fSFMS.fit(logMstar[iscen], logSFR[iscen], fit_range=[9.0, 12.], method='gaussmix') 

    fig = plt.figure(figsize=(5*(len(mranges)+1),4.5)) 
    
    sub1 = fig.add_subplot(1,3,1)
    DFM.hist2d(logMstar[iscen], logSFR[iscen], color='#ee6a50',
            levels=[0.68, 0.95], range=[[9., 12.], [-3.5, 1.5]], 
            plot_datapoints=True, fill_contours=False, plot_density=True, ax=sub1) 
    sub1.scatter(_fit_logm, _fit_logsfr, c='k', marker='x', lw=3, s=40)

    for i_m, mrange in enumerate(mranges): 
        sub1.fill_between(mrange, [2.,2.], [-5.,-5], color=cols[i_m], linewidth=0, alpha=0.25)
        sub1.text(mrange[0], -2.75, panels[i_m], ha='left', va='center', fontsize=20)
    sub1.set_xticks([9., 10., 11., 12.])
    sub1.set_xlabel('log$(\; M_*\; [M_\odot]\;)$', fontsize=20)
    sub1.set_ylim([-3.25, 1.75]) 
    sub1.set_yticks([-3., -2., -1., 0., 1.])
    sub1.set_ylabel('log$(\; \mathrm{SFR}\; [M_\odot/\mathrm{yr}]\;)$', fontsize=20)
    sub1.text(0.05, 0.95, 'Illustris',
            ha='left', va='top', transform=sub1.transAxes, fontsize=25)
    
    for i_m, mrange in enumerate(mranges): 
        fit_logm, _ = fSFMS.fit(logMstar[iscen], logSFR[iscen], method='gaussmix', fit_range=mrange, forTest=True) 
        i_fit = np.abs(fit_logm - np.mean(mrange)).argmin()
        #if i_m == 0: 
        #    subs.text(0.95, 0.1, 'Illustris',
        #            ha='left', va='center', transform=sub1.transAxes, fontsize=20)

        # P(log SSFR) 
        sub2 = fig.add_subplot(1,3,i_m+2)
        inmbin = np.where((logMstar[iscen] > mrange[0]) & (logMstar[iscen] < mrange[1]))

        _ = sub2.hist((logSFR[iscen] - logMstar[iscen])[inmbin], bins=40, 
                range=[-14., -8.], normed=True, histtype='step', color='k', linewidth=1.75)

        # overplot GMM component for SFMS
        gmm_weights = fSFMS._gmix_weights[i_fit]
        gmm_means = fSFMS._gmix_means[i_fit]
        gmm_vars = fSFMS._gmix_covariances[i_fit]
    
        # colors 
        if len(gmm_means) == 3: cs = ['C1', 'C2', 'C0'] 
        elif len(gmm_means) == 2: cs = ['C1', 'C0'] 
        elif len(gmm_means) == 1: cs = ['C0'] 

        for ii, icomp in enumerate(np.argsort(gmm_means)): 
            xx = np.linspace(-14., -9, 100)
            if ii == len(gmm_means)-1: 
                sub2.plot(xx, gmm_weights[icomp]*MNorm.pdf(xx, gmm_means[icomp], gmm_vars[icomp]), 
                        c=cs[ii], linewidth=2)
            else: 
                sub2.plot(xx, gmm_weights[icomp]*MNorm.pdf(xx, gmm_means[icomp], gmm_vars[icomp]), 
                        c=cs[ii])

        for i_comp in range(len(gmm_vars)): 
            if i_comp == 0: 
                gmm_tot = gmm_weights[i_comp]*MNorm.pdf(xx, gmm_means[i_comp], gmm_vars[i_comp])
            else: 
                gmm_tot += gmm_weights[i_comp]*MNorm.pdf(xx, gmm_means[i_comp], gmm_vars[i_comp])
        sub2.plot(xx, gmm_tot, color='k', linestyle=':', linewidth=2)
        
        if i_m == 0: 
            sub2.set_ylabel('$p\,(\;\mathrm{log}\; \mathrm{SSFR}\; [\mathrm{yr}^{-1}]\;)$', fontsize=20)
        sub2.set_xlabel('log$(\; \mathrm{SSFR}\; [\mathrm{yr}^{-1}]\;)$', fontsize=20) 
        sub2.set_xlim([-13.25, -9.]) 
        sub2.set_xticks([-9., -10., -11., -12., -13.][::-1])
        sub2.set_ylim([0.,2.1]) 
        sub2.set_yticks([0., 0.5, 1., 1.5, 2.])
        # mass bin 
        sub2.text(0.05, 0.95, panels[i_m], ha='left', va='top', transform=sub2.transAxes, fontsize=25)
        sub2.text(0.95, 0.9, '$'+str(mrange[0])+'< \mathrm{log}\, M_* <'+str(mrange[1])+'$',
                ha='right', va='center', transform=sub2.transAxes, fontsize=18)

    fig.subplots_adjust(wspace=.3)
    fig_name = ''.join([UT.fig_dir(), 'SFMSfit_demo.pdf'])
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close()
    return None


def _SFMSfit_assess(name, method='gaussmix'):
    ''' Assess the quality of the SFMS fits by comparing to the actual 
    P(sSFR) in mass bins. 
    '''
    cat = Cats.Catalog() # read in catalog
    _logm, _logsfr, _, censat = cat.Read(name)  

    iscen = (censat == 1) # centrals only 
    logm = _logm[iscen]
    logsfr = _logsfr[iscen]
    
    # fit the SFMS  
    fSFMS = fstarforms()
    fit_logm, fit_logsfr = fSFMS.fit(logm, logsfr, method=method, forTest=True) 
    F_sfms = fSFMS.powerlaw()
    
    # common sense cuts imposed in fSFMS
    logm = logm[fSFMS._sensecut]
    logsfr = logsfr[fSFMS._sensecut]

    n_col = int(np.ceil(float(len(fSFMS._tests['gbests'])+1)/3))
    fig = plt.figure(1, figsize=(5*n_col, 12))
    xx = np.linspace(-14, -8, 50) 

    # plot log SFR - log M* relation 
    sub = fig.add_subplot(3,n_col,1)
    DFM.hist2d(logm, logsfr, color='black', 
            ax=sub, levels=[0.68, 0.95], range=[[7., 12.], [-4., 2.]], fill_contours=False) 
    sub.scatter(fit_logm, fit_logsfr, c='b', marker='x', lw=3, s=40)
    sub.text(0.9, 0.1, cat.CatalogLabel(name),
            ha='right', va='center', transform=sub.transAxes, fontsize=20)
    sub.set_ylabel(r'log ( SFR  $[M_\odot / yr]$ )', fontsize=20) 
    sub.set_xlabel(r'log$\; M_* \;\;[M_\odot]$', fontsize=20) 

    for i_m in range(len(fSFMS._tests['gbests'])):  
        sub = fig.add_subplot(3, n_col, i_m+2)
    
        # within mass bin 
        mbin_mid = fSFMS._tests['mbin_mid'][i_m]
        in_mbin = np.where(
                (logm > mbin_mid-0.5*fSFMS._dlogm) & 
                (logm < mbin_mid+0.5*fSFMS._dlogm))
        # whether or not the mbin was skiped
        in_mbin_fit = (fit_logm > mbin_mid-0.5*fSFMS._dlogm) & (fit_logm < mbin_mid+0.5*fSFMS._dlogm)
        if np.sum(in_mbin_fit) > 0: 
            tcolor = 'k'
            sub.vlines(fit_logsfr[in_mbin_fit] - fit_logm[in_mbin_fit], 0., 1.4, 
                    color='b', linestyle='--') 
        else: tcolor = 'r'
        sub.text(0.1, 0.9, ''.join([str(round(mbin_mid-0.5*fSFMS._dlogm,1)), 
                    '$<$ log$\,M_* <$', str(round(mbin_mid+0.5*fSFMS._dlogm,1))]), 
                ha='left', va='center', transform=sub.transAxes, color=tcolor, fontsize=20)
        
        # P(SSFR) histogram 
        _ = sub.hist(logsfr[in_mbin] - logm[in_mbin], bins=32, 
                range=[-14., -8.], normed=True, histtype='step', color='k', linewidth=1.75)

        # plot the fits 
        if 'gaussmix' in method:
            gmm_weights = fSFMS._tests['gbests'][i_m].weights_
            gmm_means = fSFMS._tests['gbests'][i_m].means_.flatten() 
            gmm_vars = fSFMS._tests['gbests'][i_m].covariances_.flatten() 

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
        # y-axis
        sub.set_ylim([0., 1.4])
        sub.set_yticks([0., 0.5, 1.])

        sub.set_ylabel(r'P(log SSFR  $[yr^{-1}])$', fontsize=20) 
        sub.set_xlabel(r'log SSFR  $[yr^{-1}]$', fontsize=20) 

    fig.subplots_adjust(hspace=0.3, wspace=0.4)
    fig_name = ''.join([UT.fig_dir(), 'Pssfr_fit.assess.', name, '.', method, '.png'])
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close() 
    return None


def _SFR_tscales(name):  
    ''' compare the SFRs of different timescales 
    '''
    Cat = Cats.Catalog() # read in the two different SFR timescales 
    fig = plt.figure(figsize=(16,16))
    i_p = 1
    for i1, tscale1 in enumerate(['inst', '10myr', '100myr', '1gyr']): 
        for i0, tscale0 in enumerate(['inst', '10myr', '100myr', '1gyr'][::-1]): 
            sub = fig.add_subplot(4,4,i_p)
            i_p += 1
    #sub.set_ylabel(r'log ( SFR '+lbl0.split('_')[-1]+'$[M_\odot \, yr^{-1}]$ )', fontsize=25) 
            sub.set_xlim([-4., 2.]) 
            sub.set_ylim([-4., 2.]) 
            if i0 == 0: 
                sub.set_ylabel(r'log ( SFR ['+tscale1+'] )', fontsize=20) 
            if i1 == 3: 
                sub.set_xlabel(r'log ( SFR ['+tscale0+'] )', fontsize=20) 
            if tscale0 == tscale1: 
                continue 
            try: 
                _, logsfr0, _, _ = Cat.Read(name+'_'+tscale0)
                _, logsfr1, _, _ = Cat.Read(name+'_'+tscale1)
                DFM.hist2d(logsfr0, logsfr1, color='C0', 
                        levels=[0.68, 0.95], range=[[-4., 2.], [-4., 2.]], 
                        plot_datapoints=True, fill_contours=False, plot_density=True, 
                        ax=sub) 
                #sub.scatter(logsfr0, logsfr1, c='C0') 
            except (ValueError, NotImplementedError):
                pass 
            
            #lbl0 = Cat.CatalogLabel(name+'_'+tscale0)
            #lbl1 = Cat.CatalogLabel(name+'_'+tscale1)
            sub.plot([-4., 2.], [-4., 2.], c='k', lw=2, ls='--') 

    
    fig_name = ''.join([UT.fig_dir(), 'SFRcomparison.', name, '.png'])
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close() 


if __name__=="__main__": 
    #Catalogs_SFR_Mstar()

    #SFMSfit_example()

    #for tscale in ['inst', '10myr', '100myr', '1gyr']: 
    #    Catalog_SFMS_fit(tscale)
    #for c in ['illustris', 'eagle', 'mufasa']:
    #    _SFR_tscales(c)
    for c in ['scsam']: #'illustris', 'eagle', 'mufasa']:
        for tscale in ['inst', '10myr', '100myr', '1gyr']: 
            try: 
                _SFMSfit_assess(c+'_'+tscale, method='gaussmix')
            except (ValueError, NotImplementedError): 
                continue 
