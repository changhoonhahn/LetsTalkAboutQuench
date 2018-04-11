'''

Plots for Paper I of quenched galaxies series 


Author(s): ChangHoon Hahn 


'''
import numpy as np 
import corner as DFM 
from scipy import linalg
from scipy.stats import multivariate_normal as MNorm

import matplotlib.pyplot as plt 
from matplotlib import lines as mlines
from matplotlib.patches import Rectangle
#import matplotlib.patheffects as path_effects

# -- Local --
import util as UT
import catalogs as Cats
import galprop as Gprop
from fstarforms import fstarforms
from fstarforms import sfr_mstar_gmm

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


def Catalogs_Pssfr(mbin=[10.4, 10.6]): 
    ''' Compare the SSFR distribution for central galaxies from various simulations for
    specified stellar mass bin. This is to further emphasize the challenge of fitting 
    the different simulations
    '''
    Cat = Cats.Catalog()
    tscales = ['inst', '100myr'] # timescales 
    sims_list = ['illustris', 'eagle', 'mufasa', 'scsam'] # various data sets
    sims_label = ['Illustris', 'EAGLE', 'MUFASA', 'SC SAM'] 

    fig = plt.figure(1, figsize=(8,4))
    bkgd = fig.add_subplot(111, frameon=False)
    # plot p(SSFR) for the simulations 
    for i_t, tscale in enumerate(tscales): 
        sub = fig.add_subplot(1,2,1+i_t)

        for i_c, cat in enumerate(sims_list): 
            logMstar, logSFR, weight, censat = Cat.Read(cat+'_'+tscale, keepzeros=True)
            #psat = Cat.GroupFinder(cat+'_'+tscale)
            #iscen = ((psat < cut) & np.invert(Cat.zero_sfr))
            iscen = ((censat == 1) & np.invert(Cat.zero_sfr)) 
            inmbin = (iscen & (logMstar > mbin[0]) & (logMstar < mbin[1]))   

            ssfr_i = logSFR[inmbin] - logMstar[inmbin]
            _ = sub.hist(ssfr_i, bins=40, 
                    range=[-14., -8.], normed=True, histtype='step', color='C'+str(i_c+2), linewidth=1.75, 
                    label=sims_label[i_c])

        sub.set_xlim([-13.25, -9.]) 
        sub.set_xticks([-9., -10., -11., -12., -13.][::-1])
        sub.set_ylim([0.,2.4]) 
        sub.set_yticks([0., 0.5, 1., 1.5, 2.])
        if i_t != 0: sub.set_yticklabels([]) 
        
        lbl = Cat.CatalogLabel(cat+'_'+tscale)
        sub.text(0.075, 0.93, 'SFR ['+(lbl.split('[')[-1]).split(']')[0]+']', 
                ha='left', va='top', transform=sub.transAxes, fontsize=18)
        if i_t == 0: 
            sub.text(0.075, 0.77,'$'+str(mbin[0])+'< \mathrm{log}\, M_* <'+str(mbin[1])+'$',
                    ha='left', va='top', transform=sub.transAxes, fontsize=15)

            sub.legend(loc='lower left', bbox_to_anchor=(0.01, 0.25), frameon=False, prop={'size': 15})
    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel('log$(\; \mathrm{SSFR}\; [\mathrm{yr}^{-1}]\;)$', labelpad=5, fontsize=25) 
    bkgd.set_ylabel('$p\,(\;\mathrm{log}\; \mathrm{SSFR}\; [\mathrm{yr}^{-1}]\;)$', labelpad=5, fontsize=25)
    fig.subplots_adjust(wspace=0.15)
    fig_name = ''.join([UT.fig_dir(), 'Catalogs_pSSFR.pdf'])
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close()
    return None 


def GroupFinder(): 
    ''' plot the purity and completeness of "pure centrals" from Jeremy's group finder 
    for all of the simulations 
    
    Purity is defined as follows
        P_cen = N_{cen|cen}/(N_{cen|cen} + N_{cen|sat}) 

    Completeness is defined as follows 
        C_cen = N_{cen|cen}/(N_{cen|cen} + N_{sat|cen}) 

    '''
    names = ['illustris_100myr', 'eagle_100myr', 'mufasa_100myr', 'scsam_100myr']
    fig = plt.figure(figsize=(4*len(names),4)) 
    bkgd = fig.add_subplot(111, frameon=False) 
    Cata = Cats.Catalog()
    for i_n, name in enumerate(names): 
        logM, _, _, censat = Cata.Read(name, keepzeros=True) 
        psat = Cata.GroupFinder(name)
        
        if len(psat) != len(logM): 
            print name 
            print 'N_gal group finder = ', len(psat)
            print 'N_gal = ', len(logM)
            raise ValueError

        ispurecen = (psat < 0.01) 
        isnotpure = (psat >= 0.01) 
        iscen = (psat < 0.5) 

        mbin = np.linspace(8., 12., 17) 
        mmids, fp_pc = [], []  # purity fraction for pure central (pc) and central (c)
        fcomp_pc = []  # completeness fraction 
        for im in range(len(mbin)-1): 
            inmbin = (logM > mbin[im]) & (logM < mbin[im+1])

            if np.sum(inmbin) > 0: 
                mmids.append(0.5*(mbin[im] + mbin[im+1]))
                # fraction of pure centrals that are also identified as centrals 
                # by the simulation 
                N_cencen = float(np.sum(censat[ispurecen & inmbin] == 1))
                N_censat = float(np.sum(censat[ispurecen & inmbin] == 0))
                N_satcen = float(np.sum(censat[isnotpure & inmbin] == 1)) 
            
                # purity
                fp_pc.append(N_cencen/(N_cencen + N_censat))
                # completeness
                fcomp_pc.append(N_cencen/(N_cencen + N_satcen))
        
        sub = fig.add_subplot(1,len(names),i_n+1) 
        sub.plot(mmids, fp_pc, 
                label='Purity: '+str(round(np.mean(fp_pc),2)))
        sub.plot(mmids, fcomp_pc, 
                label='Completeness: '+str(round(np.mean(fcomp_pc),2))) 
        sub.set_xlim([8., 12.]) 
        sub.set_ylim([0., 1.]) 
        lbl = Cata.CatalogLabel(name)
        sub.set_title(lbl.split('[')[0], fontsize=20) 
        sub.legend(loc='lower left', frameon=False, prop={'size':15}) 
        
    bkgd.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', labelpad=10, fontsize=25) 
    bkgd.set_ylabel(r'Purity and Completeness', labelpad=10, fontsize=20) 
    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    fig.savefig(''.join([UT.fig_dir(), 'groupfinder.pdf']), bbox_inches='tight') 
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
    sub0 = fig.add_subplot(233)
    for i_c, cat in enumerate(obvs_list): 
        logMstar, logSFR, weight, censat = Cat.Read(cat)
        iscen = (censat == 1)

        # fit the SFMS  
        fSFMS = fstarforms()
        fit_logm, fit_logsfr = fSFMS.fit(logMstar[iscen], logSFR[iscen], 
                method='gaussmix', forTest=True) 

        DFM.hist2d(logMstar[iscen], logSFR[iscen], color='C'+str(i_c), 
                levels=[0.68, 0.95], range=plot_range, 
                plot_datapoints=True, fill_contours=False, plot_density=True, alpha=0.5, 
                ax=sub0) 
        sub0.scatter(fit_logm, fit_logsfr, c='k', marker='x', lw=3, s=40)

    sub0.text(0.9, 0.1, 'SDSS Centrals', ha='right', va='center', 
                transform=sub0.transAxes, fontsize=20)

    fit_logms = [None for i in sims_list]
    fit_logsfrs = [None for i in sims_list]
    for i_c, cc in enumerate(sims_list): 
        cat = '_'.join([cc, tscale]) 
        sub = fig.add_subplot(2,3,3*(i_c/2)+(i_c % 2)+1) 
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
                    ha='left', va='top', transform=sub.transAxes, fontsize=20)

        iscen = (censat == 1)

        # including SFR uncertainties for 100Myr 
        if (tscale == '100myr') and (cc in ['mufasa', 'illustris', 'eagle']): 
            if cc == 'mufasa': sfrerr = 0.182
            elif cc == 'illustris': sfrerr = 0.016
            elif cc == 'eagle': sfrerr = 0.018
            logsfr_err = 0.434*sfrerr/(10.**logSFR[iscen]) 

        # fit the SFMS  
        fSFMS = fstarforms()
        if (tscale == '100myr') and (cc in ['mufasa', 'illustris', 'eagle']): 
            # extreme-deconvolution
            fit_logm, fit_logsfr = fSFMS.fit(logMstar[iscen], logSFR[iscen], 
                    logsfr_err=logsfr_err, method='gaussmix_err', forTest=True) 
        else: 
            fit_logm, fit_logsfr = fSFMS.fit(logMstar[iscen], logSFR[iscen], 
                    method='gaussmix', forTest=True) 
        fit_logms[i_c] = fit_logm 
        fit_logsfrs[i_c] = fit_logsfr

        DFM.hist2d(logMstar[iscen], logSFR[iscen], color='C'+str(i_c+2), 
                levels=[0.68, 0.95], range=plot_range, 
                plot_datapoints=True, fill_contours=False, plot_density=True, ax=sub) 
        sub.text(0.9, 0.1, lbl.split('[')[0], ha='right', va='center', 
                transform=sub.transAxes, fontsize=20)
        sub.scatter(fit_logm, fit_logsfr, c='k', marker='x', lw=3, s=40)
        sub.set_xticks([8., 10., 12.]) 
        sub.set_yticks([-4., -2., 0., 2.]) 
    
    sub = fig.add_subplot(2,3,6) 
    for i in range(len(fit_logms)):   
        if fit_logms[i] is not None: 
            sub.scatter(fit_logms[i], fit_logsfrs[i], c='C'+str(i+2), marker='x', lw=3, s=40) 
    sub.set_xlim(plot_range[0]) 
    sub.set_ylim(plot_range[1]) 
    sub.set_xticks([8., 10., 12.]) 
    sub.set_yticks([-4., -2., 0., 2.]) 

    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', labelpad=15, fontsize=25) 
    bkgd.set_ylabel(r'log ( SFR $[M_\odot \, yr^{-1}]$ )', labelpad=15, fontsize=25) 
    
    fig.subplots_adjust(wspace=0.15, hspace=0.15)
    
    fig_name = ''.join([UT.fig_dir(), 'Catalogs_SFMSfit_SFR', tscale, '.pdf'])
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close()
    return None 


def Catalogs_SFMS_powerlawfit(): 
    ''' Compare the power-law fit of the GMM SFMS fits 
    '''
    Cat = Cats.Catalog()
    # tscales 
    tscales = ['inst', '100myr']
    # simulations 
    sims_list = ['illustris', 'eagle', 'mufasa', 'scsam'] 

    fig = plt.figure(1, figsize=(8,4))
    bkgd = fig.add_subplot(111, frameon=False)
    m_arr = np.linspace(8., 12., 100) 

    for i_t, tscale in enumerate(tscales): 
        sub = fig.add_subplot(1,2,1+i_t)
        for i_c, cc in enumerate(sims_list): 
            cat = '_'.join([cc, tscale]) 

            try: 
                lbl = Cat.CatalogLabel(cat)
                logMstar, logSFR, weight, censat = Cat.Read(cat, keepzeros=True)
                #psat = Cat.GroupFinder(cat+'_'+tscale)
            except (ValueError, NotImplementedError): 
                continue 

            if i_c == 0: 
                sub.text(0.1, 0.9, 'SFR ['+(lbl.split('[')[-1]).split(']')[0]+']', 
                        ha='left', va='top', transform=sub.transAxes, fontsize=20)

            # only keep centrals
            #iscen = ((psat < cut) & np.invert(Cat.zero_sfr))
            iscen = ((censat == 1) & np.invert(Cat.zero_sfr)) 

            fSFMS = fstarforms() # fit the SFMS  
            _ = fSFMS.fit(logMstar[iscen], logSFR[iscen], method='gaussmix', forTest=True) 
            # power-law fit of the SFMS fit 
            f_sfms = fSFMS.powerlaw(logMfid=10.5) 

            sub.plot(m_arr, f_sfms(m_arr), c='C'+str(i_c+2), lw=2, label=lbl.split('[')[0]) 
        sub.set_xlim([8.2, 11.8]) 
        sub.set_xticks([9., 10., 11.]) 
        sub.set_ylim([-2., 2.]) 
        sub.set_yticks([-4., -2., 0., 2.]) 
    sub.legend(loc='lower right', frameon=False, prop={'size': 15}) 
    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', labelpad=10, fontsize=25) 
    bkgd.set_ylabel(r'log ( SFR $[M_\odot \, yr^{-1}]$ )', labelpad=10, fontsize=25) 
    
    fig.subplots_adjust(wspace=0.15, hspace=0.15)
    
    fig_name = ''.join([UT.fig_dir(), 'Catalogs_SFMS_powerlawfit.pdf'])
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
    #panels = ['a)', 'b)']

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
    #sub1.scatter(_fit_logm, _fit_logsfr, c='k', marker='x', lw=3, s=40)

    for i_m, mrange in enumerate(mranges): 
        sub1.fill_between(mrange, [2.,2.], [-5.,-5], color=cols[i_m], linewidth=0, alpha=0.25)
        #sub1.text(mrange[0], -2.75, panels[i_m], ha='left', va='center', fontsize=20)
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
        #sub2.text(0.05, 0.95, panels[i_m], ha='left', va='top', transform=sub2.transAxes, fontsize=25)
        sub2.text(0.5, 0.9, '$'+str(mrange[0])+'< \mathrm{log}\, M_* <'+str(mrange[1])+'$',
                ha='center', va='center', transform=sub2.transAxes, fontsize=18)

    fig.subplots_adjust(wspace=.3)
    fig_name = ''.join([UT.fig_dir(), 'SFMSfit_demo.pdf'])
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close()
    return None


def SFRMstar_2Dgmm(n_comp_max=30): 
    ''' Compare the bestfit 2D GMM of the SFR-M* relation
    '''
    Cat = Cats.Catalog()

    fig = plt.figure(1,figsize=(4,4))
    plot_range = [[7., 12.], [-4., 2.]]

    sim = 'eagle_100myr'
    sub = fig.add_subplot(111)
    lbl = Cat.CatalogLabel(sim)
    _logm, _logsfr, weight, censat = Cat.Read(sim)
    iscen = (censat == 1)
    logm = _logm[iscen] 
    logsfr = _logsfr[iscen]

    gbest = sfr_mstar_gmm(logm, logsfr, n_comp_max=n_comp_max)
    
    DFM.hist2d(logm, logsfr, color='k', levels=[0.68, 0.95], range=plot_range, 
            plot_datapoints=True, fill_contours=False, plot_density=False, 
            contour_kwargs={'linewidths':1, 'linestyles':'dashed'}, 
            ax=sub) 
    #sub.scatter(logm, logsfr, color='k', s=4) 

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

    sub.set_xlim([7.5, 12.]) 
    sub.set_xticks([8., 10., 12.]) 
    sub.set_xlabel(r'log$\; M_* \;\;[M_\odot]$', labelpad=10, fontsize=20) 
    sub.set_ylim([-4., 2.]) 
    sub.set_yticks([-4., -2., 0., 2.]) 
    sub.set_ylabel(r'log ( SFR $[M_\odot \, yr^{-1}]$ )', labelpad=10, fontsize=20) 
    
    sub.text(0.1, 0.9, 'EAGLE',
            ha='left', va='top', transform=sub.transAxes, fontsize=20)

    fig_name = ''.join([UT.fig_dir(), 'SFRMstar_2Dgmm.pdf'])
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close() 
    return None 


def Catalog_GMMcomps(): 
    ''' Mark the GMM components on the SFR-M* relation of the 
    simulations/observations
    '''
    tscales = ['inst', '100myr']

    Cat = Cats.Catalog()
    # Read in various data sets
    sims_list = ['illustris', 'eagle', 'mufasa', 'scsam'] 
    obvs_list = ['tinkergroup', 'nsa_dickey'] 

    fig = plt.figure(1, figsize=(20,7.5))
    bkgd = fig.add_subplot(111, frameon=False)
    plot_range = [[7., 12.], [-4., 2.]]

    # plot SFR-M* for the observations 
    sub0 = fig.add_subplot(2,5,5)
    for i_c, cat in enumerate(obvs_list): 
        logMstar, logSFR, weight, censat = Cat.Read(cat)
        iscen = (censat == 1)

        # fit the SFMS  
        fSFMS = fstarforms()
        fit_logm, fit_logsfr = fSFMS.fit(logMstar[iscen], logSFR[iscen], 
                method='gaussmix', forTest=True) 

        DFM.hist2d(logMstar[iscen], logSFR[iscen], color='k',#'C'+str(i_c), 
                levels=[0.68, 0.95], range=plot_range, 
                plot_datapoints=True, fill_contours=False, plot_density=False, 
                contour_kwargs={'linewidths':1, 'linestyles':'dotted'}, 
                alpha=0.5, ax=sub0) 

        sig_sfms, w_sfms = np.zeros(len(fit_logm)), np.zeros(len(fit_logm))
        for i_m, fitlogm in enumerate(fit_logm): 
            # sfms component 
            sfms = (fSFMS._gmix_means[i_m] == fit_logsfr[i_m]-fit_logm[i_m])
            sig_sfms[i_m] = np.sqrt(fSFMS._gmix_covariances[i_m][sfms])
            w_sfms[i_m] = fSFMS._gmix_weights[i_m][sfms][0]
            # "quenched" component 
            quenched = (range(len(fSFMS._gmix_means[i_m])) == fSFMS._gmix_means[i_m].argmin()) & (fSFMS._gmix_means[i_m] != fit_logsfr[i_m]-fit_logm[i_m])
            if np.sum(quenched) > 0: 
                sub0.errorbar([fitlogm+0.01], fSFMS._gmix_means[i_m][quenched]+fitlogm, 
                             yerr=np.sqrt(fSFMS._gmix_covariances[i_m][quenched]), 
                             fmt='.C1')
                #sub0.scatter([fitlogm+0.01], fSFMS._gmix_means[i_m][quenched]+fitlogm, #s=40.*np.array(fSFMS._gmix_weights[i_m][quenched]), 
                #        color='C1')
            # other component 
            other = (fSFMS._gmix_means[i_m] != fit_logsfr[i_m]-fit_logm[i_m]) & (fSFMS._gmix_means[i_m] != fSFMS._gmix_means[i_m].min())
            if np.sum(other) > 0: 
                sub0.errorbar([fitlogm + 0.01*(i+2) for i in range(np.sum(other))], 
                             fSFMS._gmix_means[i_m][other]+fitlogm, 
                             yerr=np.sqrt(fSFMS._gmix_covariances[i_m][other]), 
                             fmt='.C2')
                #sub0.scatter([fitlogm + 0.01*(i+2) for i in range(np.sum(other))], fSFMS._gmix_means[i_m][other]+fitlogm, #s=40.*np.array(fSFMS._gmix_weights[i_m][other]), color='C2')
        sub0.errorbar(fit_logm, fit_logsfr, yerr=sig_sfms, fmt='.C0')
        #sub0.scatter(fit_logm, fit_logsfr, #s=40.*np.array(w_sfms), 
        #        color='C0')

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
            iscen = (censat == 1)
            if i_c == 0: 
                sub.text(0.1, 0.9, 'SFR ['+(lbl.split('[')[-1]).split(']')[0]+']', 
                        ha='left', va='center', transform=sub.transAxes, fontsize=20)

            # fit the SFMS  
            fSFMS = fstarforms()
            fit_logm, fit_logsfr = fSFMS.fit(logMstar[iscen], logSFR[iscen], 
                    method='gaussmix', forTest=True) 

            DFM.hist2d(logMstar[iscen], logSFR[iscen], color='k',#C'+str(i_c+2), 
                    levels=[0.68, 0.95], range=plot_range, 
                    plot_datapoints=True, fill_contours=False, plot_density=False, 
                    contour_kwargs={'linewidths':1, 'linestyles':'dotted'}, 
                    ax=sub) 
            if i_t == 1: 
                sub.text(0.9, 0.1, lbl.split('[')[0], ha='right', va='center', 
                        transform=sub.transAxes, fontsize=20)

            sig_sfms, w_sfms = np.zeros(len(fit_logm)), np.zeros(len(fit_logm))
            for i_m, fitlogm in enumerate(fit_logm): 
                # sfms component 
                sfms = (fSFMS._gmix_means[i_m] == fit_logsfr[i_m]-fit_logm[i_m])
                sig_sfms[i_m] = np.sqrt(fSFMS._gmix_covariances[i_m][sfms])
                w_sfms[i_m] = fSFMS._gmix_weights[i_m][sfms][0]
                # "quenched" component 
                quenched = (range(len(fSFMS._gmix_means[i_m])) == fSFMS._gmix_means[i_m].argmin()) & (fSFMS._gmix_means[i_m] != fit_logsfr[i_m]-fit_logm[i_m])
                if np.sum(quenched) > 0: 
                    sub.errorbar([fitlogm+0.01], fSFMS._gmix_means[i_m][quenched]+fitlogm, 
                                 yerr=np.sqrt(fSFMS._gmix_covariances[i_m][quenched]), 
                                 fmt='.C1')
                    #sub.scatter([fitlogm+0.01], fSFMS._gmix_means[i_m][quenched]+fitlogm, #s=40.*np.array(fSFMS._gmix_weights[i_m][quenched]), 
                    #             color='C1')
                # other component 
                other = (fSFMS._gmix_means[i_m] != fit_logsfr[i_m]-fit_logm[i_m]) & (fSFMS._gmix_means[i_m] != fSFMS._gmix_means[i_m].min())
                if np.sum(other) > 0: 
                    sub.errorbar([fitlogm + 0.01*(i+2) for i in range(np.sum(other))], 
                                 fSFMS._gmix_means[i_m][other]+fitlogm, 
                                 yerr=np.sqrt(fSFMS._gmix_covariances[i_m][other]), 
                                 fmt='.C2')
                    #sub.scatter([fitlogm + 0.01*(i+2) for i in range(np.sum(other))], 
                    #             fSFMS._gmix_means[i_m][other]+fitlogm, #s=40.*np.array(fSFMS._gmix_weights[i_m][other]), 
                    #            color='C2')
            sub.errorbar(fit_logm, fit_logsfr, yerr=sig_sfms, fmt='.C0')
            #sub.scatter(fit_logm, fit_logsfr, #s=40.*np.array(w_sfms), color='C0')
    
    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', labelpad=15, fontsize=25) 
    bkgd.set_ylabel(r'log ( SFR $[M_\odot \, yr^{-1}]$ )', labelpad=15, fontsize=25) 
    
    fig.subplots_adjust(wspace=0.15, hspace=0.15)
    
    fig_name = ''.join([UT.fig_dir(), 'Catalogs_GMMcomps.pdf'])
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close()
    return None 


def GMMcomp_composition(n_mc=10): 
    ''' Plot the fractional composition of the different GMM components 
    along with galaxies with zero SFRs for the different catalogs
    '''
    cats = ['illustris', 'eagle', 'mufasa', 'scsam']
    tscales = ['inst', '100myr']

    fig = plt.figure(figsize=(4*len(cats),4*len(tscales)))
    bkgd = fig.add_subplot(111, frameon=False)

    for i_c, c in enumerate(cats): 
        for i_t, tscale in enumerate(tscales):
            Cat = Cats.Catalog()
            logM, logSFR, w, censat = Cat.Read(c+'_'+tscale, keepzeros=True, silent=True)
            iscen = (censat == 1)
            iscen_nz = iscen & np.invert(Cat.zero_sfr) # SFR > 0 
            iscen_z = iscen & Cat.zero_sfr # SFR == 0 
            assert np.sum(iscen) == np.sum(iscen_nz) + np.sum(iscen_z) # snaity check 
            
            f_zeros, f_sfmss, f_qs, f_other0s, f_other1s= [], [], [], [], []  
            for i in range(n_mc): 
                # fit the SFMS using GMM fitting
                fSFMS = fstarforms()
                fit_logm, fit_logsfr = fSFMS.fit(logM[iscen_nz], logSFR[iscen_nz],
                        method='gaussmix', fit_range=[8.4, 12.], dlogm=0.2, Nbin_thresh=0, 
                        forTest=True, silent=True) 
                
                mbins = fSFMS._tests['mbin_mid'] 
                gbests = fSFMS._tests['gbests'] 
                
                f_zero  = np.zeros(len(mbins))
                f_sfms  = np.zeros(len(mbins))
                f_q     = np.zeros(len(mbins))
                f_other0= np.zeros(len(mbins))
                f_other1= np.zeros(len(mbins))
                
                for i_m, mbin, gbest in zip(range(len(mbins)), mbins, gbests): 
                    means_i = gbest.means_.flatten() 
                    weights_i = gbest.weights_
                    ncomp_i = len(weights_i) 
                    
                    sfthresh = (means_i > -11.) 

                    # sfms component 
                    if np.sum(sfthresh) > 1: sfms = (means_i == (means_i[sfthresh])[weights_i[sfthresh].argmax()])
                    else: sfms = sfthresh.copy()
                    if np.sum(sfms): 
                        f_sfms[i_m] = np.sum(weights_i[sfms])
                        mu_sfms = means_i[sfms]
                    elif np.sum(sfms) > 1: raise ValueError

                    # "quenched" component 
                    quenched = np.zeros(ncomp_i, dtype=bool) 
                    quenched[means_i.argmin()] = True 
                    quenched = quenched & np.invert(sfthresh)
                    f_q[i_m] = np.sum(weights_i[quenched])

                    # other in between the SFMS and quenched components
                    if np.sum(sfms): other0 = np.invert(sfms) & np.invert(quenched) & (means_i < mu_sfms)
                    else: other0 = np.invert(sfms) & np.invert(quenched)
                    if np.sum(other0): f_other0[i_m] = np.sum(weights_i[other0])
                    
                    # other above the SFMS
                    if np.sum(sfms): other1 = np.invert(sfms) & np.invert(quenched) & (means_i > mu_sfms)
                    else: other1 = np.zeros(len(means_i), dtype=bool) 
                    if np.sum(other1): f_other1[i_m] = np.sum(weights_i[other1])

                    assert ncomp_i == np.sum(sfms) + np.sum(quenched) + np.sum(other0) + np.sum(other1)
                    
                    mbin_iscen_z = iscen & Cat.zero_sfr & \
                            (logM > mbin-0.5*fSFMS._dlogm) & (logM < mbin+0.5*fSFMS._dlogm)
                    mbin_iscen = iscen & \
                            (logM > mbin-0.5*fSFMS._dlogm) & (logM < mbin+0.5*fSFMS._dlogm)
                    f_zero[i_m] = float(np.sum(mbin_iscen_z))/float(np.sum(mbin_iscen))
                f_zeros.append(f_zero)
                f_sfmss.append((1.-f_zero)*f_sfms)
                f_qs.append((1.-f_zero)*f_q)
                f_other0s.append((1.-f_zero)*f_other0)
                f_other1s.append((1.-f_zero)*f_other1)
            
            f_zero = np.mean(f_zeros, axis=0) 
            f_sfms = np.mean(f_sfmss, axis=0) 
            f_q = np.mean(f_qs, axis=0) 
            f_other0 = np.mean(f_other0s, axis=0) 
            f_other1 = np.mean(f_other1s, axis=0) 

            sub = fig.add_subplot(len(tscales), len(cats), 1+i_c+len(cats)*i_t)  
            sub.fill_between(mbins, np.zeros(len(mbins)), f_zero, # SFR = 0 
                    linewidth=0, color='C3') 
            sub.fill_between(mbins, f_zero, f_zero+f_q,              # Quenched
                    linewidth=0, color='C1') 
            sub.fill_between(mbins, f_zero+f_q, f_zero+f_q+f_other0,   # other0
                    linewidth=0, color='C2') 
            sub.fill_between(mbins, f_zero+f_q+f_other0, f_zero+f_q+f_other0+f_sfms, # SFMS 
                    linewidth=0, color='C0') 
            sub.fill_between(mbins, f_zero+f_q+f_other0+f_sfms, f_zero+f_q+f_other0+f_sfms+f_other1, # SFMS 
                    linewidth=0, color='C2') 
            #sub.set_xlim([fit_logm.min(), fit_logm.max()]) 
            sub.set_xlim([8.8, 11.5])#fit_logm.min(), fit_logm.max()]) 
            sub.set_xticks([9., 10., 11.]) 
            if i_t == 0: sub.set_xticklabels([]) 
            sub.set_ylim([0., 1.]) 
            if i_c != 0: sub.set_yticks([]) 

            lbl = Cat.CatalogLabel(c+'_'+tscale)
            if i_c == 0: 
                sub.text(0.1, 0.875, 'SFR ['+(lbl.split('[')[-1]).split(']')[0]+']', color='white', 
                        ha='left', va='center', transform=sub.transAxes, fontsize=20)
            if i_t == 1: 
                sub.text(0.9, 0.1, lbl.split('[')[0], ha='right', va='center', color='white', 
                        transform=sub.transAxes, fontsize=20)#, path_effects=[path_effects.withSimplePatchShadow()])
            #if i_c == 0: 
            #    sub.text(0.05, 0.95, lbl.split('[')[0]+'\n ['+lbl.split('[')[-1], ha='left', va='top', color='white', 
            #            transform=sub.transAxes, fontsize=15)
            #else: 
            #    sub.text(0.05, 0.95, lbl.split('[')[0], ha='left', va='top', color='white', 
            #            transform=sub.transAxes, fontsize=15)
            if (i_t == 0) and (i_c == len(cats)-1):  
                p1 = Rectangle((0, 0), 1, 1, linewidth=0, fc="C3")
                p2 = Rectangle((0, 0), 1, 1, linewidth=0, fc="C1")
                p3 = Rectangle((0, 0), 1, 1, linewidth=0, fc="C2")
                p4 = Rectangle((0, 0), 1, 1, linewidth=0, fc="C0")
                sub.legend([p1, p2, p3, p4][::-1], ['SFR = 0', '``quenched"', 'other', 'SFMS'][::-1], 
                        loc='upper right', prop={'size': 12}) #bbox_to_anchor=(1.1, 1.05))

    bkgd.set_xlabel(r'log$\; M_* \;\;[M_\odot]$', labelpad=10, fontsize=30) 
    bkgd.set_ylabel(r'GMM component fractions', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    fig.subplots_adjust(wspace=0.05, hspace=0.1)
    fig_name = ''.join([UT.fig_dir(), 'GMMcomp_composition.pdf'])
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close()
    return None 


def Pssfr_res_impact(): 
    ''' Plot illustrating the impact of star-particle resolution  
    on the P(SSFR) distribution. 
    '''
    n_mc = 100
    fig = plt.figure(figsize=(12,6))
    bkgd = fig.add_subplot(111, frameon=False)
    for i_c, cat_name in enumerate(['illustris_100myr', 'eagle_100myr', 'mufasa_100myr']):
        for i_m in range(2):  
            if cat_name == 'illustris_100myr':
                if i_m == 0: mbin = [8.2, 8.4]
                else: mbin = [10.6, 10.8]
                dsfr = 0.016
            elif cat_name == 'eagle_100myr':
                if i_m == 0: mbin = [8.4, 8.6]
                else: mbin = [10.6, 10.8]
                dsfr = 0.018
            elif cat_name == 'mufasa_100myr':
                if i_m == 0: mbin = [9.2, 9.4]
                else: mbin = [10.6, 10.8]
                dsfr = 0.182
            # read in SFR and M* 
            Cat = Cats.Catalog()
            _logM, _logSFR, w, censat = Cat.Read(cat_name, keepzeros=True)
            _SFR = 10**_logSFR

            sub = fig.add_subplot(2,3,i_c+1+3*i_m)
            iscen = (censat == 1)
            iscen_nz_mbin = iscen & np.invert(Cat.zero_sfr) & (_logM > mbin[0]) & (_logM < mbin[1])
            iscen_z_mbin  = iscen & Cat.zero_sfr & (_logM > mbin[0]) & (_logM < mbin[1])
            ngal_bin = float(np.sum(iscen & (_logM > mbin[0]) & (_logM < mbin[1])))

            hs, hs_nz = [], []
            for ii in range(n_mc):
                sfr_nz = _SFR[iscen_nz_mbin] + dsfr*2*np.random.uniform(size=np.sum(iscen_nz_mbin))
                sfr_z = dsfr * np.random.uniform(size=np.sum(iscen_z_mbin))

                logssfr_nz = np.log10(sfr_nz) - _logM[iscen_nz_mbin]
                logssfr_z = np.log10(sfr_z) - _logM[iscen_z_mbin]
                logssfr = np.concatenate([logssfr_nz, logssfr_z])

                h0, h1 = np.histogram(logssfr, bins=40, range=[-16., -8.])
                hs.append(h0)

            for ii in range(n_mc):
                sfr_nz = _SFR[iscen_nz_mbin] + dsfr*2*np.random.uniform(size=np.sum(iscen_nz_mbin))
                logssfr_nz = np.log10(sfr_nz) - _logM[iscen_nz_mbin]
                h0_nz, _ = np.histogram(logssfr_nz, bins=40, range=[-16., -8.])
                hs_nz.append(h0_nz)

            hs = np.array(hs)/ngal_bin
            hs_nz = np.array(hs_nz)/ngal_bin

            bar_x, bar_y = UT.bar_plot(h1, np.mean(hs,axis=0))
            sub.plot(bar_x, bar_y, c='C0', label='w/ SFR $=0$')#/ngal_bin)
            sub.errorbar(0.5*(h1[1:] + h1[:-1])-0.02, np.mean(hs, axis=0), yerr=np.std(hs, axis=0),
                         fmt='.C0', markersize=.5)
            
            bar_x, bar_y = UT.bar_plot(h1, np.mean(hs_nz,axis=0))
            sub.plot(bar_x, bar_y, c='k', ls='--', label='w/o SFR $=0$')
            sub.errorbar(0.5*(h1[1:] + h1[:-1])+0.02, np.mean(hs_nz, axis=0), 
                    yerr=np.std(hs_nz, axis=0), fmt='.k', markersize=.5)

            sub.set_xlim([-13.25, -8.8])
            if i_m == 0: sub.set_xticks([])
            else: sub.set_xticks([-13., -11., -9.]) 
            sub.set_ylim([0., 0.25]) 
            if i_c == 0: sub.set_yticks([0., 0.1, 0.2]) 
            else: sub.set_yticks([]) 
            lbl = Cat.CatalogLabel(cat_name).split('[')[0]
            if i_m == 0: sub.set_title(lbl, fontsize=20) 
            sub.text(0.5, 0.92, '$'+str(mbin[0])+'< \log M_* <'+str(mbin[1])+'$',
                ha='center', va='top', transform=sub.transAxes, fontsize=15)
    
            if (i_c == 2) and (i_m == 0): sub.legend(loc='lower left', bbox_to_anchor=(0.01, 0.48), frameon=False, prop={'size':15}) 
    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_ylabel(r'P(log SSFR  $[yr^{-1}])$', labelpad=10, fontsize=20)
    bkgd.set_xlabel(r'log SSFR  $[yr^{-1}]$', labelpad=10, fontsize=20)

    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    fig_name = ''.join([UT.fig_dir(), 'Pssfr_res_impact.pdf'])
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close()
    return None


def Mlim_res_impact(n_mc=20, seed=10): 
    '''
    '''
    catalogs = ['illustris_100myr', 'eagle_100myr', 'mufasa_100myr']
    catnames = ['Illustris', 'EAGLE', 'MUFASA']
    fig = plt.figure(figsize=(4*len(catalogs),4))
    bkgd = fig.add_subplot(111, frameon=False)

    Cat = Cats.Catalog()
    for i_n, name in enumerate(catalogs): 
        _logm, _logsfr, _, censat = Cat.Read(name, keepzeros=True, silent=True)
        iscen_nz = (censat == 1) & np.invert(Cat.zero_sfr)
        iscen_z  = (censat == 1) & Cat.zero_sfr
        ngal_nz = np.sum(iscen_nz)
        ngal_z = np.sum(iscen_z) 

        _sfr = 10**_logsfr
        dsfr = Cat._SFR_resolution(name) # SFR resolution 
            
        logm = np.concatenate([_logm[iscen_nz], _logm[iscen_z]]) # all log M*
    
        # fit with scattered SFR 
        fSFMS_s = fstarforms()
        fit_logsfr_s = [] 
        np.random.seed(seed)
        for i in range(n_mc): 
            sfr_low = _sfr[iscen_nz] - dsfr
            sfr_low = np.clip(sfr_low, 0., None) 
            logsfr_nz = np.log10(np.random.uniform(sfr_low, _sfr[iscen_nz]+dsfr, size=ngal_nz))
            logsfr_z = np.log10(dsfr * np.random.uniform(size=ngal_z))
            logsfr = np.concatenate([logsfr_nz, logsfr_z]) # scattered log SFR
        
            fit_logm_s, fit_logsfr_s_i = fSFMS_s.fit(logm, logsfr, method='gaussmix', 
                    dlogm=0.2, fit_range=[8.,11.], maxcomp=4, forTest=True, silent=True) 
            fit_logsfr_s.append(fit_logsfr_s_i)
    
        sig_fit_logsfr_s = np.std(np.array(fit_logsfr_s), ddof=1, axis=0) # scatter in the fits
        fit_logsfr_s = np.mean(fit_logsfr_s, axis=0) # mean fit

        # fit without scatter
        fSFMS = fstarforms()
        fit_logsfr_ns = [] 
        for i in range(n_mc): 
            fit_logm_ns, fit_logsfr_ns_i = fSFMS.fit(_logm[iscen_nz], _logsfr[iscen_nz], method='gaussmix', 
                    dlogm=0.2, fit_range=[8.,11.], maxcomp=3, forTest=True, silent=True) 
            fit_logsfr_ns.append(fit_logsfr_ns_i)
        sig_fit_logsfr_ns = np.std(np.array(fit_logsfr_ns), ddof=1, axis=0) # scatter in the fits
        fit_logsfr_ns = np.mean(fit_logsfr_ns, axis=0) # mean fit
        sig_tot = np.clip(sig_fit_logsfr_s + sig_fit_logsfr_ns, 0.1, None) 
        #print 'sig=', sig_tot

        # check that the mbins are equal between the fit w/ scatter and fit w/o scatter
        assert np.array_equal(fSFMS._tests['mbin_mid'], fSFMS_s._tests['mbin_mid'])

        dfit = fit_logsfr_ns - fit_logsfr_s # change in SFMS fit caused by resolution limit 
        #print np.abs(dfit)/sig_tot 

        # log M* where resolution limit causes the SFMS to shift by 0.1 dex
        mbin_mid = np.array(fSFMS._tests['mbin_mid'])
        mlim = (mbin_mid[(np.abs(dfit) > 0.1)]).max() + 0.5*fSFMS._dlogm 
        
        # lets plot this stuff 
        sub = fig.add_subplot(1,len(catalogs),i_n+1)
        DFM.hist2d(_logm[iscen_nz], _logsfr[iscen_nz], color='k',
                levels=[0.68, 0.95], range= [[7., 12.], [-4., 2.]],
                plot_datapoints=False, fill_contours=False, plot_density=False, 
                contour_kwargs={'linewidths':1, 'linestyles':'dotted'}, ax=sub) 

        sub.scatter(fit_logm_s, fit_logsfr_s, marker='x', color='C1', lw=1, s=40, 
                label='SFR resamp.')
        sub.scatter(fit_logm_ns, fit_logsfr_ns, marker='x', color='k', lw=1, s=40, 
                label='SFR no resamp.')
        sub.text(0.95, 0.05, '$\log\,M_{\lim}='+str(round(mlim,2))+'$', 
                 ha='right', va='bottom', transform=sub.transAxes, fontsize=15)
        sub.vlines(mlim, -4., 2., color='k', linestyle='--', linewidth=0.5)
        sub.fill_between([7.5, mlim], [-3., -3.], [2., 2.], color='k', alpha=0.2)
        sub.set_xlim([7.5, 11.8])
        sub.set_xticks([8., 9., 10., 11.]) 
        sub.set_ylim([-3., 2.])
        sub.set_title(catnames[i_n], fontsize=20)
        if i_n != 0: sub.set_yticks([]) 
        if i_n == len(catalogs)-1: sub.legend(loc='upper left', bbox_to_anchor=(-0.075, 1.), 
                handletextpad=-0.02, frameon=False, prop={'size':15}) 
    
    bkgd.set_xlabel(r'log $M_* \;\;[M_\odot]$', labelpad=10, fontsize=25) 
    bkgd.set_ylabel(r'log SFR $[M_\odot \, yr^{-1}]$', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    fig.subplots_adjust(wspace=0.05)
    fig_name = ''.join([UT.fig_dir(), 'Mlim_res_impact.pdf'])
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


def _GMM_comp_test(name): 
    ''' Test GMM with more than 3 components. More specifically 
    plot the P(SSFR) for stellar mass bins where BIC prefers
    more than 3 components.
    '''
    Cat = Cats.Catalog()
    # Read in various data sets
        
    logMstar, logSFR, weight, censat = Cat.Read(name)
    iscen = (censat == 1)
    logm = logMstar[iscen]
    logsfr = logSFR[iscen]

    # fit the SFMS  
    fSFMS = fstarforms()
    _, _ = fSFMS.fit(logm, logsfr, method='gaussmix', forTest=True) 
    
    fSFMS_more = fstarforms()
    _, _ = fSFMS_more.fit(logm, logsfr, method='gaussmix', max_comp=30, forTest=True) 
    
    assert fSFMS._tests['mbin_mid'] == fSFMS_more._tests['mbin_mid']

    bins = [] 
    for im in range(len(fSFMS._tests['mbin_mid'])): 
        if len(fSFMS_more._tests['gbests'][im].means_) > 3: 
            bins.append(im) 
    if len(bins) == 0: return None 

    fig = plt.figure(1, figsize=(4*len(bins),8))
    bkgd = fig.add_subplot(111, frameon=False)
    xx = np.linspace(-14., -9, 100)

    for ii, im in enumerate(bins): 
        sub1 = fig.add_subplot(2,len(bins),ii+1)
        sub2 = fig.add_subplot(2,len(bins),ii+1+len(bins))
        # within mass bin 
        mbin_mid = fSFMS._tests['mbin_mid'][im]
        in_mbin = np.where(
                (logm > mbin_mid-0.5*fSFMS._dlogm) & 
                (logm < mbin_mid+0.5*fSFMS._dlogm))
        sub1.text(0.5, 0.9, ''.join([str(round(mbin_mid-0.5*fSFMS._dlogm,1)), 
                    '$<$ log$\,M_* <$', str(round(mbin_mid+0.5*fSFMS._dlogm,1))]), 
                ha='center', va='center', transform=sub1.transAxes, fontsize=20)
        sub1.text(0.9, 0.5, '$k = '+str(len(fSFMS._tests['gbests'][im].means_))+'$', 
                ha='right', va='center', transform=sub1.transAxes, fontsize=20)
        sub2.text(0.9, 0.5, '$k = '+str(len(fSFMS_more._tests['gbests'][im].means_))+'$', 
                ha='right', va='center', transform=sub2.transAxes, fontsize=20)
        
        # P(SSFR) histogram 
        _ = sub1.hist(logsfr[in_mbin] - logm[in_mbin], bins=32, 
                range=[-14., -8.], normed=True, histtype='step', color='k', linewidth=1.75)
        _ = sub2.hist(logsfr[in_mbin] - logm[in_mbin], bins=32, 
                range=[-14., -8.], normed=True, histtype='step', color='k', linewidth=1.75)
        # plot the fits 
        gmm_weights = fSFMS._tests['gbests'][im].weights_
        gmm_means = fSFMS._tests['gbests'][im].means_.flatten() 
        gmm_vars = fSFMS._tests['gbests'][im].covariances_.flatten() 
        for ii, icomp in enumerate(gmm_means.argsort()[::-1]): 
            if ii == 0: 
                sub1.plot(xx, gmm_weights[icomp]*MNorm.pdf(xx, gmm_means[icomp], gmm_vars[icomp]), lw=2, ls='--')
                gmm_tot = gmm_weights[icomp]*MNorm.pdf(xx, gmm_means[icomp], gmm_vars[icomp])
            else: 
                sub1.plot(xx, gmm_weights[icomp]*MNorm.pdf(xx, gmm_means[icomp], gmm_vars[icomp]), lw=1.5, ls='--')
                gmm_tot += gmm_weights[icomp]*MNorm.pdf(xx, gmm_means[icomp], gmm_vars[icomp])
        #sub.plot(xx, gmm_tot, c='r', lw=2, ls='-')
        gmm_weights = fSFMS_more._tests['gbests'][im].weights_
        gmm_means = fSFMS_more._tests['gbests'][im].means_.flatten() 
        gmm_vars = fSFMS_more._tests['gbests'][im].covariances_.flatten() 
        for ii, icomp in enumerate(gmm_means.argsort()[::-1]): 
            if ii == 0: 
                sub2.plot(xx, gmm_weights[icomp]*MNorm.pdf(xx, gmm_means[icomp], gmm_vars[icomp]), lw=2, ls='--')
                gmm_tot = gmm_weights[icomp]*MNorm.pdf(xx, gmm_means[icomp], gmm_vars[icomp])
            else: 
                sub2.plot(xx, gmm_weights[icomp]*MNorm.pdf(xx, gmm_means[icomp], gmm_vars[icomp]), lw=1.5, ls='--')
                gmm_tot += gmm_weights[icomp]*MNorm.pdf(xx, gmm_means[icomp], gmm_vars[icomp])

        sub1.set_xlim([-13, -8]) # x-axis
        sub1.set_xticks([-13, -12, -11, -10, -9, -8])
        sub2.set_xlim([-13, -8]) # x-axis
        sub2.set_xticks([-13, -12, -11, -10, -9, -8])
        sub1.set_ylim([0., 1.4]) # y-axis
        sub1.set_yticks([0., 0.5, 1.])
        sub2.set_ylim([0., 1.4]) # y-axis
        sub2.set_yticks([0., 0.5, 1.])

    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_ylabel(r'P(log SSFR  $[yr^{-1}])$', labelpad=10, fontsize=20) 
    bkgd.set_xlabel(r'log SSFR  $[yr^{-1}]$', labelpad=10, fontsize=20) 
    
    fig.subplots_adjust(wspace=0.2, hspace=0.2)
    fig_name = ''.join([UT.fig_dir(), 'GMMcomp.test.', name, '.png'])
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close() 
    return None


if __name__=="__main__": 
    #SFRMstar_2Dgmm(n_comp_max=50)
    #Catalogs_SFR_Mstar()
    #Catalogs_Pssfr()
    GroupFinder()
    #Catalogs_SFMS_powerlawfit()
    #SFMSfit_example()

    #for tscale in ['100myr']:# 'inst', '10myr', '100myr', '1gyr']: 
    #    Catalog_SFMS_fit(tscale)
    #Catalog_GMMcomps()
    #_GMM_comp_test('tinkergroup')
    #_GMM_comp_test('nsa_dickey')
    #Pssfr_res_impact()
    #Mlim_res_impact(n_mc=10)
    #GMMcomp_composition(n_mc=50)
    #for c in ['illustris', 'eagle', 'mufasa', 'scsam']: 
    #    for tscale in ['inst', '100myr']:#'10myr', '100myr', '1gyr']: 
    #        _GMM_comp_test(c+'_'+tscale)
    #for c in ['illustris', 'eagle', 'mufasa']:
    #    _SFR_tscales(c)
    #for c in ['scsam']: #'illustris', 'eagle', 'mufasa']:
    #    for tscale in ['inst', '10myr', '100myr', '1gyr']: 
    #        try: 
    #            _SFMSfit_assess(c+'_'+tscale, method='gaussmix')
    #        except (ValueError, NotImplementedError): 
    #            continue 
