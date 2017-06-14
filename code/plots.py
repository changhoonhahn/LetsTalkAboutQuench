import numpy as np 
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt 
from matplotlib import lines as mlines
from ChangTools.plotting import prettyplot
from ChangTools.plotting import prettycolors

# -- Local --
import util as UT
import catalogs as Cats
import galprop as Gprop
import corner as DFM 


def SFR_Mstar(name, yrange=None, **kwargs): 
    ''' plot SFR histogram in log M* bins for specified catalog
    '''
    assert name

    catalog_list = ['santacruz1', 'santacruz2', 'tinkergroup', 'illustris1', 'illustris2', 'nsa_dickey', 'mufasa']
    if name not in catalog_list: 
        raise ValueError("catalog must be one of ["+', '.join(catalog_list)+"]")
    
    Cat = Cats.Catalog()
    logMstar, logSFR, weight = Cat.Read(name) # read in values 
    
    logSFR_bin_mids, PlogSFRs, mass_bins, counts, wtots = Gprop.sSFR_Mstar(logSFR, logMstar, **kwargs)
    n_mbins = len(PlogSFRs)
    
    # determine paneling 
    if n_mbins > 10: 
        panels = (3, int(np.ceil(np.float(n_mbins+1)/3.)))
    else: 
        panels = (2, int(np.ceil(np.float(n_mbins+1)/2.)))
    
    prettyplot()
    fig = plt.figure(1, figsize=(8*panels[1], 6*panels[0]))
    bkgd = fig.add_subplot(111, frameon=False)
    
    sub = fig.add_subplot(panels[0], panels[1], 1)
    #bovy.scatterplot(logMstar, logSFR, cmap='Blues', 
    #        levels=[0.68, 0.95], aspect=0.75, gcf=True, scatter=True, 
    #        xrange=[6., 12.], yrange=[-4., 2.])
    sub.set_ylabel(r'$\mathtt{log \; M_* \;\; [M_\odot]}$', fontsize=20) 
    sub.set_xlabel(r'$\mathtt{log \; SFR \;\; [M_\odot yr^{-1}]}$', fontsize=20) 

    for i_m in range(n_mbins): 
        sub = fig.add_subplot(panels[0], panels[1], i_m+2)
        sub.plot(logSFR_bin_mids[i_m], PlogSFRs[i_m], lw=3)

        sub.text(0.8, 0.9, str(round(mass_bins[i_m][0],2))+'-'+str(round(mass_bins[i_m][1],2)), 
                fontsize=30, ha='center', va='center', transform=sub.transAxes)

        if 'weights' in kwargs.keys(): 
            sub.text(0.3, 0.9, "$N_{gal}=$"+str(counts[i_m])+', $w_{tot}$ = '+str(wtots[i_m]),
                    ha='center', va='center', transform=sub.transAxes)
        else: 
            sub.text(0.3, 0.9, "$N_{gal}=$"+str(counts[i_m]), 
                    ha='center', va='center', transform=sub.transAxes)
        sub.set_xlim([-12.5, -9.])
        if yrange is not None: 
            sub.set_ylim(yrange)

    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_ylabel(r'$\mathtt{log \; M_* \;\; [M_\odot]}$', labelpad=20, fontsize=30) 
    bkgd.set_xlabel(r'$\mathtt{log \; SFR \;\; [M_\odot yr^{-1}]}$', labelpad=20, fontsize=30) 
    
    fig_name = ''.join([UT.fig_dir(), name, '.SFR_Mstar.png'])
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close()
    return None 


def Pssfr_Catalogs(xrange=[-12.5, -9.], yrange=[0., 1.8], **kwargs): 
    ''' compare SFR/sSFR histograms of various simlations and data in log M* bins 
    '''
    # Read in various data sets
    Cat = Cats.Catalog()
    catalog_list = Cat.catalog_list

    catalog_labels = [] # Labels 
    logSFRs, logMstars, weights = [], [], [] 
    for cat in catalog_list: 
        catalog_labels.append(Cat.CatalogLabel(cat))
        logMstar, logSFR, weight = Cat.Read(cat)
        logSFRs.append(logSFR)
        logMstars.append(logMstar) 
        weights.append(weight)

    prettyplot()
    for i_data in range(len(logSFRs)):
        logSFR_bin_edges, PlogSFRs, mass_bins, counts, wtots = \
                Gprop.sSFR_Mstar(logSFRs[i_data], logMstars[i_data], 
                        weights=weights[i_data], **kwargs)
        n_mbins = len(PlogSFRs)
        
        # determine figure paneling 
        if i_data == 0: 
            if n_mbins > 10: 
                panels = (3, int(np.ceil(np.float(n_mbins)/3.)))
            else: 
                panels = (2, int(np.ceil(np.float(n_mbins)/2.)))
            
            fig = plt.figure(1, figsize=(8*panels[1], 6*panels[0]))
            bkgd = fig.add_subplot(111, frameon=False)

        for i_m in range(n_mbins): 
            sub = fig.add_subplot(panels[0], panels[1], i_m+1)

            xx_bar, yy_bar = UT.bar_plot(logSFR_bin_edges[i_m], PlogSFRs[i_m])
            sub.plot(xx_bar, yy_bar, lw=5, c=Cat.CatalogColors(catalog_list[i_data]), 
                    label=catalog_labels[i_data], alpha=0.75)

            sub.text(0.95, 0.9, 
                    str(round(mass_bins[i_m][0],2))+'-'+str(round(mass_bins[i_m][1],2)), 
                    fontsize=30, ha='right', va='center', transform=sub.transAxes)
            if xrange is not None: 
                sub.set_xlim(xrange)
            if yrange is not None: 
                sub.set_ylim(yrange)
            if i_m == 0:  
                sub.legend(loc='upper left', prop={'size':17})

    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_ylabel(r'$\mathtt{P(SSFR)}$', labelpad=20, fontsize=30) 
    bkgd.set_xlabel(r'$\mathtt{log \; SSFR \;\;[yr^{-1}]}$', labelpad=20, fontsize=30) 
    
    fig_name = ''.join([UT.fig_dir(), 'Pssfr_catalogs.', str(n_mbins), 'Mbins', '.png'])
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close()
    return None 


def SFR_Mstar_Catalogs(contour='dfm'): 
    ''' Compare SFR vs M* relation of various simlations and data
    '''
    Cat = Cats.Catalog()
    # Read in various data sets
    catalog_list = Cat.catalog_list

    catalog_labels = [] # Labels 
    logSFRs, logMstars, weights = [], [], [] 
    for cat in catalog_list: 
        catalog_labels.append(Cat.CatalogLabel(cat))
        logMstar, logSFR, weight = Cat.Read(cat)
        logSFRs.append(logSFR)
        logMstars.append(logMstar) 
        weights.append(weight)

    prettyplot()
    n_rows = int(np.ceil(np.float(len(logSFRs))/4.))
    fig = plt.figure(1, figsize=(32, 6*n_rows))
    bkgd = fig.add_subplot(111, frameon=False)

    for i_data in range(len(logSFRs)):
        sub = fig.add_subplot(n_rows, 4, i_data+1)
        if contour == 'dfm':  
            DFM.hist2d(logMstars[i_data], logSFRs[i_data], color='#1F77B4', 
                    levels=[0.68, 0.95], range=[[6., 12.], [-4., 2.]], 
                    plot_datapoints=True, fill_contours=False, plot_density=True, ax=sub) 
        elif contour == 'gaussianKDE': 
            xx, yy, f = UT.gaussianKDE_contour(logMstars[i_data], logSFRs[i_data], 
                    xmin=6., xmax=12., ymin=-4., ymax=2.)
            sub.contourf(xx, yy, f, cmap='Blues')
            sub.contour(xx, yy, f, colors='k')
            sub.set_xlim([6., 12.]) 
            sub.set_ylim([-4., 2.]) 
        elif contour == False: 
            sub.scatter(logMstars[i_data], logSFRs[i_data], c='k', s=2, lw=0)
            sub.set_xlim([6., 12.]) 
            sub.set_ylim([-4., 2.]) 
        else: 
            raise ValueError() 
        sub.text(0.1, 0.9, catalog_labels[i_data],
                ha='left', va='center', transform=sub.transAxes, fontsize=17)

    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel(r'$\mathtt{log \; M_* \;\;[M_\odot]}$', labelpad=20, fontsize=30) 
    bkgd.set_ylabel(r'$\mathtt{log \; SFR \;\;[M_\odot \, yr^{-1}]}$', labelpad=20, fontsize=30) 
    
    fig.subplots_adjust(wspace=0.15, hspace=0.15)
    
    if contour: 
        fig_name = ''.join([UT.fig_dir(), 'SFR_Mstar_catalogs.', contour, '.png'])
    else: 
        fig_name = ''.join([UT.fig_dir(), 'SFR_Mstar_catalogs.scatter.png'])
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close()
    return None 


def SFR_Mstar_SFMScut_QF(fit_method='lowMbin_extrap', scatter_method='constant', **fit_kwargs): 
    ''' compare SFR vs M* relation of various simlations and data
    '''
    Cat = Cats.Catalog()
    # Read in various data sets
    catalog_list = Cat.catalog_list 
    catalog_labels = [] # Labels 
    logSFRs, logMstars, weights = [], [], [] 
    for cat in catalog_list: 
        catalog_labels.append(Cat.CatalogLabel(cat))
        logMstar, logSFR, weight = Cat.Read(cat)
        logSFRs.append(logSFR)
        logMstars.append(logMstar) 
        weights.append(weight)
        
    prettyplot()
    n_rows = int(np.ceil(np.float(len(logSFRs))/4.))
    fig = plt.figure(1, figsize=(32, 6*n_rows))
    bkgd = fig.add_subplot(111, frameon=False)
    title = ''.join(['SFMS fit method: ', fit_method.replace('_', ' '), '; scatter method: ', scatter_method.replace('_', ' ')])
    plt.title(title)
        
    qfs = [] 
    for i_data in range(len(logSFRs)):
        sub = fig.add_subplot(n_rows, 4, i_data+1)
            
        DFM.hist2d(logMstars[i_data], logSFRs[i_data], color='#1F77B4', ax=sub,
                levels=[0.68, 0.95], range=[[6., 12.], [-4., 2.]]) 
        
        # plot best-fit SFMS 
        m_arr, sfr_sfms, sig_sfr = Gprop.SFMS_scatter(logSFRs[i_data], logMstars[i_data], 
                fit_method=fit_method, method=scatter_method, **fit_kwargs)
        sub.plot(m_arr, sfr_sfms, c='b', lw=3, ls='--', label='SFMS')
        #  SFMS cut at 3 sigma below best-fit SFMS
        sub.plot(m_arr, sfr_sfms - 3. * sig_sfr, c='r', lw=3, ls='-.', label='SFR cut')

        sub.text(0.1, 0.9, catalog_labels[i_data],
                ha='left', va='center', transform=sub.transAxes, fontsize=20)

        if i_data == 3: 
            sub.legend(loc='lower right', prop={'size':25}) 

        # calculate the quiescent fraction you would get using this cut off 
        sfr_cut = np.interp(logMstars[i_data], m_arr, sfr_sfms - 3. * sig_sfr)

        mbin_qf = np.arange(6., 12.5, 0.5) 
        mass_qf, qf = [], [] 
        for i_m in range(len(mbin_qf)-1): 
            in_mbin = np.where(
                    (logMstars[i_data] >= mbin_qf[i_m]) & 
                    (logMstars[i_data] < mbin_qf[i_m+1]) & 
                    (np.isnan(logSFRs[i_data]) == False))

            if len(in_mbin[0]) > 0: 
                mass_qf.append(0.5 * (mbin_qf[i_m] + mbin_qf[i_m+1]))
                qf.append(np.float(np.sum(logSFRs[i_data][in_mbin] < sfr_cut[in_mbin])) / np.float(len(in_mbin[0])))
        qfs.append([np.array(mass_qf), np.array(qf)])

    # plot the quiescent fraction you would get using this cut off 
    sub = fig.add_subplot(n_rows, 4, i_data+2)
    for i_qf in range(len(qfs)): 
        sub.plot(qfs[i_qf][0], qfs[i_qf][1], label=catalog_labels[i_qf], 
                c=Cat.CatalogColors(catalog_list[i_qf]), lw=3)
    sub.set_xlim([6., 12.]) 
    sub.set_ylim([0., 1.]) 
    sub.set_ylabel('$\mathtt{f_{Q}}$', fontsize=25)
    sub.yaxis.tick_right()
    sub.yaxis.set_label_position("right")
    sub.legend(loc='upper left', prop={'size':17}) 

    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel(r'$\mathtt{log \;M_* \;\;[M_\odot]}$', labelpad=20, fontsize=30) 
    bkgd.set_ylabel(r'$\mathtt{log \;SFR \;\;[M_\odot/yr^{-1}]}$', labelpad=20, fontsize=30) 

    fig.subplots_adjust(wspace=0.15, hspace=0.15)
    fig_name = ''.join([UT.fig_dir(), 'SFR_Mstar', 
        '.SFMSfit_', fit_method, '.scatter_', scatter_method, '.png'])
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close()
    return None 


def SFMS_fitting_comparison():    
    ''' Compare the different methods of fitting the SFMS on Santa Cruz 2, 
    Tinker Group Catalog, and NSA Dickey catalog. Tinker covers high mass data, 
    NSA covers low mass data, and Santa Cruz covers the whole range. 
    '''
    prettyplot()
    pretty_colors = prettycolors()
    fig = plt.figure(1, figsize=(24, 18))
    bkgd = fig.add_subplot(111, frameon=False)
    
    # read in catalogs
    Cat = Cats.Catalog()
    catalog_list = ['santacruz2', 'tinkergroup', 'nsa_dickey']
    catalog_labels = [] # Labels 
    logSFRs, logMstars, weights = [], [], [] 
    for cat in catalog_list: 
        catalog_labels.append(Cat.CatalogLabel(cat))
        logMstar, logSFR, weight = Cat.Read(cat)
        logSFRs.append(logSFR)
        logMstars.append(logMstar) 
        weights.append(weight)

    for i_cat in range(len(catalog_list)):  
        for i_fit in range(3): 
            sub = fig.add_subplot(len(catalog_list), 3, i_cat*3 + i_fit + 1)

            # plot SFR-Mstar contours 
            DFM.hist2d(logMstars[i_cat], logSFRs[i_cat], color='#1F77B4', ax=sub,
                    levels=[0.68, 0.95], range=[[6., 12.], [-4., 2.]], 
                    fill_contours=False) 

            if i_fit == 0: 
                # low Mass extrapolation 
                fit_label = r'$\mathtt{low \; M_* \;  extrap.}$'

                sfms_fit, med = Gprop.SFMS_bestfit(logSFRs[i_cat], logMstars[i_cat], 
                        method='lowMbin_extrap', forTest=True, fit_Mrange=[9., 10.]) 
                sub.scatter(med[0], med[1], c='k',marker='x', lw=3, s=40) 
                mm = np.arange(6., 12., 0.5)
                sub.plot(mm, sfms_fit(mm), c='k', lw=2, ls='--')
            elif i_fit == 1: 
                # Gaussian fit --> Linear fit with preliminary constant SSFR cut
                fit_label = '$\mathtt{constant\; logSSFR\; cut;}$\n $\mathtt{Gauss.\; fit}$'

                f_SFRcut = lambda mm: -11. + mm

                sfms_fit, med = Gprop.SFMS_bestfit(logSFRs[i_cat], logMstars[i_cat], 
                        method='SSFRcut_gaussfit_linearfit', forTest=True, 
                        fit_Mrange=[7., 11.], f_SFRcut=f_SFRcut) 
                
                sub.scatter(med[0], med[1], c='k',marker='x', lw=3, s=40) 
                mm = np.arange(6., 12.5, 0.5)
                sub.plot(mm, sfms_fit(mm), c='k', lw=2, ls='--', label='Best-fit')
                sub.plot(mm, f_SFRcut(mm), c='r', lw=2, ls='-.', label='SSFR cut')

            elif i_fit == 2: 
                # Gaussian fit --> Linear fit with preliminary M* depend SSFR cut
                fit_label = '$\mathtt{M_*\; dep.\; logSSFR\; cut;}$\n $\mathtt{Gauss.\; fit}$'

                f_SFRcut = lambda mm: -1. + 0.8 * (mm - 10.)

                sfms_fit, med = Gprop.SFMS_bestfit(logSFRs[i_cat], logMstars[i_cat], 
                        method='SSFRcut_gaussfit_linearfit', forTest=True, 
                        fit_Mrange=[7., 11.], f_SFRcut=f_SFRcut) 
                
                sub.scatter(med[0], med[1], c='k',marker='x', lw=3, s=40) 
                mm = np.arange(6., 12.5, 0.5)
                sub.plot(mm, sfms_fit(mm), c='k', lw=2, ls='--', label='Best-fit')
                sub.plot(mm, f_SFRcut(mm), c='r', lw=2, ls='-.', label='SSFR cut')

            if i_cat == 0: 
                sub.text(0.075, 0.925, fit_label,
                        ha='left', va='top', transform=sub.transAxes, fontsize=25)
            if i_cat == len(catalog_list)-1: 
                if i_fit == 2: 
                    sub.legend(loc='upper left', prop={'size':25})



    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_ylabel(r'$\mathtt{log \; M_* \;\;[M_\odot]}$', labelpad=20, fontsize=30) 
    bkgd.set_xlabel(r'$\mathtt{log \; SFR \;\;[M_\odot \, yr^{-1}]}$', labelpad=20, fontsize=30) 

    fig_name = ''.join([UT.fig_dir(), 'SFMS_fitting.comparison.png'])
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close() 
    return None


def SFR_Mstar_sfrtimescale_comparison(): 
    ''' Compare the SFR - M* relation of simulations with different timescale SFR.
    '''
    prettyplot()
    pretty_colors = prettycolors()
    fig = plt.figure(1, figsize=(16, 6))
    bkgd = fig.add_subplot(111, frameon=False)

    Cat = Cats.Catalog()

    # Read in santa cruz data sets
    sc_labels, sc_logSFRs, sc_logMstars, sc_weights = [], [], [], [] 
    for cat in ['santacruz1', 'santacruz2']: 
        sc_labels.append(Cat.CatalogLabel(cat))
        logMstar, logSFR, weight = Cat.Read(cat)
        sc_logSFRs.append(logSFR)
        sc_logMstars.append(logMstar) 
        sc_weights.append(weight)

    # Read in Illustris data sets
    ill_labels, ill_logSFRs, ill_logMstars, ill_weights = [], [], [], [] 
    for cat in ['illustris1', 'illustris2']: 
        ill_labels.append(Cat.CatalogLabel(cat))
        logMstar, logSFR, weight = Cat.Read(cat)
        ill_logSFRs.append(logSFR)
        ill_logMstars.append(logMstar) 
        ill_weights.append(weight)
    
    # plot Santa Cruz SFRs
    sub = fig.add_subplot(1, 2, 1)
    DFM.hist2d(sc_logMstars[0], sc_logSFRs[0], color='#1F77B4',  
            levels=[0.68, 0.95], range=[[6., 12.], [-4., 2.]], 
            plot_datapoints=False, fill_contours=True, ax=sub) 
    DFM.hist2d(sc_logMstars[1], sc_logSFRs[1], color='#FF7F0E', label=sc_labels[1],
            levels=[0.68, 0.95], range=[[6., 12.], [-4., 2.]],
            plot_datapoints=False, fill_contours=True, ax=sub) 

    thick_line1 = mlines.Line2D([], [], ls='-', c='#1F77B4', linewidth=12, alpha=0.5,
            label=sc_labels[0])
    thick_line2 = mlines.Line2D([], [], ls='-', c='#FF7F0E', linewidth=12, alpha=0.5,
            label=sc_labels[1])
    sub.legend(loc='upper left', handles=[thick_line1, thick_line2], frameon=False, fontsize=17)
    #sub.text(0.1, 0.9, catalog_labels[i_data],
    #        ha='left', va='center', transform=sub.transAxes, fontsize=17)
    
    sub = fig.add_subplot(1, 2, 2)
    DFM.hist2d(ill_logMstars[0], ill_logSFRs[0], color='#1F77B4', label=ill_labels[0], 
            levels=[0.68, 0.95], range=[[6., 12.], [-4., 2.]], 
            plot_datapoints=False, fill_contours=True, ax=sub) 
    DFM.hist2d(ill_logMstars[1], ill_logSFRs[1], color='#FF7F0E', label=ill_labels[1],
            levels=[0.68, 0.95], range=[[6., 12.], [-4., 2.]],
            plot_datapoints=False, fill_contours=True, ax=sub) 
    thick_line1 = mlines.Line2D([], [], ls='-', c='#1F77B4', linewidth=12, alpha=0.5,
            label=ill_labels[0])
    thick_line2 = mlines.Line2D([], [], ls='-', c='#FF7F0E', linewidth=12, alpha=0.5,
            label=ill_labels[1])
    sub.legend(loc='upper left', handles=[thick_line1, thick_line2], frameon=False, fontsize=17)

    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel(r'$\mathtt{log \; M_* \;\;[M_\odot]}$', labelpad=20, fontsize=30) 
    bkgd.set_ylabel(r'$\mathtt{log \; SFR \;\;[M_\odot \, yr^{-1}]}$', labelpad=20, fontsize=30) 
    
    fig_name = ''.join([UT.fig_dir(), 'SFR_Mstar.SFR_timescales.png'])
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close()
    return None 


### plot fQ using different definition of quenched 
# - SFMS fit constant offset
# - SFMS fit measured scatter offset 
# - constant SSFR cut
# - zero w.e. thingy



if __name__=='__main__': 
    #massbins = [[6., 6.5], [7., 7.5], [8., 8.5], [9., 9.5], [10., 10.5], [11., 11.5], [12., 12.5]]
    #Pssfr_Catalogs(logmstar_massbins=massbins, logsfr_nbin=25) 
    #SFR_Mstar_Catalogs(contour='dfm')
    #SFR_Mstar_Catalogs(contour='gaussianKDE')
    #SFR_Mstar_Catalogs(contour=False)
    #SFR_Mstar_sfrtimescale_comparison()
    #SFMS_fitting_comparison()
    #SFR_Mstar_SFMScut_QF(fit_method='lowMbin_extrap', scatter_method='constant', fit_Mrange=[9., 10.])
    SFR_Mstar_SFMScut_QF(fit_method='SSFRcut_gaussfit_linearfit', scatter_method='constant', 
            fit_Mrange=[7., 11.], f_SFRcut=lambda mm: -11.+ mm)

