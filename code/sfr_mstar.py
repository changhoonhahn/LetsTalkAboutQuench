import numpy as np 
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt 
from ChangTools.plotting import prettyplot

# -- Local --
import util as UT
import catalogs as Cats
import galpy.util.bovy_plot as bovy 


def sSFR_Mstar(logsfr, logmstar, weights=None, logmstar_min=8., logmstar_max=12., dlogmstar=0.5, logmstar_massbins=None, 
        logsfr_min=-13., logsfr_max=-8., logsfr_nbin=50, normed=True): 
    ''' SFR histogram in log M* bins 
    '''
    logsfr = logsfr - logmstar
    if len(logsfr) != len(logmstar): 
        raise ValueError("SFR and M* do not have the same length!")
    # the mass bins  
    if logmstar_massbins is None: 
        mass_bin_edges = np.arange(logmstar_min, logmstar_max + dlogmstar, dlogmstar)
        mass_bins = [] 
        for i in range(len(mass_bin_edges)-1): 
            mass_bins.append([mass_bin_edges[i], mass_bin_edges[i+1]]) 
    else: 
        mass_bins = logmstar_massbins
        
    logsfr_range = [logsfr_min, logsfr_max]
    
    PlogSFR, logSFR_bin_edges, counts, wtots = [], [], [], []
    # loop through the mass bins
    for i_m, mass_bin in enumerate(mass_bins): 
        mass_lim = np.where(
                (logmstar >= mass_bin[0]) & 
                (logmstar < mass_bin[1])
                )
        n_bin = len(mass_lim[0])

        # calculate the SFR distribution for mass bin 
        if weights is not None: 
            ws = weights[mass_lim]
            wtots.append(np.sum(ws))
        else: 
            ws = None

        dist, bin_edges = np.histogram(
                logsfr[mass_lim], 
                weights=ws, 
                range=logsfr_range, 
                bins=logsfr_nbin,  
                normed=normed)

        PlogSFR.append(dist)
        logSFR_bin_edges.append(bin_edges)
        #logSFR_bin_mid.append(0.5 * (bin_edges[:-1] + bin_edges[1:]))
        counts.append(n_bin)
    
    return [logSFR_bin_edges, PlogSFR, mass_bins, counts, wtots]#, logSFR_bin_edges]


def Plot_SFR_Mtar(logsfr, logmstar, file_name=None, yrange=None, **kwargs): 
    ''' plot SFR histogram in log M* bins 
    '''
    if file_name is None: 
        raise ValueError("specify figure file name") 
    
    logSFR_bin_mids, PlogSFRs, mass_bins, counts, wtots = sSFR_Mstar(logsfr, logmstar, **kwargs)
    n_mbins = len(PlogSFRs)
    
    # determine paneling 
    if n_mbins > 10: 
        panels = (3, int(np.ceil(np.float(n_mbins+1)/3.)))
    else: 
        panels = (2, int(np.ceil(np.float(n_mbins+1)/2.)))
    
    prettyplot()
    fig = plt.figure(1, figsize=(8*panels[1], 6*panels[0]))

    sub = fig.add_subplot(panels[0], panels[1], 1)
    sub.scatter(logmstar, logsfr, c='k', lw=0)
    sub.set_xlabel('log $M_*$')
    sub.set_xlim([6., 12.]) 
    sub.set_ylim([-4., 2.]) 
    sub.set_ylabel('log SFR')

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

    fig.savefig(file_name)
    plt.close()
    return None 


def plotSSFR_Mstar_Comparison(xrange=[-12.5, -9.], yrange=[0., 1.6], **kwargs): 
    ''' compare SFR/sSFR histograms of various simlations and data in log M* bins 
    '''
    Cat = Cats.Catalog()
    # Read in various data sets
    catalog_list = ['santacruz1', 'santacruz2', 'tinkergroup', 'illustris1', 'illustris2', 'nsa_dickey', 'mufasa']

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
                sSFR_Mstar(logSFRs[i_data], logMstars[i_data], weights=weights[i_data], **kwargs)
        n_mbins = len(PlogSFRs)
        
        # determine paneling 
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
            sub.plot(xx_bar, yy_bar, lw=5, label=catalog_labels[i_data], alpha=0.75)

            sub.text(0.95, 0.9, str(round(mass_bins[i_m][0],2))+'-'+str(round(mass_bins[i_m][1],2)), 
                    fontsize=30, ha='right', va='center', transform=sub.transAxes)
            if xrange is not None: 
                sub.set_xlim(xrange)#[-12.5, -9.])
            if yrange is not None: 
                sub.set_ylim(yrange)
            if i_m == 0:  
                sub.legend(loc='upper left', prop={'size':17})

    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_ylabel(r'$\mathtt{P(SSFR)}$', labelpad=20, fontsize=30) 
    bkgd.set_xlabel(r'$\mathtt{log \; sSFR \;\;[yr^{-1}]}$', labelpad=20, fontsize=30) 
    
    fig_name = ''.join([UT.fig_dir(), 'Pssfr_comp', '.', str(n_mbins), 'Mbins', '.png'])
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close()
    return None 


def plotSFR_Mstar_Comparison(): 
    ''' Compare SFR vs M* relation of various simlations and data
    '''
    Cat = Cats.Catalog()
    # Read in various data sets
    catalog_list = ['santacruz1', 'santacruz2', 'tinkergroup', 'illustris1', 'illustris2', 'nsa_dickey', 'mufasa']

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
        #sub.scatter(logMstars[i_data], logSFRs[i_data], c='k', s=2, lw=0)

        bovy.scatterplot(logMstars[i_data], logSFRs[i_data], #weights=weights[i_data],
                cmap='Blues', levels=[0.68, 0.95], aspect=0.75, gcf=True, scatter=True, s=3, xrange=[6., 12.], yrange=[-4., 2.])
        #xx, yy, f = UT.gaussianKDE_contour(logMstars[i_data], logSFRs[i_data], 
        #        xmin=6., xmax=12., ymin=-4., ymax=2.)
        #sub.contourf(xx, yy, f, cmap='Blues')
        #sub.contour(xx, yy, f, colors='k')
        #sub.set_xlim([6., 12.]) 
        #sub.set_ylim([-4., 2.]) 
        sub.text(0.1, 0.9, catalog_labels[i_data],
                ha='left', va='center', transform=sub.transAxes, fontsize=17)

    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_ylabel(r'$\mathtt{log \; M_* \;\;[M_\odot]}$', labelpad=20, fontsize=30) 
    bkgd.set_xlabel(r'$\mathtt{log \; SFR \;\;[M_\odot \, yr^{-1}]}$', labelpad=20, fontsize=30) 
    
    fig.subplots_adjust(wspace=0.15, hspace=0.15)

    fig_name = ''.join([UT.fig_dir(), 'sfr_Mstar_comp.png'])
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close()
    return None 


def SFMS_bestfit(logSFR, logMstar, method='lowMbin_extrap', forTest=False): 
    ''' calculate the best fit of the SFMS for a given sample of log SFR and log M* 
    returns 
    
    log SFR_sfms(M*) best fit
    '''
    if method == 'lowMbin_extrap': 
        # fit the SFMS with median SFR in mass range where f_Q ~ 1. 
        # then assume the slope is the same at higher masses 
        # motivated by Bluck et al. 2016
        
        fit_Mrange = [9., 10.]  # hardcoded for now 
        Mfid = np.mean(fit_Mrange)  # fiducial M* 
        dlogM = 0.2 # dex

        Mass_cut = np.where((logMstar >= fit_Mrange[0]) & (logMstar < fit_Mrange[1])) 

        if len(Mass_cut[0]) == 0: 
            raise ValueError("No galaxies within the fitting stellar mass range") 
        
        logMstar_fit = logMstar[Mass_cut]
        logSFR_fit = logSFR[Mass_cut]

        # calculate the median SFRs within the fitting M* bin 
        logM_bins = np.arange(fit_Mrange[0], fit_Mrange[1]+dlogM, dlogM) 

        med_Mstar, med_SFR = [], [] 
        for i in range(len(logM_bins)-1):  
            in_mbin = np.where((logMstar_fit >= logM_bins[i]) & (logMstar_fit < logM_bins[i+1]) & (np.isnan(logSFR_fit) == False))
            
            if len(in_mbin[0]) > 0: 
                med_Mstar.append(np.median(logMstar_fit[in_mbin]))
                med_SFR.append(np.median(logSFR_fit[in_mbin]))

        # now least square fit line to the values
        xx = np.array(med_Mstar) - Mfid  # log Mstar - log M_fid
        yy = np.array(med_SFR)
        A = np.vstack([xx, np.ones(len(xx))]).T
        m, c = np.linalg.lstsq(A, yy)[0] 

        sfms_fit = lambda mm: m * (mm - Mfid) + c
        
        if forTest: 
            return sfms_fit, [med_Mstar, med_SFR]
        else: 
            return sfms_fit 

    elif method == 'SSFRcut_gaussfit_linearfit': 
        # fit P(SSFR) distribution above some hard SSFR cut with a Gaussian 
        # then fit the mus you get from the Gaussian with a linear fit 
        fit_Mrange = [logMstar.min(), 11.0]  # hardcoded for now 
        SSFR_cut = -11.

        Mfid = 10.  # fiducial M* hardcoded at 10 (no good reason)
        dlogM = 0.5 # dex

        Mass_cut = np.where((logMstar >= fit_Mrange[0]) & (logMstar < fit_Mrange[1])) 

        if len(Mass_cut[0]) == 0: 
            raise ValueError("No galaxies within the fitting stellar mass range") 
        
        logMstar_fit = logMstar[Mass_cut]
        logSFR_fit = logSFR[Mass_cut]

        logM_bins = np.arange(fit_Mrange[0], fit_Mrange[1]+dlogM, dlogM) 

        fit_Mstar, fit_SFR = [], [] 
        for i in range(len(logM_bins)-1):  
            SFR_cut = SSFR_cut + 0.5 * (logM_bins[i] + logM_bins[i+1])
            in_mbin = np.where((logMstar_fit >= logM_bins[i]) & (logMstar_fit < logM_bins[i+1]) & (logSFR_fit > SFR_cut))
            
            if len(in_mbin[0]) > 20: 
                yy, xx_edges = np.histogram(logSFR_fit[in_mbin], bins=10, normed=True)
                xx = 0.5 * (xx_edges[1:] + xx_edges[:-1])

                gaus = lambda xx, aa, x0, sig: aa * np.exp(-(xx - x0)**2/(2*sig**2))
                
                print logM_bins[i], ' - ', logM_bins[i+1]
                print -10. + 0.5 * (logM_bins[i] + logM_bins[i+1])

                try: 
                    popt, pcov = curve_fit(gaus, xx, yy, p0=[1., np.median(logSFR_fit[in_mbin]), 0.3])
                except RuntimeError: 
                    fig = plt.figure(2)
                    plt.scatter(xx, yy)
                    plt.show()
                    plt.close()
                    raise ValueError

                fit_Mstar.append(np.median(logMstar_fit[in_mbin]))
                fit_SFR.append(popt[1])
        
        # now least square fit line to the values
        xx = np.array(fit_Mstar) - Mfid  # log Mstar - log M_fid
        yy = np.array(fit_SFR)
        A = np.vstack([xx, np.ones(len(xx))]).T
        m, c = np.linalg.lstsq(A, yy)[0] 
        
        sfms_fit = lambda mm: m * (mm - Mfid) + c

        if forTest: 
            return sfms_fit, [fit_Mstar, fit_SFR]
        else: 
            return sfms_fit 

    elif method == 'SSFRcut_gaussfit_kinkedlinearfit': 
        # fit P(SSFR) distribution above some hard SSFR cut with a Gaussian 
        # then fit the mus you get from the Gaussian witha kinked linear fit 
        fit_Mrange = [logMstar.min(), 11.0]  # hardcoded for now 
        SSFR_cut = -11.

        Mkink = 9.5  # fiducial M* hardcoded at 10 (no good reason)
        dlogM = 0.5 # dex

        Mass_cut = np.where((logMstar >= fit_Mrange[0]) & (logMstar < fit_Mrange[1])) 

        if len(Mass_cut[0]) == 0: 
            raise ValueError("No galaxies within the fitting stellar mass range") 
        
        logMstar_fit = logMstar[Mass_cut]
        logSFR_fit = logSFR[Mass_cut]

        logM_bins = np.arange(fit_Mrange[0], fit_Mrange[1]+dlogM, dlogM) 

        fit_Mstar, fit_SFR = [], [] 
        for i in range(len(logM_bins)-1):  
            SFR_cut = SSFR_cut + 0.5 * (logM_bins[i] + logM_bins[i+1])
            in_mbin = np.where((logMstar_fit >= logM_bins[i]) & (logMstar_fit < logM_bins[i+1]) & (logSFR_fit > SFR_cut))
            
            if len(in_mbin[0]) > 20: 
                yy, xx_edges = np.histogram(logSFR_fit[in_mbin], bins=10, normed=True)
                xx = 0.5 * (xx_edges[1:] + xx_edges[:-1])

                gaus = lambda xx, aa, x0, sig: aa * np.exp(-(xx - x0)**2/(2*sig**2))
                
                try: 
                    popt, pcov = curve_fit(gaus, xx, yy, p0=[1., np.median(logSFR_fit[in_mbin]), 0.3])
                except RuntimeError: 
                    fig = plt.figure(2)
                    plt.scatter(xx, yy)
                    plt.show()
                    plt.close()
                    raise ValueError

                fit_Mstar.append(np.median(logMstar_fit[in_mbin]))
                fit_SFR.append(popt[1])
        
        def kinked(mm, m1, m2, c): 
            sfr_out = np.zeros(len(mm))
            highM = np.where(mm > Mkink)
            if len(highM[0]) > 0: 
                sfr_out[highM] = m1 * (mm[highM] - Mkink) + c

            lowM = np.where(mm <= Mkink)
            if len(lowM[0]) > 0:  
                sfr_out[lowM] = m2 * (mm[lowM] - Mkink) + c
            return sfr_out                    
        # now least square fit line to the values
        xx = np.array(fit_Mstar)  # log Mstar
        yy = np.array(fit_SFR)

        popt, pcov = curve_fit(kinked, xx, yy, p0=[1., 1., -1.])
        
        def sfms_fit(mm): 
            sfr_out = np.zeros(len(mm))
            highM = np.where(mm > Mkink)
            if len(highM[0]) > 0: 
                sfr_out[highM] = popt[0] * (mm[highM] - Mkink) + popt[2] 
            lowM = np.where(mm <= Mkink)
            if len(lowM[0]) > 0:  
                sfr_out[lowM] = popt[1] * (mm[lowM] - Mkink) + popt[2] 
            return sfr_out

        if forTest: 
            return sfms_fit, [fit_Mstar, fit_SFR]
        else: 
            return sfms_fit 


def SFMS_fit(logSFR, logMstar, method='lowMbin_extrap'): 
    ''' calculate linear fit for the SFMS of a given sample of log SFR and log M* 
    returns [m, logMfid, c] where linear SFMS is parameterized as 

    log SFR_SFMS = m * (log M* - log Mfid) + c
    '''
    if method == 'lowMbin_extrap': 
        # fit the SFMS with median SFR in mass range where f_Q ~ 1. 
        # then assume the slope is the same at higher masses 
        # motivated by Bluck et al. 2016
        
        fit_Mrange = [9., 10.]  # hardcoded for now 
        Mfid = np.mean(fit_Mrange)  # fiducial M* 
        dlogM = 0.2 # dex

        Mass_cut = np.where((logMstar >= fit_Mrange[0]) & (logMstar < fit_Mrange[1])) 

        if len(Mass_cut[0]) == 0: 
            raise ValueError("No galaxies within the fitting stellar mass range") 
        
        logMstar_fit = logMstar[Mass_cut]
        logSFR_fit = logSFR[Mass_cut]

        # calculate the median SFRs within the fitting M* bin 
        logM_bins = np.arange(fit_Mrange[0], fit_Mrange[1]+dlogM, dlogM) 

        med_Mstar, med_SFR = [], [] 
        for i in range(len(logM_bins)-1):  
            in_mbin = np.where((logMstar_fit >= logM_bins[i]) & (logMstar_fit < logM_bins[i+1]) & (np.isnan(logSFR_fit) == False))
            
            if len(in_mbin[0]) > 0: 
                med_Mstar.append(np.median(logMstar_fit[in_mbin]))
                med_SFR.append(np.median(logSFR_fit[in_mbin]))

        # now least square fit line to the values
        xx = np.array(med_Mstar) - Mfid  # log Mstar - log M_fid
        yy = np.array(med_SFR)
        A = np.vstack([xx, np.ones(len(xx))]).T
        m, c = np.linalg.lstsq(A, yy)[0] 
        return [m, Mfid, c]

    elif method == 'SSFRcut_gaussfit_linearfit': 
        # fit P(SSFR) distribution above some hard SSFR cut
        fit_Mrange = [logMstar.min(), 11.0]  # hardcoded for now 
        SSFR_cut = -11.

        Mfid = 10.  # fiducial M* hardcoded at 10 (no good reason)
        dlogM = 0.5 # dex

        Mass_cut = np.where((logMstar >= fit_Mrange[0]) & (logMstar < fit_Mrange[1])) 

        if len(Mass_cut[0]) == 0: 
            raise ValueError("No galaxies within the fitting stellar mass range") 
        
        logMstar_fit = logMstar[Mass_cut]
        logSFR_fit = logSFR[Mass_cut]

        logM_bins = np.arange(fit_Mrange[0], fit_Mrange[1]+dlogM, dlogM) 

        fit_Mstar, fit_SFR = [], [] 
        for i in range(len(logM_bins)-1):  
            SFR_cut = SSFR_cut + 0.5 * (logM_bins[i] + logM_bins[i+1])
            in_mbin = np.where((logMstar_fit >= logM_bins[i]) & (logMstar_fit < logM_bins[i+1]) & (logSFR_fit > SFR_cut))
            
            if len(in_mbin[0]) > 20: 
                yy, xx_edges = np.histogram(logSFR_fit[in_mbin], bins=10, normed=True)
                xx = 0.5 * (xx_edges[1:] + xx_edges[:-1])

                gaus = lambda xx, aa, x0, sig: aa * np.exp(-(xx - x0)**2/(2*sig**2))
                
                print logM_bins[i], ' - ', logM_bins[i+1]
                print -10. + 0.5 * (logM_bins[i] + logM_bins[i+1])

                try: 
                    popt, pcov = curve_fit(gaus, xx, yy, p0=[1., np.median(logSFR_fit[in_mbin]), 0.3])
                except RuntimeError: 
                    fig = plt.figure(2)
                    plt.scatter(xx, yy)
                    plt.show()
                    plt.close()
                    raise ValueError

                fit_Mstar.append(np.median(logMstar_fit[in_mbin]))
                fit_SFR.append(popt[1])
        
        # now least square fit line to the values
        xx = np.array(fit_Mstar) - Mfid  # log Mstar - log M_fid
        yy = np.array(fit_SFR)
        A = np.vstack([xx, np.ones(len(xx))]).T
        m, c = np.linalg.lstsq(A, yy)[0] 

        return [m, Mfid, c]


def SFMS_scatter(logSFR, logMstar, fit_method='lowMbin_extrap', method='constant'): 
    ''' calculate the scatter in the SFMS using different methods 

    returns [log M*, log SFR_SFMS, sigma_log SFR_SFMS]
    '''
    # best fit SFMS 
    f_sfms = SFMS_bestfit(logSFR, logMstar, method=fit_method)
    
    m_arr = np.arange(6., 12.5, 0.5)
    
    if method == 'constant': 
        # observations find ~0.3 dex scatter in SFMS so just return that 
        sig_SFR = np.repeat(0.3, len(m_arr))

        return [m_arr, f_sfms(m_arr), sig_SFR]

    elif method == 'half_gauss': 
        # fit the scatter of the SFMS by 
        # assuming SFMS is a log-normal distirbution centered at the SFR 
        # determined from the SFMS fit. Then, fit the scatter of SFMS SFRs 
        # of galaxies with SFRs *above* the SFMS fit to avoid 
        # contamination from quenching galaxies
        sfr_sfms_interp = f_sfms(logMstar) 

        m_low = m_arr - 0.25 
        m_high = m_arr + 0.25
        
        sig_SFR = np.zeros(len(m_arr)) 
        for i_m in range(len(m_arr)): 
            in_mbin = np.where(
                    (logMstar >= m_low[i_m]) & 
                    (logMstar < m_high[i_m]) & 
                    (logSFR > sfr_sfms_interp)) 
            
            if len(in_mbin[0]) > 20: 
                dlogSFR = logSFR[in_mbin] - sfr_sfms_interp[in_mbin]

                yy, xx_edges = np.histogram(dlogSFR, bins=10)
                xx = 0.5 * (xx_edges[1:] + xx_edges[:-1])

                gaus = lambda xx, aa, sig: aa * np.exp(-xx**2/(2*sig**2))

                popt, pcov = curve_fit(gaus, xx, yy, p0=[1., 0.3])
                
                sig_SFR[i_m] = np.abs(popt[1])

        return [m_arr, f_sfms(m_arr), sig_SFR]


def plotSFR_Mstar_SFMSfits(fit_method='lowMbin_extrap'): 
    ''' compare SFR vs M* relation of various simlations and data
    '''
    # Read in various data sets
    logSFRs, logMstars, data_labels = [], [], [] 
    ## santa cruz
    sc_logM, sc_logSFR1, sc_logSFR2, sc_w = np.loadtxt("sc_sam_sfr_mstar_correctedweights.txt", unpack=True, skiprows=1) # logM*, logSFR (10^5yr), logSFR(10^8yr)
    logSFRs.append(sc_logSFR1)
    logMstars.append(sc_logM)
    data_labels.append('SantaCruz')
    ## Tinker group catalog 
    tink_Mstar, tink_logSSFR = np.loadtxt('tinker_SDSS_centrals_M9.7.dat', unpack=True, skiprows=2, usecols=[0, 7])
    tink_logM = np.log10(tink_Mstar)
    tink_logSFR = tink_logSSFR + tink_logM
    logSFRs.append(tink_logSFR)
    logMstars.append(tink_logM)
    data_labels.append('Tinker')
    ## illustris
    ill_logMstar, ill_ssfr1, ill_ssfr2 = np.loadtxt('Illustris1_SFR_M_values.csv', unpack=True, skiprows=1, delimiter=',')
    ill_logSFR1 = np.log10(ill_ssfr1) + ill_logMstar
    ill_logSFR2 = np.log10(ill_ssfr2) + ill_logMstar
    logSFRs.append(ill_logSFR1)
    logMstars.append(ill_logMstar)
    data_labels.append('Illustris')
    # NSA 
    dic_logM, dic_logSFR =  np.loadtxt('dickey_NSA_iso_lowmass_gals.txt', unpack=True, skiprows=1, usecols=[1,5]) 
    logSFRs.append(dic_logSFR)
    logMstars.append(dic_logM)
    data_labels.append('NSA')
    # MUFASA
    muf_M, muf_SSFR =  np.loadtxt('Mufasa_m50n512_z0.cat', unpack=True, skiprows=1, usecols=[0, 1]) 
    muf_logM = np.log10(muf_M)
    muf_logSSFR = np.log10(muf_SSFR)
    muf_logSFR = muf_logSSFR + muf_logM
    logSFRs.append(muf_logSFR)
    logMstars.append(muf_logM)
    data_labels.append('MUFASA')
        
    prettyplot()
        
    fig = plt.figure(1, figsize=(8*(len(logSFRs)), 6))

    qfs = [] 
    for i_data in range(len(logSFRs)):
        sub = fig.add_subplot(1, len(logSFRs), i_data+1)
        sub.scatter(logMstars[i_data], logSFRs[i_data], c='k', lw=0, s=5)
        
        # plot best-fit SFMS 
        sfms_bestfit, fit_data = SFMS_bestfit(logSFRs[i_data], logMstars[i_data], method=fit_method, forTest=True)
        m_arr = np.arange(6., 12.1, 0.1)
        sub.scatter(fit_data[0], fit_data[1], c='r', lw=3, marker='x', s=30)
        sub.plot(m_arr, sfms_bestfit(m_arr), c='b', lw=3, ls='--', label='SFMS')
        sub.set_xlim([6., 12.]) 
        sub.set_ylim([-4., 2.]) 
        if i_data == 0: 
            sub.set_ylabel('log SFR')
        sub.set_xlabel('log $M_*$')
        sub.text(0.3, 0.9, data_labels[i_data],
                ha='center', va='center', transform=sub.transAxes)

        if i_data == 1: 
            sub.legend(loc='lower right') 

    fig_name = ''.join(['SFMS_bestfit_', fit_method, '.png'])
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close()
    return None 


def plotSFR_Mstar_SFMScut_QF(fit_method='lowMbin_extrap', scatter_method='constant'): 
    ''' compare SFR vs M* relation of various simlations and data
    '''
    Cat = Cats.Catalog()
    # Read in various data sets
    catalog_list = ['santacruz1', 'santacruz2', 'tinkergroup', 'illustris1', 'illustris2', 'nsa_dickey', 'mufasa']

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
        
        bovy.scatterplot(logMstars[i_data], logSFRs[i_data], #weights=weights[i_data],
                levels=[0.68, 0.95], aspect=0.75, gcf=True, scatter=True,
                xrange=[6., 12.], yrange=[-4., 2.], overplot=True)
        
        # plot best-fit SFMS 
        m_arr, sfr_sfms, sig_sfr = SFMS_scatter(logSFRs[i_data], logMstars[i_data], 
                fit_method=fit_method, method=scatter_method)
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
        sub.plot(qfs[i_qf][0], qfs[i_qf][1], label=catalog_labels[i_qf], lw=3)
    
    sub.set_xlim([6., 12.]) 
    sub.set_ylim([0., 1.]) 
    sub.set_ylabel('$\mathtt{f_{Q}}$', fontsize=20)
    sub.yaxis.tick_right()
    sub.yaxis.set_label_position("right")
    sub.legend(loc='upper left', prop={'size':17}) 

    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_ylabel(r'$\mathtt{log \;M_* \;\;[M_\odot]}$', labelpad=20, fontsize=30) 
    bkgd.set_xlabel(r'$\mathtt{log \;SFR \;\;[M_\odot/yr^{-1}]}$', labelpad=20, fontsize=30) 

    fig.subplots_adjust(wspace=0.15, hspace=0.15)
    fig_name = ''.join([UT.fig_dir(), 'sfr_Mstar', 
        '.SFMSfit_', fit_method, 
        '.scatter_', scatter_method, 
        '.png'])
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close()
    return None 


def plot_Dquench_Mstar(logmstar_massbins=None, fit_method='lowMbin_extrap'): 
    ''' Plot the D_quench distribution as a function of M*. We define
    D_quench as the "distance" from the "center" of the SFMS. 
    '''
    if logmstar_massbins is None: 
        raise ValueError

    Cat = Cats.Catalog()
    # Read in various data sets
    catalog_list = ['santacruz1', 'santacruz2', 'tinkergroup', 'illustris1', 'illustris2', 'nsa_dickey', 'mufasa']

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
    qfs = [] 
    for i_data in range(len(logSFRs)):
        sub = fig.add_subplot(n_rows, 4, i_data+1)
        # best-fit SFMS 
        sfr_sfms = SFMS_bestfit(logSFRs[i_data], logMstars[i_data], method=fit_method)

        # calculate d_Quench
        d_Q = logSFRs[i_data] - sfr_sfms(logMstars[i_data])
        
        for i_m in range(len(logmstar_massbins)): 
            in_mbin = np.where(
                    (logMstars[i_data] >= logmstar_massbins[i_m][0]) & 
                    (logMstars[i_data] < logmstar_massbins[i_m][1]) & 
                    (np.isnan(logSFRs[i_data]) == False))  
        
            # histogram of d_Q 
            P_d_Q, bin_edges = np.histogram(
                    d_Q[in_mbin],
                    range=[-4., 2.], 
                    bins=30,  
                    normed=False)
            xx_bar, yy_bar = UT.bar_plot(bin_edges, P_d_Q)
            sub.plot(xx_bar, yy_bar, lw=5, alpha=0.75,
                    label=str(logmstar_massbins[i_m][0])+'-'+str(logmstar_massbins[i_m][1]))
    
        sub.set_xlim([-3, 2]) 
        if i_data == len(logSFRs)-1: 
            sub.legend(loc='upper left', prop={'size':17})
        
        sub.text(0.95, 0.9, catalog_labels[i_data],
                ha='right', va='center', transform=sub.transAxes, fontsize=20)

    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_ylabel(r'$\mathtt{P(d_{MS})}$', labelpad=20, fontsize=30) 
    bkgd.set_xlabel(r'$\mathtt{d_{MS} \;\;[dex]}$', labelpad=20, fontsize=30) 

    fig.subplots_adjust(wspace=0.15, hspace=0.15)
    fig_name = ''.join([UT.fig_dir(), 'dMS_Mstar', '.SFMSfit_', fit_method, '.png'])
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close()
    return None 


if __name__=='__main__': 
    #brooks = np.loadtxt('brooks_mstar_sfr.dat', unpack=True, skiprows=1) 
    #print np.log10(brooks[0]).min(), np.log10(brooks[0]).max(), len(brooks[0])
    #Plot_SFR_Mtar(np.log10(brooks[2]), np.log10(brooks[0]), file_name='testing.png', logmstar_min=6.0, logmstar_max=10., dlogmstar=1., normed=False)
    #plotSFR_Mstar_Comparison()
    massbins = [[6., 6.5], [7., 7.5], [8., 8.5], [9., 9.5], [10., 10.5], [11., 11.5], [12., 12.5]]
    #plotSSFR_Mstar_Comparison(logmstar_massbins=massbins, logsfr_nbin=20, xrange=[-13., -8])
    #plotSFR_Mstar_SFMSfits(fit_method='lowMbin_extrap')
    #plotSFR_Mstar_SFMSfits(fit_method='SSFRcut_gaussfit_linearfit')
    #plotSFR_Mstar_SFMSfits(fit_method='SSFRcut_gaussfit_kinkedlinearfit')
    
    for fit in ['lowMbin_extrap', 'SSFRcut_gaussfit_linearfit', 'SSFRcut_gaussfit_kinkedlinearfit']: 
        #plot_Dquench_Mstar(logmstar_massbins=massbins, fit_method=fit)
        for scat in ['constant', 'half_gauss']: 
            plotSFR_Mstar_SFMScut_QF(fit_method=fit, scatter_method=scat)
