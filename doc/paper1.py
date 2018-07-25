'''

Plots for Paper I of quenched galaxies series 


Author(s): ChangHoon Hahn 


'''
import pickle
import numpy as np 
import scipy as sp
import corner as DFM 
from scipy import linalg
from scipy.stats import multivariate_normal as MNorm
from sklearn.mixture import GaussianMixture as GMix

from matplotlib import lines as mlines
from matplotlib.patches import Rectangle
#import matplotlib.patheffects as path_effects

# -- Local --
from letstalkaboutquench import util as UT
from letstalkaboutquench import catalogs as Cats
from letstalkaboutquench import galprop as Gprop
from letstalkaboutquench.fstarforms import fstarforms
from letstalkaboutquench.fstarforms import sfr_mstar_gmm

from ChangTools.plotting import prettycolors
import matplotlib as mpl
import matplotlib.pyplot as plt 
from matplotlib.font_manager import FontProperties
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


illustris_mmin = 8.4
eagle_mmin = 8.4
mufasa_mmin = 9.2
scsam_mmin = 8.8
tinker_mmin = 9.7
dickey_mmax = 9.7


def Catalogs_SFR_Mstar(): 
    ''' Compare SFR vs M* relation of central galaxies from various simlations and 
    observations 
    '''
    Cat = Cats.Catalog()
    tscales = ['inst', '100myr']
    sims_list = ['illustris', 'eagle', 'mufasa', 'scsam'] 
    obvs_list = ['tinkergroup', 'nsa_dickey']
    
    plot_range = [[7.8, 12.], [-4., 2.]]

    fig = plt.figure(figsize=(20,8))
    gs = mpl.gridspec.GridSpec(1,5, figure=fig) 

    # plot SFR-M* for the observations 
    gs_i = mpl.gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[4])
    sub0 = plt.subplot(gs_i[1:3,:])
    for i_c, cat in enumerate(obvs_list): 
        logMstar, logSFR, weight, censat = Cat.Read(cat)
        iscen = (censat == 1)
        DFM.hist2d(logMstar[iscen], logSFR[iscen], color='C'+str(i_c), 
                levels=[0.68, 0.95], range=plot_range, 
                plot_datapoints=True, fill_contours=False, plot_density=True, 
                ax=sub0) 
        sub0.set_xlim([7.8, 11.8]) 
        sub0.set_xticks([8., 9., 10., 11.]) 
        sub0.set_ylim([-3.5, 2.]) 
        sub0.set_yticks([-3, -2., -1., 0., 1, 2.]) 

    sub0.text(0.95, 0.05, 'SDSS Centrals', ha='right', va='bottom', 
                transform=sub0.transAxes, fontsize=20)

    for i_c, cc in enumerate(sims_list): 
        gs_i = mpl.gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[i_c])
        for i_t, tscale in enumerate(tscales): 
            cat = '_'.join([cc, tscale]) 
            sub = plt.subplot(gs_i[i_t,0]) 
            lbl = Cat.CatalogLabel(cat)
            logMstar, logSFR, weight, censat = Cat.Read(cat)

            if i_c == 0: 
                sub.text(0.05, 0.95, 'SFR ['+(lbl.split('[')[-1]).split(']')[0]+']', 
                        ha='left', va='top', transform=sub.transAxes, fontsize=20)
            if i_t == 1: 
                sub.text(0.95, 0.05, lbl.split('[')[0], ha='right', va='bottom', 
                        transform=sub.transAxes, fontsize=20)

            # only pure central galaxies identified from the group catalog
            psat = Cat.GroupFinder(cat)
            iscen = (psat < 0.01) 
            #iscen = (censat == 1)

            DFM.hist2d(logMstar[iscen], logSFR[iscen], color='C'+str(i_c+2), 
                    levels=[0.68, 0.95], range=plot_range, 
                    plot_datapoints=True, fill_contours=False, plot_density=True, ax=sub) 
            sub.set_xlim([7.8, 11.8]) 
            sub.set_xticks([8., 9., 10., 11.]) 
            if i_t == 0: sub.set_xticklabels([]) 
            sub.set_ylim([-3.5, 2.]) 
            sub.set_yticks([-3, -2., -1., 0., 1, 2.]) 
            #if i_c != 0: sub.set_yticklabels([]) 

    fig.text(0.5, 0.01, r'log$\; M_* \;\;[M_\odot]$', ha='center', fontsize=30) 
    fig.text(0.08, 0.5, r'log ( SFR $[M_\odot \, yr^{-1}]$ )', 
            rotation='vertical', va='center', fontsize=30) 
    fig.subplots_adjust(wspace=0.15, hspace=0.05)
    fig_name = ''.join([UT.doc_dir(), 'figs/Catalogs_SFR_Mstar.pdf'])
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
            psat = Cat.GroupFinder(cat+'_'+tscale)
            iscen_nz = ((psat < 0.01) & ~Cat.zero_sfr) #iscen = ((censat == 1) & np.invert(Cat.zero_sfr)) 
            iscen_z = ((psat < 0.01) & Cat.zero_sfr)
            inmbin_nz = (iscen_nz & (logMstar > mbin[0]) & (logMstar < mbin[1]))   
            inmbin_z = (iscen_z & (logMstar > mbin[0]) & (logMstar < mbin[1]))   

            ssfr_i = np.concatenate([logSFR[inmbin_nz] - logMstar[inmbin_nz], 
                np.tile(-13.2, np.sum(inmbin_z))])
            _ = sub.hist(ssfr_i, bins=40, 
                    range=[-14., -8.], 
                    normed=True, histtype='step', color='C'+str(i_c+2), linewidth=1.75, 
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

            sub.legend(loc='lower left', bbox_to_anchor=(0.01, 0.25), 
                    frameon=False, prop={'size': 15})
    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel('log$(\; \mathrm{SSFR}\; [\mathrm{yr}^{-1}]\;)$', labelpad=5, fontsize=25) 
    bkgd.set_ylabel('$p\,(\;\mathrm{log}\; \mathrm{SSFR}\; [\mathrm{yr}^{-1}]\;)$', 
            labelpad=5, fontsize=25)
    fig.subplots_adjust(wspace=0.15)
    fig_name = ''.join([UT.doc_dir(), 'figs/Catalogs_pSSFR.pdf'])
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
        logM, _, _, censat = Cata.Read(name, keepzeros=True, silent=True) 
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

        purity_threshold = 0.8 # purity threshold 

        notpure = (np.array(fp_pc) < 0.8) 
        if np.sum(notpure) > 0:  
            print('-- %s Catalog -- ' % name) 
            print('M_* < %f ' % (np.array(mmids)[notpure].max()+0.5*(mbin[1]-mbin[0])))
        if 'scsam' in name: 
            print('%f' % np.mean(np.array(fp_pc)[notpure]))
            print('%f' % np.mean(np.array(fp_pc)[np.invert(notpure)]))
            print('%f' % (float(len(logM[logM < 8.5]))/float(len(logM))))

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
    fig.savefig(''.join([UT.doc_dir(), 'figs/groupfinder.pdf']), bbox_inches='tight') 
    plt.close() 
    return None 


def Catalog_SFMS_fit(tscale, nosplashback=False, sb_cut='3vir'): 
    ''' Compare the GMM fits to the SFMS 
    '''
    if tscale not in ['inst', '10myr', '100myr', '1gyr']: raise ValueError
    
    Cat = Cats.Catalog()
    # Read in various data sets
    sims_list = ['illustris', 'eagle', 'mufasa', 'scsam'] 
    obvs_list = ['tinkergroup', 'nsa_dickey'] 

    # file with best-fit GMM 
    f_gmm = lambda name: _fGMM(name, nosplashback=nosplashback, sb_cut=sb_cut)

    fig = plt.figure(1, figsize=(12,8))
    bkgd = fig.add_subplot(111, frameon=False)
    plot_range = [[8., 12.], [-4., 2.]]

    # plot SFR-M* for the observations 
    sub0 = fig.add_subplot(233)
    for i_c, cat in enumerate(obvs_list): 
        logMstar, logSFR, weight, censat = Cat.Read(cat)
        if cat == 'nsa_dickey': 
            sample_cut = ((censat == 1) & ~Cat.zero_sfr & (logMstar < dickey_mmax)) 
        elif cat == 'tinkergroup': 
            sample_cut = ((censat == 1) & ~Cat.zero_sfr & (logMstar > tinker_mmin)) 

        DFM.hist2d(logMstar[sample_cut], logSFR[sample_cut], 
                color='C'+str(i_c), levels=[0.68, 0.95], range=plot_range, 
                plot_datapoints=True, fill_contours=False, plot_density=True, alpha=0.5, 
                ax=sub0) 
        
        fSFS = pickle.load(open(f_gmm(cat), 'rb'))
        sub0.errorbar(fSFS._fit_logm, fSFS._fit_logsfr, fSFS._fit_err_logssfr, fmt='.k')
    
    sub0.text(0.9, 0.1, 'SDSS Centrals', ha='right', va='center', 
                transform=sub0.transAxes, fontsize=20)
    sub0.set_xlim([8.0, 12.]) 
    sub0.set_ylim([-4., 2.]) 
    sub0.set_xticks([8., 9., 10., 11., 12.]) 
    sub0.set_xticklabels([]) 
    sub0.set_yticklabels([]) 

    for i_c, cc in enumerate(sims_list): 
        cat = '_'.join([cc, tscale]) 
        sub = fig.add_subplot(2,3,3*(i_c/2)+(i_c % 2)+1) 

        logMstar, logSFR, weight, censat = Cat.Read(cat)
        psat = Cat.GroupFinder(cat) # group finder 
        sample_cut = ((psat < 0.01) & ~Cat.zero_sfr)
            
        # fit the SFMS  
        if cc == 'scsam': 
            sample_cut = sample_cut & (logMstar > scsam_mmin)
        elif cat == 'illustris_100myr': 
            sample_cut = sample_cut & (logMstar > illustris_mmin)
        elif cat == 'eagle_100myr': 
            sample_cut = sample_cut & (logMstar > eagle_mmin)
        elif cat == 'mufasa_100myr': 
            sample_cut = sample_cut & (logMstar > mufasa_mmin)
        
        DFM.hist2d(logMstar[sample_cut], logSFR[sample_cut], color='C'+str(i_c+2), 
                levels=[0.68, 0.95], range=plot_range, 
                plot_datapoints=True, fill_contours=False, plot_density=True, ax=sub) 
        # SFS fit
        fSFS = pickle.load(open(f_gmm(cat), 'rb'))
        sub.errorbar(fSFS._fit_logm, fSFS._fit_logsfr, fSFS._fit_err_logssfr, fmt='.k')

        lbl = Cat.CatalogLabel(cat)
        sub.text(0.9, 0.1, lbl.split('[')[0], ha='right', va='center', 
                transform=sub.transAxes, fontsize=20)
        sub.set_xlim([8., 12.]) 
        sub.set_ylim([-4., 2.]) 
        sub.set_xticks([8., 9., 10., 11., 12.]) 
        if i_c == 0: 
            sub.text(0.1, 0.9, 'SFR ['+(lbl.split('[')[-1]).split(']')[0]+']', 
                    ha='left', va='top', transform=sub.transAxes, fontsize=20)
        if i_c < 2:   sub.set_xticklabels([]) 
        if i_c not in [0, 2]: sub.set_yticklabels([]) 
    
    sub = fig.add_subplot(2,3,6) 
    for i_c, cc in enumerate(sims_list): 
        cat = '_'.join([cc, tscale]) 
        fSFS = pickle.load(open(f_gmm(cat), 'rb'))
        sub.fill_between(fSFS._fit_logm, 
                fSFS._fit_logsfr - fSFS._fit_err_logssfr, 
                fSFS._fit_logsfr + fSFS._fit_err_logssfr, 
                color='C'+str(i_c+2), linewidth=0.5, alpha=0.75) 
    sub.set_xlim([8., 12.]) 
    sub.set_ylim([-4., 2.]) 
    sub.set_xticks([8., 9., 10., 11., 12.]) 
    sub.set_yticklabels([]) 
    sub.text(0.9, 0.1, 'Best-fit SFMS', ha='right', va='center', 
            transform=sub.transAxes, fontsize=20)

    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', labelpad=15, fontsize=25) 
    bkgd.set_ylabel(r'log ( SFR $[M_\odot \, yr^{-1}]$ )', labelpad=15, fontsize=25) 
    
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    
    if not nosplashback: 
        fig_name = ''.join([UT.doc_dir(), 'figs/Catalogs_SFMSfit_SFR', tscale, '.pdf'])
    else: 
        fig_name = ''.join([UT.doc_dir(), 'figs/', 
            'Catalogs_SFMSfit_SFR', tscale, '.nosplbacks.', sb_cut, '.pdf'])
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close()
    return None 


def Catalogs_SFMS_powerlawfit(): 
    ''' Compare the power-law fit of the GMM SFMS fits 
    '''
    tscales = ['inst', '100myr'] # tscales 
    sims_list = ['illustris', 'eagle', 'mufasa', 'scsam'] # simulations 

    # read in power-law fits from file 
    f_powerlaw = ''.join([UT.dat_dir(), 'paper1/SFMS_powerlawfit.txt']) 
    power_fits = {} 
    with open(f_powerlaw) as f: 
        line = f.readline()
        while line: 
            if line[0] == '#': 
                pass 
            elif line[0] == '-': 
                key = line.split(' ')[1]
                if line.split(' ')[2] == 'logM*': 
                    key += '_mlim'
                m = float(f.readline().split(':')[-1])
                b = float(f.readline().split(':')[-1])
                power_fits[key] = [m, b]
            line = f.readline() 

    Cat = Cats.Catalog()
    m_arr = np.linspace(8., 12., 100) 
    fig = plt.figure(1, figsize=(8,4))
    bkgd = fig.add_subplot(111, frameon=False)
    for i_t, tscale in enumerate(tscales): 
        sub = fig.add_subplot(1,2,1+i_t)
        for i_c, cc in enumerate(sims_list): 
            cat = '_'.join([cc, tscale]) 
            lbl = Cat.CatalogLabel(cat)
            # power-law fit of the SFMS fit 
            m, b = power_fits[cat] 
            f_sfms = lambda mm: m * (mm - 10.5) + b
            
            if 'mufasa' not in cat: 
                sub.plot(m_arr, f_sfms(m_arr), c='C'+str(i_c+2), lw=2, label=lbl.split('[')[0]) 
            else: 
                sub.plot(m_arr, f_sfms(m_arr), c='C'+str(i_c+2), lw=2, ls=':')#, label=lbl.split('[')[0]) 
                m, b = power_fits[cat+'_mlim'] 
                f_sfms = lambda mm: m * (mm - 10.5) + b
                sub.plot(m_arr, f_sfms(m_arr), c='C'+str(i_c+2), lw=2, label=r'$\mathrm{'+lbl.split('[')[0]+'}^*$') 
            
            if i_c == 0: 
                sub.text(0.1, 0.9, 'SFR ['+(lbl.split('[')[-1]).split(']')[0]+']', 
                        ha='left', va='top', transform=sub.transAxes, fontsize=20)
        sub.set_xlim([8.2, 11.8]) 
        sub.set_xticks([8.5, 9.5, 10.5, 11.5]) 
        sub.set_ylim([-4., 2.]) 
        sub.set_yticks([-4., -3., -2., -1., 0., 1., 2.]) 
        if i_t != 0: sub.set_yticklabels([]) 
    sub.legend(loc='lower right', frameon=False, prop={'size': 15}) 
    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', labelpad=10, fontsize=25) 
    bkgd.set_ylabel(r'log ( SFR $[M_\odot \, yr^{-1}]$ )', labelpad=10, fontsize=25) 
    
    fig.subplots_adjust(wspace=0.05)
    fig_name = ''.join([UT.doc_dir(), 'figs/Catalogs_SFMS_powerlawfit.pdf'])
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close()
    return None 


def Catalogs_SFMS_width(): 
    ''' Compare the wdith of the GMM SFMS fits
    '''
    Cat = Cats.Catalog()
    tscales = ['inst', '100myr'] # tscales 
    sims_list = ['illustris', 'eagle', 'mufasa', 'scsam'] # simulations 
    
    f_gmm = lambda name: ''.join([UT.dat_dir(), 
        'paper1/', 'gmmSFSfit.', name, '.gfcentral.mlim.p'])

    fig = plt.figure(1, figsize=(8,4))
    bkgd = fig.add_subplot(111, frameon=False)
    m_arr = np.linspace(8., 12., 100) 

    for i_t, tscale in enumerate(tscales): 
        sub = fig.add_subplot(1,2,1+i_t)
        for i_c, cc in enumerate(sims_list): 
            cat = '_'.join([cc, tscale]) 
            lbl = Cat.CatalogLabel(cat)
            _label = None
            if i_t == 0 and i_c in [0, 1]: 
                _label = lbl.split('[')[0]
            elif i_t == 1 and i_c in [2, 3]: 
                _label = lbl.split('[')[0]

            fSFMS = pickle.load(open(f_gmm(cat), 'rb'))
            sub.fill_between(fSFMS._fit_logm, 
                    fSFMS._fit_sig_logssfr - fSFMS._fit_err_sig_logssfr, 
                    fSFMS._fit_sig_logssfr + fSFMS._fit_err_sig_logssfr, 
                    color='C'+str(i_c+2), alpha=0.75, linewidth=0, label=_label) 
            sub.legend(loc='lower right', frameon=False, handletextpad=0.2, prop={'size': 15}) 
            if i_c == 0: 
                sub.text(0.05, 0.95, 'SFR ['+(lbl.split('[')[-1]).split(']')[0]+']', 
                        ha='left', va='top', transform=sub.transAxes, fontsize=20)
            
            # fit the widths to a line 
            xx = fSFMS._fit_logm - 10.5   # log Mstar - log M_fid
            yy = fSFMS._fit_sig_logssfr
            #yy = np.sqrt(fSFMS._fit_sig_logssfr)
            #chisq = lambda theta: np.sum((theta[0] * xx + theta[1] - yy)**2/fSFMS._fit_err_sig_logssfr**2)
            #tt = sp.optimize.minimize(chisq, np.array([0.0, 0.2])) 
            chisq = lambda theta: np.sum((theta[0] - yy)**2/fSFMS._fit_err_sig_logssfr**2)
            tt = sp.optimize.minimize(chisq, np.array([0.2])) 
            print('--%s--' % cat)
            #print('slope %f; y-int %f' % (tt['x'][0], tt['x'][1])) 
            print('amplitude %f' % (tt['x'][0])) 

        sub.plot([8., 12.], [0.3, 0.3], c='k', ls='--') 
        sub.set_xlim([8.2, 11.5]) 
        sub.set_xticks([8.5, 9.5, 10.5, 11.5]) 
        sub.set_ylim([-0.05, 0.6]) 
        if i_t != 0: sub.set_yticklabels([]) 
    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', labelpad=10, fontsize=25) 
    bkgd.set_ylabel(r'$\sigma_{\mathrm{SFMS}}$ [dex]', labelpad=10, fontsize=25) 
    
    fig.subplots_adjust(wspace=0.05)
    fig_name = ''.join([UT.doc_dir(), 'figs/Catalogs_SFMS_width.pdf'])
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

    Cat = Cats.Catalog()
    logMstar, logSFR, weight, censat = Cat.Read(sim)
    psat = Cat.GroupFinder(sim)
    iscen = ((psat < 0.01) & np.invert(Cat.zero_sfr)) 
    #iscen = (censat == 1)
        
    # fit the SFMS  
    fSFMS = fstarforms() 
    _fit_logm, _fit_logsfr = fSFMS.fit(logMstar[iscen], logSFR[iscen], fit_range=[9.0, 12.], method='gaussmix') 

    fig = plt.figure(figsize=(4*(len(mranges)+1),4)) 
    
    sub1 = fig.add_subplot(1,3,1)
    DFM.hist2d(logMstar[iscen], logSFR[iscen], color='#ee6a50',
            levels=[0.68, 0.95], range=[[9., 12.], [-3.5, 1.5]], 
            plot_datapoints=True, fill_contours=False, plot_density=True, ax=sub1) 

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
            sub2.set_ylabel('$p\,(\;\mathrm{log}\; \mathrm{SSFR}\;)$', fontsize=20)
        sub2.set_xlabel('log$(\; \mathrm{SSFR}\; [\mathrm{yr}^{-1}]\;)$', fontsize=20) 
        sub2.set_xlim([-13.25, -9.]) 
        sub2.set_xticks([-9., -10., -11., -12., -13.][::-1])
        sub2.set_ylim([0.,2.1]) 
        sub2.set_yticks([0., 0.5, 1., 1.5, 2.])
        # mass bin 
        #sub2.text(0.05, 0.95, panels[i_m], ha='left', va='top', transform=sub2.transAxes, fontsize=25)
        sub2.text(0.5, 0.9, '$'+str(mrange[0])+'< \mathrm{log}\, M_* <'+str(mrange[1])+'$',
                ha='center', va='center', transform=sub2.transAxes, fontsize=18)

    #fig.subplots_adjust(wspace=.3)
    fig.tight_layout() 
    fig_name = ''.join([UT.doc_dir(), 'figs/SFMSfit_demo.pdf'])
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

    fig_name = ''.join([UT.doc_dir(), 'figs/SFRMstar_2Dgmm.pdf'])
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
    
    f_gmm = lambda name: ''.join([UT.dat_dir(), 
        'paper1/', 'gmmSFSfit.', name, '.gfcentral.mlim.p'])

    fig = plt.figure(1, figsize=(16,8))#(20,7.5))
    bkgd = fig.add_subplot(111, frameon=False)
    plot_range = [[8., 12.], [-4., 2.]]

    for i_t, tscale in enumerate(tscales): 
        for i_c, cc in enumerate(sims_list): 
            cat = '_'.join([cc, tscale]) 
            sub = fig.add_subplot(2,4,1+i_c+i_t*4) 

            logMstar, logSFR, weight, censat = Cat.Read(cat)
            psat = Cat.GroupFinder(cat) # group finder 
            sample_cut = ((psat < 0.01) & ~Cat.zero_sfr)
                
            # fit the SFMS  
            if cc == 'scsam': 
                sample_cut = sample_cut & (logMstar > scsam_mmin)
            elif cat == 'illustris_100myr': 
                sample_cut = sample_cut & (logMstar > illustris_mmin)
            elif cat == 'eagle_100myr': 
                sample_cut = sample_cut & (logMstar > eagle_mmin)
            elif cat == 'mufasa_100myr': 
                sample_cut = sample_cut & (logMstar > mufasa_mmin)
            
            DFM.hist2d(logMstar[sample_cut], logSFR[sample_cut], color='k', 
                    levels=[0.68, 0.95], range=plot_range, 
                    plot_datapoints=True, fill_contours=False, plot_density=False, 
                    contour_kwargs={'linewidths':1, 'linestyles':'dotted'}, 
                    ax=sub) 

            fSFMS = pickle.load(open(f_gmm(cat), 'rb'))
            fit_logm = fSFMS._fit_logm
            fit_logsfr = fSFMS._fit_logsfr

            sig_sfms, w_sfms = np.zeros(len(fit_logm)), np.zeros(len(fit_logm))
            for i_m, fitlogm in enumerate(fit_logm): 
                # sfms component 
                _means = fSFMS._gbests[i_m].means_.flatten()
                _covs = fSFMS._gbests[i_m].covariances_.flatten()
                #print _means, _covs

                sfms = (_means == fit_logsfr[i_m]-fit_logm[i_m])
                sig_sfms[i_m] = np.sqrt(_covs[sfms])
                # "quenched" component 
                quenched = ((range(len(_means)) == _means.argmin()) & (_means != fit_logsfr[i_m]-fit_logm[i_m])) 
                if np.sum(quenched) > 0: 
                    sub.errorbar([fitlogm+0.01], _means[quenched]+fitlogm, yerr=np.sqrt(_covs[quenched]), fmt='.C1')
                # other component 
                starburst = (_means > fit_logsfr[i_m]-fit_logm[i_m])
                if np.sum(starburst) > 0: 
                    sub.errorbar([fitlogm + 0.01*(i+2) for i in range(np.sum(starburst))], _means[starburst]+fitlogm, 
                                 yerr=np.sqrt(_covs[starburst]), fmt='.C4')
                transition = (~sfms & ~quenched & ~starburst) 
                if np.sum(transition) > 0: 
                    sub.errorbar([fitlogm + 0.01*(i+2) for i in range(np.sum(transition))], _means[transition]+fitlogm, 
                                 yerr=np.sqrt(_covs[transition]), fmt='.C2')
            sub.errorbar(fit_logm, fit_logsfr, yerr=sig_sfms, fmt='.C0')
            sub.set_xlim([8., 12.]) 

            lbl = Cat.CatalogLabel(cat)
            if i_c == 0: 
                sub.text(0.1, 0.9, 'SFR ['+(lbl.split('[')[-1]).split(']')[0]+']', 
                        ha='left', va='center', transform=sub.transAxes, fontsize=20)
            if i_t == 1: 
                font0 = FontProperties() 
                font0.set_weight('heavy') 
                sub.text(0.9, 0.1, lbl.split('[')[0], ha='right', va='center', 
                        transform=sub.transAxes, fontsize=20, fontproperties=font0)
            if i_t == 0: sub.set_xticklabels([]) 
            if i_c != 0: sub.set_yticklabels([]) 
    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', labelpad=15, fontsize=25) 
    bkgd.set_ylabel(r'log ( SFR $[M_\odot \, yr^{-1}]$ )', labelpad=15, fontsize=25) 
    
    fig.subplots_adjust(wspace=0.1, hspace=0.075)
    fig_name = ''.join([UT.doc_dir(), 'figs/Catalogs_GMMcomps.pdf'])
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close()
    return None 


def Pssfr_GMMcomps(timescale='inst'): 
    ''' Plot P(SSFR) along with the best-fit GMM components
    in order to demonstrate that the GMM model is not overfitting 
    '''
    Cat = Cats.Catalog()
    # Read in various data sets
    sims_list = ['illustris', 'eagle', 'mufasa', 'scsam'] 
    
    f_gmm = lambda name: ''.join([UT.dat_dir(), 'paper1/', 'gmmSFSfit.', name, '.gfcentral.mlim.p'])

    fig = plt.figure(1, figsize=(16,14))#(20,7.5))
    bkgd = fig.add_subplot(1, 1, 1, frameon=False)
    for i_c, cc in enumerate(sims_list): 
        for i_mass_bin, mass_bin in enumerate([[9.2, 9.4], [9.8, 10.], [10.6, 10.8]]): 
            cat = '_'.join([cc, timescale]) 
            logMstar, logSFR, weight, censat = Cat.Read(cat)
            psat = Cat.GroupFinder(cat) # group finder 
            iscen = (psat < 0.01)
                
            # fit the SFMS  
            if cc == 'scsam': 
                mlim = (logMstar > scsam_mmin)
            elif cat == 'illustris_100myr': 
                mlim = (logMstar > illustris_mmin)
            elif cat == 'eagle_100myr': 
                mlim = (logMstar > eagle_mmin)
            elif cat == 'mufasa_100myr': 
                mlim = (logMstar > mufasa_mmin)
            else: 
                mlim = np.ones(len(logMstar)).astype(bool) 

            # SFMS fit 
            fSFS = pickle.load(open(f_gmm(cat), 'rb'))

            # P(SSFR) plot within mass bin 
            sub = fig.add_subplot(4,3,1+3*i_c+i_mass_bin) 
            font0 = FontProperties() 
            font0.set_weight('heavy') 
            lbl = Cat.CatalogLabel(cat)
            if i_mass_bin == 0: 
                sub.text(0.05, 0.9, lbl.split('[')[0], ha='left', va='top', 
                        transform=sub.transAxes, fontsize=25, fontproperties=font0)

            if mass_bin[0] >= logMstar[mlim].min(): # make sure there are galaxies in the massbin

                inmbin = ((logMstar > mass_bin[0]) & (logMstar < mass_bin[1])) 
                inmbin_nz = (inmbin & iscen & mlim & ~Cat.zero_sfr)
                inmbin_z = (inmbin & iscen & mlim & Cat.zero_sfr)

                ssfrs = np.concatenate([logSFR[inmbin_nz] - logMstar[inmbin_nz], 
                    np.repeat(-13.5, np.sum(inmbin_z))])
                print('%i of %i galaxies have zero SFR' % 
                        (np.sum(inmbin_z), np.sum(inmbin_nz)+np.sum(inmbin_z)))
                f_nz = float(np.sum(inmbin_nz)) / float(np.sum(inmbin_nz)+np.sum(inmbin_z))
                _ = sub.hist(ssfrs, bins=40, 
                        range=[-14., -8.], density=True, histtype='stepfilled', 
                        color='k', alpha=0.25, linewidth=1.75)
                i_mbin = np.where((fSFS._fit_logm > mass_bin[0]) & (fSFS._fit_logm < mass_bin[1]))[0]
                if len(i_mbin) != 1: raise ValueError
                i_mbin = i_mbin[0]
                gmm_weights = fSFS._gbests[i_mbin].weights_
                gmm_means = fSFS._gbests[i_mbin].means_
                gmm_vars = fSFS._gbests[i_mbin].covariances_
                icomps = fSFS._GMM_idcomp(fSFS._gbests[i_mbin], SSFR_cut=-11.)
                n_comp_best = len(gmm_means) 
                x_ssfr = np.linspace(-14., -9, 100)

                for i_gmm, gmm, bic in zip(
                        range(len(fSFS._gmms[i_mbin])), fSFS._gmms[i_mbin], fSFS._bics[i_mbin]):
                    gmm_ws = gmm.weights_.flatten()
                    gmm_mus = gmm.means_.flatten()
                    gmm_vars = gmm.covariances_.flatten()
                    n_comp = len(gmm_mus) 
                
                    sub.text(0.075, 0.72-float(i_gmm)*0.1, 
                            '$\mathrm{BIC}_{k='+str(i_gmm+1)+'}='+str(round(bic,2))+'$', 
                            ha='left', va='top', color='C'+str(i_gmm),
                            transform=sub.transAxes, fontsize=15)
                    lww = 0.75
                    if n_comp == n_comp_best: 
                        lww = 1

                    allgmm = np.zeros(len(x_ssfr))
                    for icomp in range(n_comp):  
                        allgmm += f_nz * gmm_ws[icomp] * MNorm.pdf(x_ssfr, gmm_mus[icomp], gmm_vars[icomp]) 
                        sub.plot(x_ssfr, 
                                f_nz * gmm_ws[icomp] * MNorm.pdf(x_ssfr, gmm_mus[icomp], gmm_vars[icomp]), 
                                c='C'+str(i_gmm), lw=0.75, ls='--') 
                    sub.plot(x_ssfr, allgmm, c='C'+str(i_gmm), lw=1.5) 
            if i_c == 0: 
                sub.set_title(str(mass_bin[0])+'$<$ log $M_*$ $<$'+str(mass_bin[1])+'', 
                        fontsize=25)
                #sub.text(0.9, 0.95, 
                #        str(mass_bin[0])+'$<$ log $M_*$ $<$'+str(mass_bin[1])+'', 
                #        ha='right', va='top',
                #        transform=sub.transAxes, fontsize=25)
            sub.set_xlim([-13.6, -9.]) 
            sub.set_xticks([-9., -10., -11., -12., -13.][::-1])
            sub.set_ylim([0.,2.]) 
            if i_c in [0, 1, 2]: sub.set_xticklabels([]) 
            if i_mass_bin in [1, 2]:  sub.set_yticklabels([]) 
            #sub.set_yticks([0., 0.5, 1., 1.5, 2.])
    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel('log$(\; \mathrm{SSFR}\; [\mathrm{yr}^{-1}]\;)$', labelpad=5, fontsize=25) 
    bkgd.set_ylabel('$p\,(\;\mathrm{log}\; \mathrm{SSFR}\; [\mathrm{yr}^{-1}]\;)$', 
            labelpad=5, fontsize=25)
        
    fig.subplots_adjust(wspace=0.1, hspace=0.075)
    fig_name = ''.join([UT.doc_dir(), 'figs/Pssfr_GMMcomps_', timescale, '.pdf'])
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close()
    return None 


def GMMcomp_weights(n_bootstrap=10, nosplashback=False, sb_cut='3vir'): 
    ''' Plot the fractional composition of the different GMM components 
    along with galaxies with zero SFRs for the different catalogs
    '''
    cats = ['illustris', 'eagle', 'mufasa', 'scsam']
    tscales = ['inst', '100myr']

    fig = plt.figure(figsize=(20,8))
    gs = mpl.gridspec.GridSpec(1,5, figure=fig) 

    mbinss, f_comps_uncs, f_compss = [], [], [] 
    for i_c, c in enumerate(cats): 
        gs_i = mpl.gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[i_c])
        for i_t, tscale in enumerate(tscales):
            name = c+'_'+tscale
            mbins, f_comps, f_comps_unc = _GMM_fcomp(name, groupfinder=True, 
                    n_bootstrap=n_bootstrap, nosplashback=nosplashback, sb_cut=sb_cut)
            mbinss.append(mbins)
            f_compss.append(f_comps)
            f_comps_uncs.append(f_comps_unc)

            f_zero, f_sfms, f_q, f_other0, f_other1 = list(f_comps)
            
            sub = plt.subplot(gs_i[i_t,0]) 
            sub.fill_between(mbins, np.zeros(len(mbins)), f_zero, # SFR = 0 
                    linewidth=0, color='C3') 
            sub.fill_between(mbins, f_zero, f_zero+f_q,              # Quenched
                    linewidth=0, color='C1') 
            sub.fill_between(mbins, f_zero+f_q, f_zero+f_q+f_other0,   # other0
                    linewidth=0, color='C2') 
            sub.fill_between(mbins, f_zero+f_q+f_other0, f_zero+f_q+f_other0+f_sfms, # SFMS 
                    linewidth=0, color='C0') 
            sub.fill_between(mbins, f_zero+f_q+f_other0+f_sfms, f_zero+f_q+f_other0+f_sfms+f_other1, # star-burst 
                    linewidth=0, color='C4') 
            if name == 'mufasa_100myr':
                mmin = mufasa_mmin
                sub.fill_between([0., mmin+0.1], [0.0, 0.0], [1., 1.], 
                        linewidth=0, color='k', alpha=0.8) 
            elif 'scsam' in name: 
                mmin = scsam_mmin
                sub.fill_between([0., mmin+0.1], [0.0, 0.0], [1., 1.], 
                        linewidth=0, color='k', alpha=0.8) 
            if c == 'mufasa': 
                sub.set_xlim([8.8, 11.3])
            else: 
                sub.set_xlim([8.8, 11.5])
            sub.set_xticks([9., 10., 11.]) 
            if i_t == 0: sub.set_xticklabels([]) 
            sub.set_ylim([0.0, 1.]) 
            if i_c != 0: sub.set_yticks([]) 

            Cat = Cats.Catalog()
            lbl = Cat.CatalogLabel(name)
            if i_c == 0: 
                sub.text(0.1, 0.875, 'SFR ['+(lbl.split('[')[-1]).split(']')[0]+']', 
                        color='white', ha='left', va='center', 
                        transform=sub.transAxes, fontsize=20)
            if i_t == 1: 
                sub.text(0.9, 0.1, lbl.split('[')[0], ha='right', va='center', color='white', 
                        transform=sub.transAxes, fontsize=20)#, path_effects=[path_effects.withSimplePatchShadow()])
            if (i_t == 0) and (i_c == 1):#len(cats)-1):  
                p1 = Rectangle((0, 0), 1, 1, linewidth=0, fc="C3")
                p2 = Rectangle((0, 0), 1, 1, linewidth=0, fc="C1")
                p3 = Rectangle((0, 0), 1, 1, linewidth=0, fc="C2")
                p4 = Rectangle((0, 0), 1, 1, linewidth=0, fc="C0")
                p5 = Rectangle((0, 0), 1, 1, linewidth=0, fc="C4")
                sub.legend([p1, p2, p3, p5, p4][::-1], ['SFR=0', 'low SF', 'intermediate SF', 'high SF', 'SFS'][::-1], 
                        loc='upper left', prop={'size': 12}) #bbox_to_anchor=(1.1, 1.05))

    obvs = ['nsa_dickey', 'tinkergroup']
    for i_c, c in enumerate(obvs): 
        if c == 'nsa_dickey': 
            mbins_nsa, f_comps_nsa, f_comps_unc_nsa = _GMM_fcomp(c, 
                    groupfinder=False, n_bootstrap=n_bootstrap)
        elif c == 'tinkergroup': 
            mbins_sdss, f_comps_sdss, f_comps_unc_sdss = _GMM_fcomp(c, 
                    groupfinder=False, n_bootstrap=n_bootstrap)

    mbins = np.concatenate([mbins_nsa, mbins_sdss]) 
    f_comps = np.concatenate((f_comps_nsa, f_comps_sdss), axis=1) 

    f_zero, f_sfms, f_q, f_other0, f_other1 = list(f_comps)

    gs_i = mpl.gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[4])
    sub = plt.subplot(gs_i[1:3,:])
    sub.fill_between(mbins, np.zeros(len(mbins)), f_zero, # SFR = 0 
            linewidth=0, color='C3') 
    sub.fill_between(mbins, f_zero, f_zero+f_q,              # Quenched
            linewidth=0, color='C1') 
    sub.fill_between(mbins, f_zero+f_q, f_zero+f_q+f_other0,   # other0
            linewidth=0, color='C2') 
    sub.fill_between(mbins, f_zero+f_q+f_other0, f_zero+f_q+f_other0+f_sfms, # SFMS 
            linewidth=0, color='C0') 
    sub.fill_between(mbins, f_zero+f_q+f_other0+f_sfms, f_zero+f_q+f_other0+f_sfms+f_other1, # star-burst 
            linewidth=0, color='C4') 
    sub.vlines(mbins_nsa.max(), 0., 1., linewidth=2, linestyle='--', color='k') 
    sub.set_xlim([8.8, 11.5])
    sub.set_xticks([9., 10., 11.]) 
    sub.set_ylim([0.0, 1.]) 
    sub.set_yticklabels([]) 
    if i_c == 1: 
        sub.text(0.375, 0.95, 'SDSS', ha='left', va='top', color='white', 
                transform=sub.transAxes, fontsize=20)
        sub.text(0.3, 0.05, 'NSA', ha='right', va='bottom', color='white', 
                transform=sub.transAxes, fontsize=20)
    #fig.text(r'log$\; M_* \;\;[M_\odot]$', labelpad=10, fontsize=30) 
    fig.text(0.075, 0.5, r'GMM component fractions', rotation='vertical', va='center', fontsize=25) 
    fig.text(0.5, 0.025, r'log$\; M_* \;\;[M_\odot]$', ha='center', fontsize=30) 
    fig.subplots_adjust(wspace=0.05, hspace=0.1)
    if not nosplashback: 
        fig_name = ''.join([UT.doc_dir(), 'figs/GMMcomp_composition.pdf'])
    else: 
        fig_name = ''.join([UT.doc_dir(), 'figs/GMMcomp_composition.nosplbacks.', sb_cut, '.pdf'])
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close() 
    
    # plot the uncertainties from bootstrap resampling 
    fig1 = plt.figure(figsize=(20,8))
    gs = mpl.gridspec.GridSpec(1,5, figure=fig1) 

    for i_c, c in enumerate(cats): 
        gs_i = mpl.gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[i_c])
        for i_t, tscale in enumerate(tscales):
            mbins = mbinss[i_t+2*i_c] 
            f_comps = f_compss[i_t+2*i_c]
            f_comps_unc = f_comps_uncs[i_t+2*i_c]
            
            f_zero, f_sfms, f_q, f_other0, f_other1 = list(f_comps)
            f_zero_unc, f_sfms_unc, f_q_unc, f_other0_unc, f_other1_unc = list(f_comps_unc)

            sub = plt.subplot(gs_i[i_t,0]) 
            sub.fill_between(mbins, f_zero-f_zero_unc, f_zero+f_zero_unc, 
                    color='C3', alpha=0.5, linewidth=1)
            sub.fill_between(mbins, f_q-f_q_unc, f_q+f_q_unc, 
                    color='C1', alpha=0.5, linewidth=1) 
            sub.fill_between(mbins, f_other0-f_other0_unc, f_other0+f_other0_unc, 
                    color='C2', alpha=0.5, linewidth=1)   # other0
            sub.fill_between(mbins, f_sfms-f_sfms_unc, f_sfms+f_sfms_unc, 
                    color='C0', alpha=0.5, linewidth=1) # SFMS 
            sub.fill_between(mbins, f_other1-f_other1_unc, f_other1+f_other1_unc, 
                    color='C4', alpha=0.5, linewidth=1) # Star-burst 
            if (c == 'mufasa') and (tscale == '100myr'):  
                sub.fill_between([0., mufasa_mmin+0.1], [0.0, 0.0], [1., 1.], linewidth=0, color='k', alpha=0.8) 
            elif (c == 'scsam'):
                sub.fill_between([0., scsam_mmin+0.1], [0.0, 0.0], [1., 1.], linewidth=0, color='k', alpha=0.8) 
            
            if c == 'mufasa': sub.set_xlim([8.8, 11.3])
            else: sub.set_xlim([8.8, 11.5])
            sub.set_xticks([9., 10., 11.]) 
            if i_t == 0: sub.set_xticklabels([]) 
            sub.set_ylim([0.0, 1.]) 
            if i_c != 0: sub.set_yticks([]) 

            lbl = Cat.CatalogLabel(c+'_'+tscale)
            if i_c == 0: 
                sub.text(0.1, 0.875, 'SFR ['+(lbl.split('[')[-1]).split(']')[0]+']', color='k', #'white', 
                        ha='left', va='center', transform=sub.transAxes, fontsize=20)
            if i_t == 1: 
                sub.text(0.9, 0.1, lbl.split('[')[0], ha='right', va='center', color='k', #'white', 
                        transform=sub.transAxes, fontsize=20) 
            if (i_t == 0) and (i_c == len(cats)-2):  
                p3 = Rectangle((0, 0), 1, 1, linewidth=0, alpha=0.5, fc="C2")
                p4 = Rectangle((0, 0), 1, 1, linewidth=0, alpha=0.5, fc="C0")
                p5 = Rectangle((0, 0), 1, 1, linewidth=0, alpha=0.5, fc="C4")
                sub.legend([p4, p5, p3][::-1], ['SFS', 'high SF', 'intermediate SF'][::-1], 
                        ncol=1, loc='upper left', frameon=False, prop={'size': 12})
            elif (i_t == 0) and (i_c == len(cats)-1): 
                p1 = Rectangle((0, 0), 1, 1, linewidth=0, alpha=0.5, fc="C3")
                p2 = Rectangle((0, 0), 1, 1, linewidth=0, alpha=0.5, fc="C1")
                sub.legend([p2, p1], ['low SF', 'SFR=0'],
                        ncol=1, loc='upper left', frameon=False, prop={'size': 12}) #bbox_to_anchor=(1.1, 1.05))

    mbins = np.concatenate([mbins_nsa, mbins_sdss]) 
    f_comps = np.concatenate((f_comps_nsa, f_comps_sdss), axis=1) 
    f_comps_unc = np.concatenate((f_comps_unc_nsa, f_comps_unc_sdss), axis=1) 

    f_zero, f_sfms, f_q, f_other0, f_other1 = list(f_comps)
    f_zero_unc, f_sfms_unc, f_q_unc, f_other0_unc, f_other1_unc = list(f_comps_unc)

    gs_i = mpl.gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[4])
    sub = plt.subplot(gs_i[1:3,:])
    sub.fill_between(mbins, f_zero-f_zero_unc, f_zero+f_zero_unc, 
            color='C3', alpha=0.5, linewidth=1)
    sub.fill_between(mbins, f_q-f_q_unc, f_q+f_q_unc, 
            color='C1', alpha=0.5, linewidth=1) 
    sub.fill_between(mbins, f_other0-f_other0_unc, f_other0+f_other0_unc, 
            color='C2', alpha=0.5, linewidth=1)   # other0
    sub.fill_between(mbins, f_sfms-f_sfms_unc, f_sfms+f_sfms_unc, 
            color='C0', alpha=0.5, linewidth=1) # SFMS 
    sub.fill_between(mbins, f_other1-f_other1_unc, f_other1+f_other1_unc, 
            color='C4', alpha=0.5, linewidth=1) # Star-burst 
    sub.vlines(mbins_nsa.max(), 0., 1., linewidth=2, linestyle='--', color='k') 
    sub.set_xlim([8.8, 11.5])
    sub.set_xticks([9., 10., 11.]) 
    sub.set_ylim([0.0, 1.]) 
    sub.set_yticklabels([]) 
    if i_c == 1: 
        sub.text(0.375, 0.95, 'SDSS', ha='left', va='top', color='white', 
                transform=sub.transAxes, fontsize=20)
        sub.text(0.3, 0.05, 'NSA', ha='right', va='bottom', color='white', 
                transform=sub.transAxes, fontsize=20)
    fig1.text(0.075, 0.5, r'GMM component fractions', rotation='vertical', va='center', fontsize=25) 
    fig1.text(0.5, 0.025, r'log$\; M_* \;\;[M_\odot]$', ha='center', fontsize=30) 
    fig1.subplots_adjust(wspace=0.05, hspace=0.1)
    if not nosplashback: 
        fig_name = ''.join([UT.doc_dir(), 'figs/GMMcomp_comp_uncertainty.pdf'])
    else: 
        fig_name = ''.join([UT.doc_dir(), 'figs/GMMcomp_comp_uncertainty.nosplbacks.', sb_cut, '.pdf'])
    fig1.savefig(fig_name, bbox_inches='tight')
    plt.close()
    return None 


def _GMM_fcomp(name, groupfinder=True, n_bootstrap=10, nosplashback=False, sb_cut='3vir', silent=True):         
    Cat = Cats.Catalog()
    logM, logSFR, w, censat = Cat.Read(name, keepzeros=True, silent=True)
    if groupfinder: 
        if not nosplashback: 
            psat = Cat.GroupFinder(name) 
            iscen = (psat < 0.01)
        else: 
            iscen = Cat.noGFSplashbacks(name, cut=sb_cut) 
    else: 
        iscen = (censat == 1)

    # same stellar mass limits as run/project1.py
    if 'scsam' in name: 
        mlim = (logM > scsam_mmin)
    elif name == 'illustris_100myr': 
        mlim = (logM > illustris_mmin)
    elif name == 'eagle_100myr': 
        mlim = (logM > eagle_mmin)
    elif name == 'mufasa_100myr': 
        mlim = (logM > mufasa_mmin)
    elif name == 'nsa_dickey': 
        mlim = (logM < dickey_mmax)
    elif name == 'tinkergroup': 
        mlim = (logM > tinker_mmin)
    else: 
        mlim = np.ones(len(logM)).astype(bool) 

    # read in the best-fit SFS from the GMM fitting
    f_gmm = ''.join([UT.dat_dir(), 'paper1/', 'gmmSFSfit.', name, '.gfcentral.lowNbinthresh.mlim.p'])
    fSFMS = pickle.load(open(f_gmm, 'rb'))
    
    mbin0 = fSFMS._mbins[fSFMS._mbins_nbinthresh,0]
    mbin1 = fSFMS._mbins[fSFMS._mbins_nbinthresh,1]
    gbests = fSFMS._gbests
    assert len(mbin0) == len(gbests)
       
    nmbin = len(mbin0) 
    f_comps = np.zeros((5, nmbin)) # zero, sfms, q, other0, other1
    f_comps_unc = np.zeros((5, nmbin))
    
    for i_m, gbest in zip(range(len(mbin0)), gbests): 
        # calculate the fraction of galaxies have that zero SFR
        mbin_iscen = iscen & (logM > mbin0[i_m]) & (logM < mbin1[i_m])
        mbin_iscen_z = mbin_iscen & Cat.zero_sfr
        f_comps[0, i_m] = float(np.sum(mbin_iscen_z))/float(np.sum(mbin_iscen))

        weights_i = gbest.weights_

        i_sfms, i_q, i_int, i_sb = fSFMS._GMM_idcomp(gbest, silent=True)

        f_nz = 1. - f_comps[0, i_m]  # multiply by non-zero fraction
        if i_sfms is not None: 
            f_comps[1, i_m] = f_nz * np.sum(weights_i[i_sfms])
        if i_q is not None: 
            f_comps[2, i_m] = f_nz * np.sum(weights_i[i_q])
        if i_int is not None: 
            f_comps[3, i_m] = f_nz * np.sum(weights_i[i_int])
        if i_sb is not None: 
            f_comps[4, i_m] = f_nz * np.sum(weights_i[i_sb])

        # bootstrap uncertainty 
        X = logSFR[mbin_iscen] - logM[mbin_iscen] # logSSFRs
        n_best = len(gbest.means_.flatten())

        f_boots = np.zeros((5, n_bootstrap))
        for i_boot in range(n_bootstrap): 
            X_boot = np.random.choice(X.flatten(), size=len(X), replace=True) 
            if name not in ['eagle_inst', 'eagle_100myr', 'mufasa_insta', 'mufasa_100myr']: 
                zero = np.invert(np.isfinite(X_boot))
            else: 
                zero = (np.invert(np.isfinite(X_boot)) | (X_boot <= -99.))

            f_boots[0,i_boot] = float(np.sum(zero))/float(len(X))
            if not silent: print('%f - %f : %f' % (mbin0[i_m], mbin1[i_m], f_boots[0,i_boot]))
                
            gmm_boot = GMix(n_components=n_best)
            gmm_boot.fit(X_boot[~zero].reshape(-1,1))
            weights_i = gmm_boot.weights_

            i_sfms, i_q, i_int, i_sb = fSFMS._GMM_idcomp(gmm_boot, silent=True)
        
            f_nonzero = 1. - f_boots[0,i_boot] 
            if i_sfms is not None: 
                f_boots[1,i_boot] = f_nonzero * np.sum(weights_i[i_sfms])
            if i_q is not None: 
                f_boots[2,i_boot] = f_nonzero * np.sum(weights_i[i_q])
            if i_int is not None: 
                f_boots[3,i_boot] = f_nonzero * np.sum(weights_i[i_int])
            if i_sb is not None: 
                f_boots[4,i_boot] = f_nonzero * np.sum(weights_i[i_sb])
        for i_b in range(f_boots.shape[0]): 
            f_comps_unc[i_b,i_m] = np.std(f_boots[i_b,:]) 
    return 0.5*(mbin0 + mbin1),  f_comps, f_comps_unc 


def rhoSF(cumulative=True):
    ''' Calculate the star formation density of the simulations
    '''
    tscales = ['inst', '100myr']
    sims_list = ['illustris', 'eagle', 'mufasa', 'scsam']#, 'mufasa'
    sims_vol = [106**3, 100**3, (50/0.68)**3, (100/0.678)**3] # volume 

    mbin = np.linspace(8.8, 12., 20) 
    Cat = Cats.Catalog()
    for i_t, tscale in enumerate(tscales): 
        fig = plt.figure(figsize=(6,6))
        bkgd = fig.add_subplot(111, frameon=False)
        sub = fig.add_subplot(111)
        for i_c, cc in enumerate(sims_list): 
            cat = '_'.join([cc, tscale]) 
            logMstar, logSFR, weight, censat = Cat.Read(cat)

            sfr_tot = np.sum(10**logSFR) # Msun/yr
            print("%s %s, log(SF density) %f Msun/yr/Mpc^3" % (cc, tscale, np.log10(sfr_tot/sims_vol[i_c])))

            rho_cum = np.zeros(len(mbin))
            rho_cum_cen = np.zeros(len(mbin))
            rho_cum_sat = np.zeros(len(mbin))
            for i_m in range(len(mbin)-1): #, m in enumerate(mbin): 
                if cumulative: 
                    lessthan = (logMstar < mbin[i_m]) & np.isfinite(logSFR)
                else: 
                    lessthan = (logMstar > mbin[i_m]) & (logMstar < mbin[i_m+1]) & np.isfinite(logSFR)
                lessthan_cen = lessthan & (censat == 1)
                lessthan_sat = lessthan & (censat != 1)
                if np.sum(lessthan) == 0: 
                    continue 
                rho_cum[i_m] = np.log10(np.sum(10**logSFR[lessthan])/sims_vol[i_c])
                if np.sum(lessthan_cen) > 0: 
                    rho_cum_cen[i_m] = np.log10(np.sum(10**logSFR[lessthan_cen])/sims_vol[i_c])
                if np.sum(lessthan_sat) > 0: 
                    rho_cum_sat[i_m] = np.log10(np.sum(10**logSFR[lessthan_sat])/sims_vol[i_c])

            lbl = Cat.CatalogLabel(cat)
            sub.plot(mbin, 10.**rho_cum, c='C'+str(i_c+2), label=lbl.split('[')[0]) 
            sub.plot(mbin, 10.**rho_cum_cen, c='C'+str(i_c+2), ls='--')#, label='Centrals') 
            sub.plot(mbin, 10.**rho_cum_sat, c='C'+str(i_c+2), ls=':')#, label='Satellites') 
            sub.set_xlim([8.8, 12.]) 
            if cumulative:
                sub.set_ylim([1e-4, 1e-1]) 
            else: 
                sub.set_ylim([1e-5, 10**-1.5]) 
            sub.set_yscale('log') 
            #if i_c != 0: sub.set_yticklabels([])
            #if i_c == 0: sub.legend(loc='upper left', frameon=False, prop={'size': 20})
            #sub.text(0.9, 0.1, lbl.split('[')[0], ha='right', va='center', transform=sub.transAxes, fontsize=20)
        sub.legend(loc='upper left', frameon=False, prop={'size': 20})
        bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        bkgd.set_ylabel(r'cumulative log $\rho_\mathrm{SFR}$  $[M_\odot/yr/\mathrm{Mpc}^3])$', labelpad=10, fontsize=20)
        bkgd.set_xlabel(r'log $M_*$  $[M_\odot]$', labelpad=10, fontsize=20)

        fig.subplots_adjust(wspace=0.05, hspace=0.05)
        if cumulative:
            fig_name = ''.join([UT.doc_dir(), 'figs/rhoSFR_cum_', tscale, '.pdf'])
        else: 
            fig_name = ''.join([UT.doc_dir(), 'figs/rhoSFR_', tscale, '.pdf'])
        fig.savefig(fig_name, bbox_inches='tight')
    return None 


def SMF(): 
    '''
    '''
    sims_list = ['illustris', 'eagle', 'mufasa', 'scsam']#, 'mufasa'
    sims_vol = [106**3, 100**3, (50/0.68)**3, (100/0.678)**3] # volume 
    Cat = Cats.Catalog()

    fig = plt.figure(figsize=(20,4))
    bkgd = fig.add_subplot(111, frameon=False)
    sub0 = fig.add_subplot(1,5,1)
    for i_c, cc in enumerate(sims_list): 
        cat = '_'.join([cc, 'inst']) 
        logMstar, logSFR, weight, censat = Cat.Read(cat)
       
        mbin = np.linspace(8.8, 12., 10) 
        phi, phi_cen, phi_sat = [], [], [] 
        for i_m in range(len(mbin)-1): 
            inmbin = ((logMstar > mbin[i_m]) & (logMstar < mbin[i_m+1]))
            inmbin_cen = inmbin & (censat == 1)
            inmbin_sat = inmbin & (censat != 1)

            phi.append(float(np.sum(inmbin)) / sims_vol[i_c])
            phi_cen.append(float(np.sum(inmbin_cen)) / sims_vol[i_c])
            phi_sat.append(float(np.sum(inmbin_sat)) / sims_vol[i_c])
    
        lbl = Cat.CatalogLabel(cat)
        sub0.plot(0.5*(mbin[1:] + mbin[:-1]), np.log10(phi), c='C'+str(i_c+2), label=lbl.split('[')[0]) 
        sub0.set_xlim([8.8, 12.]) 
        sub0.set_ylim([-5., -1.8]) 
    
        sub = fig.add_subplot(1,5,i_c+2)
        sub.plot(0.5*(mbin[1:] + mbin[:-1]), np.log10(phi), c='k')
        sub.plot(0.5*(mbin[1:] + mbin[:-1]), np.log10(phi_cen), c='C0', label='Centrals') 
        sub.plot(0.5*(mbin[1:] + mbin[:-1]), np.log10(phi_sat), c='C1', label='Satellites') 
        sub.set_xlim([8.8, 12.]) 
        sub.set_ylim([-5., -1.8]) 
        sub.set_yticklabels([])
        if i_c == 3: sub.legend(loc='lower left', frameon=False, prop={'size': 15})
        sub.text(0.95, 0.95, lbl.split('[')[0], ha='right', va='top', transform=sub.transAxes, fontsize=20)

    sub0.legend(loc='lower left', frameon=False, prop={'size': 15})
    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_ylabel(r'log $\Phi$  $[\mathrm{Mpc}^{-3}])$', labelpad=10, fontsize=20)
    bkgd.set_xlabel(r'log $M_*$  $[M_\odot]$', labelpad=10, fontsize=20)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    fig_name = ''.join([UT.doc_dir(), 'figs/SMF_sims.pdf'])
    fig.savefig(fig_name, bbox_inches='tight')
    return None 


def fsat():
    sims_list = ['illustris', 'eagle', 'mufasa', 'scsam']#, 'mufasa'
    sims_vol = [106**3, 100**3, (50/0.68)**3, (100/0.678)**3] # volume 
    Cat = Cats.Catalog()

    fig = plt.figure(figsize=(6,6))
    bkgd = fig.add_subplot(111, frameon=False)
    sub = fig.add_subplot(111) 
    for i_c, cc in enumerate(sims_list): 
        cat = '_'.join([cc, 'inst']) 
        logMstar, logSFR, weight, censat = Cat.Read(cat)
       
        mbin = np.linspace(8.0, 12., 20) 
        fsat = np.zeros(len(mbin)-1)
        for i_m in range(len(mbin)-1): 
            inmbin = ((logMstar > mbin[i_m]) & (logMstar < mbin[i_m+1]))
            if np.sum(inmbin) ==  0: 
                continue 
            inmbin_sat = inmbin & (censat != 1)
            fsat[i_m] = float(np.sum(inmbin_sat))/float(np.sum(inmbin))

        lbl = Cat.CatalogLabel(cat)
        sub.plot((0.5*(mbin[1:] + mbin[:-1]))[fsat > 0], fsat[fsat > 0], c='C'+str(i_c), label=lbl.split('[')[0]) 
    sub.legend(loc='upper right', frameon=False, prop={'size': 20})  
    sub.set_xlim([8.0, 12.]) 
    sub.set_ylim([0.,1.]) 
    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_ylabel(r'$f_\mathrm{sat}$', labelpad=10, fontsize=20)
    bkgd.set_xlabel(r'log $M_*$  $[M_\odot]$', labelpad=10, fontsize=20)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    fig_name = ''.join([UT.doc_dir(), 'figs/fsat_sims.pdf'])
    fig.savefig(fig_name, bbox_inches='tight')
    return None 


def dSFS(method='interpexterp'): 
    ''' plot galaxies that would be classified as quiescent based on 
    a d_SFMS < -1 cut. 
    ''' 
    f_sfs = lambda name: ''.join([UT.dat_dir(), 'paper1/dsfs.', name, '.gfcentrals.mlim.dat']) 
    f_gmm = lambda name: ''.join([UT.dat_dir(), 'paper1/', 'gmmSFSfit.', name, '.gfcentral.mlim.p'])
    if method == 'powerlaw': 
        icol_dsfs = 2 
    elif method == 'interpexterp': 
        icol_dsfs = 3 

    fig = plt.figure(figsize=(20,8))
    gs = mpl.gridspec.GridSpec(1,5, figure=fig) 
    # plot SFR-M* for the observations 
    gs_i = mpl.gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[4])
    sub0 = plt.subplot(gs_i[1:3,:])
    for cat in ['tinkergroup', 'nsa_dickey']:
        logmstar, logsfr, dsfs = np.loadtxt(f_sfs(cat), 
                skiprows=4, unpack=True, delimiter=',', usecols=[0,1,icol_dsfs]) 
        fSFMS = pickle.load(open(f_gmm(cat), 'rb'))
        
        sub0.scatter(logmstar, logsfr, c='k', s=1) 
        sub0.errorbar(fSFMS._fit_logm, fSFMS._fit_logsfr, fSFMS._fit_err_logssfr, fmt='.C0')
        quiescent = ((dsfs < -1.) & (dsfs != -999.)) 
        sub0.scatter(logmstar[quiescent], logsfr[quiescent], c='C1', s=1) 
    sub0.set_xlim([8., 12.]) 
    sub0.set_ylim([-4., 2.]) 
    sub0.text(0.9, 0.1, 'SDSS Centrals', ha='right', va='center', 
                transform=sub0.transAxes, fontsize=20)
    
    Cat = Cats.Catalog()
    tscales = ['inst', '100myr']
    sims_list = ['illustris', 'eagle', 'mufasa', 'scsam']  # simulations 
    for i_c, cc in enumerate(sims_list): 
        gs_i = mpl.gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[i_c])
        for i_t, tscale in enumerate(tscales): 
            cat = '_'.join([cc, tscale]) 
            sub = plt.subplot(gs_i[i_t,0]) 
        
            logmstar, logsfr, dsfs = np.loadtxt(f_sfs(cat), 
                    skiprows=4, unpack=True, delimiter=',', usecols=[0,1,icol_dsfs]) 
            fSFMS = pickle.load(open(f_gmm(cat), 'rb'))

            fpowerlaw = fSFMS.powerlaw(logMfid=10.5) 
            sub.scatter(logmstar, logsfr, c='k', s=1) 
            sub.errorbar(fSFMS._fit_logm, fSFMS._fit_logsfr, fSFMS._fit_err_logssfr, fmt='.C0')
            quiescent = ((dsfs < -1.) & (dsfs != -999.)) 
            sub.scatter(logmstar[quiescent], logsfr[quiescent], c='C1', s=1) 
            
            marr = np.linspace(8., 12., 20) 
            sub.plot(marr, fpowerlaw(marr), c='r', ls='--')
            sub.set_xlim([8., 12.]) 
            sub.set_ylim([-4., 2.]) 
            
            lbl = Cat.CatalogLabel(cat)
            if i_c == 0: 
                sub.text(0.1, 0.9, 'SFR ['+(lbl.split('[')[-1]).split(']')[0]+']', 
                        ha='left', va='center', transform=sub.transAxes, fontsize=20)
            if i_t == 1: 
                sub.text(0.9, 0.1, lbl.split('[')[0], ha='right', va='center', 
                        transform=sub.transAxes, fontsize=20)

    fig.text(0.5, 0.025, r'log$\; M_* \;\;[M_\odot]$', ha='center', fontsize=30) 
    fig.text(0.075, 0.5, r'log ( SFR $[M_\odot \, yr^{-1}]$ )', rotation='vertical', va='center', fontsize=25) 
    fig.subplots_adjust(wspace=0.2, hspace=0.15)
    fig_name = ''.join([UT.doc_dir(), 'figs/Catalogs_dSFS.', method, '.png'])
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close()
    return None 


##############################
# Appendix: 100Myr SFR Resolution Effect
##############################
def Pssfr_res_impact(n_mc=100, seed=1, poisson=False): 
    ''' Plot illustrating the impact of 100Myr SFR resolution  
    on the P(SSFR) distribution. 
    '''
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
            psat = Cat.GroupFinder(cat_name)
            _SFR = 10**_logSFR

            sub = fig.add_subplot(2,3,i_c+1+3*i_m)
            iscen = (psat < 0.01) # only pure central galaxies identified from the group catalog 
            iscen_nz_mbin = iscen & np.invert(Cat.zero_sfr) & (_logM > mbin[0]) & (_logM < mbin[1])
            iscen_z_mbin  = iscen & Cat.zero_sfr & (_logM > mbin[0]) & (_logM < mbin[1])
            ngal_bin = float(np.sum(iscen & (_logM > mbin[0]) & (_logM < mbin[1])))

            hs_uniform, hs_poisson = [], [] 
            rand = np.random.RandomState(seed)
            for ii in range(n_mc):
                sfr_nz = _SFR[iscen_nz_mbin] + dsfr*rand.uniform(size=np.sum(iscen_nz_mbin))
                sfr_z = dsfr * np.random.uniform(size=np.sum(iscen_z_mbin))

                logssfr_nz = np.log10(sfr_nz) - _logM[iscen_nz_mbin]
                logssfr_z = np.log10(sfr_z) - _logM[iscen_z_mbin]
                logssfr = np.concatenate([logssfr_nz, logssfr_z])

                h0, h1 = np.histogram(logssfr, bins=40, range=[-16., -8.])
                hs_uniform.append(h0)
                
                sfr_nz = _SFR[iscen_nz_mbin] + dsfr*rand.poisson(size=np.sum(iscen_nz_mbin))
                sfr_z = dsfr*rand.poisson(size=np.sum(iscen_z_mbin))

                logssfr_nz = np.log10(sfr_nz) - _logM[iscen_nz_mbin]
                logssfr_z = np.log10(sfr_z) - _logM[iscen_z_mbin]
                logssfr = np.concatenate([logssfr_nz, logssfr_z])

                h0, h1 = np.histogram(logssfr, bins=40, range=[-16., -8.])
                hs_poisson.append(h0)

            hs_uniform = np.array(hs_uniform)/ngal_bin
            hs_poisson = np.array(hs_poisson)/ngal_bin
               
            h0, h1 = np.histogram(_logSFR[iscen_nz_mbin] - _logM[iscen_nz_mbin], bins=40, range=[-16., -8.])
            bar_x, bar_y = UT.bar_plot(h1, h0/ngal_bin)
            sub.plot(bar_x, bar_y, c='k', ls='-', lw=1.5)#, label='w/o SFR $=0$')

            bar_x, bar_y = UT.bar_plot(h1, np.mean(hs_uniform,axis=0))
            sub.plot(bar_x, bar_y, c='C0', lw=1, label=r"$\mathrm{SFR}_i' \in [\mathrm{SFR}, \mathrm{SFR}+\Delta_\mathrm{SFR}]$")
            sub.errorbar(0.5*(h1[1:] + h1[:-1])-0.02, np.mean(hs_uniform, axis=0), yerr=np.std(hs_uniform, axis=0),
                         fmt='.C0', markersize=.5)
            
            if poisson:
                bar_x, bar_y = UT.bar_plot(h1, np.mean(hs_poisson,axis=0))
                sub.plot(bar_x, bar_y, c='C1', ls='--', lw=1)#, label=r"$\mathrm{SFR}_i' \in [\mathrm{SFR}, \mathrm{SFR}+\Delta_\mathrm{SFR}]$")
                sub.errorbar(0.5*(h1[1:] + h1[:-1])-0.02, np.mean(hs_poisson, axis=0), yerr=np.std(hs_poisson, axis=0),
                             fmt='.C1', markersize=.5)
            #sub.errorbar(0.5*(h1[1:] + h1[:-1])+0.02, np.mean(hs_nz, axis=0), 
            #        yerr=np.std(hs_nz, axis=0), fmt='.k', markersize=.5)

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
    
            if (i_c == 2) and (i_m == 1): 
                sub.legend(loc='lower left', bbox_to_anchor=(0.01, 0.65), frameon=False, prop={'size':13}) 
    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_ylabel(r'P(log SSFR  $[yr^{-1}])$', labelpad=10, fontsize=20)
    bkgd.set_xlabel(r'log SSFR  $[yr^{-1}]$', labelpad=10, fontsize=20)

    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    if poisson:
        fig_name = ''.join([UT.doc_dir(), 'figs/Pssfr_res_impact_poisson.pdf'])
    else: 
        fig_name = ''.join([UT.doc_dir(), 'figs/Pssfr_res_impact.pdf'])
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close()
    return None


def Mlim_res_impact(n_mc=20, seed=10, threshold=0.2): 
    ''' Measure the stellar mass where the resolution limit of simulations
    impact the SFMS fits.

    Threshold is in units of dex
    '''
    catalogs = ['illustris_100myr', 'eagle_100myr', 'mufasa_100myr']
    catnames = ['Illustris', 'EAGLE', 'MUFASA']
    fig = plt.figure(figsize=(4*len(catalogs),4))
    bkgd = fig.add_subplot(111, frameon=False)

    Cat = Cats.Catalog()
    for i_n, name in enumerate(catalogs): 
        _logm, _logsfr, _, censat = Cat.Read(name, keepzeros=True, silent=True)
        psat = Cat.GroupFinder(name)
        iscen_nz = (psat < 0.01) & np.invert(Cat.zero_sfr)
        iscen_z  = (psat < 0.01) & Cat.zero_sfr
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
            #sfr_low = _sfr[iscen_nz] - dsfr
            #sfr_low = np.clip(sfr_low, 0., None) 
            #logsfr_nz = np.log10(np.random.uniform(sfr_low, _sfr[iscen_nz]+dsfr, size=ngal_nz))
            logsfr_nz = np.log10(_sfr[iscen_nz] + dsfr * np.random.uniform(size=ngal_nz))
            logsfr_z = np.log10(dsfr * np.random.uniform(size=ngal_z))
            logsfr = np.concatenate([logsfr_nz, logsfr_z]) # scattered log SFR
        
            fit_logm_s, fit_logsfr_s_i = fSFMS_s.fit(logm, logsfr, method='gaussmix', 
                    dlogm=0.2, fit_range=[8.,11.], max_comp=4, silent=True) 
            fit_logsfr_s.append(fit_logsfr_s_i)
    
        sig_fit_logsfr_s = np.std(np.array(fit_logsfr_s), ddof=1, axis=0) # scatter in the fits
        fit_logsfr_s = np.mean(fit_logsfr_s, axis=0) # mean fit

        # fit without scatter
        fSFMS = fstarforms() # fit the SFMS  
        fit_logm_ns, fit_logsfr_ns, sig_fit_logsfr_ns = fSFMS.fit(_logm[iscen_nz], _logsfr[iscen_nz], method='gaussmix', 
                    dlogm=0.2, fit_range=[8.,11.], max_comp=3, fit_error='bootstrap', 
                    n_bootstrap=n_mc, silent=True) 

        #sub.fill_between(fSFMS._fit_logm, fSFMS._fit_sig_logssfr - fSFMS._fit_err_sig_logssfr, 
        #        fSFMS._fit_sig_logssfr + fSFMS._fit_err_sig_logssfr, color='C'+str(i_c+2), 
        #        alpha=0.75, linewidth=0, label=_label) 
        #fit_logsfr_ns = [] 
        #for i in range(n_mc): 
        #    fit_logm_ns, fit_logsfr_ns_i = fSFMS.fit(_logm[iscen_nz], _logsfr[iscen_nz], method='gaussmix', 
        #            dlogm=0.2, fit_range=[8.,11.], maxcomp=3, forTest=True, silent=True) 
        #    fit_logsfr_ns.append(fit_logsfr_ns_i)
        #sig_fit_logsfr_ns = np.std(np.array(fit_logsfr_ns), ddof=1, axis=0) # scatter in the fits
        #fit_logsfr_ns = np.mean(fit_logsfr_ns, axis=0) # mean fit
        sig_tot = np.clip(sig_fit_logsfr_s + sig_fit_logsfr_ns, 0.1, None) 
        #print 'sig=', sig_tot

        # check that the mbins are equal between the fit w/ scatter and fit w/o scatter
        assert np.array_equal(fSFMS._mbins, fSFMS_s._mbins)

        dfit = fit_logsfr_ns - fit_logsfr_s # change in SFMS fit caused by resolution limit 
        #print np.abs(dfit)/sig_tot 

        # log M* where resolution limit causes the SFMS to shift by 0.2 dex
        mbin_mid = 0.5 * (fSFMS._mbins[:,0] + fSFMS._mbins[:,1])
        if dfit.shape[0] != mbin_mid.shape[0]: 
            i0 = np.max(np.where(mbin_mid < fit_logm_s.min()-0.1)[0])
            mbin_mid = mbin_mid[i0+1:]
            assert len(mbin_mid) == len(dfit)
        mlim = (mbin_mid[(np.abs(dfit) > threshold)]).max() + 0.5 * fSFMS._dlogm 
        for tt in [0.1, 0.15, 0.2, 0.25]: 
            print('-------------------------') 
            print('%s' % name)
            print('for %f dex threshold: M_lim = %f' % (tt, (mbin_mid[(np.abs(dfit) > tt)]).max() + 0.5 * fSFMS._dlogm))
        
        # lets plot this stuff 
        sub = fig.add_subplot(1,len(catalogs),i_n+1)
        DFM.hist2d(_logm[iscen_nz], _logsfr[iscen_nz], color='k',
                levels=[0.68, 0.95], range= [[7., 12.], [-4., 2.]],
                plot_datapoints=False, fill_contours=False, plot_density=False, 
                contour_kwargs={'linewidths':1, 'linestyles':'dotted'}, ax=sub) 

        #sub.scatter(fit_logm_ns, fit_logsfr_ns, marker='x', color='k', lw=1, s=40, 
        #        label='w/ Res. Effect')
        if i_n == 1: lbl0 = 'with SFR resolution'
        else: lbl0 = None 
        sub.errorbar(fit_logm_ns, fit_logsfr_ns, sig_fit_logsfr_ns, fmt='.k',
                label=lbl0)
        if i_n == 2: lbl1 = 'without SFR resolution' #r"$\mathrm{SFR}_i' \in [\mathrm{SFR}_i, \mathrm{SFR}_i+\Delta_\mathrm{SFR}]$"
        else: lbl1 = None 
        sub.scatter(fit_logm_s, fit_logsfr_s, marker='x', color='C1', lw=1, s=40, 
                label=lbl1)
        sub.text(0.95, 0.05, '$\log\,M_{\lim}='+str(round(mlim,2))+'$', 
                 ha='right', va='bottom', transform=sub.transAxes, fontsize=15)
        sub.vlines(mlim, -4., 2., color='k', linestyle='--', linewidth=0.5)
        sub.fill_between([7.5, mlim], [-3., -3.], [2., 2.], color='k', alpha=0.2)
        sub.set_xlim([7.5, 11.8])
        sub.set_xticks([8., 9., 10., 11.]) 
        sub.set_ylim([-3., 2.])
        sub.set_title(catnames[i_n], fontsize=20)
        if i_n != 0: sub.set_yticklabels([]) 
        #if i_n == len(catalogs)-1: 
        sub.legend(loc='upper right', #bbox_to_anchor=(-0.075, 1.), 
                handletextpad=-0.02, frameon=False, prop={'size':15}) 
    
    bkgd.set_xlabel(r'log $M_* \;\;[M_\odot]$', labelpad=10, fontsize=25) 
    bkgd.set_ylabel(r'log SFR $[M_\odot \, yr^{-1}]$', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    fig.subplots_adjust(wspace=0.05)
    fig_name = ''.join([UT.doc_dir(), 'figs/Mlim_res_impact.pdf'])
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close()
    return None


def GMMcomp_weights_res_impact(n_bootstrap=10): 
    ''' Plot the fractional composition of the different GMM components 
    along with galaxies with zero SFRs for the different catalogs
    '''
    cats = ['illustris', 'eagle', 'mufasa']

    fig = plt.figure(figsize=(12,8))

    for i_c, c in enumerate(cats): 
        name = c+'_100myr'
        mbins, f_comps, f_comps_unc = _GMM_fcomp_res_impact(name, groupfinder=True, n_bootstrap=n_bootstrap)

        f_zero, f_sfms, f_q, f_other0, f_other1 = list(f_comps)

        sub = fig.add_subplot(2,3,i_c+1) 
        sub.fill_between(mbins, np.zeros(len(mbins)), f_zero, # SFR = 0 
                linewidth=0, color='C3') 
        sub.fill_between(mbins, f_zero, f_zero+f_q,              # Quenched
                linewidth=0, color='C1') 
        sub.fill_between(mbins, f_zero+f_q, f_zero+f_q+f_other0,   # other0
                linewidth=0, color='C2') 
        sub.fill_between(mbins, f_zero+f_q+f_other0, f_zero+f_q+f_other0+f_sfms, # SFMS 
                linewidth=0, color='C0') 
        sub.fill_between(mbins, f_zero+f_q+f_other0+f_sfms, f_zero+f_q+f_other0+f_sfms+f_other1, # star-burst 
                linewidth=0, color='C4') 
        if c == 'mufasa':
            mmin = 9.2
            sub.fill_between([0., mmin+0.1], [0.0, 0.0], [1., 1.], linewidth=0, color='k', alpha=0.8) 
            sub.set_xlim([8.8, 11.3])
        else: 
            sub.set_xlim([8.8, 11.5])
        sub.set_xticks([9., 10., 11.]) 
        sub.set_ylim([0.0, 1.]) 
        if i_c != 0: sub.set_yticks([]) 

        Cat = Cats.Catalog()
        lbl = Cat.CatalogLabel(name)
        #if i_c == 0: 
        #    sub.text(0.1, 0.875, 'SFR ['+(lbl.split('[')[-1]).split(']')[0]+']', color='white', 
        #            ha='left', va='center', transform=sub.transAxes, fontsize=20)
        sub.text(0.9, 0.1, lbl.split('[')[0], ha='right', va='center', color='white', 
                transform=sub.transAxes, fontsize=20)#, path_effects=[path_effects.withSimplePatchShadow()])
        if i_c == 0:#len(cats)-1):  
            p2 = Rectangle((0, 0), 1, 1, linewidth=0, fc="C1")
            p3 = Rectangle((0, 0), 1, 1, linewidth=0, fc="C2")
            p4 = Rectangle((0, 0), 1, 1, linewidth=0, fc="C0")
            p5 = Rectangle((0, 0), 1, 1, linewidth=0, fc="C4")
            sub.legend([p2, p3, p5, p4][::-1], ['``quenched"', 'other', 'other', 'SFMS'][::-1], 
                    loc='upper left', prop={'size': 12}) #bbox_to_anchor=(1.1, 1.05))

        # plot the uncertainties from bootstrap resampling 
        f_zero_unc, f_sfms_unc, f_q_unc, f_other0_unc, f_other1_unc = list(f_comps_unc)

        sub = fig.add_subplot(2,3,i_c+4) 
        sub.fill_between(mbins, f_zero-f_zero_unc, f_zero+f_zero_unc, 
                color='C3', alpha=0.5, linewidth=1)
        sub.fill_between(mbins, f_q-f_q_unc, f_q+f_q_unc, 
                color='C1', alpha=0.5, linewidth=1) 
        sub.fill_between(mbins, f_other0-f_other0_unc, f_other0+f_other0_unc, 
                color='C2', alpha=0.5, linewidth=1)   # other0
        sub.fill_between(mbins, f_sfms-f_sfms_unc, f_sfms+f_sfms_unc, 
                color='C0', alpha=0.5, linewidth=1) # SFMS 
        sub.fill_between(mbins, f_other1-f_other1_unc, f_other1+f_other1_unc, 
                color='C4', alpha=0.5, linewidth=1) # Star-burst 
        if c == 'mufasa':  
            sub.fill_between([0., mmin+0.1], [0.0, 0.0], [1., 1.], linewidth=0, color='k', alpha=0.8) 
            sub.set_xlim([8.8, 11.3])
        else: 
            sub.set_xlim([8.8, 11.5])
        sub.set_xticks([9., 10., 11.]) 
        sub.set_ylim([0.0, 1.]) 
        if i_c != 0: sub.set_yticks([]) 

    fig.text(0.05, 0.5, r'GMM component fractions', rotation='vertical', va='center', fontsize=25) 
    fig.text(0.5, 0.025, r'log$\; M_* \;\;[M_\odot]$', ha='center', fontsize=30) 
    fig.subplots_adjust(wspace=0.05, hspace=0.1)
    fig_name = ''.join([UT.doc_dir(), 'figs/GMMcomp_comp_res_impact.pdf'])
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close() 
    return None 


def _GMM_fcomp_res_impact(name, groupfinder=True, n_bootstrap=10, seed=1, silent=True):         
    Cat = Cats.Catalog()
    logM, logSFR, w, censat = Cat.Read(name, keepzeros=True, silent=True)
    if groupfinder: 
        psat = Cat.GroupFinder(name) 
        iscen = (psat < 0.01)
    else: 
        iscen = (censat == 1)
    iscen_nz = iscen & np.invert(Cat.zero_sfr) # SFR > 0 
    iscen_z = iscen & Cat.zero_sfr # SFR == 0 
    assert np.sum(iscen) == np.sum(iscen_nz) + np.sum(iscen_z) # snaity check 

    # fit range
    mmin = 8.4
    mmax = 12.
    if name == 'illustris_100myr':
        dsfr = 0.016
    elif name == 'eagle_100myr':
        dsfr = 0.018
    elif name == 'mufasa_100myr':
        dsfr = 0.182
        mmin = 9.2
    else: 
        raise ValueError

    # sample SFR within the resolution range 
    rand = np.random.RandomState(seed)
    SFR_nz_p = 10**logSFR[iscen_nz] + dsfr * rand.uniform(size=np.sum(iscen_nz))
    SFR_z_p = dsfr * rand.uniform(size=np.sum(iscen_z))
    logM_p = np.concatenate([logM[iscen_nz], logM[iscen_z]]) 
    SFR_p = np.concatenate([SFR_nz_p, SFR_z_p]) 
    logSFR_p = np.log10(SFR_p)
    
    f_zeros, f_sfmss, f_qs, f_other0s, f_other1s= [], [], [], [], []  
    # fit the SFMS using GMM fitting
    fSFMS_p = fstarforms()
    fit_logm_p, fit_logsfr_p = fSFMS_p.fit(logM[iscen_nz], logSFR[iscen_nz], method='gaussmix', 
            fit_range=[mmin, mmax], dlogm=0.2, Nbin_thresh=10, max_comp=3, 
            silent=True) 

    fSFMS = fstarforms()
    fit_logm, fit_logsfr = fSFMS.fit(logM_p, logSFR_p, method='gaussmix', 
            fit_range=[mmin, mmax], dlogm=0.2, Nbin_thresh=10, max_comp=4, 
            silent=True) 
        
    mbin0 = fSFMS._mbins[:,0]
    mbin1 = fSFMS._mbins[:,1]
    gbests = fSFMS._gbests
       
    nmbin = len(mbin0) 
    f_comps = np.zeros((5, nmbin)) # zero, sfms, q, other0, other1
    f_comps_unc = np.zeros((5, nmbin))
    if not silent: print('---------- %s ------------' % name)
    
    for i_m, gbest in zip(range(len(mbin0)), gbests): 
        # calculate the fraction of galaxies have that zero SFR
        mbin_iscen = (logM_p > mbin0[i_m]) & (logM_p < mbin1[i_m])
        mbin_iscen_z = mbin_iscen & (SFR_p == 0.) 
        f_comps[0, i_m] = float(np.sum(mbin_iscen_z))/float(np.sum(mbin_iscen))

        weights_i = gbest.weights_
        i_sfms, i_q, i_int, i_sb = fSFMS._GMM_idcomp(gbest, silent=True)
        
        f_nz = 1. - f_comps[0, i_m]  # multiply by non-zero fraction
        if i_sfms is not None: 
            f_comps[1, i_m] = f_nz * np.sum(weights_i[i_sfms])
        if i_q is not None: 
            f_comps[2, i_m] = f_nz * np.sum(weights_i[i_q])
        if i_int is not None: 
            f_comps[3, i_m] = f_nz * np.sum(weights_i[i_int])
        if i_sb is not None: 
            f_comps[4, i_m] = f_nz * np.sum(weights_i[i_sb])
        
        inmbin = (fit_logm_p > mbin0[i_m]) & (fit_logm_p < mbin1[i_m])
        if np.sum(inmbin) > 0: 
            if np.sum(inmbin) > 1: raise ValueError
            ggg = fSFMS_p._gbests[np.arange(len(fit_logm_p))[inmbin][0]]
            i_sfms_p, i_q_p, i_int_p, i_sb_p = fSFMS_p._GMM_idcomp(ggg, silent=True)
            if i_q_p is not None: 
                ssfr_q = ggg.means_.flatten()[i_q_p] + ggg.covariances_.flatten()[i_q_p]
                if not silent: 
                    print('%f - %f' % (mbin0[i_m], mbin1[i_m]))
                    print('-------- SSFR_q = %f --------' % ssfr_q) 
                if i_int is not None: 
                    if not silent: 
                        print gbest.means_.flatten()[i_int]
                        print f_comps[3, i_m]
                    below = (gbest.means_.flatten()[i_int] < ssfr_q.max() + 0.1) 
                    f_comps[2, i_m] += f_nz * np.sum(weights_i[i_int][below])
                    f_comps[3, i_m] -= f_nz * np.sum(weights_i[i_int][below])
                    if not silent: print f_comps[3, i_m]

        # bootstrap uncertainty 
        X = logSFR_p[mbin_iscen] - logM_p[mbin_iscen] # logSSFRs
        n_best = len(gbest.means_.flatten())

        f_boots = np.zeros((5, n_bootstrap))

        for i_boot in range(n_bootstrap): 
            X_boot = np.random.choice(X.flatten(), size=len(X), replace=True) 
            if name not in ['eagle_inst', 'eagle_100myr', 'mufasa_insta', 'mufasa_100myr']: zero = np.invert(np.isfinite(X_boot))
            else: zero = (np.invert(np.isfinite(X_boot)) | (X_boot <= -99.))

            f_boots[0,i_boot] = float(np.sum(zero))/float(len(X))
            if not silent: print('%f - %f : %f' % (mbin0[i_m], mbin1[i_m], f_boots[0,i_boot]))
                
            gmm_boot = GMix(n_components=n_best)
            gmm_boot.fit(X_boot[np.invert(zero)].reshape(-1,1))
            weights_i = gmm_boot.weights_

            i_sfms, i_q, i_int, i_sb = fSFMS._GMM_idcomp(gmm_boot, silent=True)
        
            f_nonzero = 1. - f_boots[0,i_boot] 
            if i_sfms is not None: 
                f_boots[1,i_boot] = f_nonzero * np.sum(weights_i[i_sfms])
            if i_q is not None: 
                f_boots[2,i_boot] = f_nonzero * np.sum(weights_i[i_q])
            if i_int is not None: 
                f_boots[3,i_boot] = f_nonzero * np.sum(weights_i[i_int])
            if i_sb is not None: 
                f_boots[4,i_boot] = f_nonzero * np.sum(weights_i[i_sb])

            if np.sum(inmbin) > 0: 
                if i_q_p is not None: 
                    if i_int is not None: 
                        below = (gbest.means_.flatten()[i_int] < ssfr_q.max() + 0.1) 
                        f_boots[2,i_boot] += f_nz * np.sum(weights_i[i_int][below])
                        f_boots[3,i_boot] -= f_nz * np.sum(weights_i[i_int][below])

        for i_b in range(f_boots.shape[0]): 
            f_comps_unc[i_b,i_m] = np.std(f_boots[i_b,:]) 
    return 0.5*(mbin0 + mbin1),  f_comps, f_comps_unc 


def _SFMSfit_assess(name, fit_range=None, method='gaussmix'):
    ''' Assess the quality of the SFMS fits by comparing to the actual 
    P(sSFR) in mass bins. 
    '''
    cat = Cats.Catalog() # read in catalog
    _logm, _logsfr, _, censat = cat.Read(name)  
    if name in ['nsa_dickey', 'tinkergroup']: 
        iscen = (censat == 1) # centrals only 
    else:
        psat = cat.GroupFinder(name)
        iscen = ((psat < 0.01) & np.invert(cat.zero_sfr))
    logm = _logm[iscen]
    logsfr = _logsfr[iscen]
    
    # fit the SFMS  
    fSFMS = fstarforms()
    fit_logm, fit_logsfr = fSFMS.fit(logm, logsfr, fit_range=fit_range, method=method) 
    
    n_col = int(np.ceil(float(len(fSFMS._gbests)+1)/3))
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

    for i_m in range(len(fSFMS._gbests)):  
        sub = fig.add_subplot(3, n_col, i_m+2)
    
        # within mass bin 
        mbin_mid = np.mean(fSFMS._mbins[i_m, :]) 
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
            gmm_weights = fSFMS._gbests[i_m].weights_
            gmm_means = fSFMS._gbests[i_m].means_.flatten() 
            gmm_vars = fSFMS._gbests[i_m].covariances_.flatten() 

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

    
    fig_name = ''.join([UT.doc_dir(), 'SFRcomparison.', name, '.png'])
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
    fig_name = ''.join([UT.doc_dir(), 'GMMcomp.test.', name, '.png'])
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close() 
    return None


def _fGMM(name, nosplashback=False, sb_cut='3vir'): 
    if nosplashback and name not in ['nsa_dickey', 'tinkergroup']: 
        fgmm = ''.join([UT.dat_dir(), 'paper1/', 
            'gmmSFSfit.', name, '.gfcentral.nosplbacks.', sb_cut, '.mlim.p'])
        print fgmm
        return fgmm 
    else: 
        return ''.join([UT.dat_dir(), 'paper1/', 'gmmSFSfit.', name, '.gfcentral.mlim.p'])


##############################
# Appendix: previous method 
##############################
def Catalogs_SFR_Mstar_SD14like():
    ''' Compare SFR vs M* relation plotted like in Somerville & Dave 2014 (fits from the different modelers)
    '''
    Cat = Cats.Catalog()
    sims_list = ['illustris', 'eagle', 'mufasa', 'scsam'] # simulations 

    fig = plt.figure(figsize=(5,5))
    sub = fig.add_subplot(111)
    for i_c, sim in enumerate(sims_list):
        cat = '_'.join([sim, 'inst'])
        logMstar, logSFR, weight, censat = Cat.Read(cat)

        lbl = Cat.CatalogLabel(cat)
        if i_c == 0:
            sub.text(0.95, 0.05, 'SFR ['+(lbl.split('[')[-1]).split(']')[0]+']',
                     ha='right', va='bottom', transform=sub.transAxes, fontsize=25)

        if sim == 'illustris': 
            nbins, binsize, fitmin = 12, 0.3, 8.05
            all_or_sf = 'all'
        elif sim == 'eagle': 
            nbins, binsize, fitmin = 12, 0.3, 8.1
            all_or_sf = 'sf'
        elif sim == 'mufasa': 
            nbins, binsize, fitmin = 11, 0.25, 8.525
            all_or_sf = 'all'
        elif sim == 'scsam': 
            nbins, binsize, fitmin = 10, 0.3, 8.7
            all_or_sf = 'sf'

        logSFRfit1 = np.zeros(nbins)
        logSFRfit2 = np.zeros(nbins)
        logMstarfit = np.zeros(nbins)
        for i_b in range(nbins):
            # in stellar mass bin 
            inbin = ((logMstar > fitmin + binsize * i_b) & (logMstar < fitmin + binsize * (i_b+1)))
            logMstarfit[i_b] = fitmin + binsize*(i_b+0.5)
            logSFRfit2[i_b] = np.log10(np.median(10.0**(logSFR[inbin])))
        
            # in stellar mass bin above ssfr > -11
            inbinsf = (inbin & (logSFR-logMstar > -11.0)) 
            logSFRfit1[i_b] = np.log10(np.median(10.0**(logSFR[inbinsf])))

        if all_or_sf == 'all': 
            sub.plot(logMstarfit, logSFRfit2, color='C'+str(i_c+2), 
                    label=lbl.split('[')[0]+' (all galaxies)')
            #sub.scatter(logMstarfit, logSFRfit2, color='C'+str(i_c+2))
        elif all_or_sf == 'sf': 
            sub.plot(logMstarfit, logSFRfit1, color='C'+str(i_c+2), 
                    label=lbl.split('[')[0]+' ($\log\,\mathrm{SSFR} > -11$)')
        sub.set_xlim([8, 12])
        sub.set_ylim([-3, 3])
    sub.legend(loc='upper left', handletextpad=0.5, frameon=False, fontsize=15)
    sub.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', fontsize=25)
    sub.set_ylabel(r'log ( SFR $[M_\odot \, yr^{-1}]$ )', fontsize=25)
    fig.subplots_adjust(wspace=0.2, hspace=0.15)
    fig_name = ''.join([UT.doc_dir(), 'figs/Catalogs_SFR_Mstar_SD14like.pdf'])
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close()
    return None


def Catalogs_SFR_Mstar_testSimpleFits():
    ''' Compare SFR vs M* relation plotting medians for SF galaxies or all galaxies
    '''
    Cat = Cats.Catalog()
    sims_list = ['illustris', 'eagle', 'mufasa', 'scsam'] # simulations 

    f_gmm = lambda name: _fGMM(name)
        
    fig = plt.figure(figsize=(10,5))
    bkgd = fig.add_subplot(111, frameon=False)
    sub1 = fig.add_subplot(121)
    sub2 = fig.add_subplot(122)
    for i_c, sim in enumerate(sims_list):
        cat = '_'.join([sim, 'inst'])
        logMstar, logSFR, weight, censat = Cat.Read(cat)

        lbl = Cat.CatalogLabel(cat)
        if i_c == 0:
            sub2.text(0.95, 0.05, 'SFR ['+(lbl.split('[')[-1]).split(']')[0]+']',
                     ha='right', va='bottom', transform=sub2.transAxes, fontsize=25)

        if sim == 'illustris': 
            nbins, binsize, fitmin = 12, 0.3, 8.05
            all_or_sf = 'all'
        elif sim == 'eagle': 
            nbins, binsize, fitmin = 12, 0.3, 8.1
            all_or_sf = 'sf'
        elif sim == 'mufasa': 
            nbins, binsize, fitmin = 11, 0.25, 8.525
            all_or_sf = 'all'
        elif sim == 'scsam': 
            nbins, binsize, fitmin = 10, 0.3, 8.7
            all_or_sf = 'sf'

        logSFRfit1 = np.zeros(nbins)
        logSFRfit2 = np.zeros(nbins)
        logMstarfit = np.zeros(nbins)
        for i_b in range(nbins):
            # in stellar mass bin 
            inbin = ((logMstar > fitmin + binsize * i_b) & (logMstar < fitmin + binsize * (i_b+1)))
            logMstarfit[i_b] = fitmin + binsize*(i_b+0.5)
            logSFRfit2[i_b] = np.log10(np.median(10.0**(logSFR[inbin])))
        
            # in stellar mass bin above ssfr > -11
            inbinsf = (inbin & (logSFR-logMstar > -11.0)) 
            logSFRfit1[i_b] = np.log10(np.median(10.0**(logSFR[inbinsf])))

        fSFS = pickle.load(open(f_gmm(cat), 'rb'))
        sub2.errorbar(fSFS._fit_logm, fSFS._fit_logsfr, fSFS._fit_err_logssfr, 
                fmt='.C'+str(i_c+2), label='Hahn et al.(2018) \nCentral SFS GMM fit')#, linewidth=0.5, alpha=0.75) 
        if i_c == 0: sub2.legend(loc='upper left', handletextpad=0., frameon=False, fontsize=20) 

        sub1.plot(logMstarfit, logSFRfit2, color='C'+str(i_c+2), 
                label=lbl.split('[')[0])
        sub2.plot(logMstarfit, logSFRfit1, color='C'+str(i_c+2)) 
        sub1.set_xlim([8, 12])
        sub1.set_ylim([-3, 3])
        sub2.set_xlim([8, 12])
        sub2.set_ylim([-3, 3])
    sub1.legend(loc='upper left', handletextpad=0.5, frameon=False, fontsize=20)
    sub1.set_title('All Galaxies', fontsize=20) 
    sub2.set_title('All Galaxies w/ $\log(\mathrm{SSFR}){>}-11$', fontsize=20) 
    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', labelpad=5, fontsize=25) 
    bkgd.set_ylabel(r'median log ( SFR $[M_\odot \, yr^{-1}]$ )', labelpad=5, fontsize=25) 
    fig.subplots_adjust(wspace=0.1)
    fig_name = ''.join([UT.doc_dir(), 'figs/Catalogs_SFR_Mstar_SimpleFitsMedian.pdf'])
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close()
    return None


if __name__=="__main__": 
    #Catalogs_SFR_Mstar()
    #Catalogs_Pssfr()
    #GroupFinder()
    #SFMSfit_example()
    #for tt in ['inst', '100myr']:
    #    Catalog_SFMS_fit(tt)
    #    Catalog_SFMS_fit(tt, nosplashback=True, sb_cut='geha')
    #Catalogs_SFMS_powerlawfit()
    #Catalogs_SFMS_width()
    #Catalog_GMMcomps()
    #Pssfr_GMMcomps(timescale='inst')
    #Pssfr_GMMcomps(timescale='100myr')
    GMMcomp_weights(n_bootstrap=100)
    #GMMcomp_weights(n_bootstrap=10, nosplashback=True, sb_cut='geha')
    #_GMM_comp_test('tinkergroup')
    #_GMM_comp_test('nsa_dickey')
    #rhoSF(cumulative=False)
    #rhoSF(cumulative=True)
    #SMF()
    #fsat()
    #dSFS('powerlaw')
    #dSFS('interpexterp')
    #GMMcomp_weights_res_impact(n_bootstrap=100)
    #Pssfr_res_impact(n_mc=100)
    #Mlim_res_impact(n_mc=100)
    #for c in ['illustris', 'eagle', 'mufasa', 'scsam']: 
    #    for tscale in ['inst', '100myr']:#'10myr', '100myr', '1gyr']: 
    #        _GMM_comp_test(c+'_'+tscale)
    #for c in ['illustris', 'eagle', 'mufasa']:
    #    _SFR_tscales(c)
    #for c in ['mufasa']: #'illustris', 'eagle', 'mufasa', 'scsam']: 
    #    for tscale in ['inst', '100myr']: 
    #        _SFMSfit_assess(c+'_'+tscale, method='gaussmix')
    #_SFMSfit_assess('nsa_dickey', fit_range=(8.4, 9.7), method='gaussmix')
    #_SFMSfit_assess('tinkergroup', fit_range=(9.8, 12.), method='gaussmix')
    #SFRMstar_2Dgmm(n_comp_max=50)
    #Catalogs_SFR_Mstar_SD14like()
    #Catalogs_SFR_Mstar_testSimpleFits()
