'''

IQ paper 4 --- High z galaxies


'''
import numpy as np 
import corner as DFM 
# -- letstalkaboutquench --
from letstalkaboutquench import util as UT
from letstalkaboutquench import catalogs as Cats
from letstalkaboutquench import galprop as Gprop
from letstalkaboutquench.fstarforms import fstarforms
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
mpl.rcParams['legend.frameon'] = False


def dSFS(name, method='interpexterp'): 
    ''' calculate dSFS for high z catalog 
    ''' 
    zlo = [0.5, 1., 1.4, 1.8, 2.2, 2.6]
    zhi = [1., 1.4, 1.8, 2.2, 2.6, 3.0]
    logms, logsfrs = [], [] 
    sfms_fits, dsfss = [], [] 
    for i in range(1,len(zlo)+1): 
        logm, logsfr = readHighz(name, i, keepzeros=False)
        logms.append(logm)
        logsfrs.append(logsfr)
        # fit the SFMSes
        fSFMS = fstarforms()
        sfms_fit = fSFMS.fit(logm, logsfr, 
                method='gaussmix',      # Gaussian Mixture Model fitting 
                fit_range=[8.5, 12.0],  # stellar mass range
                dlogm=0.4,            # stellar mass bins of 0.4 dex
                SSFR_cut= -10.5, 
                Nbin_thresh=100,        # at least 100 galaxies in bin 
                fit_error='bootstrap',  # uncertainty estimate method 
                n_bootstrap=100)        # number of bootstrap bins
        sfms_fits.append(sfms_fit) 
        if method == 'powerlaw': 
            _ = fSFMS.powerlaw(logMfid=10.5) 
            dsfs = fSFMS.d_SFS(logm, logsfr, method=method, silent=False) 
        elif method == 'interpexterp': 
            dsfs = fSFMS.d_SFS(logm, logsfr, method=method, err_thresh=0.2, silent=False) 
        dsfss.append(dsfs) 
        # save d_SFS to file 
        f_hz = fHighz(name, i)
        if 'illustris' in name: 
            if name == 'illustris_10myr': 
                f_sfs = ''.join([f_hz.split('.txt')[0], '.10myr.dsfs.', method, '.txt']) 
            elif name == 'illustris_100myr': 
                f_sfs = ''.join([f_hz.split('.txt')[0], '.100myr.dsfs.', method, '.txt']) 
            elif name == 'illustris_1gyr': 
                f_sfs = ''.join([f_hz.split('.txt')[0], '.1gyr.dsfs.', method, '.txt']) 
            else: raise ValueError
        else: 
            f_sfs = ''.join([f_hz.split('.txt')[0], '.dsfs.', method, '.txt']) 
        if method == 'powerlaw': 
            hdr = '\n'.join(['distance to the SF sequence', 
                'SFMS fit choices: dlogm=0.4, SSFR_cut=-10.5, Nbin_thresh=100', 
                'dSFS choices: logM_fid = 10.5']) 
        elif method == 'interpexterp':
            hdr = '\n'.join(['distance to the SF sequence', 
                'SFMS fit choices: dlogm=0.4, SSFR_cut=-10.5, Nbin_thresh=100', 
                'dSFS choices: interp=True, extrap=True, err_thresh=0.2']) 
        np.savetxt(f_sfs, dsfs, header=hdr) 

    # SFMS overplotted ontop of SFR--M* relation 
    fig = plt.figure(figsize=(12,8))
    bkgd = fig.add_subplot(111, frameon=False)
    for i_z in range(len(zlo)): 
        sub = fig.add_subplot(2,3,i_z+1) 
        sub.scatter(logms[i_z], logsfrs[i_z], color='k', s=1) 
        sub.errorbar(sfms_fits[i_z][0], sfms_fits[i_z][1], sfms_fits[i_z][2], fmt='.C0')
        is_quiescent = ((dsfss[i_z] < -1.) & (dsfss[i_z] != -999.)) 
        sub.scatter(logms[i_z][is_quiescent], logsfrs[i_z][is_quiescent], color='C1', s=1) 
        sub.set_xlim([8.5, 12.]) 
        sub.set_ylim([-3., 4.]) 
        sub.text(0.95, 0.05, '$'+str(zlo[i_z])+'< z <'+str(zhi[i_z])+'$', 
                ha='right', va='bottom', transform=sub.transAxes, fontsize=20)
        if i_z == 0: 
            sub.text(0.05, 0.95, ' '.join(name.upper().split('_')),
                    ha='left', va='top', transform=sub.transAxes, fontsize=20)

    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', labelpad=15, fontsize=25) 
    bkgd.set_ylabel(r'log ( SFR $[M_\odot \, yr^{-1}]$ )', labelpad=15, fontsize=25) 
    fig.subplots_adjust(wspace=0.2, hspace=0.15)
    fig_name = ''.join([UT.doc_dir(), 'highz/figs/', name.lower(), '_dsfs.', method, '.png'])
    fig.savefig(fig_name, bbox_inches='tight')
    return None 


def candels(): 
    ''' SFR -- M* relation of CANDLES galaxies in the 6 redshift bins
    '''
    zlo = [0.5, 1., 1.4, 1.8, 2.2, 2.6]
    zhi = [1., 1.4, 1.8, 2.2, 2.6, 3.0]

    logms, logsfrs = [], [] 
    for i in range(1,len(zlo)+1): 
        logm, logsfr = readHighz('candels', i, keepzeros=False)
        logms.append(logm)
        logsfrs.append(logsfr)
    
    fig = plt.figure(figsize=(12,8))
    bkgd = fig.add_subplot(111, frameon=False)
    for i_z in range(len(zlo)): 
        sub = fig.add_subplot(2,3,i_z+1) 
        DFM.hist2d(logms[i_z], logsfrs[i_z], color='C0', 
                levels=[0.68, 0.95], range=[[7.8, 12.], [-4., 4.]], 
                plot_datapoints=True, fill_contours=False, plot_density=True, 
                ax=sub) 
        sub.set_xlim([8.5, 12.]) 
        sub.set_ylim([-3., 4.]) 
        sub.text(0.95, 0.05, '$'+str(zlo[i_z])+'< z <'+str(zhi[i_z])+'$', 
                ha='right', va='bottom', transform=sub.transAxes, fontsize=20)
        if i_z == 0: 
            sub.text(0.05, 0.95, 'CANDELS', 
                    ha='left', va='top', transform=sub.transAxes, fontsize=25)
    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', labelpad=15, fontsize=25) 
    bkgd.set_ylabel(r'log ( SFR $[M_\odot \, yr^{-1}]$ )', labelpad=15, fontsize=25) 
    fig.subplots_adjust(wspace=0.2, hspace=0.15)
    fig_name = ''.join([UT.doc_dir(), 'highz/figs/candels_sfrM.pdf'])
    fig.savefig(fig_name, bbox_inches='tight')
   

def highz_sfms(name):
    ''' SFMS fits to the SFR -- M* relation of CANDLES galaxies in 
    the 6 redshift bins
    '''
    zlo = [0.5, 1., 1.4, 1.8, 2.2, 2.6]
    zhi = [1., 1.4, 1.8, 2.2, 2.6, 3.0]

    fSFMS = fstarforms()
    logms, logsfrs = [], [] 
    sfms_fits = [] 
    for i in range(1,len(zlo)+1): 
        logm, logsfr = readHighz(name, i, keepzeros=False)
        logms.append(logm)
        logsfrs.append(logsfr)
        # fit the SFMSes
        sfms_fit = fSFMS.fit(logm, logsfr, 
                method='gaussmix',      # Gaussian Mixture Model fitting 
                fit_range=[8.5, 12.0],  # stellar mass range
                dlogm = 0.4,            # stellar mass bins of 0.4 dex
                Nbin_thresh=100,        # at least 100 galaxies in bin 
                fit_error='bootstrap',  # uncertainty estimate method 
                n_bootstrap=100)        # number of bootstrap bins
        sfms_fits.append(sfms_fit) 
    
    # SFMS overplotted ontop of SFR--M* relation 
    fig = plt.figure(figsize=(12,8))
    bkgd = fig.add_subplot(111, frameon=False)
    for i_z in range(len(zlo)): 
        sub = fig.add_subplot(2,3,i_z+1) 
        DFM.hist2d(logms[i_z], logsfrs[i_z], color='C0', 
                levels=[0.68, 0.95], range=[[7.8, 12.], [-4., 4.]], 
                plot_datapoints=True, fill_contours=False, plot_density=True, 
                ax=sub) 
        # plot SFMS fit
        sub.errorbar(sfms_fits[i_z][0], sfms_fits[i_z][1], sfms_fits[i_z][2], fmt='.k')
        sub.set_xlim([8.5, 12.]) 
        sub.set_ylim([-3., 4.]) 
        sub.text(0.95, 0.05, '$'+str(zlo[i_z])+'< z <'+str(zhi[i_z])+'$', 
                ha='right', va='bottom', transform=sub.transAxes, fontsize=20)
        if i_z == 0: 
            sub.text(0.05, 0.95, ' '.join(name.upper().split('_')),
                    ha='left', va='top', transform=sub.transAxes, fontsize=20)

    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', labelpad=15, fontsize=25) 
    bkgd.set_ylabel(r'log ( SFR $[M_\odot \, yr^{-1}]$ )', labelpad=15, fontsize=25) 
    fig.subplots_adjust(wspace=0.2, hspace=0.15)
    fig_name = ''.join([UT.doc_dir(), 'highz/figs/', name.lower(), '_sfms.pdf'])
    fig.savefig(fig_name, bbox_inches='tight')
    
    # SFMS fit redshift evolution 
    fig = plt.figure(figsize=(6,6))
    sub = fig.add_subplot(111)
    for i_z, sfms_fit in enumerate(sfms_fits):  
        sub.fill_between(sfms_fit[0], sfms_fit[1]-sfms_fit[2], sfms_fit[1]+sfms_fit[2], 
                color='C'+str(i_z), linewidth=0.5, alpha=0.75, 
                label='$'+str(zlo[i_z])+'< z <'+str(zhi[i_z])+'$') 
    sub.text(0.95, 0.05, ' '.join(name.upper().split('_')),
            ha='right', va='bottom', transform=sub.transAxes, fontsize=20)
    sub.legend(loc='upper left', prop={'size':15}) 
    sub.set_xlim([8.5, 12.]) 
    sub.set_ylim([-0.5, 4.]) 
    fig_name = ''.join([UT.doc_dir(), 'highz/figs/', name.lower(), '_sfmsz.pdf'])
    fig.savefig(fig_name, bbox_inches='tight')
    return None 


def sfms_comparison(): 
    ''' Compare the SFMS fits among the data and simulation  
    '''
    names = ['candels', 'illustris_10myr', 'illustris_1gyr', 'eagle'] 

    zlo = [0.5, 1., 1.4, 1.8, 2.2, 2.6]
    zhi = [1., 1.4, 1.8, 2.2, 2.6, 3.0]
    
    fSFMS = fstarforms()
    sfms_dict = {} 
    for name in names:  
        sfms_fits = [] 
        for i in range(1,len(zlo)+1): 
            logm, logsfr = readHighz(name, i, keepzeros=False)
            # fit the SFMSes
            sfms_fit = fSFMS.fit(logm, logsfr, 
                    method='gaussmix',      # Gaussian Mixture Model fitting 
                    fit_range=[8.5, 12.0],  # stellar mass range
                    dlogm = 0.4,            # stellar mass bins of 0.4 dex
                    Nbin_thresh=100,        # at least 100 galaxies in bin 
                    fit_error='bootstrap',  # uncertainty estimate method 
                    n_bootstrap=100)        # number of bootstrap bins
            sfms_fits.append(sfms_fit) 
        sfms_dict[name] = sfms_fits
    
    # SFMS overplotted ontop of SFR--M* relation 
    fig = plt.figure(figsize=(12,8))
    bkgd = fig.add_subplot(111, frameon=False)
    for i_z in range(len(zlo)): 
        sub = fig.add_subplot(2,3,i_z+1) 
        # plot SFMS fits
        for i_n, name in enumerate(names):  
            if name == 'candels': colour = 'k'
            else: colour = 'C'+str(i_n) 
            sub.fill_between(sfms_dict[name][i_z][0], 
                    sfms_dict[name][i_z][1] - sfms_dict[name][i_z][2], 
                    sfms_dict[name][i_z][1] + sfms_dict[name][i_z][2], 
                    color=colour, alpha=0.75, label=' '.join(name.upper().split('_')))
        sub.set_xlim([8.5, 12.]) 
        sub.set_ylim([-1., 4.]) 
        sub.text(0.95, 0.05, '$'+str(zlo[i_z])+'< z <'+str(zhi[i_z])+'$', 
                ha='right', va='bottom', transform=sub.transAxes, fontsize=20)
        if i_z == 0: 
            sub.legend(loc='upper left', handletextpad=0.5, prop={'size': 15}) 
            #sub.text(0.05, 0.95, ' '.join(name.upper().split('_')),
            #        ha='left', va='top', transform=sub.transAxes, fontsize=20)
    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', labelpad=15, fontsize=25) 
    bkgd.set_ylabel(r'log ( SFR $[M_\odot \, yr^{-1}]$ )', labelpad=15, fontsize=25) 
    fig.subplots_adjust(wspace=0.2, hspace=0.15)
    fig_name = ''.join([UT.doc_dir(), 'highz/figs/sfms_comparison.pdf'])
    fig.savefig(fig_name, bbox_inches='tight')
    return None


def readHighz(name, i_z, keepzeros=False): 
    ''' read in CANDELS, Illustris, or EAGLE M* and SFR data in the 
    iz redshift bin 
    '''
    f_data = fHighz(name, i_z)
    name = name.lower() 
    if name == 'candels': 
        _, ms, sfr = np.loadtxt(f_data, skiprows=2, unpack=True) # z, M*, SFR
        logms = np.log10(ms) 
        logsfr = np.log10(sfr) 
        notzero = (sfr != 0.)
    elif 'illustris' in name: 
        if name == 'illustris_10myr': 
            ms, sfr, _, _ = np.loadtxt(f_data, skiprows=2, unpack=True) # M*, SFR 10Myr, SFR 1Gyr 
        elif name == 'illustris_100myr': 
            ms, _, _, sfr = np.loadtxt(f_data, skiprows=2, unpack=True) # M*, SFR 10Myr, SFR 1Gyr 
        elif name == 'illustris_1gyr': 
            ms, _, sfr, _ = np.loadtxt(f_data, skiprows=2, unpack=True) # M*, SFR 10Myr, SFR 1Gyr 
        logms = np.log10(ms) 
        logsfr = np.log10(sfr) 
        notzero = (sfr != 0.)
    elif name == 'eagle': 
        ms, sfr = np.loadtxt(f_data, skiprows=2, unpack=True) # M*, SFR instantaneous
        logms = ms
        logsfr = sfr 
        notzero = np.isfinite(sfr)

    if not keepzeros: 
        return logms[notzero], logsfr[notzero]
    else: 
        return logms, logsfr, notzero


def fHighz(name, i_z): 
    ''' High z project file names
    '''
    dat_dir = ''.join([UT.dat_dir(), 'highz/'])
    if name == 'candels': 
        f_data = ''.join([dat_dir, 'CANDELS/CANDELS_z', str(i_z), '.txt']) 
    elif 'illustris' in name: 
        f_data = ''.join([dat_dir, 'Illustris/Illustris_z', str(i_z), '.txt']) 
    elif name == 'eagle': 
        f_data = ''.join([dat_dir, 'EAGLE/EAGLE_z', str(i_z), '.txt']) 
    else: 
        raise NotImplementedError
    return f_data


if __name__=="__main__": 
    #sfms_comparison()
    for name in ['illustris_100myr']: #'eagle', 'illustris_10myr', 'illustris_1gyr']:
        for method in ['interpexterp', 'powerlaw']:  
            #highz_sfms(name)
            dSFS(name, method=method) 
