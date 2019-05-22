'''

IQ paper 4 --- High z galaxies


'''
import os
import pickle
import numpy as np 
import corner as DFM 
from scipy.stats import multivariate_normal as MNorm
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
        fSFS = highzSFSfit(name, i_z)
        sfms_fit = [fSFS._fit_logm, fSFS._fit_logsfr, fSFS._fit_err_logssfr]
        sfms_fits.append(sfms_fit) 
        if method == 'powerlaw': 
            _ = fSFS.powerlaw(logMfid=10.5) 
            dsfs = fSFS.d_SFS(logm, logsfr, method=method, silent=False) 
        elif method == 'interpexterp': 
            dsfs = fSFS.d_SFS(logm, logsfr, method=method, err_thresh=0.2, silent=False) 
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
   

def highz_sfms(name, noise=False, seed=1):
    ''' SFMS fits to the SFR -- M* relation of CANDLES galaxies in 
    the 6 redshift bins
    '''
    zlo = [0.5, 1., 1.4, 1.8, 2.2, 2.6]
    zhi = [1., 1.4, 1.8, 2.2, 2.6, 3.0]

    logms, logsfrs = [], [] 
    sfms_fits = [] 
    for i in range(1,len(zlo)+1): 
        logm, logsfr = readHighz(name, i, keepzeros=False, noise=noise, seed=seed)
        logms.append(logm)
        logsfrs.append(logsfr)
        # fit the SFMSes
        fSFS = highzSFSfit(name, i, noise=noise, seed=seed)
        sfms_fit = [fSFS._fit_logm, fSFS._fit_logsfr, fSFS._fit_err_logssfr]
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
    if not noise: fig_name = os.path.join(UT.doc_dir(), 'highz', 'figs', '%s%s' % (name.lower(), '_sfms.pdf'))
    else: fig_name = os.path.join(UT.doc_dir(), 'highz', 'figs', '%s%s' % (name.lower(), '_wnoise_sfms.pdf'))
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
    if not noise: fig_name = os.path.join(UT.doc_dir(), 'highz', 'figs', '%s%s' % (name.lower(), '_sfmsz.pdf'))
    else: fig_name = os.path.join(UT.doc_dir(), 'highz', 'figs', '%s%s' % (name.lower(), '_wnoise_sfmsz.pdf'))
    fig.savefig(fig_name, bbox_inches='tight')
    return None 


def pssfr(name, i_z, noise=False, seed=1): 
    zlo = [0.5, 1., 1.4, 1.8, 2.2, 2.6]
    zhi = [1., 1.4, 1.8, 2.2, 2.6, 3.0]

    logm, logsfr = readHighz(name, i_z, keepzeros=False, noise=noise, seed=seed)
    logssfr = logsfr - logm 

    # fit the SFMS
    fSFS = highzSFSfit(name, i_z, noise=noise, seed=seed)
    mbins = fSFS._mbins[fSFS._mbins_sfs]
    nmbin = np.sum(fSFS._mbins_sfs)
    nrow, ncol = 2, int(np.ceil(0.5*nmbin))

    fig = plt.figure(figsize=(5*ncol,5*nrow))
    bkgd = fig.add_subplot(1,1,1, frameon=False)
    for imbin in range(nmbin): 
        inmbin = ((logm > mbins[imbin][0]) & (logm <= mbins[imbin][1]))
        
        sub = fig.add_subplot(nrow, ncol, imbin+1)
        _ = sub.hist(logssfr[inmbin], bins=40, 
                range=[-14., -8.], density=True, histtype='stepfilled', 
                color='k', alpha=0.25, linewidth=1.75)
    
        i_mbin = np.where((fSFS._fit_logm > mbins[imbin][0]) & (fSFS._fit_logm < mbins[imbin][1]))[0][0]
        gmm_ws = fSFS._gbests[i_mbin].weights_.flatten()
        gmm_mus = fSFS._gbests[i_mbin].means_.flatten()
        gmm_vars = fSFS._gbests[i_mbin].covariances_.flatten()
        icomps = fSFS._GMM_idcomp(fSFS._gbests[i_mbin], SSFR_cut=-11.)
        isfs = icomps[0]
        
        x_ssfr = np.linspace(-14., -8, 100)
        for icomp in range(len(gmm_mus)):  
            sub.plot(x_ssfr, gmm_ws[icomp] * MNorm.pdf(x_ssfr, gmm_mus[icomp], gmm_vars[icomp]), c='k', lw=0.75, ls=':') 
        sub.plot(x_ssfr, gmm_ws[isfs] * MNorm.pdf(x_ssfr, gmm_mus[isfs], gmm_vars[isfs]), c='b', lw=1, ls='-') 
        sub.legend(loc='upper left', prop={'size':15}) 
        sub.set_xlim([-13.6, -8.]) 
        
        if imbin == 0:
            _name = ' '.join(name.upper().split('_'))+'\n $'+str(zlo[i_z-1])+'< z <'+str(zhi[i_z-1])+'$'
            sub.text(0.05, 0.95, _name,
                    ha='left', va='top', transform=sub.transAxes, fontsize=20)
        sub.set_title(str(round(mbins[imbin][0],1))+'$<$ log $M_*$ $<$'+str(round(mbins[imbin][1],1))+'', 
                    fontsize=25)

    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel('log$(\; \mathrm{SSFR}\; [\mathrm{yr}^{-1}]\;)$', labelpad=5, fontsize=25) 
    bkgd.set_ylabel('$p\,(\;\mathrm{log}\; \mathrm{SSFR}\; [\mathrm{yr}^{-1}]\;)$', 
            labelpad=5, fontsize=25)
    #fig.subplots_adjust(wspace=0.1, hspace=0.075)
    if not noise: fig_name = ''.join([UT.doc_dir(), 'highz/figs/', name.lower(), '_z', str(i_z), '_pssfr.pdf'])
    else: fig_name = ''.join([UT.doc_dir(), 'highz/figs/', name.lower(), '_z', str(i_z), '_wnoise_pssfr.pdf'])
    fig.savefig(fig_name, bbox_inches='tight')
    return None 


def highzSFSfit(name, i_z, noise=False, seed=1, overwrite=False): 
    f_highz = fHighz(name, i_z, noise=noise, seed=seed)
    f_sfs =  ''.join([f_highz.rsplit('/', 1)[0], '/sfs_fit', f_highz.rsplit('/', 1)[1].rsplit('.txt',1)[0], '.p']) 
    
    if os.path.isfile(f_sfs) and not overwrite: 
        fSFS = pickle.load(open(f_sfs, 'rb'))
    else: 
        logm, logsfr = readHighz(name, i_z, keepzeros=False, noise=noise, seed=seed)
        logssfr = logsfr - logm 

        # fit the SFMS
        fSFS = fstarforms()
        sfms_fit = fSFS.fit(logm, logsfr, 
                method='gaussmix',      # Gaussian Mixture Model fitting 
                fit_range=[8.5, 12.0],  # stellar mass range
                dlogm = 0.4,            # stellar mass bins of 0.4 dex
                Nbin_thresh=100,        # at least 100 galaxies in bin 
                fit_error='bootstrap',  # uncertainty estimate method 
                n_bootstrap=100)        # number of bootstrap bins
    
        pickle.dump(fSFS, open(f_sfs, 'wb'))
    return fSFS


def readHighz(name, i_z, keepzeros=False, noise=False, seed=1): 
    ''' read in CANDELS, Illustris, or EAGLE M* and SFR data in the 
    iz redshift bin 
    '''
    f_data = fHighz(name, i_z)
    name = name.lower() 
    if not noise: 
        if name == 'candels': 
            _, logms, logsfr = np.loadtxt(f_data, skiprows=2, unpack=True) # z, M*, SFR
            notzero = np.isfinite(logsfr)
        elif 'illustris' in name: 
            if name == 'illustris_10myr': isfr = 1
            elif name == 'illustris_100myr': isfr = 3 
            elif name == 'illustris_1gyr': isfr = 2 
            ms, sfr = np.loadtxt(f_data, skiprows=2, unpack=True, usecols=[0, isfr]) # M*, SFR 10Myr, SFR 1Gyr 
            logms = np.log10(ms) 
            logsfr = np.log10(sfr) 
            notzero = (sfr != 0.)
        elif name == 'tng':
            logms, logsfr = np.loadtxt(f_data, skiprows=2, unpack=True) # log M*, log SFR 
            notzero = np.isfinite(logsfr)
        elif name == 'eagle': 
            ms, sfr = np.loadtxt(f_data, skiprows=2, unpack=True) # M*, SFR instantaneous
            logms = ms
            logsfr = np.log10(sfr) 
            notzero = (sfr != 0.)
        elif 'sam-light' in name: 
            _, ms, sfr = np.loadtxt(f_data, skiprows=2, unpack=True) 
            logms = np.log10(ms)
            logsfr = np.log10(sfr)
            notzero = (sfr != 0.)  
        elif name == 'simba': 
            ms, sfr = np.loadtxt(f_data, skiprows=2, unpack=True) # M*, SFR instantaneous
            logms = np.log10(ms)
            logsfr = np.log10(sfr) 
            notzero = (sfr != 0.)
    else: 
        f_data = fHighz(name, i_z, noise=True, seed=seed) 
        logms, logsfr = np.loadtxt(f_data, unpack=True, usecols=[0,1], skiprows=2) 
        notzero = np.isfinite(logsfr)

    if not keepzeros: 
        return logms[notzero], logsfr[notzero]
    else: 
        return logms, logsfr, notzero 


def fHighz(name, i_z, noise=False, seed=1): 
    ''' High z project file names
    '''
    dat_dir = ''.join([UT.dat_dir(), 'highz/'])
    if name == 'candels': 
        f_data = ''.join([dat_dir, 'CANDELS/CANDELS_Iyer_z', str(i_z), '.txt']) 
    elif 'illustris' in name: 
        f_data = ''.join([dat_dir, 'Illustris/Illustris_z', str(i_z), '.txt']) 
    elif name == 'tng': 
        f_data = ''.join([dat_dir, 'Illustris/IllustrisTNG_z', str(i_z), '.txt']) 
    elif name == 'eagle': 
        f_data = ''.join([dat_dir, 'EAGLE/EAGLE_z', str(i_z), '.txt']) 
    elif name == 'sam-light-full': # SAM light cone full 
        f_data = ''.join([dat_dir, 'SAM_lightcone/SAMfull_z', str(i_z), '.txt'])
    elif name == 'sam-light-slice': # SAM light cone dz=0.01 slice around the median redshift 
        f_data = ''.join([dat_dir, 'SAM_lightcone/SAMslice_z', str(i_z), '.txt'])
    elif name == 'simba': 
        f_data = ''.join([dat_dir, 'SIMBA/SIMBA_z', str(i_z), '.txt'])
    else: 
        raise NotImplementedError

    if noise: 
        f_data = f_data.replace('.txt', '.wnoise.seed%i.txt' % seed)
    return f_data


def add_uncertainty(name, i_z, seed=1): 
    ''' add in measurement uncertainties for SFR and M*. sigma_logSFR = 0.33 dex and 
    sigma_logM* = 0.07 dex. This is done in the simplest way possible --- i.e. add 
    gaussian noise 
    '''
    sig_logsfr = 0.33
    sig_logms = 0.07

    np.random.seed(seed)
    # read in log M* and log SFR 
    logms, logsfr, notzero = readHighz(name, i_z, keepzeros=True)

    if np.sum(~notzero) == 0:
        dlogsfr = sig_logsfr * np.random.randn(len(logsfr))
        logsfr_new = logsfr + dlogsfr
        
        dlogms = sig_logms * np.random.randn(len(logms))
        logms_new = logms + dlogms
    else: 
        # for now, add noise to non-zero SFRs but leaver zero SFRs alone. 
        dlogsfr = sig_logsfr * np.random.randn(len(logsfr))
        logsfr_new = np.zeros(len(logsfr))
        logsfr_new[notzero] = logsfr[notzero] + dlogsfr[notzero]
        logsfr_new[~notzero] = logsfr[~notzero]
        
        dlogms = sig_logms * np.random.randn(len(logms))
        logms_new = logms + dlogms

    # save to file 
    fnew = fHighz(name, i_z, noise=True, seed=seed) 
    hdr = 'Gaussian noise with sigma_logSFR = 0.33 dex and sigma_M* = 0.07 dex added to data\n logM*, logSFR' 
    np.savetxt(fnew, np.array([logms_new, logsfr_new]).T, delimiter='\t', fmt='%.5f %.5f', header=hdr) 
    return None 

################################################
# figures: SFS 
################################################
def SFR_Mstar_comparison(noise=False, seed=1):  
    ''' Compare the SFS fits among the data and simulation  
    '''
    names = ['sam-light-slice', 'eagle', 'illustris_100myr', 'tng', 'simba', 'candels']
    lbls = ['SC-SAM', 'EAGLE', 'Illustris', 'Illustris TNG', 'SIMBA', 'CANDELS'] 
    zlo = [0.5, 1., 1.4, 1.8, 2.2, 2.6]
    zhi = [1., 1.4, 1.8, 2.2, 2.6, 3.0]
    
    # SFMS overplotted ontop of SFR--M* relation 
    fig = plt.figure(figsize=(18,18))
    bkgd = fig.add_subplot(111, frameon=False)
    for i_z in range(len(zlo)): 
        for i_n, name in enumerate(names):  # plot SFMS fits
            sub = fig.add_subplot(6,6,i_z*6+i_n+1) 

            # fit SFR-M* 
            if name != 'candels': 
                logm, logsfr = readHighz(name, i_z+1, keepzeros=False, noise=noise, seed=seed)
                fSFS = highzSFSfit(name, i_z+1, noise=noise, seed=seed) # fit the SFMSes
            else: 
                logm, logsfr = readHighz(name, i_z+1, keepzeros=False, noise=False)
                fSFS = highzSFSfit(name, i_z+1, noise=False) # fit the SFMSes
            sfms_fit = [fSFS._fit_logm, fSFS._fit_logsfr, fSFS._fit_err_logssfr]

            DFM.hist2d(logm, logsfr, color='C0', 
                    levels=[0.68, 0.95], range=[[7.8, 12.], [-4., 4.]], 
                    plot_datapoints=True, fill_contours=False, plot_density=True, 
                    ax=sub) 
            # plot SFMS fit
            sub.errorbar(sfms_fit[0], sfms_fit[1], sfms_fit[2], fmt='.k')
            sub.set_xlim([8.5, 12.]) 
            sub.set_ylim([-3., 4.]) 
            if i_z < 5: sub.set_xticklabels([]) 
            if i_n != 0: sub.set_yticklabels([]) 
            if i_n == 5: sub.text(0.95, 0.05, '$'+str(zlo[i_z])+'< z <'+str(zhi[i_z])+'$', 
                    ha='right', va='bottom', transform=sub.transAxes, fontsize=25)
            if i_z == 0: sub.set_title(lbls[i_n], fontsize=25) 
    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', labelpad=15, fontsize=25) 
    bkgd.set_ylabel(r'log ( SFR $[M_\odot \, yr^{-1}]$ )', labelpad=15, fontsize=25) 
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    if noise: 
        fig_name = ''.join([UT.doc_dir(), 'highz/figs/sfr_mstar_comparison_wnoise_seed%i.pdf' % seed])
    else: 
        fig_name = ''.join([UT.doc_dir(), 'highz/figs/sfr_mstar_comparison.pdf'])
    fig.savefig(fig_name, bbox_inches='tight')
    return None


def SFS_comparison(noise=False, seed=1): 
    ''' Compare the SFS fits among the data and simulation  
    '''
    names = ['sam-light-slice', 'eagle', 'illustris_100myr', 'tng', 'simba', 'candels']
    lbls = ['SC-SAM', 'EAGLE', 'Illustris', 'Illustris TNG', 'SIMBA', 'CANDELS'] 
    zlo = [0.5, 1., 1.4, 1.8, 2.2, 2.6]
    zhi = [1., 1.4, 1.8, 2.2, 2.6, 3.0]
    
    sfms_dict = {} 
    for name in names:  
        sfms_fits = [] 
        for i in range(1,len(zlo)+1): 
            if name != 'candels': 
                logm, logsfr = readHighz(name, i, keepzeros=False, noise=noise, seed=seed)
                fSFS = highzSFSfit(name, i, noise=noise, seed=seed) # fit the SFMSes
            else: 
                logm, logsfr = readHighz(name, i, keepzeros=False, noise=False)
                fSFS = highzSFSfit(name, i, noise=False) # fit the SFMSes
            sfms_fit = [fSFS._fit_logm, fSFS._fit_logsfr, fSFS._fit_err_logssfr]
            sfms_fits.append(sfms_fit) 
        sfms_dict[name] = sfms_fits
    
    # SFMS overplotted ontop of SFR--M* relation 
    fig = plt.figure(figsize=(12,8))
    bkgd = fig.add_subplot(111, frameon=False)
    for i_z in range(len(zlo)): 
        sub = fig.add_subplot(2,3,i_z+1) 

        plts = []
        for i_n, name in enumerate(names):  # plot SFMS fits
            if name == 'candels': 
                _plt = sub.errorbar(sfms_dict[name][i_z][0], 
                        sfms_dict[name][i_z][1], yerr=sfms_dict[name][i_z][2], fmt='.k')
            else: 
                colour = 'C'+str(i_n) 
                _plt = sub.fill_between(sfms_dict[name][i_z][0], 
                        sfms_dict[name][i_z][1] - sfms_dict[name][i_z][2], 
                        sfms_dict[name][i_z][1] + sfms_dict[name][i_z][2], 
                        color='C%i' % i_n, alpha=0.75, linewidth=0.)
            plts.append(_plt) 
        sub.set_xlim([8.5, 12.]) 
        sub.set_ylim([-1., 4.]) 
        if i_z < 3: sub.set_xticklabels([]) 
        if i_z not in [0, 3]: sub.set_yticklabels([]) 
        sub.text(0.95, 0.05, '$'+str(zlo[i_z])+'< z <'+str(zhi[i_z])+'$', 
                ha='right', va='bottom', transform=sub.transAxes, fontsize=20)
        if i_z == 0: 
            sub.legend(plts[:3], lbls[:3], loc='upper left', handletextpad=0.5, prop={'size': 17}) 
        elif i_z == 1: 
            sub.legend(plts[3:], lbls[3:], loc='upper left', handletextpad=0.5, prop={'size': 17}) 
            #sub.text(0.05, 0.95, ' '.join(name.upper().split('_')),
            #        ha='left', va='top', transform=sub.transAxes, fontsize=20)
    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', labelpad=15, fontsize=25) 
    bkgd.set_ylabel(r'log ( SFR $[M_\odot \, yr^{-1}]$ )', labelpad=15, fontsize=25) 
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    if noise: 
        fig_name = ''.join([UT.doc_dir(), 'highz/figs/sfs_comparison_wnoise_seed%i.pdf' % seed])
    else: 
        fig_name = ''.join([UT.doc_dir(), 'highz/figs/sfs_comparison.pdf'])
    fig.savefig(fig_name, bbox_inches='tight')
    return None


def SFS_zevo_comparison(noise=False, seed=1): 
    ''' Compare the SFMS fits among the data and simulation  
    '''
    names = ['sam-light-slice', 'eagle', 'illustris_100myr', 'tng', 'simba', 'candels']
    lbls = ['SC-SAM', 'EAGLE', 'Illustris', 'Illustris TNG', 'SIMBA', 'CANDELS'] 
    zlo = [0.5, 1., 1.4, 1.8, 2.2, 2.6]
    zhi = [1., 1.4, 1.8, 2.2, 2.6, 3.0]
    zlbls = ['$0.5 < z < 1.0$', '$1.0 < z < 1.4$', '$1.4 < z < 1.8$', '$1.8 < z < 2.2$', '$2.2 < z < 2.6$', '$2.6 < z < 3.0$']
    
    sfms_dict = {} 
    for name in names:  
        sfms_fits = [] 
        for i in range(1,len(zlo)+1): 
            if name != 'candels': 
                logm, logsfr = readHighz(name, i, keepzeros=False, noise=noise, seed=seed)
                fSFS = highzSFSfit(name, i, noise=noise, seed=seed) # fit the SFMSes
            else: 
                logm, logsfr = readHighz(name, i, keepzeros=False, noise=False)
                fSFS = highzSFSfit(name, i, noise=False) # fit the SFMSes
            sfms_fit = [fSFS._fit_logm, fSFS._fit_logsfr, fSFS._fit_err_logssfr]
            sfms_fits.append(sfms_fit) 
        sfms_dict[name] = sfms_fits
    
    # SFMS overplotted ontop of SFR--M* relation 
    fig = plt.figure(figsize=(12,8))
    bkgd = fig.add_subplot(111, frameon=False)

    for i_n, name in enumerate(names): 
        sub = fig.add_subplot(2,3,i_n+1) 
        plts = []
        for i_z in range(len(zlo)): 
            _plt = sub.fill_between(sfms_dict[name][i_z][0], 
                    sfms_dict[name][i_z][1] - sfms_dict[name][i_z][2], 
                    sfms_dict[name][i_z][1] + sfms_dict[name][i_z][2], 
                    color='C%i' % i_z, alpha=0.75, linewidth=0.)
            plts.append(_plt) 
        sub.set_xlim([8.5, 12.]) 
        sub.set_ylim([-1., 4.]) 
        if i_n < 3: sub.set_xticklabels([]) 
        if i_n not in [0, 3]: sub.set_yticklabels([]) 
        sub.text(0.95, 0.05, lbls[i_n], ha='right', va='bottom', transform=sub.transAxes, fontsize=20)
        if i_n == 0: 
            sub.legend(plts[:3], zlbls[:3], loc='upper left', handletextpad=0.5, prop={'size': 17}) 
        elif i_n == 1: 
            sub.legend(plts[3:], zlbls[3:], loc='upper left', handletextpad=0.5, prop={'size': 17}) 
    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', labelpad=15, fontsize=25) 
    bkgd.set_ylabel(r'log ( SFR $[M_\odot \, yr^{-1}]$ )', labelpad=15, fontsize=25) 
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    if noise: 
        fig_name = ''.join([UT.doc_dir(), 'highz/figs/sfs_zevo_comparison_wnoise_seed%i.pdf' % seed])
    else: 
        fig_name = ''.join([UT.doc_dir(), 'highz/figs/sfs_zevo_comparison.pdf'])
    fig.savefig(fig_name, bbox_inches='tight')
    return None

################################################
# figures: QF 
################################################
def fcomp(name, i_z, noise=False, seed=1):
    ''' derive quiescent fraction from GMM best-fit. quiescent fraction defined as all components below SFS 
    '''
    logm, logsfr, nonzero = readHighz(name, i_z, keepzeros=True, noise=noise, seed=seed)
    fSFS = highzSFSfit(name, i_z, noise=noise, seed=seed)
    
    # M* bins where SFS is reasonably fit 
    mbin0 = fSFS._mbins[fSFS._mbins_nbinthresh,0]
    mbin1 = fSFS._mbins[fSFS._mbins_nbinthresh,1]
    nmbin = len(mbin0) 

    gbests = fSFS._gbests # best fit GMM
    i_sfss, i_qs, i_ints, i_sbs = fSFS._GMM_compID(gbests, dev_thresh=0.5)

    nmbin = len(mbin0) 
    f_comps = np.zeros((5, nmbin)) # zero, sfms, q, other0, other1
    for i_m, gbest in zip(range(nmbin), gbests): 
        # calculate the fraction of galaxies have that zero SFR
        inmbin      = (logm > mbin0[i_m]) & (logm < mbin1[i_m]) # within bin 
        inmbin_z    = inmbin & ~nonzero # has SFR = 0 
        f_comps[0, i_m] = float(np.sum(inmbin_z))/float(np.sum(inmbin))

        weights_i = gbest.weights_

        i_sfs = i_sfss[i_m]
        i_q = i_qs[i_m]
        i_int = i_ints[i_m]
        i_sb = i_sbs[i_m]

        f_nz = 1. - f_comps[0, i_m]  # multiply by non-zero fraction
        if i_sfs is not None: 
            f_comps[1, i_m] = f_nz * np.sum(weights_i[i_sfs])
        if i_q is not None: 
            f_comps[2, i_m] = f_nz * np.sum(weights_i[i_q])
        if i_int is not None: 
            f_comps[3, i_m] = f_nz * np.sum(weights_i[i_int])
        if i_sb is not None: 
            f_comps[4, i_m] = f_nz * np.sum(weights_i[i_sb])

    return 0.5*(mbin0 + mbin1), f_comps


def QF(name, i_z, noise=False, seed=1):
    ''' derive quiescent fraction from GMM best-fit. quiescent fraction defined as all components below SFS 
    '''
    mmid, fcomps = fcomp(name, i_z, noise=noise, seed=seed) 
    f_Q = fcomps[0,:] + fcomps[2] + fcomps[3]
    return mmid, f_Q 


def fcomp_comparison(noise=False, seed=1): 
    '''
    '''
    names = ['sam-light-slice', 'eagle', 'illustris_100myr', 'tng', 'simba', 'candels']
    lbls = ['SC-SAM', 'EAGLE', 'Illustris', 'Illustris TNG', 'SIMBA', 'CANDELS'] 
    zlo = [0.5, 1., 1.4, 1.8, 2.2, 2.6]
    zhi = [1., 1.4, 1.8, 2.2, 2.6, 3.0]
    
    # SFMS overplotted ontop of SFR--M* relation 
    fig = plt.figure(figsize=(18,18))
    bkgd = fig.add_subplot(111, frameon=False)
    for i_z in range(len(zlo)): 
        for i_n, name in enumerate(names):  # plot SFMS fits
            sub = fig.add_subplot(6,6,i_z*6+i_n+1) 
            if name != 'candels': mmid, f_comps = fcomp(name, i_z+1, noise=noise, seed=seed)
            else: mmid, f_comps = fcomp(name, i_z+1, noise=False)
            
            f_zero, f_sfms, f_q, f_other0, f_other1 = list(f_comps)
            
            sub.fill_between(mmid, np.zeros(len(mmid)), f_zero, # SFR = 0 
                    linewidth=0, color='C3') 
            sub.fill_between(mmid, f_zero, f_zero+f_q,              # Quenched
                    linewidth=0, color='C1') 
            sub.fill_between(mmid, f_zero+f_q, f_zero+f_q+f_other0,   # other0
                    linewidth=0, color='C2') 
            sub.fill_between(mmid, f_zero+f_q+f_other0, f_zero+f_q+f_other0+f_sfms, # SFMS 
                    linewidth=0, color='C0') 
            sub.fill_between(mmid, f_zero+f_q+f_other0+f_sfms, f_zero+f_q+f_other0+f_sfms+f_other1, # star-burst 
                    linewidth=0, color='C4') 

            sub.set_xlim([8.5, 12.]) 
            sub.set_ylim([0., 1.]) 
            if i_z < 5: sub.set_xticklabels([]) 
            if i_n != 0: sub.set_yticklabels([]) 
            if i_n == 5: sub.text(0.95, 0.05, '$'+str(zlo[i_z])+'< z <'+str(zhi[i_z])+'$', 
                    ha='right', va='bottom', transform=sub.transAxes, fontsize=25)
            if i_z == 0: sub.set_title(lbls[i_n], fontsize=25) 
    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', labelpad=15, fontsize=25) 
    bkgd.set_ylabel(r'log ( SFR $[M_\odot \, yr^{-1}]$ )', labelpad=15, fontsize=25) 
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    if noise: 
        fig_name = ''.join([UT.doc_dir(), 'highz/figs/fcomp_comparison_wnoise_seed%i.pdf' % seed])
    else: 
        fig_name = ''.join([UT.doc_dir(), 'highz/figs/fcomp_comparison.pdf'])
    fig.savefig(fig_name, bbox_inches='tight')
    return None


def QF_comparison(noise=False, seed=1): 
    ''' Compare the QF derived from GMMs among the data and simulation  
    '''
    names = ['sam-light-slice', 'eagle', 'illustris_100myr', 'tng', 'simba', 'candels']
    lbls = ['SC-SAM', 'EAGLE', 'Illustris', 'Illustris TNG', 'SIMBA', 'CANDELS'] 
    zlo = [0.5, 1., 1.4, 1.8, 2.2, 2.6]
    zhi = [1., 1.4, 1.8, 2.2, 2.6, 3.0]
    
    fq_dict = {} 
    for name in names:  
        fqs = [] 
        for i in range(1,len(zlo)+1): 
            if name != 'candels': marr, fq = QF(name, i, noise=noise, seed=seed)
            else: marr, fq = QF(name, i, noise=False)
            fqs.append([marr, fq]) 
        fq_dict[name] = fqs
    
    fig = plt.figure(figsize=(12,8))
    bkgd = fig.add_subplot(111, frameon=False)
    for i_z in range(len(zlo)): 
        sub = fig.add_subplot(2,3,i_z+1) 

        plts = []
        for i_n, name in enumerate(names):  # plot fQ fits
            if name == 'candels': colour = 'k'
            else: colour = 'C'+str(i_n) 
            print fq_dict[name][i_z][0], fq_dict[name][i_z][1]
            _plt, = sub.plot(fq_dict[name][i_z][0], fq_dict[name][i_z][1], color=colour)
            plts.append(_plt) 
        sub.set_xlim([8.5, 12.]) 
        sub.set_ylim([0., 1.]) 
        if i_z < 3: sub.set_xticklabels([]) 
        if i_z not in [0, 3]: sub.set_yticklabels([]) 
        sub.text(0.95, 0.05, '$'+str(zlo[i_z])+'< z <'+str(zhi[i_z])+'$', 
                ha='right', va='bottom', transform=sub.transAxes, fontsize=20)
        if i_z == 0: 
            sub.legend(plts[:3], lbls[:3], loc='upper left', handletextpad=0.5, prop={'size': 17}) 
        elif i_z == 1: 
            sub.legend(plts[3:], lbls[3:], loc='upper left', handletextpad=0.5, prop={'size': 17}) 
            #sub.text(0.05, 0.95, ' '.join(name.upper().split('_')),
            #        ha='left', va='top', transform=sub.transAxes, fontsize=20)
    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', labelpad=15, fontsize=25) 
    bkgd.set_ylabel(r'log ( SFR $[M_\odot \, yr^{-1}]$ )', labelpad=15, fontsize=25) 
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    if noise: 
        fig_name = ''.join([UT.doc_dir(), 'highz/figs/fq_comparison_wnoise_seed%i.pdf' % seed])
    else: 
        fig_name = ''.join([UT.doc_dir(), 'highz/figs/fq_comparison.pdf'])
    fig.savefig(fig_name, bbox_inches='tight')
    return None


def QF_zevo_comparison(noise=False, seed=1): 
    ''' Compare the QF derived from GMMs among the data and simulation  
    '''
    names = ['sam-light-slice', 'eagle', 'illustris_100myr', 'tng', 'simba', 'candels']
    lbls = ['SC-SAM', 'EAGLE', 'Illustris', 'Illustris TNG', 'SIMBA', 'CANDELS'] 
    zlo = [0.5, 1., 1.4, 1.8, 2.2, 2.6]
    zhi = [1., 1.4, 1.8, 2.2, 2.6, 3.0]
    zlbls = ['$0.5 < z < 1.0$', '$1.0 < z < 1.4$', '$1.4 < z < 1.8$', '$1.8 < z < 2.2$', '$2.2 < z < 2.6$', '$2.6 < z < 3.0$']
    
    fq_dict = {} 
    for name in names:  
        fqs = [] 
        for i in range(1,len(zlo)+1): 
            if name != 'candels': marr, fq = QF(name, i, noise=noise, seed=seed)
            else: marr, fq = QF(name, i, noise=False)
            fqs.append([marr, fq]) 
        fq_dict[name] = fqs
    
    fig = plt.figure(figsize=(12,8))
    bkgd = fig.add_subplot(111, frameon=False)
    for i_n, name in enumerate(names):  # plot fQ fits
        sub = fig.add_subplot(2,3,i_n+1) 

        plts = []
        for i_z in range(len(zlo)): 
            _plt, = sub.plot(fq_dict[name][i_z][0], fq_dict[name][i_z][1], color='C'+str(i_z))
            plts.append(_plt) 

        sub.set_xlim([8.5, 12.]) 
        sub.set_ylim([0., 1.]) 
        if i_n < 3: sub.set_xticklabels([]) 
        if i_n not in [0, 3]: sub.set_yticklabels([]) 
        sub.text(0.95, 0.05, lbls[i_n], ha='right', va='bottom', transform=sub.transAxes, fontsize=20)
        if i_n == 0: 
            sub.legend(plts[:3], zlbls[:3], loc='upper left', handletextpad=0.5, prop={'size': 17}) 
        elif i_n == 1: 
            sub.legend(plts[3:], zlbls[3:], loc='upper left', handletextpad=0.5, prop={'size': 17}) 
    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', labelpad=15, fontsize=25) 
    bkgd.set_ylabel(r'log ( SFR $[M_\odot \, yr^{-1}]$ )', labelpad=15, fontsize=25) 
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    if noise: 
        fig_name = ''.join([UT.doc_dir(), 'highz/figs/fq_zevo_comparison_wnoise_seed%i.pdf' % seed])
    else: 
        fig_name = ''.join([UT.doc_dir(), 'highz/figs/fq_zevo_comparison.pdf'])
    fig.savefig(fig_name, bbox_inches='tight')
    return None

##################################################
# appendix: resolution effects 
##################################################
def Pssfr_res_impact(n_mc=20, noise=False, seed=1, poisson=False): 
    ''' Plot the impact of SFR resolution on the P(SSFR) distribution. 
    '''
    names = ['eagle', 'illustris_100myr', 'tng', 'simba']
    lbls = ['EAGLE', 'Illustris', 'Illustris TNG', 'SIMBA'] 
    SFRres_dict = {'eagle': 0.018, 'illustris_100myr': 0.0126, 'tng': 0.014, 'simba': 0.182} 
    mbins = [[8.9, 9.4], [8.9, 9.4], [8.9, 9.4], [9.4, 9.9]] 

    fig = plt.figure(figsize=(16,6))
    bkgd = fig.add_subplot(111, frameon=False)
    for i_c, name in enumerate(names):
        for i_m in range(2):  
            if i_m == 0: mbin = mbins[i_c] 
            else: mbin = [10.5, 10.9]

            logm, logsfr, nonzero = readHighz(name, 1, keepzeros=True, noise=noise, seed=seed)
            
            inmbin = (logm > mbin[0]) & (logm < mbin[1]) # in M* bin 
            ngal_bin = float(np.sum(inmbin)) 

            dsfr_res = SFRres_dict[name] # SFR resolution 

            hs_uniform, hs_poisson = [], [] 
            rand = np.random.RandomState(seed)
            for i_mc in range(n_mc): # loop through everything n_mc times 
                # sample logSFR' from [logSFR + dlogSFR] 
                logsfr_nz   = np.log10(10**logsfr[nonzero] + dsfr_res * np.random.uniform(size=np.sum(nonzero)))
                logsfr_z    = np.log10(dsfr_res * np.random.uniform(size=np.sum(~nonzero)))
                _logsfr     = np.zeros(len(logsfr))
                _logsfr[nonzero] = logsfr_nz
                _logsfr[~nonzero] = logsfr_z 
                _logssfr = _logsfr - logm 
                
                h0, h1 = np.histogram(_logssfr, bins=40, range=[-16., -8.])
                hs_uniform.append(h0)
                
                logsfr_nz   = np.log10(10**logsfr[nonzero] + dsfr_res * rand.poisson(size=np.sum(nonzero))) 
                logsfr_z    = np.log10(dsfr_res * rand.poisson(size=np.sum(~nonzero))) 
                _logsfr     = np.zeros(len(logsfr))
                _logsfr[nonzero] = logsfr_nz
                _logsfr[~nonzero] = logsfr_z 
                _logssfr = _logsfr - logm 

                h0, h1 = np.histogram(_logssfr, bins=40, range=[-16., -8.])
                hs_poisson.append(h0)

            hs_uniform = np.array(hs_uniform)/ngal_bin
            hs_poisson = np.array(hs_poisson)/ngal_bin
               
            sub = fig.add_subplot(2,4,i_c+1+4*i_m)
            h0, h1 = np.histogram(logsfr[nonzero] - logm[nonzero], bins=40, range=[-16., -8.])
            bar_x, bar_y = UT.bar_plot(h1, h0/ngal_bin)
            sub.plot(bar_x, bar_y, c='k', ls='-', lw=1.5)#, label='w/o SFR $=0$')

            bar_x, bar_y = UT.bar_plot(h1, np.mean(hs_uniform,axis=0))
            sub.plot(bar_x, bar_y, c='C1', lw=1, label=r"$\mathrm{SFR}_i' \in [\mathrm{SFR}, \mathrm{SFR}+\Delta_\mathrm{SFR}]$")
            sub.errorbar(0.5*(h1[1:] + h1[:-1])-0.02, np.mean(hs_uniform, axis=0), yerr=np.std(hs_uniform, axis=0),
                         fmt='.C1', markersize=.5)
            
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
            if i_m == 0: sub.set_ylim([0., 1.]) 
            else:  sub.set_ylim([0., 2.]) 
            if i_c != 0: sub.set_yticks([]) 
            if i_m == 0: sub.set_title(lbls[i_c], fontsize=20) 
            sub.text(0.5, 0.92, '$'+str(mbin[0])+'< \log M_* <'+str(mbin[1])+'$',
                ha='center', va='top', transform=sub.transAxes, fontsize=15)
    
            if (i_c == 2) and (i_m == 1): 
                sub.legend(loc='lower left', bbox_to_anchor=(0.01, 0.65), frameon=False, prop={'size':13}) 
    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    bkgd.set_ylabel(r'P(log SSFR  $[yr^{-1}])$', labelpad=10, fontsize=20)
    bkgd.set_xlabel(r'log SSFR  $[yr^{-1}]$', labelpad=10, fontsize=20)

    fig.subplots_adjust(wspace=0.05, hspace=0.05)

    if noise: 
        fig_name = ''.join([UT.doc_dir(), 'highz/figs/Pssfr_res_impact_wnoise_seed%i.pdf' % seed])
    else: 
        fig_name = ''.join([UT.doc_dir(), 'highz/figs/Pssfr_res_impact_comparison.pdf'])
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close()
    return None


def Mlim_res_impact(n_mc=20, noise=False, seed=1, threshold=0.2): 
    ''' determine M_lim stellar mass where the resolution limit of simulations
    impact the SFS fits.
    '''
    names = ['eagle', 'illustris_100myr', 'tng', 'simba']
    lbls = ['EAGLE', 'Illustris', 'Illustris TNG', 'SIMBA'] 
    zlo = [0.5, 1., 1.4, 1.8, 2.2, 2.6]
    zhi = [1., 1.4, 1.8, 2.2, 2.6, 3.0]
    SFRres_dict = {'eagle': 0.018, 'illustris_100myr': 0.0126, 'tng': 0.014, 'simba': 0.182} 
    
    # SFMS overplotted ontop of SFR--M* relation 
    fig = plt.figure(figsize=(18,3*len(names)))
    bkgd = fig.add_subplot(111, frameon=False)
    for i_n, name in enumerate(names):  # plot SFMS fits
        for i_z in range(len(zlo)): 
            # read in log M* and log SFR of sim (including SFR=0) 
            logm, logsfr, nonzero = readHighz(name, i_z+1, keepzeros=True, noise=noise, seed=seed)
            # standard SFS fit 
            sfs_std = highzSFSfit(name, i_z+1, noise=noise, seed=seed)
            sfs_std_sfr = np.zeros(sfs_std._mbins.shape[0]) 
            sfs_std_sfr[sfs_std._mbins_sfs] = sfs_std._fit_logsfr
            sfs_std_err_ssfr = np.zeros(sfs_std._mbins.shape[0]) 
            sfs_std_err_ssfr[sfs_std._mbins_sfs] = sfs_std._fit_err_logssfr

            dsfr_res = SFRres_dict[name] # SFR resolution 

            sfs_mc = np.zeros((n_mc, sfs_std._mbins.shape[0]))
            sfs_mc_mbins_sfs = [] 
            for i_mc in range(n_mc): # loop through everything n_mc times 
                # sample logSFR' from [logSFR + dlogSFR] 
                logsfr_nz   = np.log10(10**logsfr[nonzero] + dsfr_res * np.random.uniform(size=np.sum(nonzero)))
                logsfr_z    = np.log10(dsfr_res * np.random.uniform(size=np.sum(~nonzero)))
                _logsfr     = np.zeros(len(logsfr))
                _logsfr[nonzero] = logsfr_nz
                _logsfr[~nonzero] = logsfr_z 
            
                fSFS_mc = fstarforms() # initialize 
                _sfs_fit = fSFS_mc.fit(logm, _logsfr, # fit SFS 
                        method='gaussmix', 
                        fit_range=[8.5, 12.0],  
                        dlogm=0.4, 
                        n_bootstrap=1, 
                        Nbin_thresh=100) 
                sfs_mc[i_mc, fSFS_mc._mbins_sfs] = _sfs_fit[1]
                sfs_mc_mbins_sfs.append(fSFS_mc._mbins_sfs) 
                
            mu_sfs_mc = np.sum(sfs_mc, axis=0)/np.sum(np.array(sfs_mc_mbins_sfs), axis=0)  # average SFS SFR of n_mc
            joint_mbins = sfs_std._mbins_sfs & (np.sum(np.array(sfs_mc_mbins_sfs), axis=0) == n_mc)
            #np.isfinite(mu_sfs_mc) # M* bins with SFS 

            dsfs_res = sfs_std_sfr[joint_mbins] - mu_sfs_mc[joint_mbins] # change in SFS fit from resolution limit 
            print name, i_z
            print dsfs_res
            
            # determine logM* limit based on when dsfs_res shifts by > threshold dex 
            mbin_mid = 0.5 * (sfs_std._mbins[:,0] + sfs_std._mbins[:,1])
            above_thresh = (np.abs(dsfs_res) > threshold) & (mbin_mid[joint_mbins] < 10.) 
            if np.sum(above_thresh) > 0:
                mlim = (mbin_mid[joint_mbins][(np.abs(dsfs_res) > threshold)]).max() + 0.5 * sfs_std._dlogm 
            else: 
                mlim = None 
            
            # --- plot the comparison --- 
            sub = fig.add_subplot(len(names),6,i_n*6+i_z+1) 
            DFM.hist2d(logm[nonzero], logsfr[nonzero], color='C0', 
                    levels=[0.68, 0.95], range=[[7.8, 12.], [-4., 4.]], 
                    plot_datapoints=True, fill_contours=False, plot_density=True, 
                    ax=sub) 
            # plot standard SFS fit
            plt_std = sub.errorbar(mbin_mid[joint_mbins], sfs_std_sfr[joint_mbins], sfs_std_err_ssfr[joint_mbins], fmt='.k')
            plt_res = sub.scatter(mbin_mid[joint_mbins], mu_sfs_mc[joint_mbins], marker='x', color='C1', lw=1, s=40)

            if i_n == 3 and i_z == 5: sub.legend([plt_std, plt_res], 
                    ['w/ SFR res. eff.', r"$\mathrm{SFR}_i' \in [\mathrm{SFR}, \mathrm{SFR}+\Delta_\mathrm{SFR}]$"], 
                    loc='lower right', handletextpad=-0.02, prop={'size': 10}) 
            if i_n < len(names)-1: sub.set_xticklabels([]) 
            if i_z > 0: sub.set_yticklabels([]) 
            else: sub.text(0.05, 0.95, lbls[i_n], ha='left', va='top', transform=sub.transAxes, fontsize=20)
            
            if mlim is not None: 
                sub.text(0.95, 0.05, '$\log\,M_{\lim}='+str(round(mlim,2))+'$', 
                         ha='right', va='bottom', transform=sub.transAxes, fontsize=15)
                sub.vlines(mlim, -4., 4., color='k', linestyle='--', linewidth=0.5)
                sub.fill_between([7.5, mlim], [-3., -3.], [4., 4.], color='k', alpha=0.2)
            sub.set_xlim([8.5, 12.]) 
            sub.set_xticks([9., 10., 11., 12.]) 
            sub.set_ylim([-3., 4.]) 

    bkgd.set_xlabel(r'log $M_* \;\;[M_\odot]$', labelpad=10, fontsize=25) 
    bkgd.set_ylabel(r'log SFR $[M_\odot \, yr^{-1}]$', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    if noise: 
        fig_name = ''.join([UT.doc_dir(), 'highz/figs/Mlim_res_impact_wnoise_seed%i.pdf' % seed])
    else: 
        fig_name = ''.join([UT.doc_dir(), 'highz/figs/Mlim_res_impact_comparison.pdf'])
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close()
    return None

##################################################
# appendix: slice vs full  
##################################################
def SFS_SAM_comparison(noise=False, seed=1): 
    ''' Compare the SFMS fits among the data and simulation  
    '''
    names = ['sam-light-slice', 'sam-light-full']
    zlo = [0.5, 1., 1.4, 1.8, 2.2, 2.6]
    zhi = [1., 1.4, 1.8, 2.2, 2.6, 3.0]
    
    sfms_dict = {} 
    for name in names:  
        sfms_fits = [] 
        for i in range(1,len(zlo)+1): 
            if noise and 'sam-light' in name: 
                logm, logsfr = readHighz(name, i, keepzeros=False, noise=True, seed=1)
                fSFS = highzSFSfit(name, i, noise=True, seed=1) # fit the SFMSes
            else:
                logm, logsfr = readHighz(name, i, keepzeros=False)
                fSFS = highzSFSfit(name, i) # fit the SFMSes
            sfms_fit = [fSFS._fit_logm, fSFS._fit_logsfr, fSFS._fit_err_logssfr]
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
                    color=colour, alpha=0.75, linewidth=0., label=' '.join(name.upper().split('_')))
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
    if noise: 
        fig_name = ''.join([UT.doc_dir(), 'highz/figs/sfms_SAM_comparison.wnoise.seed%i.pdf' % seed])
    else: 
        fig_name = ''.join([UT.doc_dir(), 'highz/figs/sfms_SAM_comparison.pdf'])
    fig.savefig(fig_name, bbox_inches='tight')
    return None


if __name__=="__main__": 
    #for name in ['illustris_100myr']: #'eagle', 'illustris_10myr', 'illustris_1gyr']:
    #    for method in ['interpexterp', 'powerlaw']:  
    #        #highz_sfms(name)
    #        dSFS(name, method=method) 
    for name in ['eagle', 'illustris_100myr', 'tng', 'simba', 'sam-light-full', 'sam-light-slice', 'candels']:
        continue 
        print('--- %s ---' % name) 
        for iz in range(1,7): 
            print('--- %i of 7 ---' % iz) 
            _ = highzSFSfit(name, iz, overwrite=True)
            pssfr(name, iz)  
        highz_sfms(name)

    for name in ['eagle', 'illustris_100myr', 'tng', 'simba', 'sam-light-full', 'sam-light-slice']:
        continue 
        for iz in range(1,7): 
            add_uncertainty(name, iz)
            highzSFSfit(name, iz, noise=True, seed=1, overwrite=True)
            pssfr(name, iz, noise=True, seed=1)  
        highz_sfms(name, noise=True, seed=1)
    #SFS_comparison()
    #SFS_comparison(noise=True, seed=1)
    
    #SFS_zevo_comparison()
    #SFS_zevo_comparison(noise=True, seed=1)
    
    #SFR_Mstar_comparison()
    #SFR_Mstar_comparison(noise=True, seed=1)
    
    #Mlim_res_impact(n_mc=20, noise=False, seed=1, threshold=0.2)
    #Mlim_res_impact(n_mc=100, noise=True, seed=1, threshold=0.2)

    #Pssfr_res_impact(n_mc=20, noise=False, seed=1, poisson=False)
    #Pssfr_res_impact(n_mc=100, noise=True, seed=1, poisson=False)

    #fcomp_comparison(noise=False, seed=1)
    #fcomp_comparison(noise=True, seed=1)

    #QF_comparison(noise=False, seed=1)
    #QF_comparison(noise=True, seed=1)
    
    QF_zevo_comparison(noise=False, seed=1)
    QF_zevo_comparison(noise=True, seed=1)
    
    #sfms_SAM_comparison()
    #sfms_SAM_comparison(noise=True, seed=1)
