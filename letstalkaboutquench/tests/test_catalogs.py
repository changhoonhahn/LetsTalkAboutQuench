'''
'''
import numpy as np 

# --- Local ---
import env 
from catalogs import Catalog as Cat
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


def GroupFinder_purity(): 
    ''' Test Catalog.GroupFinder by reproducing the purity plots 
    that Tjitske generated
    '''
    names = ['illustris_100myr', 'eagle_100myr', 'mufasa_100myr']
    fig = plt.figure(figsize=(4*len(names),4)) 
    bkgd = fig.add_subplot(111, frameon=False) 
    Cata = Cat()
    for i_n, name in enumerate(names): 
        logM, _, _, censat = Cata.Read(name, keepzeros=True) 
        psat = Cata.GroupFinder(name)
        
        if len(psat) != len(logM): 
            print name 
            print 'N_gal group finder = ', len(psat)
            print 'N_gal = ', len(logM)
            raise ValueError

        ispurecen = (psat < 0.01) 
        iscen = (psat < 0.5) 

        mbin = np.linspace(8., 12., 17) 
        mmids, fp_pc, fp_c = [], [], []  # purity fraction for pure central (pc) and central (c)
        for im in range(len(mbin)-1): 
            inmbin = (logM > mbin[im]) & (logM < mbin[im+1])
            if np.sum(inmbin) > 0: 
                mmids.append(0.5*(mbin[im] + mbin[im+1]))
                fp_pc.append(float(np.sum(censat[ispurecen & inmbin] == 1))/float(np.sum(ispurecen & inmbin)))
                fp_c.append(float(np.sum(censat[iscen & inmbin] == 1))/float(np.sum(iscen & inmbin)))
        
        sub = fig.add_subplot(1,len(names),i_n+1) 
        sub.plot(mmids, fp_pc, 
                label='Pure Centrals '+str(round(np.mean(fp_pc),2)))
        sub.plot(mmids, fp_c, 
                label='Centrals '+str(round(np.mean(fp_c),2))) 
        sub.set_xlim([8., 12.]) 
        sub.set_ylim([0., 1.]) 

        sub.legend(loc='lower left', prop={'size':15}) 
        
    bkgd.set_xlabel(r'log ( $M_* \;\;[M_\odot]$ )', labelpad=10, fontsize=25) 
    bkgd.set_ylabel(r'$f_\mathrm{purity}$', labelpad=10, fontsize=25) 
    bkgd.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    fig.savefig(''.join([UT.fig_dir(), 'groupfinder.purity.png']), bbox_inches='tight') 
    plt.close() 
    return None 


def SFR_Mstar_SSFRcut(name, xrange=None, yrange=None): 
    ''' Test that the logSFR-logM* relation is reasonable and that
    read works.
    '''
    cat = Cat() 
    logM, logSFR, w = cat.Read(name)
    
    prettyplot() 
    fig = plt.figure() 
    sub = fig.add_subplot(111)
    if xrange is None:
        xrange = [7., 12.]
    if yrange is None:
        yrange = [-4., 2.]
    
    DFM.hist2d(logM, logSFR, color='#1F77B4', 
                levels=[0.68, 0.95], range=[xrange, yrange], 
                plot_datapoints=True, fill_contours=False, plot_density=True, ax=sub) 
    f_SSFRcut = cat._default_fSSFRcut()
    sub.plot(np.linspace(xrange[0], xrange[1], 10), [f_SSFRcut(m)+m for m in np.linspace(xrange[0], xrange[1], 10)], c='r', ls='--', lw=2)

    sub.set_xlim(xrange)
    sub.set_ylim(yrange)
    sub.set_xlabel(r'$\mathtt{log \; M_* \;\;[M_\odot]}$', labelpad=10, fontsize=25) 
    sub.set_ylabel(r'$\mathtt{log \; SFR \;\;[M_\odot \, yr^{-1}]}$', labelpad=10, fontsize=25) 
    fig.savefig(''.join([UT.fig_dir(), name, '.sfr_mstar.ssfrcut.png']), bbox_inches='tight') 
    plt.close() 
    return None 


def SFR_Mstar(name, xrange=None, yrange=None): 
    ''' Test that the logSFR-logM* relation is reasonable and that
    read works.
    '''
    cat = Cat() 
    _logM, _logSFR, w, censat = cat.Read(name)
    iscen = (censat == 1) 
    logM = _logM[iscen]
    logSFR = _logSFR[iscen]

    prettyplot() 
    fig = plt.figure() 
    sub = fig.add_subplot(111)
    if xrange is None:
        xrange = [7., 12.]
    if yrange is None:
        yrange = [-4., 2.]
    
    #DFM.hist2d(logM, logSFR, color='#1F77B4', 
    #            levels=[0.68, 0.95], range=[xrange, yrange], 
    #            plot_datapoints=True, fill_contours=False, plot_density=True, ax=sub) 
    print logSFR
    sub.scatter(logM, logSFR, s=2)
    sub.set_xlim(xrange)
    sub.set_ylim(yrange)
    sub.set_xlabel(r'$\mathtt{log \; M_* \;\;[M_\odot]}$', labelpad=10, fontsize=25) 
    sub.set_ylabel(r'$\mathtt{log \; SFR \;\;[M_\odot \, yr^{-1}]}$', labelpad=10, fontsize=25) 
    fig.savefig(''.join([UT.fig_dir(), name, '.sfr_mstar.png']), bbox_inches='tight') 
    plt.close() 
    return None 


def pssfr(name, Mrange=[10.,10.5], xrange=None): 
    '''
    '''
    cat = Cat() 
    logM, logSFR, w = cat.Read(name)
    
    # mass bins
    inmbin = np.where((logM >= Mrange[0]) & (logM < Mrange[1]))
    
    prettyplot() 
    fig = plt.figure() 
    sub = fig.add_subplot(111)
    
    if xrange is None: 
        xrange = [-2.5, 2]
    yy, xx_edges = np.histogram(logSFR[inmbin], bins=20, range=[-2.5, 2], normed=True)
    x_plot, y_plot = UT.bar_plot(xx_edges, yy)
    sub.plot(x_plot, y_plot, lw=3)
    sub.set_xlim(xrange)
    sub.set_xlabel(r'$\mathtt{log \; SFR \;\;[M_\odot \, yr^{-1}]}$', labelpad=10, fontsize=25) 
    sub.set_ylabel(r'$\mathtt{P(log \; SFR)\;}$', labelpad=10, fontsize=25) 
    f_fig = ''.join([UT.fig_dir(), name, '.Psfr.mstar', str(Mrange[0]), '_', str(Mrange[1]), '.png'])
    fig.savefig(f_fig, bbox_inches='tight') 
    plt.close() 
    return None



if __name__=='__main__': 
    GroupFinder_purity()
