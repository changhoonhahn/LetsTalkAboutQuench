'''

fStarForMS = fitting the STAR FORming Main Sequence 

'''
import numpy as np 
import types
import warnings 
from sklearn.mixture import GaussianMixture as GMix

import util as UT 


class fstarforms(object): 
    ''' class object for fitting the star formation main sequence
    of a galaxy population. 

    Main functionality of this class include : 
    *

    '''
    def __init__(self):
        pass

    def fit(self, logmstar, logsfr, method=None, fit_range=None, dlogm=0.2,
            Nbin_thresh=100, SSFR_cut=None, **kwargs): 
        '''Given log SFR and log Mstar values of a galaxy population, 
        return the power-law best fit to the SFMS. After some initial 
        common sense cuts, P(log SSFR) in bins of stellar mass are fit 
        using specified method. 

        Parameters
        ----------
        logmstar : (np.array) 
            log stellar mass of galaxy population 

        logsfr : (np.array) 
            log sfr of galaxy population 

        method : (str)
            string specifying the method of the fit. Options are
            ['logMbin_extrap', 'gaussfit', 'negbinomfit', 'gaussmix'].

        fit_range : (list) 
            Optional. 2 element list specifying the fitting stellar
            mass range -- [logM*_min, logM*_max]

        dlogm : (float) 
            Optional. Default 0.2 dex. Width of logM* bins. 

        Nbin_thresh : (float)
            Optional. Default is 100. If a logM* bin has less than 
            100 galaxies, the bin is omitted. 

        SSFR_cut : (float or function) 
            Optional. Some SSFR cut to impose on the galaxy population
            before doing the fitting. For 'gaussfit' and 'negbinomfit' 
            default SSFR_cut is logSSFR > -11.

        Notes
        -----
        - Since the inputs are logM* and logSFR, SFR=0 is by construction 
        not accepted. 
            

        References
        ---------- 
        - Bluck et al., 2016 (arXiv:1607.03318)
        - Feldmann, 2017 (arXiv:1705.03014) 
        - Bisigello et al., 2017 (arXiv: 1706.06154)
        '''
        if len(logmstar) != len(logsfr): 
            raise ValueError("logmstar and logsfr are not the same length arrays") 

        if method not in ['logMbin_extrap', 'gaussfit', 'negbinomfit', 'gaussmix']: 
            raise ValueError("not one of the methods!") 

        # only keep sensible logmstar and log sfr
        sense = (logmstar > 0.) & (logmstar < 13) & (logsfr > -5) & (logsfr < 4) & (np.isnan(logsfr) == False))
        if len(logmstar) - np.sum(sense) > 0:  
            warnings.warn(str(len(logmstar) - np.sum(sense))+' galaxies have nonsensical logM* or logSFR values')  
        logmstar = logmstar[np.where(sense)]
        logsfr = logsfr[np.where(sense)]

        # fitting M* range
        if fit_range is None:
            if method == 'lowMbin_extrap': 
                warnings.warn('Specify fitting range of lowMbin_extrap fit method will return garbage') 
            fit_range = [logmstar.min(), logmstar.max()]
        mass_cut = (logmstar > fit_range[0]) & (logmstar < fit_range[1])
        if np.sum(mass_cut) == 0: 
            raise ValueError("no galaxies within that cut!")
        
        # logM* binning 
        mbin_low = np.arange(fit_range[0], fit_range[1], dlogm)
        mbin_high = mbin_low + dlogm

        if method == 'lowMbin_extrap': 
            # fit the SFMS with median SFR in the low mass range where 
            # f_Q ~ 0 then extrapolate the fit to higher masses
            # such a method was used in Bluck et al. 2016
            if fit_range[1] > 9.5: 
                warnings.warn('hmmm, you sure you want to use lowMbin_extrap?'+\
                        'Observations find the quiescent fraction greater than'+\
                        ' 0 in these stellar mass ranges') 

            fit_logm, fit_logsfr = [], []
            for i in range(len(mbin_low)): 
                in_mbin = np.where(
                        (logmstar > mbin_low[i]) & 
                        (logmstar < mbin_high[i])) 

                if len(in_mbin[0]) > Nbin_thresh: 
                    fit_logm.append(np.median(logm[in_mbin]))
                    fit_logsfr.append(np.median(logsfr[in_mbin]))
        
        elif method == 'gaussfit': 
            # in stellar mass bins, fit P(log SSFR) distribution above some SSFR cut with a Gaussian 
            # then fit the mus you get from the Gaussian with a linear fit 
            # this is motivated by the fact that observations find SFMS 
            # to be a log-normal distrubiton (see put references here) 
            if SSFR_cut is None: 
                ssfrcut = (logsfr - logmstar > -11.)
            else: 
                if isinstance(SSFR_cut, float): 
                    ssfrcut = (logsfr - logmstar > SSFR_cut)
                elif callable(SSFR_cut):  
                    # if SSFR_cut is a function of log M*
                    ssfrcut = (logsfr - logmstar > SSFR_cut(logmstar))

            fit_logm, fit_logsfr = [], []
            for i in range(len(mbin_low)): 
                in_mbin = np.where(
                        (logmstar > mbin_low[i]) & 
                        (logmstar < mbin_high[i]) & 
                        ssfrcut) 

                if len(in_mbin[0]) > Nbin_thresh: 
                    # now fit a gaussian to the distribution 
                    yy, xx_edges = np.histogram(logSFR_fit[in_mbin]-logMstar_fit[in_mbin],
                            bins=20, range=[-14, -8], normed=True)
                    xx = 0.5 * (xx_edges[1:] + xx_edges[:-1])

                    gauss = lambda xx, aa, x0, sig: aa * np.exp(-(xx - x0)**2/(2*sig**2))
                    
                    try: 
                        popt, pcov = curve_fit(gauss, xx, yy, 
                                p0=[1., np.median(logsfr[in_mbin]-logmstar[in_mbin]), 0.3])
                    except RuntimeError: 
                        plt.scatter(xx, yy)
                        plt.show()
                        plt.close()
                        raise ValueError

                    fit_logm.append(np.median(logmstar[in_mbin]))
                    fit_logsfr.append(popt[1])

        elif method == 'negbinomfit': 
            # in stellar mass bins, fit P(SSFR) distribution above some SSFR cut with a 
            # negative binomial distribution (see ipython notebook for details on binomial distribution)
            if SSFR_cut is None: 
                ssfrcut = (logsfr - logmstar > -11.)
            else: 
                if isinstance(SSFR_cut, float): 
                    ssfrcut = (logsfr - logmstar > SSFR_cut)
                elif callable(SSFR_cut):  
                    # if SSFR_cut is a function of log M*
                    ssfrcut = (logsfr - logmstar > SSFR_cut(logmstar))
            
            fit_logm, fit_logsfr = [], []
            for i in range(len(mbin_low)): 
                in_mbin = np.where(
                        (logmstar > mbin_low[i]) & 
                        (logmstar < mbin_high[i]) & 
                        ssfrcut) 

                if len(in_mbin[0]) > Nbin_thresh: 
                    # now fit a negative binomal to the distribution 
                    yy, xx_edges = np.histogram(logsfr[in_mbin]-logmstar[in_mbin],
                            bins=20, range=[-14, -8], normed=True)
                    xx = 0.5 * (xx_edges[1:] + xx_edges[:-1])

                    # negative binomial PDF 
                    NB_fit = lambda xx, aa, mu, theta: UT.NB_pdf_logx(np.power(10., xx+aa), mu, theta)

                    try: 
                        popt, pcov = curve_fit(NB_fit, xx, yy, 
                                p0=[12., 100, 1.5])
                    except RuntimeError: 
                        fig = plt.figure(2)
                        plt.scatter(xx, yy)
                        plt.show()
                        plt.close()
                        raise ValueError

                    fit_logm.append(np.median(logmstar[in_mbin]))
                    fit_logsfr.append(np.log10(popt[1]) - popt[0])

        elif method == 'gaussmix':
            # in stellar mass bins, fit P(log SSFR) distribution using a Gaussian mixture model 
            # with at most 3 components and some common sense priors based on the fact
            # that SFMS is roughly a log-normal distribution. This does not require a log(SSFR) 
            # cut like gaussfit and negbinomfit, which is nice.
            # (see ipython notebook for examples of the gaussian mixture model) 

            fit_logm, fit_logsfr = [], []
            for i in range(len(mbin_low)): 
                in_mbin = np.where(
                        (logmstar > mbin_low[i]) & 
                        (logmstar < mbin_high[i]) & 
                        ssfrcut) 
                gmms, bics = [], []  
                for i_n, n in enumerate([1,2,3]): 
                    X = logsfr[in_mbin] - logmstar[in_mbin]
                    gmm = Mix.GaussianMixture(n_components=n)
                    gmm.fit(X)
                    gmms.append(gmm)
                    bics.append(gmm.bic(X)) # bayesian information criteria

                # components with the lowest BIC (preferred)
                i_best = np.array(bics).argmin()
                gbest = gmms[i_best]
                if gbest.means_.flatten().max() < -11.:
                    # this means that the bestfitting GMM does not find a gaussian with 
                    # log SSFR > -11. we take this to mean that SFMS is not well 
                    # defined in this mass bin 
                    continue 
                if i_best > 0 and np.sum(gbest.means_.flatten() > -11) > 1: 
                    # if best fit has more than one component make sure that it's not
                    # 'overfitting' the sfms. Check whether the two gaussians with
                    # means log SSFR > -11 have comparable weights
                    in_sf = np.where(gbest.means_.flatten() > -11) > 1)

                    if gbest.weights_[in_sf].min()/gbest.weights_[in_sf].max() > 0.33: 
                        #### finish this up 
                        #### finish this up 
                        #### finish this up 
                        #### finish this up 
                        #### finish this up 
                        #### finish this up 
                        #### finish this up 

        if ('linearfit' in method) or (method == 'lowMbin_extrap'): 
            # now fit line to the fit_Mstar and fit_SSFR values
            xx = np.array(fit_Mstar) - Mfid  # log Mstar - log M_fid
            yy = np.array(fit_SSFR)
            A = np.vstack([xx, np.ones(len(xx))]).T
            m, c = np.linalg.lstsq(A, yy)[0] 
            
            sfms_fit = lambda mm: m * (mm - Mfid) + c + mm

            if forTest: 
                return sfms_fit, [np.array(fit_Mstar), np.array(fit_SSFR)]
            else: 
                return sfms_fit 

