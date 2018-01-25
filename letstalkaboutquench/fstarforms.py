'''

fStarForMS = fitting the STAR FORming Main Sequence 

'''
import numpy as np 
import warnings 
from scipy.optimize import curve_fit
from sklearn.mixture import GaussianMixture as GMix

import util as UT 


class fstarforms(object): 
    ''' class object for fitting the star formation main sequence
    of a galaxy population. 

    Main functionality of this class include : 
    * fitting log SFR of the SFMS in bins of log M*  
    * fitting parameterizations of the SFMS using log SFR 


    to-do
    * implement calculate f_SFMS (the SFMS fraction)
    '''
    def __init__(self):
        self._fit_logm = None 
        self._fit_logssfr = None
        self._fit_logsfr = None

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

        Returns 
        -------
        fit_logm, fit_logsfr : (array, array)
             

        Notes
        -----
        - Since the inputs are logM* and logSFR, SFR=0 is by construction 
        not accepted. 
            

        References
        ---------- 
        - Bluck et al., 2016 (arXiv:1607.03318)
        - Feldmann, 2017 (arXiv:1705.03014) 
        - Bisigello et al., 2017 (arXiv: 1706.06154)
        Gaussian Mixture Modelling: 
        - Kuhn and Feigelson, 2017 (arXiv:1711.11101) 
        - McLachlan and Peel, 2000
        - Lloyd, 1982 (k-means algorithm)
        - Dempster, Laird, Rubin, 1977 (EM algoritmh) 
        - Wu, 1983 (EM convergence) 
        '''
        if len(logmstar) != len(logsfr): 
            raise ValueError("logmstar and logsfr are not the same length arrays") 

        if method not in ['logMbin_extrap', 'gaussfit', 'negbinomfit', 'gaussmix']: 
            raise ValueError("not one of the methods!") 

        # only keep sensible logmstar and log sfr
        sense = (logmstar > 0.) & (logmstar < 13) & (logsfr > -5) & (logsfr < 4) & (np.isnan(logsfr) == False)
        if len(logmstar) - np.sum(sense) > 0:  
            warnings.warn(str(len(logmstar) - np.sum(sense))+' galaxies have nonsensical logM* or logSFR values')  
        self._sensecut = sense
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
        self._dlogm = dlogm
        self._Nbin_thresh = Nbin_thresh

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

            fit_logm, fit_logssfr = [], []
            fit_popt, fit_amp = [], []
            frac_sfms = [] 
            for i in range(len(mbin_low)): 
                in_mbin = np.where(
                        (logmstar > mbin_low[i]) & 
                        (logmstar < mbin_high[i]) & 
                        ssfrcut) 

                if len(in_mbin[0]) > Nbin_thresh: 
                    # now fit a gaussian to the distribution 
                    yy, xx_edges = np.histogram(logsfr[in_mbin]-logmstar[in_mbin],
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
                    fit_logssfr.append(popt[1])
                    famp = float(len(in_mbin[0]))/np.sum((logmstar > mbin_low[i]) & (logmstar < mbin_high[i]))
                    fit_amp.append(famp)
                    fit_popt.append(popt)
                    frac_sfms.append(popt[0] * np.sqrt(2.*np.pi*popt[2]**2) * famp)
            self._fit_amp = fit_amp
            self._fit_popt = fit_popt

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
            
            fit_logm, fit_logssfr = [], []
            fit_popt, fit_amp = [], []
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
                    fit_logssfr.append(np.log10(popt[1]) - popt[0])
                    fit_amp.append(float(len(in_mbin[0]))/np.sum((logmstar > mbin_low[i]) & (logmstar < mbin_high[i])))
                    fit_popt.append(popt)
            self._fit_amp = fit_amp
            self._fit_popt = fit_popt
            frac_sfms = ['not worth integrating negative binomial distribution'] 

        elif method == 'gaussmix':
            # in stellar mass bins, fit P(log SSFR) distribution using a Gaussian mixture model 
            # with at most 3 components and some common sense priors based on the fact
            # that SFMS is roughly a log-normal distribution. This does not require a log(SSFR) 
            # cut like gaussfit and negbinomfit, which is nice.
            # (see ipython notebook for examples of the gaussian mixture model) 

            fit_logm, fit_logssfr = [], []
            gmix_weights_, gmix_means_, gmix_covariances_ = [], [], []
            frac_sfms = [] 
            for i in range(len(mbin_low)): 
                in_mbin = np.where(
                        (logmstar > mbin_low[i]) & 
                        (logmstar < mbin_high[i])) 
                X = logsfr[in_mbin] - logmstar[in_mbin] # logSSFRs
                X = np.reshape(X, (-1,1))
                if len(in_mbin[0]) <= Nbin_thresh: 
                    continue

                n_comps = [1,2,3]
                gmms, bics = [], []  
                for i_n, n in enumerate(n_comps): 
                    gmm = GMix(n_components=n)
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
                    warnings.warn('SFMS is not well defined in the M* bin'+str(mbin_low[i])+'-'+(str(mbin_high[i])))
                    continue 

                if n_comps[i_best] > 1 and np.sum(gbest.means_.flatten() > -11) > 1: 
                    # if best fit has more than one component make sure that it's not
                    # 'overfitting' the sfms. Check whether the two gaussians with
                    # means log SSFR > -11 have comparable weights
                    in_sf = np.where(gbest.means_.flatten() > -11)

                    if gbest.weights_[in_sf].min()/gbest.weights_[in_sf].max() > 0.33: 
                        # there are two components that contribute significantly 
                        # to the SF population distribution. The second component 
                        # cannot be considered a nuissance component.  
                        bics[i_best] = np.inf 
                        i_best = np.array(bics).argmin() # next best fit 
                        gbest = gmms[i_best]
                        if n_comps[i_best] > 1 and np.sum(gbest.means_.flatten() > -11) > 1: 
                            in_sf = np.where(gbest.means_.flatten() > -11)
                            if gbest.weights_[in_sf].min()/gbest.weights_[in_sf].max() > 0.33: 
                                warnings.warn('GMM does not provide a sensible fit to the SFMS '+\
                                        'in the M* bin'+str(mbin_low[i])+'-'+(str(mbin_high[i])))
                                continue 
                            else: 
                                fit_logm.append(np.median(logmstar[in_mbin])) 
                                sf_comp = self._GMM_SFMS_logSSFR(gbest.means_.flatten(), gbest.weights_)
                                fit_logssfr.append(gbest.means_.flatten()[sf_comp])
                                gmix_weights_.append(gbest.weights_)
                                gmix_means_.append(gbest.means_.flatten())
                                gmix_covariances_.append(gbest.covariances_.flatten())
                        elif gbest.means_.flatten().max() < -11.:
                            warnings.warn('SFMS is not well defined in the M* bin'+str(mbin_low[i])+'-'+(str(mbin_high[i])))
                            continue 
                        else: 
                            fit_logm.append(np.median(logmstar[in_mbin])) 
                            sf_comp = self._GMM_SFMS_logSSFR(gbest.means_.flatten(), gbest.weights_)
                            fit_logssfr.append(gbest.means_.flatten()[sf_comp])
                            gmix_weights_.append(gbest.weights_)
                            gmix_means_.append(gbest.means_.flatten())
                            gmix_covariances_.append(gbest.covariances_.flatten())
                    else: 
                        # there are two components that contribute to the SF population 
                        # distribution. hOwever, the second component does not contribute
                        # significantly to the distribution. 
                        fit_logm.append(np.median(logmstar[in_mbin])) 
                        sf_comp = self._GMM_SFMS_logSSFR(gbest.means_.flatten(), gbest.weights_)
                        fit_logssfr.append(gbest.means_.flatten()[sf_comp])
                        gmix_weights_.append(gbest.weights_)
                        gmix_means_.append(gbest.means_.flatten())
                        gmix_covariances_.append(gbest.covariances_.flatten())
                else: 
                    fit_logm.append(np.median(logmstar[in_mbin])) 
                    sf_comp = self._GMM_SFMS_logSSFR(gbest.means_.flatten(), gbest.weights_)
                    fit_logssfr.append(gbest.means_.flatten()[sf_comp])
                    gmix_weights_.append(gbest.weights_)
                    gmix_means_.append(gbest.means_.flatten())
                    gmix_covariances_.append(gbest.covariances_.flatten())

                # calculate the star formation main sequence fraction 
                # an estimate of the fraction of galaxies in this mass bin that are 
                # on the star formation main sequence 
                frac_sfms.append(gbest.weights_[sf_comp])

            # save the gmix fit values 
            self._gmix_weights = gmix_weights_ 
            self._gmix_means = gmix_means_
            self._gmix_covariances = gmix_covariances_

        self._frac_sfms = np.array(frac_sfms)

        # save the fit ssfr and logm 
        self._fit_logm = np.array(fit_logm)  
        self._fit_logssfr = np.array(fit_logssfr)  
        self._fit_logsfr = self._fit_logm + self._fit_logssfr
        return [self._fit_logm, self._fit_logsfr]

    def powerlaw(self, logMfid=None): 
        ''' Find the best-fit power-law parameterization of the 
        SFMS from the logM* and log SFR_SFMS fit from the `fit` 
        method above. This is the simplest fit possible

        f_SFMS(log M*)  = a * (log M* - logM_fid) + b 

        Parameters
        ----------
        Mid : (float) 
            Fiducial 

        Returns
        -------
        sfms_fit : (function)
            f_SFMS(logM*)
        '''
        if self._fit_logm is None  or self._fit_logssfr is None or self._fit_logsfr is None: 
            raise ValueError('Run `fit` method first')

        # fiducial log M*  
        if logMfid is None: 
            logMfid = int(np.round(np.median(self._fit_logm)/0.5))*0.5
            print('fiducial log M* ='+str(logMfid))
        self._logMfid = logMfid

        # now fit line to the fit_Mstar and fit_SSFR values
        xx = self._fit_logm - logMfid  # log Mstar - log M_fid
        yy = self._fit_logsfr
        A = np.vstack([xx, np.ones(len(xx))]).T
        m, c = np.linalg.lstsq(A, yy)[0] 
        self._powerlaw_m = m 
        self._powerlaw_c = c
        
        sfms_fit = lambda mm: m * (mm - logMfid) + c
        print 'logSFR_SFMS = '+str(round(m, 3))+' (logM* - '+str(round(logMfid,3))+') + '+str(round(c, 3))
        return sfms_fit 

    def frac_SFMS(self): 
        ''' Return the estimate of the fraction of galaxies that are on 
        the star formation main sequence as a function of mass produce from 
        the fit. 
        '''
        if self._fit_logm is None  or self._frac_sfms is None:
            raise ValueError('Run `fit` method first')
        if isinstance(self._frac_sfms[0], str): 
            raise NotImplementedError(self._frac_sfms[0]) 
        return [self._fit_logm, self._frac_sfms]

    def _GMM_SFMS_logSSFR(self, means, weights): 
        ''' Given means and weights of a GMM, determine which component corresponds to the
        SFMS portion
        '''
        issf = np.where(means > -11.)
        if len(issf[0]) == 0: # there's only one component thats in the SF portion  
            return issf[0]
        else: 
            return issf[0][weights[issf].argmax()]
