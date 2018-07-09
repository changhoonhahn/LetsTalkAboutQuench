'''

fStarForMS = fitting the STAR FORming Main Sequence 

'''
import numpy as np 
import scipy as sp 
import warnings 
from scipy.optimize import curve_fit
from extreme_deconvolution import extreme_deconvolution
from sklearn.mixture import GMM 
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
        self._sfms_fit = None 

    def fit(self, logmstar, logsfr, logsfr_err=None, method=None, fit_range=None, dlogm=0.2,
            Nbin_thresh=100, SSFR_cut=None, max_comp=3, fit_error=None, n_bootstrap=None,
            silent=False): 
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
            ['logMbin_extrap', 'gaussfit', 'negbinomfit', 'gaussmix', 'gaussmix_err'].
            'gaussmix' uses Gaussian Mixture Models to fit the P(SSFR) distribution. 
            'gaussmix_err' uses Gaussian Mixture Models *but* accounts for uncertainty
            in the SFRs. 

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

        fit_error : 
            Optional. Default is None. If specified, estimates the error
            in the fit using either boostrap (fit_error='bootstrap') or 
            jackknifing (fit_error='jackknife') 

        n_bootstrap : 
            Optional. If fit_error='bootstrap' specify, the number of 
            bootstrap samples. 

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
        self._check_input(logmstar, logsfr)  # check the logmstar and logsfr inputs 
        if logsfr_err is not None: 
            if np.sum(np.invert(np.isfinite(logsfr_err))) > 0: 
                raise ValueError("There are non-finite log SFR error values")  

        if method not in ['logMbin_extrap', 'gaussfit', 'negbinomfit', 'gaussmix', 'gaussmix_err']: 
            raise ValueError(method+" is not one of the methods!") 

        if fit_error is not None: 
            if fit_error not in ['bootstrap', 'jackknife']: 
                raise ValueError("fitting currently only supports fit_error=bootstrap or jackknife") 

        if method == 'gaussmix_err' and fit_error is not None: 
            raise NotImplementedError("error estimation for gaussian mixture model fitting"+\
                    " using extreme deconvolution is currently not supported. Considering how long"+\
                    "extreme deconvolution takes, not sure if it should be...")  

        if fit_range is None: 
            # if fitting M* range is not specified, use all the galaxies
            # the fit range will be a bit padded. 
            if method == 'lowMbin_extrap': 
                warnings.warn('Specify fitting range of lowMbin_extrap '+\
                        'fit method will return garbage') 
            fit_range = [int(logmstar.min()/dlogm)*dlogm, np.ceil(logmstar.max()/dlogm)*dlogm]

        mass_cut = (logmstar > fit_range[0]) & (logmstar < fit_range[1])
        if np.sum(mass_cut) == 0: 
            print("trying to fit SFMS over range %f < log M* < %f" % (fit_range[0], fit_range[1]))
            print("input spans %f < log M* < %f" % (logmstar.min(), logmstar.max()))
            raise ValueError("no galaxies within that cut!")
        # log M* binning 
        mbin_low = np.arange(fit_range[0], fit_range[1], dlogm)
        mbin_high = mbin_low + dlogm
        mbins =  np.array([mbin_low, mbin_high]).T
        self._mbins = mbins
        self._dlogm = dlogm
        self._Nbin_thresh = Nbin_thresh

        if 'gaussmix' in method:
            # for each logM* bin, fit P(log SSFR) distribution using a Gaussian mixture model 
            # with at most 3 components and some common sense priors based on the fact
            # that SFMS is roughly a log-normal distribution. This does not require a log(SSFR) 
            # cut like gaussfit and negbinomfit and can be flexibly applied to a wide range of 
            # SSFR distributions.
            if (method == 'gaussmix_err') and  (logsfr_err is None): 
                raise ValueError("This method requires logSFR errors") 

            fit_logm = [] 
            fit_logssfr, fit_sig_logssfr = [], [] # mean and variance of the SFMS component
            if fit_error is not None: fit_err_logssfr, fit_err_sig_logssfr = [], [] # uncertainty in the mean and variance
            gbests =  []
            
            for i in range(mbins.shape[0]): # log M* bins  
                in_mbin = (logmstar > mbins[i,0]) & (logmstar < mbins[i,1])

                X = logsfr[in_mbin] - logmstar[in_mbin] # logSSFRs
                X = np.reshape(X, (-1,1))
                if method == 'gaussmix_err':
                    # error bars on of logSFR
                    Xerr = logsfr_err[in_mbin] 
                    Xerr = np.reshape(Xerr, (-1,1,1))

                if np.sum(in_mbin) <= Nbin_thresh: # not enough galaxies
                    continue

                n_comps = range(1, max_comp+1)# [1,2,3] default max_comp = 3
                gmms, bics = [], []  
                for i_n, n in enumerate(n_comps): 
                    if method == 'gaussmix': 
                        gmm = GMix(n_components=n)
                        gmm.fit(X)
                        bics.append(gmm.bic(X)) # bayesian information criteria
                    elif method == 'gaussmix_err': 
                        gmm =xdGMM(n_components=n)
                        gmm.fit(X, Xerr)
                        bics.append(gmm.bic(X, Xerr)) # bayesian information criteria
                    gmms.append(gmm)

                # components with the lowest BIC (preferred)
                i_best = np.array(bics).argmin()
                n_best = n_comps[i_best] # number of components of the best-fit 
                gbest = gmms[i_best] # best fit GMM 
                gbests.append(gbest)

                # identify the different components
                i_sfms, i_q, i_int, i_sb = self._GMM_idcomp(gbest, silent=True)
    
                if i_sfms is None: 
                    continue 
                
                # save the SFMS log M* and log SSFR values 
                fit_logm.append(np.median(logmstar[in_mbin])) 
                fit_logssfr.append(UT.flatten(gbest.means_.flatten()[i_sfms]))
                fit_sig_logssfr.append(np.sqrt(UT.flatten(gbest.covariances_.flatten()[i_sfms])))

                # calculate the uncertainty of logSSFR fit 
                if fit_error is None: 
                    pass 
                elif fit_error == 'bootstrap': 
                    # using bootstrap resampling 
                    boot_mu_logssfr, boot_sig_logssfr = [], []
                    for i_boot in range(n_bootstrap): 
                        # resample the data w/ replacement 
                        X_boot = np.random.choice(X.flatten(), size=len(X), replace=True) 
                        gmm_boot = GMix(n_components=n_best)
                        gmm_boot.fit(X_boot.reshape(-1,1))
                
                        i_sfms_boot, _, _, _ = self._GMM_idcomp(gmm_boot, silent=True)
                        if i_sfms_boot is None: 
                            continue 
                        boot_mu_logssfr.append(UT.flatten(gmm_boot.means_.flatten()[i_sfms_boot]))
                        boot_sig_logssfr.append(np.sqrt(UT.flatten(gmm_boot.covariances_.flatten()[i_sfms_boot])))
                    fit_err_logssfr.append(np.std(np.array(boot_mu_logssfr)))
                    fit_err_sig_logssfr.append(np.std(np.array(boot_sig_logssfr)))
                else: 
                    raise NotImplementedError("not yet implemented") 
                
            # save the bestfit GMM  
            self._gbests = gbests
        else: 
            raise NotImplementedError("other fitting methods below need to be significantly corrected") 
            '''
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
            '''
        # save the fit ssfr and logm 
        self._fit_logm = np.array(fit_logm)  
        self._fit_logssfr = np.array(fit_logssfr)  
        self._fit_logsfr = self._fit_logm + self._fit_logssfr
        self._fit_sig_logssfr = np.array(fit_sig_logssfr)
        if fit_error is None: 
            self._fit_err_logssfr = None 
            self._fit_err_sig_logssfr = None
            return [self._fit_logm, self._fit_logsfr]
        else: 
            self._fit_err_logssfr = np.array(fit_err_logssfr) 
            self._fit_err_sig_logssfr = np.array(fit_err_sig_logssfr)
            return [self._fit_logm, self._fit_logsfr, self._fit_err_logssfr]

    def powerlaw(self, logMfid=None, silent=True): 
        ''' Find the best-fit power-law parameterization of the 
        SFMS from the logM* and log SFR_SFMS fit from the `fit` 
        method above. This is the simplest fit possible

        f_SFMS(log M*)  = a * (log M* - logM_fid) + b 

        Parameters
        ----------
        logMfid : (float) 
            Fiducial log M_*. 

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

        # chi-squared
        chisq = lambda theta: np.sum((theta[0] * xx + theta[1] - yy)**2/self._fit_err_logssfr**2)

        #A = np.vstack([xx, np.ones(len(xx))]).T
        #m, c = np.linalg.lstsq(A, yy)[0] 
        tt = sp.optimize.minimize(chisq, np.array([0.8, 0.3])) 

        self._powerlaw_m = tt['x'][0]
        self._powerlaw_c = tt['x'][1]
        
        sfms_fit = lambda mm: tt['x'][0] * (mm - logMfid) + tt['x'][1]
        self._sfms_fit = sfms_fit 
        if not silent: 
            print('logSFR_SFMS = %s (logM* - %s) + %s' % (str(round(m, 3)), str(round(logMfid,3)), str(round(c, 3))))
        return sfms_fit 
    
    def d_MS(self, logmstar, logsfr): 
        ''' Calculate the `distance` from the best-fit main sequence 
        '''
        if self._sfms_fit is None: 
            raise ValueError("Run `fit` and `powerlaw` methods first") 
        return logsfr - self._sfms_fit(logmstar) 

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

    def _GMM_idcomp(self, gbest, silent=True): 
        ''' Given the best-fit GMM, identify all the components
        '''
        mu_gbest = gbest.means_.flatten()
        w_gbest  = gbest.weights_
        n_gbest  = len(mu_gbest) 
        
        i_sfms = None # sfms 
        i_sb = None # star-burst 
        i_int = None # intermediate 
        i_q = None # quenched
    
        highsfr = (mu_gbest > -11) 
        if np.sum(highsfr) == 1: 
            # only one component with high sfr. This is the one 
            i_sfms = np.arange(n_gbest)[highsfr]
        elif np.sum(mu_gbest > -11) > 1: 
            # if best fit has more than one component with high SFR (logSSFR > -11), 
            # we designate the component with the highest weight as the SFMS 
            highsfr = (mu_gbest > -11)
            i_sfms = (np.arange(n_gbest)[highsfr])[w_gbest[highsfr].argmax()]
        else: 
            # no components with high sfr -- i.e. no SFMS component 
            pass 

        # lowest SSFR component with SSFR less than SFMS will be designated as the 
        # quenched component 
        if i_sfms is not None: 
            notsf = (mu_gbest < mu_gbest[i_sfms]) #& (mu_gbest < -11.) 
            if np.sum(notsf) > 0: 
                i_q = (np.arange(n_gbest)[notsf])[mu_gbest[notsf].argmin()]
                # check if there's an intermediate population 
                interm = (mu_gbest < mu_gbest[i_sfms]) & (mu_gbest > mu_gbest[i_q]) 
            #else: 
            #    interm = (mu_gbest < mu_gbest[i_sfms]) & (mu_gbest > -11.) 
                if np.sum(interm) > 0: 
                    i_int = np.arange(n_gbest)[interm]
        else: # no SFMS 
            #notsf = (mu_gbest < -11.) 
            #if np.sum(notsf) > 0: 
                #i_q = (np.arange(n_gbest)[notsf])[mu_gbest[notsf].argmin()]
            i_q = (np.arange(n_gbest))[mu_gbest.argmin()]
            # check if there's an intermediate population 
            interm = (mu_gbest > mu_gbest[i_q]) 
            if np.sum(interm) > 0: 
                i_int = np.arange(n_gbest)[interm]

        # if there's a component with high SFR than SFMS -- i.e. star-burst 
        if i_sfms is not None: 
            above_sfms = (mu_gbest > mu_gbest[i_sfms])
            if np.sum(above_sfms) > 0: 
                i_sb = np.arange(n_gbest)[above_sfms]
        return [i_sfms, i_q, i_int, i_sb] 
    
    def _check_input(self, logmstar, logsfr): 
        ''' check that input logMstar or logSFR values do not make sense!
        '''
        if len(logmstar) != len(logsfr): 
            raise ValueError("logmstar and logsfr are not the same length arrays") 
        if np.sum(logmstar < 0.) > 0: 
            raise ValueError("There are negative values of log M*")  
        if np.sum(logmstar > 13.) > 0: 
            warnings.warn("There are galaxies with log M* > 13. ... that's weird") 
        if np.sum(np.invert(np.isfinite(logsfr))) > 0: 
            raise ValueError("There are non-finite log SFR values")  
        return None 


class xdGMM(object): 
    ''' Wrapper for extreme_deconovolution. Methods are structured similar
    to GMM
    '''
    def __init__(self, n_components): 
        '''
        '''
        self.n_components = n_components
        self.l = None
        self.weights_ = None
        self.means_ = None
        self.covariances_ = None

    def fit(self, X, Xerr): 
        ''' fit GMM to X and Xerr
        '''
        X, Xerr = self._X_check(X, Xerr)
        self._X = X 
        self._Xerr = Xerr
        gmm = GMM(self.n_components, n_iter=10, covariance_type='full').fit(X)
        w, m, c = gmm.weights_.copy(), gmm.means_.copy(), gmm.covars_.copy()
        l = extreme_deconvolution(X, Xerr, w, m, c)
        self.l = l 
        self.weights_ = w 
        self.means_ = m
        self.covariances_ = c
        return None

    def logL(self, X, Xerr): 
        ''' log Likelihood of the fit. 
        '''
        if (self.l is None) or (not np.array_equal(X, self._X)) or (not np.array_equal(Xerr, self._Xerr)): 
            self.fit(X, Xerr)
        X, Xerr = self._X_check(X, Xerr)
        return self.l * X.shape[0]

    def _X_check(self, X, Xerr): 
        ''' correct array shape of X and Xerr to be compatible 
        with all the methods in this class 
        '''
        if len(X.shape) == 1: 
            X = np.reshape(X, (-1,1))
        if len(Xerr.shape) == 1:   
            Xerr = np.reshape(Xerr, (-1,1,1))
        return X, Xerr
    
    def bic(self, X, Xerr): 
        ''' calculate the bayesian information criteria
        -2 ln(L) + Npar ln(Nsample) 
        '''
        if (self.l is None) or (not np.array_equal(X, self._X)) or (not np.array_equal(Xerr, self._Xerr)): 
            self.fit(X, Xerr)
        X, Xerr = self._X_check(X, Xerr)
        assert np.array_equal(X, self._X)  
        return (-2 * self.l * X.shape[0] + self._n_parameters() * np.log(X.shape[0])) 
    
    def _n_parameters(self): 
        ''' number of paramters in the model. 
        '''
        _, n_features = self.means_.shape
        cov_params = self.n_components * n_features * (n_features + 1) / 2.
        mean_params = n_features  * self.n_components
        return int(cov_params + mean_params + self.n_components - 1)


def sfr_mstar_gmm(logmstar, logsfr, n_comp_max=30, silent=False): 
    ''' Fit a 2D gaussian mixture model to the 
    log(M*) and log(SFR) sample of galaxies, 
    '''
    # only keep sensible logmstar and log sfr
    sense = (logmstar > 0.) & (logmstar < 13) & (logsfr > -5) & (logsfr < 4) & (np.isnan(logsfr) == False)
    if (len(logmstar) - np.sum(sense) > 0) and not silent: 
        warnings.warn(str(len(logmstar) - np.sum(sense))+' galaxies have nonsensical logM* or logSFR values')  
    logmstar = logmstar[np.where(sense)]
    logsfr = logsfr[np.where(sense)]

    X = np.array([logmstar, logsfr]).T # (n_sample, n_features) 

    gmms, bics = [], []  
    for i_n, n in enumerate(range(1, n_comp_max)): 
        gmm = GMix(n_components=n)
        gmm.fit(X)
        gmms.append(gmm)
        bics.append(gmm.bic(X)) # bayesian information criteria
    ibest = np.array(bics).argmin() # lower the better!
    gbest = gmms[ibest]

    if not silent: 
        print(str(len(gbest.means_))+' components') 
    return gbest 
