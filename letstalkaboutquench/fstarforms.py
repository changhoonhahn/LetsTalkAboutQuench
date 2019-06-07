'''

fStarForMS = fitting the STAR FORming Main Sequence 

'''
import numpy as np 
import scipy as sp 
import warnings 
from scipy.optimize import curve_fit
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
        self._fit_method = None
        self._fit_logm = None 
        self._fit_logssfr = None
        self._fit_logsfr = None
        self._sfms_fit = None 

    def fit(self, logmstar, logsfr, logsfr_err=None, method='gaussmix', fit_range=None, dlogm=0.2,
            Nbin_thresh=100, dev_thresh=0.5, max_comp=3, fit_error='bootstrap', n_bootstrap=None,
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
        self._dlogm = dlogm
        self._Nbin_thresh = Nbin_thresh

        # check inputs 
        self._check_input(logmstar, logsfr)  
        if logsfr_err is not None: 
            if not np.all(np.isfinite(logsfr_err)): 
                raise ValueError("There are non-finite log SFR error values")  
        if method not in ['gaussmix']: 
            raise ValueError(method+" is not one of the methods!") 
        self._fit_method = method 
        if fit_error not in ['bootstrap']: 
            raise ValueError("fitting currently only supports fit_error=bootstrap") 
    
        # fitting M* range  
        if fit_range is None: 
            # if fitting M* range is not specified, use all the galaxies
            # the fit range will be a bit padded. 
            fit_range = [int(logmstar.min()/dlogm)*dlogm, np.ceil(logmstar.max()/dlogm)*dlogm]

        mass_cut = (logmstar > fit_range[0]) & (logmstar < fit_range[1])
        if np.sum(mass_cut) == 0: 
            print("trying to fit SFMS over range %f < log M* < %f" % (fit_range[0], fit_range[1]))
            print("input spans %f < log M* < %f" % (logmstar.min(), logmstar.max()))
            raise ValueError("no galaxies within that cut!")

        # log M* binning 
        mbin_low = np.arange(fit_range[0], fit_range[1], dlogm)
        mbin_high = mbin_low + dlogm
        mbins = np.array([mbin_low, mbin_high]).T
        self._mbins = mbins
        
        # log M* bins above the threshold  
        self._mbins_nbinthresh = np.ones(mbins.shape[0]).astype(bool)
        for i in range(mbins.shape[0]): # log M* bins  
            in_mbin = (logmstar > mbins[i,0]) & (logmstar < mbins[i,1])
            if np.sum(in_mbin) < Nbin_thresh: # not enough galaxies
                self._mbins_nbinthresh[i] = False

        self._mbins_sfs = self._mbins_nbinthresh.copy() 

        if 'gaussmix' in method:
            # for each logM* bin, fit P(log SSFR) distribution using a Gaussian mixture model 
            # with at most 3 components and some common sense priors based on the fact
            # that SFMS is roughly a log-normal distribution. This does not require a log(SSFR) 
            # cut like gaussfit and negbinomfit and can be flexibly applied to a wide range of 
            # SSFR distributions.
            logm_median, gbests, nbests, _gmms, _bics = self._GMM_pssfr(logmstar, logsfr, 
                    self._mbins[self._mbins_nbinthresh,:], max_comp=max_comp)
            if logm_median[0] > 10.: 
                warnings.warn("The lowest M* bin is greater than 10^10, this may compromise the SFS identification scheme") 
            # save the bestfit GMMs
            self._gbests = gbests 
            self._gmms = _gmms
            self._bics = _bics

            # identify the SFS and other components 
            i_sfss, i_qs, i_ints, i_sbs = self._GMM_compID(gbests, dev_thresh=dev_thresh)
            self._mbins_sfs[np.where(self._mbins_nbinthresh)[0][np.array(i_sfss) == None]] = False # M* bins without SFS
            assert np.sum(self._mbins_nbinthresh) == np.sum(self._mbins_sfs) + np.sum(np.array(i_sfss) == None)

            m_gmm = np.tile(-999., (len(gbests), 8)) # positions of the best-fit GMM components in each of the logM* bins 
            s_gmm = np.tile(-999., (len(gbests), 8)) # width of the best-fit GMM components in each of the logM* bins 
            w_gmm = np.tile(-999., (len(gbests), 8)) # weights of the best-fit GMM components in each of the logM* bins 

            for i, i0, i1, i2, i3 in zip(range(len(gbests)), i_sfss, i_qs, i_ints, i_sbs): 
                if i0 is not None:
                    m_gmm[i,0] = gbests[i].means_.flatten()[i0]
                    s_gmm[i,0] = np.sqrt(UT.flatten(gbests[i].covariances_.flatten()[i0]))
                    w_gmm[i,0] = gbests[i].weights_[i0]
                if i1 is not None:
                    m_gmm[i,1] = gbests[i].means_.flatten()[i1]
                    s_gmm[i,1] = np.sqrt(UT.flatten(gbests[i].covariances_.flatten()[i1]))
                    w_gmm[i,1] = gbests[i].weights_[i1]
                if i2 is not None:
                    assert len(i2) <= 3
                    for ii, i2i in enumerate(i2): 
                        m_gmm[i,2+ii] = gbests[i].means_.flatten()[i2i]
                        s_gmm[i,2+ii] = np.sqrt(UT.flatten(gbests[i].covariances_.flatten()[i2i]))
                        w_gmm[i,2+ii] = gbests[i].weights_[i2i]
                if i3 is not None:
                    assert len(i3) <= 3
                    for ii, i3i in enumerate(i3): 
                        m_gmm[i,5+ii] = gbests[i].means_.flatten()[i3i]
                        s_gmm[i,5+ii] = np.sqrt(UT.flatten(gbests[i].covariances_.flatten()[i3i]))
                        w_gmm[i,5+ii] = gbests[i].weights_[i3i]

            # get bootstrap errors for all GMM parameters
            m_gmm_boot = np.tile(-999., (n_bootstrap, len(gbests), 8)) # positions of the bootstrap GMMs  
            s_gmm_boot = np.tile(-999., (n_bootstrap, len(gbests), 8)) # widths of the bootstrap GMMs  
            w_gmm_boot = np.tile(-999., (n_bootstrap, len(gbests), 8)) # weights of the boostrap GMMs
            for ibs in range(n_bootstrap): 
                # resample the data w/ replacement 
                i_boot = np.random.choice(np.arange(len(logmstar)), size=len(logmstar), replace=True) 
                # fit GMMs with n_best components 
                gboots = self._GMM_pssfr_nbest(logmstar[i_boot], logsfr[i_boot],
                        self._mbins[self._mbins_sfs,:], nbests=nbests)

                # identify the SFS components 
                _i_sfss, _i_qs, _i_ints, _i_sbs = self._GMM_compID(gboots, dev_thresh=dev_thresh)

                for i, i0, i1, i2, i3 in zip(range(len(gbests)), _i_sfss, _i_qs, _i_ints, _i_sbs): 
                    if i0 is not None:
                        m_gmm_boot[ibs,i,0] = gboots[i].means_.flatten()[i0]
                        s_gmm_boot[ibs,i,0] = np.sqrt(UT.flatten(gboots[i].covariances_.flatten()[i0]))
                        w_gmm_boot[ibs,i,0] = gboots[i].weights_[i0]
                    if i1 is not None:
                        m_gmm_boot[ibs,i,1] = gboots[i].means_.flatten()[i1]
                        s_gmm_boot[ibs,i,1] = np.sqrt(UT.flatten(gboots[i].covariances_.flatten()[i1]))
                        w_gmm_boot[ibs,i,1] = gboots[i].weights_[i1]
                    if i2 is not None:
                        assert len(i2) <= 3
                        for ii, i2i in enumerate(i2): 
                            m_gmm_boot[ibs,i,2+ii] = gboots[i].means_.flatten()[i2i]
                            s_gmm_boot[ibs,i,2+ii] = np.sqrt(UT.flatten(gboots[i].covariances_.flatten()[i2i]))
                            w_gmm_boot[ibs,i,2+ii] = gboots[i].weights_[i2i]
                    if i3 is not None:
                        assert len(i3) <= 3 
                        for ii, i3i in enumerate(i3): 
                            m_gmm_boot[ibs,i,5+ii] = gboots[i].means_.flatten()[i3i]
                            s_gmm_boot[ibs,i,5+ii] = np.sqrt(UT.flatten(gboots[i].covariances_.flatten()[i3i]))
                            w_gmm_boot[ibs,i,5+ii] = gboots[i].weights_[i3i]
            
            # now loop through the logM* bins and calculate bootstrap uncertainties 
            merr_boot = np.tile(-999., (len(gbests), 8))
            serr_boot = np.tile(-999., (len(gbests), 8))
            werr_boot = np.tile(-999., (len(gbests), 8))
            for i in range(len(gbests)): 
                for icomp, comp in zip([0, 1, 2, 5], [i_sfss, i_qs, i_ints, i_sbs]):
                    if comp[i] is None:
                        continue 
                    if icomp <= 1: 
                        hascomp = (m_gmm_boot[:,i,icomp] != -999.) 
                        merr_boot[i,icomp] = np.std(m_gmm_boot[hascomp,i,icomp]) 
                        serr_boot[i,icomp] = np.std(s_gmm_boot[hascomp,i,icomp]) 
                        werr_boot[i,icomp] = np.std(np.concatenate([w_gmm_boot[hascomp,i,icomp], np.zeros(np.sum(~hascomp))]))
                    else: 
                        for ii in range(len(comp[i])): 
                            hascomp = (m_gmm_boot[:,i,icomp+ii] != -999.) 
                            merr_boot[i,icomp+ii] = np.std(m_gmm_boot[hascomp,i,icomp+ii]) 
                            serr_boot[i,icomp+ii] = np.std(s_gmm_boot[hascomp,i,icomp+ii]) 
                            werr_boot[i,icomp+ii] = np.std(np.concatenate([w_gmm_boot[hascomp,i,icomp+ii], np.zeros(np.sum(~hascomp))]))

            tt_sfs  = [] # logM*, mu, sigma, weight of bestfit GMM  
            tt_q    = [] 
            tt_int  = [] 
            tt_int1 = [] 
            tt_int2 = [] 
            tt_sbs  = [] 
            tt_sbs1 = [] 
            tt_sbs2 = [] 
            tt_sfs_boot     = [] # mu_err, sigma_err, weight_err from boostrap 
            tt_q_boot       = [] 
            tt_int_boot     = [] 
            tt_int1_boot    = [] 
            tt_int2_boot    = [] 
            tt_sbs_boot     = [] 
            tt_sbs1_boot    = [] 
            tt_sbs2_boot    = [] 
            for _i, logm_med in enumerate(logm_median): 
                if i_sfss[_i] is not None: 
                    tt_sfs.append(np.array([logm_med, m_gmm[_i,0], s_gmm[_i,0], w_gmm[_i,0]]))
                    tt_sfs_boot.append(np.array([merr_boot[_i,0], serr_boot[_i,0], werr_boot[_i,0]]))
                if i_qs[_i] is not None: 
                    tt_q.append(np.array([logm_med, m_gmm[_i,1], s_gmm[_i,1], w_gmm[_i,1]]))
                    tt_q_boot.append(np.array([merr_boot[_i,1], serr_boot[_i,1], werr_boot[_i,1]]))
                if i_ints[_i] is not None: 
                    for ii, _tt_int, _tt_int_boot in zip(range(len(i_ints[_i])), [tt_int, tt_int1, tt_int2], [tt_int_boot, tt_int1_boot, tt_int2_boot]): 
                        _tt_int.append(np.array([logm_med, m_gmm[_i,2+ii], s_gmm[_i,2+ii], w_gmm[_i,2+ii]]))
                        _tt_int_boot.append(np.array([merr_boot[_i,2+ii], serr_boot[_i,2+ii], werr_boot[_i,2+ii]]))
                if i_sbs[_i] is not None: 
                    for ii, _tt_sbs, _tt_sbs_boot in zip(range(len(i_sbs[_i])), [tt_sbs, tt_sbs1, tt_sbs2], [tt_sbs_boot, tt_sbs1_boot, tt_sbs2_boot]): 
                        _tt_sbs.append(np.array([logm_med, m_gmm[_i,5+ii], s_gmm[_i,5+ii], w_gmm[_i,5+ii]])) 
                        _tt_sbs_boot.append(np.array([merr_boot[_i,5+ii], serr_boot[_i,5+ii], werr_boot[_i,5+ii]]))

            self._theta_sfs = np.array(tt_sfs) 
            self._err_sfs = np.array(tt_sfs_boot) 
            self._theta_q = np.array(tt_q) 
            self._err_q = np.array(tt_q_boot) 
            self._theta_int = np.array(tt_int) 
            self._err_int = np.array(tt_int_boot) 
            self._theta_int1 = np.array(tt_int1) 
            self._err_int1 = np.array(tt_int1_boot) 
            self._theta_int2 = np.array(tt_int2) 
            self._err_int2 = np.array(tt_int2_boot) 
            self._theta_sbs = np.array(tt_sbs) 
            self._err_sbs = np.array(tt_sbs_boot) 
            self._theta_sbs1 = np.array(tt_sbs1) 
            self._err_sbs1 = np.array(tt_sbs1_boot) 
            self._theta_sbs2 = np.array(tt_sbs2) 
            self._err_sbs2 = np.array(tt_sbs2_boot) 
        else: 
            raise NotImplementedError

        # save the fit ssfr and logm 
        self._fit_logm = self._theta_sfs[:,0]
        self._fit_logssfr = self._theta_sfs[:,1]
        self._fit_logsfr = self._fit_logm + self._fit_logssfr
        self._fit_err_logssfr = self._err_sfs[:,1] 
        return [self._fit_logm, self._fit_logsfr, self._fit_err_logssfr]

    def _GMM_pssfr(self, logmstar, logsfr, mbins, max_comp=3, n_bootstrap=10): 
        ''' Fit GMM components to P(SSFR) of given data and return best-fit
        '''
        fit_logm = [] 
        gbests, nbests = [], [] 
        _gmms, _bics = [], [] 

        # loop through log M* bins  
        for i in range(mbins.shape[0]): 
            in_mbin = (logmstar > mbins[i,0]) & (logmstar < mbins[i,1])

            x = logsfr[in_mbin] - logmstar[in_mbin] # logSSFRs
            x = np.reshape(x, (-1,1))

            # save the SFS log M* and log SSFR values 
            fit_logm.append(np.median(logmstar[in_mbin])) 
            
            # fit GMMs with a range of components 
            ncomps = range(1, max_comp+1)
            gmms, bics = [], []  
            for i_n, n in enumerate(ncomps): 
                gmm = GMix(n_components=n)
                gmm.fit(x)
                bics.append(gmm.bic(x)) # bayesian information criteria
                gmms.append(gmm)

            # components with the lowest BIC (preferred)
            i_best = np.array(bics).argmin()
            n_best = ncomps[i_best] # number of components of the best-fit 
            gbest = gmms[i_best] # best fit GMM 
            
            # save the best gmm, all the gmms, and bics 
            nbests.append(n_best) 
            gbests.append(gbest)
            _gmms.append(gmms) 
            _bics.append(bics)

        return fit_logm, gbests, nbests, _gmms, _bics
    
    def _GMM_pssfr_nbest(self, logmstar, logsfr, mbins, nbests=None): 
        ''' Fit GMM components to P(SSFR) of given data and return best-fit
        '''
        gmms = [] 
        for i in range(mbins.shape[0]): # log M* bins  
            in_mbin = (logmstar > mbins[i,0]) & (logmstar < mbins[i,1])
            X = logsfr[in_mbin] - logmstar[in_mbin] # logSSFRs
            X = np.reshape(X, (-1,1))
    
            gmm = GMix(n_components=nbests[i])
            gmm.fit(X)
            # save the best gmm, all the gmms, and bics 
            gmms.append(gmm) 
        return gmms

    def _GMM_compID(self, gbests, dev_thresh=0.5): 
        ''' Given the best-fit GMMs for all the stellar mass bins, identify the SFS 
        and the other components. STarting from the lowest M* bin, we identify
        the SFS based on the highest weight component. Then in the next M* bin, we 
        iteratively determine whether the highest weight component is with dev_thresh 
        of the previous M* bin. 
        '''
        i_sfss, i_qs, i_ints, i_sbs = [], [], [], [] 
        for ibin, gbest in enumerate(gbests): 
            mu_gbest = gbest.means_.flatten()
            w_gbest  = gbest.weights_
            n_gbest  = len(mu_gbest) 

            i_sfs, i_q, i_int, i_sb = None, None, None, None
            if ibin == 0: # lowest M* bin SFS is the highest weight bin 
                i_sfs = np.argmax(w_gbest)
                mu_sfs_im1 = mu_gbest[i_sfs]
            else: 
                i_sfs = np.argmax(w_gbest)
                i_comps = np.ones(n_gbest).astype(bool) 
                if np.abs(mu_gbest - mu_sfs_im1).min() < dev_thresh: 
                    while np.abs(mu_gbest[i_sfs] - mu_sfs_im1) > dev_thresh:  
                        i_comps = (i_comps & (np.arange(n_gbest) != i_sfs))
                        i_sfs = np.arange(n_gbest)[w_gbest == np.max(w_gbest[i_comps])]
                    mu_sfs_im1 = mu_gbest[i_sfs]
                else: 
                    i_sfs = None 
        
            # if there's a component with high SFR than SFMS -- i.e. star-burst 
            if i_sfs is not None: 
                above_sfs = (mu_gbest > mu_gbest[i_sfs])
                if np.sum(above_sfs) > 0: 
                    i_sb = np.arange(n_gbest)[above_sfs]

            # lowest SSFR component with SSFR less than SFMS will be designated as the 
            # quenched component 
            if i_sfs is not None: 
                notsf = (mu_gbest < mu_gbest[i_sfs]) #& (mu_gbest < -11.) 
                if np.sum(notsf) > 0: 
                    i_q = (np.arange(n_gbest)[notsf])[mu_gbest[notsf].argmin()]
                    # check if there's an intermediate population 
                    interm = (mu_gbest < mu_gbest[i_sfs]) & (mu_gbest > mu_gbest[i_q]) 
                    if np.sum(interm) > 0: 
                        i_int = np.arange(n_gbest)[interm]
            else: # no SFMS 
                i_q = (np.arange(n_gbest))[mu_gbest.argmin()]
                # check if there's an intermediate population 
                interm = (mu_gbest > mu_gbest[i_q]) 
                if np.sum(interm) > 0: 
                    i_int = np.arange(n_gbest)[interm]
            i_sfss.append(i_sfs)
            i_qs.append(i_q)
            i_ints.append(i_int) 
            i_sbs.append(i_sb) 
        return i_sfss, i_qs, i_ints, i_sbs

    def powerlaw(self, logMfid=None, mlim=None, silent=True): 
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
        err = self._fit_err_logssfr
        if mlim is not None: 
            mcut = ((self._fit_logm > mlim[0]) & (self._fit_logm < mlim[1])) 
            xx = xx[mcut]
            yy = yy[mcut] 
            err = err[mcut]

        # chi-squared
        chisq = lambda theta: np.sum((theta[0] * xx + theta[1] - yy)**2/err**2)

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
    
    def d_SFS(self, logmstar, logsfr, method='interpexterp', err_thresh=None, silent=True): 
        ''' Calculate the `distance` from the best-fit star-forming sequence 
        '''
        # check that .fit() has been run
        if self._fit_method is None: 
            msg_err = ''.join(["Cannot calculate the distance to the best-fit", 
                " star forming sequence without first fitting the sequence"]) 
            raise ValueError(msg_err) 

        if method == 'powerlaw':  
            # fit a powerlaw to the GMM SFS fits and then use it 
            # to measure the dSFS 
            fsfms = lambda mm: self._powerlaw_m * (mm - self._logMfid) + self._powerlaw_c 
            dsfs = logsfr - fsfms(logmstar) 

        elif method in ['interpexterp', 'nointerp']: 
            # instead of fitting a powerlaw use the actual SFS fits in order 
            # to calculate the dSFS values 

            # get stellar mass bins 
            mlow, mhigh = self._mbins.T
            n_mbins = len(mlow) 
            hasfit = np.zeros(n_mbins).astype(bool) 
            for i_m in range(n_mbins): 
                fitinmbin = ((self._fit_logm >= mlow[i_m]) & (self._fit_logm < mhigh[i_m])) 
                if np.sum(fitinmbin) == 1: 
                    hasfit[i_m] = True
                elif np.sum(fitinmbin) > 1: 
                    raise ValueError 
            mlow = mlow[hasfit]
            mhigh = mhigh[hasfit]
            n_mbins = np.sum(hasfit)
        
            # impose stellar mass limit based on the stellar mass range of the SFMS fits
            inmlim = ((logmstar > mlow.min()) & (logmstar < mhigh.max()) & np.isfinite(logsfr)) 
            if not silent: 
                print('SFMS fit ranges in logM* from %f to %f' % (mlow.min(), mhigh.max())) 
                print('%i objects are outside of this range and be assigned d_SFS = -999.' % (np.sum(~inmlim)))
            
            # error threshold to remove noisy SFMS bins
            if err_thresh is not None: 
                if self._fit_err_logssfr is None: 
                    raise ValueError("run fit with fit_error enabled")

                notnoisy = (self._fit_err_logssfr < err_thresh) 
                fit_logm = self._fit_logm[notnoisy]
                fit_logsfr = self._fit_logsfr[notnoisy]
            else: 
                fit_logm = self._fit_logm
                fit_logsfr = self._fit_logsfr

            # calculate dsfs 
            dsfs = np.tile(-999., len(logmstar))
            if method == 'interpexterp': 
                # linear interpolation with extrapolation beyond
                fsfms = sp.interpolate.interp1d(fit_logm, fit_logsfr, kind='linear', 
                        fill_value='extrapolate') 
                #if not extrap: 
                #    dsfs[inmlim] = logsfr[inmlim] - fsfms(logmstar[inmlim]) 
                dsfs = logsfr - fsfms(logmstar) 
            elif method == 'nointerp': 
                fsfms = sp.interpolate.interp1d(fit_logm, fit_logsfr, kind='nearest') 
                in_interp = (logmstar >= self._fit_logm.min()) & (logmstar <= self._fit_logm.max())
                dsfs[inmlim & in_interp] = logsfr[inmlim & in_interp] - fsfms(logmstar[inmlim & in_interp]) 
                below = (logmstar < self._fit_logm.min())
                dsfs[inmlim & below] = logsfr[inmlim & below] - \
                        self._fit_logsfr[np.argmin(self._fit_logm)]
                above = (logmstar > self._fit_logm.max())
                dsfs[inmlim & above] = logsfr[inmlim & above] - \
                        self._fit_logsfr[np.argmax(self._fit_logm)]
        return dsfs 

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

    def _GMM_idcomp(self, gbest, SSFR_cut=None, silent=True): 
        ''' Given the best-fit GMM, identify all the components
        '''
        if SSFR_cut is None: 
            SSFR_cut = -11.
        mu_gbest = gbest.means_.flatten()
        w_gbest  = gbest.weights_
        n_gbest  = len(mu_gbest) 
        
        i_sfms = None # sfms 
        i_sb = None # star-burst 
        i_int = None # intermediate 
        i_q = None # quenched
    
        highsfr = (mu_gbest > SSFR_cut) 
        if np.sum(highsfr) == 1: 
            # only one component with high sfr. This is the one 
            i_sfms = np.arange(n_gbest)[highsfr][0]
        elif np.sum(mu_gbest > SSFR_cut) > 1: 
            # if best fit has more than one component with high SFR (logSSFR > -11), 
            # we designate the component with the highest weight as the SFMS 
            highsfr = (mu_gbest > SSFR_cut)
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

    def _fit(self, logmstar, logsfr, logsfr_err=None, method=None, fit_range=None, dlogm=0.2,
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
        self._fit_method = method 

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
        mbins = np.array([mbin_low, mbin_high]).T
        self._mbins = mbins
        self._mbins_nbinthresh = np.ones(mbins.shape[0]).astype(bool)
        self._mbins_sfs = np.ones(mbins.shape[0]).astype(bool)
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
            gbests = []
            _gmms, _bics = [], [] 
            for i in range(mbins.shape[0]): # log M* bins  
                in_mbin = (logmstar > mbins[i,0]) & (logmstar < mbins[i,1])

                X = logsfr[in_mbin] - logmstar[in_mbin] # logSSFRs
                X = np.reshape(X, (-1,1))
                if method == 'gaussmix_err':
                    # error bars on of logSFR
                    Xerr = logsfr_err[in_mbin] 
                    Xerr = np.reshape(Xerr, (-1,1,1))

                if np.sum(in_mbin) < Nbin_thresh: # not enough galaxies
                    self._mbins_nbinthresh[i] = False
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
                
                # save the best gmm, all the gmms, and bics 
                gbests.append(gbest)
                _gmms.append(gmms) 
                _bics.append(bics)

                # identify the different components
                i_sfms, i_q, i_int, i_sb = self._GMM_idcomp(gbest, SSFR_cut=SSFR_cut, silent=True)
    
                if i_sfms is None: 
                    self._mbins_sfs[i] = False 
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
                
                        i_sfms_boot, _, _, _ = self._GMM_idcomp(gmm_boot, SSFR_cut=SSFR_cut, silent=True)
                        if i_sfms_boot is None: 
                            continue 
                        boot_mu_logssfr.append(UT.flatten(gmm_boot.means_.flatten()[i_sfms_boot]))
                        boot_sig_logssfr.append(np.sqrt(UT.flatten(gmm_boot.covariances_.flatten()[i_sfms_boot])))
                    fit_err_logssfr.append(np.std(np.array(boot_mu_logssfr)))
                    fit_err_sig_logssfr.append(np.std(np.array(boot_sig_logssfr)))
                else: 
                    raise NotImplementedError("not yet implemented") 
                
            self._gbests = gbests # save the bestfit GMM  
            self._gmms = _gmms
            self._bics = _bics
        else: 
            raise NotImplementedError

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
        from extreme_deconvolution import extreme_deconvolution
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
