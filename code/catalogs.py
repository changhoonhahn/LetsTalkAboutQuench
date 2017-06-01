import numpy as np 

    
    
    
class Catalog: 
    ''' class object for different catalogs
    '''

    def __init__(self, name): 


    def
    
    
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

