# arrays and stuff
import numpy as np
import numpy.ma as ma

# miscellaneous plotting stuff
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

# fits files and fitting
from astropy.io import fits
import scipy.interpolate as interp
import scipy.signal as signal
from scipy.optimize import curve_fit

# font matters
from matplotlib import rc
font = {'family':'serif','size':12}
rc('font',**font)

# this project
import fsps
import h5py

age_bounds = np.array(((0.000, 0.0025, 0.005), (0.005, 0.010, 0.015), (0.015, 0.020, 0.025), (0.025, 0.030, 0.035), 
                       (0.035, 0.0400, 0.045), (0.045, 0.050, 0.055), (0.055, 0.060, 0.065), (0.065, 0.070, 0.075), 
                       (0.075, 0.0800, 0.085), (0.085, 0.090, 0.095), (0.095, 0.100, 0.125), (0.125, 0.150, 0.175), 
                       (0.175, 0.20, 0.225),   (0.225, 0.250, 0.275), (0.275, 0.300, 0.325), (0.325, 0.350, 0.375), 
                       (0.375, 0.40, 0.425),   (0.425, 0.450, 0.475), (0.475, 0.5125, 0.55), (0.550, 0.600, 0.650), 
                       (0.650, 0.70, 0.750),   (0.75 , 0.80,  0.85),  (0.85 , 0.90,  0.95),  (0.95, 1.0375, 1.125), 
                       (1.125, 1.25, 1.375),   (1.375, 1.50,  1.625), (1.625, 1.75, 1.875),  (1.875, 2.00, 2.125),    
                       (2.125, 2.25, 2.375),   (2.375, 2.50, 2.625),  (2.625, 2.75, 2.875),  (2.875, 3.00, 3.125),   
                       (3.125, 3.25, 3.375),   (3.375, 3.50, 3.625),  (3.625, 3.75, 3.875),
                       (3.875, 4.00, 4.25),    (4.25, 4.50, 4.75),    (4.75, 5.00, 5.25), (5.25, 5.50, 5.75), 
                       (5.75 , 6.00, 6.25),    (6.25, 6.50, 6.75),    (6.75 , 7.00, 7.25), (7.25 , 7.50, 7.75),
                       (7.75 , 8.00, 8.25), (8.25 , 8.5, 8.75), (8.75 ,9.0, 9.25), (9.25 , 9.5, 9.75),
                       (9.75 , 10.0, 10.25), (10.25 , 10.5, 10.75), (10.75 , 11.0, 11.25), (11.25 , 11.5, 11.75),
                       (11.75 , 12.0, 12.25), (12.25 , 12.5, 12.75), (12.75 , 13.0, 13.25), (13.25 , 13.5,13.75),
                       (13.75, 14.0, 14.0)))



met_center_bins = [-2.5, -2.05, -1.75, -1.45, -1.15, -0.85, -0.55, -0.35, -0.25, -0.15, 
                   -0.05, 0.05, 0.15, 0.25, 0.4, 0.5]

sp = fsps.StellarPopulation(zcontinuous=1,sfh=3)
test_age = age_bounds[:,1]
fsps_age = 14-test_age[::-1]

#num = [0,1000,2000,4000,6000,7000,8000,9000,10000,11000,12000,13000,14000,15000,16000,17000,18000,19000,20000,21000,22000,23000,24000,25000,26000,26500,27000,27500,28000,28500,29000,29400,29800]
#num = [29800,29809]
num = [29810,29811]
for filenr in range(len(num)-1):
    print filenr, ": ", num[filenr]
    f = h5py.File('EAGLE/%dEAGLE_SFRHs.hdf5' %(num[filenr]),'r')
    UVJ = np.zeros((num[filenr+1]-num[filenr], 3))
    spectra = np.zeros((num[filenr+1]-num[filenr], 4192))
    
    for gal in range(num[filenr+1]-num[filenr]):
        sp.params['add_neb_emission'] = True
        sp.params['add_neb_continuum'] = True

        specs_em = np.zeros((16,4192))
        U = np.zeros((16))
        V= np.zeros((16))
        J= np.zeros((16))
        for i in range(16):
            sfr = f['SFRH'][gal][:,i]
            sp.params['logzsol'] = met_center_bins[i]
            #sp.set_tabular_sfh(fsps_age,np.flip(sfr,axis=0))
            sfh = np.concatenate((np.reshape(fsps_age, (56,1)), np.reshape(sfr[::-1],(56,1))), axis=1)
            sp.set_tabular_sfh(sfh[:,0],sfh[:,1])
            wave, spec = sp.get_spectrum() 
            U[i] = -2.5*np.log10(np.sum(10**(sp.get_mags(bands=['u'])/-2.5)))
            V[i] = -2.5*np.log10(np.sum(10**(sp.get_mags(bands=['v'])/-2.5)))
            J[i] = -2.5*np.log10(np.sum(10**(sp.get_mags(bands=['2mass_j'])/-2.5)))
            optical = (wave > 3700) & (wave < 9000)
            specs_em[i] = np.sum(spec[:,optical], axis=0)

        Utot = -2.5*np.log10(np.sum(10**(U*1.0/-2.5)))
        Vtot = -2.5*np.log10(np.sum(10**(V*1.0/-2.5)))
        Jtot = -2.5*np.log10(np.sum(10**(J*1.0/-2.5)))
#        print 'UVJ: ', Utot,Vtot,Jtot
        UVJ[gal,:] = [Utot, Vtot, Jtot]
        spectra[gal,:] = np.sum(specs_em,axis=0)
    np.savetxt('EAGLE_spectra_%05d.txt'%(num[filenr]), spectra)
    np.savetxt('EAGLE_UVJ_%05d.txt'%(num[filenr]), UVJ)
