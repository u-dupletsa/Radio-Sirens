import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

import icarogw
import healpy as hp
import numpy as np
from  astropy.cosmology import FlatLambdaCDM
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import bilby
import GWFish.modules as gw
import os
import pickle


##########################################################################
##########################################################################
# INITIAL CONFIGURATION
##########################################################################
H0_value = 67.7
nside = 16
random_seed_gw_run = 1
snr_thr = 150
use_HI_in_likelihood = False
use_HI_in_gw_data = True
inference_on_bias_parameters = False
perfect_PEs = False
# icarogw settings
my_nparallel=5000 # number of PE samples per each GW event (1 if perfect_PEs=True)
my_neffPE=10 # number of effective PEs (-1 if perfect_PEs=True)
my_neffINJ=None # number of effective injections
##########################################################################
if use_HI_in_likelihood:
    cosmoPE_lbl = 'HI'
else:
    cosmoPE_lbl = 'NoHI'

if use_HI_in_gw_data:
    gw_lbl = 'HI'
else:
    gw_lbl = 'NoHI'
##########################################################################
##########################################################################



##########################################################################
# Reference cosmology and maps
##########################################################################
npix = hp.nside2npix(nside)
lmax = 3*nside 

# redhsift range for reference
zmin = 0.005
zmax = 3.025

# Reference cosmology
cosmo_ref = icarogw.cosmology.astropycosmology(zmax=20.)
cosmo_ref.build_cosmology(FlatLambdaCDM(H0=H0_value,Om0=0.308)) 

PATH_TO_DENSITY_MATRIX = 'hi_data/'
if use_HI_in_likelihood:
    density_matrix = pd.read_hdf(PATH_TO_DENSITY_MATRIX + 'map4icaro_lmax' + str(lmax) + '_zmin' + str(zmin) + '_zmax' + str(zmax) + '_nside' + str(nside) + '.hdf5').to_numpy()
else:
    density_matrix = pd.read_hdf(PATH_TO_DENSITY_MATRIX + 'map4icaro_lmax' + str(lmax) + '_zmin' + str(zmin) + '_zmax' + str(zmax) + '_nside' + str(nside) + '.hdf5').to_numpy()
    density_matrix = np.ones_like(density_matrix)

redshift_grid = np.array([0.005, 0.015, 0.035, 0.06 , 0.085, 0.11 , 0.135, 0.17 , 0.205,
        0.24 , 0.28 , 0.32 , 0.36 , 0.405, 0.455, 0.505, 0.56 , 0.62 ,
        0.68 , 0.745, 0.82 , 0.895, 0.975, 1.06 , 1.155, 1.26 , 1.365,
        1.485, 1.615, 1.755, 1.905, 2.05 , 2.205, 2.39 , 2.585, 2.795,
        3.025])
pixel_grid = np.arange(0, npix, 1).astype(int) # generates an array of (integer) numbers from 0 to npix-1
HI = icarogw.HI.HI_map(redshift_grid, pixel_grid, density_matrix , bGW = 1., alphaGW = 0.)
##########################################################################


##########################################################################
# INJECTION PART
##########################################################################
# read injection results
df_inj_snr = pd.read_hdf('gw_data/bbh_1e7_injections_for_gwfish_SNR_results_5_300_Msol.hdf5')
injections_dict = {'luminosity_distance':df_inj_snr['luminosity_distance'].to_numpy(),
                'right_ascension':df_inj_snr['ra'].to_numpy(),
                'declination':df_inj_snr['dec'].to_numpy(),
                'snr': df_inj_snr['SNR'].to_numpy()
                }
Ninjections = int(1e7)
prior = df_inj_snr['inj_prior'].to_numpy()
inj = icarogw.injections.injections(injections_dict, prior, Ninjections, 1.)
print('You have this number of detected injections ', 
    len(np.where(injections_dict['snr']>snr_thr)[0]))
inj.update_cut(injections_dict['snr']>snr_thr)
inj.pixelize(nside)
##########################################################################


##########################################################################
# PE part
##########################################################################
# Results from Fisher runs
error_file = 'gw_data/gwfish_results_raw/Errors_ET_cryo_10_bbh_1e4_events_rs%s_nside%s_%s_SNR%s.txt' %(random_seed_gw_run, nside, gw_lbl, snr_thr)
fisher_file = 'gw_data/gwfish_results_raw/fisher_matrices_ET_cryo_10_bbh_1e4_events_rs%s_nside%s_%s_SNR%s.npy' %(random_seed_gw_run, nside, gw_lbl, snr_thr)
inv_fisher_file = 'gw_data/gwfish_results_raw/inv_fisher_matrices_ET_cryo_10_bbh_1e4_events_rs%s_nside%s_%s_SNR%s.npy' %(random_seed_gw_run, nside, gw_lbl, snr_thr)
if perfect_PEs:
    # read the gwfish results
    lbs = ['SNR', 'mass_1', 'mass_2', 'luminosity_distance', 'theta_jn', 'ra', 'dec',
       'psi', 'phase', 'geocent_time', 'a_1', 'a_2', 'tilt_1', 'tilt_2',
       'phi_12', 'phi_jl',
        'err_mass_1', 'err_mass_2', 'err_luminosity_distance', 'err_theta_jn',  'err_ra', 'err_dec', 
        'err_psi', 'err_phase', 'err_geocent_time', 'err_a_1', 'err_a_2',
        'err_tilt_1', 'err_tilt_2', 'err_phi_12', 'err_phi_jl', 'err_sky_loc']
    gwfish_errs = pd.read_csv(error_file, names=lbs, skiprows=1, delimiter=' ')  
    gwfish_obs = gwfish_errs[gwfish_errs['SNR'] > snr_thr]

    posteriors = {}
    for key in range(len(gwfish_obs)):
        posterior_dict = {'luminosity_distance':np.array([gwfish_obs.iloc[key]['luminosity_distance']]),
                        'right_ascension':np.array([gwfish_obs.iloc[key]['ra']]),
                        'declination':np.array([gwfish_obs.iloc[key]['dec']])}

        prior = np.ones_like(posterior_dict['luminosity_distance'])
        
        posteriors[str(key)] = icarogw.posterior_samples.posterior_samples(posterior_dict, prior)

    posteriors = icarogw.posterior_samples.posterior_samples_catalog(posteriors)
    posteriors.pixelize(nside)
    
else:
    # check if the pickle samples are in place, otherwise compute
    filename = 'gw_data/gwfish_results_resampled/samples_from_gwfish_1e4events_SNR%s_rs%s_nside%s_%s.pkl' %(snr_thr, random_seed_gw_run, nside, gw_lbl)

    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            samples_tmvn = pickle.load(f)
        print(f"Loaded: {filename}")
    else:
        print(f"Generating samples for: {filename}")
        from get_samples_from_gwfish import *
        get_tmvn_samples_from_gwfish_results(error_file, fisher_file, inv_fisher_file, random_seed_run=random_seed_gw_run, snr_thr=snr_thr, nside=nside, hi_probability=gw_lbl)
        with open(filename, 'rb') as f:
            samples_tmvn = pickle.load(f)
        print(f"Loaded: {filename}")
        
    posteriors = {}
    for key in samples_tmvn:
        posterior_dict = {'luminosity_distance':samples_tmvn[key][2,:],
                        'right_ascension':samples_tmvn[key][4,:],
                        'declination':samples_tmvn[key][5,:]}

        prior = np.ones_like(posterior_dict['luminosity_distance'])
        
        posteriors[str(key)] = icarogw.posterior_samples.posterior_samples(posterior_dict, prior)

    posteriors = icarogw.posterior_samples.posterior_samples_catalog(posteriors)
    posteriors.pixelize(nside)
##########################################################################



##########################################################################
# icarogw likelihood and run
cw = icarogw.wrappers.FlatLambdaCDM_wrap(zmax=20.)
rw = icarogw.wrappers.rateevolution_Madau()
rw.update(gamma=2.7,kappa=6.,zp=1.)
rate = icarogw.rates.CBC_HI_vanilla_rate(HI,cw,rw,scale_free=True)

likelihood = icarogw.likelihood.hierarchical_likelihood(posteriors,
                                                       inj,
                                                       rate,
                                                       nparallel=my_nparallel,
                                                       neffPE=my_neffPE, 
                                                       neffINJ=my_neffINJ) 

if inference_on_bias_parameters:
    priors_dict={'H0':bilby.prior.Uniform(10.,240.),
            'Om0':bilby.prior.Uniform(0.1,0.9),
            'gamma':bilby.prior.Uniform(0.,12.),
            'kappa':bilby.prior.Uniform(0.,12.),
            'zp':bilby.prior.Uniform(0.,4.),
            'bGW': bilby.prior.Uniform(0.1, 5.0),  
            'alphaGW': bilby.prior.Uniform(-1.0, 1.0)}
else:
    priors_dict={'H0':bilby.prior.Uniform(10.,240.),
            'Om0':bilby.prior.Uniform(0.1,0.9),
            'gamma':bilby.prior.Uniform(0.,12.),
            'kappa':bilby.prior.Uniform(0.,12.),
            'zp':bilby.prior.Uniform(0.,4.),
            'alphaGW':0.,
            'bGW':1.}

result=bilby.run_sampler(likelihood, priors_dict, sampler='nessai', nlive=1000, npool=4, 
                         outdir='results/cosmoPEs_snr%s_rs%s_nside%s_%s%s' %(snr_thr, random_seed_gw_run, nside, gw_lbl, cosmoPE_lbl))
result.plot_corner()
##########################################################################