import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import GWFish.modules as gw

def get_tmvn_samples_from_gwfish_results(data_file, fisher_matrix_file, cov_matrix_file, random_seed_run=1, snr_thr=150, nside=16, hi_probability='HI'):
    # read the gwfish results
    lbs = ['SNR', 'mass_1', 'mass_2', 'luminosity_distance', 'theta_jn', 'ra', 'dec',
       'psi', 'phase', 'geocent_time', 'a_1', 'a_2', 'tilt_1', 'tilt_2',
       'phi_12', 'phi_jl',
        'err_mass_1', 'err_mass_2', 'err_luminosity_distance', 'err_theta_jn',  'err_ra', 'err_dec', 
        'err_psi', 'err_phase', 'err_geocent_time', 'err_a_1', 'err_a_2',
        'err_tilt_1', 'err_tilt_2', 'err_phi_12', 'err_phi_jl', 'err_sky_loc']
    gwfish_errs = pd.read_csv(data_file, names=lbs, skiprows=1, delimiter=' ')    
    gwfish_obs = gwfish_errs[gwfish_errs['SNR'] > snr_thr]
    cov_matrices = np.load(cov_matrix_file)
    fisher_matrices = np.load(fisher_matrix_file)

    # filter only detected signals with SNR > snr_thr
    cov_matrices_filt = []
    fisher_matrices_filt = []
    for i, row in gwfish_errs.iterrows():
        if row['SNR'] > snr_thr:
            cov_matrices_filt.append(cov_matrices[i])
            fisher_matrices_filt.append(fisher_matrices[i])

    params = ['mass_1', 'mass_2', 'luminosity_distance', 'theta_jn', 'ra', 'dec',
       'psi', 'phase', 'geocent_time', 'a_1', 'a_2', 'tilt_1', 'tilt_2',
       'phi_12', 'phi_jl']
    

    # include priors (just uniform, to assure physical ranges)
    np.random.seed(42)
    min_array = np.array([-np.inf, -np.inf, 0.01, 0, 0, -np.pi/2, 0, 0, -np.inf, 0, 0, 0, 0, 0, 0])
    max_array = np.array([np.inf, np.inf, 50000, np.pi, 2*np.pi, np.pi/2, 2*np.pi, 2*np.pi, np.inf, 0.99, 0.99, np.pi, np.pi, 2*np.pi, 2*np.pi])

    samples_tmvn = {}
    skipped = []

    for i in tqdm(range(len(gwfish_obs))):

        try:
            mns = gwfish_obs[params].iloc[i].to_numpy()
            cov = cov_matrices_filt[i]
            epsilon = 1e-10  # small value for regularization
            regularized_cov = cov + epsilon * np.eye(cov.shape[0])
            
            tmvn = gw.minimax_tilting_sampler.TruncatedMVN(mns, regularized_cov, min_array, max_array)
            samples_tmvn[i] = tmvn.sample(5_000)
            
        except RuntimeError as e:
            print(f"Skipping index {i} due to error: {e}")
            skipped.append(i)
            continue

    # save the samples

    with open('gw_data/gwfish_results_resampled/test_samples_from_gwfish_1e4events_SNR%s_rs%s_nside%s_%s.pkl' %(snr_thr, random_seed_run, nside, hi_probability), 'wb') as f:
        pickle.dump(samples_tmvn, f)