import os
import numpy as np
import pandas as pd
import GWFish.modules as gw
from joblib import Parallel, delayed

inj_df = pd.read_hdf('gw_data/bbh_1e7_injections_for_gwfish_5_300_Msol.hdf5')

waveform_model = 'IMRPhenomXPHM'
f_ref = 20.

detectors = ['ET_cryo_10']
yaml_file = 'my_detectors'
ConfigDet = os.path.join(
    'gw_data/', yaml_file + '.yaml'
)

network = gw.detection.Network(
    detector_ids=detectors,
    detection_SNR=(0., 8.),
    config=ConfigDet
)

def compute_snr(chunk):
    snr = gw.utilities.get_snr(chunk, network, waveform_model, f_ref)
    return snr['ET_cryo_10'].to_numpy()

# Split into chunks (tune chunk size!)
n_jobs = os.cpu_count()
chunks = np.array_split(inj_df, n_jobs * 4)

snr_list = Parallel(n_jobs=n_jobs, backend="loky")(
    delayed(compute_snr)(chunk) for chunk in chunks
)

snr_all = np.concatenate(snr_list)

df_binaries = inj_df.copy()
df_binaries['SNR'] = snr_all

df_binaries.to_hdf(
    'results/bbh_1e7_injections_for_gwfish_SNR_5_300_Msol_SNR.hdf5',
    key='df',
    mode='w'
)
