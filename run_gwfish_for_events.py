import GWFish.modules as gw
import os 
import pandas as pd

##################################################################################
# GENERAL SETTINGS for gw data file to analyze
##################################################################################
hi_probability = True
my_random_seed = 1
nside          = 16
# choose between 'HI' and 'NoHI':
hi_probability = 'HI' 
#hi_probability = 'NoHI'
gw_data_file = 'bbh_1e4_events_for_gwfish_rs%s_nside%s_%s' %(my_random_seed, nside, hi_probability)
waveform_model = 'IMRPhenomXPHM'
f_ref = 20.
##################################################################################
##################################################################################


# The networks are the combinations of detectors that will be used for the analysis
# The detection_SNR is the minimum SNR for a detection:
#   --> The first entry specifies the minimum SNR for a detection in a single detector
#   --> The second entry specifies the minimum network SNR for a detection
detectors = ['ET_cryo_10']
yaml_file = 'my_detectors'
ConfigDet = os.path.join('gw_data/' + yaml_file + '.yaml')
network = gw.detection.Network(detector_ids = detectors, detection_SNR = (0., 150.), config=ConfigDet)
fisher_parameters = ['mass_1', 'mass_2', 'luminosity_distance', 'theta_jn', 'ra', 'dec',
       'psi', 'phase', 'geocent_time', 'a_1', 'a_2', 'tilt_1', 'tilt_2',
       'phi_12', 'phi_jl']
df_binaries = pd.read_hdf('gw_data/' + gw_data_file + '.hdf5')
print('Analyzing data file: ', gw_data_file)
gw.fishermatrix.analyze_and_save_to_txt(network = network,
                                parameter_values  = df_binaries,
                                fisher_parameters = fisher_parameters,
                                sub_network_ids_list = [[0]],
                                population_name = 'bbh_1e4_events_rs%s_nside%s_%s' %(my_random_seed, nside, hi_probability),
                                waveform_model = 'IMRPhenomXPHM',
                                save_matrices = True,
                                use_duty_cycle = False,
                                decimal_output_format='%.15f')