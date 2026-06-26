import pickle
from tqdm import tqdm
import grmhd_library as grmhd
import numpy as np
import glob
import os

"""
Many different files with many origins which I want to unify.
"""

#Labels by hand.
fee_hand_labels = "./labels/fee_files_090925.txt"
nonfee_hand_labels = "./labels/nonfee_mad_sane_weird_090925.txt"

#High cadence
'''
labelFolder = "./hicad_labels"
fluxesFile = "../../data_products/koral_fluxes_fixphi_2M.pkl"
'''

#Normal cadence
labelFolder = "./used_labels"
fluxesFile = "../../data_products/koral_fluxes_fixphi.pkl"

#Variable kappa, 2M
'''
paperVIIIFile = './data_products/paperVIII_analysis_varkappa_2M.pkl'
outputName = './data_products/combinedFEEDictionary_2M_variablekappa_061226.pkl'
'''

#Thermal, 2M
'''
paperVIIIFile = './data_products/paperVIII_analysis_2M.pkl'
outputName = './data_products/combinedFEEDictionary_2M_061226.pkl'
'''

#Thermal, normal cadence
'''
paperVIIIFile = './data_products/paperVIII_analysis.pkl'
outputName = './data_products/combinedFEEDictionary_061226.pkl'
'''

#Variable kappa, normal cadence
paperVIIIFile = './data_products/paperVIII_analysis_varkappa.pkl'
outputName = './data_products/combinedFEEDictionary_variablekappa_061226.pkl'

#Many dictionaries
labelFiles = glob.glob(os.path.join(labelFolder, "*pkl"))
with open(fluxesFile, 'rb') as f:
	D_fluxes = pickle.load(f)
with open(paperVIIIFile, 'rb') as f:
	D_paperVIII = pickle.load(f)
fee_list = np.loadtxt(fee_hand_labels, dtype=str)
nonfee_list = np.loadtxt(nonfee_hand_labels, dtype=str)

#Hacky function to match parameters to the keys in the fluxes dictionary
def D_flux_key_match(Bstate, spin):
	if Bstate == 'MAD':
		if spin == 0:
			spin_string = '0'
		else:
			spin_string = f"{np.abs(spin):1.2f}".strip('0')
			if spin < 0:
				spin_string = 'm' + spin_string
		output = "analysis_ipole_a" + spin_string

		#For some reason, Ramesh added more text here.
		if "2M" in fluxesFile:
			output += "_4400-9400"

		return output
	else:
		#Currently, I don't have the fluxes.
		return None

#Little arrays I'll use to match labels later.
fee_matcher = []
nonfee_matcher = []
for item in fee_list:
	params = grmhd.read_parameters_from_name(item)
	fee_matcher.append([params['Bstate'], params['spin'], params['t']])
for item in nonfee_list:
	params = grmhd.read_parameters_from_name(item)
	nonfee_matcher.append([params['Bstate'], params['spin'], params['t']])

#Match parameters to files later
if "2M" in fluxesFile:
	if 'varkappa' in paperVIIIFile:
		image_matcher = grmhd.FileMatcher(startingDirectory='/n/holylfs05/LABS/bhi/Lab/narayan_lab/ipole_libraries/koral_library_2M_nonthermal_M87')
		image_matcher_labels = grmhd.FileMatcher(startingDirectory='/n/holylfs05/LABS/bhi/Lab/narayan_lab/ipole_libraries/koral_library_2M_M87')
	else:
		image_matcher = grmhd.FileMatcher(startingDirectory='/n/holylfs05/LABS/bhi/Lab/narayan_lab/ipole_libraries/koral_library_2M_M87')
		image_matcher_labels = image_matcher
else:
	if 'varkappa' in paperVIIIFile:
		image_matcher = grmhd.FileMatcher(startingDirectory='/n/holylfs05/LABS/bhi/Lab/narayan_lab/ipole_libraries/koral_library_varkappa_sigmacut20_M87')
		image_matcher_labels = grmhd.FileMatcher(startingDirectory='/n/holylfs05/LABS/bhi/Lab/narayan_lab/ipole_libraries/koral_library')
	else:
		image_matcher = grmhd.FileMatcher(startingDirectory='/n/holylfs05/LABS/bhi/Lab/narayan_lab/ipole_libraries/koral_library')
		image_matcher_labels = image_matcher

#Start with the labeled file.
D = {}
with open(labelFiles[0], 'rb')as f:
	D_fold = pickle.load(f)
	D['filenames'] = np.array(D_fold['names'])
N = len(D['filenames'])
D['isFEE_model'] = np.zeros((N,len(labelFiles)))
for fold in range(len(labelFiles)):
	with open(labelFiles[fold], 'rb') as f:
		D_fold = pickle.load(f)
	assert np.all(D['filenames'] == D_fold['names'])
	D['isFEE_model'][:,fold] = D_fold['labels']

#Placeholder
D['isFEE_byhand'] = np.full(N, -1.0)

#GRMHD/GRRT input parameters
D['Bstate'] = np.array(['SANE']*N)
D['spin'] = np.zeros(N, dtype=float)
D['t'] = np.zeros(N, dtype=float)
D['Rl'] = np.zeros(N, dtype=float)
D['Rh'] = np.zeros(N, dtype=float)
D['freq'] = np.zeros(N, dtype=float)
D['inc'] = np.zeros(N, dtype=float)

#A subset of interesting analysis parameters.
D['Fnu'] = np.zeros(N, dtype=float)
D['mdot_unscaled'] = np.zeros(N, dtype=float)
D['mdot_scaled'] = np.zeros(N, dtype=float)
D['phi'] = np.zeros(N, dtype=float)
D['mnet'] = np.zeros(N, dtype=float)
D['vnet'] = np.zeros(N, dtype=float)
D['mavg'] = np.zeros(N, dtype=float)
D['vavg'] = np.zeros(N, dtype=float)
for j in range(1,6):
	for component in ['amp','phase']:
		D[f'b{j}_{component}'] = np.zeros(N, float)
D['tauI'] = np.zeros(N, dtype=float)
D['tauF'] = np.zeros(N, dtype=float)
D['major_FWHM'] = np.zeros(N, dtype=float)
D['minor_FWHM'] = np.zeros(N, dtype=float)

print("Main loop...")
for i in tqdm(range(N)):
	name = D['filenames'][i]

	#First, just read some parameters.
	params = grmhd.read_parameters_from_name(name)

	#Skipping extra combos that were definitely not done for variable kappa.
	if 'varkappa' in paperVIIIFile:
		if not np.isin(params['freq'], [228,345]):
			continue
		if params['Rh'] == 20:
			continue
		if params['Bstate'] == 'SANE':
			continue

	for key in params.keys():
		if not key in ['Munit_a', 'Munit_b']:
			D[key][i] = params[key]

	#Now, let's overwrite label if possible.
	matcher = [params['Bstate'], params['spin'], params['t']]
	if matcher in fee_matcher:
		D['isFEE_byhand'][i] = 1.0
	elif matcher in nonfee_matcher:
		D['isFEE_byhand'][i] = 0.0
	if D['Bstate'][i] == 'SANE':
		D['isFEE_byhand'][i] = 0.0

	#Match columns in the fluxes file.  SANEs are missing, but we probably don't need them.
	if params['Bstate'] == 'MAD':
		subDict = D_fluxes[D_flux_key_match(params['Bstate'], params['spin'])]
		i_fluxes = np.where(np.isclose(subDict['t'], params['t'], rtol=0, atol=1))[0][0]

		D['mdot_unscaled'][i] = subDict['mdot'][i_fluxes]
		D['mdot_scaled'][i] = subDict['mdot'][i_fluxes] * np.exp(params['Munit_a'] + params['Munit_b']*params['t']/1e6)
		D['phi'][i] = subDict['phi_BH'][i_fluxes]

	#And then the Paper VIII file.  This is seemingly overly convoluted because sometimes I'll use labels from one set of ray traced images on another.
	params = grmhd.read_parameters_from_name(name)
	matchingFile = image_matcher.match_files_from_params(Bstate=params['Bstate'], spin=params['spin'], t=params['t'], Rl=params['Rl'], Rh=params['Rh'], inc=params['inc'], freq=params['freq'])[0].split('/')[-1]
	i_paperVIII = np.where(matchingFile == D_paperVIII['filenames'])[0][0]
	for pol in ['m','v']:
		for kind in ['net','avg']:
			D[pol+kind][i] = D_paperVIII[pol+kind][i_paperVIII]
	for j in range(1,6):
		for component in ['amp','phase']:
			D[f'b{j}_{component}'][i] = D_paperVIII[f'b{j}_{component}'][i_paperVIII]
	D['Fnu'][i] = D_paperVIII['Ftot'][i_paperVIII]
	D['tauI'][i] = D_paperVIII['tauI'][i_paperVIII]
	D['tauF'][i] = D_paperVIII['tauF'][i_paperVIII]
	D['major_FWHM'][i] = D_paperVIII['major_FWHM'][i_paperVIII]
	D['minor_FWHM'][i] = D_paperVIII['tauI'][i_paperVIII]
	
print("Now adding spectral index.")
#Now, loop over all the images to compute a spectral index.
D['alpha_230-345'] = np.zeros(N, dtype=float)
for i in tqdm(range(N)):
	name = D['filenames'][i]

	#First, just read some parameters.
	params = grmhd.read_parameters_from_name(name)

	#Will replace for all frequency values, but only stopping here.
	if params['freq'] != 228.0:
		continue

	allFiles = image_matcher_labels.match_files_from_params(Bstate=params['Bstate'], spin=params['spin'], t=params['t'], Rl=params['Rl'], Rh=params['Rh'], inc=params['inc'])
	allFrequencies = np.array([grmhd.read_parameters_from_name(name)['freq'] for name in allFiles])
	F228 = D['Fnu'][i]
	F345 = D['Fnu'][D['filenames'] == image_matcher_labels.match_files_from_params(Bstate=params['Bstate'], spin=params['spin'], t=params['t'], Rl=params['Rl'], Rh=params['Rh'], inc=params['inc'], freq=345)]
	alpha = (np.log10(F345)-np.log10(F228)) / (np.log10(345)-np.log10(228))

	for name in allFiles:
		D['alpha_230-345'][D['filenames'] == name] = alpha

#Get rid of potentially empty filler rows.
real = D['Fnu'] > 0
for key in D.keys():
	D[key] = D[key][real]

with open(outputName, 'wb') as openFile:
	pickle.dump(D, openFile)
	print(f"Saved to {outputName}.")
