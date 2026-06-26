import numpy as np
import pickle
import glob
import os

'''
infile = '../../data_products/koral_library_organization.pkl'
outfile = '../../data_products/koral_fluxes.pkl'
'''
infile = '../../data_products/koral_library_organization_2M.pkl'
outfile = '../../data_products/koral_fluxes_2M.pkl'

with open(infile, 'rb') as f:
	library = pickle.load(f)

outDict = {}
for key in library['subdirectories'].keys():
	potentialFiles = glob.glob(os.path.join(library['startingDirectory'], key, 'scalars*dat'))
	if len(potentialFiles) > 1:
		print(f"I found multiple scalars files in {os.path.join(library['startingDirectory'], key)}. Using the first one arbitrarily; double-check this.")

	if len(potentialFiles) >= 1:
		print(potentialFiles[0])
		table = np.loadtxt(potentialFiles[0])
		subDict = {}
		subDict['t'] = table[:,0]
		subDict['mdot'] = table[:,2]
		subDict['phi_BH'] = table[:,8]
		subDict['spin'] = library['subdirectories'][key]['spin']
		subDict['isMAD'] = library['subdirectories'][key]['isMAD']
		outDict[key] = subDict
	else:
		print(f"Missing file for {key}.")

with open(outfile, 'wb') as f:
	pickle.dump(outDict, f)
