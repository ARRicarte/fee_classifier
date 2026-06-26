"""
Taking outputs from Paper VIII scripts and creating a single pickle.
"""

import pickle
import numpy as np
import os
import glob
import pandas

#Standard Set
'''
infolder = '/n/home11/aricarte/projects/eht/2017_sgra_paper5/cache/koral_ricarte2022'
outfile = './data_products/paperVIII_analysis.pkl'
'''

#High-cadence
'''
infolder = '/n/home11/aricarte/projects/eht/2017_sgra_paper5/cache/koral_library_2M_M87'
outfile = './data_products/paperVIII_analysis_2M.pkl'
'''

#Variable Kappa
infolder = '/n/home11/aricarte/projects/eht/2017_sgra_paper5/cache/koral_library_varkappa_sigmacut20_M87'
outfile = './data_products/paperVIII_analysis_varkappa.pkl'

#Variable Kappa, 2M
'''
infolder = '/n/home11/aricarte/projects/eht/2017_sgra_paper5/cache/koral_library_2M_variablekappa_M87'
outfile = './data_products/paperVIII_analysis_varkappa_2M.pkl'
'''

#Excluding incomplete Rhigh=20 files.
files = [file for file in glob.glob(os.path.join(infolder, "*/*.tsv")) if 'Rh20' not in file]
D = {}
D['filenames'] = []
D['Bstate'] = []
D['spin'] = [] 
D['t'] = []
D['Rl'] = []
D['Rh'] = []
D['freq'] = []
D['inc'] = []

for i in range(len(files)):
	print(files[i])
	table = pandas.read_table(files[i])
	filename = files[i].split('/')[-1]
	D['filenames'].extend([item.split('/')[-1] for item in table['file_path']])
	D['Bstate'].extend([filename.split('_')[0]] * table.shape[0])
	D['spin'].extend([float(filename.split('_')[1].split('spin')[1])] * table.shape[0])
	D['t'].extend(table['time'].values)
	D['Rl'].extend([1.0]*table.shape[0])  #This isn't actually saved.  Assuming they are all 1.
	D['Rh'].extend([float(filename.split('_')[3].split('.')[0].split('Rh')[1])]*table.shape[0])
	D['freq'].extend([float(files[i].split('/')[-2].split('GHz')[0])]*table.shape[0])
	D['inc'].extend([float(filename.split('_')[2][1:])]*table.shape[0])

	#Now, get the actual data we want.
	for key in table.keys():
		#I don't care about these ones.
		if not key in ['file_path', 'time', 'time_hr', 'Ladv', 'Imin', 'Imax', 'Imean']:
			if not key in D.keys():
				D[key] = []
			D[key].extend(table[key].values)

for key in D.keys():
	D[key] = np.array(D[key])

with open(outfile, 'wb') as f:
	pickle.dump(D, f)
