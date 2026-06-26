"""
On Jan 30th 2026, I realized I accidentally made a bunch of images labeled spin 0.3 and 0.7 that were actually just copies of 0.5.  Need to get rid of them.
"""

import pickle
import os
import glob

infolder = "./hicad_labels_fake0.30.7"
outfolder = "./hicad_labels"
files = glob.glob(os.path.join(infolder, '*.pkl'))
for file in files:
	print(file)
	with open(file, 'rb') as f:
		D_old = pickle.load(f)

	D_new = {}
	D_new['calibratedTemperatureParameters'] = D_old['calibratedTemperatureParameters']
	D_new['names'] = []
	D_new['labels'] = []
	for i in range(len(D_old['names'])):
		if "spin0.3" in D_old['names'][i]:
			continue
		elif "spin0.7" in D_old['names'][i]:
			continue
		else:
			D_new['names'].append(D_old['names'][i])
			D_new['labels'].append(D_old['labels'][i])
		
	with open(os.path.join(outfolder, file.split('/')[-1]), 'wb') as f:
		pickle.dump(D_new, f)
