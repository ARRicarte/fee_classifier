from computeAllLabels import computeAllLabels
import os
import pickle
import sys
import glob
import numpy as np

def computeAllLabels_k(infolder, outfolder, k, calibrated=True, dataset='complete'):

	if calibrated:
		#Calibration
		temperature_file = './used_temperature/temperatures.pkl'
		with open(temperature_file, 'rb') as f:
			D = pickle.load(f)
		index = np.where([f'fold{k}' in file for file in D['calibration_files']])[0][0]
		temperature = D['temperatures'][index]
	else:
		temperature = 1.0

	all_models = np.sort(glob.glob(os.path.join(infolder, "*pth")))
	infile = all_models[k]
	outfile = os.path.join(outfolder, infile.split('/')[-1].replace('.pth', '.pkl'))

	computeAllLabels(infile, outfile, calibratedTemperatureParameters=[temperature,0,0], dataset=dataset)

if __name__ == '__main__':
	#Assuming we're only doing one of these models.
	k = int(sys.argv[1])

	infolder = './used_models'
	outfolder = './hicad_labels'
	dataset = 'hicad'
	'''
	outfolder = './used_labels'
	dataset = 'complete'
	'''

	computeAllLabels_k(infolder, outfolder, k, calibrated=False, dataset=dataset)
