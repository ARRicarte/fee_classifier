"""
Modified from original one on Illinois cluster.
"""

import h5py
import numpy as np
import glob
import os
import pickle
import grmhd_library as grmhd

def readLightCurveData(image):
	"""
	Extract unresoved Stokes + time.
	"""
	with h5py.File(image, 'r') as openFile:
		timeInM = openFile['header']['t'][()]
		scale = openFile['header']['scale'][()]
		imagep = openFile['pol'][()]
		I = np.sum(imagep[:,:,0]) * scale
		Q = np.sum(imagep[:,:,1]) * scale
		U = np.sum(imagep[:,:,2]) * scale
		V = np.sum(imagep[:,:,3]) * scale

	return timeInM, I, Q, U, V

def assembleQU(folder, outfile):

	#Empty dictionary
	outDict = {}
	outDict['Bstate'] = []
	outDict['spin'] = []
	outDict['Rh'] = []
	outDict['inc'] = []
	outDict['freq'] = []
	outDict['timeInM'] = []
	outDict['I'] = []
	outDict['Q'] = []
	outDict['U'] = []
	outDict['V'] = []

	#Find matching files and loop.
	files = glob.glob(os.path.join(folder, '*h5'))
	print(f"{len(files)} files identified.")
	for f in files:
		print(f"Processing {f}...")

		#Read file name
		params = grmhd.read_parameters_from_name(f)
		outDict['Bstate'].append(params['Bstate'])
		outDict['spin'].append(params['spin'])
		outDict['Rh'].append(params['Rh'])
		outDict['inc'].append(params['inc'])
		outDict['freq'].append(params['freq'])
		
		#Extract unresolved Stokes values
		timeInM, I, Q, U, V = readLightCurveData(f)
		outDict['timeInM'].append(timeInM)
		outDict['I'].append(I)
		outDict['Q'].append(Q)
		outDict['U'].append(U)
		outDict['V'].append(V)

	#Turn into numpy arrays, put in a sort of order.  If there are multiple models, masking appropriately will put their individual snapshots in order.
	for key in outDict.keys():
		outDict[key] = np.array(outDict[key])
	order = np.argsort(outDict['timeInM'])
	for key in outDict.keys():
		outDict[key] = outDict[key][order]

	#Save a pickled dictionary
	with open(outfile, 'wb') as openFile:
		pickle.dump(outDict, openFile)

if __name__ == '__main__':
	#Thermal set
	folder = '/n/holylfs05/LABS/bhi/Lab/narayan_lab/ipole_libraries/koral_library_2M_M87/'
	outfile = "./data_products/unnresolved_pol_light_curves_2M_thermal.pkl"
	assembleQU(folder, outfile)

	#Nonthermal set
	folder = '/n/holylfs05/LABS/bhi/Lab/narayan_lab/ipole_libraries/koral_library_2M_nonthermal_M87/'
	outfile = "./data_products/unnresolved_pol_light_curves_2M_nonthermal.pkl"
	assembleQU(folder, outfile)
