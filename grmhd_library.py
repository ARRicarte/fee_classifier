"""
Deals with interfacing with the .h5 files.
"""

import glob
import os
import numpy as np
from tqdm import tqdm

def read_parameters_from_name(name):
	"""
	Files look like...
	MAD_spin-0.30_t15400.0_Rl1.0_Rh1.0_Munita56.41_Munitb9.07_freq228.00_inc17.00_M87.h5
	"""

	if "/" in name:
		name = name.split('/')[-1]
	chunks = name.split('_')

	output = {}
	output['Bstate'] = chunks[0]
	output['spin'] = float(chunks[1].split('spin')[1])
	output['t'] = float(chunks[2].split('t')[1])
	output['Rl'] = float(chunks[3].split('Rl')[1])
	output['Rh'] = float(chunks[4].split('Rh')[1])
	output['Munit_a'] = float(chunks[5].split('Munita')[1])
	output['Munit_b'] = float(chunks[6].split('Munitb')[1])
	output['freq'] = float(chunks[7].split('freq')[1])
	output['inc'] = float(chunks[8].split('inc')[1])

	return output


class FileMatcher(object):

	def __init__(self, startingDirectory='/n/holylfs05/LABS/bhi/Lab/narayan_lab/ipole_libraries/koral_library'):

		#Find and organize files
		allPossibleFiles = glob.glob(os.path.join(startingDirectory, "*h5"))
		Bstate_arr = []
		spin_arr = []
		t_arr = []
		Rl_arr = []
		Rh_arr = []
		freq_arr = []
		inc_arr = []
		filename_arr = []

		print("Loading GRMHD file names and parameters...")
		for file in tqdm(allPossibleFiles):
			params = read_parameters_from_name(file)

			#Apparently I had some incomplete set of Rh=20 models. Ignore them.
			if params['Rh'] == 20:
				continue
			else:
				Bstate_arr.append(params['Bstate'])
				spin_arr.append(params['spin'])
				t_arr.append(params['t'])
				Rl_arr.append(params['Rl'])
				Rh_arr.append(params['Rh'])
				freq_arr.append(params['freq'])
				inc_arr.append(params['inc'])
				filename_arr.append(file)

		self.Bstate_arr = np.array(Bstate_arr)
		self.spin_arr = np.array(spin_arr)
		self.t_arr = np.array(t_arr)
		self.Rl_arr = np.array(Rl_arr)
		self.Rh_arr = np.array(Rh_arr)
		self.freq_arr = np.array(freq_arr)
		self.inc_arr = np.array(inc_arr)
		self.filename_arr = np.array(filename_arr)

	def match_files_from_params(self, Bstate=None, spin=None, t=None, Rl=None, Rh=None, freq=None, inc=None):
		"""
		Given some parameters, return the files that match them.
		"""

		mask = np.ones(len(self.filename_arr), dtype=bool)
		if Bstate is not None:
			mask = mask & (self.Bstate_arr == Bstate)
		if spin is not None:
			mask = mask & (self.spin_arr == spin)
		if t is not None:
			mask = mask & (self.t_arr == t)
		if Rl is not None:
			mask = mask & (self.Rl_arr == Rl)
		if Rh is not None:
			mask = mask & (self.Rh_arr == Rh)
		if freq is not None:
			mask = mask & (self.freq_arr == freq)
		if inc is not None:
			mask = mask & (self.inc_arr == inc)
		return self.filename_arr[mask]
