import prepare_files as file_prep 
import grmhd_library as grmhd
import glob
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

def visualizeLabels(infolder, outfolder, totalNumber=200, fee_labels='./labels/fee_files_090925.txt', nonfee_labels='./labels/nonfee_mad_sane_weird_090925.txt', 
	startingDirectory='/n/holylfs05/LABS/bhi/Lab/narayan_lab/ipole_libraries/koral_library'):

	#Load and reorganize a little.
	dictionary_list = []
	for file in glob.glob(os.path.join(infolder, '*pkl')):
		with open(file, 'rb') as f:
			D = pickle.load(f)

		#Reduce so that we only have 228 GHz.
		mask = np.array(['freq228.00' in name for name in D['names']])
		D['names'] = np.array(D['names'])[mask]
		D['labels'] = np.array(D['labels'])[mask]

		#Ack, replace this when I use the new combined dictionary...
		D['Bstate'] = np.array([name.split('/')[-1].split('_')[0] for name in D['names']])
		D['spin'] = np.array([name.split('/')[-1].split('_')[1].split('spin')[1] for name in D['names']]).astype(float)
		D['time'] = np.array([name.split('/')[-1].split('_')[2].split('t')[1] for name in D['names']]).astype(float)
		D['Rh'] = np.array([name.split('/')[-1].split('_')[4].split('Rh')[1] for name in D['names']]).astype(float).astype(int)
		D['parameter_combo'] = [[D['Bstate'][i], D['spin'][i], D['time'][i]] for i in range(len(D['Bstate']))]
		D['isFEE'] = np.array(D['labels'])
		dictionary_list.append(D)

	#Construct majority vote and consensus lists.
	votes = np.array([D['isFEE'] for D in dictionary_list])
	majority_vote = np.mean(votes, axis=0)
	consensus = np.all(np.round(votes), axis=0)

	#Truth values
	nonfee_names = np.loadtxt(nonfee_labels, dtype=str)
	fee_names = np.loadtxt(fee_labels, dtype=str)
	parameters_nonfee = [grmhd.read_parameters_from_name(name) for name in nonfee_names]
	parameters_fee = [grmhd.read_parameters_from_name(name) for name in fee_names]
	parameter_combos_nonfee = [[p['Bstate'], p['spin'], p['t']] for p in parameters_nonfee]
	parameter_combos_fee = [[p['Bstate'], p['spin'], p['t']] for p in parameters_fee]
	
	#Now, select some number and analyze
	randomIndices = np.random.randint(0, len(D['names']), size=totalNumber)
	for i in randomIndices:
		name = D['names'][i]
		parameter_combo = D['parameter_combo'][i]
		print(name)
		fig, ax = plt.subplots(figsize=(6,6))
		image = file_prep.prepare_image(name, augmentations=False)[0][0]
		ax.imshow(image, origin='lower', cmap='afmhot_us')
		ax.axis('off')
		ax.text(0.05, 0.95, f"{D['Bstate'][i]}, {D['spin'][i]}, {D['Rh'][i]}, {D['time'][i]}", transform=ax.transAxes, ha='left', va='top', fontsize=11, color='dodgerblue')
		ax.text(0.05, 0.92, f"Average: {majority_vote[i]:1.2f}", transform=ax.transAxes, ha='left', va='top', fontsize=11, color='dodgerblue')
		ax.text(0.05, 0.89, f"Consensus: {consensus[i]:d}", transform=ax.transAxes, ha='left', va='top', fontsize=11, color='dodgerblue')
		
		if parameter_combo in parameter_combos_nonfee:
			truth = '0'
		elif parameter_combo in parameter_combos_fee:
			truth = '1'
		else:
			truth = '?'
		ax.text(0.05, 0.86, f"True Label:  {truth}", transform=ax.transAxes, ha='left', va='top', fontsize=11, color='dodgerblue')

		fig.tight_layout()
		if outfolder is not None:
			fig.savefig(os.path.join(outfolder, name.split('/')[-1].replace(".h5",".pdf")), dpi=300)
		else:
			fig.show()
			input()

if __name__ == '__main__':
	infolder = './used_labels'
	outfolder = './label_inspection'
	visualizeLabels(infolder, outfolder, totalNumber=500)
	#visualizeLabels(infolder, None, totalNumber=100)
