import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import os

def plotEruptionIncidence(infolder, colors=['r', 'orange', 'b', 'purple'], ylim=(0,1), plotIndividual=False, output=None):

	fig, axarr = plt.subplots(1, 2, figsize=(8,3))

	Bstates = ['MAD', 'SANE']
	values = [[], []]

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
		D['Rh'] = np.array([name.split('/')[-1].split('_')[4].split('Rh')[1] for name in D['names']]).astype(float).astype(int)
		D['isFEE'] = np.array(D['labels'])
		dictionary_list.append(D)

	for fold in range(len(dictionary_list)):
		D = dictionary_list[fold]

		Rhigh_unique = np.sort(np.unique(D['Rh']))
		spin_unique = np.sort(np.unique(D['spin']))
		for col in range(len(axarr)):
			Bstate = Bstates[col]
			ax = axarr[col]
			for R_high_index in range(len(Rhigh_unique)):
				fee_probability = np.array([np.mean(np.round(D['isFEE'])[(D['spin'] == spin) & (D['Bstate'] == Bstate) & (D['Rh'] == Rhigh_unique[R_high_index])]) for spin in spin_unique])
				if plotIndividual:
					ax.plot(spin_unique+0.12*(R_high_index/3-.5), fee_probability, color=colors[R_high_index], marker='none', lw=1, alpha=0.2)
				values[col].append(fee_probability)

	#Now, look at majority vote and consensus results.  Assuming Rhigh_unique and spin_unique don't change.  If they did, something went wrong anyway.
	for col in range(len(axarr)):
		Bstate = Bstates[col]
		ax = axarr[col]
		majority_vote_mean = np.zeros((len(spin_unique),len(Rhigh_unique)))
		consensus_mean = np.zeros((len(spin_unique),len(Rhigh_unique)))
		for R_high_index in range(len(Rhigh_unique)):
			Rhigh = Rhigh_unique[R_high_index]
			#Assuming all the names are the same and in the same order.  If not, something went wrong anyway.
			for spin_index in range(len(spin_unique)):
				spin = spin_unique[spin_index]
				mask = (D['spin'] == spin) & (D['Bstate'] == Bstate) & (D['Rh'] == Rhigh)
				votes = np.array([D['isFEE'][mask] for D in dictionary_list])
				majority_votes = np.mean(np.round(votes), axis=0) > 0.5
				consensus = np.all(np.round(votes), axis=0)
				majority_vote_mean[spin_index,R_high_index] = np.mean(majority_votes)
				consensus_mean[spin_index,R_high_index] = np.mean(consensus)
			ax.plot(spin_unique+0.12*(R_high_index/3-.5), majority_vote_mean[:,R_high_index], color=colors[R_high_index], marker='s', lw=1, alpha=0.4, markersize=5)
			ax.plot(spin_unique+0.12*(R_high_index/3-.5), consensus_mean[:,R_high_index], color=colors[R_high_index], marker='*', lw=1, alpha=1.0, markersize=8)
		print(f"For {Bstate} models, assuming majority_vote, the models claim an overall FEE fraction of {np.mean(majority_vote_mean):3.2f} +/- {np.std(majority_vote_mean):3.2f}")
		print(f"For {Bstate} models, assuming consensus, the models claim an overall FEE fraction of {np.mean(consensus_mean):3.2f} +/- {np.std(consensus_mean):3.2f}")

	#Fake plots for labels.
	for R_high_index in range(len(Rhigh_unique)):
		axarr[1].plot([], [], color=colors[R_high_index], label='$R_\mathrm{high}$'+f'={Rhigh_unique[R_high_index]}', alpha=1.0, lw=1)
	axarr[1].plot([], [], color='grey', label=f'Majority Vote', alpha=0.4, lw=0, marker='s', markersize=5)
	axarr[1].plot([], [], color='grey', label=f'Consensus', alpha=1.0, lw=0, marker='*', markersize=8)

	for col in range(len(axarr)):
		ax = axarr[col]
		if col == 0:
			ax.set_ylabel("Inferred FEE Snapshot Fraction")
		else:
			ax.set_yticklabels([])
		ax.set_xlabel(r"$a_\bullet$")
		ax.set_xlim(-1,1)
		ax.set_ylim(ylim)
		ax.set_xticks(spin_unique)
		if col == 1:
			ax.legend(frameon=False)
		ax.text(0.05, 0.95, Bstates[col], ha='left', va='top', transform=ax.transAxes, fontsize=11)
	fig.tight_layout()
	if output is None:
		fig.show()
	else:
		fig.savefig(output, dpi=400)
		plt.close(fig)

if __name__ == '__main__':
	infolder = './used_labels'
	plotEruptionIncidence(infolder, ylim=(0,0.6), output='./figures/eruption_incidence.pdf')
