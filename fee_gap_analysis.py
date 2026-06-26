import pickle
import numpy as np
import matplotlib.pyplot as plt

def smooth_labels(array):
	"""
	Currently unused strategy to deal with spurious positives and negatives.
	"""

	output = array.copy()
	for i in range(1,len(output)-1):
		#If both neighbors are a FEE, flip to a FEE.
		if output[i] == 0:
			if (output[i-1] == 1) & (output[i+1] == 1):
				output[i] = 1
	for i in range(1,len(output)-1):
		#If neither neighbor is a FEE, flip to non-FEE
		if output[i] == 1:
			if (output[i-1] == 0) & (output[i+1] == 0):
				output[i] = 0
	return output

def convolve_labels(array, convolveBy=10):
	return np.round(np.convolve(array, np.full(convolveBy,1/convolveBy), mode='same'))

def fee_gap_analysis(D, cleanup=True, density=False, gap_bins=np.linspace(0,2500,21), duration_bins=np.linspace(0,450,21), output=None, logx=False, convolveBy=7, Rhighs=None, minimumDuration=7):

	#Some pre-processing
	consensus = np.all(D['isFEE_model']>0.5, axis=1).astype(int)
	majority_vote = np.mean(np.round(D['isFEE_model']), axis=1) > 0.5

	if Rhighs is None:
		Rhighs = np.sort(np.unique(D['Rh']))
	spins = np.sort(np.unique(D['spin']))
	durations_consensus = [[[] for R in Rhighs] for a in spins]
	durations_majority = [[[] for R in Rhighs] for a in spins]
	gaps_consensus = [[[] for R in Rhighs] for a in spins]
	gaps_majority = [[[] for R in Rhighs] for a in spins]
	for a_index in range(len(spins)):
		for r_index in range(len(Rhighs)):
			mask = (D['spin'] == spins[a_index]) & (D['Rh'] == Rhighs[r_index]) & (D['freq'] == 228)
			if np.sum(mask) == 0:
				continue
			t = D['t'][mask]
			order = np.argsort(t)
			t = t[order]

			#Consensus
			consensus_used = consensus[mask][order]
			if cleanup:
				consensus_used = convolve_labels(consensus_used, convolveBy=convolveBy)
			starts = t[np.where(np.diff(consensus_used)==1)[0]]
			ends = t[np.where(np.diff(consensus_used)==-1)[0]]
			if consensus_used[-1] == 1:
				#We don't actually know how long this is.
				starts = starts[:-1]
			if consensus_used[0] == 1:
				#We don't actually know when this started.
				ends = ends[1:]

			while True:
				durations = ends-starts
				gaps = starts[1:] - ends[:-1]
				if np.all(gaps > minimumDuration):
					break
				else:
					toMerge = gaps <= minimumDuration
					starts = np.delete(starts, np.where(toMerge)[0][0])
					ends = np.delete(ends, np.where(toMerge)[0][0])

			long_enough = durations > minimumDuration
			durations_consensus[a_index][r_index].append(durations[long_enough])
			gaps_consensus[a_index][r_index].append(starts[long_enough][1:]-ends[long_enough][:-1])

			#Majority Vote
			majority_vote_used = majority_vote[mask][order].astype(int)
			if cleanup:
				majority_vote_used = convolve_labels(majority_vote_used, convolveBy=convolveBy)
			starts = t[np.where(np.diff(majority_vote_used)==1)[0]]
			ends = t[np.where(np.diff(majority_vote_used)==-1)[0]]
			if majority_vote_used[-1] == 1:
				#We don't actually know how long this is.
				starts = starts[:-1]
			if majority_vote_used[0] == 1:
				#We don't actually know when this started.
				ends = ends[1:]
			durations = ends-starts
			long_enough = durations > minimumDuration
			durations_majority[a_index][r_index].append(durations[long_enough])
			gaps_majority[a_index][r_index].append(starts[long_enough][1:]-ends[long_enough][:-1])
	
	fig, axarr = plt.subplots(1, 2, figsize=(8,3))

	#Only considering R_high=1
	gaps_consensus_merged = np.concatenate([gaps_consensus[i][0][0] for i in range(len(gaps_consensus))])
	gaps_majority_merged = np.concatenate([gaps_majority[i][0][0] for i in range(len(gaps_majority))])
	durations_consensus_merged = np.concatenate([durations_consensus[i][0][0] for i in range(len(durations_consensus))])
	durations_majority_merged = np.concatenate([durations_majority[i][0][0] for i in range(len(durations_majority))])

	print(f"We include a total of {len(durations_majority_merged)} events from majority, or {len(durations_consensus_merged)} from consensus.")
	print(f"{np.sum(durations_consensus_merged>=convolveBy*2)} have a duration of at least {convolveBy*2*2} M according to consensus.  The median duration is {np.median(durations_consensus_merged)}, and the median gap is {np.median(gaps_consensus_merged)}")
	print(f"{np.sum(durations_majority_merged>=convolveBy*2)} have a duration of at least {convolveBy*2*2} M according to majority. The median duration is {np.median(durations_majority_merged)}, and the median gap is {np.median(gaps_majority_merged)}.")

	axarr[0].hist(gaps_consensus_merged, bins=gap_bins, alpha=0.7, color='darkorange', density=density, zorder=1)
	axarr[0].hist(gaps_majority_merged, bins=gap_bins, alpha=0.7, color='dodgerblue', density=density, zorder=0)
	axarr[0].set_xlim(gap_bins[0], gap_bins[-1])
	axarr[0].text(0.95, 0.95, "Gaps Between FEEs", fontsize=11, ha='right', va='top', transform=axarr[0].transAxes)

	axarr[1].hist(durations_consensus_merged, bins=duration_bins, alpha=0.7, color='darkorange', density=density, label='Consensus', zorder=1)
	axarr[1].hist(durations_majority_merged, bins=duration_bins, alpha=0.7, color='dodgerblue', density=density, label='Majority Vote', zorder=0)
	axarr[1].set_xlim(duration_bins[0], duration_bins[-1])
	axarr[1].text(0.95, 0.95, "Durations of FEEs", fontsize=11, ha='right', va='top', transform=axarr[1].transAxes)
	axarr[1].legend(frameon=False, loc='center right')

	for row in range(len(axarr)):
		axarr[row].set_yscale('log')
		axarr[row].set_xlabel(r'$\Delta t \ [t_g]$', fontsize=11)
		axarr[row].set_ylabel('Number in Bin', fontsize=11)
		if logx:
			axarr[row].set_xscale('log')

	fig.tight_layout()
	if output is None:
		fig.show()
	else:
		fig.savefig(output, dpi=300)
		plt.close(fig)

if __name__ == '__main__':
	unifiedDictionary = './data_products/combinedFEEDictionary_2M_061226.pkl'
	with open(unifiedDictionary, 'rb') as f:
		D = pickle.load(f)

	#fee_gap_analysis(D, gap_bins=np.logspace(np.log10(2),np.log10(2500),21), duration_bins=np.logspace(np.log10(2),np.log10(350),21), logx=True, output='./figures/fee_gap_analysis_log.pdf')
	fee_gap_analysis(D, output='./figures/fee_gap_analysis.pdf')
	'''
	fee_gap_analysis(D, realness_cutoff=20)
	'''
