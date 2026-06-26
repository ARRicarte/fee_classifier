import numpy as np
import pickle
import matplotlib.pyplot as plt
import fee_gap_analysis as gap_analysis

def plotQURotationDeviations(qu_file, fee_file, spin, Rh, freq, n_convolve=0, ylim=(-20,20)):

	with open(qu_file, 'rb') as f:
		D_qu = pickle.load(f)

	mask_qu = (D_qu['spin'] == spin) & (D_qu['Rh'] == Rh) & (D_qu['freq'] == freq)
	t = np.squeeze(D_qu['t'][mask_qu,:])
	P = np.squeeze(D_qu['P'][mask_qu,:])
	s = np.squeeze(D_qu['arclength'][mask_qu,:])
	integrated_curvature = np.squeeze(D_qu['integrated_curvature'][mask_qu,:]) * 180/np.pi
	mean_rotation = np.squeeze(D_qu['mean_rotation'][mask_qu]) * 180/np.pi

	with open(fee_file, 'rb') as f:
		D_fee = pickle.load(f)
	mask_fee = (D_fee['spin'] == spin) & (D_fee['Rh'] == Rh) & (D_fee['freq'] == freq)
	isFEE = D_fee['isFEE_model'][mask_fee][np.argsort(D_fee['t'][mask_fee])]
	consensus = np.all(isFEE>0.5, axis=1).astype(int)
	majority_vote = np.mean(isFEE, axis=1)
	
	fig, ax = plt.subplots(1, 1, figsize=(8,3))
	omega = np.gradient(integrated_curvature)/np.gradient(t)
	omega_fee = np.mean(omega[consensus==1])
	omega_nonfee = np.mean(omega[consensus==0])
	print(f"Found Omega={omega_fee:2.1f} during FEEs, Omega={omega_nonfee:2.1f} otherwise.")
	print(f"Overall, Omega={mean_rotation:2.1f}")
	if n_convolve > 0:
		omega = np.convolve(omega, np.full(n_convolve, 1/n_convolve), mode='same')
	dt = np.diff(t)[0]
	t_stairs = np.concatenate([[t[0]-dt/2],t+dt/2])
	ax.stairs((majority_vote>0.5).astype(int)*(ylim[1]-ylim[0])+ylim[0], t_stairs, color='silver', fill=True, baseline=ylim[0])
	ax.stairs(consensus*(ylim[1]-ylim[0])+ylim[0], t_stairs, color='goldenrod', fill=True, baseline=ylim[0])

	ax.plot(t, np.zeros(len(t)), color='k', ls='--', lw=1)
	ax.plot(t, np.full(len(t), mean_rotation), color='mediumblue', ls=':', lw=1)
	ax.fill_between(t, mean_rotation-np.std(omega), mean_rotation+np.std(omega), color='mediumblue', alpha=0.3)
	ax.plot(t, omega, color='mediumblue')
	ax.set_xlabel('$t \ [t_g]$', fontsize=11)
	ax.set_ylabel('$\Omega_{QU} \ [\mathrm{deg} \,t_g^{-1}]$', fontsize=11)
	ax.set_xlim(t[0],t[-1])
	ax.set_ylim(ylim)
	fig.tight_layout()
	fig.show()

def plotQURotationDifference(qu_files, fee_file, ylim=(-8,-1), output=None, windows=np.linspace(1e5,1e5+1e4,4), convolveBy=7, minimumDuration=50, afterDuration=100):

	#Let's plot only R_high=1, but all spins.

	fig, axarr = plt.subplots(2, 1, figsize=(5,5))

	with open(fee_file, 'rb') as f:
		D_fee = pickle.load(f)

	for row in range(len(axarr)):
		ax = axarr[row]
		qu_file = qu_files[row]
		with open(qu_file, 'rb') as f:
			D_qu = pickle.load(f)

		#CAREFUL: change this if necessary
		spins = [0,0.5,0.9]
		for a_index in range(len(spins)):
			spin = spins[a_index]
			omega_all = []
			omega_fee = []
			omega_nonfee = []
			omega_afterfee = []

			for t_index in range(len(windows)-1):
				mask_qu = (D_qu['spin'] == spin) & (D_qu['Rh'] == 1) & (D_qu['freq'] == 228)
				if np.sum(mask_qu) == 0:
					continue
				mask_t = (D_qu['t'][mask_qu,:] >= windows[t_index]) & (D_qu['t'][mask_qu,:] < windows[t_index+1])
				t = np.squeeze(D_qu['t'][mask_qu,:][mask_t])
				P = np.squeeze(D_qu['P'][mask_qu,:][mask_t])
				integrated_curvature = np.squeeze(D_qu['integrated_curvature'][mask_qu,:][mask_t]) * 180/np.pi
				omega = np.gradient(integrated_curvature)/np.gradient(t)

				mask_fee = (D_fee['spin'] == spin) & (D_fee['Rh'] == 1) & (D_fee['freq'] == 228)
				isFEE = D_fee['isFEE_model'][mask_fee][np.argsort(D_fee['t'][mask_fee])]
				consensus = np.all(isFEE>0.5, axis=1)[np.squeeze(mask_t)]

				#Trying to define "after" a FEE.
				consensus_used = gap_analysis.convolve_labels(consensus, convolveBy=convolveBy)
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
				durations_consensus = durations[long_enough]
				gaps_consensus = starts[long_enough][1:]-ends[long_enough][:-1]
				starts_consensus = starts[long_enough]
				ends_consensus = ends[long_enough]
				after_fee = np.zeros_like(consensus)
				for ends in ends_consensus:
					after_fee[(t>ends) & (t<=ends+afterDuration) & (~consensus)] = True

				#Append to list
				omega_all.append(np.mean(omega))
				omega_fee.append(np.mean(omega[consensus]))
				omega_nonfee.append(np.mean(omega[~consensus]))
				omega_afterfee.append(np.mean(omega[after_fee]))

			ax.errorbar(a_index-0.2, np.mean(omega_fee), np.std(omega_fee), color='k', zorder=-1, capsize=2)
			ax.errorbar(a_index-0.067, np.nanmean(omega_afterfee), np.nanstd(omega_afterfee), color='k', zorder=-1, capsize=2)
			ax.errorbar(a_index+0.067, np.mean(omega_all), np.std(omega_all), color='k', zorder=-1, capsize=2)
			ax.errorbar(a_index+0.2, np.mean(omega_nonfee), np.std(omega_nonfee), color='k', zorder=-1, capsize=2)
			ax.scatter(a_index-0.2, np.mean(omega_fee), color='b', marker='*', s=80)
			ax.scatter(a_index-0.067, np.nanmean(omega_afterfee), color='m', marker='D', s=50)
			ax.scatter(a_index+0.067, np.mean(omega_all), color='k', marker='o', s=50)
			ax.scatter(a_index+0.2, np.mean(omega_nonfee), color='r', marker='s', s=50)

		ax.set_ylabel('$\Omega_{QU} \ [\mathrm{deg} \,t_g^{-1}]$', fontsize=11)
		ax.set_xticks([0,1,2])
		ax.set_xticklabels([0,0.5,0.9])
		ax.set_ylim(ylim)
		if row == 0:
			ax.set_xticklabels([])
		else:
			ax.set_xlabel(r'$a_\bullet$', fontsize=11)

	axarr[0].text(0.03, 0.95, "Thermal Models", fontsize=11, ha='left', va='top', transform=axarr[0].transAxes)
	axarr[1].text(0.03, 0.95, "Variable $\kappa$ Models", fontsize=11, ha='left', va='top', transform=axarr[1].transAxes)
	axarr[0].scatter([], [], color='b', marker='*', label='FEE', s=80)
	axarr[0].scatter([], [], color='m', marker='D', label='Post-FEE', s=50)
	axarr[0].scatter([], [], color='k', marker='o', label='Overall', s=50)
	axarr[0].scatter([], [], color='r', marker='s', label='non-FEE', s=50)
	axarr[0].legend(frameon=False, fontsize=10, loc='upper right', ncol=2)

	fig.tight_layout()
	if output is None:
		fig.show()
	else:
		fig.savefig(output, dpi=400)
		plt.close(fig)

if __name__ == '__main__':
	qu_file = './data_products/curvatures_2M_thermal.pkl'
	fee_file = './data_products/combinedFEEDictionary_2M_012726.pkl'

	#Careful: convolving can be misleading if there are sharp turns somewhere, making them look longer.
	#plotQURotationDeviations(qu_file, fee_file, 0., 1, 228, n_convolve=0)
	#plotQURotationDeviations(qu_file, fee_file, 0.9, 1, 228, n_convolve=51)

	plotQURotationDifference(['./data_products/curvatures_2M_thermal.pkl', './data_products/curvatures_2M_nonthermal.pkl'], fee_file, output='./figures/qu_rotation_difference_min50_post100.pdf')
	#plotQURotationDifference(['./data_products/curvatures_2M_thermal_maxspeed2piover5.pkl', './data_products/curvatures_2M_nonthermal_maxspeed2piover5.pkl'], fee_file)#, output='./figures/qu_rotation_difference.pdf')
