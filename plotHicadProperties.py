import matplotlib.pyplot as plt
import numpy as np
import pickle

def plotHicadProperties(D, spin, Rhigh, output=None, ylims=[[30,80],[0,2],[30,90],[0,0.4]], comparelabels_Rh1=False, tidylabels=False, D_nonthermal=None):

	consensus = np.all(D['isFEE_model']>0.5, axis=1).astype(int)
	majority_vote = np.mean(D['isFEE_model'], axis=1)
	mask = (D['spin'] == spin) & (D['Rh'] == Rhigh) & (D['freq'] == 228)
	t = D['t'][mask]
	order = np.argsort(t)
	t -= t[order][0]

	consensus_used = consensus[mask][order]
	majority_vote_used = majority_vote[mask][order]
	if comparelabels_Rh1:
		mask_rh1 = (D['spin'] == spin) & (D['Rh'] == 1) & (D['freq'] == 228)
		t_rh1 = D['t'][mask_rh1]
		order_rh1 = np.argsort(t_rh1)
		t_rh1 -= t_rh1[order_rh1][0]
		consensus_used_rh1 = consensus[mask_rh1][order_rh1]

	if tidylabels:
		#This actually didn't do much.
		for i in range(1,len(consensus_used)-1):
			#If your neighbors are FEEs, sure call yourself a FEE too.
			if (consensus_used[i] == 0):
				if (consensus_used[i-1] == 1) & (consensus_used[i+1] == 1):
					print(i)
					consensus_used[i] == 1

			#If neither of your neighbors are FEEs, I don't believe you are.
			elif (consensus_used[i] == 1):
				if (consensus_used[i-1] == 0) & (consensus_used[i+1] == 0):
					print(i)
					consensus_used[i] == 0

	fig, axarr = plt.subplots(4, 1, figsize=(10,6), sharex=True)
	for row in range(len(axarr)):
		ax = axarr[row]
		dt = np.diff(t[order])[0]
		t_stairs = np.concatenate([[t[order][0]-dt/2],t[order]+dt/2])
		#ax.stairs((majority_vote_used>0.5).astype(int)*(ylims[row][1]-ylims[row][0])+ylims[row][0], t_stairs, color='silver', fill=True)
		ax.stairs(consensus_used*(ylims[row][1]-ylims[row][0])+ylims[row][0], t_stairs, color='silver', fill=True)
		if comparelabels_Rh1:
			ax.fill_between(t_rh1[order_rh1], ylims[row][0], consensus_used_rh1*(ylims[row][1]-ylims[row][0])+ylims[row][0], color='b', alpha=0.6)
	
	#GRMHD Quantities
	axarr[0].plot(t[order], D['phi'][mask][order], color='r', lw=1)
	axarr[0].set_ylabel(r"$\phi$", fontsize=11, color='r')

	ax_mdot = axarr[0].twinx()
	ax_mdot.plot(t[order], D['mdot_scaled'][mask][order]/np.mean(D['mdot_scaled'][mask][order]), color='k', lw=1)
	ax_mdot.set_ylabel(r"$\dot{M}_\bullet/\langle \dot{M}_\bullet \rangle$", color='k', rotation=-90, labelpad=20, fontsize=11)
	ax_mdot.set_ylim(0,3)
	ax_mdot.tick_params(direction='in')

	#Total Flux & Spectral Index
	axarr[1].plot(t[order], D['Fnu'][mask][order], color='dodgerblue', lw=1)
	if D_nonthermal is not None:
		axarr[1].plot(t[order], D_nonthermal['Fnu'][mask][order], color='dodgerblue', lw=1, alpha=0.5)
	axarr[1].set_ylabel(r"$F_{\nu}$ [Jy]", fontsize=11, color='dodgerblue')

	ax_spec = axarr[1].twinx()
	ax_spec.plot(t[order], D['alpha_230-345'][mask][order], color='purple', lw=1)
	if D_nonthermal is not None:
		ax_spec.plot(t[order], D_nonthermal['alpha_230-345'][mask][order], color='purple', lw=1, alpha=0.5)
	ax_spec.set_ylabel(r"$\alpha$", fontsize=11, color='purple', rotation=-90, labelpad=20)
	ax_spec.set_yticks([-1,0])
	ax_spec.set_ylim(-1.5,0.5)
	ax_spec.tick_params(direction='in')
	
	#FWHM
	axarr[2].plot(t[order], D['major_FWHM'][mask][order], color='brown', lw=1)
	if D_nonthermal is not None:
		axarr[2].plot(t[order], D_nonthermal['major_FWHM'][mask][order], color='brown', lw=1, alpha=0.5)
	axarr[2].set_ylabel(r"FWHM [$\mu$as]", fontsize=11, color='brown')

	#Polarization
	axarr[3].plot(t[order], D['mnet'][mask][order], label=r'$m_\mathrm{net}$', lw=1, color='tab:blue')
	axarr[3].plot(t[order], D['mavg'][mask][order], label=r'$m_\mathrm{avg}$', lw=1, color='tab:orange')
	axarr[3].plot(t[order], D['b2_amp'][mask][order], label=r'$|\beta_2|$', lw=1, color='tab:green')
	if D_nonthermal is not None:
		axarr[3].plot(t[order], D_nonthermal['mnet'][mask][order], lw=1, color='tab:blue', alpha=0.5)
		axarr[3].plot(t[order], D_nonthermal['mavg'][mask][order], lw=1, color='tab:orange', alpha=0.5)
		axarr[3].plot(t[order], D_nonthermal['b2_amp'][mask][order], lw=1, color='tab:green', alpha=0.5)
		
	axarr[3].set_ylabel(r"LP Fractions", fontsize=11)
	axarr[3].legend(loc='upper left', fontsize=9, frameon=False, ncol=3)

	for row in range(len(axarr)):
		ax = axarr[row]
		ax.set_xlim(t[order][0], t[order][-1])
		ax.set_ylim(ylims[row])
		ax.set_xticks(np.linspace(0,1e4,11))
		if row == (len(axarr)-1):
			ax.set_xlabel(r"t - $10^5$ [$t_g$]", fontsize=11)
		ax.tick_params(direction='in')

	fig.tight_layout()
	fig.subplots_adjust(hspace=0.1)
	if output is None:
		fig.show()
	else:
		fig.savefig(output)
		plt.close(fig)

if __name__ == '__main__':
	unifiedDictionary = './data_products/combinedFEEDictionary_2M_061226.pkl'
	unifiedDictionary_nt = './data_products/combinedFEEDictionary_2M_variablekappa_061226.pkl'
	with open(unifiedDictionary, 'rb') as f:
		D = pickle.load(f)
	with open(unifiedDictionary_nt, 'rb') as f:
		D_nt = pickle.load(f)

	#plotHicadProperties(D, 0.9, 160.0)#, output='./figures/full_stokes_summary_a0.9_Rh160.pdf')
	#plotHicadProperties(D, 0.9, 40.0, output='./figures/full_stokes_summary_a0.9_Rh1.pdf')
	plotHicadProperties(D, 0.9, 160.0, D_nonthermal=D_nt, output='./figures/full_stokes_summary_a0.9_Rh160.pdf')
