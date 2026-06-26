import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import os
import numpy as np
import h5py
import ehtplot

def plotFrameUnpolarized(imageFile, intensityMax=None, output=None, EVPA_CONV="EofN", cmap='afmhot_us', fig_ax=None, makePlot=True, logIntensity=False, intensityDecades=3):
	#Open the IPOLE output and extract relevant data.
	with h5py.File(imageFile, 'r') as hfp:
		dx = hfp['header']['camera']['dx'][()]
		dy = hfp['header']['camera']['dy'][()]
		dsource = hfp['header']['dsource'][()]
		lunit = hfp['header']['units']['L_unit'][()]
		fov_muas = dx / dsource * lunit * 2.06265e11
		scale = hfp['header']['scale'][()]
		timeInM = hfp['header']['t'][()]
		evpa_0 = 'W'
		if 'evpa_0' in hfp['header']:
		  evpa_0 = hfp['header']['evpa_0'][()]
		I = np.copy(hfp['pol'][:,:,0]).transpose((1,0)) * scale
		pixelSize = dx * dy * (lunit / dsource * 2.06265e11)**2 / (I.shape[0] * I.shape[1])
		I /= pixelSize
	#Note the flipped x-axis.
	extent = [ fov_muas/2, -fov_muas/2, -fov_muas/2, fov_muas/2 ]

	if fig_ax is None:
		fig, ax = plt.subplots(1, 1, figsize=(5,4))
	else:
		fig, ax = fig_ax
	if intensityMax is None:
		intensityMax = np.max(I)
	if logIntensity:
		norm = LogNorm(vmax=intensityMax, vmin=intensityMax/10**intensityDecades)
	else:
		norm = Normalize(vmin=0., vmax=intensityMax)
	I_imshow = np.copy(I)
	if logIntensity:
		I_imshow[I_imshow==0] = 1e-99
		I_imshow[~np.isfinite(I_imshow)] = 1e-99
	im = ax.imshow(I_imshow, cmap=cmap, norm=norm, origin='lower', extent=extent, interpolation='nearest')

	if makePlot:
		if output is None:
			fig.show()
		else:
			fig.savefig(output, dpi=400)
			plt.close(fig)

def plotExamples(fee_files, nonfee_files, output=None, figsize=(8,3.5), startingDirectory='/n/holylfs05/LABS/bhi/Lab/narayan_lab/ipole_libraries/koral_library'):

	assert len(fee_files) == len(nonfee_files)

	fig, axarr = plt.subplots(2, len(fee_files), figsize=figsize)
	for col in range(len(fee_files)):
		plotFrameUnpolarized(os.path.join(startingDirectory, fee_files[col]), makePlot=False, fig_ax=(fig,axarr[0,col]), logIntensity=True)
		plotFrameUnpolarized(os.path.join(startingDirectory, nonfee_files[col]), makePlot=False, fig_ax=(fig,axarr[1,col]), logIntensity=True)

	for row in range(axarr.shape[0]):
		for col in range(axarr.shape[1]):
			ax = axarr[row,col]
			ax.axis('off')
	axarr[0,0].text(0.05, 0.95, "FEEs", color='white', fontsize=12, transform=axarr[0,0].transAxes, ha='left', va='top')
	axarr[1,0].text(0.05, 0.95, "Not FEEs", color='white', fontsize=12, transform=axarr[1,0].transAxes, ha='left', va='top')

	#Color bar
	fig.subplots_adjust(wspace=0,hspace=0,left=0,right=0.875,top=1,bottom=0)
	sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap('afmhot_us'), norm=LogNorm(10**(-3),1))
	sm._A = []
	cbax = fig.add_axes([0.885, 0.02, 0.03, 0.96])
	cb = fig.colorbar(sm, orientation='vertical', cax=cbax)
	cb.set_label("$I/I_\mathrm{max}$", fontsize=12, labelpad=-10)
	cb.set_ticks([10**(-3),1])

	if output is None:
		fig.show()
	else:
		fig.savefig(output)
		plt.close(fig)

if __name__ == '__main__':
	fee_table = np.loadtxt("./labels/fee_files.txt", dtype=str)
	nonfee_table = np.loadtxt("./labels/nonfee_files.txt", dtype=str)

	#Some arbitrary indices.
	plotExamples(fee_table[np.array([1,52,100,151])], nonfee_table[np.array([1,41,81,121])], output='./figures/fee_examples.pdf')
