import numpy as np
import h5py
import grmhd_library as grmhd
import os
import matplotlib.pyplot as plt
import ehtplot

def prepare_image(ipole_file, dynamic_range=3, augmentations=True):
	"""
	Input an IPOLE file, output 8 different images, normalized between 0 and 1.
	"""

	#Open an IPOLE file.  We don't care about any of the scales.
	with h5py.File(ipole_file, 'r') as open_file:
		imagep = np.copy(open_file['pol']).transpose((1,0,2))

	#Using float32 to save memory.
	I = np.float32(imagep[:,:,0])

	#Before doing anything, make sure all values are non-negative floats.
	I[np.isnan(I)] = 0
	I[I<0] = 0

	#We want to look at this in log scale, with a specified dynamic range.  Anything below this will be 0.
	I /= np.max(I)
	logI = np.log10(I)
	logI += dynamic_range
	logI[logI<0] = 0
	logI /= dynamic_range

	#Now, all values should be between 0 and 1.  Let's return a bunch of transposes and 90 degree rotations.
	output = []
	names = []
	if augmentations:
		for k in range(4):
			output.append(np.copy(np.rot90(logI)))
			names.append(ipole_file + f"_rot{90*k}_transpose0")
			output.append(np.copy(np.rot90(np.transpose(logI))))
			names.append(ipole_file + f"_rot{90*k}_transpose1")
	else:
		output.append(np.copy(logI))
		names.append(ipole_file)

	#It's possible that you will need to perform more transformations later for the CNN, but here we have 8 images by default.
	return output, names

makeImages = False
libraries = []
libraries.append('labels')
#libraries.append('/n/holylfs05/LABS/bhi/Lab/narayan_lab/ipole_libraries/koral_library_2M_M87')
#libraries.append('/n/holylfs05/LABS/bhi/Lab/narayan_lab/ipole_libraries/koral_library')

outfolders = []
outfolders.append('image_pngs/labeled')
#outfolders.append('image_pngs/hicad')
#outfolders.append('image_pngs/complete/MAD')

for library, outfolder in zip(libraries, outfolders):
	if library == 'labels':
		#Over-ride to use labeled set.
		starting = '/n/holylfs05/LABS/bhi/Lab/narayan_lab/ipole_libraries/koral_library/'
		table_fee = np.loadtxt('./labels/fee_files_090925.txt', dtype=str).tolist()
		table_nonfee = np.loadtxt('./labels/nonfee_mad_sane_weird_090925.txt', dtype=str).tolist()
		table = table_fee + table_nonfee
		names = [starting+name for name in table if 'MAD' in name]
		np.random.seed(37)
		rng = np.random.default_rng()
		rng.shuffle(names)
	else:
		file_matcher = grmhd.FileMatcher(startingDirectory=library)
		names = file_matcher.match_files_from_params(freq=228, Rh=1, Bstate='MAD').tolist()
	print(f"Found {len(names)} images.")
	if makeImages:
		for i in range(len(names)):
			smallname = names[i].split('/')[-1].split('.h5')[0]
			print(smallname)
			outname = os.path.join(outfolder, f"{i}_{smallname}.png")
			if os.path.isfile(outname):
				continue
			im_arr = prepare_image(names[i], augmentations=False)[0][0]
			fig, ax = plt.subplots(1, 1, figsize=(4,4))
			im = ax.imshow(im_arr, cmap='afmhot_us')
			ax.axis('off')
			fig.savefig(outname, dpi=200)
			plt.close(fig)
