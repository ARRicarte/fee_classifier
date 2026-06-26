import grmhd_library as grmhd
import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split, GroupShuffleSplit, GroupKFold
from tqdm import tqdm
import h5py
import os

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

"""
Define custom dataset, which we'll use for both the training and testing sets.
"""

class CustomDataset(Dataset):
	def __init__(self, images, labels, names, transform=None):
		self.images = images
		self.labels = labels
		self.names = names
		self.transform = transform

	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):
		image = self.images[idx]
		label = self.labels[idx]
		name = self.names[idx]

		if self.transform:
			image = self.transform(image)

		return image, label, name

"""
Version that only saves file names and opens them on the fly.
"""

class LazyDataset(Dataset):
	def __init__(self, paths, labels, transform=None, dynamic_range=3, augmentations=True):
		self.paths = paths
		self.labels = labels
		self.transform = transform
		self.dynamic_range = dynamic_range
		self.augmentations = augmentations

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, idx):
		path = self.paths[idx]
		label = self.labels[idx]
		image = prepare_image(path, dynamic_range=self.dynamic_range, augmentations=self.augmentations)[0][0]

		if self.transform:
			image = self.transform(image)

		return image, label, path

transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.RandomResizedCrop(#Want generality with respect to shadow size.  Hopefully this is enough.
		size=224,
		scale=(0.8,1.0),
		ratio=(0.9,1.1)
	),
	transforms.Normalize(mean=0.449, std=0.226) #Mean of RGB weights.
])

transform_noscale = transforms.Compose([
	transforms.ToTensor(),
	transforms.RandomResizedCrop(#No cropping.  Appropriate for test images.
		size=224,
		scale=(1.0,1.0),
		ratio=(1.0,1.0)
	),
	transforms.Normalize(mean=0.449, std=0.226), #Mean of RGB weights
])

def generate_dataset(fee_list="./labels/fee_files.txt", nonfee_list="./labels/nonfee_files.txt", dynamic_range=3, batch_size=32, n_threads=1, k=0, n_splits=5):

	images = []
	labels = []
	names = []
	image_matcher = grmhd.FileMatcher()

	isFee_list = [1,0]
	#For both files,
	for textfile, isFee in zip([fee_list,nonfee_list],isFee_list):
		file_list = np.loadtxt(textfile, dtype=str)
		#For each file referenced in the file,
		print(f"Loading from {textfile}...")
		for handpicked_file in tqdm(file_list):
			params = grmhd.read_parameters_from_name(handpicked_file)
			allPossibleFiles = image_matcher.match_files_from_params(Bstate=params['Bstate'], spin=params['spin'], t=params['t'], freq=params['freq'])
			assert(len(allPossibleFiles)==4)
			#For each combination of Rhigh,
			for file in allPossibleFiles:
				modifiedImages, modifiedNames = prepare_image(file, dynamic_range=dynamic_range)
				images.extend(modifiedImages)
				labels.extend([isFee]*len(modifiedImages))
				names.extend(modifiedNames)

	#Split into training and testing randomly.  I need to be careful to keep the copies in their own bins.
	group_size = 32 #2 transposes, 4 rotations, 4 Rhighs
	num_groups = len(images) // group_size
	groups = [i for i in range(num_groups) for _ in range(group_size)]
	gkf = GroupKFold(n_splits=n_splits, random_state=42, shuffle=True)
	all_splits = list(gkf.split(images, labels, groups=groups))
	train_idx, test_idx = all_splits[k]

	train_images= [images[i] for i in train_idx]
	train_labels = [labels[i] for i in train_idx]
	train_names = [names[i] for i in train_idx]
	test_images = [images[i] for i in test_idx]
	test_labels = [labels[i] for i in test_idx]
	test_names = [names[i] for i in test_idx]

	#Organize into datasets.  Note that these still reference the originals.
	train_dataset = CustomDataset(images=train_images, labels=train_labels, names=train_names, transform=transform)
	test_dataset = CustomDataset(images=test_images, labels=test_labels, names=test_names, transform=transform_noscale)

	# Create data loader objects for training and test sets.  Note that train shuffles, but testing does not.
	train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_threads)
	test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)

	return train_loader, test_loader

def generate_complete_dataset(dynamic_range=3, batch_size=32, n_threads=1):

	file_matcher = grmhd.FileMatcher(startingDirectory='/n/holylfs05/LABS/bhi/Lab/narayan_lab/ipole_libraries/koral_library')
	names = file_matcher.filename_arr.tolist()
	labels = [0] * len(names)
	images = []
	for file in names:
		image, _ = prepare_image(file, dynamic_range=dynamic_range)
		images.extend(image)

	complete_dataset = CustomDataset(images=images, labels=labels, names=names, transform=transform_noscale)
	complete_loader = DataLoader(dataset=complete_dataset, batch_size=batch_size, shuffle=False, num_workers=n_threads)
	return complete_loader

def generate_complete_lazy_dataset(dynamic_range=3, batch_size=32, n_threads=1):

	file_matcher = grmhd.FileMatcher(startingDirectory='/n/holylfs05/LABS/bhi/Lab/narayan_lab/ipole_libraries/koral_library')
	names = file_matcher.filename_arr.tolist()
	labels = [0] * len(names)
	complete_dataset = LazyDataset(names, labels, transform=transform_noscale, dynamic_range=dynamic_range)
	complete_loader = DataLoader(dataset=complete_dataset, batch_size=batch_size, shuffle=False, num_workers=n_threads)
	return complete_loader

def generate_hicad_lazy_dataset(startingDirectory='/n/holylfs05/LABS/bhi/Lab/narayan_lab/ipole_libraries/koral_library_2M_M87', dynamic_range=3, batch_size=32, n_threads=1):

	file_matcher = grmhd.FileMatcher(startingDirectory='/n/holylfs05/LABS/bhi/Lab/narayan_lab/ipole_libraries/koral_library_2M_M87')
	names = file_matcher.filename_arr.tolist()
	labels = [0] * len(names)
	complete_dataset = LazyDataset(names, labels, transform=transform_noscale, dynamic_range=dynamic_range)
	complete_loader = DataLoader(dataset=complete_dataset, batch_size=batch_size, shuffle=False, num_workers=n_threads)
	return complete_loader
