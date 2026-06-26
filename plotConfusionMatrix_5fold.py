import torch
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import numpy as np
import fee_classifier as fee
import prepare_files as file_prep
import os
import glob
import grmhd_library as grmhd
from plotConfusionMatrix import plotConfusionMatrix

folder = './used_models/'
output = './figures/confusion_matrices_5fold/'
n_threads = 24
models = glob.glob(os.path.join(folder, '*pth'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for model_name in models:
	model = fee.initialize_model(model_name)
	print(model_name)
	segments = model_name.split('_')
	k = int(segments[np.where(['fold' in thing for thing in segments])[0][0]][-1])

	train_loader, test_loader = file_prep.generate_dataset(n_threads=n_threads, nonfee_list='./labels/nonfee_mad_sane_weird_090925.txt', fee_list='./labels/fee_files_090925.txt', k=k)
	for images, labels, names in test_loader:
		images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
		params_raw = [grmhd.read_parameters_from_name(name) for name in names]
		params = {}
		params['Rh'] = np.array([D['Rh'] for D in params_raw])
		params['Bstate'] = np.array([D['Bstate'] for D in params_raw])
		params['spin'] = np.array([D['spin'] for D in params_raw])
		logits = model(images)

	guesses = torch.round(torch.sigmoid(logits))
	logits = logits.detach().numpy()
	guesses = guesses.detach().numpy()
	labels = labels.numpy()
	names = np.array(names)

	probabilities_unscaled = torch.sigmoid(torch.from_numpy(logits))
	for Bstate in np.unique(params['Bstate']):
		outname = os.path.join(output, f'confusion_{Bstate}_fold{k}.pdf')
		if not os.path.isfile(outname):
			mask = (params['Bstate'] == Bstate)
			plotConfusionMatrix(labels[mask], guesses[mask], output=outname)
