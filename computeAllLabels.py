import prepare_files as file_prep
import grmhd_library as grmhd
import fee_classifier as fee
import torch
import pickle
import os
import numpy as np
import glob
from tqdm import tqdm
import temperatureScaling

def computeAllLabels(modelName, outputName, n_threads=48, mc_dropout=False, calibratedTemperatureParameters=None, dataset='complete'):

	model = fee.initialize_model(savefile=modelName, temperatureParameters=calibratedTemperatureParameters)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	if dataset == 'complete':
		loader = file_prep.generate_complete_lazy_dataset()
	elif dataset == 'hicad':
		loader = file_prep.generate_hicad_lazy_dataset()

	#Run the model on all of the images and get the predictions
	names = []
	labels = []
	if mc_dropout: 
		labels = np.squeeze(fee.make_prediction_montecarlo_dropout(model, loader, device, n=10).detach().numpy())
		names = loader.dataset.paths
	else:
		for images_batch, _, names_batch in tqdm(loader):
			images_batch = images_batch.to(device)
			if calibratedTemperatureParameters is not None:
				params_raw = [grmhd.read_parameters_from_name(name) for name in names_batch]
				params = {}
				params['Bstate'] = np.array([D['Bstate'] for D in params_raw])
				params['spin'] = np.array([D['spin'] for D in params_raw])
				params['Rh'] = np.array([D['Rh'] for D in params_raw])
				labels_batch = torch.sigmoid(model.forward(images_batch, params=params))
			else:
				labels_batch = torch.sigmoid(model(images_batch))
			names.extend(names_batch)
			labels.extend(torch.squeeze(labels_batch.detach()).tolist())
	#Note that these labels are floats between 0 and 1, not integers.

	D = {}
	D['names'] = names
	D['labels'] = labels
	D['calibratedTemperatureParameters'] = calibratedTemperatureParameters

	with open(outputName, "wb") as openFile:
		pickle.dump(D, openFile)

if __name__ == '__main__':
	modelName = './saved_models/sanestoo_080125/eruption_model_19.pth'
	outputName = "./data_products/calibrated_080125_model_19.pkl"
	computeAllLabels(modelName, outputName)
