import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
import prepare_files as file_prep
from tqdm import tqdm
import pickle
import os
import numpy as np
import glob
import time
import temperatureScaling

def initialize_model(savefile=None, temperatureParameters=None, freeze_first_block=True):

	#Popular CNN with 18 layers and pre-trained weights.
	model = models.resnet18()

	#Modify first conv layer to accept 1 input channel
	original_conv = model.conv1
	model.conv1 = nn.Conv2d(
		in_channels=1,
		out_channels=original_conv.out_channels,
		kernel_size=original_conv.kernel_size,
		stride=original_conv.stride,
		padding=original_conv.padding,
		bias=original_conv.bias is not None
	)

	#Averaging pretrained weights because there's only one channel now.
	with torch.no_grad():
		model.conv1.weight[:] = original_conv.weight.mean(dim=1, keepdim=True)

	#Freeze first block to help overfitting?
	if freeze_first_block:
		for name, param in model.named_parameters():
			if name.startswith("conv1") or \
			   name.startswith("bn1") or \
			   name.startswith("layer1"):
				param.requires_grad = False

	model.fc = nn.Sequential(
	nn.Dropout(p=0.3),   #Regularization: dropout X% of neurons at each pass to discourage overfitting.
	nn.Linear(512, 1), #Modify so final layer is just deciding between two things.  In this case, one number between 0 and 1 is sufficient.
	)

	if savefile is not None:
		#Assuming a pth file.
		model.load_state_dict(torch.load(savefile))

	#Put in evaluation mode by default.
	model.eval()

	if temperatureParameters is None:
		return model
	else:
		calibratedModel = temperatureScaling.CalibratedModel(model)
		calibratedModel.setCalibration(temperatureParameters)
		return calibratedModel

def smooth_binary_labels(labels, smoothing=0.0):
	return labels * (1 - smoothing) + 0.5 * smoothing

def train(model, train_loader, criterion, optimizer, device, num_epochs=10, save=False, outfolder='./saved_models/', smoothing=0.0, logfile=None):
	if not save:
		model_list = []
	model.train()
	if logfile is not None:
		with open(logfile, 'w') as log:
			log.write("#epoch loss accuracy time_min\n")
	for epoch in range(num_epochs):
		t_start = time.time()
		print(f"Training model {epoch+1} of {num_epochs}.")
		running_loss = 0.0
		running_accuracy = 0.0
		for images, labels, _ in tqdm(train_loader):
			images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
			
			#Label smoothing--we're not 100% confident
			labels = smooth_binary_labels(labels, smoothing=smoothing)

			# Forward pass
			outputs = torch.sigmoid(model(images))
			loss = criterion(outputs, labels)

			# Backward pass and optimization
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			running_loss += loss.item()
			running_accuracy += (torch.round(outputs) == torch.round(labels)).sum() / len(labels)
		loss_normed = running_loss/len(train_loader)
		accuracy_normed = running_accuracy/len(train_loader)
		minutes_elapsed = (t_end-t_start)/60

		t_end = time.time()
		print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss_normed:.4f}, Accuracy: {accuracy_normed:.4f}, Time Elapsed: {minutes_elapsed:2.2f} min")
		if not save:
			model_list.append(model)
		if logfile is not None:
			with open(logfile, 'a') as log:
				log.write(f"{epoch} {loss_normed} {accuracy_normed} {minutes_elapsed}\n")
		if save:
			torch.save(model.state_dict(), os.path.join(outfolder, f"eruption_model_{epoch}.pth"))

	if not save:
		return model_list

def calcTestAccuracy(model, test_loader, device, monte_carlo=False):
	model.eval()
	correct = 0
	total = 0

	for images, labels, _ in test_loader:
		with torch.no_grad():
			if monte_carlo:
				outputs = make_prediction_montecarlo_dropout(model, test_loader, device)
				predicted = torch.round(outputs)
			else:
				images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
				outputs = torch.sigmoid(model(images))
				predicted = torch.round(outputs)  #get predicted class
		total += labels.size(0)
		correct += (predicted == labels).sum().item()

	accuracy = correct / total
	print(f"Test Accuracy: {accuracy:.4f}")
	return accuracy

def calcTestLoss(model, test_loader, device, criterion):
	model.eval()
	loss = 0.0
	with torch.no_grad():
		for images, labels, _ in test_loader:
			images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
			outputs = torch.sigmoid(model(images))
			loss += criterion(outputs, labels).item()

	loss /= len(test_loader)
	print(f"Test Loss: {loss:.4f}")
	return loss

def evaluation(model_list, test_loader, device, criterion):

	print('Beginning Evaluation.')
	# Load if given a folder name.
	if isinstance(model_list, str):
		model_files = np.array(glob.glob(os.path.join(model_list, "*pth")))
		model_order = np.argsort(np.array([name.split('/')[-1].split('.')[0].split('_')[-1] for name in model_files]).astype(int))
		model_files = model_files[model_order]
		model_list = model_files

	# Evaluation
	test_loss = []
	test_accuracy = []
	mc_test_accuracy = []
	for model in model_list:
		print(f"Evaluating epoch {len(test_loss)}...")
		if isinstance(model, str):
			model = initialize_model(model)
		model.eval()
		test_loss.append(calcTestLoss(model, test_loader, device, criterion))
		test_accuracy.append(calcTestAccuracy(model, test_loader, device))
		mc_test_accuracy.append(calcTestAccuracy(model, test_loader, device, monte_carlo=True))

	return test_loss, test_accuracy, mc_test_accuracy

def saveAnalysis(outname, training_loss, training_accuracy, test_loss, test_accuracy, mc_test_accuracy):

	# Save some numbers
	D = {}
	D['training_loss'] = training_loss
	D['training_accuracy'] = training_accuracy
	D['test_loss'] = test_loss
	D['test_accuracy'] = test_accuracy
	D['mc_test_accuracy'] = mc_test_accuracy
	with open(outname, 'wb') as f:
		pickle.dump(D, f)

def make_prediction_montecarlo_dropout(model, loader, device, n=20):

	#Put only dropout layers in training mode, turning them on.
	model.eval()
	for m in model.modules():
		if isinstance(m, torch.nn.Dropout):
			m.train()

	outputs = []
	for images, _, _ in loader:
	#for images, labels, names in loader:
		images = images.to(device)
		batch_preds = []
		#Produce n estimates
		for _ in range(n):
			#Model produces logits, which are turned into probabilities
			with torch.no_grad():
				logits = model(images)
				probs = torch.sigmoid(logits)
				batch_preds.append(probs.cpu())
		#Average the probabilities.
		batch_mean = torch.stack(batch_preds).mean(dim=0).squeeze(1)
		outputs.append(batch_mean)
	return torch.cat(outputs)

if __name__ == '__main__':
	import sys
	k = int(sys.argv[1])

	# Saved things will follow this pattern
	modelName = f'addstretch_renorm_fold{k}_012726'
	logfile = os.path.join('./logs/', modelName+'.txt')
	num_epochs = 100

	# Smoothing of labels--can cap accuracy, but my labels are genuinely ambiguous.
	smoothing = 0.00

	# Freeze the first block?  Useful for stability.
	freeze_first_block = True

	# Start with a .pth file if desired.
	savefile = None
	model = initialize_model(savefile, freeze_first_block=freeze_first_block)
	
	# Parallelization
	n_threads = 48
	torch.set_num_threads(n_threads)

	# Save models
	outfolder = './saved_models/' + modelName
	if not os.path.isdir(outfolder):
		os.system('mkdir '+outfolder)

	# Put model on GPU if available
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)

	# Create training and testing sets.
	train_loader, test_loader = file_prep.generate_dataset(n_threads=n_threads, nonfee_list='./labels/nonfee_mad_sane_weird_090925.txt', fee_list='./labels/fee_files_090925.txt', k=k)

	# Define loss function and optimizer. Adding a weight for potential class imbalance.
	imbalance = train_loader.dataset.labels.count(0) / train_loader.dataset.labels.count(1)
	criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([imbalance], device=device)) #Binary cross-entropy loss, appropriate for logit input
	optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)  #Weight decay: penalizes large weights that may not be needed to reduce overfitting.

	# Training!
	train(model, train_loader, criterion, optimizer, device, num_epochs=num_epochs, save=True, outfolder=outfolder, logfile=logfile)
	logtable = np.loadtxt(logfile)
	training_loss = logtable[:,1]
	training_accuracy = logtable[:,2]

	# Evaluate, save to a pickle
	test_loss, test_accuracy, mc_test_accuracy = evaluation(outfolder, test_loader, device, criterion)

	saveAnalysis(os.path.join("./data_products", "losses_" + modelName + '.pkl'), training_loss, training_accuracy, test_loss, test_accuracy, mc_test_accuracy)
