import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import shap
import pandas as pd
import copy

seed_value = 42
rng = np.random.default_rng(seed=seed_value)

def z_normalize(array):

	return (array - np.mean(array)) / np.std(array)

def computeRandomForest(dictionary, keys, mask, columnNames=None, groups=None, trainingFraction=0.8, doFeatureImportance=True):

	if columnNames is None:
		columnNames = keys

	keyToName = dict(zip(keys,columnNames))

	#To tune the tree, splitting into training and validation
	training = np.zeros(np.sum(mask), dtype=bool)
	training[rng.choice([True,False], size=len(training), p=[trainingFraction,1-trainingFraction])] = True
	validation = ~training
	X_training = pd.DataFrame(np.array([dictionary[key][mask][training] for key in keys]).transpose(1,0), columns=columnNames)
	y_training = np.all(dictionary['isFEE_model'][mask][training] > 0.5, axis=1).astype(int)
	true_fee_fraction = np.mean(y_training)
	sample_weight = np.ones(len(y_training), dtype=float)
	sample_weight[y_training == 1] = 1-true_fee_fraction
	sample_weight[y_training == 0] = true_fee_fraction

	#Build the tree
	rf = RandomForestClassifier(max_depth=5, n_estimators=100, random_state=seed_value)
	rf.fit(X_training, y_training, sample_weight=sample_weight)
	accuracy = rf.score(X_training, y_training, sample_weight=sample_weight)
	accuracy_unweighted = rf.score(X_training, y_training)
	predictions = rf.predict(X_training)
	print(f"Real FEE fraction: {np.mean(y_training):1.3f} | Model FEE fraction: {np.mean(predictions):1.3f}")
	print(f"Weighted Accuracy: {accuracy:1.3f} | Unweighted Accuracy: {accuracy_unweighted:1.3f}.")

	#Leave out a validation set and see how accurate the predictions are.
	if trainingFraction < 1.0:
		X_validation = pd.DataFrame(np.array([dictionary[key][mask][validation] for key in keys]).transpose(1,0), columns=columnNames)
		y_validation = np.round(np.mean(dictionary['isFEE_model'][mask][validation], axis=1))
		validation_fee_fraction = np.mean(y_validation)
		sample_weight_validation = np.ones(len(y_validation), dtype=float)
		sample_weight_validation[y_validation == 1] = 1-validation_fee_fraction
		sample_weight_validation[y_validation == 0] = true_fee_fraction
		print(f"Validation Accuracy, Weighted: {rf.score(X_validation, y_validation, sample_weight=sample_weight_validation):1.3f}.")
		print(f"Validation Accuracy, Unweighted: {rf.score(X_validation, y_validation):1.3f}.")

	if doFeatureImportance:
		if groups is None:
			explainer = shap.Explainer(rf, X_training)
			shap_values = explainer(X_training)
		else:
			feature_names = list(X.columns)

			group_names = []
			group_indices = []

			for group, features in groups.items():
				group_names.append(group)
				group_indices.append([feature_names.index(keyToName[f]) for f in features])
			explainer = shap.Explainer(rf, X, feature_names=feature_names)
			shap_values = explainer(X, groups=group_indices)

		#Now, estimate permutation importance.
		accuracy_decrease = []
		for key in columnNames:
			X_shuffle = X_training.copy()
			X_shuffle[key] = X_training.sample(frac=1)[key].values
			accuracy_decrease.append(accuracy-rf.score(X_shuffle, y_training, sample_weight=sample_weight))

		return rf, shap_values[:,:,1], accuracy_decrease
	else:
		return rf

def shapAnalysis(shap_values):

	shap.plots.bar(shap_values, max_display=shap_values.shape[1])
	shap.plots.beeswarm(shap_values, max_display=shap_values.shape[1])

def plotFeatureImportance(shap_values, accuracy_decrease, output=None):

	fig, ax = plt.subplots(1, 1, figsize=(4,4))

	order = np.flipud(np.argsort(accuracy_decrease))
	
	accuracy_normalized = accuracy_decrease / np.sum(accuracy_decrease)
	shap_importance = np.mean(np.abs(shap_values.values), axis=0)
	shap_normalized = shap_importance / np.sum(shap_importance)
	y_values = np.flipud(np.arange(shap_values.shape[1]))
	ax.scatter(accuracy_normalized[order], y_values, marker='o', color='b', s=50, label='Permutation')
	ax.scatter(shap_normalized[order], y_values, marker='s', color='r', s=50, label='SHAP')

	ax.set_xlim(0,0.5)
	ax.set_xlabel("Normalized Feature Importance", fontsize=10)
	ax.set_yticks(y_values)
	ax.set_yticklabels([shap_values.feature_names[i] for i in order])

	ax.legend(loc='lower right', frameon=False)
	fig.tight_layout()
	if output is None:
		fig.show()
	else:
		fig.savefig(output)
		plt.close(fig)

if __name__ == '__main__':
	#Thermal
	'''
	inputFile = './data_products/combinedFEEDictionary_061226.pkl'
	outputFile = './data_products/shap_thermal_everything.pkl'
	'''

	#Nonthermal
	inputFile = './data_products/combinedFEEDictionary_variablekappa_061226.pkl'
	outputFile = './data_products/shap_variablekappa_everything.pkl'
	with open(inputFile, 'rb') as f:
		D = pickle.load(f)

	#Turn this on if we're calibrating metaparameters.
	calibration = False

	#All likely useful things
	keys = np.array(['Fnu', 'mnet', 'vnet', 'mavg', 'vavg', 'b2_amp', 'b1_amp', 'major_FWHM', 'alpha_230-345', 'spin', 'Rh', 'mdot_scaled', 'phi'])
	#nicerNames = np.array([r'$F_\nu$', '$m_\mathrm{net}$', '$v_\mathrm{net}$', '$m_\mathrm{avg}$', '$v_\mathrm{avg}$', r'$|\beta_2|$', r'$|\beta_1|$', 'FWHM', r'$\alpha$', r'$a_\bullet$', r'$R_\mathrm{high}$', r'$\dot{M}_\bullet$', r'$\phi$'])
	nicerNames = keys

	#Everything
	indices = np.arange(len(keys))

	#Exclude GRMHD unobservables
	#indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

	#Exclude spatial resolution
	#indices = np.array([0, 1, 2, 8, 9, 10])

	#Compact
	#indices = np.array([0, 1, 2, 7, 8, 9, 10])

	#Testing
	#indices = np.array([0, 9, 10])

	mask = (D['Bstate'] == 'MAD') & (D['freq'] == 228)

	#z-score normalization for better interpretability
	Dz = copy.deepcopy(D)
	for key in keys:
		if key in ['spin', 'Rh']:
			continue
		for freq in np.unique(D['freq']):
			for Bstate in np.unique(D['Bstate']):
				for Rhigh in np.sort(np.unique(D['Rh'])):
					for spin in np.sort(np.unique(D['spin'])):
						match = (D['Bstate'] == Bstate) & (D['freq'] == freq) & (D['Rh'] == Rhigh) & (D['spin'] == spin)
						Dz[key][match] = z_normalize(D[key][match])

	#Did this to determine a reasonable max_depth.  Weighted validation accuracy plateaued at 5 when using the entire dataset.
	if calibration:
		rf = computeRandomForest(Dz, keys[indices], mask, columnNames=nicerNames[indices], trainingFraction=0.8, doFeatureImportance=False)
	else:
		#Just for reporting accuracies
		#rf = computeRandomForest(Dz, keys[indices], mask, columnNames=nicerNames[indices], trainingFraction=1.0, doFeatureImportance=False)
		rf, shap_values, accuracy_decrease = computeRandomForest(Dz, keys[indices], mask, columnNames=nicerNames[indices], trainingFraction=1.0)

		D_out = {"shap_values": shap_values, "accuracy_decrease": accuracy_decrease}
		with open(outputFile, 'wb') as f:
			pickle.dump(D_out, f)
		#shapAnalysis(shap_values)
		#plotFeatureImportance(shap_values, accuracy_decrease)
