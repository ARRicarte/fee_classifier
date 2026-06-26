import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
import copy

def z_normalize(array):

	return (array - np.mean(array)) / np.std(array)

def fitLogistic(dictionary, keys, mask, bootstrap=False):

	X = np.atleast_2d([dictionary[key][mask] for key in keys]).transpose(1,0)

	if 'isFEE' in dictionary.keys():
		#This is just one model.
		y = np.round(dictionary['isFEE'][mask])
	elif 'isFEE_model' in dictionary.keys():
		#Assuming multi-fold model.
		y = np.all(dictionary['isFEE_model'][mask] > 0.5, axis=1).astype(int)

	if bootstrap:
		#Sample with replacement.
		randomIndices = np.random.choice(np.arange(len(y)), size=len(y), replace=True)
		X = X[randomIndices,:]
		y = y[randomIndices]

	fee_rate = np.mean(y)
	clf = LogisticRegression(penalty='l2', class_weight='balanced')
	clf.fit(X,y)
	sample_weight = np.full(len(y), fee_rate)
	sample_weight[y==1] = 1-fee_rate
	score = clf.score(X, y, sample_weight=sample_weight)

	print(f"Obtained an accuracy of {score}.")
	print(f"The FEE rate is {np.mean(y):1.2f}.")

	return clf, score

def fitLogisticBootstrap(dictionary, keys, mask, n_boot=100):

	k_list = []
	x_list = []
	acc_list = []
	for boot in range(n_boot):
		clf, score = fitLogistic(dictionary, keys, mask, bootstrap=True)
		k_list.append(clf.coef_[0][0])
		x_list.append(clf.intercept_[0])
		acc_list.append(score)
	
	k = np.mean(k_list)
	k_err = np.std(k_list)
	x = np.mean(x_list)
	x_err = np.std(x_list)
	acc = np.mean(acc_list)
	acc_err = np.std(acc_list)
	
	print(f"k={k:1.2f}+/-{k_err:1.2f}, x={x:1.2f}+/-{x_err:1.2f}, acc={acc:1.2f}+/-{acc_err:1.2f}")
	return k, k_err, x, x_err, acc, acc_err

if __name__ == '__main__':
	#inputFile = './data_products/combinedFEEDictionary_120125.pkl'
	inputFile = './data_products/combinedFEEDictionary_variablekappa_012726.pkl'
	with open(inputFile, 'rb') as f:
		D = pickle.load(f)

	#Results depend on which parameters are included, as well as the penalty used.  Due to colinearities, be careful about interpretation.

	#Most complete
	#keys = np.array(['Fnu', 'mnet', 'vnet', 'mavg', 'vavg', 'b2_amp', 'b1_amp', 'major_FWHM', 'alpha_230-345'])

	#Only spatially unresolved
	#keys = np.array(['Fnu', 'mnet', 'vnet', 'alpha_230-345'])

	#Key pieces
	keys = np.array(['Fnu', 'mnet', 'major_FWHM', 'mavg'])

	#Stokes I
	#keys = np.array(['Fnu', 'major_FWHM', 'alpha_230-345'])

	#GRMHD
	#keys = np.array(['phi', 'mdot_scaled'])

	#One thing.
	#keys = np.array(['Fnu'])

	#Common themes during a FEE...
	# - Higher spectral index
	# - Lower flux density
	# - Higher linear polarization
	# - Larger image overall

	#Make a dictionary of z_score normalized versions.
	Dz = copy.deepcopy(D)
	for key in keys:
		for freq in np.unique(D['freq']):
			for Bstate in np.unique(D['Bstate']):
				for Rhigh in np.sort(np.unique(D['Rh'])):
					for spin in np.sort(np.unique(D['spin'])):
						mask = (D['Bstate'] == Bstate) & (D['freq'] == freq) & (D['Rh'] == Rhigh) & (D['spin'] == spin)
						Dz[key][mask] = z_normalize(D[key][mask])

	#Only use MAD and 228 GHz.
	combinedMask = (Dz['Bstate'] == 'MAD') & (Dz['freq'] == 228) & (Dz['spin'] == 0.9) & (Dz['Rh'] == 160)

	logistic, _ = fitLogistic(Dz, keys, combinedMask)
	fit = np.atleast_1d(np.squeeze(logistic.coef_))

	importance = np.flipud(np.argsort(np.abs(fit)))
	#print(f"Top Predictors: {keys[importance[0]]}, {fit[importance[0]]:1.2e}; {keys[importance[1]]}, {fit[importance[1]]:1.2e}; {keys[importance[2]]}, {fit[importance[2]]:1.2e}")
	for index in importance:
		print(f"{keys[index]}: {fit[index]:1.2e}")
	print(f"Intercept: {logistic.intercept_[0]:1.2e}")

	'''
	for Rhigh in np.sort(np.unique(D['Rh'])):
		for spin in np.sort(np.unique(D['spin'])):
			print(f"Rhigh={Rhigh}, a={spin}")
			mask = (D['Bstate'] == 'MAD') & (D['freq'] == 228) & (D['Rh'] == Rhigh) & (D['spin'] == spin)
			fit = fitLogistic(D, keys, mask).squeeze()
			importance = np.flipud(np.argsort(np.abs(fit)))
			print(f"Top Predictors: {keys[importance[0]]}, {fit[importance[0]]:1.2e}; {keys[importance[1]]}, {fit[importance[1]]:1.2e}; {keys[importance[2]]}, {fit[importance[2]]:1.2e}")
	'''
