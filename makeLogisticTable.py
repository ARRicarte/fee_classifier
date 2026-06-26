from fitLogistic import *
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
import copy

def makeLogisticTable(dictionary, keys, mask, outfile):

	with open(outfile, 'w') as table:
		table.write("Quantity, k, x0, Accuracy")
		for key in keys:
			k, kerr, x0, x0err, Accuracy, Accuracyerr = fitLogisticBootstrap(Dz, [key], mask)
			table.write(f"\n{key}, ${k:1.3f} \pm {kerr:1.3f}$, ${x0:1.3f} \pm {x0err:1.3f}$, ${Accuracy:1.3f} \pm {Accuracyerr:1.3f}$")

if __name__ == '__main__':
	inputs = ['./data_products/combinedFEEDictionary_061226.pkl', './data_products/combinedFEEDictionary_variablekappa_061226.pkl']
	outputs = ['./data_products/logistics_thermal.txt', './data_products/logistics_variablekappa.txt']
	for input, output in zip(inputs, outputs):
		with open(input, 'rb') as f:
			D = pickle.load(f)

		keys = np.array(['Fnu', 'mnet', 'vnet', 'mavg', 'vavg', 'b2_amp', 'b1_amp', 'major_FWHM', 'alpha_230-345', 'mdot_scaled', 'phi', 'tauI', 'tauF'])
		Dz = copy.deepcopy(D)
		for key in keys:
			for freq in np.unique(D['freq']):
				for Bstate in np.unique(D['Bstate']):
					for Rhigh in np.sort(np.unique(D['Rh'])):
						for spin in np.sort(np.unique(D['spin'])):
							mask = (D['Bstate'] == Bstate) & (D['freq'] == freq) & (D['Rh'] == Rhigh) & (D['spin'] == spin)
							Dz[key][mask] = z_normalize(D[key][mask])
		mask = (Dz['Bstate'] == 'MAD') & (Dz['freq'] == 228)
		makeLogisticTable(Dz, keys, mask, output)
