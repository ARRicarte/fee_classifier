import numpy as np
import pickle

def boxcar_irregular_time_weighted(t, y, width=2000.0):
	t = np.asarray(t)
	y = np.asarray(y)

	order = np.argsort(t)
	t = t[order]
	y = y[order]

	dt = np.empty_like(t, dtype=float)
	dt[1:-1] = 0.5 * (t[2:] - t[:-2])
	dt[0] = t[1] - t[0]
	dt[-1] = t[-1] - t[-2]

	half = width / 2
	wsum = np.r_[0.0, np.cumsum(y * dt)]
	dsum = np.r_[0.0, np.cumsum(dt)]

	left = np.searchsorted(t, t - half, side="left")
	right = np.searchsorted(t, t + half, side="right")

	y_smooth = (wsum[right] - wsum[left]) / (dsum[right] - dsum[left])

	out = np.empty_like(y_smooth)
	out[order] = y_smooth
	return out

files = ['../../data_products/koral_fluxes.pkl', '../../data_products/koral_fluxes_2M.pkl']
newfiles = ['../../data_products/koral_fluxes_fixphi.pkl', '../../data_products/koral_fluxes_fixphi_2M.pkl']

for i in range(len(files)):
	infile = files[i]
	outfile = newfiles[i]
	with open(infile, 'rb') as f:
		D = pickle.load(f)
	for key in D.keys():
		sub_D = D[key]
		sub_D['phi_BH_prefix'] = sub_D['phi_BH']

		#Arbitrary, but using a rolling window of 2000 M
		sub_D['phi_BH'] = sub_D['phi_BH_prefix'] * np.sqrt(sub_D['mdot']) / boxcar_irregular_time_weighted(sub_D['t'], sub_D['mdot'], width=2000)

	with open(outfile, 'wb') as f:
		pickle.dump(D, f)
