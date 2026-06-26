import pickle
import numpy as np

def firstDerivative(y, x):
	'''
	I want to be using three points consistently.

	For the first derivative, np.gradient already does this.
	'''
	return np.gradient(y, x)

def secondDerivative(y, x):
	'''
	If you naively apply np.gradient twice, you are estimating the second derivative from 5 points.
	This isn't inherently bad, but can lead to problems, especially if the curve is poorly-sampled.
	'''

	assert len(y) == len(x)
	output = np.zeros_like(y)
	dx = np.diff(x)

	#First derivative from 2 points.
	dy_dx = np.diff(y) / dx

	#Second derivative from 3 points.
	output[1:-1] = np.diff(dy_dx) / (0.5*(dx[1:] + dx[:-1]))

	#Copying the estimates to the edges.
	output[0] = output[1]
	output[-1] = output[-2]
	return output

def localCurvature(x, y, s):
	dx_ds = firstDerivative(x, s)
	dy_ds = firstDerivative(y, s)
	d2x_ds2 = secondDerivative(x, s)
	d2y_ds2 = secondDerivative(y, s)

	return (dx_ds * d2y_ds2 - dy_ds * d2x_ds2) / np.sqrt(dx_ds**2 + dy_ds**2)**3

def computeCurvatures(infile, Bstate, spin, Rh, inc, freq, cleanup=True, minCurvatureRadius=0.00, maxRotationSpeed=2*np.pi/15, convolve=False, convolveOver=4):

	#Open file, get the info we want to visualize.
	with open(infile, 'rb') as openFile:
		D = pickle.load(openFile)
	model = (D['Bstate'] == Bstate) & (D['spin'] == spin) & (D['Rh'] == Rh) & (D['inc'] == inc) & (D['freq'] == freq)
	if np.sum(model) == 0:
		return None, None, None, None, None, None

	I = D['I'][model]
	Q = D['Q'][model]
	U = D['U'][model]
	t = D['timeInM'][model]

	if convolve:
		convolutionKernel = np.full(convolveOver,1/convolveOver)
		I = np.convolve(I, convolutionKernel, mode='same')
		Q = np.convolve(Q, convolutionKernel, mode='same')
		U = np.convolve(U, convolutionKernel, mode='same')

	ds_dt = np.sqrt(firstDerivative(Q,t)**2 + firstDerivative(U,t)**2) #Curve speed, Jy per M.
	dt = np.gradient(t)
	s = np.cumsum(ds_dt*dt)
	local_curvature = localCurvature(Q, U, s)  #Radians per arclength, which is in Jy

	#Attempts to address numerical issues.
	if cleanup:
		#In QU space, enforce a minimum radius of curvature in Jy.  This is equivalent to a maximum curvature that we believe.
		too_small = 1.0/np.abs(local_curvature) < minCurvatureRadius
		local_curvature[too_small] = 0

		#Also, enforce a maximum loop speed.  We do not believe we resolve any loops faster than 2 pi / 15 M by default.
		loop_speed = local_curvature * ds_dt
		too_fast = np.abs(loop_speed) > maxRotationSpeed
		local_curvature[too_fast] = 0
		print(f"  {np.mean(too_small):1.3f} of the loops were considered too small, and {np.mean(too_fast):1.3f} of the loops were considered too fast.")

	integrated_curvature = np.cumsum(local_curvature*ds_dt*dt)

	#May be useful for masking later
	P = np.sqrt(Q**2 + U**2)
	m_net = P / I

	return t, local_curvature, s, integrated_curvature, m_net, P

def computeAllCurvatures(infile, outfile, minCurvatureRadius=0.00, maxRotationSpeed=2*np.pi/15, convolve=False, convolveOver=4):

	with open(infile, 'rb') as f:
		inDict = pickle.load(f)

	outDict = {}
	outDict['Bstate'] = []
	outDict['spin'] = []
	outDict['Rh'] = []
	outDict['inc'] = []
	outDict['t'] = []
	outDict['local_curvature'] = []
	outDict['arclength'] = []
	outDict['integrated_curvature'] = []
	outDict['mean_rotation'] = []
	outDict['m_net'] = []
	outDict['P'] = []
	outDict['filename'] = []
	outDict['freq'] = []
	for Bstate in np.sort(np.unique(inDict['Bstate'])):
		for spin in np.sort(np.unique(inDict['spin'])):
			for Rh in np.sort(np.unique(inDict['Rh'])):
				for inc in np.sort(np.unique(inDict['inc'])):
					for freq in np.sort(np.unique(inDict['freq'])):
						t, local_curvature, arclength, integrated_curvature, m_net, P = computeCurvatures(infile, Bstate, spin, Rh, inc, freq, minCurvatureRadius=minCurvatureRadius, \
						maxRotationSpeed=maxRotationSpeed, convolve=convolve, convolveOver=convolveOver)
						if t is not None:
							print(f"{Bstate} {spin} {Rh} {inc}: mean rotation rate = {(integrated_curvature[-1] - integrated_curvature[0]) / (t[-1]-t[0]):1.3e} rad per M")
							outDict['Bstate'].append(Bstate)
							outDict['spin'].append(spin)
							outDict['Rh'].append(Rh)
							outDict['inc'].append(inc)
							outDict['t'].append(t)
							outDict['local_curvature'].append(local_curvature)
							outDict['arclength'].append(arclength)
							outDict['integrated_curvature'].append(integrated_curvature)
							outDict['mean_rotation'].append((integrated_curvature[-1]-integrated_curvature[0]) / (t[-1]-t[0]))
							outDict['m_net'].append(m_net)
							outDict['P'].append(P)
							outDict['freq'].append(freq)

	for key in outDict.keys():
		outDict[key] = np.array(outDict[key])
	outDict['minCurvatureRadius'] = minCurvatureRadius
	outDict['maxRotationSpeed'] = maxRotationSpeed
	outDict['filename'] = infile
	
	with open(outfile, 'wb') as openFile:
		pickle.dump(outDict, openFile, protocol=2)

if __name__ == '__main__':
	infile = './data_products/unnresolved_pol_light_curves_2M_thermal.pkl'
	outfile = './data_products/curvatures_2M_thermal_maxspeed2piover5.pkl'
	computeAllCurvatures(infile, outfile, maxRotationSpeed=2*np.pi/5, minCurvatureRadius=0.00, convolve=False, convolveOver=0)

	infile = './data_products/unnresolved_pol_light_curves_2M_nonthermal.pkl'
	outfile = './data_products/curvatures_2M_nonthermal_maxspeed2piover5.pkl'
	computeAllCurvatures(infile, outfile, maxRotationSpeed=2*np.pi/5, minCurvatureRadius=0.00, convolve=False, convolveOver=0)
