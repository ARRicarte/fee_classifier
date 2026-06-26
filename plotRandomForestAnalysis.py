import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import shap
import pandas as pd

bad_names = ['Fnu', 'mnet', 'vnet', 'mavg', 'vavg', 'b2_amp', 'b1_amp', 'major_FWHM', 'alpha_230-345', 'spin', 'Rh', 'mdot_scaled', 'phi']
good_names = [r'$F_\nu$', '$m_\mathrm{net}$', '$v_\mathrm{net}$', '$m_\mathrm{avg}$', '$v_\mathrm{avg}$', r'$|\beta_2|$', r'$|\beta_1|$', 'FWHM', r'$\alpha$', r'$a_\bullet$', r'$R_\mathrm{high}$', r'$\dot{M}_\bullet$', r'$\phi$']
betterNames = dict(zip(bad_names, good_names))

def shapAnalysis(shap_values_list, output=None, titles=['Thermal', 'Variable $\kappa$']):

	fig, axarr = plt.subplots(1, 2, figsize=(8,4))
	fig.subplots_adjust(right=0.85, wspace=0.03)

	order = None
	for col in range(len(axarr)):
		ax = axarr[col]
		shap_values = shap_values_list[col]
		if order is None:
			order = np.flipud(np.argsort(np.mean(np.abs(shap_values.values), axis=0)))
		shap.plots.beeswarm(shap_values, max_display=shap_values.shape[1], alpha=0.7, ax=ax, show=False, plot_size=None, color_bar=False, color=shap.plots.colors.red_blue, order=order)
		ax.set_xlabel('SHAP')
		if col != 0:
			ax.set_yticklabels([])
		ax.set_title(titles[col])

	cax = fig.add_axes([0.88, axarr[-1].get_position().y0, 0.03, axarr[-1].get_position().height])
	m = cm.ScalarMappable(cmap=shap.plots.colors.red_blue)
	m.set_array([0, 1])
	cb = fig.colorbar(m, cax=cax, ticks=[0,1])
	cb.set_ticklabels(["Lowest", "Highest"])
	cb.set_label("Feature Value (Z-normed)", size=12, labelpad=-20)
	cb.ax.tick_params(labelsize=11, length=0)

	if output is None:
		fig.show()
	else:
		fig.savefig(output, dpi=300)
		plt.close(fig)

def plotFeatureImportance(shap_values_list, accuracy_decrease_list, output=None, xlim=(0,0.6)):

	fig, ax = plt.subplots(1, 1, figsize=(5,4))

	#Doing order based on thermal SHAP performance.
	order = np.flipud(np.argsort(np.mean(np.abs(shap_values_list[0].values), axis=0)))

	for i in range(len(shap_values_list)):
		shap_values = shap_values_list[i]
		accuracy_decrease = accuracy_decrease_list[i]
		accuracy_normalized = accuracy_decrease / np.sum(accuracy_decrease)
		shap_importance = np.mean(np.abs(shap_values.values), axis=0)
		shap_normalized = shap_importance / np.sum(shap_importance)
		y_values = np.flipud(np.arange(shap_values.shape[1]))

		if i == 0:
			ax.scatter(accuracy_normalized[order], y_values+0.15, marker='o', color='b', s=50, label='Permutation')
			ax.scatter(shap_normalized[order], y_values+0.15, marker='s', color='r', s=50, label='SHAP')
		else:
			ax.scatter(accuracy_normalized[order], y_values-0.15, marker='o', color='b', s=50, facecolor='white')
			ax.scatter(shap_normalized[order], y_values-0.15, marker='s', color='r', s=50, facecolor='white')
		for y in y_values:
			ax.plot(xlim, [y]*2, color='gray', ls=':', lw=1, alpha=0.3)

	ax.set_xlim(xlim)
	ax.set_xlabel("Normalized Feature Importance", fontsize=10)
	ax.set_yticks(y_values)
	ax.set_yticklabels([shap_values.feature_names[i] for i in order])

	#Extra legend items
	ax.scatter([], [], marker='s', color='k', s=50, label='Thermal')
	ax.scatter([], [], marker='s', color='k', s=50, label='Variable $\kappa$', facecolor='white')
	ax.legend(loc='lower right', frameon=True, ncols=2, edgecolor='none', framealpha=1)

	fig.tight_layout()
	if output is None:
		fig.show()
	else:
		fig.savefig(output)
		plt.close(fig)

if __name__ == '__main__':
	pairs = []
	pairs.append(['./data_products/shap_thermal_everything.pkl', './data_products/shap_variablekappa_everything.pkl'])
	#pairs.append(['./data_products/shap_thermal_everything_observable.pkl', './data_products/shap_variablekappa_everything_observable.pkl'])
	#pairs.append(['./data_products/shap_thermal_everything_observable_unresolved.pkl', './data_products/shap_variablekappa_everything_observable_unresolved.pkl'])
	#pairs.append(['./data_products/shap_thermal_compact.pkl', './data_products/shap_variablekappa_compact.pkl'])
	#pairs.append(['./data_products/shap_thermal_compact_noretrogrades.pkl', './data_products/shap_variablekappa_compact_noretrogrades.pkl'])

	for pair in pairs:
		shap_values_list = []
		accuracy_decrease_list = []
		for infile in pair:
			with open(infile, 'rb') as f:
				D = pickle.load(f)
			D['shap_values'].feature_names = [betterNames[name] for name in D['shap_values'].feature_names]

			shap_values_list.append(D['shap_values'])
			accuracy_decrease_list.append(D['accuracy_decrease'])
		name = pair[0].split('thermal_')[-1].split('.pkl')[0]
		shapAnalysis(shap_values_list, output=f'./figures/shap_summary_{name}.pdf')
		plotFeatureImportance(shap_values_list, accuracy_decrease_list, output=f'./figures/feature_importance_{name}.pdf')
		'''
		#Just an extended x-axis.
		plotFeatureImportance(shap_values_list, accuracy_decrease_list, xlim=(0,0.8), output=f'./figures/feature_importance_{name}.pdf')
		'''
