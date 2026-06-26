import os
import pickle
import numpy as np

def selectUsedModels(file_list, target, criterion='accuracy'):

	for file in file_list:
		with open(file, 'rb') as f:
			D = pickle.load(f)
		accuracy = D['test_accuracy']
		loss = D['test_loss']
		if criterion == 'accuracy':
			best_index = np.argmax(accuracy)
		elif criterion == 'loss':
			best_index = np.argmin(loss)
		print(f"For {file}, the best index is {best_index}, where the accuracy was {accuracy[best_index]:1.4f} and the loss was {loss[best_index]:1.4f}.")

		old_name = './saved_models/' + file.split('.pkl')[0].split('losses_')[-1] + f'/eruption_model_{best_index}.pth'
		new_name = os.path.join(target, file.split('.pkl')[0].split('losses_')[-1]  + f'_{best_index}.pth')
		os.system(f"cp {old_name} {new_name}")

if __name__ == '__main__':
	file_list = []
	file_list.append('./data_products/losses_addstretch_renorm_fold0_012726.pkl')
	file_list.append('./data_products/losses_addstretch_renorm_fold1_012726.pkl')
	file_list.append('./data_products/losses_addstretch_renorm_fold2_012726.pkl')
	file_list.append('./data_products/losses_addstretch_renorm_fold3_012726.pkl')
	file_list.append('./data_products/losses_addstretch_renorm_fold4_012726.pkl')

	target = './used_models'
	selectUsedModels(file_list, target, criterion='loss')
