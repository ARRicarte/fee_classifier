"""
Erandi did a blind labeling of a random selection of hicad snapshots.
Here, we compare against the performance of the CNNs.
"""

import numpy as np
import pickle
import glob
import matplotlib.pyplot as plt

erandi_file = "./erandi_labels/hicad_labels.txt"
hicad_labels = "./data_products/combinedFEEDictionary_2M_061226.pkl"

#Need to map Erandi's labels to snapshots
filenames = glob.glob("./image_pngs/hicad/*png")
images = np.array([name.split('/')[-1] for name in filenames])
indices = np.array([int(name.split('_')[0]) for name in images])
order = np.argsort(indices)
images = images[order]
indices = indices[order]
names = []
for ind in range(len(filenames)):
	pieces = images[ind].split('_')
	thing = "_".join([pieces[i] for i in range(1,len(pieces))]).replace('.png', '.h5')
	names.append(thing)
names = np.array(names)
indexToName = dict(zip(indices, names))

#Load their table
blind_table = np.loadtxt(erandi_file, dtype=str)
blind_indices = blind_table[:,0].astype(int)
blind_labels = np.zeros(blind_table.shape[0], dtype=int)
blind_labels[blind_table[:,1] == 'yes'] = 1
blind_names = np.array([indexToName[ind] for ind in blind_indices])

#Load CNN labels
with open(hicad_labels, 'rb') as f:
	D = pickle.load(f)
CNN_names = np.array([name.split('/')[-1] for name in D['filenames']])
CNN_labels = D['isFEE_model']
CNN_mean_probs = np.mean(D['isFEE_model'], axis=1)
CNN_consensus = np.all(CNN_labels>0.5, axis=1).astype(int)
CNN_majority_vote = np.mean((CNN_labels>0.5).astype(float)>0.5, axis=1).astype(int)

#Match Erandi to CNN.
erandi_to_cnn = np.array([np.where(CNN_names == name)[0][0] for name in blind_names])
CNN_labels_used = CNN_labels[erandi_to_cnn]
CNN_majority_vote_used = CNN_majority_vote[erandi_to_cnn]
CNN_consensus_used = CNN_consensus[erandi_to_cnn]
print(f"Consensus and majority vote agree {np.mean(CNN_majority_vote_used==CNN_consensus_used)*100:2.1f}% of the time.")
print(f"Erandi and consensus agree {np.mean(CNN_consensus_used==blind_labels)*100:2.1f}% of the time.")
print("Mismatches:")
print(blind_indices[CNN_consensus_used!=blind_labels])
print("Erandi vote:")
print(blind_labels[CNN_consensus_used!=blind_labels])

true_positives = (CNN_consensus_used==1) & (blind_labels==1)
true_negatives = (CNN_consensus_used==0) & (blind_labels==0)
false_positives = (CNN_consensus_used==1) & (blind_labels==0)
false_negatives = (CNN_consensus_used==0) & (blind_labels==1)
precision = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_positives))
recall = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_negatives))
mean_confidence_disagreement = np.mean(np.abs(CNN_mean_probs[erandi_to_cnn][CNN_consensus_used!=blind_labels]-0.5))
mean_confidence = np.mean(np.abs(CNN_mean_probs-0.5))
print(f"Precision: {precision:1.2f}")
print(f"Recall: {recall:1.2f}")
print(f"Mean Confidence on Disagreements: {mean_confidence_disagreement:1.2f}")
print(f"Mean Confidence Overall: {mean_confidence:1.2f}")
print(f"Fraction between 0.1 and 0.9 on disagreements:  {np.mean(np.abs(CNN_mean_probs[erandi_to_cnn][CNN_consensus_used!=blind_labels]-0.5)<0.4):1.2f}")
print(f"Fraction between 0.1 and 0.9 overall:  {np.mean(np.abs(CNN_mean_probs-0.5)<0.4):1.2f}")

#Histogram
bins = np.linspace(0,1,11)
plt.hist(CNN_mean_probs, density=True, bins=bins, alpha=0.6)
plt.hist(CNN_mean_probs[erandi_to_cnn][CNN_consensus_used!=blind_labels], density=True, bins=bins, alpha=0.6)
plt.show()
