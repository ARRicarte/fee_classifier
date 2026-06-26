import os
import glob
import numpy as np

images = np.array([name.split('/')[-1] for name in glob.glob("image_pngs/labeled/*")])
indices = np.array([int(name.split('_')[0]) for name in images])
order = np.argsort(indices)
images = images[order]
indices = indices[order]
labels = np.zeros(len(indices), dtype=int)
table_fee = np.loadtxt('./labels/fee_files_090925.txt', dtype=str).tolist()

for ind in indices:
	pieces = images[ind].split('_')
	thing = "_".join([pieces[i] for i in range(1,len(pieces))]).replace('.png', '.h5')
	labels[ind] = int(thing in table_fee)

f = open("verification_labels.txt", "w")
for i in range(len(labels)):
	f.write(f"{i} {labels[i]}\n")

f.close()
	
