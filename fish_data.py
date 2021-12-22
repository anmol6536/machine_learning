import os
import pandas as pd
import matplotlib.pyplot as plt


dataset_folder = '/Users/anmol_gorakshakar/python/machine_learning/fish_dataset/archive/Fish_Dataset/Fish_Dataset'
if os.path.exists(dataset_folder):
    fish_folders = [f"{dataset_folder}/{i}/{i}" for i in os.listdir(dataset_folder) if '.' not in i]
    holder = []
    for folder in fish_folders:
        label = folder.split('/')[-1].lower().replace('-', '_').replace(' ', '_')
        photos = pd.DataFrame([f"{folder}/{i}" for i in os.listdir(folder) if '.png' in i], columns=['path'])
        photos['label'] = label
        holder.append(photos)
    photos = pd.concat(holder).reset_index(drop=True)

print(photos['path'][1])
picture = plt.imread(photos['path'][1])
plt.imshow(picture)
plt.show()