import json
from collections import defaultdict
from global_land_mask import globe
import numpy as np
from config import clustering

with open(clustering['kmeans_path']+'/kmeans_labels_95.json', 'r') as file:
    labels_data = json.load(file)
with open('/storage/climate-memmap/train_coordinates_data/coordinates_data_95.json', 'r') as file:
    coordinates = json.load(file)
land_dict = {}
non_land_dict = {}
for i in range(4,25):
    list_labels = labels_data[str(i)]
    land_dict[str(i)] = defaultdict(int)
    non_land_dict[str(i)] = defaultdict(int)
    for j in range(len(coordinates)):
        if globe.is_land(np.float64(coordinates[j][0][0]), np.float64(coordinates[j][0][1])) == True:
            land_dict[str(i)][int(list_labels[j])]+=1
        else:
            non_land_dict[str(i)][int(list_labels[j])]+=1
with open(clustering['kmeans_path']+'/land_dict.json', 'w') as file:
    json.dump(land_dict, file)

with open(clustering['kmeans_path']+'/non_land_dict.json', 'w') as file:
    json.dump(non_land_dict, file)