import math as m
import numpy as np
import xarray as xr
from trollimage.xrimage import XRImage
from tqdm import tqdm
import random as r
import os
# import netcdf4
month = ['0'+str(i) for i in range(1,10)]
month.extend([str(i) for i in range(10,13)])
day_31 = ['0'+str(i) for i in range(1,10)]
day_31.extend([str(i) for i in range(10,32)])
day_30 = ['0'+str(i) for i in range(1,10)]
day_30.extend([str(i) for i in range(10,31)])
day_28 = ['0'+str(i) for i in range(1,10)]
day_28.extend([str(i) for i in range(10,29)])
similar_month = r.choice(month)
similar_day = r.choice(day_31)

def sample_new2(array):
    shape_sim = np.shape(array.data)
    x = r.randint(256, shape_sim[2]-256)
    y = r.randint(256, shape_sim[1]-256)
    sliced = array.data.isel(x=slice(x, x+128), y=slice(y, y+128)) # To get teh slice of the array
    nearest = sliced.isel(x=56, y=56) # To get the nearest pixel
    lat, long = nearest.coords['latitude'].data, nearest.coords['longitude'].data # To get the latitude pf nearest pixel
    return sliced.values, lat, long

def array(path):
    similar_month = r.choice(month)
    path = path + similar_month
    if int(similar_month) in [1, 3, 5, 7, 8, 10, 12]:
        similar_day = r.choice(day_31)
    elif int(similar_month) == 2:
        similar_day = r.choice(day_28)
    else:
        similar_day = r.choice(day_30)
    path = path + '/' + similar_day
    files = os.listdir(path)
    nc_path = r.choice(files)
    nc_path = path + '/' + nc_path
    ds = xr.open_dataset(nc_path, )
    img = XRImage(ds['day_microphysics'])
    img.crude_stretch(min_stretch=[2.044909, 1.795136, 236.558919], max_stretch=[78.810258, 22.026770, 300.755732])
    return img

def triplet(path):
    while True:
        try:
            similar = array(path)
        except:
            continue
        if np.shape(similar.data)[1] != 2030:
            continue 
        a, lat, long = sample_new2(similar)
        if np.any(np.isnan(a)):
            continue
        else:
            break
    return a, lat, long


# orig_memmap = np.memmap('./storage/climate-memmap/orig_memmap.memmap', dtype='float64', mode='w+', shape=(1000000, 3, 3, 128, 128))
path = '/storage/my-sdsc-storage/nrp/protected/sio/MODIS_Aqua_microphysics_images/2011/'
# for j in tqdm(range(1000000)):
#     # mem_file = np.memmap('./storage/climate-memmap/orig_memmap.memmap',  dtype='float64', mode='r+', shape=(1000000, 3, 3, 128, 128))
#     orig_memmap[j] = triplet(path)

import multiprocessing
import numpy as np
from tqdm import tqdm
import time
import json
# Function to generate triplet data
def generate_triplet(j):
    path = '/storage/my-sdsc-storage/nrp/protected/sio/MODIS_Aqua_microphysics_images/2011/'
    mem_file = np.memmap(
            '/storage/climate-memmap/coordinates_data/data/data_coords_'+str(j)+'.memmap',
            dtype='float64',
            mode='w+',
            shape=(1000, 3, 128, 128))
    coordinates = []
    for i in range(1000):
        mem_file[i], lat, long = triplet(path)
        coordinates.append([float(lat), float(long)])
    # print(coordinates)
    with open('/storage/climate-memmap/coordinates_data/coords/coord_'+str(j)+'.json', 'w') as file:
        json.dump(coordinates, file)
  
#         pass
start = time.time()
if __name__ == "__main__":
    print(start)
    num_processes = 20  # Number of processes to use
    pool = multiprocessing.Pool(processes=num_processes)

    results = pool.map(generate_triplet, range(num_processes))
print(time.time()-start)