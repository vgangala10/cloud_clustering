import math as m
import numpy as np
import xarray as xr
from trollimage.xrimage import XRImage
from tqdm import tqdm
import random as r
import os
import json
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


def sample_new2(array, distant, array_path, distant_path):
    shape_sim = np.shape(array)
    shape_dist = np.shape(distant)
    coords = []
    a = np.empty((3, 3, 128, 128))
    x = r.randint(256, shape_sim[2]-256) # To get the random x coordinate 256 pixels away from the boundary
    y = r.randint(256, shape_sim[1]-256) # To get the random y coordinate 256 pixels away from the boundary
    # similar_1 = sliced(array, coords(x,y))
    similar_1= array.isel(x=slice(x, x+128), y=slice(y, y+128)) # To get the slice of the array
    nearest = similar_1.isel(x=56, y=56) # To get the middle pixel
    lat, long = nearest.coords['latitude'].data, nearest.coords['longitude'].data # To get the coordinates of the middle pixel
    coords.append([float(lat), float(long), array_path]) 

    # for neighboring crop
    random_angle_rad = r.uniform(0, 2*m.pi) # To get the random angle
    radius = r.randint(64, 128) # To get the radius of the circle
    x_near = x + int(radius * m.cos(random_angle_rad)) # To get the x coordinate of the neighboring pixel
    y_near = y + int(radius * m.sin(random_angle_rad)) # To get the y coordinate of the neighboring pixel
    similar_2 = array.isel(x=slice(x_near, x_near+128), y=slice(y_near, y_near+128))
    nearest = similar_2.isel(x=56, y=56) # To get the nearest pixel
    lat, long = nearest.coords['latitude'].data, nearest.coords['longitude'].data # To get the latitude pf nearest pixel
    coords.append([float(lat), float(long), array_path])


    # for distant image
    x_dist = r.randint(256, shape_dist[2]-256)
    y_dist = r.randint(256, shape_dist[1]-256)
    distant_1 = distant.isel(x=slice(x_dist, x_dist+128), y=slice(y_dist, y_dist+128))
    nearest = distant_1.isel(x=56, y=56) # To get the nearest pixel
    lat, long = nearest.coords['latitude'].data, nearest.coords['longitude'].data # To get the latitude pf nearest pixel
    coords.append([float(lat), float(long), distant_path])
    a[0] = similar_1
    a[1] = similar_2
    a[2] = distant_1
    return a, coords

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
    nc_path2 = path + '/' + nc_path
    ds = xr.open_dataset(nc_path2, )
    img = XRImage(ds['day_microphysics'])
    img.crude_stretch(min_stretch=[2.044909, 1.795136, 236.558919], max_stretch=[78.810258, 22.026770, 300.755732])
    return img, nc_path

def triplet(path):
    while True:
        try:
            similar, sim_path = array(path)
            distant, dist_path = array(path)
        except:
            continue
        similar = similar.data
        distant = distant.data
        if np.shape(similar)[1]!=2030 or np.shape(distant)[1]!= 2030:
            if np.shape(similar)[1]==2040:
                similar= similar.isel(x=slice(0, 2030))
            else:
                continue
            if np.shape(distant)[1]==2040:
                distant= distant.isel(x=slice(0, 2030))
            else:
                continue 
        a, coords = sample_new2(similar, distant, sim_path, dist_path)
        if np.any(np.isnan(a)):
            continue
        else:
            break
    return a, coords


# orig_memmap = np.memmap('./storage/climate-memmap/orig_memmap.memmap', dtype='float64', mode='w+', shape=(1000000, 3, 3, 128, 128))
path = '/storage/my-sdsc-storage/nrp/protected/sio/MODIS_Aqua_microphysics_images/2011/'
# for j in tqdm(range(1000000)):
#     # mem_file = np.memmap('./storage/climate-memmap/orig_memmap.memmap',  dtype='float64', mode='r+', shape=(1000000, 3, 3, 128, 128))
#     orig_memmap[j] = triplet(path)

import multiprocessing
import numpy as np
from tqdm import tqdm
import time
# Function to generate triplet data
def generate_triplet(j):
    coordinates_data = []
    path = '/storage/my-sdsc-storage/nrp/protected/sio/MODIS_Aqua_microphysics_images/2011/'
    mem_file = np.memmap(
            '/storage/climate-memmap/triplet_data/orig_memmap'+str(j)+'.memmap',
            dtype='float64',
            mode='w+',
            shape=(10000, 3, 3, 128, 128))
    for i in range(10000):
        mem_file[i], coords = triplet(path)
        coordinates_data.append(coords)
    with open('/storage/climate-memmap/train_coordinates_data/coordinates_data_'+str(j)+'.json', 'w') as file:
        json.dump(coordinates_data, file)
        
        
#         pass
start = time.time()
if __name__ == "__main__":
    print(start)
    num_processes = 30  # Number of processes to use
    pool = multiprocessing.Pool(processes=num_processes)

    results = pool.map(generate_triplet, range(num_processes))

print(time.time()-start)