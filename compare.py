import math as m
import numpy as np
import xarray as xr
from trollimage.xrimage import XRImage
import sys
from datetime import datetime, timedelta
import glob
import matplotlib.pyplot as plt
import json
'''T'''


def array(all_paths):
    path = "/storage/my-sdsc-storage/nrp/protected/sio/MODIS_Aqua_microphysics_images/2010"
    start_date = datetime(2010, 1, 1)
    count = 0
    images = []
    labels = ['Open-cellular_MCC', 'Disorganized_MCC', 'Suppressed_Cu', 'Clustered_cumulus', 'Closed-cellular_MCC', 'Solid_Stratus']
    for i, all_path in enumerate(all_paths):
        array = []
        for path_extend in all_path: #/storage/climate-memmap/classified_cloud_images/Open-cellular_MCC/MYD021KM.A2010154.1845_index_1024_index_0256_Ref7.png
            day = int(path_extend[14:17])
            target_date = start_date + timedelta(days=(day-1))
            month = str(target_date.month).zfill(2)
            day_ = str(target_date.day).zfill(2)
            path_new = path + '/' + month + '/' + day_
            index_dim_1 = int(path_extend[29:33])
            index_dim_2 = int(path_extend[40:44])
            path_extend_ = path_extend[:22]
            pattern = f"{path_new}/{path_extend_}*"
            final_path = glob.glob(pattern)
            try:
                ds = xr.open_dataset(final_path[0], )
                img = XRImage(ds['day_microphysics'])
                img.crude_stretch(min_stretch=[2.044909, 1.795136, 236.558919], max_stretch=[78.810258, 22.026770, 300.755732])
                y = img.data.values
                y = y[:,::-1,::-1]
                slice = y[:, index_dim_1:(index_dim_1+128), index_dim_2:(index_dim_2+128)]
                array.append(slice)
                # slice = np.transpose(slice, (1,2,0))
                # plt.imshow(slice)
                # plt.axis('off')
                # plt.savefig('/storage/climate-memmap/classified_cloud_images_modified/'+labels[i]+'/'+path_extend, format='png')

            except:
                images.append([path_extend, labels[i]])
                count+=1
        print(count)  
        # return slice
        array = np.stack(array)
        print(array.shape)
        mem = np.memmap('/storage/climate-memmap/classified_cloud_images_modified/'+labels[i]+'/memmap2.memmap', dtype = 'float64', mode = 'w+', shape = array.shape)
        mem[:] = array[:]
        mem.flush()
    print(len(images))
    with open('/storage/climate-memmap/classified_cloud_images_modified/error.json', 'w') as file:
        json.dump(images, file)
with open('/storage/climate-memmap/all_files.json', 'r') as file:
    read_filenames = json.load(file)
print(len(read_filenames), len(read_filenames[0]))
array(read_filenames)
# map = np.memmap('/storage/climate-memmap/test_comp.memmap', mode = 'w+', dtype = 'float64', shape = y.shape)
# map[:] = y[:]
# /storage/my-sdsc-storage/nrp/protected/sio/MODIS_Aqua_microphysics_images/2010/06/03/MYD021KM.A2010154.1845.061.2018060063736.nc