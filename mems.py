import numpy as np
import json
for i in range(30, 32):
    with open('/storage/climate-memmap/train_coordinates_data/coordinates_data_'+str(i)+'.json', 'r') as file:
        data = json.load(file)
    dtype_mem = [('custom_int', float), ('custom_float', float), ('custom_str', 'U43')]
    data = np.array(data, dtype = dtype_mem)
    data_mem = np.memmap('/storage/climate-memmap/train_coord_memmaps/coordinates_data_'+str(i)+'.memmap', dtype = dtype_mem, mode = 'w+', shape = (10000, 3, 3))
    data_mem[:] = data[:]
    data_mem.flush()
    del data_mem, data