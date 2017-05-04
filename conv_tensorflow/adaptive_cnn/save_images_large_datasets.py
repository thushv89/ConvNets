import numpy as np
import os
from scipy.misc import imsave

dataset = 'svhn-10'
dataset_filename = 'data_non_station/svhn-10-nonstation-dataset.pkl'
col_count = (32,32,3)
start_idx = 0
stop_idx = 1000

gen_perist_dir = dataset + '-traindata'
gen_save_count = 0
fp1 = np.memmap(dataset_filename,dtype=np.float32,mode='r',
                    offset=np.dtype('float32').itemsize*col_count[0]*col_count[1]*col_count[2]*start_idx,shape=(stop_idx,col_count[0],col_count[1],col_count[2]))

train_data = fp1[:,:,:,:]

if not os.path.exists(gen_perist_dir):
    os.makedirs(gen_perist_dir)

for im in train_data:
    imsave(gen_perist_dir + os.sep + dataset + '_image_'
           + str(gen_save_count) + '_' + '.png', im)
    gen_save_count += 1