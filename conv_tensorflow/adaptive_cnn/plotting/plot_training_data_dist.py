import numpy as np
import os
from collections import Counter
import logging

dataset = 'svhn-10'
behavior = 'non-stationary'
label_filename='data_non_station'+os.sep+'svhn-10-nonstation-labels.pkl'
num_labels = 10
dataset_size = 1280000
batch_size = 128
def load_data_from_memmap(label_filename,num_labels,start_idx,size):
    global logger,tf_distort_data_batch,distorted_imgs


    fp2 = np.memmap(label_filename,dtype=np.int32,mode='r',
                    offset=np.dtype('int32').itemsize*1*start_idx,shape=(size,1))

    train_labels = fp2[:]
    return np.asarray(train_labels).astype(np.int32).flatten()


if __name__ == '__main__':

    pool_dist_logger = logging.getLogger('pool_distribution_logger')
    pool_dist_logger.setLevel(logging.INFO)
    poolHandler = logging.FileHandler('data_distribution_'+dataset+' '+behavior+'.log', mode='w')
    poolHandler.setFormatter(logging.Formatter('%(message)s'))
    pool_dist_logger.addHandler(poolHandler)
    pool_dist_logger.info('#Class distribution (train data) %s %s',dataset,behavior)

    train_labels = load_data_from_memmap(label_filename,num_labels,0,dataset_size)
    print(train_labels[:10])
    for batch_id in range(dataset_size//batch_size):
        batch_labels = train_labels[batch_id*batch_size:(batch_id+1)*batch_size]
        lbl_counts = Counter(batch_labels)

        train_dist_str = ''
        prev_dist = 0
        for li in range(num_labels):
            if li in lbl_counts:
                curr_value = (prev_dist + (lbl_counts[li] * 1.0 / batch_size))
                train_dist_str += '%.3f,'%curr_value
            else:
                curr_value = 0 + prev_dist
                train_dist_str += '%.3f,'%curr_value
            #prev_dist = curr_value

        pool_dist_logger.info('%d,%s',batch_id,train_dist_str)

