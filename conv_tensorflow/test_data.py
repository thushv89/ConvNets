__author__ = 'Thushan Ganegedara'

import load_data
from scipy.misc import imsave
import numpy as np
if __name__=='__main__':

    load_data.load_and_save_data_cifar10(filename='cifar-10-white.pickle',zca_whiten=True,return_original=True,separate_rgb=False)
    (tr_white_dataset,tr_labels),(v_white_dataset,v_labels),(ts_white_dataset,ts_labels) = load_data.reformat_data_cifar10(filename='cifar-10-white.pickle')
    #(tr_dataset,tr_labels),(v_dataset,v_labels),(ts_dataset,ts_labels) = load_data.reformat_data_cifar10(filename='cifar-10.pickle')
    for i in range(10):
        rand_idx = np.random.randint(0,5000)
        #imsave('test_img.png', tr_dataset[rand_idx,:,:,:])
        imsave('test_img_whitened_'+str(i)+'.png', tr_white_dataset[rand_idx,:,:,:])
