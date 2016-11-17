__author__ = 'Thushan Ganegedara'

import load_data
from matplotlib import pyplot as plt
from scipy.misc import imsave
import numpy as np
if __name__=='__main__':

    load_data.load_and_save_data_cifar10(filename='cifar-10-whitened.pickle',zca_whiten=True,return_original=True)
    (tr_white_dataset,tr_labels),(v_white_dataset,v_labels),(ts_white_dataset,ts_labels) = load_data.reformat_data_cifar10(filename='cifar-10-whitened.pickle')
    #(tr_dataset,tr_labels),(v_dataset,v_labels),(ts_dataset,ts_labels) = load_data.reformat_data_cifar10(filename='cifar-10.pickle')
    for i in range(5):
        rand_idx = np.random.randint(0,5000)
        #imsave('test_img.png', tr_dataset[rand_idx,:,:,:])
        imsave('test_img_whitened_'+str(i)+'.png', tr_white_dataset[rand_idx,:,:,:])
