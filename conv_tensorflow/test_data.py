__author__ = 'Thushan Ganegedara'

import load_data
from scipy.misc import imsave
if __name__=='__main__':

    (tr_dataset,tr_labels),(v_dataset,v_labels),(ts_dataset,ts_labels) = load_data.reformat_data_cifar10()
    imsave('/test_img.png', tr_dataset[1,:,:,:])
