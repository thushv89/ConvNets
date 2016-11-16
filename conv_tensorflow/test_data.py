__author__ = 'Thushan Ganegedara'

import load_data
from matplotlib import pyplot as plt
from scipy.misc import imsave
import numpy as np
if __name__=='__main__':

    (tr_dataset,tr_labels),(v_dataset,v_labels),(ts_dataset,ts_labels) = load_data.reformat_data_cifar10()
    imsave('test_img.png', tr_dataset[np.random.randint(0,10000),:,:,:])
    plt.imshow(tr_dataset[5,:,:,:])