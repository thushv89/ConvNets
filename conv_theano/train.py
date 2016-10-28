__author__ = 'thushv89'

import theano
from theano import tensor as T
import numpy as np
from ConvLayer import ConvLayer
from PoolLayer import PoolLayer
import os
import pickle
import matplotlib.pyplot as plt

datatype = 'cifar-10'
image_size = 32*32*3
preprocess = ['whiten_per_image']

def split_train_valid(dxy,tr_weights):
    dx,dy = dxy
    perm_idx = np.random.permutation(dx.shape[0])
    tr_idx = perm_idx[:int(tr_weights*dx.shape[0])]
    v_idx = perm_idx[int(tr_weights*dx.shape[0]):]
    return (dx[tr_idx,:],dy[tr_idx,]),(dx[v_idx,:],dy[v_idx,])

def load_data_cifar_10(datatype, dir):

    def shared_dataset(dxy,borrow=True):

        dx,dy = dxy
        x_shr = theano.shared(dx,borrow=borrow)
        y_shr = theano.shared(dy,borrow=borrow)

        return x_shr,T.cast(y_shr,'int32')

    data_x = []
    data_y = []
    test_data_x = []
    test_data_y = []
    if datatype == 'cifar-10':
        for i in range(1,2):
            filename = dir + os.sep + "cifar_10_data_batch_" + str(i)
            with open(filename, 'rb') as handle:
                data = pickle.load(handle)
                data_x.extend(data.get('data'))
                data_y.extend(data.get('labels'))

    (tr_x,tr_y),(v_x,v_y) = split_train_valid(
        (
            np.asarray(data_x,dtype=theano.config.floatX),
            np.asarray(data_y,dtype=theano.config.floatX)
         ),
        0.9)

    tr_x_shr,tr_y_shr = shared_dataset((tr_x,tr_y))

    pre_process_data(tr_x_shr)

def pre_process_data(x_shared):
    idx = T.iscalar('index')
    sym_x = T.dmatrix('X')

    if 'whiten_per_image' in preprocess:
        pp_sym_x = (sym_x-T.mean(sym_x,axis=1).reshape(-1,1))/T.std(sym_x,axis=1).reshape(-1,1)

    print(x_shared.eval().shape)
    test_whitening = theano.function(inputs=[idx],outputs=pp_sym_x,
                                     givens={sym_x:x_shared[idx:idx+2]}
                                     )

    plt.imshow(test_whitening(1)[0])

if __name__=='__main__':

    load_data_cifar_10(datatype,'../data')