__author__ = 'Thushan Ganegedara'

import theano
from theano import tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

import numpy as np

from ConvPoolLayer import ConvPoolLayer
from HiddenLayer import HiddenLayer
from LogisticRegression import LogisticRegression

import pickle
import os

def load_data(filename):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    with open(filename, 'rb') as handle:
        train_set, valid_set, test_set = pickle.load(handle, encoding='latin1')

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

def make_shared(batch_x, batch_y, name, normalize, normalize_thresh=1.0,turn_bw=False):
    '''' Load data into shared variables '''
    if turn_bw:
        dims = batch_x.shape[1]
        bw_data = 0.2989*batch_x[:,0:dims/3] + 0.5870 * batch_x[:,dims/3:(2*dims)/3] + 0.1140 * batch_x[:,(dims*2)/3:dims]
        batch_x = bw_data

    if not normalize:
        x_shared = theano.shared(batch_x, name + '_x_pkl')
    else:
        x_shared = theano.shared(batch_x, name + '_x_pkl')/normalize_thresh
    max_val = np.max(x_shared.eval())
    assert 0.004<=max_val<=1.
    y_shared = T.cast(theano.shared(batch_y.astype(theano.config.floatX), name + '_y_pkl'), 'int32')
    size = batch_x.shape[0]

    return x_shared, y_shared

def load_cifar_10():
    train_names = ['cifar_10_data_batch_1','cifar_10_data_batch_2','cifar_10_data_batch_3','cifar_10_data_batch_4']
    valid_name = 'cifar_10_data_batch_5'
    test_name = 'cifar_10_test_batch'

    data_x = []
    data_y = []
    for file_path in train_names:
        f = open('data' + os.sep +file_path, 'rb')
        dict = pickle.load(f,encoding='latin1')
        data_x.extend(dict.get('data'))
        data_y.extend(dict.get('labels'))

    train_x,train_y = make_shared(np.asarray(data_x,dtype=theano.config.floatX),np.asarray(data_y,theano.config.floatX),'train',True, 255.)

    f = open('data' + os.sep +valid_name, 'rb')
    dict = pickle.load(f,encoding='latin1')
    valid_x,valid_y = make_shared(np.asarray(dict.get('data'),dtype=theano.config.floatX),np.asarray(dict.get('labels'),dtype=theano.config.floatX),'valid',True, 255.)

    f = open('data' + os.sep +test_name, 'rb')
    dict = pickle.load(f,encoding='latin1')
    test_x,test_y = make_shared(np.asarray(dict.get('data'),dtype=theano.config.floatX),np.asarray(dict.get('labels'),dtype=theano.config.floatX),'test',True, 255.)

    f.close()

    return [(train_x,train_y),(valid_x,valid_y),(test_x,test_y)]

def create_image_from_vector(vec, dataset):
    from pylab import imshow,show,cm
    if dataset == 'mnist':
        imshow(np.reshape(vec*255,(-1,28)),cmap=cm.gray)
    elif dataset == 'cifar-10':
        r_val = vec[0]
        g_val = vec[1]
        b_val = vec[2]
        gr_img = 0.2989 * r_val + 0.5870 * g_val + 0.1140 * b_val
        #imshow(gr_img*255.,cmap=cm.gray)
        imshow(vec.T)
    show()

def eval_conv_net():
    rng = np.random.RandomState(23455)

    # kernel size refers to the number of feature maps in a given layer?
    nkerns=[20, 50]
    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')
    batch_size = 100
    learning_rate = 0.25

    in_w = 32 # input image width
    in_h = 32 # input image height
    in_ch = 3 # input image channels

    fil_w = 9
    fil_h = 9
    fil_count_w = in_w - fil_w + 1
    fil_count_h = in_h - fil_h + 1

    pool_w = 2
    pool_h = 2
    final_count_w = int(fil_count_w/pool_w)
    final_count_h = int(fil_count_h/pool_h)

    #datasets = load_data('data' + os.sep + 'mnist.pkl' )
    datasets = load_cifar_10()
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    print('Building the model ...')

    layer0_input = x.reshape((batch_size,in_ch,in_w,in_h))

    #test_img = train_set_x.eval().reshape((-1,in_ch,in_w,in_h))[132,:,:,:]
    #create_image_from_vector(test_img,'cifar-10')
    #overlap offest for filters is 1
    layer0 = ConvPoolLayer(rng,
                           input=layer0_input,
                           image_shape=(batch_size,in_ch,in_w,in_h),
                           filter_shape=(nkerns[0],in_ch,fil_w,fil_h),
                           poolsize=(pool_w,pool_h))

    layer1 = ConvPoolLayer(rng,
                           input=layer0.output,
                           image_shape=(batch_size,nkerns[0],final_count_w,final_count_h),
                           filter_shape=(nkerns[1], nkerns[0], fil_w, fil_h),
                           poolsize=(pool_w,pool_h)
                           )

    layer2_input = layer1.output.flatten(2)

    layer2 = HiddenLayer(rng,input=layer2_input,n_in=nkerns[1]*4*4,n_out=500,activation=T.tanh)

    layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10)

    cost = layer3.negative_log_likelihood(y)

    print('Inputs loaded ...')
    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    print('Theano Functions defined ...')
    n_epochs = 25
    n_train_batches = int(train_set_x.eval().shape[0]/batch_size)
    n_test_batches = int(test_set_x.eval().shape[0]/batch_size)

    for epoch in range(n_epochs):
        print('Epoch ',epoch)
        for minibatch_index in range(n_train_batches):
            print('\t Processing mini batch', minibatch_index)
            cost_ij = train_model(minibatch_index)

        # test it on the test set
        test_losses = [test_model(i) for i in range(n_test_batches)]
        test_score = np.mean(test_losses)
        print('     epoch ',epoch, 'test error of ', test_score * 100.)

if __name__ == '__main__':
    eval_conv_net()