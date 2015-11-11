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

        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)

        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

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

    def shared_dataset(data_xy, borrow=True):

        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX)/255.,
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)

        return shared_x, T.cast(shared_y, 'int32')

    train_x,train_y = shared_dataset([data_x,data_y])

    f = open('data' + os.sep +valid_name, 'rb')
    dict = pickle.load(f,encoding='latin1')
    valid_x,valid_y = shared_dataset([np.asarray(dict.get('data'),dtype=theano.config.floatX),np.asarray(dict.get('labels'),dtype=theano.config.floatX)])

    f = open('data' + os.sep +test_name, 'rb')
    dict = pickle.load(f,encoding='latin1')
    test_x,test_y = shared_dataset([np.asarray(dict.get('data'),dtype=theano.config.floatX),np.asarray(dict.get('labels'),dtype=theano.config.floatX)])

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

def calc_conv_and_pool_params(in_w,in_h,in_ch,fil_w,fil_h, pool_w, pool_h):
    # number of filters per feature map after convolution
    fil_count_w = in_w - fil_w + 1
    fil_count_h = in_h - fil_h + 1

    # number of filters per feature map after max pooling
    out_w = int(fil_count_w/pool_w)
    out_h = int(fil_count_h/pool_h)

    return [out_w,out_h]


def eval_conv_net():
    rng = np.random.RandomState(23455)

    # kernel size refers to the number of feature maps in a given layer?
    nkerns=[40, 50, 50]
    full_conn_layer_sizes = [1024,1024,1024]
    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')
    batch_size = 100
    learning_rate = 0.25

    in_w = 32 # input image width
    in_h = 32 # input image height
    in_ch = 3 # input image channels

    fil_w = 5
    fil_h = 5

    pool_w = 2
    pool_h = 2

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

    l0_out_w,l0_out_h = calc_conv_and_pool_params(in_w,in_h,in_ch,fil_w,fil_h,pool_w,pool_h)

    layer1 = ConvPoolLayer(rng,
                           input=layer0.output,
                           image_shape=(batch_size,nkerns[0],l0_out_w,l0_out_h),
                           filter_shape=(nkerns[1], nkerns[0], fil_w, fil_h),
                           poolsize=(pool_w,pool_h)
                           )

    l1_out_w,l1_out_h = calc_conv_and_pool_params(l0_out_w,l0_out_h,in_ch,fil_w,fil_h,pool_w,pool_h)

    layer2 = ConvPoolLayer(rng,
                           input=layer1.output,
                           image_shape=(batch_size,nkerns[1],l1_out_w,l1_out_h),
                           filter_shape=(nkerns[2], nkerns[1], fil_w, fil_h),
                           poolsize=(1,1)
                           )

    l2_out_w, l2_out_h = calc_conv_and_pool_params(l1_out_w,l1_out_h,in_ch,fil_w,fil_h,1,1)

    layer3_input = layer2.output.flatten(2)

    layer3 = HiddenLayer(rng,input=layer3_input,n_in=nkerns[2]* l2_out_w * l2_out_h,n_out=full_conn_layer_sizes[0],activation=T.tanh)
    layer4 = HiddenLayer(rng,input=layer3.output,n_in=full_conn_layer_sizes[0],n_out=full_conn_layer_sizes[1],activation=T.tanh)
    layer5 = HiddenLayer(rng,input=layer4.output,n_in=full_conn_layer_sizes[1],n_out=full_conn_layer_sizes[2],activation=T.tanh)

    layer6 = LogisticRegression(input=layer5.output, n_in=full_conn_layer_sizes[2], n_out=10)

    cost = layer6.negative_log_likelihood(y)

    print('Inputs loaded ...')
    test_model = theano.function(
        [index],
        layer6.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer6.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer6.params + layer5.params + layer4.params + layer3.params + layer2.params + layer1.params + layer0.params

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
    n_train_batches = int(train_set_x.get_value().shape[0]/batch_size)
    n_test_batches = int(test_set_x.get_value().shape[0]/batch_size)

    check_model_io = theano.function(
        [index],
        [layer0.input.shape,layer0.output.shape,layer1.input.shape,layer1.output.shape],
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        },
        on_unused_input='warn'
    )

    for epoch in range(n_epochs):
        print('Epoch ',epoch)
        for minibatch_index in range(n_train_batches):
            print('\t Processing mini batch', minibatch_index)
            cost_ij= train_model(minibatch_index)


        # test it on the test set
        test_losses = [test_model(i) for i in range(n_test_batches)]
        test_score = np.mean(test_losses)
        print('     epoch ',epoch, 'test error of ', test_score * 100.)

if __name__ == '__main__':
    eval_conv_net()