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
        train_set, valid_set, test_set = pickle.load(handle)

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

def calc_chained_out_shape(layer,img_w,img_h,filters, pools,pooling=True):

    fil_count_w = img_w
    fil_count_h = img_h

    out_w = fil_count_w
    out_h = fil_count_h

    #if i<0
    for i in range(layer+1):
        # number of filters per feature map after convolution
        fil_count_w = out_w - filters[i][0] + 1
        fil_count_h = out_h - filters[i][0] + 1

        # number of filters per feature map after max pooling
        if pooling:
            out_w = int(fil_count_w/pools[i][0])
            out_h = int(fil_count_h/pools[i][0])
        else:
            out_w = fil_count_w
            out_h = fil_count_h

    return out_w,out_h

def eval_conv_net():
    rng = np.random.RandomState(23455)

    # kernel size refers to the number of feature maps in a given layer
    # 1st one being number of channels in the image
    conv_activation = 'relu'

    nkerns=[3, 64, 64, 64]
    nkerns_maxout=[3,1, 1, 1]
    fulcon_layer_sizes = [512,256]
    n_conv_layers = len(nkerns)-1
    n_fulcon_layers = len(fulcon_layer_sizes)


    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')
    batch_size = 100
    learning_rate = 0.01
    pooling = True
    img_w = 32 # input image width
    img_h = 32 # input image height
    labels = 10
    # filter width and height
    filters = [(5,5),(3,3),(3,3),(3,3),(1,1)]

    # pool width and height
    pools = [(1,1),(1,1),(1,1),(2,2),(2,2)]

    #datasets = load_data('data' + os.sep + 'mnist.pkl' )
    datasets = load_cifar_10()
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    print('Building the model ...')
    print('Pooling: ',pooling,' ',pools)
    print('Learning Rate: ', learning_rate)
    print('Image(Channels x Width x Height): ',nkerns[0],'x',img_w,'x',img_h)
    layer0_input = x.reshape((batch_size,nkerns[0],img_w,img_h))
    in_shapes = [(img_w,img_h)]
    in_shapes.extend([calc_chained_out_shape(i,img_w,img_h,filters,pools,pooling) for i in range(n_conv_layers)])

    print('Convolutional layers')

    #convolution / max-pooling layers
    if not  conv_activation=='maxout':
        conv_layers = [ConvPoolLayer(rng,
                               image_shape=(batch_size,nkerns[i],in_shapes[i][0],in_shapes[i][1]),
                               filter_shape=(nkerns[i+1], nkerns[i], filters[i][0], filters[i][1]),
                               poolsize=(pools[i][0],pools[i][1]),pooling=pooling,activation=conv_activation)
                       for i in range(n_conv_layers)]
    else:
        # for maxout the number of input kernels will always be 1 since we're only taking the maximum feature map
        conv_layers = [ConvPoolLayer(rng,
                               image_shape=(batch_size,nkerns_maxout[i],in_shapes[i][0],in_shapes[i][1]),
                               filter_shape=(nkerns[i+1], nkerns_maxout[i], filters[i][0], filters[i][1]),
                               poolsize=(pools[i][0],pools[i][1]),pooling=pooling,activation=conv_activation)
                       for i in range(n_conv_layers)]

    # set the input
    for i,layer in enumerate(conv_layers):
        if i==0:
            input = layer0_input
        else:
            input = conv_layers[i-1].output
        layer.process(input)

    print('\nConvolution Layer Info ...')
    check_model_io = [theano.function(
        [],
        [conv_layers[layer_idx].input.shape,conv_layers[layer_idx].output.shape],
        givens={
            x: train_set_x[0 * batch_size: 1 * batch_size]
        },
        on_unused_input='warn'
    ) for layer_idx in range(n_conv_layers)]

    for i_conv in range(n_conv_layers):
        info = check_model_io[i_conv]()
        print('\tConvolutional Layer ',i_conv)
        print('\t\tInput: ',info[0])
        print('\t\tOutput: ',info[1])

    print('\nConvolutional layers created with Max-Pooling ...')

    fulcon_start_in = conv_layers[-1].output.flatten(2)

    fulcon_layers = [
        HiddenLayer(rng,n_in=fulcon_layer_sizes[i-1],n_out=fulcon_layer_sizes[i],activation=T.tanh) if i>0 else
        HiddenLayer(
            rng,
            n_in=(nkerns[-1] if not conv_activation=='maxout' else nkerns_maxout[-1])* in_shapes[-1][0] * in_shapes[-1][1], #if it is maxout there will be only 1 kernel
            n_out=fulcon_layer_sizes[0],activation=T.tanh)
        for i in range(n_fulcon_layers)
    ]

    for i,layer in enumerate(fulcon_layers):
        if i==0:
            input = fulcon_start_in
        else:
            input = fulcon_layers[i-1].output
        layer.process(input)

    print('Fully connected hidden layers created ...')

    classif_layer = LogisticRegression(input=fulcon_layers[-1].output, n_in=fulcon_layer_sizes[-1], n_out=labels)

    cost = classif_layer.negative_log_likelihood(y)

    print('Inputs loaded ...')
    test_model = theano.function(
        [index],
        classif_layer.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        classif_layer.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = []
    params += [l.params[0] for l in fulcon_layers] + [l.params[1] for l in fulcon_layers]
    params += [l.params[0] for l in conv_layers] + [l.params[1] for l in conv_layers]
    params += classif_layer.params

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
    n_valid_batches = int(valid_set_x.get_value().shape[0]/batch_size)
    n_test_batches = int(test_set_x.get_value().shape[0]/batch_size)

    for epoch in range(n_epochs):
        print('Epoch ',epoch)
        for minibatch_index in range(n_train_batches):
            cost_ij= train_model(minibatch_index)
            print('\t Finished mini batch', minibatch_index,' Cost: ',cost_ij)
            if (minibatch_index+1)%100==0:
                v_losses = [validate_model(v_mini_idx) for v_mini_idx in range(n_valid_batches)]
                print('\t Valid Error: ',np.mean(v_losses)*100.)

        # test it on the test set
        test_losses = [test_model(i) for i in range(n_test_batches)]
        test_score = np.mean(test_losses)
        print('     epoch ',epoch, 'test error of ', test_score * 100.)

if __name__ == '__main__':
    eval_conv_net()