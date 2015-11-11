__author__ = 'Thushan Ganegedara'

import theano
import numpy as np
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

class ConvPoolLayer(object):

    #filter is the receptive field for a single neurons
    #pool size is the size of the subsample where one max value from that will be chosen to bring forward
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2,2) ):

        assert image_shape[1] == filter_shape[1]
        self.input = input

        fan_in = np.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:])/np.prod(poolsize))

        W_bound = np.sqrt(6./(fan_in + fan_out))

        self.W = theano.shared(np.asarray(
    rng.uniform( low=-1.0/W_bound, high= 1./W_bound,
                 size=filter_shape),
    dtype=theano.config.floatX),name='W',borrow=True)

        b_values = np.zeros((filter_shape[0],),dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        conv_out = conv.conv2d(input=input,filters=self.W,filter_shape=filter_shape,image_shape=image_shape)

        pooled_out = downsample.max_pool_2d(input=conv_out,ds=poolsize,ignore_border=True)

        self.output = T.tanh(pooled_out + self.b.dimshuffle('x',0,'x','x'))

        self.params = [self.W, self.b]

        self.input = input