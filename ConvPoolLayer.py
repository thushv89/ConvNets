__author__ = 'Thushan Ganegedara'

import theano
import numpy as np
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

class ConvPoolLayer(object):

    #filter is the receptive field for a single neurons
    #pool size is the size of the subsample where one max value from that will be chosen to bring forward
    def __init__(self, rng, filter_shape, image_shape, poolsize=(2,2), pooling=True):

        assert image_shape[1] == filter_shape[1]
        self.input,self.output = None,None
        self.pooling = pooling
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.pool_size = poolsize

        fan_in = np.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:])/np.prod(poolsize))

        W_bound = np.sqrt(6./(fan_in + fan_out))

        self.W = theano.shared(np.asarray(
                    rng.uniform( low=-1.0/W_bound, high= 1./W_bound,
                    size=filter_shape), dtype=theano.config.floatX),name='W',borrow=True)

        b_values = np.asarray(np.random.rand(filter_shape[0],)*0.1,dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True,name='b')

        self.params = [self.W, self.b]



    def process(self,input):

        self.input = input

        conv_out = conv.conv2d(input=input,filters=self.W,filter_shape=self.filter_shape,image_shape=self.image_shape)

        if self.pooling:
            pooled_out = downsample.max_pool_2d(input=conv_out,ds=self.pool_size,ignore_border=True)
        else:
            pooled_out = conv_out
        # linear rectified units are said to process faster (available as T.nnet.relu in 0.7.1)
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x',0,'x','x'))

