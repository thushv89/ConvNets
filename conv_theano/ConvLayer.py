__author__ = 'thushv89'

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv

class ConvLayer(object):

    #filter is the receptive field for a single neurons
    #pool size is the size of the subsample where one max value from that will be chosen to bring forward
    def __init__(self, rng, filter_shape, input_shape, activation='relu',layer_id=-1):
        '''
        Initialize the weights and define hyperparameters for the conv layer
        :param rng: Random generator
        :param filter_shape: Receptive field of a neuron (prev_kernel,curr_kernel,filter_width,filter_height)
        :param input_shape: Input shape (batch_size,perv_kernel,img_width,img_height)
        :param activation: Activation type (Recommended: relu)
        :return:
        '''

        assert input_shape[1] == filter_shape[1]
        self.sym_x,self.sym_y = None,None
        self.activation = activation
        self.filter_shape = filter_shape
        self.input_shape = input_shape

        # used for weight initialization
        fan_in = np.prod(filter_shape[1:]) #kernel x f_width x f_height
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:])) #prev_kernel x i_width x i_height

        W_bound = np.sqrt(6./(fan_in + fan_out))

        self.W = theano.shared(np.asarray(
                    rng.uniform( low=-1.0/W_bound, high= 1./W_bound,
                    size=filter_shape), dtype=theano.config.floatX),name='W-conv-'+str(layer_id),borrow=True)

        b_values = np.asarray(np.random.rand(filter_shape[0],)*0.1,dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True,name='b-conv-'+str(layer_id))

        self.params = [self.W, self.b]

    def process(self,sym_x):
        self.sym_x = sym_x

        #subsample is the stride shape of (x,y)
        conv_out = conv.conv2d(input=self.sym_x,filters=self.W,filter_shape=self.filter_shape,image_shape=self.input_shape)

        #
        if self.activation == 'relu':
            self.sym_y = T.nnet.relu(conv_out + self.b.dimshuffle('x',0,'x','x'))
        elif self.activation == 'tanh':
            self.sym_y = T.tanh(conv_out + self.b.dimshuffle('x',0,'x','x'))
        else:
            raise NotImplementedError
