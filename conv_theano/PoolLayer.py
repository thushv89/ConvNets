__author__ = 'thushv89'

from theano.tensor.signal import downsample

class PoolLayer(object):
    #filter is the receptive field for a single neurons
    #pool size is the size of the subsample where one max value from that will be chosen to bring forward
    def __init__(self, rng, input_shape, layer_id=-1):
        '''
        Initialize the weights and define hyperparameters for the conv layer
        :param rng: Random generator
        :param input_shape: Input shape (batch_size,perv_kernel,img_width,img_height)
        :param layer_id: ID of the layer
        :return:
        '''

        self.sym_x,self.sym_y = None,None
        self.input_shape = input_shape

    def process(self,sym_x):
        self.sym_x = sym_x

        #subsample is the stride shape of (x,y)
        self.sym_y = downsample.max_pool_2d(input=sym_x,ds=self.pool_size,ignore_border=True)


