import tensorflow as tf
import numpy as np
import logging
import sys

logging_level = logging.INFO
logging_format = '[%(funcName)s] %(message)s'

class BasicCNNOpsFacade(object):

    def __init__(self, **params):
        self.basic_hyperparameters = params['basic_hyperparameters']
        self.batch_size = self.basic_hyperparameters['batch_size']
        self.use_l2 = self.basic_hyperparameters['use_l2']
        self.beta = self.basic_hyperparameters['beta']

        self.cnn_hyperparameters = params['cnn_hyperparameters']
        self.cnn_ops = params['cnn_ops']

        self.global_variables = params['global_variables']

        self.logger = logging.getLogger('basic_logger')
        self.logger.setLevel(logging_level)
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter(logging_format))
        console.setLevel(logging_level)
        self.logger.addHandler(console)

    def initialize_cnn_with_ops(self):
        tf_layer_activations = self.global_variables['tf_layer_activations']
        weights,biases = {},{}

        self.logger.info('Initializing the iConvNet (conv_global,pool_global,classifier)...\n')
        for op in self.cnn_ops:
            if 'conv' in op:
                weights[op] = tf.Variable(tf.truncated_normal(
                    self.cnn_hyps[op]['weights'],
                    stddev=2./max(100,self.cnn_hyperparameters[op]['weights'][0]*self.cnn_hyperparameters[op]['weights'][1])
                ),validate_shape = False, expected_shape = self.cnn_hyperparameters[op]['weights'],name=op+'_weights',dtype=tf.float32)
                biases[op] = tf.Variable(tf.constant(
                    np.random.random()*0.001,shape=[self.cnn_hyperparameters[op]['weights'][3]]
                ),validate_shape = False, expected_shape = [self.cnn_hyperparameters[op]['weights'][3]],name=op+'_bias',dtype=tf.float32)

                tf_layer_activations[op] = tf.Variable(tf.zeros(shape=[self.cnn_hyperparameters[op]['weights'][3]], dtype=tf.float32, name = op+'_activations'),validate_shape=False)

                self.logger.debug('Weights for %s initialized with size %s',op,str(self.cnn_hyperparameters[op]['weights']))
                self.logger.debug('Biases for %s initialized with size %d',op,self.cnn_hyperparameters[op]['weights'][3])

            if 'fulcon' in op:
                weights[op] = tf.Variable(tf.truncated_normal(
                    [self.cnn_hyperparameters[op]['in'],self.cnn_hyperparameters[op]['out']],
                    stddev=2./self.cnn_hyperparameters[op]['in']
                ), validate_shape = False, expected_shape = [self.cnn_hyperparameters[op]['in'],self.cnn_hyperparameters[op]['out']], name=op+'_weights',dtype=tf.float32)
                biases[op] = tf.Variable(tf.constant(
                    np.random.random()*0.001,shape=[self.cnn_hyperparameters[op]['out']]
                ), validate_shape = False, expected_shape = [self.cnn_hyperparameters[op]['out']], name=op+'_bias',dtype=tf.float32)

                self.logger.debug('Weights for %s initialized with size %d,%d',
                    op,self.cnn_hyperparameters[op]['in'],self.cnn_hyperparameters[op]['out'])
                self.logger.debug('Biases for %s initialized with size %d',op,self.cnn_hyperparameters[op]['out'])

        return weights,biases

    def initialize_cnn_with_ops_fixed(self):
        weights,biases = {},{}

        self.logger.info('Initializing the iConvNet (conv_global,pool_global,classifier)...\n')
        for op in self.cnn_ops:
            if 'conv' in op:
                weights[op] = tf.Variable(tf.truncated_normal(
                    self.cnn_hyperparameters[op]['weights'],
                    stddev=2./max(100,self.cnn_hyperparameters[op]['weights'][0]*self.cnn_hyperparameters[op]['weights'][1])
                ),validate_shape = True,name=op+'_weights',dtype=tf.float32)
                biases[op] = tf.Variable(tf.constant(
                    np.random.random()*0.001,shape=[self.cnn_hyperparameters[op]['weights'][3]]
                ),validate_shape = True,name=op+'_bias',dtype=tf.float32)

                self.logger.info('Weights for %s initialized with size %s',op,str(self.cnn_hyperparameters[op]['weights']))
                self.logger.info('Biases for %s initialized with size %d',op,self.cnn_hyperparameters[op]['weights'][3])

            if 'fulcon' in op:
                weights[op] = tf.Variable(tf.truncated_normal(
                    [self.cnn_hyperparameters[op]['in'],self.cnn_hyperparameters[op]['out']],
                    stddev=2./self.cnn_hyperparameters[op]['in']
                ), validate_shape = True, name=op+'_weights',dtype=tf.float32)
                biases[op] = tf.Variable(tf.constant(
                    np.random.random()*0.001,shape=[self.cnn_hyperparameters[op]['out']]
                ), validate_shape = True, name=op+'_bias',dtype=tf.float32)

                self.logger.info('Weights for %s initialized with size %d,%d',
                    op,self.cnn_hyperparameters[op]['in'],self.cnn_hyperparameters[op]['out'])
                self.logger.info('Biases for %s initialized with size %d',op,self.cnn_hyperparameters[op]['out'])

        return weights,biases


    def get_logits(self,dataset,weights,biases):

        cnn_ops = self.cnn_ops
        tf_layer_activations = self.global_variables['tf_layer_activations']
        tf_cnn_hyperparameters = self.global_variables['tf_cnn_hyperparameters']

        first_fc = 'fulcon_out' if 'fulcon_0' not in weights else 'fulcon_0'

        self.logger.debug('Defining the logit calculation ...')
        self.logger.debug('\tCurrent set of operations: %s'%cnn_ops)
        activation_ops = []

        x = dataset
        self.logger.debug('\tReceived data for X(%s)...'%x.get_shape().as_list())

        #need to calculate the output according to the layers we have
        for op in cnn_ops:
            if 'conv' in op:
                self.logger.debug('\tConvolving (%s) With Weights:%s Stride:%s'%(op,self.cnn_hyperparameters[op]['weights'],self.cnn_hyperparameters[op]['stride']))
                #logger.debug('\t\tX before convolution:%s'%(x.get_shape().as_list()))
                self.logger.debug('\t\tWeights: %s',tf.shape(weights[op]).eval())
                x = tf.nn.conv2d(x, weights[op], self.cnn_hyperparameters[op]['stride'], padding=self.cnn_hyperparameters[op]['padding'])

                #logger.debug('\t\t Relu with x(%s) and b(%s)'%(tf.shape(x).eval(),tf.shape(biases[op]).eval()))
                x = tf.nn.relu(x + biases[op])
                #logger.debug('\t\tX after %s:%s'%tf.shape(weights[op]).eval())
                activation_ops.append(tf.assign(tf_layer_activations[op],tf.reduce_mean(x,[0,1,2]),validate_shape=False))

            if 'pool' in op:
                self.logger.debug('\tPooling (%s) with Kernel:%s Stride:%s'%(op,self.cnn_hyperparameters[op]['kernel'],self.cnn_hyperparameters[op]['stride']))
                if self.cnn_hyperparameters[op]['type'] is 'max':
                    x = tf.nn.max_pool(x,ksize=self.cnn_hyperparameters[op]['kernel'],strides=self.cnn_hyperparameters[op]['stride'],padding=self.cnn_hyperparameters[op]['padding'])
                elif self.cnn_hyperparameters[op]['type'] is 'avg':
                    x = tf.nn.avg_pool(x,ksize=self.cnn_hyperparameters[op]['kernel'],strides=self.cnn_hyperparameters[op]['stride'],padding=self.cnn_hyperparameters[op]['padding'])

                #logger.debug('\t\tX after %s:%s'%(op,tf.shape(x).eval()))

            if 'fulcon' in op:
                if first_fc==op:
                    # we need to reshape the output of last subsampling layer to
                    # convert 4D output to a 2D input to the hidden layer
                    # e.g subsample layer output [batch_size,width,height,depth] -> [batch_size,width*height*depth]

                    self.logger.debug('Input size of fulcon_out : %d', self.cnn_hyperparameters[op]['in'])
                    x = tf.reshape(x, [self.batch_size, tf_cnn_hyperparameters[op]['in']])
                    x = tf.nn.relu(tf.matmul(x, weights[op]) + biases[op])
                elif 'fulcon_out' == op:
                    x = tf.matmul(x, weights['fulcon_out']) + biases['fulcon_out']
                else:
                    x = tf.nn.relu(tf.matmul(x, weights[op]) + biases[op])

        return x,activation_ops

    def calc_loss(self, logits, labels, weights, weighted=False, tf_data_weights=None):
        # Training computation.
        if self.use_l2:
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels)) + \
                   (self.beta/2)*tf.reduce_sum([tf.nn.l2_loss(w) if 'fulcon' in kw or 'conv' in kw else 0 for kw,w in weights.items()])
        else:
            # use weighted loss
            if weighted:
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels) * tf_data_weights)
            else:
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))

        return loss


    def calc_loss_vector(self, logits,labels):
        return tf.nn.softmax_cross_entropy_with_logits(logits, labels)

    def inc_global_step(self,global_step):
        return global_step.assign(global_step + 1)

    def predict_with_logits(self,logits):
        # Predictions for the training, validation, and test data.
        prediction = tf.nn.softmax(self,logits)
        return prediction

    def predict_with_dataset(self,dataset):
        logits, _ = self.get_logits(dataset, False)
        prediction = tf.nn.softmax(logits)
        return prediction

    def accuracy(self, predictions, labels):
        assert predictions.shape[0] == labels.shape[0]
        return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
                / predictions.shape[0])
