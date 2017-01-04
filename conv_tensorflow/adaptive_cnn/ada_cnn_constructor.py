__author__ = 'Thushan Ganegedara'

import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import numpy as np
import os
from math import ceil,floor
import load_data
import logging
import sys
import time
import qlearner
from data_pool import Pool

logger = None
logging_level = logging.DEBUG
logging_format = '[%(funcName)s] %(message)s'

batch_size = 128 # number of datapoints in a single batch

start_lr = 0.001
decay_learning_rate = True

#dropout seems to be making it impossible to learn
#maybe only works for large nets
dropout_rate = 0.25
use_dropout = False

include_l2_loss = True
#keep beta small (0.2 is too much >0.002 seems to be fine)
beta = 1e-5


def initialize_cnn_with_ops(cnn_ops,cnn_hyps):
    weights,biases = {},{}

    print('Initializing the iConvNet (conv_global,pool_global,classifier)...')
    for op in cnn_ops:
        if 'conv' in op:
            weights[op] = tf.Variable(tf.truncated_normal(
                cnn_hyps[op]['weights'],
                stddev=2./(cnn_hyps[op]['weights'][0]*cnn_hyps[op]['weights'][1])
            ),name=op+'_weights')
            biases[op] = tf.Variable(tf.constant(
                np.random.random()*0.01,shape=[cnn_hyps[op]['weights'][3]]
            ),name=op+'_bias')

            print('Weights for %s initialized with size %s'%(op,str(cnn_hyps[op]['weights'])))
            print('Biases for %s initialized with size %d'%(op,cnn_hyps[op]['weights'][3]))

        if 'fulcon' in op:
            weights['fulcon_out'] = tf.Variable(tf.truncated_normal(
                [cnn_hyps[op]['in'],cnn_hyps[op]['out']],
                stddev=2./cnn_hyps[op]['in']
            ),name=op+'_weights')
            biases[op] = tf.Variable(tf.constant(
                np.random.random()*0.01,shape=[cnn_hyps[op]['out']]
            ),name=op+'_bias')

            print('Weights for %s initialized with size %d,%d'%(
                op,cnn_hyps[op]['in'],cnn_hyps[op]['out']
            ))
            print('Biases for %s initialized with size %d'%(
                op,cnn_hyps[op]['out']
            ))
    return weights,biases

def get_logits_with_ops(dataset,cnn_ops,cnn_hyps,weights,biases):
    logger.info('Current set of operations: %s'%cnn_ops)
    x = dataset
    logger.debug('Received data for X(%s)...'%x.get_shape().as_list())

    logger.info('Performing the specified operations ...')

    #need to calculate the output according to the layers we have
    for op in cnn_ops:
        if 'conv' in op:
            logger.debug('\tConvolving (%s) With Weights:%s Stride:%s'%(op,cnn_hyps[op]['weights'],cnn_hyps[op]['stride']))
            logger.debug('\t\tX before convolution:%s'%(x.get_shape().as_list()))
            logger.debug('\t\tWeights: %s',weights[op].get_shape().as_list())
            x = tf.nn.conv2d(x, weights[op], cnn_hyps[op]['stride'], padding=cnn_hyps[op]['padding'])
            logger.debug('\t\t Relu with x(%s) and b(%s)'%(x.get_shape().as_list(),biases[op].get_shape().as_list()))
            x = tf.nn.relu(x + biases[op])
            logger.debug('\t\tX after %s:%s'%(op,x.get_shape().as_list()))
        if 'pool' in op:
            logger.debug('\tPooling (%s) with Kernel:%s Stride:%s'%(op,cnn_hyps[op]['kernel'],cnn_hyps[op]['stride']))
            if cnn_hyps[op]['type'] is 'max':
                x = tf.nn.max_pool(x,ksize=cnn_hyps[op]['kernel'],strides=cnn_hyps[op]['stride'],padding=cnn_hyps[op]['padding'])
            elif cnn_hyps[op]['type'] is 'avg':
                x = tf.nn.avg_pool(x,ksize=cnn_hyps[op]['kernel'],strides=cnn_hyps[op]['stride'],padding=cnn_hyps[op]['padding'])

            logger.debug('\t\tX after %s:%s'%(op,x.get_shape().as_list()))

        if 'fulcon' in op:
            break

    # we need to reshape the output of last subsampling layer to
    # convert 4D output to a 2D input to the hidden layer
    # e.g subsample layer output [batch_size,width,height,depth] -> [batch_size,width*height*depth]
    shape = x.get_shape().as_list()
    rows = shape[0]

    fin_xw,fin_xh = get_final_x(None,None)
    logger.debug("My calculation, TensorFlow calculation: (%d,%d), (%d,%d)"%(fin_xw,fin_xh,shape[1],shape[2]))
    assert fin_xw == shape[1] and fin_xh == shape[2]

    print('Unwrapping last convolution layer %s to %s hidden layer'%(shape,(rows,cnn_hyps['fulcon_out']['in'])))
    x = tf.reshape(x, [rows,cnn_hyps['fulcon_out']['in']])

    return tf.matmul(x, weights['fulcon_out']) + biases['fulcon_out']

def calc_loss(logits,labels):
    # Training computation.
    if include_l2_loss:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels)) + \
               (beta/2)*tf.reduce_sum([tf.nn.l2_loss(w) if 'fulcon' in kw or 'conv' in kw else 0 for kw,w in weights.items()])
    else:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))

    return loss

def calc_loss_vector(logits,labels):
    return tf.nn.softmax_cross_entropy_with_logits(logits, labels)

def optimize_func(loss,global_step):
    # Optimizer.
    if decay_learning_rate:
        learning_rate = tf.train.exponential_decay(start_lr, global_step,decay_steps=1,decay_rate=0.95)
    else:
        learning_rate = start_lr

    optimizer = tf.train.MomentumOptimizer(learning_rate,momentum=0.9).minimize(loss)
    return optimizer,learning_rate

def optimize_with_fixed_lr_func(loss,learning_rate):
    optimizer = tf.train.MomentumOptimizer(learning_rate,momentum=0.9).minimize(loss)
    return optimizer

def inc_global_step(global_step):
    return global_step.assign(global_step+1)

def predict_with_logits(logits):
    # Predictions for the training, validation, and test data.
    prediction = tf.nn.softmax(logits)
    return prediction

def predict_with_dataset(dataset,cnn_ops,cnn_hyps,weights,biases):
    prediction = tf.nn.softmax(get_logits_with_ops(dataset,cnn_ops,cnn_hyps,weights,biases))
    return prediction

def accuracy(predictions, labels):
    assert predictions.shape[0]==labels.shape[0]
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])

if __name__=='__main__':

    #type of data training
    datatype = 'cifar-10'
    if datatype=='cifar-10':
        image_size = 32
        num_labels = 10
        num_channels = 3 # rgb
    elif datatype=='notMNIST':
        image_size = 28
        num_labels = 10
        num_channels = 1 # grayscale

    logger = logging.getLogger('main_logger')
    logger.setLevel(logging_level)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(logging_format))
    console.setLevel(logging_level)
    logger.addHandler(console)


    #TODO: Load data 75*batch_size (train) & 25*batch_size (valid)
    data_dir = '..'+os.sep+'..'+os.sep+'data'
    (dataset,labels),(_,_),(_,_)=load_data.reformat_data_cifar10(data_dir,'cifar-10.pickle')
    train_dataset,train_labels = dataset[:75*batch_size,:,:,:],labels[:75*batch_size,:]
    valid_dataset,valid_labels = dataset[75*batch_size:100*batch_size,:,:,:],labels[75*batch_size:100*batch_size,:]
    train_size,valid_size = train_dataset.shape[0],valid_dataset.shape[0]

    policy_iterations = 50
    construction_epochs = 10

    #Construction Policy Learner
    constructor = qlearner.AdaCNNConstructionQLearner(learning_rate=0.1,discount_rate=0.99,image_size=image_size,upper_bound=8,epsilon=0.99)

    for pIter in range(policy_iterations):
        traj_states = constructor.output_trajectory()
        cnn_ops = []
        cnn_hyperparameters = {}
        last_feature_map_depth = 0 # need this to calculate the fulcon layer in size
        prev_conv_hyp = None
        # building ops and hyperparameters out of states
        for state in traj_states:
            # state (layer_depth,op=(type,kernel,stride,depth),out_size)
            op = state[1] # op => type,kernel,stride,depth
            depth_index = state[0]
            if op[0] is 'C':
                op_id = 'conv_'+str(depth_index)
                if prev_conv_hyp is None:
                    hyps = {'weights':[op[1],op[1],num_channels,op[3]],'stride':[1,op[2],op[2],1],'padding':'SAME'}
                else:
                    hyps = {'weights':[op[1],op[1],prev_conv_hyp['weights'][3],op[3]],'stride':[1,op[2],op[2],1],'padding':'SAME'}

                cnn_ops.append(op_id)
                cnn_hyperparameters[op_id]=hyps
                prev_conv_hyp = hyps # need this to set the input depth for a conv layer
                last_feature_map_depth = op[3]

            elif op[0] is 'P':
                op_id = 'pool_'+str(depth_index)
                hyps = {'type':'max','kernel':[1,op[1],op[1],1],'stride':[1,op[2],op[2],1],'padding':'SAME'}
                cnn_ops.append(op_id)
                cnn_hyperparameters[op_id]=hyps
            elif op[0] == 'Terminate':
                output_size = traj_states[-2][2]
                if output_size>1:
                    cnn_ops.append('pool_global')
                    pg_hyps =  {'type':'avg','kernel':[1,output_size,output_size,1],'stride':[1,1,1,1],'padding':'VALID'}

                op_id = 'fulcon_out'
                hyps = {'in':output_size*output_size*last_feature_map_depth,'out':num_labels}
                cnn_ops.append(op_id)
                cnn_hyperparameters[op_id]=hyps
            elif op[0] == 'Init':
                continue
            else:
                print('='*60)
                print(op[0])
                print('='*60)
                raise NotImplementedError

        weights,biases = initialize_cnn_with_ops(cnn_ops,cnn_hyperparameters)

        graph = tf.Graph()

        with tf.Session(graph=graph,config = tf.ConfigProto(allow_soft_placement=True)) as session, tf.device('/gpu:0'):
            global_step = tf.Variable(0, trainable=False)

            logger.info('Input data defined...\n')
            # Input train data
            tf_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
            tf_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

            tf_pool_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
            tf_pool_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

            # Valid data
            tf_valid_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
            tf_valid_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

            logits = get_logits_with_ops(tf_dataset,cnn_ops,cnn_hyperparameters,weights,biases)
            loss = calc_loss(logits,tf_labels)
            pred = predict_with_logits(logits)
            optimize = optimize_func(loss,global_step)
            inc_gstep = inc_global_step(global_step)

            # valid predict function
            pred_valid = predict_with_dataset(tf_valid_dataset,cnn_ops,cnn_hyperparameters,weights,biases)

            tf.initialize_all_variables().run()

            for cEpoch in range(construction_epochs):
                for batch_id in range(ceil(train_size//batch_size)-1):

                    batch_data = train_dataset[batch_id*batch_size:(batch_id+1)*batch_size, :, :, :]
                    batch_labels = train_labels[batch_id*batch_size:(batch_id+1)*batch_size, :]

                    feed_dict = {tf_dataset : batch_data, tf_labels : batch_labels}
                    _, l, (_,updated_lr), predictions = session.run(
                            [logits,loss,optimize,pred], feed_dict=feed_dict
                    )

                full_valid_predictions = None
                for vbatch_id in range(ceil(train_size//batch_size)-1):
                    # validation batch
                    batch_valid_data = valid_dataset[vbatch_id*batch_size:(vbatch_id+1)*batch_size, :, :, :]
                    batch_valid_labels = valid_labels[vbatch_id*batch_size:(vbatch_id+1)*batch_size, :]

                    feed_valid_dict = {tf_valid_dataset:batch_valid_data, tf_valid_labels:batch_valid_labels}
                    valid_predictions = session.run([pred_valid],feed_dict=feed_valid_dict)

                    if full_valid_predictions is None:
                        full_valid_predictions = np.asarray(valid_predictions[0],dtype=np.float32)
                    else:
                        full_valid_predictions = np.append(full_valid_predictions,valid_predictions[0],axis=0)

                valid_accuracy = accuracy(full_valid_predictions, valid_labels)
                _ = session.run([inc_gstep])

            del weights,biases
            constructor.update_policy({'accuracy':valid_accuracy,'trajectory':traj_states})