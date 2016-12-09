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
import learn_best_actions
from data_pool import Pool

logger = None
logging_level = logging.DEBUG
logging_format = '[%(funcName)s] %(message)s'

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

batch_size = 16 # number of datapoints in a single batch

num_iterations = 3 #

start_lr = 0.01
decay_learning_rate = True

#dropout seems to be making it impossible to learn
#maybe only works for large nets
dropout_rate = 0.25
use_dropout = True

include_l2_loss = True
#keep beta small (0.2 is too much >0.002 seems to be fine)
beta = 1e-3

summary_frequency = 5
assert_true = True

train_dataset, train_labels = None,None
valid_dataset, valid_labels = None,None
test_dataset, test_labels = None,None

layer_count = 0 #ordering of layers
time_stamp = 0 #use this to indicate when the layer was added

incrementally_add_layers = False
if incrementally_add_layers:
    iconv_ops = ['conv_1','pool_1','conv_balance','pool_balance','fulcon_out'] #ops will give the order of layers
else:
    iconv_ops = ['conv_1','pool_1','conv_2','pool_2','conv_3','pool_2','conv_balance','pool_balance','fulcon_out'] #ops will give the order of layers

depth_conv = {'conv_1':64,'conv_2':96,'conv_3':128,'conv_balance':128}

final_2d_output = (4,4)
current_final_2d_output = (0,0)

conv_1_hyparams = {'weights':[5,5,num_channels,depth_conv['conv_1']],'stride':[1,1,1,1],'padding':'SAME'}
conv_2_hyparams = {'weights':[5,5,int(depth_conv['conv_1']),depth_conv['conv_2']],'stride':[1,1,1,1],'padding':'SAME'}
conv_3_hyparams = {'weights':[5,5,int(depth_conv['conv_2']),depth_conv['conv_3']],'stride':[1,1,1,1],'padding':'SAME'}
if incrementally_add_layers:
    conv_balance_hyparams = {'weights':[1,1,int(depth_conv['conv_1']),128],'stride':[1,1,1,1],'padding':'SAME'}
else:
    conv_balance_hyparams = {'weights':[1,1,int(depth_conv['conv_3']),128],'stride':[1,1,1,1],'padding':'SAME'}
pool_1_hyparams = {'type':'max','kernel':[1,3,3,1],'stride':[1,2,2,1],'padding':'SAME'}
pool_2_hyparams = {'type':'max','kernel':[1,3,3,1],'stride':[1,2,2,1],'padding':'SAME'}
if incrementally_add_layers:
    pool_balance_hyparams = {'type':'avg','kernel':[1,2,2,1],'stride':[1,4,4,1],'padding':'SAME'}
else:
    pool_balance_hyparams = {'type':'avg','kernel':[1,2,2,1],'stride':[1,1,1,1],'padding':'SAME'}
out_hyparams = {'in':final_2d_output[0]*final_2d_output[1]*depth_conv['conv_balance'],'out':10}

hyparams = {'conv_1': conv_1_hyparams, 'conv_2': conv_2_hyparams, 'conv_3':conv_3_hyparams,'conv_balance':conv_balance_hyparams,
           'pool_1': pool_1_hyparams, 'pool_2':pool_2_hyparams, 'pool_balance':pool_balance_hyparams,
           'fulcon_out':out_hyparams}


conv_depths = {} # store the in and out depth of each convolutional layer
conv_order  = {} # layer order (in terms of convolutional layer) in the full structure

weights,biases = {},{}

research_parameters = {
    'init_weights_with_existing':False,'seperate_cp_best_actions':False,
    'freeze_threshold':3000,'use_valid_pool':False,'fixed_fulcon':False
}

def accuracy(predictions, labels):
    assert predictions.shape[0]==labels.shape[0]
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])

def init_iconvnet():

    print('Initializing the iConvNet...')

    # Convolutional layers
    for op in iconv_ops:
        if 'conv' in op:
                print('\tDefining weights and biases for %s (weights:%s)'%(op,hyparams[op]['weights']))
                print('\t\tWeights:%s'%hyparams[op]['weights'])
                print('\t\tBias:%d'%hyparams[op]['weights'][3])
                weights[op]=tf.Variable(
                    tf.truncated_normal(hyparams[op]['weights'],
                                        stddev=2./min(10,hyparams[op]['weights'][0])
                                        )
                )
                biases[op] = tf.Variable(tf.constant(np.random.random()*0.0001,shape=[hyparams[op]['weights'][3]]))

    # Fully connected classifier
    weights['fulcon_out'] = tf.Variable(tf.truncated_normal(
        [hyparams['fulcon_out']['in'],hyparams['fulcon_out']['out']],
        stddev=2./hyparams['fulcon_out']['in']
    ))
    biases['fulcon_out'] = tf.Variable(tf.constant(
        np.random.random()*0.01,shape=[hyparams['fulcon_out']['out']]
    ))

    print('Weights for %s initialized with size %d,%d'%(
        'fulcon_out',hyparams['fulcon_out']['in'], num_labels
    ))
    print('Biases for %s initialized with size %d'%(
        'fulcon_out',num_labels
    ))

def append_new_layer_id(layer_id):
    global iconv_ops
    out_id = iconv_ops.pop(-1)
    pool_balance_id = iconv_ops.pop(-1)
    conv_balance_id = iconv_ops.pop(-1)
    iconv_ops.append(layer_id)
    iconv_ops.append(conv_balance_id)
    iconv_ops.append(pool_balance_id)
    iconv_ops.append(out_id)

def update_balance_layer(new_in_depth):
    '''
    If a convolutional layer (intermediate) is removed. This function will
     Correct the dimention mismatch either by adding more weights or removing
    :param to_d: the depth to be updated to
    :return:
    '''
    balance_layer_id = 'conv_balance'
    current_in_depth = hyparams[balance_layer_id]['weights'][2]
    conv_weights = hyparams[balance_layer_id]['weights']

    logger.info("Need to update the layer %s in_depth from %d to %d"%(balance_layer_id,current_in_depth,new_in_depth))
    hyparams[balance_layer_id]['weights'] =[conv_weights[0],conv_weights[1],new_in_depth,conv_weights[3]]

    if new_in_depth<current_in_depth:
        logger.info("\tRequired to remove %d depth layers"%(current_in_depth-new_in_depth))
        # update hyperparameters
        logger.debug('\tNew weights should be: %s'%hyparams[balance_layer_id]['weights'])
        # update weights
        weights[balance_layer_id] = tf.slice(weights[balance_layer_id],
                                                  [0,0,0,0],[conv_weights[0],conv_weights[1],new_in_depth,conv_weights[3]]
                                                  )
        # no need to update biase
    elif new_in_depth>current_in_depth:
        logger.info("\tRequired to add %d depth layers"%(new_in_depth-current_in_depth))

        conv_new_weights = tf.Variable(tf.truncated_normal(
            [conv_weights[0],conv_weights[1],new_in_depth-current_in_depth,conv_weights[3]],
            stddev=2./(conv_weights[0]*conv_weights[1])
        ))
        tf.initialize_variables([conv_new_weights]).run()
        weights[balance_layer_id] = tf.concat(2,[weights[balance_layer_id],conv_new_weights])
    else:
        logger.info('\tNo modifications done...')
        return

    new_shape = weights[balance_layer_id].get_shape().as_list()
    logger.info('Shape of new weights after the modification: %s'%new_shape)

    assert np.all(np.asarray(new_shape)>0)
    assert new_shape == hyparams[balance_layer_id]['weights']

def add_conv_layer(w,stride,conv_id,init_random):
    '''
    Specify the tensor variables and hyperparameters for a convolutional neuron layer. And update conv_ops list
    :param w: Weights of the receptive field
    :param stride: Stried of the receptive field
    :return: None
    '''
    global layer_count,weights,biases,hyparams

    hyparams[conv_id]={'weights':w,'stride':stride,'padding':'SAME'}

    if research_parameters['init_weights_with_existing']:

        raise NotImplementedError

    if not research_parameters['init_weights_with_existing']:
        logger.debug('Initializing %s random ...'%conv_id)
        weights[conv_id]= tf.Variable(tf.truncated_normal(w,stddev=2./(w[0]*w[1])),name='w_'+conv_id)
        biases[conv_id] = tf.Variable(tf.constant(np.random.random()*0.01,shape=[w[3]]),name='b_'+conv_id)

        tf.initialize_variables([weights[conv_id],biases[conv_id]]).run()

    # add the new conv_op to ordered operation list
    append_new_layer_id(conv_id)

    weight_shape = weights[conv_id].get_shape().as_list()

    for op in reversed(iconv_ops[:iconv_ops.index(conv_id)]):
        if 'conv' in op:
            prev_conv_op = op
            break

    assert prev_conv_op != conv_id

    if hyparams[prev_conv_op]['weights'][3]!=hyparams[conv_id]['weights'][3]:
        print('Out Weights of (Previous) %s (%d) and (New) %s (%d) do not match'%(prev_conv_op,hyparams[prev_conv_op]['weights'][3],conv_id,hyparams[conv_id]['weights'][3]))
        update_balance_layer(hyparams[conv_id]['weights'][3])

    if stride[1]>1 or stride[2]>1:
        raise NotImplementedError

    assert np.all(np.asarray(weight_shape)>0)

def add_pool_layer(ksize,stride,type,pool_id):
    '''
    Specify the hyperparameters for a pooling layer. And update conv_ops
    :param ksize: Kernel size
    :param stride: Stride
    :param type: avg or max
    :return: None
    '''
    global layer_count,hyparams
    hyparams[pool_id] = {'type':type,'kernel':ksize,'stride':stride,'padding':'SAME'}

    append_new_layer_id(pool_id)

    if stride[1]>1 or stride[2]>1:
        print('Before changing pool_balance layer %d,%d'%(hyparams['pool_balance']['stride'][1],hyparams['pool_balance']['stride'][2]))
        hyparams['pool_balance']['stride'][1] = np.max([1,hyparams['pool_balance']['stride'][1]-stride[1]])
        hyparams['pool_balance']['stride'][2] = np.max([1,hyparams['pool_balance']['stride'][2]-stride[2]])
        print('After changing pool_balance layer %d,%d'%(hyparams['pool_balance']['stride'][1],hyparams['pool_balance']['stride'][2]))


def get_logits(dataset):
    global final_2d_output,current_final_2d_output

    logger.info('Current set of operations: %s'%iconv_ops)
    x = dataset
    logger.debug('Received data for X(%s)...'%x.get_shape().as_list())

    logger.info('Performing the specified operations ...')

    #need to calculate the output according to the layers we have
    for op in iconv_ops:
        if 'conv' in op:
            logger.debug('\tConvolving (%s) With Weights:%s Stride:%s'%(op,hyparams[op]['weights'],hyparams[op]['stride']))
            logger.debug('\t\tX before convolution:%s'%(x.get_shape().as_list()))
            logger.debug('\t\tWeights: %s',weights[op].get_shape().as_list())
            x = tf.nn.conv2d(x, weights[op], hyparams[op]['stride'], padding=hyparams[op]['padding'])
            logger.debug('\t\t Relu with x(%s) and b(%s)'%(x.get_shape().as_list(),biases[op].get_shape().as_list()))
            x = tf.nn.relu(x + biases[op])
            logger.debug('\t\tX after %s:%s'%(op,x.get_shape().as_list()))

        if 'pool' in op:
            logger.debug('\tPooling (%s) with Kernel:%s Stride:%s'%(op,hyparams[op]['kernel'],hyparams[op]['stride']))

            x = tf.nn.max_pool(x,ksize=hyparams[op]['kernel'],strides=hyparams[op]['stride'],padding=hyparams[op]['padding'])
            logger.debug('\t\tX after %s:%s'%(op,x.get_shape().as_list()))

        if 'fulcon' in op:
            break


    # we need to reshape the output of last subsampling layer to
    # convert 4D output to a 2D input to the hidden layer
    # e.g subsample layer output [batch_size,width,height,depth] -> [batch_size,width*height*depth]
    shape = x.get_shape().as_list()
    current_final_2d_output = (shape[1],shape[2])

    logger.debug("Req calculation, Actual calculation: (%d,%d), (%d,%d)"
                 %(final_2d_output[0],final_2d_output[1],current_final_2d_output[0],current_final_2d_output[1]))
    assert final_2d_output == current_final_2d_output
    rows = shape[0]


    print('Unwrapping last convolution layer %s to %s hidden layer'%(shape,(rows,hyparams['fulcon_out']['in'])))
    x = tf.reshape(x, [rows,hyparams['fulcon_out']['in']])

    return tf.matmul(x, weights['fulcon_out']) + \
           biases['fulcon_out']


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
        learning_rate = tf.train.exponential_decay(start_lr, global_step,decay_steps=1,decay_rate=0.99)
    else:
        learning_rate = start_lr

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    return optimizer,learning_rate

def optimize_with_fixed_lr_func(loss,learning_rate):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    return optimizer

def inc_global_step(global_step):
    return global_step.assign(global_step+1)

def predict_with_logits(logits):
    # Predictions for the training, validation, and test data.
    prediction = tf.nn.softmax(logits)
    return prediction

def predict_with_dataset(dataset):
    prediction = tf.nn.softmax(get_logits(dataset))
    return prediction

if __name__=='__main__':
    global logger

    logger = logging.getLogger('main_logger')
    logger.setLevel(logging_level)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(logging_format))
    console.setLevel(logging_level)
    logger.addHandler(console)

    # Value logger will log info used to calculate policies
    test_logger = logging.getLogger('test_logger')
    test_logger.setLevel(logging.INFO)
    fileHandler = logging.FileHandler('test_log', mode='w')
    fileHandler.setFormatter(logging.Formatter('%(message)s'))
    test_logger.addHandler(fileHandler)


    (train_dataset,train_labels),(valid_dataset,valid_labels),(test_dataset,test_labels) = load_data.reformat_data_cifar10('cifar-10-zca.pickle')
    train_size,valid_size,test_size = train_dataset.shape[0],valid_dataset.shape[0],test_dataset.shape[0]

    graph = tf.Graph()

    valid_accuracy_log = []
    train_accuracy_log = []
    train_loss_log = []
    test_accuracy_log = []

    with tf.Session(graph=graph) as session:

        #global step is used to decay the learning rate
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

        tf_test_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
        tf_test_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

        init_iconvnet() #initialize the initial conv net

        logits = get_logits(tf_dataset)
        loss = calc_loss(logits,tf_labels)
        pred = predict_with_logits(logits)
        optimize = optimize_func(loss,global_step)
        inc_gstep = inc_global_step(global_step)
        loss_vector = calc_loss_vector(logits,tf_labels)

        # valid predict function
        pred_valid = predict_with_dataset(tf_valid_dataset)
        pred_test = predict_with_dataset(tf_test_dataset)

        tf.initialize_all_variables().run()

        for epoch in range(100):

            if epoch==5 and incrementally_add_layers:
                conv_id = 'conv_2'
                add_conv_layer(hyparams[conv_id]['weights'],hyparams[conv_id]['stride'],conv_id,True)
                pool_id = 'pool_2'
                add_pool_layer(hyparams[pool_id]['kernel'],hyparams[pool_id]['stride'],hyparams[pool_id]['type'],pool_id)
                get_logits(tf_dataset)

            elif epoch == 10 and incrementally_add_layers:
                conv_id = 'conv_3'
                add_conv_layer(hyparams[conv_id]['weights'],hyparams[conv_id]['stride'],conv_id,True)
                pool_id = 'pool_2'
                add_pool_layer(hyparams[pool_id]['kernel'],hyparams[pool_id]['stride'],hyparams[pool_id]['type'],pool_id)
                get_logits(tf_dataset)

            for batch_id in range(ceil(train_size//batch_size)-1):

                batch_data = train_dataset[batch_id*batch_size:(batch_id+1)*batch_size, :, :, :]
                batch_labels = train_labels[batch_id*batch_size:(batch_id+1)*batch_size, :]

                batch_start_time = time.clock()
                feed_dict = {tf_dataset : batch_data, tf_labels : batch_labels}
                _, l, l_vec, (_,updated_lr), predictions = session.run(
                        [logits,loss,loss_vector,optimize,pred], feed_dict=feed_dict
                )

                assert not np.isnan(l)
                train_accuracy_log.append(accuracy(predictions, batch_labels))
                train_loss_log.append(l)

            _ = session.run([inc_gstep])

            logger.info('\tGlobal Step %d' %global_step.eval())
            logger.info('\tMean loss at epoch %d: %.5f' % (epoch, np.mean(train_loss_log)))
            logger.debug('\tLearning rate: %.3f'%updated_lr)
            logger.info('\tMinibatch accuracy: %.2f%%\n' % np.mean(train_accuracy_log))
            train_accuracy_log = []
            train_loss_log = []

            for valid_batch_id in range(ceil(valid_size//batch_size)-1):

                # validation batch
                batch_valid_data = valid_dataset[(valid_batch_id)*batch_size:(valid_batch_id+1)*batch_size, :, :, :]
                batch_valid_labels = valid_labels[(valid_batch_id)*batch_size:(valid_batch_id+1)*batch_size, :]

                valid_feed_dict = {tf_valid_dataset:batch_valid_data,tf_valid_labels:batch_valid_labels}
                valid_predictions = session.run([pred_valid],feed_dict=valid_feed_dict)
                valid_accuracy_log.append(accuracy(valid_predictions[0],batch_valid_labels))

            for test_batch_id in range(ceil(test_size//batch_size)-1):

                # validation batch
                batch_test_data = test_dataset[(test_batch_id)*batch_size:(test_batch_id+1)*batch_size, :, :, :]
                batch_test_labels = test_labels[(test_batch_id)*batch_size:(test_batch_id+1)*batch_size, :]

                test_feed_dict = {tf_test_dataset:batch_test_data,tf_test_labels:batch_test_labels}
                test_predictions = session.run([pred_test],feed_dict=test_feed_dict)
                test_accuracy_log.append(accuracy(test_predictions[0],batch_test_labels))

            if epoch > 0 and epoch % summary_frequency == 0:

                logger.info('\n==================== Epoch: %d ====================='%epoch)
                logger.debug('\tGlobal step: %d'%global_step.eval())
                logger.debug('\tCurrent Ops: %s'%iconv_ops)
                mean_valid_accuracy = np.mean(valid_accuracy_log)
                mean_test_accuracy = np.mean(test_accuracy_log)
                logger.debug('\tMean Valid accuracy: %.2f%%' %mean_valid_accuracy)
                logger.debug('\tMean Test accuracy: %.2f%%\n' %mean_test_accuracy)
                valid_accuracy_log = []
                test_accuracy_log = []
