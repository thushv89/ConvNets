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

logger = None
logging_level = logging.INFO
logging_format = '[%(name)s] [%(funcName)s] %(message)s'

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

start_lr = 0.1
decay_learning_rate = True

#dropout seems to be making it impossible to learn
#maybe only works for large nets
dropout_rate = 0.25
use_dropout = True

include_l2_loss = True
#keep beta small (0.2 is too much >0.002 seems to be fine)
beta = 1e-8

policy_interval = 100

train_dataset, train_labels = None,None
valid_dataset, valid_labels = None,None
test_dataset, test_labels = None,None

layer_count = 0 #ordering of layers
time_stamp = 0 #use this to indicate when the layer was added

iconv_ops = ['fulcon_out'] #ops will give the order of layers
hyparams = {
    'fulcon_out':{'in':image_size*image_size*num_channels, 'out':num_labels, 'whd':[image_size,image_size,num_channels]}
}
weights,biases = {},{}

# it's not wise to include always changing values such as (layer_count) in the id
def get_layer_id(layer_name):
    return layer_name

def init_iconvnet():

    print('Initializing the iConvNet...')
    weights[get_layer_id('fulcon_out')] = tf.Variable(tf.truncated_normal(
        [hyparams[get_layer_id('fulcon_out')]['in'],hyparams[get_layer_id('fulcon_out')]['out']],
        stddev=2./hyparams[get_layer_id('fulcon_out')]['in']
    ))
    biases[get_layer_id('fulcon_out')] = tf.Variable(tf.constant(
        np.random.random()*0.01,shape=[hyparams[get_layer_id('fulcon_out')]['out']]
    ))

    print('Weights for %s initialized with size %d,%d'%(
        get_layer_id('fulcon_out'),image_size*image_size*num_channels, num_labels
    ))
    print('Biases for %s initialized with size %d'%(
        get_layer_id('fulcon_out'),num_labels
    ))

#update the fulcon_out layer as we add more layers
#Should use ways to incorporate already existing trained weights
#without randomly initializing
#to do this we do not use the just fan_in but width,height,depth (whd) separately

#devicing better initialization techniques might require (Next release)
def update_fulcon_out(to_w,to_h,to_d):
    global weights

    from_w,from_h,from_d = hyparams[get_layer_id('fulcon_out')]['whd']
    print('\nUpdating fulcon_out base from (%d,%d,%d) to (%d,%d,%d)'%(from_w,from_h,from_d,to_w,to_h,to_d))
    weight_shape = weights[get_layer_id('fulcon_out')].get_shape().as_list()
    print('\tCurrent fulcon_out Weight shape: %s'%weight_shape)

    # adding neurons will give + and removing will give -
    print('\tDetermining the number of weights to be added/removed ...')
    mod_fmap_h = to_h-from_h
    mod_fmap_w = to_w-from_w
    mod_fmap_total = to_w*mod_fmap_h + from_h*mod_fmap_w
    mod_depth = to_d-from_d
    mod_total = mod_fmap_total*to_d + (from_w*from_h)*mod_depth
    print('\tWill modify %s,%s weights (+ is adding, - is removing)'%(mod_total,num_labels))

    if mod_total<0:
        #remove weights (Half from each side on dim 0)
        mod_total = -mod_total
        new_weights = tf.slice(weights[get_layer_id('fulcon_out')],
                               [ceil(mod_total//2),0],[weight_shape[0]-mod_total,num_labels])
        print('\t\tRemoved with begin[%s,%s] and size[%s,%s]'%(ceil(mod_total//2),0,weight_shape[0]-mod_total,num_labels))
        print('\t\tRemoved (%s,%s) weights'%(mod_total,num_labels))


    elif mod_total>0:
        #add weights
        add_weights = tf.Variable(tf.truncated_normal([mod_total,num_labels],stddev=2./mod_total),name='add_weights')
        tf.initialize_variables([add_weights]).run()
        new_weights = tf.concat(0,[weights[get_layer_id('fulcon_out')],add_weights])
        print('\t\tAdded (%s,%s) weights'%(mod_total,num_labels))

    else:
        #do nothing
        print('\t\tNo modification required... Skipping the update')
        new_weights = weights[get_layer_id('fulcon_out')]

    new_shape = new_weights.get_shape().as_list()
    print('\tSize (fulcon_out) after modification %s'%new_shape)
    # assigning the new weights
    hyparams[get_layer_id('fulcon_out')]['in']=new_shape[0]
    weights[get_layer_id('fulcon_out')] = new_weights


def add_conv_layer(w,stride):
    '''
    Specify the tensor variables and hyperparameters for a convolutional neuron layer. And update conv_ops list
    :param w: Weights of the receptive field
    :param stride: Stried of the receptive field
    :return: None
    '''
    global layer_count,weights,biases,hyparams

    conv_id = get_layer_id('conv_'+str(layer_count))
    hyparams[conv_id]={'weights':w,'stride':stride,'padding':'SAME'}
    weights[conv_id]= tf.Variable(tf.truncated_normal(w,stddev=2./(w[0]*w[1])),name='w_'+conv_id)
    biases[conv_id] = tf.Variable(tf.constant(np.random.random()*0.01,shape=[w[3]]),name='b_'+conv_id)

    tf.initialize_variables([weights[conv_id],biases[conv_id]]).run()

    out_id = iconv_ops.pop(-1)
    iconv_ops.append(conv_id)
    iconv_ops.append(out_id)

    layer_count += 1

def add_pool_layer(ksize,stride,type):
    '''
    Specify the hyperparameters for a pooling layer. And update conv_ops
    :param ksize: Kernel size
    :param stride: Stride
    :param type: avg or max
    :return: None
    '''
    global layer_count,hyparams
    pool_id = get_layer_id('pool_'+str(layer_count))
    hyparams[pool_id] = {'type':type,'kernel':ksize,'stride':stride,'padding':'SAME'}

    out_id = iconv_ops.pop(-1)
    iconv_ops.append(pool_id)
    iconv_ops.append(out_id)
    layer_count += 1

def get_logits(dataset):

    print('Current set of operations: %s'%iconv_ops)
    x = dataset
    print('Received data for X(%s)...'%x.get_shape().as_list())

    print('\nPerforming the specified operations ...')
    # if we have added atleaset one layer
    if len(iconv_ops)>1:
        #need to calculate the output according to the layers we have
        for op in iconv_ops:
            if 'conv' in op:
                print('\tCovolving data (%s)'%op)
                x = tf.nn.conv2d(x, weights[op], hyparams[op]['stride'], padding=hyparams[op]['padding'])

                x = tf.nn.relu(x + biases[op])
                print('\t\tX after %s:%s'%(op,x.get_shape().as_list()))
            if 'pool' in op:
                print('\tPooling data')
                x = tf.nn.max_pool(x,ksize=hyparams[op]['kernel'],strides=hyparams[op]['stride'],padding=hyparams[op]['padding'])
                print('\t\tX after %s:%s'%(op,x.get_shape().as_list()))

            if 'fulcon' in op:
                break

        # we need to reshape the output of last subsampling layer to
        # convert 4D output to a 2D input to the hidden layer
        # e.g subsample layer output [batch_size,width,height,depth] -> [batch_size,width*height*depth]
        shape = x.get_shape().as_list()
        rows = shape[0]

        # updating fulcon_layer to match the changed layers (if needed)
        update_fulcon_out(shape[1],shape[2],shape[3])
        hyparams[get_layer_id('fulcon_out')]['whd']=[shape[1],shape[2],shape[3]]
        print('Unwrapping last convolution layer %s to %s hidden layer'%(shape,(rows,hyparams['fulcon_out']['in'])))
        x = tf.reshape(x, [rows,hyparams[get_layer_id('fulcon_out')]['in']])

        return tf.matmul(x, weights[get_layer_id('fulcon_out')]) + \
               biases[get_layer_id('fulcon_out')]

    else:
        # havent added an convolution layers
        shape = x.get_shape().as_list()
        print('Reshaping X of size %s',shape)
        x = tf.reshape(x, [shape[0],hyparams[get_layer_id('fulcon_out')]['in']])
        print('After reshaping X %s',x.get_shape().as_list())
        return tf.matmul(x, weights[get_layer_id('fulcon_out')]) + biases[get_layer_id('fulcon_out')]

def calc_loss(logits,labels):
    # Training computation.
    if include_l2_loss:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels)) + \
               (beta/2)*tf.reduce_sum([tf.nn.l2_loss(w) if 'fulcon' in kw or 'conv' in kw else 0 for kw,w in weights.items()])
    else:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))

    return loss

def optimize_func(loss,global_step):
    # Optimizer.
    if decay_learning_rate:
        learning_rate = tf.train.exponential_decay(start_lr, global_step,decay_steps=500,decay_rate=0.99)
    else:
        learning_rate = start_lr

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    return optimizer,learning_rate

def inc_global_step(global_step):
    return global_step.assign(global_step+1)

def predict_with_logits(logits):
    # Predictions for the training, validation, and test data.
    prediction = tf.nn.softmax(logits)
    return prediction

def predict_with_dataset(dataset):
    prediction = tf.nn.softmax(get_logits(dataset))
    return prediction

def get_parameter_count():
    '''
    Get the total number of parameters in the network
    :return: total number of parametrs as an integer
    '''
    global weights
    param_count = 0
    for op in iconv_ops:
        if 'conv' in op:
            w_shape = weights[op].get_shape().as_list()
            param_count += w_shape[0]*w_shape[1]*w_shape[2]*w_shape[3]
        if 'fulcon' in op:
            w_shape = weights[op].get_shape().as_list()
            param_count += w_shape[0]*w_shape[1]

    return param_count

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])

if __name__=='__main__':
    global logger

    logger = logging.getLogger('main_logger')
    logger.setLevel(logging_level)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(logging_format))
    console.setLevel(logging_level)
    logger.addHandler(console)

    # Value logger will log info used to calculate policies
    value_logger = logging.getLogger('value_logger')
    value_logger.setLevel(logging.INFO)
    value_console = logging.StreamHandler(sys.stdout)
    value_console.setFormatter(logging.Formatter('%(message)s'))
    value_console.setLevel(logging.INFO)
    value_logger.addHandler(value_console)

    load_data.load_and_save_data_cifar10()
    (train_dataset,train_labels),(valid_dataset,valid_labels),(test_dataset,test_labels) = load_data.reformat_data_cifar10()
    train_size,valid_size,test_size = train_dataset.shape[0],valid_dataset.shape[0],test_dataset.shape[0]

    graph = tf.Graph()

    policy_learner = qlearner.ContinuousState(
        {'learning_rate':0.5,'discount_rate':0.9,
         'policy_interval':policy_interval,'gp_interval':5*policy_interval,
         'eta_1':10*policy_interval,'eta_2':20*policy_interval}
    )
    action_picker = learn_best_actions.
    valid_accuracies = []
    with tf.Session(graph=graph) as session:

        #global step is used to decay the learning rate
        global_step = tf.Variable(0, trainable=False)

        logger.info('Input data defined...\n')
        # Input train data
        tf_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
        tf_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

        # Valid data
        tf_valid_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
        tf_valid_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

        init_iconvnet() #initialize the initial conv net

        logits = get_logits(tf_dataset)
        loss = calc_loss(logits,tf_labels)
        pred = predict_with_logits(logits)
        optimize = optimize_func(loss,global_step)
        inc_gstep = inc_global_step(global_step)

        # valid predict function
        pred_valid = predict_with_dataset(tf_valid_dataset)

        tf.initialize_all_variables().run()

        start_time = time.clock()
        for batch_id in range(ceil(train_size//batch_size)-1):
            time_stamp = batch_id
            batch_data = train_dataset[batch_id*batch_size:(batch_id+1)*batch_size, :, :, :]
            batch_labels = train_labels[batch_id*batch_size:(batch_id+1)*batch_size, :]

            # validation batch
            batch_valid_data = train_dataset[(batch_id+1)*batch_size:(batch_id+2)*batch_size, :, :, :]
            batch_valid_labels = train_labels[(batch_id+1)*batch_size:(batch_id+2)*batch_size, :]


            feed_dict = {tf_dataset : batch_data, tf_labels : batch_labels}
            _, l, (_,updated_lr), predictions,_ = session.run([logits,loss,optimize,pred,inc_gstep], feed_dict=feed_dict)

            valid_feed_dict = {tf_valid_dataset:batch_valid_data,tf_valid_labels:batch_valid_labels}
            valid_predictions = session.run([pred_valid],feed_dict=valid_feed_dict)
            valid_accuracies.append(accuracy(valid_predictions[0],batch_valid_labels))

            if batch_id>0 and batch_id % policy_interval == 0:
                end_time = time.clock()
                print('Global step: %d'%global_step.eval())
                print('Minibatch loss at step %d: %f' % (batch_id, l))
                print('Learning rate: %.3f'%updated_lr)
                print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
                mean_valid_accuracy = np.mean(valid_accuracies[len(valid_accuracies)//2:])
                print('%d Mean Valid accuracy (Latter half): %.1f%%' % (time_stamp,mean_valid_accuracy))
                param_count = get_parameter_count()
                print('%d Parameter count: %d'%(time_stamp,param_count))
                duration = end_time-start_time
                print('%d Time taken: %.3f'%(time_stamp,duration))
                start_time = time.clock()
                valid_accuracies = []

                policy_learner.update_policy(time_stamp,{'error_t':mean_valid_accuracy,'time_cost':duration,'param_cost':param_count})

            if batch_id == 101:
                print('\n======================== Adding a convolutional layer ==========================')
                #pred = predict_with_logits(logits)
                add_conv_layer([3,3,3,16],[1,1,1,1])
                logits = get_logits(tf_dataset)
                print('================================================================================\n')
            if batch_id == 301:
                print('\n================= Adding a pooling layer ===========================')
                add_pool_layer([1,2,2,1],[1,2,2,1],'avg')
                logits = get_logits(tf_dataset)
                print('====================================================================\n')
            if batch_id == 1001:
                break
