__author__ = 'Thushan Ganegedara'

'''==================================================================
#   This is an experiment to check if incrementally adding layers   #
#   help to reach higher accuracy quicker than starting with all    #
#   the layers at once.                                             #
=================================================================='''

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
import getopt
from scipy.misc import imsave
from skimage.transform import rotate

logger = None
logging_level = logging.DEBUG
logging_format = '[%(funcName)s] %(message)s'

backprop_feature_dir = 'backprop_features'

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

batch_size = 128 # number of datapoints in a single batch


start_lr = 0.001
decay_learning_rate = True

#dropout seems to be making it impossible to learn
#maybe only works for large nets
dropout_rate = 0.25
use_dropout = False

include_l2_loss = False
# keep beta small (0.2 is too much >0.002 seems to be fine)
beta = 1e-1

use_local_res_norm = False
summary_frequency = 5
assert_true = True

train_dataset, train_labels = None,None
valid_dataset, valid_labels = None,None
test_dataset, test_labels = None,None

layer_count = 0 #ordering of layers
time_stamp = 0 #use this to indicate when the layer was added

incrementally_add_layers = True

if incrementally_add_layers:
    if use_local_res_norm:
        iconv_ops = ['conv_1','pool_1','loc_res_norm','conv_balance','pool_balance','fulcon_out'] #ops will give the order of layers
    else:
        iconv_ops = ['conv_1','pool_1','conv_balance','pool_balance','fulcon_out'] #ops will give the order of layers
else:
    if use_local_res_norm:
        iconv_ops = ['conv_1','pool_1','loc_res_norm','conv_2','pool_2','loc_res_norm','conv_3','pool_2','loc_res_norm','conv_balance','pool_balance','fulcon_out'] #ops will give the order of layers
    else:
        iconv_ops = ['conv_1','pool_1','conv_2','pool_2','conv_3','pool_2','conv_balance','pool_balance','fulcon_out'] #ops will give the order of layers

depth_conv = {'conv_1':128,'conv_2':256,'conv_3':512,'conv_balance':1024}

final_2d_output = (4,4)
current_final_2d_output = (0,0)

conv_1_hyparams = {'weights':[5,5,num_channels,depth_conv['conv_1']],'stride':[1,1,1,1],'padding':'SAME'}
conv_2_hyparams = {'weights':[5,5,int(depth_conv['conv_1']),depth_conv['conv_2']],'stride':[1,1,1,1],'padding':'SAME'}
conv_3_hyparams = {'weights':[5,5,int(depth_conv['conv_2']),depth_conv['conv_3']],'stride':[1,1,1,1],'padding':'SAME'}

if incrementally_add_layers:
    conv_balance_hyparams = {'weights':[1,1,int(depth_conv['conv_1']),depth_conv['conv_balance']],'stride':[1,1,1,1],'padding':'SAME'}
else:
    conv_balance_hyparams = {'weights':[1,1,int(depth_conv['conv_3']),depth_conv['conv_balance']],'stride':[1,1,1,1],'padding':'SAME'}

pool_1_hyparams = {'type':'max','kernel':[1,3,3,1],'stride':[1,2,2,1],'padding':'SAME'}
pool_2_hyparams = {'type':'max','kernel':[1,3,3,1],'stride':[1,2,2,1],'padding':'SAME'}
if incrementally_add_layers:
    pool_balance_hyparams = {'type':'avg','kernel':[1,2,2,1],'stride':[1,4,4,1],'padding':'SAME'}
else:
    pool_balance_hyparams = {'type':'avg','kernel':[1,2,2,1],'stride':[1,1,1,1],'padding':'SAME'}
out_hyparams = {'in':final_2d_output[0]*final_2d_output[1]*depth_conv['conv_balance'],'out':10}

hyparams = {'conv_1': conv_1_hyparams, 'conv_2': conv_2_hyparams, 'conv_3':conv_3_hyparams, 'conv_balance':conv_balance_hyparams,
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
                                        stddev=2./(hyparams[op]['weights'][0]*hyparams[op]['weights'][1])
                                        )
                )
                biases[op] = tf.Variable(tf.constant(np.random.random()*1e-6,shape=[hyparams[op]['weights'][3]]))

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
    if use_local_res_norm:
        append_new_layer_id('loc_res_norm')

    if stride[1]>1 or stride[2]>1:
        print('Before changing pool_balance layer %d,%d'%(hyparams['pool_balance']['stride'][1],hyparams['pool_balance']['stride'][2]))
        hyparams['pool_balance']['stride'][1] = np.max([1,hyparams['pool_balance']['stride'][1]-stride[1]])
        hyparams['pool_balance']['stride'][2] = np.max([1,hyparams['pool_balance']['stride'][2]-stride[2]])
        print('After changing pool_balance layer %d,%d'%(hyparams['pool_balance']['stride'][1],hyparams['pool_balance']['stride'][2]))


def get_logits(dataset):
    global final_2d_output,current_final_2d_output
    outputs = []
    logger.info('Current set of operations: %s'%iconv_ops)
    outputs.append(dataset)
    logger.debug('Received data for X(%s)...'%outputs[-1].get_shape().as_list())

    logger.info('Performing the specified operations ...')

    #need to calculate the output according to the layers we have
    for op in iconv_ops:
        if 'conv' in op:
            logger.debug('\tConvolving (%s) With Weights:%s Stride:%s'%(op,hyparams[op]['weights'],hyparams[op]['stride']))
            logger.debug('\t\tX before convolution:%s'%(outputs[-1].get_shape().as_list()))
            logger.debug('\t\tWeights: %s',weights[op].get_shape().as_list())
            outputs.append(tf.nn.conv2d(outputs[-1], weights[op], hyparams[op]['stride'], padding=hyparams[op]['padding']))
            logger.debug('\t\t Relu with x(%s) and b(%s)'%(outputs[-1].get_shape().as_list(),biases[op].get_shape().as_list()))
            outputs[-1] = tf.nn.relu(outputs[-1] + biases[op])
            logger.debug('\t\tX after %s:%s'%(op,outputs[-1].get_shape().as_list()))

        if 'pool' in op:
            logger.debug('\tPooling (%s) with Kernel:%s Stride:%s'%(op,hyparams[op]['kernel'],hyparams[op]['stride']))

            outputs.append(tf.nn.max_pool(outputs[-1],ksize=hyparams[op]['kernel'],strides=hyparams[op]['stride'],padding=hyparams[op]['padding']))
            logger.debug('\t\tX after %s:%s'%(op,outputs[-1].get_shape().as_list()))

        if op=='loc_res_norm':
            print('\tLocal Response Normalization')
            outputs.append(tf.nn.local_response_normalization(outputs[-1], depth_radius=3, bias=None, alpha=1e-2, beta=0.75))

        if 'fulcon' in op:
            break


    # we need to reshape the output of last subsampling layer to
    # convert 4D output to a 2D input to the hidden layer
    # e.g subsample layer output [batch_size,width,height,depth] -> [batch_size,width*height*depth]
    shape = outputs[-1].get_shape().as_list()
    current_final_2d_output = (shape[1],shape[2])

    logger.debug("Req calculation, Actual calculation: (%d,%d), (%d,%d)"
                 %(final_2d_output[0],final_2d_output[1],current_final_2d_output[0],current_final_2d_output[1]))
    assert final_2d_output == current_final_2d_output
    rows = shape[0]

    print('Unwrapping last convolution layer %s to %s hidden layer'%(shape,(rows,hyparams['fulcon_out']['in'])))
    reshaped_output = tf.reshape(outputs[-1], [rows,hyparams['fulcon_out']['in']])

    outputs.append(tf.matmul(reshaped_output, weights['fulcon_out']) + biases['fulcon_out'])

    return outputs

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
        learning_rate = tf.maximum(tf.constant(start_lr*0.1),tf.train.exponential_decay(start_lr, global_step,decay_steps=1,decay_rate=0.99))
    else:
        learning_rate = start_lr

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    return optimizer,learning_rate

def clip_weights_with_threshold(max_threshold):
    global weights
    for op,w in weights.items():
        if 'conv' in op:
            weights[op] = tf.clip_by_value(weights[op], -max_threshold, max_threshold, name=None)
        elif 'fulcon' in op:
            weights[op] = tf.clip_by_value(weights[op], -max_threshold, max_threshold, name=None)

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
    prediction = tf.nn.softmax(get_logits(dataset)[-1])
    return prediction

def get_weight_stats():
    global weights
    stats=dict()
    for op,w in weights.items():
        if 'conv' in op:
            w_nd = w.eval()
            stats[op]= {'min':np.min(w_nd),'max':np.max(w_nd),'mean':np.mean(w_nd),'stddev':np.std(w_nd)}

    return stats


def find_max_activations_for_layer(activations):
    # this should return the id of the image that had max activation for all feature maps

    # this matrix of size b x d
    max_activations_images = tf.reduce_max(activations,axis=[1,2])
    img_id_with_max_activation = tf.argmax(max_activations_images, axis=0)
    return max_activations_images,img_id_with_max_activation

def deconv_featuremap_with_data(layer_id,featuremap_id,tf_selected_dataset,guided_backprop=False):
    global weights,biases

    pool_switches = {}
    activation_masks = {} # used for guided_backprop
    outputs_fwd = []
    logger.info('Current set of operations: %s'%iconv_ops)
    outputs_fwd.append(tf_selected_dataset)
    logger.debug('Received data for X(%s)...'%outputs_fwd[-1].get_shape().as_list())

    logger.info('Performing the forward pass ...')

    #need to calculate the output according to the layers we have
    for op in iconv_ops:
        if 'conv' in op:
            logger.debug('\tConvolving (%s) With Weights:%s Stride:%s'%(op,hyparams[op]['weights'],hyparams[op]['stride']))
            logger.debug('\t\tX before convolution:%s'%(outputs_fwd[-1].get_shape().as_list()))
            logger.debug('\t\tWeights: %s',weights[op].get_shape().as_list())
            outputs_fwd.append(tf.nn.conv2d(outputs_fwd[-1], weights[op], hyparams[op]['stride'], padding=hyparams[op]['padding']))
            logger.debug('\t\t Relu with x(%s) and b(%s)'%(outputs_fwd[-1].get_shape().as_list(),biases[op].get_shape().as_list()))
            outputs_fwd[-1] = tf.nn.relu(outputs_fwd[-1] + biases[op])
            logger.debug('\t\tX after %s:%s'%(op,outputs_fwd[-1].get_shape().as_list()))

            if guided_backprop:
                activation_masks[op] = tf.not_equal(outputs_fwd[-1],tf.constant(0,dtype=tf.float32))
                assert activation_masks[op].get_shape().as_list() == outputs_fwd[-1].get_shape().as_list()

            if op==layer_id:
                break

        if 'pool' in op:
            logger.debug('\tPooling (%s) with Kernel:%s Stride:%s'%(op,hyparams[op]['kernel'],hyparams[op]['stride']))
            pool_out,switch = tf.nn.max_pool_with_argmax(outputs_fwd[-1],ksize=hyparams[op]['kernel'],strides=hyparams[op]['stride'],padding=hyparams[op]['padding'])
            outputs_fwd.append(pool_out)
            pool_switches[op] = switch
            logger.debug('\t\tX after %s:%s'%(op,outputs_fwd[-1].get_shape().as_list()))

        if 'fulcon' in op:
            break

    logger.info('Performing the backward pass ...\n')
    logger.debug('\tInput Size (Non-Zeroed): %s',str(outputs_fwd[-1].get_shape().as_list()))
    # b - batch size, h - height, w - width, d - number of filters
    b,h,w,d = outputs_fwd[-1].get_shape().as_list()

    # outputs[-1] will have the required activation
    # will be of size b x 1 x 1

    # we create a tensor from the activations of the layer which only has non-zeros
    # for the selected feature map (layer_activations_2)
    layer_activations = tf.transpose(outputs_fwd[-1],[3,0,1,2])
    layer_indices = tf.constant([[featuremap_id]])
    layer_updates = tf.expand_dims(layer_activations[featuremap_id,:,:,:],0)
    layer_activations_2 = tf.scatter_nd(layer_indices,layer_updates,tf.constant(layer_activations.get_shape().as_list()))
    layer_activations_2 = tf.transpose(layer_activations_2,[1,2,3,0])
    assert outputs_fwd[-1].get_shape().as_list() == layer_activations_2.get_shape().as_list()

    # single out only the maximally activated neuron and set the zeros
    argmax_indices = tf.argmax(tf.reshape(layer_activations_2,[b,h*w*d]),axis=1)
    batch_range = tf.range(b,dtype=tf.int64)
    nonzero_indices = tf.stack([batch_range,argmax_indices],axis=1)
    updates = tf.gather_nd(tf.reshape(layer_activations_2,[b,h*w*d]),nonzero_indices)
    logger.debug('\tNon-zero indices shape: %s',nonzero_indices.get_shape().as_list())
    logger.debug('\tNon-zero updates shape: %s',updates.get_shape().as_list())
    # Creating the new activations (of size: b x w x h x d)
    # with only the highest activation of given feature map ID non-zero and rest set to zero
    zeroed_activations = tf.scatter_nd(nonzero_indices,updates,tf.constant([b,h*w*d],dtype=tf.int64))
    zeroed_activations = tf.reshape(zeroed_activations,[b,h,w,d])

    outputs_bckwd = [zeroed_activations]
    op_index = iconv_ops.index(layer_id)

    logger.debug('Input Size (Zeroed): %s',str(outputs_bckwd[-1].get_shape().as_list()))

    for op in reversed(iconv_ops[:op_index+1]):
        if 'conv' in op:

            # Deconvolution

            logger.debug('\tDeConvolving (%s) With Weights:%s Stride:%s'%(op,weights[op].get_shape().as_list(),hyparams[op]['stride']))
            logger.debug('\t\tX before deconvolution:%s'%(outputs_bckwd[-1].get_shape().as_list()))
            logger.debug('\t\tWeights: %s',weights[op].get_shape().as_list())

            output_shape = outputs_bckwd[-1].get_shape().as_list()
            output_shape[1] *= hyparams[op]['stride'][1]
            output_shape[2] *= hyparams[op]['stride'][2]
            output_shape[3] = hyparams[op]['weights'][2]
            logger.debug('\t\tExpected output shape: %s',output_shape)
            outputs_bckwd.append(
                    tf.nn.conv2d_transpose(outputs_bckwd[-1], filter=weights[op], strides=hyparams[op]['stride'],
                                           padding=hyparams[op]['padding'],output_shape=tf.constant(output_shape))
            )

            logger.debug('\t\tX after %s:%s'%(op,outputs_bckwd[-1].get_shape().as_list()))

        if 'pool' in op:

            # find previous conv_op
            previous_conv_op = None
            for before_op in reversed(iconv_ops[:iconv_ops.index(op)+1]):
                if 'conv' in before_op:
                    previous_conv_op = before_op
                    break

            logger.debug('Detected previous conv op %s',previous_conv_op)

            # Unpooling operation and Rectification
            logger.debug('\tUnPooling (%s) with Kernel:%s Stride:%s'%(op,hyparams[op]['kernel'],hyparams[op]['stride']))
            logger.debug('\t\tInput shape: %s',outputs_bckwd[-1].get_shape().as_list())

            output_shape = outputs_bckwd[-1].get_shape().as_list()
            output_shape[1] *= hyparams[op]['stride'][1]
            output_shape[2] *= hyparams[op]['stride'][2]

            logger.debug('\t\tExpected output shape: %s',output_shape)
            # Unpooling

            # Switch variable returns an array of size b x h x w x d. But only provide flattened indices
            # Meaning that if you have an output of size 4x4 it will flatten it to a 16 element long array

            # we're goin go make a batch_range which is like [0,0,...,0,1,1,...,1,...]
            # so each unique number will have (h/stride * w/stride * d) elements
            # first it will be of shape b x h/stride x w/stride x d
            # but then we reshape it to b x (h/stride * w/stride * d)
            tf_switches = pool_switches[op]
            tf_batch_range = tf.reshape(tf.range(b,dtype=tf.int64),[b,1,1,1])
            tf_ones_mask = tf.ones_like(tf_switches)
            tf_multi_batch_range = tf_ones_mask * tf_batch_range

            # here we have indices that looks like b*(h/stride)*(w/stride) x 2
            tf_indices = tf.stack([tf.reshape(tf_multi_batch_range,[-1]),tf.reshape(tf_switches,[-1])],axis=1)

            updates = tf.reshape(outputs_bckwd[-1],[-1])

            ref = tf.Variable(tf.zeros([b,output_shape[1]*output_shape[2]*output_shape[3]],dtype=tf.float32),dtype=tf.float32,name='ref_'+op)
            
            session.run(tf.variables_initializer([ref]))

            updated_unpool = tf.scatter_nd(tf.to_int32(tf_indices),updates,tf.constant([b,output_shape[1]*output_shape[2]*output_shape[3]]),name='updated_unpool_'+op)

            outputs_bckwd.append(tf.reshape(updated_unpool,[b,output_shape[1],output_shape[2],output_shape[3]]))

            outputs_bckwd[-1] = tf.nn.relu(outputs_bckwd[-1])

            if guided_backprop and previous_conv_op is not None:
                logger.info('Output-bckwd: %s',outputs_bckwd[-1].get_shape().as_list())
                logger.info('Activation mask %s',activation_masks[previous_conv_op].get_shape().as_list())
                assert outputs_bckwd[-1].get_shape().as_list() == activation_masks[previous_conv_op].get_shape().as_list()
                outputs_bckwd[-1] = outputs_bckwd[-1] * tf.to_float(activation_masks[previous_conv_op])

            logger.debug('\t\tX after %s:%s'%(op,outputs_bckwd[-1].get_shape().as_list()))

    return outputs_fwd,outputs_bckwd

def visualize_with_deconv(session,layer_id,all_x,guided_backprop=False):
    global logger, weights, biases
    '''
    DECONV works the following way.
    # Pick a layer
    # Pick a subset of feature maps or all the feature maps in the layer (if small)
    # For each feature map
    #     Pick the n images that maximally activate that feature map
    #     For each image
    #          Do back propagation for the given activations from that layer until the pixel input layer
    '''

    selected_featuremap_ids = list(np.random.randint(0,depth_conv[layer_id],(20,)))

    examples_per_featuremap = 10
    images_for_featuremap = {} # a dictionary with featuremmap_id : an ndarray with size num_of_images_per_featuremap x image_size
    mean_activations_for_featuremap = {} # this is a dictionary containing featuremap_id : [list of mean activations for each image in order]

    layer_index = iconv_ops.index(layer_id)
    tf_deconv_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_selected_image = tf.placeholder(tf.float32, shape=(examples_per_featuremap, image_size, image_size, num_channels))

    tf_activations = get_logits(tf_deconv_dataset)
    # layer_index+1 because we have input as 0th index
    tf_max_activations, tf_img_ids = find_max_activations_for_layer(tf_activations[layer_index+1])

    # activations for batch of data for a layer of size w(width),h(height),d(depth) will be = b x w x h x d
    # reduce this to b x d by using reduce sum
    image_shape = all_x.shape[1:]
    for batch_id in range(all_x.shape[0]//batch_size):

        batch_data = all_x[batch_id*batch_size:(batch_id+1)*batch_size, :, :, :]
        feed_dict = {tf_deconv_dataset:batch_data}

        # max_activations b x d, img_ids_for_max will be 1 x d
        max_activations,img_ids_for_max = session.run([tf_max_activations,tf_img_ids],feed_dict=feed_dict)
        if batch_id==0:
            logger.debug('Max activations for batch %d size: %s',batch_id,str(max_activations.shape))
            logger.debug('Image ID  for batch %d size: %s',batch_id,str(img_ids_for_max.shape))

        for d_i in range(img_ids_for_max.shape[0]):

            # we only run this for selected set of featurmaps
            if d_i not in selected_featuremap_ids:
                continue

            img_id = np.asscalar(img_ids_for_max[d_i])

            if d_i==selected_featuremap_ids[1]:
                logger.debug('Found %d image id for depth %d',img_id,d_i)

            if d_i not in mean_activations_for_featuremap:
                mean_activations_for_featuremap[d_i] = [np.asscalar(max_activations[img_id,d_i])]
                images_for_featuremap[d_i] = np.reshape(batch_data[img_id,:,:,:],(1,image_shape[0],image_shape[1],image_shape[2]))
            else:
                if len(mean_activations_for_featuremap[d_i])>= examples_per_featuremap:
                    # delete the minimum

                    min_idx = np.asscalar(np.argmin(np.asarray(mean_activations_for_featuremap[d_i])))
                    if d_i==selected_featuremap_ids[1]:
                        logger.debug('Mean activations: %s',mean_activations_for_featuremap[d_i])
                        logger.debug('\tFound minimum activation with %.2f at index %d',np.min(mean_activations_for_featuremap[d_i]),min_idx)

                    del mean_activations_for_featuremap[d_i][min_idx]
                    images_for_featuremap[d_i] = np.delete(images_for_featuremap[d_i],[min_idx],axis=0)
                reshp_img = np.reshape(batch_data[img_id,:,:,:],(1,image_shape[0],image_shape[1],image_shape[2]))
                images_for_featuremap[d_i] = np.append(images_for_featuremap[d_i],reshp_img,axis=0)
                mean_activations_for_featuremap[d_i].append(np.asscalar(max_activations[img_id,d_i]))

    logger.info('Size of image set for feature map: %s',str(len(mean_activations_for_featuremap[selected_featuremap_ids[0]])))

    # TODO: run the following command for all the selected featuremap_id
    all_deconv_outputs = []
    all_images = []
    for d_i in selected_featuremap_ids:
        tf_fwd_outputs, tf_bck_outputs = deconv_featuremap_with_data(layer_id,d_i,tf_selected_image,guided_backprop)
        fwd_outputs, deconv_outputs = session.run([tf_fwd_outputs,tf_bck_outputs],
                                                  feed_dict={tf_selected_image:images_for_featuremap[d_i]})
        all_deconv_outputs.append(deconv_outputs[-1])
        all_images.append(images_for_featuremap[d_i])
    return all_deconv_outputs, all_images

if __name__=='__main__':
    global logger

    try:
        opts,args = getopt.getopt(
            sys.argv[1:],"",['data=',"log_suffix="])
    except getopt.GetoptError as err:
        print('<filename>.py --data= --log_suffix=')

    if len(opts)!=0:
        for opt,arg in opts:
            if opt == '--data':
                data_filename = arg
            if opt == '--log_suffix':
                log_suffix = arg

    logger = logging.getLogger('main_logger')
    logger.setLevel(logging_level)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(logging_format))
    console.setLevel(logging_level)
    logger.addHandler(console)

    # Value logger will log info used to calculate policies
    test_logger = logging.getLogger('test_logger')
    test_logger.setLevel(logging.INFO)
    fileHandler = logging.FileHandler('test_log'+log_suffix+'.log', mode='w')
    fileHandler.setFormatter(logging.Formatter('%(message)s'))
    test_logger.addHandler(fileHandler)

    (train_dataset,train_labels),(valid_dataset,valid_labels),(test_dataset,test_labels) = load_data.reformat_data_cifar10(data_filename)
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

        logits = get_logits(tf_dataset)[-1]
        loss = calc_loss(logits,tf_labels)
        pred = predict_with_logits(logits)
        optimize = optimize_func(loss,global_step)
        inc_gstep = inc_global_step(global_step)
        loss_vector = calc_loss_vector(logits,tf_labels)

        # valid predict function
        pred_valid = predict_with_dataset(tf_valid_dataset)
        pred_test = predict_with_dataset(tf_test_dataset)

        tf.global_variables_initializer().run()

        for epoch in range(31):

            if epoch==5 and incrementally_add_layers:
                conv_id = 'conv_2'
                add_conv_layer(hyparams[conv_id]['weights'],hyparams[conv_id]['stride'],conv_id,True)
                pool_id = 'pool_2'
                add_pool_layer(hyparams[pool_id]['kernel'],hyparams[pool_id]['stride'],hyparams[pool_id]['type'],pool_id)
                get_logits(tf_dataset)[-1]

            elif epoch == 10 and incrementally_add_layers:
                conv_id = 'conv_3'
                add_conv_layer(hyparams[conv_id]['weights'],hyparams[conv_id]['stride'],conv_id,True)
                pool_id = 'pool_2'
                add_pool_layer(hyparams[pool_id]['kernel'],hyparams[pool_id]['stride'],hyparams[pool_id]['type'],pool_id)
                get_logits(tf_dataset)[-1]

            for batch_id in range(ceil(train_size//batch_size)-1):

                batch_data = train_dataset[batch_id*batch_size:(batch_id+1)*batch_size, :, :, :]
                batch_labels = train_labels[batch_id*batch_size:(batch_id+1)*batch_size, :]

                batch_start_time = time.clock()
                feed_dict = {tf_dataset : batch_data, tf_labels : batch_labels}
                _, l, l_vec, (_,updated_lr), predictions = session.run(
                        [logits,loss,loss_vector,optimize,pred], feed_dict=feed_dict
                )

                #print(l)
                #print(get_weight_stats())
                assert not np.isnan(l)
                train_accuracy_log.append(accuracy(predictions, batch_labels))
                train_loss_log.append(l)

            _ = session.run([inc_gstep])

            logger.info('\tGlobal Step %d' %global_step.eval())
            logger.info('\tMean loss at epoch %d: %.5f' % (epoch, np.mean(train_loss_log)))
            logger.debug('\tLearning rate: %.5f'%updated_lr)
            logger.info('\tMinibatch accuracy: %.2f%%\n' % np.mean(train_accuracy_log))
            train_accuracy_log = []
            train_loss_log = []

            if epoch > 0 and epoch % summary_frequency == 0:
                for valid_batch_id in range(ceil(valid_size//batch_size)-1):
                    # validation batch
                    batch_valid_data = valid_dataset[(valid_batch_id)*batch_size:(valid_batch_id+1)*batch_size, :, :, :]
                    batch_valid_labels = valid_labels[(valid_batch_id)*batch_size:(valid_batch_id+1)*batch_size, :]

                    valid_feed_dict = {tf_valid_dataset:batch_valid_data,tf_valid_labels:batch_valid_labels}
                    valid_predictions = session.run([pred_valid],feed_dict=valid_feed_dict)
                    valid_accuracy_log.append(accuracy(valid_predictions[0],batch_valid_labels))

                for test_batch_id in range(ceil(test_size//batch_size)-1):
                    # test batch
                    batch_test_data = test_dataset[(test_batch_id)*batch_size:(test_batch_id+1)*batch_size, :, :, :]
                    batch_test_labels = test_labels[(test_batch_id)*batch_size:(test_batch_id+1)*batch_size, :]

                    test_feed_dict = {tf_test_dataset:batch_test_data,tf_test_labels:batch_test_labels}
                    test_predictions = session.run([pred_test],feed_dict=test_feed_dict)
                    test_accuracy_log.append(accuracy(test_predictions[0],batch_test_labels))

                logger.info('\n==================== Epoch: %d ====================='%epoch)
                logger.debug('\tGlobal step: %d'%global_step.eval())
                logger.debug('\tCurrent Ops: %s'%iconv_ops)
                mean_valid_accuracy = np.mean(valid_accuracy_log)
                mean_test_accuracy = np.mean(test_accuracy_log)
                logger.debug('\tMean Valid accuracy: %.2f%%' %mean_valid_accuracy)
                logger.debug('\tMean Test accuracy: %.2f%%\n' %mean_test_accuracy)
                valid_accuracy_log = []
                test_accuracy_log = []

                test_logger.info('%d,%.2f,%.2f',epoch,mean_valid_accuracy,mean_test_accuracy)

        layer_ids = ['conv_2','conv_3']
        rotate_required = True
        for lid in layer_ids:
            all_deconvs,all_images = visualize_with_deconv(session,lid,valid_dataset,True)
            d_i = 0
            for deconv_di, images_di in zip(all_deconvs,all_images):
                local_dir = backprop_feature_dir + os.sep + lid + '_' + str(d_i)
                if not os.path.exists(local_dir):
                    os.mkdir(local_dir)

                # saving deconv images
                for img_i in range(deconv_di.shape[0]):
                    if rotate_required:
                        local_img = (deconv_di[img_i,:,:,:]-np.min(deconv_di[img_i,:,:,:])).astype('uint16')
                        local_img = rotate(local_img,270)
                    else:
                        local_img = deconv_di[img_i,:,:,:]
                    imsave(local_dir + os.sep + 'deconv_' + lid +'_'+str(d_i)+'_'+str(img_i)+'.png', local_img)

                # saving original images
                for img_i in range(images_di.shape[0]):
                    if rotate_required:
                        images_di[img_i,:,:,:] = images_di[img_i,:,:,:] - np.min(images_di[img_i,:,:,:])
                        images_di[img_i,:,:,:] = images_di[img_i,:,:,:]*255.0/np.max(images_di[img_i,:,:,:])
                        local_img = images_di[img_i,:,:,:].astype('uint16')

                        local_img = rotate(local_img,270)
                    else:
                        local_img = images_di[img_i,:,:,:]
                    imsave(local_dir + os.sep + 'image_' + lid +'_'+str(d_i)+'_'+str(img_i)+'.png', local_img)

                d_i += 1
