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
import qlearner
from data_pool import Pool
from collections import Counter
from scipy.misc import imsave
import getopt
import time
import utils
import BasicCNNOpsFacade
import OptimizerOpsFacade

##################################################################
# AdaCNN Adapter
# ===============================================================
# AdaCNN Adapter will adapt the final model proposed by AdaCNN Constructor
# Then runs e-greedy Q-Learn for all the layer to decide of more filters required
# or should remove some. But keep in mind number of layers will be fixed
##################################################################

logger = None
logging_level = logging.INFO
logging_format = '[%(funcName)s] %(message)s'

batch_size = 128 # number of datapoints in a single batch
start_lr = 0.001
decay_learning_rate = False
dropout_rate = 0.25
use_dropout = False
#keep beta small (0.2 is too much >0.002 seems to be fine)
include_l2_loss = False
beta = 1e-5
check_early_stopping_from = 5
accuracy_drop_cap = 3
iterations_per_batch = 1
epochs = 3


def initialize_cnn_with_ops(cnn_ops,cnn_hyps):
    global tf_layer_activations
    weights,biases = {},{}

    logger.info('Initializing the iConvNet (conv_global,pool_global,classifier)...\n')
    for op in cnn_ops:
        if 'conv' in op:
            weights[op] = tf.Variable(tf.truncated_normal(
                cnn_hyps[op]['weights'],
                stddev=2./max(100,cnn_hyps[op]['weights'][0]*cnn_hyps[op]['weights'][1])
            ),validate_shape = False, expected_shape = cnn_hyps[op]['weights'],name=op+'_weights',dtype=tf.float32)
            biases[op] = tf.Variable(tf.constant(
                np.random.random()*0.001,shape=[cnn_hyps[op]['weights'][3]]
            ),validate_shape = False, expected_shape = [cnn_hyps[op]['weights'][3]],name=op+'_bias',dtype=tf.float32)

            tf_layer_activations[op] = tf.Variable(tf.zeros(shape=[cnn_hyps[op]['weights'][3]], dtype=tf.float32, name = op+'_activations'),validate_shape=False)

            logger.debug('Weights for %s initialized with size %s',op,str(cnn_hyps[op]['weights']))
            logger.debug('Biases for %s initialized with size %d',op,cnn_hyps[op]['weights'][3])

        if 'fulcon' in op:
            weights[op] = tf.Variable(tf.truncated_normal(
                [cnn_hyps[op]['in'],cnn_hyps[op]['out']],
                stddev=2./cnn_hyps[op]['in']
            ), validate_shape = False, expected_shape = [cnn_hyps[op]['in'],cnn_hyps[op]['out']], name=op+'_weights',dtype=tf.float32)
            biases[op] = tf.Variable(tf.constant(
                np.random.random()*0.001,shape=[cnn_hyps[op]['out']]
            ), validate_shape = False, expected_shape = [cnn_hyps[op]['out']], name=op+'_bias',dtype=tf.float32)

            logger.debug('Weights for %s initialized with size %d,%d',
                op,cnn_hyps[op]['in'],cnn_hyps[op]['out'])
            logger.debug('Biases for %s initialized with size %d',op,cnn_hyps[op]['out'])

    return weights,biases


def initialize_cnn_with_ops_fixed(cnn_ops,cnn_hyps):
    weights,biases = {},{}

    logger.info('Initializing the iConvNet (conv_global,pool_global,classifier)...\n')
    for op in cnn_ops:
        if 'conv' in op:
            weights[op] = tf.Variable(tf.truncated_normal(
                cnn_hyps[op]['weights'],
                stddev=2./max(100,cnn_hyps[op]['weights'][0]*cnn_hyps[op]['weights'][1])
            ),validate_shape = True,name=op+'_weights',dtype=tf.float32)
            biases[op] = tf.Variable(tf.constant(
                np.random.random()*0.001,shape=[cnn_hyps[op]['weights'][3]]
            ),validate_shape = True,name=op+'_bias',dtype=tf.float32)

            logger.info('Weights for %s initialized with size %s',op,str(cnn_hyps[op]['weights']))
            logger.info('Biases for %s initialized with size %d',op,cnn_hyps[op]['weights'][3])

        if 'fulcon' in op:
            weights[op] = tf.Variable(tf.truncated_normal(
                [cnn_hyps[op]['in'],cnn_hyps[op]['out']],
                stddev=2./cnn_hyps[op]['in']
            ), validate_shape = True, name=op+'_weights',dtype=tf.float32)
            biases[op] = tf.Variable(tf.constant(
                np.random.random()*0.001,shape=[cnn_hyps[op]['out']]
            ), validate_shape = True, name=op+'_bias',dtype=tf.float32)

            logger.info('Weights for %s initialized with size %d,%d',
                op,cnn_hyps[op]['in'],cnn_hyps[op]['out'])
            logger.info('Biases for %s initialized with size %d',op,cnn_hyps[op]['out'])

    return weights,biases


def get_logits_with_ops( dataset, use_dropout):
    global cnn_ops,tf_cnn_hyperparameters
    global tf_layer_activations
    global weights,biases

    first_fc = 'fulcon_out' if 'fulcon_0' not in weights else 'fulcon_0'

    logger.debug('Defining the logit calculation ...')
    logger.debug('\tCurrent set of operations: %s'%cnn_ops)
    activation_ops = []

    x = dataset
    logger.debug('\tReceived data for X(%s)...'%x.get_shape().as_list())

    #need to calculate the output according to the layers we have
    for op in cnn_ops:
        if 'conv' in op:
            logger.debug('\tConvolving (%s) With Weights:%s Stride:%s'%(op,cnn_hyperparameters[op]['weights'],cnn_hyperparameters[op]['stride']))
            #logger.debug('\t\tX before convolution:%s'%(x.get_shape().as_list()))
            logger.debug('\t\tWeights: %s',tf.shape(weights[op]).eval())
            x = tf.nn.conv2d(x, weights[op], cnn_hyperparameters[op]['stride'], padding=cnn_hyperparameters[op]['padding'])

            '''if use_dropout:
                x_2d_shape = tf.shape(x).eval()[:2]
                dropout_selected_ids = np.random.randint(0,cnn_hyperparameters[op]['weights'][3],(int((1-dropout_rate)*cnn_hyperparameters[op]['weights'][3]),)).tolist()
                dropout_mask = tf.scatter_nd([[idx] for idx in dropout_selected_ids],
                                             tf.ones(shape=[len(dropout_selected_ids), batch_size, x_2d_shape[0],
                                                            x_2d_shape[1]], dtype=tf.float32),
                                             shape=[cnn_hyperparameters[op]['weights'][3],batch_size,
                                                    x_2d_shape]
                                             )
                dropout_mask = tf.transpose(dropout_mask,[1,2,3,0])
                x = dropout_mask * x'''
            #logger.debug('\t\t Relu with x(%s) and b(%s)'%(tf.shape(x).eval(),tf.shape(biases[op]).eval()))
            x = tf.nn.relu(x + biases[op])
            #logger.debug('\t\tX after %s:%s'%tf.shape(weights[op]).eval())
            activation_ops.append(tf.assign(tf_layer_activations[op],tf.reduce_mean(x,[0,1,2]),validate_shape=False))

        if 'pool' in op:
            logger.debug('\tPooling (%s) with Kernel:%s Stride:%s'%(op,cnn_hyperparameters[op]['kernel'],cnn_hyperparameters[op]['stride']))
            if cnn_hyperparameters[op]['type'] is 'max':
                x = tf.nn.max_pool(x,ksize=cnn_hyperparameters[op]['kernel'],strides=cnn_hyperparameters[op]['stride'],padding=cnn_hyperparameters[op]['padding'])
            elif cnn_hyperparameters[op]['type'] is 'avg':
                x = tf.nn.avg_pool(x,ksize=cnn_hyperparameters[op]['kernel'],strides=cnn_hyperparameters[op]['stride'],padding=cnn_hyperparameters[op]['padding'])

            #logger.debug('\t\tX after %s:%s'%(op,tf.shape(x).eval()))

        if 'fulcon' in op:
            if first_fc==op:
                # we need to reshape the output of last subsampling layer to
                # convert 4D output to a 2D input to the hidden layer
                # e.g subsample layer output [batch_size,width,height,depth] -> [batch_size,width*height*depth]

                logger.debug('Input size of fulcon_out : %d', cnn_hyperparameters[op]['in'])
                x = tf.reshape(x, [batch_size, tf_cnn_hyperparameters[op]['in']])
                x = tf.nn.relu(tf.matmul(x, weights[op]) + biases[op])
            elif 'fulcon_out' == op:
                x = tf.matmul(x, weights['fulcon_out']) + biases['fulcon_out']
            else:
                x = tf.nn.relu(tf.matmul(x, weights[op]) + biases[op])

    return x,activation_ops


def calc_loss(logits,labels,weighted=False,tf_data_weights=None):
    # Training computation.
    if include_l2_loss:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels)) + \
               (beta/2)*tf.reduce_sum([tf.nn.l2_loss(w) if 'fulcon' in kw or 'conv' in kw else 0 for kw,w in weights.items()])
    else:
        # use weighted loss
        if weighted:
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels) * tf_data_weights)
        else:
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))

    return loss


def calc_loss_vector(logits,labels):
    return tf.nn.softmax_cross_entropy_with_logits(logits, labels)


def optimize_func(loss,global_step,learning_rate):
    global research_parameters
    global weights,biases

    optimize_ops = []
    # Optimizer.
    optimize_ops.append(tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss))

    return optimize_ops,learning_rate


def optimize_with_momenutm_func(loss, global_step, learning_rate):
    global tf_weight_vels,tf_bias_vels
    vel_update_ops, optimize_ops = [],[]

    if research_parameters['adapt_structure'] or research_parameters['use_custom_momentum_opt']:
        # custom momentum optimizing
        # apply_gradient([g,v]) does the following v -= eta*g
        # eta is learning_rate
        # Since what we need is
        # v(t+1) = mu*v(t) - eta*g
        # theta(t+1) = theta(t) + v(t+1) --- (2)
        # we form (2) in the form v(t+1) = mu*v(t) + eta*g
        # theta(t+1) = theta(t) - v(t+1)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)

        for op in tf_weight_vels.keys():
            [(grads_w,w),(grads_b,b)] = optimizer.compute_gradients(loss, [weights[op], biases[op]])

            # update velocity vector
            vel_update_ops.append(tf.assign(tf_weight_vels[op], research_parameters['momentum']*tf_weight_vels[op] + grads_w))
            vel_update_ops.append(tf.assign(tf_bias_vels[op], research_parameters['momentum']*tf_bias_vels[op] + grads_b))

            optimize_ops.append(optimizer.apply_gradients(
                        [(tf_weight_vels[op]*learning_rate,weights[op]),(tf_bias_vels[op]*learning_rate,biases[op])]
            ))
    else:
        optimize_ops.append(
            tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                       momentum=research_parameters['momentum']).minimize(loss))

    return optimize_ops,vel_update_ops,learning_rate

def optimize_with_momenutm_func_pool(loss, global_step, learning_rate):
    global tf_pool_w_vels,tf_pool_b_vels
    vel_update_ops, optimize_ops = [],[]

    if research_parameters['adapt_structure'] or research_parameters['use_custom_momentum_opt']:
        # custom momentum optimizing
        # apply_gradient([g,v]) does the following v -= eta*g
        # eta is learning_rate
        # Since what we need is
        # v(t+1) = mu*v(t) - eta*g
        # theta(t+1) = theta(t) + v(t+1) --- (2)
        # we form (2) in the form v(t+1) = mu*v(t) + eta*g
        # theta(t+1) = theta(t) - v(t+1)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)

        for op in tf_pool_w_vels.keys():
            [(grads_w,w),(grads_b,b)] = optimizer.compute_gradients(loss, [weights[op], biases[op]])

            # update velocity vector
            vel_update_ops.append(tf.assign(tf_pool_w_vels[op], research_parameters['momentum']*tf_pool_w_vels[op] + grads_w))
            vel_update_ops.append(tf.assign(tf_pool_b_vels[op], research_parameters['momentum']*tf_pool_b_vels[op] + grads_b))

            optimize_ops.append(optimizer.apply_gradients(
                        [(tf_pool_w_vels[op]*learning_rate,weights[op]),(tf_pool_b_vels[op]*learning_rate,biases[op])]
            ))
    else:
        optimize_ops.append(
            tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                       momentum=research_parameters['momentum']).minimize(loss))

    return optimize_ops,vel_update_ops,learning_rate


def optimize_with_tensor_slice_func(loss, filter_indices_to_replace, op, w, b):
    global weights,biases
    global cnn_hyperparameters,tf_cnn_hyperparameters

    learning_rate = tf.constant(start_lr,dtype=tf.float32,name='learning_rate')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    grads_wb = optimizer.compute_gradients(loss,[w,b])

    (grads_w,w),(grads_b,b) = grads_wb[0],grads_wb[1]

    curr_weight_shape = tf_cnn_hyperparameters[op]['weights'].eval()
    transposed_shape = [curr_weight_shape[3],curr_weight_shape[0],curr_weight_shape[1], curr_weight_shape[2]]

    replace_amnt = filter_indices_to_replace.size

    logger.debug('Applying gradients for %s',op)
    logger.debug('\tAnd filter IDs: %s',filter_indices_to_replace)

    mask_grads_w = tf.scatter_nd(
            filter_indices_to_replace.reshape(-1,1),
            tf.ones(shape=[replace_amnt,transposed_shape[1],transposed_shape[2],transposed_shape[3]], dtype=tf.float32),
            shape=transposed_shape
    )

    mask_grads_w = tf.transpose(mask_grads_w,[1,2,3,0])

    mask_grads_b = tf.scatter_nd(
            filter_indices_to_replace.reshape(-1,1),
            tf.ones_like(filter_indices_to_replace, dtype=tf.float32),
            shape=[curr_weight_shape[3]]
    )

    new_grads_w = grads_w * mask_grads_w
    new_grads_b = grads_b * mask_grads_b

    grad_apply_op = optimizer.apply_gradients([(new_grads_w,w),(new_grads_b,b)])

    return grad_apply_op


def optimize_with_min_indices_mom(loss, opt_ind_dict,indices_size, learning_rate):
    global weights, biases
    global tf_weight_vels, tf_bias_vels
    global cnn_ops, cnn_hyperparameters, tf_cnn_hyperparameters

    grad_ops,vel_update_ops = [],[]

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
    for op in cnn_ops:

        [(grads_w, w), (grads_b, b)] = optimizer.compute_gradients(loss, [weights[op], biases[op]])

        vel_update_ops.append(
            tf.assign(tf_weight_vels[op], research_parameters['momentum'] * tf_weight_vels[op] + grads_w))
        vel_update_ops.append(
            tf.assign(tf_bias_vels[op], research_parameters['momentum'] * tf_bias_vels[op] + grads_b))

        if 'conv' in op:

            transposed_shape = [tf_cnn_hyperparameters[op]['weights'][3], tf_cnn_hyperparameters[op]['weights'][0],
                                tf_cnn_hyperparameters[op]['weights'][1], tf_cnn_hyperparameters[op]['weights'][2]]

            logger.debug('Applying gradients for %s', op)

            mask_grads_w = tf.scatter_nd(
                tf.reshape(opt_ind_dict[op], [-1, 1]),
                tf.ones(shape=[indices_size, transposed_shape[1], transposed_shape[2], transposed_shape[3]],
                        dtype=tf.float32),
                shape=transposed_shape
            )

            mask_grads_w = tf.transpose(mask_grads_w, [1, 2, 3, 0])

            mask_grads_b = tf.scatter_nd(
                tf.reshape(opt_ind_dict[op], [-1, 1]),
                tf.ones_like(opt_ind_dict[op], dtype=tf.float32),
                shape=[tf_cnn_hyperparameters[op]['weights'][3]]
            )

            grads_w = tf_weight_vels[op] * learning_rate * mask_grads_w
            grads_b = tf_bias_vels[op] * learning_rate * mask_grads_b
            grad_ops.append(optimizer.apply_gradients([(grads_w, w), (grads_b, b)]))

        elif 'fulcon' in op:

            # use dropout (random) for this layer because we can't just train
            # min activations of the last layer (classification)
            logger.debug('Applying gradients for %s', op)

            mask_grads_w = tf.scatter_nd(
                tf.reshape(opt_ind_dict[op], [-1, 1]),
                tf.ones(shape=[indices_size, tf_cnn_hyperparameters[op]['out']],
                        dtype=tf.float32),
                shape=[tf_cnn_hyperparameters[op]['in'], tf_cnn_hyperparameters[op]['out']]
            )
            mask_grads_b = tf.scatter_nd(
                tf.reshape(opt_ind_dict[op], [-1,1]),
                tf.ones_like(opt_ind_dict[op],dtype=tf.float32),
                shape=[tf_cnn_hyperparameters[op]['out']]
            )

            grads_w = tf_weight_vels[op] * learning_rate * mask_grads_w
            grads_b = tf_bias_vels[op] * learning_rate * mask_grads_b

            grad_ops.append(
                optimizer.apply_gradients([(grads_w, weights[op]), (grads_b, biases[op])]))


def optimize_all_affected_with_indices(loss, filter_indices_to_replace, op, w, b, indices_size):
    '''
    Any adaptation of a convolutional layer would result in a change in the following layer.
    This optimization optimize the filters/weights responsible in both those layer
    :param loss:
    :param filter_indices_to_replace:
    :param op:
    :param w:
    :param b:
    :param cnn_hyps:
    :param cnn_ops:
    :return:
    '''
    global weights,biases,tf_pool_w_vels,tf_pool_b_vels
    global cnn_ops,cnn_hyperparameters,tf_cnn_hyperparameters

    vel_update_ops =[]
    grad_ops = []
    grads_w,grads_b= {},{}
    mask_grads_w,mask_grads_b = {},{}
    learning_rate = tf.constant(start_lr,dtype=tf.float32,name='learning_rate')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)

    replace_amnt = indices_size

    if 'conv' in op:
        [(grads_w[op], w), (grads_b[op], b)] = optimizer.compute_gradients(loss, [w, b])

        transposed_shape = [tf_cnn_hyperparameters[op]['weights'][3],tf_cnn_hyperparameters[op]['weights'][0],
                            tf_cnn_hyperparameters[op]['weights'][1], tf_cnn_hyperparameters[op]['weights'][2]]

        logger.debug('Applying gradients for %s',op)
        logger.debug('\tAnd filter IDs: %s',filter_indices_to_replace)

        mask_grads_w[op] = tf.scatter_nd(
                tf.reshape(filter_indices_to_replace,[-1,1]),
                tf.ones(shape=[replace_amnt,transposed_shape[1],transposed_shape[2],transposed_shape[3]], dtype=tf.float32),
                shape=transposed_shape
        )

        mask_grads_w[op] = tf.transpose(mask_grads_w[op],[1,2,3,0])

        mask_grads_b[op] = tf.scatter_nd(
                tf.reshape(filter_indices_to_replace, [-1, 1]),
                tf.ones_like(filter_indices_to_replace, dtype=tf.float32),
                shape=[tf_cnn_hyperparameters[op]['weights'][3]]
        )

        grads_w[op] = grads_w[op] * mask_grads_w[op]
        grads_b[op] = grads_b[op] * mask_grads_b[op]

        vel_update_ops.append(
            tf.assign(tf_pool_w_vels[op], research_parameters['momentum'] * tf_pool_w_vels[op] + grads_w[op]))
        vel_update_ops.append(
            tf.assign(tf_pool_b_vels[op], research_parameters['momentum'] * tf_pool_b_vels[op] + grads_b[op]))

        grad_ops.append(optimizer.apply_gradients([(tf_pool_w_vels[op]*learning_rate,w),(tf_pool_b_vels[op]*learning_rate,b)]))

    next_op = None
    for tmp_op in cnn_ops[cnn_ops.index(op)+1:]:
        if 'conv' in tmp_op or 'fulcon' in tmp_op:
            next_op = tmp_op
            break
    logger.debug('Next conv op: %s',next_op)
    [(grads_w[next_op], w)] = optimizer.compute_gradients(loss, [weights[next_op]])

    if 'conv' in next_op:

        transposed_shape = [tf_cnn_hyperparameters[next_op]['weights'][2], tf_cnn_hyperparameters[next_op]['weights'][0],
                            tf_cnn_hyperparameters[next_op]['weights'][1],tf_cnn_hyperparameters[next_op]['weights'][3]]

        logger.debug('Applying gradients for %s', next_op)
        logger.debug('\tAnd filter IDs: %s', filter_indices_to_replace)

        mask_grads_w[next_op] = tf.scatter_nd(
            tf.reshape(filter_indices_to_replace, [-1, 1]),
            tf.ones(shape=[replace_amnt, transposed_shape[1], transposed_shape[2], transposed_shape[3]],
                    dtype=tf.float32),
            shape=transposed_shape
        )

        mask_grads_w[next_op] = tf.transpose(mask_grads_w[next_op], [1, 2, 0, 3])
        grads_w[next_op] = grads_w[next_op] * mask_grads_w[next_op]

        vel_update_ops.append(
            tf.assign(tf_pool_w_vels[next_op], research_parameters['momentum'] * tf_pool_w_vels[next_op] + grads_w[next_op]))

        grad_ops.append(optimizer.apply_gradients([(tf_pool_w_vels[next_op]*learning_rate, weights[next_op])]))

    elif 'fulcon' in next_op:

        logger.debug('Applying gradients for %s', next_op)
        logger.debug('\tAnd filter IDs: %s', filter_indices_to_replace)

        mask_grads_w[next_op] = tf.scatter_nd(
            tf.reshape(filter_indices_to_replace, [-1, 1]),
            tf.ones(shape=[replace_amnt, tf_cnn_hyperparameters[next_op]['out']],
                    dtype=tf.float32),
            shape=[tf_cnn_hyperparameters[next_op]['in'],tf_cnn_hyperparameters[next_op]['out']]
        )

        grads_w[next_op] = grads_w[next_op] * mask_grads_w[next_op]

        vel_update_ops.append(
            tf.assign(tf_pool_w_vels[next_op],
                      research_parameters['momentum'] * tf_pool_w_vels[next_op] + grads_w[next_op]))

        grad_ops.append(optimizer.apply_gradients([(tf_pool_w_vels[next_op]*learning_rate, weights[next_op])]))

    return grad_ops,vel_update_ops


def inc_global_step(global_step):
    return global_step.assign(global_step+1)


def predict_with_logits(logits):
    # Predictions for the training, validation, and test data.
    prediction = tf.nn.softmax(logits)
    return prediction


def predict_with_dataset(dataset):
    logits,_ = get_logits_with_ops(dataset,False)
    prediction = tf.nn.softmax(logits)
    return prediction


def accuracy(predictions, labels):
    assert predictions.shape[0]==labels.shape[0]
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


def add_with_action(
        op, tf_action_info, tf_weights_this, tf_bias_this,
        tf_weights_next, tf_activations, tf_wvelocity_this,
        tf_bvelocity_this, tf_wvelocity_next
):
    global cnn_hyperparameters, tf_cnn_hyperparameters
    global logger,weights,biases,tf_weight_vels,tf_bias_vels,tf_pool_w_vels,tf_pool_b_vels
    global weight_velocity_vectors,bias_velocity_vectors

    first_fc = 'fulcon_out' if 'fulcon_0' not in weights else 'fulcon_0'
    update_ops = []

    # find the id of the last conv operation of the net
    for tmp_op in reversed(cnn_ops):
        if 'conv' in tmp_op:
            last_conv_id = tmp_op
            break

    logger.debug('Running action add for op %s',op)

    amount_to_add = tf_action_info[2] # amount of filters to add
    assert 'conv' in op

    # calculating new weights
    tf_new_weights = tf.concat(3,[weights[op],tf_weights_this])
    tf_new_biases = tf.concat(0,[biases[op],tf_bias_this])

    # updating velocity vectors
    if research_parameters['optimizer']=='Momentum':

        weight_vel = tf.concat(3,[tf_weight_vels[op], tf_wvelocity_this])
        bias_vel = tf.concat(0,[tf_bias_vels[op],tf_bvelocity_this])
        pool_w_vel = tf.concat(3,[tf_pool_w_vels[op],tf_wvelocity_this])
        pool_b_vel = tf.concat(0,[tf_pool_b_vels[op],tf_bvelocity_this])

        update_ops.append(tf.assign(tf_weight_vels[op],weight_vel,validate_shape=False))
        update_ops.append(tf.assign(tf_bias_vels[op],bias_vel,validate_shape=False))
        update_ops.append(tf.assign(tf_pool_w_vels[op],pool_w_vel,validate_shape=False))
        update_ops.append(tf.assign(tf_pool_b_vels[op], pool_b_vel, validate_shape=False))

    update_ops.append(tf.assign(weights[op], tf_new_weights, validate_shape=False))
    update_ops.append(tf.assign(biases[op], tf_new_biases, validate_shape = False))

    # ================ Changes to next_op ===============
    # Very last convolutional layer
    # this is different from other layers
    # as a change in this require changes to FC layer
    if op==last_conv_id:
        # change FC layer
        # the reshaping is required because our placeholder for weights_next is Rank 4
        tf_weights_next = tf.squeeze(tf_weights_next)
        tf_new_weights = tf.concat(0,[weights[first_fc],tf_weights_next])

        # updating velocity vectors
        if research_parameters['optimizer'] == 'Momentum':
            tf_wvelocity_next = tf.squeeze(tf_wvelocity_next)
            weight_vel = tf.concat(0,[tf_weight_vels[first_fc],tf_wvelocity_next])
            pool_w_vel = tf.concat(0, [tf_pool_w_vels[first_fc], tf_wvelocity_next])
            update_ops.append(tf.assign(tf_weight_vels[first_fc],weight_vel,validate_shape=False))
            update_ops.append(tf.assign(tf_pool_w_vels[first_fc],pool_w_vel,validate_shape=False))

        update_ops.append(tf.assign(weights[first_fc], tf_new_weights, validate_shape=False))

    else:

        # change in hyperparameter of next conv op
        next_conv_op = [tmp_op for tmp_op in cnn_ops[cnn_ops.index(op)+1:] if 'conv' in tmp_op][0]
        assert op!=next_conv_op

        # change only the weights in next conv_op
        tf_new_weights=tf.concat(2,[weights[next_conv_op],tf_weights_next])

        if research_parameters['optimizer']=='Momentum':
            weight_vel = tf.concat(2,[tf_weight_vels[next_conv_op], tf_wvelocity_next])
            pool_w_vel = tf.concat(2,[tf_pool_w_vels[next_conv_op],tf_wvelocity_next])
            update_ops.append(tf.assign(tf_weight_vels[next_conv_op], weight_vel, validate_shape=False))
            update_ops.append(tf.assign(tf_pool_w_vels[next_conv_op],pool_w_vel, validate_shape=False))

        update_ops.append(tf.assign(weights[next_conv_op], tf_new_weights, validate_shape=False))

    return update_ops


def get_rm_indices_with_distance(op,tf_action_info):
    global weights,biases

    amount_to_rmv = tf_action_info[2]
    reshaped_weight = tf.transpose(weights[op], [3, 0, 1, 2])
    reshaped_weight = tf.reshape(weights[op], [tf_cnn_hyperparameters[op]['weights'][3],
                                               tf_cnn_hyperparameters[op]['weights'][0] *
                                               tf_cnn_hyperparameters[op]['weights'][1] *
                                               tf_cnn_hyperparameters[op]['weights'][2]]
                                 )
    cos_sim_weights = tf.matmul(reshaped_weight, tf.transpose(reshaped_weight), name='dot_prod_cos_sim') / tf.matmul(
        tf.sqrt(tf.reduce_sum(reshaped_weight ** 2, axis=1, keep_dims=True)),
        tf.sqrt(tf.transpose(tf.reduce_sum(reshaped_weight ** 2, axis=1, keep_dims=True)))
        , name='norm_cos_sim')

    upper_triang_cos_sim = tf.matrix_band_part(cos_sim_weights, 0, -1, name='upper_triang_cos_sim')
    zero_diag_triang_cos_sim = tf.matrix_set_diag(upper_triang_cos_sim,
                                                  tf.zeros(shape=[tf_cnn_hyperparameters[op]['weights'][3]]),
                                                  name='zero_diag_upper_triangle')
    flattened_cos_sim = tf.reshape(zero_diag_triang_cos_sim, shape=[-1], name='flattend_cos_sim')

    # we are finding top amount_to_rmv + epsilon amount because
    # to avoid k_values = {...,(83,1)(139,94)(139,83),...} like incidents
    # above case will ignore both indices of (139,83) resulting in a reduction < amount_to_rmv
    [high_sim_values, high_sim_indices] = tf.nn.top_k(flattened_cos_sim,
                                                      k=tf.minimum(amount_to_rmv + 10,
                                                                   tf_cnn_hyperparameters[op]['weights'][3]),
                                                      name='top_k_indices')

    tf_indices_to_remove_1 = tf.reshape(tf.mod(high_sim_indices, tf_cnn_hyperparameters[op]['weights'][3]), shape=[-1],
                                        name='mod_indices')
    tf_indices_to_remove_2 = tf.reshape(tf.floor_div(high_sim_indices, tf_cnn_hyperparameters[op]['weights'][3]),
                                        shape=[-1], name='floor_div_indices')
    # concat both mod and floor_div indices
    tf_indices_to_rm = tf.reshape(tf.stack([tf_indices_to_remove_1, tf_indices_to_remove_2], name='all_rm_indices'),
                                  shape=[-1])
    # return both values and indices of unique values (discard indices)
    tf_unique_rm_ind, _ = tf.unique(tf_indices_to_rm, name='unique_rm_indices')

    return tf_unique_rm_ind


def remove_with_action(op, tf_action_info, tf_activations):
    global cnn_hyperparameters, tf_cnn_hyperparameters
    global logger,weights,biases,tf_weight_vels,tf_bias_vels
    global weight_velocity_vectors,bias_velocity_vectors

    first_fc = 'fulcon_out' if 'fulcon_0' not in weights else 'fulcon_0'
    update_ops = []

    for tmp_op in reversed(cnn_ops):
        if 'conv' in tmp_op:
            last_conv_id = tmp_op
            break

    # this is trickier than adding weights
    # We remove the given number of filters
    # which have the least rolling mean activation averaged over whole map
    amount_to_rmv = tf_action_info[2]
    assert 'conv' in op

    if research_parameters['remove_filters_by']=='Activation':
        neg_activations = -1.0 * tf_activations
        [min_act_values,tf_unique_rm_ind] = tf.nn.top_k(neg_activations,k=amount_to_rmv,name='min_act_indices')

    elif research_parameters['remove_filters_by']=='Distance':
        # calculate cosine distance for F filters (FxF matrix)
        # take one side of diagonal, find (f1,f2) pairs with least distance
        # select indices amnt f2 indices
        tf_unique_rm_ind = get_rm_indices_with_distance(op,tf_action_info)

    tf_indices_to_rm = tf.reshape(tf.slice(tf_unique_rm_ind,[0],[amount_to_rmv]),shape=[amount_to_rmv,1],name='indices_to_rm')
    tf_rm_ind_scatter = tf.scatter_nd(tf_indices_to_rm,tf.ones(shape=[amount_to_rmv],dtype=tf.int32),shape=[tf_cnn_hyperparameters[op]['weights'][3]])

    tf_indices_to_keep_boolean = tf.equal(tf_rm_ind_scatter,tf.constant(0,dtype=tf.int32))
    tf_indices_to_keep = tf.reshape(tf.where(tf_indices_to_keep_boolean),shape=[-1,1],name='indices_to_keep')

    # currently no way to generally slice using gather
    # need to do a transoformation to do this.
    # change both weights and biase in the current op

    tf_new_weights = tf.transpose(weights[op],[3,0,1,2])
    tf_new_weights = tf.gather_nd(tf_new_weights,tf_indices_to_keep)
    tf_new_weights = tf.transpose(tf_new_weights,[1,2,3,0],name='new_weights')

    update_ops.append(tf.assign(weights[op],tf_new_weights,validate_shape=False))

    tf_new_biases=tf.reshape(tf.gather(biases[op],tf_indices_to_keep),shape=[-1],name='new_bias')
    update_ops.append(tf.assign(biases[op],tf_new_biases,validate_shape=False))

    if research_parameters['optimizer'] == 'Momentum':
        weight_vel = tf.transpose(tf_weight_vels[op], [3, 0, 1, 2])
        weight_vel = tf.gather_nd(weight_vel, tf_indices_to_keep)
        weight_vel = tf.transpose(weight_vel, [1, 2, 3, 0])

        pool_w_vel = tf.transpose(tf_pool_w_vels[op], [3,0,1,2])
        pool_w_vel = tf.gather_nd(pool_w_vel, tf_indices_to_keep)
        pool_w_vel = tf.transpose(pool_w_vel, [1,2,3,0])

        bias_vel = tf.reshape(tf.gather(tf_bias_vels[op],tf_indices_to_keep),[-1])
        pool_b_vel = tf.reshape(tf.gather(tf_pool_b_vels[op],tf_indices_to_keep),[-1])

        update_ops.append(tf.assign(tf_weight_vels[op], weight_vel, validate_shape=False))
        update_ops.append(tf.assign(tf_pool_w_vels[op], pool_w_vel, validate_shape=False))
        update_ops.append(tf.assign(tf_bias_vels[op],bias_vel,validate_shape=False))
        update_ops.append(tf.assign(tf_pool_b_vels[op], pool_b_vel, validate_shape=False))

    if op==last_conv_id:

        tf_new_weights = tf.gather_nd(weights[first_fc],tf_indices_to_keep)
        update_ops.append(tf.assign(weights[first_fc],tf_new_weights,validate_shape=False))

        if research_parameters['optimizer']=='Momentum':
            weight_vel=tf.gather_nd(tf_weight_vels[first_fc],tf_indices_to_keep)
            pool_w_vel = tf.gather_nd(tf_pool_w_vels[first_fc],tf_indices_to_keep)
            update_ops.append(tf.assign(tf_weight_vels[first_fc],weight_vel,validate_shape=False))
            update_ops.append(tf.assign(tf_pool_w_vels[first_fc],pool_w_vel,validate_shape=False))

    else:
        # change in hyperparameter of next conv op
        next_conv_op = [tmp_op for tmp_op in cnn_ops[cnn_ops.index(op)+1:] if 'conv' in tmp_op][0]
        assert op!=next_conv_op

        # change only the weights in next conv_op

        tf_new_weights = tf.transpose(weights[next_conv_op],[2,0,1,3])
        tf_new_weights = tf.gather_nd(tf_new_weights,tf_indices_to_keep)
        tf_new_weights = tf.transpose(tf_new_weights,[1,2,0,3])

        update_ops.append(tf.assign(weights[next_conv_op],tf_new_weights,validate_shape=False))

        if research_parameters['optimizer']=='Momentum':
            weight_vel = tf.transpose(tf_weight_vels[next_conv_op],[2,0,1,3])
            weight_vel = tf.gather_nd(weight_vel,tf_indices_to_keep)
            weight_vel = tf.transpose(weight_vel,[1,2,0,3])

            pool_w_vel = tf.transpose(tf_pool_w_vels[next_conv_op], [2, 0, 1, 3])
            pool_w_vel = tf.gather_nd(pool_w_vel, tf_indices_to_keep)
            pool_w_vel = tf.transpose(pool_w_vel, [1, 2, 0, 3])

            update_ops.append(tf.assign(tf_weight_vels[next_conv_op],weight_vel,validate_shape=False))
            update_ops.append(tf.assign(tf_pool_w_vels[next_conv_op], pool_w_vel, validate_shape=False))

    return update_ops,tf_indices_to_rm


def load_data_from_memmap(dataset_info,dataset_filename,label_filename,start_idx,size):
    global logger
    num_labels = dataset_info['num_labels']
    col_count = (dataset_info['image_w'],dataset_info['image_w'],dataset_info['num_channels'])
    logger.info('Processing files %s,%s'%(dataset_filename,label_filename))
    fp1 = np.memmap(dataset_filename,dtype=np.float32,mode='r',
                    offset=np.dtype('float32').itemsize*col_count[0]*col_count[1]*col_count[2]*start_idx,shape=(size,col_count[0],col_count[1],col_count[2]))

    fp2 = np.memmap(label_filename,dtype=np.int32,mode='r',
                    offset=np.dtype('int32').itemsize*1*start_idx,shape=(size,1))

    train_dataset = fp1[:,:,:,:]
    # labels is nx1 shape
    train_labels = fp2[:]

    train_ohe_labels = (np.arange(num_labels) == train_labels[:]).astype(np.float32)
    del fp1,fp2

    assert np.all(np.argmax(train_ohe_labels[:10],axis=1).flatten()==train_labels[:10].flatten())
    return train_dataset,train_ohe_labels


def init_velocity_vectors(cnn_ops,cnn_hyps):
    global weight_velocity_vectors,bias_velocity_vectors

    for op in cnn_ops:
        if 'conv' in op:
            weight_velocity_vectors[op] = tf.zeros(shape=cnn_hyps[op]['weights'],dtype=tf.float32)
            bias_velocity_vectors[op] = tf.zeros(shape=[cnn_hyps[op]['weights'][3]],dtype=tf.float32)
        elif 'fulcon' in op:
            weight_velocity_vectors[op] = tf.zeros(shape=[cnn_hyps[op]['in'],cnn_hyps[op]['out']],dtype=tf.float32)
            bias_velocity_vectors[op] = tf.zeros(shape=[cnn_hyps[op]['out']],dtype=tf.float32)


def init_tf_hyperparameters():
    global cnn_ops, cnn_hyperparameters,tf_cnn_hyperparameters

    for op in cnn_ops:
        if 'conv' in op:
            tf_cnn_hyperparameters[op] = {'weights':tf.Variable(cnn_hyperparameters[op]['weights'],dtype=tf.int32,name='hyp_'+op+'_weights')}
        if 'fulcon' in op:
            tf_cnn_hyperparameters[op] = {
                'in': tf.Variable(cnn_hyperparameters[op]['in'], dtype=tf.int32, name='hyp_' + op + '_in'),
                'out': tf.Variable(cnn_hyperparameters[op]['out'], dtype=tf.int32, name='hyp_' + op + '_out')
            }


def update_tf_hyperparameters(op,tf_weight_shape,tf_in_size):
    global cnn_ops, cnn_hyperparameters, tf_cnn_hyperparameters
    update_ops = []
    if 'conv' in op:
        update_ops.append(tf.assign(tf_cnn_hyperparameters[op]['weights'],tf_weight_shape))
    if 'fulcon' in op:
        update_ops.append(tf.assign(tf_cnn_hyperparameters[op]['in'],tf_in_size))

    return update_ops


tf_weight_vels, tf_bias_vels = {}, {}
tf_pool_w_vels, tf_pool_b_vels = {},{}
weights,biases = None,None
tf_layer_activations = {}

research_parameters = {
    'save_train_test_images':False,
    'log_class_distribution':True,'log_distribution_every':128,
    'adapt_structure' : True,
    'hard_pool_acceptance_rate':0.1, 'accuracy_threshold_hard_pool':50,
    'replace_op_train_rate':0.5, # amount of batches from hard_pool selected to train
    'optimizer':'Momentum','momentum':0.9,
    'use_custom_momentum_opt':True,
    'remove_filters_by':'Activation',
    'optimize_end_to_end':True, # if true functions such as add and finetune will optimize the network from starting layer to end (fulcon_out)
    'loss_diff_threshold':0.02,
    'debugging':True if logging_level==logging.DEBUG else False,
    'stop_training_at':11000,
    'train_min_activation':False
}

interval_parameters = {
    'history_dump_interval':500,
    'policy_interval' : 50, #number of batches to process for each policy iteration
    'test_interval' : 100
}

state_action_history = {}

cnn_ops, cnn_hyperparameters, tf_cnn_hyperparameters = None,None,{}

if __name__=='__main__':
    global weights,biases,tf_weight_vels,tf_bias_vels,tf_pool_w_vels,tf_pool_b_vels
    global weight_velocity_vectors,bias_velocity_vectors
    global cnn_ops,cnn_hyperparameters

    try:
        opts,args = getopt.getopt(
            sys.argv[1:],"",["output_dir="])
    except getopt.GetoptError as err:
        print('<filename>.py --output_dir=')

    if len(opts)!=0:
        for opt,arg in opts:
            if opt == '--output_dir':
                output_dir = arg

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #type of data training
    datatype = 'cifar-10'
    behavior = 'non-stationary'

    dataset_info = {'dataset_type':datatype,'behavior':behavior}
    dataset_filename,label_filename = None,None
    test_dataset,test_labels = None,None
    if datatype=='cifar-10':
        image_size = 32
        num_labels = 10
        num_channels = 3 # rgb
        dataset_size = 50000

        if behavior == 'non-stationary':
            dataset_filename='data_non_station'+os.sep+'cifar-10-nonstation-dataset.pkl'
            label_filename='data_non_station'+os.sep+'cifar-10-nonstation-labels.pkl'
            dataset_size = 1280000
            chunk_size = 25600
        elif behavior == 'stationary':
            dataset_filename='data_non_station'+os.sep+'cifar-10-station-dataset.pkl'
            label_filename='data_non_station'+os.sep+'cifar-10-station-labels.pkl'
            dataset_size = 1280000
            chunk_size = 25600

        test_size=10000
        test_dataset_filename='data_non_station'+os.sep+'cifar-10-nonstation-test-dataset.pkl'
        test_label_filename = 'data_non_station'+os.sep+'cifar-10-nonstation-test-labels.pkl'

    elif datatype=='imagenet-100':
        image_size = 224
        num_labels = 100
        num_channels = 3
        dataset_size = 128000
        if behavior == 'non-stationary':
            dataset_filename='..'+os.sep+'imagenet_small'+os.sep+'imagenet-100-non-station-dataset.pkl'
            label_filename='..'+os.sep+'imagenet_small'+os.sep+'imagenet-100-non-station-label.pkl'
            image_size = 128
            dataset_size = 1280000
            chunk_size = 12800

        test_size=5000
        test_dataset_filename='..'+os.sep+'imagenet_small'+os.sep+'imagenet-100-non-station-test-dataset.pkl'
        test_label_filename = '..'+os.sep+'imagenet_small'+os.sep+'imagenet-100-non-station-test-label.pkl'

    elif datatype=='svhn-10':
        image_size = 32
        num_labels = 10
        num_channels = 3
        dataset_size = 128000
        if behavior == 'non-stationary':
            dataset_filename='data_non_station'+os.sep+'svhn-10-nonstation-dataset.pkl'
            label_filename='data_non_station'+os.sep+'svhn-10-nonstation-labels.pkl'
            dataset_size = 1280000
            chunk_size = 25600
        elif behavior == 'stationary':
            dataset_filename='data_non_station'+os.sep+'svhn-10-station-dataset.pkl'
            label_filename='data_non_station'+os.sep+'svhn-10-station-labels.pkl'
            dataset_size = 1280000
            chunk_size = 25600

        test_size = 26032
        test_dataset_filename = 'data_non_station' + os.sep + 'svhn-10-nonstation-test-dataset.pkl'
        test_label_filename = 'data_non_station' + os.sep + 'svhn-10-nonstation-test-labels.pkl'

    dataset_info['image_w']=image_size
    dataset_info['num_labels']=num_labels
    dataset_info['num_channels']=num_channels
    dataset_info['dataset_size']=dataset_size
    dataset_info['chunk_size']=chunk_size
    dataset_info['num_labels']=num_labels

    logger = logging.getLogger('main_logger')
    logger.setLevel(logging_level)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(logging_format))
    console.setLevel(logging_level)
    logger.addHandler(console)

    error_logger = logging.getLogger('error_logger')
    error_logger.setLevel(logging.INFO)
    errHandler = logging.FileHandler(output_dir + os.sep + 'Error.log', mode='w')
    errHandler.setFormatter(logging.Formatter('%(message)s'))
    error_logger.addHandler(errHandler)
    error_logger.info('#Batch_ID,Loss(Train),Loss(Valid Seen),Loss(Valid Unseen),Valid(Seen),Valid(Unseen),Test')

    perf_logger = logging.getLogger('time_logger')
    perf_logger.setLevel(logging.INFO)
    perfHandler = logging.FileHandler(output_dir + os.sep + 'time.log', mode='w')
    perfHandler.setFormatter(logging.Formatter('%(message)s'))
    perf_logger.addHandler(perfHandler)
    perf_logger.info('#Batch_ID,Time(Full),Time(Train),Op count, Var count')

    if research_parameters['adapt_structure']:
        cnn_structure_logger = logging.getLogger('cnn_structure_logger')
        cnn_structure_logger.setLevel(logging.INFO)
        structHandler = logging.FileHandler(output_dir + os.sep + 'cnn_structure.log', mode='w')
        structHandler.setFormatter(logging.Formatter('%(message)s'))
        cnn_structure_logger.addHandler(structHandler)
        cnn_structure_logger.info('#batch_id:state:action:reward:#layer_1_hyperparameters#layer_2_hyperparameters#...')

        q_logger = logging.getLogger('q_logger')
        q_logger.setLevel(logging.INFO)
        qHandler = logging.FileHandler(output_dir + os.sep + 'QMetric.log', mode='w')
        qHandler.setFormatter(logging.Formatter('%(message)s'))
        q_logger.addHandler(qHandler)
        q_logger.info('#batch_id,pred_q')


    if research_parameters['log_class_distribution']:
        class_dist_logger = logging.getLogger('class_dist_logger')
        class_dist_logger.setLevel(logging.INFO)
        class_distHandler = logging.FileHandler(output_dir + os.sep + 'class_distribution.log', mode='w')
        class_distHandler.setFormatter(logging.Formatter('%(message)s'))
        class_dist_logger.addHandler(class_distHandler)

    pool_dist_logger = logging.getLogger('pool_distribution_logger')
    pool_dist_logger.setLevel(logging.INFO)
    poolHandler = logging.FileHandler(output_dir + os.sep + 'pool_distribution.log', mode='w')
    poolHandler.setFormatter(logging.Formatter('%(message)s'))
    pool_dist_logger.addHandler(poolHandler)
    pool_dist_logger.info('#Class distribution')

    # Loading test data
    train_dataset,train_labels = None,None

    test_dataset,test_labels = load_data_from_memmap(dataset_info,test_dataset_filename,test_label_filename,0,test_size)

    assert chunk_size%batch_size==0
    batches_in_chunk = chunk_size//batch_size

    logger.info('='*80)
    logger.info('\tTrain Size: %d'%dataset_size)
    logger.info('='*80)

    #cnn_string = "C,5,1,256#P,3,2,0#C,5,1,512#C,3,1,128#FC,2048,0,0#Terminate,0,0,0"
    if not research_parameters['adapt_structure']:
        cnn_string = "C,5,1,256#P,3,2,0#C,5,1,256#P,3,2,0#C,3,1,512#Terminate,0,0,0"
    else:
        cnn_string = "C,5,1,64#P,3,2,0#C,5,1,64#P,3,2,0#C,3,1,64#Terminate,0,0,0"
    #cnn_string = "C,3,1,128#P,5,2,0#C,5,1,128#C,3,1,512#C,5,1,128#C,5,1,256#P,2,2,0#C,5,1,64#Terminate,0,0,0"
    #cnn_string = "C,3,4,128#P,5,2,0#Terminate,0,0,0"

    cnn_ops,cnn_hyperparameters = utils.get_ops_hyps_from_string(dataset_info,cnn_string)

    hyp_logger = logging.getLogger('hyperparameter_logger')
    hyp_logger.setLevel(logging.INFO)
    hypHandler = logging.FileHandler(output_dir + os.sep + 'Hyperparameter.log', mode='w')
    hypHandler.setFormatter(logging.Formatter('%(message)s'))
    hyp_logger.addHandler(hypHandler)
    hyp_logger.info('#Starting hyperparameters')
    hyp_logger.info(cnn_hyperparameters)
    hyp_logger.info('#Dataset info')
    hyp_logger.info(dataset_info)
    hyp_logger.info('Learning rate: %.5f', start_lr)
    hyp_logger.info('Batch size: %d', batch_size)
    hyp_logger.info('#Research parameters')
    hyp_logger.info(research_parameters)
    hyp_logger.info('#Interval parameters')
    hyp_logger.info(interval_parameters)

    # Resetting this every policy interval
    rolling_data_distribution = {}
    mean_data_distribution = {}
    prev_mean_distribution = {}
    for li in range(num_labels):
        rolling_data_distribution[li]=0.0
        mean_data_distribution[li]=1.0/float(num_labels)
        prev_mean_distribution[li]=1.0/float(num_labels)

    graph = tf.Graph()
    config = tf.ConfigProto()
    config.allow_soft_placement=True
    config.log_device_placement=False
    #config.gpu_options.per_process_gpu_memory_fraction=0.65
    with tf.Session(graph=graph,config=config) as session, tf.device('/gpu:0'):

        hardness = 0.5
        hard_pool = Pool(size=12800,batch_size=batch_size,image_size=image_size,num_channels=num_channels,num_labels=num_labels,assert_test=False)

        init_tf_hyperparameters()

        if research_parameters['adapt_structure'] or research_parameters['use_custom_momentum_opt']:
            weights,biases = initialize_cnn_with_ops(cnn_ops,cnn_hyperparameters)
        else:
            weights, biases = initialize_cnn_with_ops_fixed(cnn_ops, cnn_hyperparameters)

        first_fc = 'fulcon_out' if 'fulcon_0' not in weights else 'fulcon_0'
        # -1 is because we don't want to count pool_global
        layer_count = len([op for op in cnn_ops if 'conv' in op or 'pool' in op])-1

        # ids of the convolution ops
        convolution_op_ids = []
        for op_i, op in enumerate(cnn_ops):
            if 'conv' in op:
                convolution_op_ids.append(op_i)

        # Adapting Policy Learner
        adapter = qlearner.AdaCNNAdaptingQLearner(
            discount_rate=0.5, fit_interval = 1,
            exploratory_tries = 10, exploratory_interval = 100,
            filter_upper_bound=512, filter_min_bound=128,
            conv_ids=convolution_op_ids, net_depth=layer_count,
            n_conv = len([op for op in cnn_ops if 'conv' in op]),
            epsilon=1.0, target_update_rate=25,
            batch_size=32, persist_dir = output_dir,
            session = session, random_mode = False
        )

        global_step = tf.Variable(0, dtype=tf.int32,trainable=False,name='global_step')

        logger.info('Input data defined...\n')
        # Input train data
        tf_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels),name='TrainDataset')
        tf_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels),name='TrainLabels')
        tf_data_weights = tf.placeholder(tf.float32, shape=(batch_size),name='TrainLabels')

        # Pool data
        tf_pool_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels),name='PoolDataset')
        tf_pool_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels),name='PoolLabels')

        # Valid data (Next train batch) Unseen
        tf_valid_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels),name='ValidDataset')
        tf_valid_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels),name='ValidLabels')

        # Test data (Global)
        tf_test_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels),name='TestDataset')
        tf_test_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels),name='TestLabels')

        # if using momentum
        if research_parameters['optimizer']=='Momentum':
            for tmp_op in cnn_ops:
                if 'conv' in tmp_op:
                    tf_weight_vels[tmp_op] = tf.Variable(
                        tf.zeros(shape=cnn_hyperparameters[tmp_op]['weights'], dtype=tf.float32),
                        name='w_vel_'+tmp_op)
                    tf_pool_w_vels[tmp_op] = tf.Variable(
                        tf.zeros(shape=cnn_hyperparameters[tmp_op]['weights'], dtype=tf.float32),
                        name='poolw_vel'+tmp_op)

                    tf_bias_vels[tmp_op] = tf.Variable(
                        tf.zeros(shape=[cnn_hyperparameters[tmp_op]['weights'][3]], dtype=tf.float32),
                        name='b_vel_'+tmp_op)
                    tf_pool_b_vels[tmp_op] = tf.Variable(
                        tf.zeros(shape=[cnn_hyperparameters[tmp_op]['weights'][3]], dtype=tf.float32),
                        name='poolb_vel_' + tmp_op)
                elif 'fulcon' in tmp_op:
                    tf_weight_vels[tmp_op] = tf.Variable(tf.zeros(shape=[cnn_hyperparameters[tmp_op]['in'], cnn_hyperparameters[tmp_op]['out']], dtype=tf.float32),
                                                           name='w_vel_'+tmp_op)
                    tf_pool_w_vels[tmp_op] = tf.Variable(
                        tf.zeros(shape=[cnn_hyperparameters[tmp_op]['in'], cnn_hyperparameters[tmp_op]['out']],
                                 dtype=tf.float32),
                        name='poolw_vel_' + tmp_op)

                    tf_bias_vels[tmp_op] = tf.Variable(
                        tf.zeros(shape=[cnn_hyperparameters[tmp_op]['out']], dtype=tf.float32),
                        name='b_vel_'+tmp_op)
                    tf_pool_b_vels[tmp_op] = tf.Variable(
                        tf.zeros(shape=[cnn_hyperparameters[tmp_op]['out']], dtype=tf.float32), name='poolb_vel_' + tmp_op)

            #init_velocity_vectors(cnn_ops,cnn_hyperparameters)

        init_op = tf.global_variables_initializer()
        _ = session.run(init_op)

        # Tensorflow operations related to training data
        logits,tf_activation_ops = get_logits_with_ops(tf_dataset,use_dropout)
        loss = calc_loss(logits,tf_labels,True,tf_data_weights)
        loss_vec = calc_loss_vector(logits,tf_labels)
        pred = predict_with_logits(logits)

        if not research_parameters['train_min_activation']:
            if research_parameters['optimizer']=='SGD':
                optimize,upd_lr = optimize_func(loss,global_step,tf.constant(start_lr,dtype=tf.float32))
            elif research_parameters['optimizer']=='Momentum':
                optimize, velocity_update, upd_lr = optimize_with_momenutm_func(loss, global_step, tf.constant(start_lr,dtype=tf.float32))

        inc_gstep = inc_global_step(global_step)

        init_op = tf.global_variables_initializer()
        _ = session.run(init_op)

        # Tensorflow operations for validation data
        tf_valid_logits, _ = get_logits_with_ops(tf_valid_dataset,False)
        tf_valid_predictions = predict_with_logits(tf_valid_logits)
        tf_valid_loss = calc_loss(tf_valid_logits,tf_valid_labels,False,None)

        # Tensorflow operations for test data
        pred_test = predict_with_dataset(tf_test_dataset)

        # Tensorflow operations for hard_pool
        pool_logits, _ = get_logits_with_ops(tf_pool_dataset,use_dropout)
        pool_loss = calc_loss(pool_logits, tf_pool_labels,False,None)
        pool_pred = predict_with_dataset(tf_pool_dataset)

        if research_parameters['adapt_structure']:
            # Tensorflow operations that are defined one for each convolution operation
            tf_indices = tf.placeholder(dtype=tf.int32,shape=(None,),name='optimize_indices')
            tf_indices_size = tf.placeholder(tf.int32)
            tf_slice_optimize = {}
            tf_slice_vel_update = {}
            tf_add_filters_ops,tf_rm_filters_ops,tf_replace_ind_ops = {},{},{}

            tf_action_info = tf.placeholder(shape=[3], dtype=tf.int32,
                                            name='tf_action')  # [op_id,action_id,amount] (action_id 0 - add, 1 -remove)
            tf_running_activations = tf.placeholder(shape=(None,), dtype=tf.float32, name='running_activations')

            tf_weights_this = tf.placeholder(shape=[None,None,None,None],dtype=tf.float32,name='new_weights_current')
            tf_bias_this = tf.placeholder(shape=(None,), dtype=tf.float32, name='new_bias_current')
            tf_weights_next = tf.placeholder(shape=[None,None,None,None],dtype=tf.float32,name='new_weights_next')

            #tf_wvelocity_this = tf.zeros([cnn_hyps[op]['weights'][0],
            #          cnn_hyps[op]['weights'][1], cnn_hyps[op]['weights'][2], amount_to_add], dtype=tf.float32)
            # tf_bvelocity_this = tf.zeros([amount_to_add],dtype=tf.float32)
            # tf_wvelocity_next = tf.zeros([amount_to_add,num_labels])
            # tf_wvelocity_next =tf.zeros([tf_cnn_hyperparameters[next_conv_op]['weights'][0],tf_cnn_hyperparameters[next_conv_op]['weights'][1],
            #              amount_to_add,tf_cnn_hyperparameters[next_conv_op]['weights'][3]])

            tf_wvelocity_this = tf.placeholder(shape=[None,None,None,None], dtype=tf.float32)
            tf_bvelocity_this = tf.placeholder(shape=(None,),dtype=tf.float32)
            tf_wvelocity_next = tf.placeholder(shape=[None,None,None,None],dtype=tf.float32)

            tf_weight_shape = tf.placeholder(shape=[4],dtype=tf.int32,name='weight_shape')
            tf_in_size = tf.placeholder(dtype=tf.int32)
            tf_update_hyp_ops={}

            if research_parameters['train_min_activation']:
                if research_parameters['optimizer']=='SGD':
                    raise NotImplementedError
                elif research_parameters['optimizer']=='Momentum':
                    optimize, velocity_update, upd_lr = optimize_with_momenutm_func(loss, global_step, tf.constant(start_lr,dtype=tf.float32))

            for tmp_op in cnn_ops:
                if 'conv' in tmp_op:
                    tf_update_hyp_ops[tmp_op] = update_tf_hyperparameters(tmp_op,tf_weight_shape,tf_in_size)
                    tf_add_filters_ops[tmp_op] = add_with_action(tmp_op,tf_action_info,tf_weights_this,
                                                                 tf_bias_this,tf_weights_next,tf_running_activations,
                                                                 tf_wvelocity_this,tf_bvelocity_this,tf_wvelocity_next)
                    tf_rm_filters_ops[tmp_op] = remove_with_action(tmp_op,tf_action_info,tf_running_activations)
                    tf_replace_ind_ops[tmp_op] = get_rm_indices_with_distance(tmp_op,tf_action_info)
                    tf_slice_optimize[tmp_op],tf_slice_vel_update[tmp_op] = optimize_all_affected_with_indices(
                        pool_loss, tf_indices,
                        tmp_op, weights[tmp_op], biases[tmp_op], tf_indices_size
                    )
                elif 'fulcon' in tmp_op:
                    tf_update_hyp_ops[tmp_op] = update_tf_hyperparameters(tmp_op, tf_weight_shape, tf_in_size)

            if research_parameters['optimize_end_to_end']:
                # Lower learning rates for pool op doesnt help
                # Momentum or SGD for pooling? Go with SGD
                #optimize_with_pool, _ = optimize_func(pool_loss, global_step, tf.constant(start_lr,dtype=tf.float32))
                optimize_with_pool, pool_vel_updates, _ = optimize_with_momenutm_func_pool(pool_loss, global_step, tf.constant(start_lr, dtype=tf.float32))
            else:
                raise NotImplementedError

        logger.info('Tensorflow functions defined')
        logger.info('Variables initialized...')

        train_losses = []
        mean_train_loss = 0

        rolling_ativation_means = {}
        for op in cnn_ops:
            if 'conv' in op:
                logger.debug('\tDefining rolling activation mean for %s',op)
                rolling_ativation_means[op]=np.zeros([cnn_hyperparameters[op]['weights'][3]])

        decay = 0.9
        act_decay = 0.5
        current_state,current_action = None,None
        prev_valid_accuracy,next_valid_accuracy = 0,0

        current_q_learn_op_id = 0
        logger.info('Convolutional Op IDs: %s',convolution_op_ids)

        logger.info('Starting Training Phase')
        #TODO: Think about decaying learning rate (should or shouldn't)

        logger.info('Dataset Size: %d',dataset_size)
        logger.info('Batch Size: %d',batch_size)
        logger.info('Chunk Size: %d',chunk_size)
        logger.info('Batches in Chunk : %d',batches_in_chunk)

        previous_loss = 1e5 # used for the check to start adapting
        prev_pool_accuracy = 0
        prev_state,prev_action,prev_invalid_actions = None,None,None
        start_adapting = False

        batch_id_multiplier = (dataset_size//batch_size) - interval_parameters['test_interval']

        for epoch in range(epochs):
            memmap_idx = 0

            for batch_id in range(ceil(dataset_size//batch_size)-1):

                if batch_id+1>=research_parameters['stop_training_at']:
                    break

                t0 = time.clock() # starting time for a batch

                logger.debug('='*80)
                logger.debug('tf op count: %d',len(graph.get_operations()))
                logger.debug('=' * 80)

                logger.debug('\tTraining with batch %d',batch_id)
                for op in cnn_ops:
                    if 'pool' not in op:
                        assert weights[op].name in [v.name for v in tf.trainable_variables()]

                chunk_batch_id = batch_id%batches_in_chunk

                if chunk_batch_id==0:
                    # We load 1 extra batch (chunk_size+1) because we always make the valid batch the batch_id+1
                    logger.info('\tCurrent memmap start index: %d', memmap_idx)
                    if memmap_idx+chunk_size+batch_size<dataset_size:
                        train_dataset,train_labels = load_data_from_memmap(dataset_info,dataset_filename,label_filename,memmap_idx,chunk_size+batch_size)
                    else:
                        train_dataset, train_labels = load_data_from_memmap(dataset_info, dataset_filename, label_filename,
                                                                            memmap_idx, chunk_size)
                    memmap_idx += chunk_size
                    logger.info('Loading dataset chunk of size (chunk size + batch size): %d',train_dataset.shape[0])
                    logger.info('\tNext memmap start index: %d',memmap_idx)

                batch_data = train_dataset[chunk_batch_id*batch_size:(chunk_batch_id+1)*batch_size, :, :, :]
                batch_labels = train_labels[chunk_batch_id*batch_size:(chunk_batch_id+1)*batch_size, :]

                cnt = Counter(np.argmax(batch_labels, axis=1))
                if behavior=='non-stationary':
                    batch_weights = np.zeros((batch_size,))
                    batch_labels_int = np.argmax(batch_labels, axis=1)

                    for li in range(num_labels):
                        batch_weights[np.where(batch_labels_int==li)[0]] = 1.0 - (cnt[li]/batch_size)
                elif behavior=='stationary':
                    batch_weights = np.ones((batch_size,))
                else:
                    raise NotImplementedError

                feed_dict = {tf_dataset : batch_data, tf_labels : batch_labels, tf_data_weights:batch_weights}

                t0_train = time.clock()
                for _ in range(iterations_per_batch):
                    if research_parameters['optimizer']=='Momentum' and research_parameters['use_custom_momentum_opt']:
                        _, _, l, l_vec, _, _, updated_lr, predictions, current_activations = session.run(
                            [logits, tf_activation_ops, loss, loss_vec, optimize, velocity_update,upd_lr, pred, tf_layer_activations], feed_dict=feed_dict
                        )
                    else:
                        _, _, l,l_vec, _,updated_lr, predictions, current_activations = session.run(
                                [logits, tf_activation_ops,loss,loss_vec,optimize,upd_lr,pred,tf_layer_activations], feed_dict=feed_dict
                        )

                t1_train = time.clock()

                # this snippet logs the normalized class distribution every specified interval
                if chunk_batch_id%research_parameters['log_distribution_every']==0:
                    dist_cnt = Counter(np.argmax(batch_labels, axis=1))
                    norm_dist = ''
                    for li in range(num_labels):
                        norm_dist += '%.5f,'%(dist_cnt[li]*1.0/batch_size)
                    class_dist_logger.info(norm_dist)

                if chunk_batch_id % 50<10:
                    logger.debug('=' * 80)
                    logger.debug('Actual Train Labels')
                    logger.debug(np.argmax(batch_labels, axis=1).flatten()[:5])
                    logger.debug('=' * 80)
                    logger.debug('Predicted Train Labels')
                    logger.debug(np.argmax(predictions, axis=1).flatten()[:5])
                    logger.debug('=' * 80)
                    logger.debug('Train: %d, %.3f', chunk_batch_id, accuracy(predictions, batch_labels))
                    logger.debug('')

                if np.isnan(l):
                    logger.critical('NaN detected: (batchID) %d (last Cost) %.3f',chunk_batch_id,train_losses[-1])
                if np.random.random()<research_parameters['hard_pool_acceptance_rate']:
                    hard_pool.add_hard_examples(batch_data,batch_labels,l_vec,1.0-(accuracy(predictions, batch_labels)/100.0))

                assert not np.isnan(l)

                # rolling activation mean update
                for op,op_activations in current_activations.items():
                    logger.debug('checking %s',op)
                    logger.debug('\tRolling size (%s): %s',op,rolling_ativation_means[op].shape)
                    logger.debug('\tCurrent size (%s): %s', op, op_activations.shape)
                for op, op_activations in current_activations.items():
                    assert current_activations[op].size == cnn_hyperparameters[op]['weights'][3]
                    rolling_ativation_means[op]=(1-act_decay)*rolling_ativation_means[op] + decay*current_activations[op]

                train_losses.append(l)

                # validation batch (Unseen)
                batch_valid_data = train_dataset[(chunk_batch_id+1)*batch_size:(chunk_batch_id+2)*batch_size, :, :, :]
                batch_valid_labels = train_labels[(chunk_batch_id+1)*batch_size:(chunk_batch_id+2)*batch_size, :]

                feed_valid_dict = {tf_valid_dataset:batch_valid_data, tf_valid_labels:batch_valid_labels}
                next_valid_loss, next_valid_predictions = session.run([tf_valid_loss,tf_valid_predictions],feed_dict=feed_valid_dict)
                next_valid_accuracy = accuracy(next_valid_predictions, batch_valid_labels)

                # validation batch (Seen) random from last 50 batches
                if chunk_batch_id>0:
                    prev_valid_batch_id = np.random.randint(max(0,chunk_batch_id-50),chunk_batch_id)
                else:
                    prev_valid_batch_id = 0

                batch_valid_data = train_dataset[prev_valid_batch_id*batch_size:(prev_valid_batch_id+1)*batch_size, :, :, :]
                batch_valid_labels = train_labels[prev_valid_batch_id*batch_size:(prev_valid_batch_id+1)*batch_size, :]

                feed_valid_dict = {tf_valid_dataset:batch_valid_data, tf_valid_labels:batch_valid_labels}
                prev_valid_loss, prev_valid_predictions = session.run([tf_valid_loss,tf_valid_predictions],feed_dict=feed_valid_dict)
                prev_valid_accuracy = accuracy(prev_valid_predictions, batch_valid_labels)

                if (batch_id+1)%interval_parameters['test_interval']==0:
                    mean_train_loss = np.mean(train_losses)
                    logger.info('='*60)
                    logger.info('\tBatch ID: %d'%batch_id)
                    logger.info('\tLearning rate: %.5f'%updated_lr)
                    logger.info('\tMinibatch Mean Loss: %.3f'%mean_train_loss)
                    logger.info('\tValidation Accuracy (Unseen): %.3f'%next_valid_accuracy)
                    logger.debug('\tValidation Loss (Unseen): %.3f'%next_valid_loss)
                    logger.info('\tValidation Accuracy (Seen): %.3f'%prev_valid_accuracy)
                    logger.debug('\tValidation Loss (Unseen): %.3f'%prev_valid_loss)

                    test_accuracies = []
                    for test_batch_id in range(test_size//batch_size):
                        batch_test_data = test_dataset[test_batch_id*batch_size:(test_batch_id+1)*batch_size, :, :, :]
                        batch_test_labels = test_labels[test_batch_id*batch_size:(test_batch_id+1)*batch_size, :]

                        feed_test_dict = {tf_test_dataset:batch_test_data, tf_test_labels:batch_test_labels}
                        test_predictions = session.run(pred_test,feed_dict=feed_test_dict)
                        test_accuracies.append(accuracy(test_predictions, batch_test_labels))
                        if test_batch_id<10:
                            logger.debug('='*80)
                            logger.debug('Actual Test Labels %d',test_batch_id)
                            logger.debug(np.argmax(batch_test_labels, axis=1).flatten()[:5])
                            logger.debug('Predicted Test Labels %d',test_batch_id)
                            logger.debug(np.argmax(test_predictions,axis=1).flatten()[:5])
                            logger.debug('Test: %d, %.3f',test_batch_id,accuracy(test_predictions, batch_test_labels))
                            logger.debug('=' * 80)

                    logger.info('\tTest Accuracy: %.3f'%np.mean(test_accuracies))
                    logger.info('='*60)
                    logger.info('')

                    error_logger.info('%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f',
                                      batch_id_multiplier*epoch + batch_id,mean_train_loss,prev_valid_loss,next_valid_loss,
                                      prev_valid_accuracy,next_valid_accuracy,
                                      np.mean(test_accuracies)
                                      )
                    if research_parameters['adapt_structure'] and \
                            not start_adapting and previous_loss - mean_train_loss < research_parameters['loss_diff_threshold']:
                        start_adapting = True
                        logger.info('=' * 80)
                        logger.info('Loss Stabilized: Starting structural adaptations...')
                        logger.info('=' * 80)
                    previous_loss = mean_train_loss

                    if research_parameters['save_train_test_images']:
                        local_dir = output_dir+ os.sep + 'saved_images_'+str(batch_id)
                        if not os.path.exists(local_dir):
                            os.mkdir(local_dir)
                        train_dir = local_dir + os.sep + 'train'
                        if not os.path.exists(train_dir):
                            os.mkdir(train_dir)
                        for img_id in range(50):
                            img_lbl = np.asscalar(np.argmax(batch_labels[img_id,:]))
                            img_data = batch_data[img_id,:,:,:]
                            if not os.path.exists(train_dir+os.sep+str(img_lbl)):
                                os.mkdir(train_dir+os.sep+str(img_lbl))
                            imsave(train_dir+os.sep+ str(img_lbl) + os.sep + 'image_' + str(img_id) + '.png',img_data)

                        test_dir = local_dir + os.sep + 'test'
                        if not os.path.exists(test_dir):
                            os.mkdir(test_dir)
                        for img_id in range(50):
                            img_lbl = np.asscalar(np.argmax(test_labels[img_id, :]))
                            img_data = test_dataset[img_id, :, :, :]
                            if not os.path.exists(test_dir + os.sep + str(img_lbl)):
                                os.mkdir(test_dir + os.sep + str(img_lbl))
                            imsave(test_dir + os.sep + str(img_lbl) + os.sep + 'image_' + str(img_id) + '.png', img_data)

                    # reset variables
                    mean_train_loss = 0.0
                    train_losses = []

                if start_adapting and research_parameters['adapt_structure']:

                    # calculate rolling mean of data distribution (amounts of different classes)
                    for li in range(num_labels):
                        rolling_data_distribution[li]=(1-decay)*rolling_data_distribution[li] + decay*float(cnt[li])/float(batch_size)
                        mean_data_distribution[li] += float(cnt[li])/(float(batch_size)*float(interval_parameters['policy_interval']))

                    # ====================== Policy update and output ======================

                    if batch_id>0 and batch_id%interval_parameters['policy_interval']==0:

                        # update distance measure for class distirbution
                        distMSE = 0.0
                        for li in range(num_labels):
                            distMSE += (prev_mean_distribution[li]-rolling_data_distribution[li])**2
                        distMSE = np.sqrt(distMSE)

                        filter_dict,filter_list = {},[]
                        for op_i,op in enumerate(cnn_ops):
                            if 'conv' in op:
                                filter_dict[op_i]=cnn_hyperparameters[op]['weights'][3]
                                filter_list.append(cnn_hyperparameters[op]['weights'][3])
                            elif 'pool' in op and op!='pool_global':
                                filter_dict[op_i]=0
                                filter_list.append(0)

                        current_state,current_action,curr_invalid_actions = adapter.output_action({'distMSE':distMSE,'filter_counts':filter_dict,'filter_counts_list':filter_list})

                        for li,la in enumerate(current_action):
                            # pooling and fulcon layers
                            if la is None:
                                continue

                            logger.info('Got state: %s, action: %s',str(current_state),str(la))
                            state_action_history[batch_id_multiplier*epoch + batch_id]={'states':current_state,'actions':current_action}

                            # where all magic happens (adding and removing filters)
                            si,ai = current_state,la
                            current_op = cnn_ops[li]

                            for tmp_op in reversed(cnn_ops):
                                if 'conv' in tmp_op:
                                    last_conv_id = tmp_op
                                    break

                            if 'conv' in current_op and ai[0]=='add' :

                                amount_to_add = ai[1]

                                if current_op != last_conv_id:
                                    next_conv_op = [tmp_op for tmp_op in cnn_ops[cnn_ops.index(current_op) + 1:] if 'conv' in tmp_op][0]

                                _ = session.run(tf_add_filters_ops[current_op],
                                                feed_dict={
                                                    tf_action_info:np.asarray([li,1,ai[1]]),
                                                    tf_weights_this:np.random.normal(scale=0.01,size=(
                                                        cnn_hyperparameters[current_op]['weights'][0],cnn_hyperparameters[current_op]['weights'][1],
                                                        cnn_hyperparameters[current_op]['weights'][2],amount_to_add)),
                                                    tf_bias_this:np.random.normal(scale=0.01,size=(amount_to_add)),

                                                    tf_weights_next:np.random.normal(scale=0.01,size=(
                                                        cnn_hyperparameters[next_conv_op]['weights'][0],cnn_hyperparameters[next_conv_op]['weights'][1],
                                                        amount_to_add,cnn_hyperparameters[next_conv_op]['weights'][3])
                                                                                     ) if last_conv_id != current_op else
                                                    np.random.normal(scale=0.01,size=(amount_to_add,cnn_hyperparameters[first_fc]['out'],1,1)),
                                                    tf_running_activations:rolling_ativation_means[current_op],

                                                    tf_wvelocity_this:np.zeros(shape=(
                                                        cnn_hyperparameters[current_op]['weights'][0],cnn_hyperparameters[current_op]['weights'][1],
                                                        cnn_hyperparameters[current_op]['weights'][2],amount_to_add),dtype=np.float32),
                                                    tf_bvelocity_this:np.zeros(shape=(amount_to_add,),dtype=np.float32),
                                                    tf_wvelocity_next:np.zeros(shape=(
                                                        cnn_hyperparameters[next_conv_op]['weights'][0],cnn_hyperparameters[next_conv_op]['weights'][1],
                                                        amount_to_add,cnn_hyperparameters[next_conv_op]['weights'][3]),dtype=np.float32) if last_conv_id != current_op else
                                                    np.zeros(shape=(amount_to_add,cnn_hyperparameters[first_fc]['out'],1,1),dtype=np.float32),
                                                })

                                # change both weights and biase in the current op
                                logger.debug('\tAdding %d new weights',amount_to_add)

                                if research_parameters['debugging']:
                                    logger.debug('\tSummary of changes to weights of %s ...', current_op)
                                    logger.debug('\t\tNew Weights: %s', str(tf.shape(weights[current_op]).eval()))

                                # change out hyperparameter of op
                                cnn_hyperparameters[current_op]['weights'][3]+=amount_to_add
                                if research_parameters['debugging']:
                                    assert cnn_hyperparameters[current_op]['weights'][2]==tf.shape(weights[current_op]).eval()[2]

                                session.run(tf_update_hyp_ops[current_op], feed_dict={
                                    tf_weight_shape:cnn_hyperparameters[current_op]['weights']
                                })

                                if current_op == last_conv_id:
                                    cnn_hyperparameters[first_fc]['in']+=amount_to_add

                                    if research_parameters['debugging']:
                                        logger.debug('\tNew %s in: %d',first_fc,cnn_hyperparameters[first_fc]['in'])
                                        logger.debug('\tSummary of changes to weights of %s',first_fc)
                                        logger.debug('\t\tNew Weights: %s', str(tf.shape(weights[first_fc]).eval()))

                                    session.run(tf_update_hyp_ops[first_fc],feed_dict={
                                        tf_in_size:cnn_hyperparameters[first_fc]['in']
                                    })
                                else:

                                    next_conv_op = \
                                    [tmp_op for tmp_op in cnn_ops[cnn_ops.index(current_op) + 1:] if 'conv' in tmp_op][0]
                                    assert current_op != next_conv_op

                                    if research_parameters['debugging']:
                                        logger.debug('\tSummary of changes to weights of %s',next_conv_op)
                                        logger.debug('\t\tCurrent Weights: %s',str(tf.shape(weights[next_conv_op]).eval()))

                                    cnn_hyperparameters[next_conv_op]['weights'][2] += amount_to_add

                                    if research_parameters['debugging']:
                                        assert cnn_hyperparameters[next_conv_op]['weights'][2]==tf.shape(weights[next_conv_op]).eval()[2]

                                    session.run(tf_update_hyp_ops[next_conv_op], feed_dict={
                                        tf_weight_shape: cnn_hyperparameters[next_conv_op]['weights']
                                    })

                                #changed_var_names = [v.name for v in changed_vars]
                                #logger.debug('All variable names: %s', str([v.name for v in tf.all_variables()]))
                                #logger.debug('Changed var names: %s', str(changed_var_names))
                                #changed_var_values = session.run(changed_vars)

                                #logger.debug('Changed Variable Values')
                                #for n, v in zip(changed_var_names,changed_var_values):
                                    #logger.debug('\t%s,%s',n,str(v.shape))

                                # optimize the newly added fiterls only
                                pool_dataset, pool_labels = hard_pool.get_pool_data()['pool_dataset'], \
                                                            hard_pool.get_pool_data()['pool_labels']

                                # this was done to increase performance and reduce overfitting
                                # instead of optimizing with every single batch in the pool
                                # we select few at random

                                logger.info('\t(Before) Size of Rolling mean vector for %s: %s', current_op,
                                             rolling_ativation_means[current_op].shape)

                                rolling_ativation_means[current_op] = np.append(rolling_ativation_means[current_op],np.zeros(ai[1]))

                                logger.info('\tSize of Rolling mean vector for %s: %s', current_op,
                                             rolling_ativation_means[current_op].shape)

                                # This is a pretty important step
                                # Unless you run this onces, the sizes of weights do not change
                                _ = session.run([logits,tf_activation_ops],feed_dict=feed_dict)
                                for pool_id in range((hard_pool.get_size() // batch_size) - 1):
                                    if np.random.random() < research_parameters['replace_op_train_rate']:
                                        pbatch_data = pool_dataset[pool_id * batch_size:(pool_id + 1) * batch_size, :, :, :]
                                        pbatch_labels = pool_labels[pool_id * batch_size:(pool_id + 1) * batch_size, :]

                                        if pool_id == 0:
                                            _ = session.run([pool_logits,pool_loss],feed_dict= {tf_pool_dataset: pbatch_data,
                                                                                    tf_pool_labels: pbatch_labels}
                                                        )

                                        current_activations,_,_ = session.run([tf_layer_activations,tf_slice_optimize[current_op],tf_slice_vel_update[current_op]],
                                                        feed_dict={tf_pool_dataset: pbatch_data,
                                                                   tf_pool_labels: pbatch_labels,
                                                                   tf_indices:np.arange(cnn_hyperparameters[current_op]['weights'][3]-ai[1],cnn_hyperparameters[current_op]['weights'][3]),
                                                                   tf_indices_size:ai[1]}
                                                        )

                                        # update rolling activation means
                                        for op, op_activations in current_activations.items():
                                            assert current_activations[op].size == cnn_hyperparameters[op]['weights'][3]
                                            rolling_ativation_means[op] = (1 - act_decay) * rolling_ativation_means[
                                                op] + decay * current_activations[op]

                            elif 'conv' in current_op and ai[0]=='remove' :

                                _,rm_indices = session.run(tf_rm_filters_ops[current_op],
                                                feed_dict={
                                                    tf_action_info: np.asarray([li, 0, ai[1]]),
                                                    tf_running_activations: rolling_ativation_means[current_op]
                                                })
                                rm_indices = rm_indices.flatten()
                                amount_to_rmv = ai[1]

                                if research_parameters['remove_filters_by'] == 'Activation':
                                    logger.debug('\tRemoving filters for op %s', current_op)
                                    logger.debug('\t\t\tIndices: %s', rm_indices[:10])

                                elif research_parameters['remove_filters_by'] == 'Distance':
                                    logger.debug('\tRemoving filters for op %s', current_op)

                                    logger.debug('\t\tSimilarity summary')
                                    logger.debug('\t\t\tIndices: %s', rm_indices[:10])

                                    logger.debug('\t\tSize of indices to remove: %s/%d', rm_indices.size,
                                                 cnn_hyperparameters[current_op]['weights'][3])
                                    indices_of_filters_keep = list(
                                        set(np.arange(cnn_hyperparameters[current_op]['weights'][3])) - set(rm_indices.tolist()))
                                    logger.debug('\t\tSize of indices to keep: %s/%d', len(indices_of_filters_keep),
                                                 cnn_hyperparameters[current_op]['weights'][3])

                                cnn_hyperparameters[current_op]['weights'][3] -= amount_to_rmv
                                if research_parameters['debugging']:
                                    logger.debug('\tSize after feature map reduction: %s,%s', current_op, tf.shape(weights[current_op]).eval())
                                    assert tf.shape(weights[current_op]).eval()[3] == cnn_hyperparameters[current_op]['weights'][3]

                                session.run(tf_update_hyp_ops[current_op], feed_dict={
                                    tf_weight_shape: cnn_hyperparameters[current_op]['weights']
                                })

                                if current_op == last_conv_id:
                                    cnn_hyperparameters[first_fc]['in'] -= amount_to_rmv
                                    if research_parameters['debugging']:
                                        logger.debug('\tSize after feature map reduction: %s,%s',
                                                     first_fc,str(tf.shape(weights[first_fc]).eval()))

                                    session.run(tf_update_hyp_ops[first_fc], feed_dict={
                                        tf_in_size: cnn_hyperparameters[first_fc]['in']
                                    })

                                else:
                                    next_conv_op = [tmp_op for tmp_op in cnn_ops[cnn_ops.index(current_op) + 1:] if 'conv' in tmp_op][0]
                                    assert current_op != next_conv_op

                                    cnn_hyperparameters[next_conv_op]['weights'][2] -= amount_to_rmv

                                    if research_parameters['debugging']:
                                        logger.debug('\tSize after feature map reduction: %s,%s', next_conv_op,
                                                     str(tf.shape(weights[next_conv_op]).eval()))
                                        assert tf.shape(weights[next_conv_op]).eval()[2] == \
                                               cnn_hyperparameters[next_conv_op]['weights'][2]

                                    session.run(tf_update_hyp_ops[next_conv_op], feed_dict={
                                        tf_weight_shape: cnn_hyperparameters[next_conv_op]['weights']
                                    })

                                logger.info('\t(Before) Size of Rolling mean vector for %s: %s', current_op,
                                            rolling_ativation_means[current_op].shape)
                                rolling_ativation_means[current_op] = np.delete(rolling_ativation_means[current_op], rm_indices)
                                logger.info('\tSize of Rolling mean vector for %s: %s',current_op,rolling_ativation_means[current_op].shape)

                                # This is a pretty important step
                                # Unless you run this onces, the sizes of weights do not change
                                _ = session.run([logits, tf_activation_ops], feed_dict=feed_dict)

                            elif 'conv' in current_op and (ai[0]=='replace'):

                                if research_parameters['remove_filters_by'] == 'Activation':
                                    layer_activations = rolling_ativation_means[current_op]
                                    indices_of_filters_replace = (np.argsort(layer_activations).flatten()[:ai[1]]).astype(
                                        'int32')
                                elif research_parameters['remove_filters_by'] == 'Distance':
                                    # calculate cosine distance for F filters (FxF matrix)
                                    # take one side of diagonal, find (f1,f2) pairs with least distance
                                    # select indices amnt f2 indices
                                    indices_of_filters_replace = session.run(tf_rm_filters_ops[current_op])

                                for pool_id in range((hard_pool.get_size() // batch_size) - 1):
                                    if np.random.random() < research_parameters['replace_op_train_rate']:
                                        pbatch_data = pool_dataset[pool_id * batch_size:(pool_id + 1) * batch_size, :, :, :]
                                        pbatch_labels = pool_labels[pool_id * batch_size:(pool_id + 1) * batch_size, :]

                                        if pool_id==0:
                                            _ = session.run([pool_logits,pool_loss],feed_dict= {tf_pool_dataset: pbatch_data,
                                                                                    tf_pool_labels: pbatch_labels}
                                                            )
                                        current_activations,_,_ = session.run([tf_layer_activations,tf_slice_optimize[current_op],tf_slice_vel_update[current_op]],
                                                        feed_dict={tf_pool_dataset: pbatch_data,
                                                                   tf_pool_labels: pbatch_labels,
                                                                   tf_indices:indices_of_filters_replace,
                                                                   tf_indices_size:ai[1]}
                                                        )

                                        # update rolling activation means
                                        for op, op_activations in current_activations.items():
                                            assert current_activations[op].size == cnn_hyperparameters[op]['weights'][3]
                                            rolling_ativation_means[op] = (1 - act_decay) * rolling_ativation_means[
                                                op] + decay * current_activations[op]


                            elif 'conv' in current_op and ai[0]=='finetune':
                                # pooling takes place here

                                op = cnn_ops[li]
                                pool_dataset,pool_labels = hard_pool.get_pool_data()['pool_dataset'],hard_pool.get_pool_data()['pool_labels']

                                logger.debug('Only tuning following variables...')
                                logger.debug('\t%s,%s',weights[op].name,str(weights[op]))
                                logger.debug('\t%s,%s',biases[op].name,str(biases[op]))

                                assert weights[op].name in [v.name for v in tf.trainable_variables()]
                                assert biases[op].name in [v.name for v in tf.trainable_variables()]



                                # without if can give problems in exploratory stage because of no data in the pool
                                if hard_pool.get_size()>batch_size:
                                    for pool_id in range((hard_pool.get_size()//batch_size)-1):

                                        pbatch_data = pool_dataset[pool_id*batch_size:(pool_id+1)*batch_size, :, :, :]
                                        pbatch_labels = pool_labels[pool_id*batch_size:(pool_id+1)*batch_size, :]
                                        pool_feed_dict = {tf_pool_dataset:pbatch_data,tf_pool_labels:pbatch_labels}

                                        if pool_id == 0:
                                            _ = session.run([pool_logits, pool_loss], feed_dict={tf_pool_dataset:pbatch_data,tf_pool_labels:pbatch_labels})
                                        _, _, _, _ = session.run([pool_logits, pool_loss, optimize_with_pool, pool_vel_updates], feed_dict=pool_feed_dict)

                                break # otherwise will do this repeadedly for the number of layers

                        # updating the policy
                        if prev_state is not None and prev_action is not None:

                            pool_accuracy = []
                            pool_dataset, pool_labels = hard_pool.get_pool_data()['pool_dataset'], \
                                                        hard_pool.get_pool_data()['pool_labels']
                            for pool_id in range((hard_pool.get_size()//batch_size)-1):
                                pbatch_data = pool_dataset[pool_id*batch_size:(pool_id+1)*batch_size, :, :, :]
                                pbatch_labels = pool_labels[pool_id*batch_size:(pool_id+1)*batch_size, :]
                                pool_feed_dict = {tf_pool_dataset:pbatch_data,tf_pool_labels:pbatch_labels}

                                _, _, p_predictions = session.run([pool_logits, pool_loss, pool_pred], feed_dict=pool_feed_dict)
                                pool_accuracy.append(accuracy(p_predictions,pbatch_labels))

                            # don't use current state as the next state, current state is for a different layer
                            next_state = [distMSE]
                            for li,la in enumerate(prev_action):
                                if la is None:
                                    assert li not in convolution_op_ids
                                    next_state.append(0)
                                    continue
                                elif la[0]=='add':
                                    next_state.append(prev_state[li+1] + la[1])
                                elif la[0]=='remove':
                                    next_state.append(prev_state[li+1] - la[1])
                                else:
                                    next_state.append(prev_state[li + 1])

                            next_state = tuple(next_state)

                            logger.info('\tState (prev): %s', str(prev_state))
                            logger.info('\tAction (prev): %s', str(prev_action))
                            logger.info('\tState (next): %s\n', str(next_state))

                            #logger.info('\tValid accuracy Gain: %.2f (Unseen) %.2f (seen)',
                            #            (next_valid_accuracy_after - next_valid_accuracy),
                            #            (prev_valid_accuracy_after-prev_valid_accuracy)
                            #            )
                            adapter.update_policy({'prev_state': prev_state, 'prev_action': prev_action,
                                                   'curr_state': next_state,
                                                   'next_accuracy': None,
                                                   'prev_accuracy': None,
                                                   'pool_accuracy': np.mean(pool_accuracy),
                                                   'prev_pool_accuracy': prev_pool_accuracy,
                                                   'invalid_actions':prev_invalid_actions})

                            prev_pool_accuracy = np.mean(pool_accuracy)

                            cnn_structure_logger.info(
                                '%d:%s:%s:%.5f:%s', (batch_id_multiplier*epoch)+batch_id, current_state, current_action,np.mean(pool_accuracy),
                                utils.get_cnn_string_from_ops(cnn_ops, cnn_hyperparameters)
                            )
                            #q_logger.info('%d,%.5f',epoch*batch_id_multiplier + batch_id,adapter.get_average_Q())

                        #reset rolling activation means because network changed
                        #rolling_ativation_means = {}
                        #for op in cnn_ops:
                        #    if 'conv' in op:
                        #        logger.debug("Resetting activation means to zero for %s with %d",op,cnn_hyperparameters[op]['weights'][3])
                        #        rolling_ativation_means[op]=np.zeros([cnn_hyperparameters[op]['weights'][3]])

                        logger.debug('Resetting both data distribution means')

                        prev_mean_distribution = mean_data_distribution
                        for li in range(num_labels):
                            rolling_data_distribution[li]=0.0
                            mean_data_distribution[li]=0.0

                        prev_state = current_state
                        prev_action = current_action
                        prev_invalid_actions = curr_invalid_actions

                    if batch_id>0 and batch_id%interval_parameters['history_dump_interval']==0:
                        with open(output_dir + os.sep + 'state_actions_' + str(epoch) + "_" + str(batch_id)+'.pickle', 'wb') as f:
                            pickle.dump(state_action_history, f, pickle.HIGHEST_PROTOCOL)
                            state_action_history = {}

                        #with open(output_dir + os.sep + 'Q_' + str(epoch) + "_" + str(batch_id)+'.pickle', 'wb') as f:
                        #    pickle.dump(adapter.get_Q(), f, pickle.HIGHEST_PROTOCOL)

                        pool_dist_string = ''
                        for val in hard_pool.get_class_distribution():
                            pool_dist_string += str(val)+','

                        pool_dist_logger.info('%s%d',pool_dist_string,hard_pool.get_size())

                t1 = time.clock()
                op_count = len(graph.get_operations())
                var_count = len(tf.global_variables()) + len(tf.local_variables()) + len(tf.model_variables())
                perf_logger.info('%d,%.5f,%.5f,%d,%d',(batch_id_multiplier*epoch)+batch_id,t1-t0,t1_train-t0_train,op_count,var_count)
