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
from collections import Counter
from scipy.misc import imsave
import getopt
from functools import partial
import time


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


def get_final_x(cnn_ops,cnn_hyps):
    '''
    Takes a new operations and concat with existing set of operations
    then output what the final output size will be
    :param op_list: list of operations
    :param hyp: Hyperparameters fo the operation
    :return: a tuple (width,height) of x
    '''

    x = image_size
    for op in cnn_ops:
        hyp_op = cnn_hyps[op]
        if 'conv' in op:
            f = hyp_op['weights'][0]
            s = hyp_op['stride'][1]

            x = ceil(float(x)/float(s))
        elif 'pool' in op:
            if hyp_op['padding']=='SAME':
                f = hyp_op['kernel'][1]
                s = hyp_op['stride'][1]
                x = ceil(float(x)/float(s))
            elif hyp_op['padding']=='VALID':
                f = hyp_op['kernel'][1]
                s = hyp_op['stride'][1]
                x = ceil(float(x - f + 1)/float(s))

    return x


def initialize_cnn_with_ops(cnn_ops,cnn_hyps):
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

            logger.debug('Weights for %s initialized with size %s',op,str(cnn_hyps[op]['weights']))
            logger.debug('Biases for %s initialized with size %d',op,cnn_hyps[op]['weights'][3])

        if 'fulcon' in op:
            weights['fulcon_out'] = tf.Variable(tf.truncated_normal(
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


def get_logits_with_ops(dataset, weights, biases, use_dropout):
    global cnn_ops,tf_cnn_hyperparameters

    logger.debug('Defining the logit calculation ...')
    logger.debug('\tCurrent set of operations: %s'%cnn_ops)
    act_means = {}
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
            act_means[op]=tf.reduce_mean(x,[0,1,2])

        if 'pool' in op:
            logger.debug('\tPooling (%s) with Kernel:%s Stride:%s'%(op,cnn_hyperparameters[op]['kernel'],cnn_hyperparameters[op]['stride']))
            if cnn_hyperparameters[op]['type'] is 'max':
                x = tf.nn.max_pool(x,ksize=cnn_hyperparameters[op]['kernel'],strides=cnn_hyperparameters[op]['stride'],padding=cnn_hyperparameters[op]['padding'])
            elif cnn_hyperparameters[op]['type'] is 'avg':
                x = tf.nn.avg_pool(x,ksize=cnn_hyperparameters[op]['kernel'],strides=cnn_hyperparameters[op]['stride'],padding=cnn_hyperparameters[op]['padding'])

            #logger.debug('\t\tX after %s:%s'%(op,tf.shape(x).eval()))

        if 'fulcon' in op:
            break

    # we need to reshape the output of last subsampling layer to
    # convert 4D output to a 2D input to the hidden layer
    # e.g subsample layer output [batch_size,width,height,depth] -> [batch_size,width*height*depth]

    fin_x = get_final_x(cnn_ops,cnn_hyperparameters)

    logger.debug('Input size of fulcon_out : %d',cnn_hyperparameters['fulcon_out']['in'])
    x = tf.reshape(x, [batch_size,tf_cnn_hyperparameters['fulcon_out']['in']])

    if use_dropout:
        x = tf.nn.dropout(x,1-dropout_rate)

    return tf.matmul(x, weights['fulcon_out']) + biases['fulcon_out'],act_means


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


def optimize_func(loss,global_step,start_lr=start_lr):
    global research_parameters
    global weights,biases

    optimize_ops = []
    # Optimizer.
    if research_parameters['optimizer'] == 'SGD':
        learning_rate = tf.constant(start_lr, dtype=tf.float32, name='learning_rate')
        optimize_ops.append(tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss))

    elif research_parameters['optimizer'] == 'Momentum':
        # custom momentum optimizing
        # apply_gradient([g,v]) does the following v -= eta*g
        # eta is learning_rate
        # Since what we need is
        # v(t+1) = mu*v(t) - eta*g
        # theta(t+1) = theta(t) + v(t+1) --- (2)
        # we form (2) in the form theta(t+1) -= 1.0*(-v(t+1))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        learning_rate = tf.constant(start_lr, dtype=tf.float32, name='learning_rate')

        for op in weight_velocity_vectors.keys():
            [(grads_w,w),(grads_b,b)] = optimizer.compute_gradients(loss, [weights[op], biases[op]])

            # update velocity vector
            weight_velocity_vectors[op] = research_parameters['momentum']*weight_velocity_vectors[op] + (learning_rate*grads_w)
            bias_velocity_vectors[op] = research_parameters['momentum']*bias_velocity_vectors[op] + (learning_rate*grads_b)

            optimize_ops.append(optimizer.apply_gradients(
                        [(weight_velocity_vectors[op],weights[op]),(bias_velocity_vectors[op],biases[op])]
            ))
    else:
        raise NotImplementedError

    return optimize_ops,learning_rate


def optimize_with_variable_func(loss,global_step,var_ops):
    # Optimizer.
    optimize_ops = []
    if research_parameters['optimizer'] == 'SGD':
        if decay_learning_rate:
            learning_rate = tf.train.exponential_decay(start_lr, global_step,decay_steps=1,decay_rate=0.95)
        else:
            learning_rate = tf.constant(start_lr,dtype=tf.float32,name='learning_rate')
        variable_list = [weights[op] for op in var_ops]
        variable_list.extend([biases[op] for op in var_ops])
        assert variable_list is not None and len(variable_list)>0
        optimize_ops.append(
            tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss,var_list=variable_list)
        )

    elif research_parameters['optimizer'] == 'Momentum':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        learning_rate = tf.constant(start_lr, dtype=tf.float32, name='learning_rate')
        for op in var_ops:
            grads_wb = optimizer.compute_gradients(loss, [weights[op], biases[op]])
            (grads_w, w), (grads_b, b) = grads_wb[0], grads_wb[1]
            # update velocity vector

            with tf.control_dependencies([
                tf.assign(weight_velocity_vectors[op],
                          - (research_parameters['momentum'] * weight_velocity_vectors[op] - (learning_rate * grads_w))),
                tf.assign(bias_velocity_vectors[op],
                          -(research_parameters['momentum'] * bias_velocity_vectors[op] - (learning_rate * grads_b)))
            ]):
                optimize_ops.append(optimizer.apply_gradients(
                    [(weight_velocity_vectors[op], weights[op]), (bias_velocity_vectors[op], biases[op])]))

    else:
        raise NotImplementedError

    return optimize_ops,learning_rate


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
    global weights,biases
    global cnn_ops,cnn_hyperparameters,tf_cnn_hyperparameters

    grad_ops = []
    grads_w,grads_b= {},{}
    mask_grads_w,mask_grads_b = {},{}
    learning_rate = tf.constant(start_lr,dtype=tf.float32,name='learning_rate')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

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
        grad_ops.append(optimizer.apply_gradients([(grads_w[op],w),(grads_b[op],b)]))

    next_op = None
    for tmp_op in cnn_ops[cnn_ops.index(op)+1:]:
        if 'conv' in tmp_op or 'fulcon' in tmp_op:
            next_op = tmp_op
            break
    logger.debug('Next conv op: %s',next_op)
    [(grads_w[next_op], w),(grads_b[next_op],b)] = optimizer.compute_gradients(loss, [weights[next_op],biases[next_op]])

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
        grad_ops.append(optimizer.apply_gradients([(grads_w[next_op], weights[next_op]),(grads_b[next_op],biases[next_op])]))

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
        grad_ops.append(optimizer.apply_gradients([(grads_w[next_op], weights[next_op]),(grads_b[next_op],biases[next_op])]))

    if research_parameters['optimize_end_to_end']:
        for tmp_op in cnn_ops[cnn_ops.index(next_op)+1:]:
            if 'conv' in tmp_op or 'fulcon' in tmp_op:
                [(grads_w[tmp_op], w), (grads_b[tmp_op], b)] = optimizer.compute_gradients(loss, [w, b])
            grad_ops.append(optimizer.apply_gradients([(grads_w[next_op], weights[next_op]),(grads_b[next_op], biases[next_op])]))

    return grad_ops

def inc_global_step(global_step):
    return global_step.assign(global_step+1)


def predict_with_logits(logits):
    # Predictions for the training, validation, and test data.
    prediction = tf.nn.softmax(logits)
    return prediction


def predict_with_dataset(dataset, weights,biases):
    logits,_ = get_logits_with_ops(dataset,weights,biases,False)
    prediction = tf.nn.softmax(logits)
    return prediction


def accuracy(predictions, labels):
    assert predictions.shape[0]==labels.shape[0]
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


def get_ops_hyps_from_string(net_string):
    # E.g. String
    # Init,0,0,0#C,1,1,64#C,5,1,64#C,5,1,128#P,5,2,0#C,1,1,64#P,2,2,0#Terminate,0,0,0
    cnn_ops = []
    cnn_hyperparameters = {}
    prev_conv_hyp = None

    op_tokens = net_string.split('#')
    depth_index = 0
    last_feature_map_depth = 3 # need this to calculate the fulcon layer in size
    for token in op_tokens:
        # state (layer_depth,op=(type,kernel,stride,depth),out_size)
        token_tokens = token.split(',')
        # op => type,kernel,stride,depth
        op = (token_tokens[0],int(token_tokens[1]),int(token_tokens[2]),int(token_tokens[3]))
        if op[0] == 'C':
            op_id = 'conv_'+str(depth_index)
            if prev_conv_hyp is None:
                hyps = {'weights':[op[1],op[1],num_channels,op[3]],'stride':[1,op[2],op[2],1],'padding':'SAME'}
            else:
                hyps = {'weights':[op[1],op[1],prev_conv_hyp['weights'][3],op[3]],'stride':[1,op[2],op[2],1],'padding':'SAME'}

            cnn_ops.append(op_id)
            cnn_hyperparameters[op_id]=hyps
            prev_conv_hyp = hyps # need this to set the input depth for a conv layer
            last_feature_map_depth = op[3]
            depth_index += 1

        elif op[0] == 'P':
            op_id = 'pool_'+str(depth_index)
            hyps = {'type':'max','kernel':[1,op[1],op[1],1],'stride':[1,op[2],op[2],1],'padding':'SAME'}
            cnn_ops.append(op_id)
            cnn_hyperparameters[op_id]=hyps
            depth_index += 1

        elif op[0] == 'Terminate':
            if len(op_tokens)>2:
                output_size = get_final_x(cnn_ops,cnn_hyperparameters)
            # this could happen if we get terminal state without any other states in trajectory
            else:
                output_size = image_size

            if output_size>1:
                cnn_ops.append('pool_global')
                pg_hyps =  {'type':'avg','kernel':[1,output_size,output_size,1],'stride':[1,1,1,1],'padding':'VALID'}
                cnn_hyperparameters['pool_global']=pg_hyps

            op_id = 'fulcon_out'
            hyps = {'in':1*1*last_feature_map_depth,'out':num_labels}
            cnn_ops.append(op_id)
            cnn_hyperparameters[op_id]=hyps
        elif op[0] == 'Init':
            continue
        else:
            print('='*40)
            print(op[0])
            print('='*40)
            raise NotImplementedError

    return cnn_ops,cnn_hyperparameters


def get_cnn_string_from_ops(cnn_ops,cnn_hyps):
    current_cnn_string = ''
    for op in cnn_ops:
        if 'conv' in op:
            current_cnn_string += '#C,'+str(cnn_hyps[op]['weights'][0])+','+str(cnn_hyps[op]['stride'][1])+','+str(cnn_hyps[op]['weights'][3])
        if 'pool' in op:
            current_cnn_string += '#P,'+str(cnn_hyps[op]['kernel'][0])+','+str(cnn_hyps[op]['stride'][1])+','+str(0)
        if 'fulcon' in op:
            current_cnn_string += '#Terminate,0,0,0'

    return current_cnn_string

def add_with_action(op, tf_action_info, tf_activations):
    global cnn_hyperparameters, tf_cnn_hyperparameters
    global logger,weights,biases
    global weight_velocity_vectors,bias_velocity_vectors

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
    tf_new_weights = tf.concat(3,[weights[op],
                                        tf.truncated_normal([tf_cnn_hyperparameters[op]['weights'][0],
                                                             tf_cnn_hyperparameters[op]['weights'][1],tf_cnn_hyperparameters[op]['weights'][2],amount_to_add],stddev=0.02)
                                        ])
    tf_new_biases = tf.concat(0,[biases[op],tf.truncated_normal([amount_to_add],stddev=0.001)])

    # updating velocity vectors
    '''if research_parameters['optimizer']=='Momentum':
        weight_velocity_vectors[op] = tf.concat(3,[weight_velocity_vectors[op],
                                              tf.zeros([cnn_hyps[op]['weights'][0],
                                                        cnn_hyps[op]['weights'][1],cnn_hyps[op]['weights'][2],amount_to_add],dtype=tf.float32)
                                              ])
        bias_velocity_vectors[op] = tf.concat(0,[bias_velocity_vectors[op],tf.zeros([amount_to_add],dtype=tf.float32)])'''

    update_ops.append(tf.assign(weights[op], tf_new_weights, validate_shape=False))
    update_ops.append(tf.assign(biases[op], tf_new_biases, validate_shape = False))

    # ================ Changes to next_op ===============
    # Very last convolutional layer
    # this is different from other layers
    # as a change in this require changes to FC layer
    if op==last_conv_id:
        # change FC layer
        tf_new_weights = tf.concat(0,[weights['fulcon_out'],tf.truncated_normal([amount_to_add,num_labels],stddev=0.01)])


        # updating velocity vectors
        '''if research_parameters['optimizer'] == 'Momentum':
            weight_velocity_vectors['fulcon_out'] = tf.concat(0,[weight_velocity_vectors['fulcon_out'],tf.zeros([amount_to_add,num_labels])])'''

        update_ops.append(tf.assign(weights['fulcon_out'], tf_new_weights, validate_shape=False))

    else:

        # change in hyperparameter of next conv op
        next_conv_op = [tmp_op for tmp_op in cnn_ops[cnn_ops.index(op)+1:] if 'conv' in tmp_op][0]
        assert op!=next_conv_op

        # change only the weights in next conv_op
        tf_new_weights=tf.concat(2,[weights[next_conv_op],
                                      tf.truncated_normal([tf_cnn_hyperparameters[next_conv_op]['weights'][0],tf_cnn_hyperparameters[next_conv_op]['weights'][1],
                                                           amount_to_add,tf_cnn_hyperparameters[next_conv_op]['weights'][3]],stddev=0.01)])

        '''if research_parameters['optimizer']=='Momentum':
            weight_velocity_vectors[next_conv_op] = tf.concat(2,[
                weight_velocity_vectors[next_conv_op],
                tf.zeros([tf_cnn_hyperparameters[next_conv_op]['weights'][0],tf_cnn_hyperparameters[next_conv_op]['weights'][1],
                          amount_to_add,tf_cnn_hyperparameters[next_conv_op]['weights'][3]])]
                                                         )'''
        update_ops.append(tf.assign(weights[next_conv_op], tf_new_weights, validate_shape=False))

    return update_ops

def remove_with_action(op, tf_action_info, tf_activations):
    global cnn_hyperparameters, tf_cnn_hyperparameters
    global logger,weights,biases
    global weight_velocity_vectors,bias_velocity_vectors

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
        indices_of_filters_keep = (np.argsort(layer_activations).flatten()[amount_to_rmv:]).astype('int32')
        raise NotImplementedError
    elif research_parameters['remove_filters_by']=='Distance':
        # calculate cosine distance for F filters (FxF matrix)
        # take one side of diagonal, find (f1,f2) pairs with least distance
        # select indices amnt f2 indices

        reshaped_weight = tf.transpose(weights[op],[3,0,1,2])
        reshaped_weight = tf.reshape(weights[op],[tf_cnn_hyperparameters[op]['weights'][3],
                                                  tf_cnn_hyperparameters[op]['weights'][0]*tf_cnn_hyperparameters[op]['weights'][1]*tf_cnn_hyperparameters[op]['weights'][2]]
                                     )
        cos_sim_weights = tf.matmul(reshaped_weight,tf.transpose(reshaped_weight),name='dot_prod_cos_sim')/ tf.matmul(
            tf.sqrt(tf.reduce_sum(reshaped_weight**2,axis=1,keep_dims=True)),
            tf.sqrt(tf.transpose(tf.reduce_sum(reshaped_weight**2,axis=1,keep_dims=True)))
        ,name='norm_cos_sim')

        upper_triang_cos_sim = tf.matrix_band_part(cos_sim_weights, 0, -1,name='upper_triang_cos_sim')
        zero_diag_triang_cos_sim = tf.matrix_set_diag(upper_triang_cos_sim, tf.zeros(shape=[tf_cnn_hyperparameters[op]['weights'][3]]),name='zero_diag_upper_triangle')
        flattened_cos_sim = tf.reshape(zero_diag_triang_cos_sim,shape=[-1],name='flattend_cos_sim')

        # we are finding top amount_to_rmv + epsilon amount because
        # to avoid k_values = {...,(83,1)(139,94)(139,83),...} like incidents
        # above case will ignore both indices of (139,83) resulting in a reduction < amount_to_rmv
        [high_sim_values, high_sim_indices] = tf.nn.top_k(flattened_cos_sim,
                                                          k=amount_to_rmv, name='top_k_indices')


        tf_indices_to_remove_1 = tf.reshape(tf.mod(high_sim_indices, tf_cnn_hyperparameters[op]['weights'][3]),shape=[-1],name='mod_indices')
        tf_indices_to_remove_2 = tf.reshape(tf.floor_div(high_sim_indices, tf_cnn_hyperparameters[op]['weights'][3]),shape=[-1],name='floor_div_indices')
        # concat both mod and floor_div indices
        tf_indices_to_rm = tf.reshape(tf.stack([tf_indices_to_remove_1,tf_indices_to_remove_2],name='all_rm_indices'),shape=[-1])
        # return both values and indices of unique values (discard indices)
        tf_unique_rm_ind,_ = tf.unique(tf_indices_to_rm,name='unique_rm_indices')
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

    '''if research_parameters['optimizer'] == 'Momentum':
        weight_velocity_vectors[op] = tf.transpose(weight_velocity_vectors[op], [3, 0, 1, 2])
        weight_velocity_vectors[op] = tf.gather_nd(weight_velocity_vectors[op], [[idx] for idx in indices_of_filters_keep])
        weight_velocity_vectors[op] = tf.transpose(weight_velocity_vectors[op], [1, 2, 3, 0])

        bias_velocity_vectors[op] = tf.gather(bias_velocity_vectors[op],indices_of_filters_keep)'''

    if op==last_conv_id:

        tf_new_weights = tf.gather_nd(weights['fulcon_out'],tf_indices_to_keep)

        update_ops.append(tf.assign(weights['fulcon_out'],tf_new_weights,validate_shape=False))

        '''if research_parameters['optimizer']=='Momentum':
            weight_velocity_vectors['fulcon_out']=tf.gather_nd(weight_velocity_vectors['fulcon_out'],[[idx] for idx in indices_of_filters_keep])'''

    else:
        # change in hyperparameter of next conv op
        next_conv_op = [tmp_op for tmp_op in cnn_ops[cnn_ops.index(op)+1:] if 'conv' in tmp_op][0]
        assert op!=next_conv_op

        # change only the weights in next conv_op

        tf_new_weights = tf.transpose(weights[next_conv_op],[2,0,1,3])
        tf_new_weights = tf.gather_nd(tf_new_weights,tf_indices_to_keep)
        tf_new_weights = tf.transpose(tf_new_weights,[1,2,0,3])

        update_ops.append(tf.assign(weights[next_conv_op],tf_new_weights,validate_shape=False))

        '''if research_parameters['optimizer']=='Momentum':
            weight_velocity_vectors[next_conv_op] = tf.transpose(weight_velocity_vectors[next_conv_op],[2,0,1,3])
            weight_velocity_vectors[next_conv_op] = tf.gather_nd(weight_velocity_vectors[next_conv_op],[[idx] for idx in indices_of_filters_keep])
            weight_velocity_vectors[next_conv_op] = tf.transpose(weight_velocity_vectors[next_conv_op],[1,2,3,0])'''

    return update_ops,tf_indices_to_rm

def load_data_from_memmap(dataset_info,dataset_filename,label_filename,start_idx,size):
    global logger
    num_labels = dataset_info['num_labels']
    col_count = (dataset_info['image_size'],dataset_info['image_size'],dataset_info['num_channels'])
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


weight_velocity_vectors, bias_velocity_vectors = {}, {}
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


weights,biases = None,None

research_parameters = {
    'save_train_test_images':False,
    'log_class_distribution':True,'log_distribution_every':128,
    'adapt_structure' : True,
    'hard_pool_acceptance_rate':0.1, 'accuracy_threshold_hard_pool':50,
    'replace_op_train_rate':0.5, # amount of batches from hard_pool selected to train
    'optimizer':'SGD','momentum':0.9,
    'remove_filters_by':'Distance',
    'optimize_end_to_end':True, # if true functions such as add and finetune will optimize the network from starting layer to end (fulcon_out)
    'loss_diff_threshold':0.02,
    'debugging':True if logging_level==logging.DEBUG else False
}

interval_parameters = {
    'history_dump_interval':500,
    'policy_interval' : 20, #number of batches to process for each policy iteration
    'test_interval' : 25
}

state_action_history = {}

cnn_ops, cnn_hyperparameters, tf_cnn_hyperparameters = None,None,{}

if __name__=='__main__':
    global weights,biases
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
    behavior = 'stationary'

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
            chunk_size = 51200
        elif behavior == 'stationary':
            dataset_filename='data_non_station'+os.sep+'cifar-10-station-dataset.pkl'
            label_filename='data_non_station'+os.sep+'cifar-10-station-labels.pkl'
            dataset_size = 1280000
            chunk_size = 51200

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

    dataset_info['image_size']=image_size
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
        cnn_structure_logger.info('#batch_id:state:action:#layer_1_hyperparameters#layer_2_hyperparameters#...')


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

    # Loading data
    memmap_idx = 0
    train_dataset,train_labels = None,None

    test_dataset,test_labels = load_data_from_memmap(dataset_info,test_dataset_filename,test_label_filename,0,test_size)

    assert chunk_size%batch_size==0
    batches_in_chunk = chunk_size//batch_size

    logger.info('='*80)
    logger.info('\tTrain Size: %d'%dataset_size)
    logger.info('='*80)

    cnn_string = "C,5,1,256#P,3,2,0#C,5,1,512#C,3,1,128#Terminate,0,0,0"
    #cnn_string = "C,3,1,128#P,5,2,0#C,5,1,128#C,3,1,512#C,5,1,128#C,5,1,256#P,2,2,0#C,5,1,64#Terminate,0,0,0"
    #cnn_string = "C,3,4,128#P,5,2,0#Terminate,0,0,0"

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

        cnn_ops,cnn_hyperparameters = get_ops_hyps_from_string(cnn_string)
        init_tf_hyperparameters()

        weights,biases = initialize_cnn_with_ops(cnn_ops,cnn_hyperparameters)

        # -1 is because we don't want to count pool_global
        layer_count = len([op for op in cnn_ops if 'conv' in op or 'pool' in op])-1

        # Adapting Policy Learner
        adapter = qlearner.AdaCNNAdaptingQLearner(learning_rate=0.1,
                                                  discount_rate=0.9,
                                                  fit_interval = 5,
                                                  even_tries = 3,
                                                  filter_upper_bound=1024,
                                                  net_depth=layer_count,
                                                  upper_bound=10,
                                                  epsilon=0.99)

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

        if research_parameters['optimizer']=='Momentum':
            init_velocity_vectors(cnn_ops,cnn_hyperparameters)

        init_op = tf.global_variables_initializer()
        _ = session.run(init_op)

        logits,act_means = get_logits_with_ops(tf_dataset,weights,biases,use_dropout)
        loss = calc_loss(logits,tf_labels,True,tf_data_weights)
        loss_vec = calc_loss_vector(logits,tf_labels)

        pred = predict_with_logits(logits)
        optimize,upd_lr = optimize_func(loss,global_step,start_lr)
        inc_gstep = inc_global_step(global_step)

        init_op = tf.global_variables_initializer()
        _ = session.run(init_op)

        # valid predict function
        tf_valid_logits, _ = get_logits_with_ops(tf_valid_dataset,weights,biases,False)
        tf_valid_predictions = predict_with_logits(tf_valid_logits)
        tf_valid_loss = calc_loss(tf_valid_logits,tf_valid_labels,False,None)

        pred_test = predict_with_dataset(tf_test_dataset, weights,biases)

        pool_logits, _ = get_logits_with_ops(tf_pool_dataset, weights, biases,use_dropout)
        pool_loss = calc_loss(pool_logits, tf_pool_labels,False,None)

        tf_indices = tf.placeholder(dtype=tf.int32,shape=(None,),name='optimize_indices')
        tf_indices_size = tf.placeholder(tf.int32)
        tf_slice_optimize = {}
        tf_add_filters_ops,tf_rm_filters_ops = {},{}
        tf_action_info = tf.placeholder(shape=[3], dtype=tf.int32,
                                        name='tf_action')  # [op_id,action_id,amount] (action_id 0 - add, 1 -remove)
        tf_running_activations = tf.placeholder(shape=(None,), dtype=tf.float32, name='running_activations')

        tf_weight_shape = tf.placeholder(shape=[4],dtype=tf.int32,name='weight_shape')
        tf_in_size = tf.placeholder(dtype=tf.int32)
        tf_update_hyp_ops={}
        for tmp_op in cnn_ops:
            if 'conv' in tmp_op:
                tf_update_hyp_ops[tmp_op] = update_tf_hyperparameters(tmp_op,tf_weight_shape,tf_in_size)
                tf_add_filters_ops[tmp_op] = add_with_action(tmp_op,tf_action_info,tf_running_activations)
                tf_rm_filters_ops[tmp_op] = remove_with_action(tmp_op,tf_action_info,tf_running_activations)
                tf_slice_optimize[tmp_op] = optimize_all_affected_with_indices(
                    pool_loss, tf_indices,
                    tmp_op, weights[tmp_op], biases[tmp_op], tf_indices_size
                )
            elif 'fulcon' in tmp_op:
                tf_update_hyp_ops[tmp_op] = update_tf_hyperparameters(tmp_op, tf_weight_shape, tf_in_size)

        if research_parameters['optimize_end_to_end']:
            optimize_with_variable, upd_lr = optimize_func(pool_loss, global_step, start_lr)
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

        convolution_op_ids = []
        for op_i, op in enumerate(cnn_ops):
            if 'conv' in op:
                convolution_op_ids.append(op_i)
        current_q_learn_op_id = 0
        logger.info('Convolutional Op IDs: %s',convolution_op_ids)

        logger.info('Starting Training Phase')
        #TODO: Think about decaying learning rate (should or shouldn't)

        logger.info('Dataset Size: %d',dataset_size)
        logger.info('Batch Size: %d',batch_size)
        logger.info('Chunk Size: %d',chunk_size)
        logger.info('Batches in Chunk : %d',batches_in_chunk)

        previous_loss = 1e5 # used for the check to start adapting
        start_adapting = False
        for batch_id in range(ceil(dataset_size//batch_size)- batches_in_chunk + 1):

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
                train_dataset,train_labels = load_data_from_memmap(dataset_info,dataset_filename,label_filename,memmap_idx,chunk_size+batch_size)
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
            #print(cnt)
            #print(batch_weights)
            #print(batch_labels_int)
            feed_dict = {tf_dataset : batch_data, tf_labels : batch_labels, tf_data_weights:batch_weights}

            t0_train = time.clock()
            for _ in range(iterations_per_batch):
                _, activation_means, l,l_vec, _,updated_lr, predictions = session.run(
                        [logits,act_means,loss,loss_vec,optimize,upd_lr,pred], feed_dict=feed_dict
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
            for op,op_activations in activation_means.items():
                rolling_ativation_means[op]=(1-act_decay)*rolling_ativation_means[op] + decay*activation_means[op]

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

            if batch_id%interval_parameters['test_interval']==0:
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
                                  batch_id,mean_train_loss,prev_valid_loss,next_valid_loss,
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

                    filter_dict = {}
                    for op_i,op in enumerate(cnn_ops):
                        if 'conv' in op:
                            filter_dict[op_i]=cnn_hyperparameters[op]['weights'][3]
                        elif 'pool' in op:
                            filter_dict[op_i]=0

                    current_state,current_action = adapter.output_action({'distMSE':distMSE,'filter_counts':filter_dict},convolution_op_ids[current_q_learn_op_id])
                    current_q_learn_op_id = (current_q_learn_op_id + 1) % len(
                        convolution_op_ids)  # we update each convolutional layer in a circular pattern

                    logger.info('Got state: %s, action: %s',str(current_state),str(current_action))
                    state_action_history[batch_id]={'states':current_state,'actions':current_action}

                    # where all magic happens (adding and removing filters)
                    si,ai = current_state,current_action
                    current_op = cnn_ops[si[0]]
                    if 'conv' in current_op and ai[0]=='add' :

                        _ = session.run(tf_add_filters_ops[current_op],
                                        feed_dict={
                                            tf_action_info:np.asarray([si[0],1,ai[1]]),
                                            tf_running_activations:rolling_ativation_means[current_op]
                                        })

                        amount_to_add = ai[1]
                        for tmp_op in reversed(cnn_ops):
                            if 'conv' in tmp_op:
                                last_conv_id = tmp_op
                                break

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
                            cnn_hyperparameters['fulcon_out']['in']+=amount_to_add

                            if research_parameters['debugging']:
                                logger.debug('\tNew %s in: %d','fulcon_out',cnn_hyperparameters['fulcon_out']['in'])
                                logger.debug('\tSummary of changes to weights of fulcon_out')
                                logger.debug('\t\tNew Weights: %s', str(tf.shape(weights['fulcon_out']).eval()))

                            session.run(tf_update_hyp_ops['fulcon_out'],feed_dict={
                                tf_in_size:cnn_hyperparameters['fulcon_out']['in']
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
                        for pool_id in range((hard_pool.get_size() // batch_size) - 1):
                            if np.random.random() < research_parameters['replace_op_train_rate']:
                                pbatch_data = pool_dataset[pool_id * batch_size:(pool_id + 1) * batch_size, :, :, :]
                                pbatch_labels = pool_labels[pool_id * batch_size:(pool_id + 1) * batch_size, :]
                                _ = session.run([pool_logits,pool_loss],feed_dict= {tf_pool_dataset: pbatch_data,
                                                                        tf_pool_labels: pbatch_labels}
                                                )
                                _ = session.run(tf_slice_optimize[current_op],
                                                feed_dict={tf_pool_dataset: pbatch_data,
                                                           tf_pool_labels: pbatch_labels,
                                                           tf_indices:np.arange(cnn_hyperparameters[current_op]['weights'][3]-ai[1],cnn_hyperparameters[current_op]['weights'][3]),
                                                           tf_indices_size:ai[1]}
                                                )

                    elif 'conv' in current_op and ai[0]=='remove' :

                        _,rm_indices = session.run(tf_rm_filters_ops[current_op],
                                        feed_dict={
                                            tf_action_info: np.asarray([si[0], 0, ai[1]]),
                                            tf_running_activations: rolling_ativation_means[current_op]
                                        })
                        rm_indices = rm_indices.flatten()
                        amount_to_rmv = ai[1]

                        if research_parameters['remove_filters_by'] == 'Activation':
                            raise NotImplementedError

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
                            cnn_hyperparameters['fulcon_out']['in'] -= amount_to_rmv
                            if research_parameters['debugging']:
                                logger.debug('\tSize after feature map reduction: fulcon_out,%s',
                                             str(tf.shape(weights['fulcon_out']).eval()))

                            session.run(tf_update_hyp_ops['fulcon_out'], feed_dict={
                                tf_in_size: cnn_hyperparameters['fulcon_out']['in']
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

                    elif 'conv' in current_op and (ai[0]=='replace'):

                        if research_parameters['remove_filters_by'] == 'Activation':
                            layer_activations = rolling_ativation_means[current_op]
                            indices_of_filters_replace = (np.argsort(layer_activations).flatten()[:ai[1]]).astype(
                                'int32')
                        elif research_parameters['remove_filters_by'] == 'Distance':
                            # calculate cosine distance for F filters (FxF matrix)
                            # take one side of diagonal, find (f1,f2) pairs with least distance
                            # select indices amnt f2 indices
                            raise NotImplementedError

                    elif 'conv' in current_op and ai[0]=='finetune':
                        # pooling takes place here
                        #pool_logits, _ = get_logits_with_ops(tf_pool_dataset, cnn_ops, cnn_hyperparameters, weights, biases,use_dropout)
                        #pool_loss = calc_loss(pool_logits, tf_pool_labels, False, None)

                        op = cnn_ops[si[0]]
                        pool_dataset,pool_labels = hard_pool.get_pool_data()['pool_dataset'],hard_pool.get_pool_data()['pool_labels']

                        logger.debug('Only tuning following variables...')
                        logger.debug('\t%s,%s',weights[op].name,str(weights[op]))
                        logger.debug('\t%s,%s',biases[op].name,str(biases[op]))

                        assert weights[op].name in [v.name for v in tf.trainable_variables()]
                        assert biases[op].name in [v.name for v in tf.trainable_variables()]

                        for pool_id in range((hard_pool.get_size()//batch_size)-1):
                            pbatch_data = pool_dataset[pool_id*batch_size:(pool_id+1)*batch_size, :, :, :]
                            pbatch_labels = pool_labels[pool_id*batch_size:(pool_id+1)*batch_size, :]
                            pool_feed_dict = {tf_pool_dataset:pbatch_data,tf_pool_labels:pbatch_labels}
                            if research_parameters['optimizer']=='SGD':
                                _, _, _ = session.run([pool_logits,pool_loss,optimize_with_variable],feed_dict=pool_feed_dict)

                    cnn_structure_logger.info(
                        '%d:%s:%s%s', batch_id, current_state,current_action,get_cnn_string_from_ops(cnn_ops, cnn_hyperparameters)
                    )

                    # updating the policy
                    if current_state is not None and current_action is not None:

                        logger.info('Calculating Valid accuracy Gains')
                        feed_valid_dict = {tf_valid_dataset: batch_valid_data, tf_valid_labels: batch_valid_labels}
                        #_ = session.run(tf_valid_logits,feed_dict=feed_valid_dict)
                        next_valid_predictions = session.run(tf_valid_predictions,
                                                             feed_dict=feed_valid_dict)
                        next_valid_accuracy_after = accuracy(next_valid_predictions, batch_valid_labels)

                        batch_valid_data = train_dataset[
                                           prev_valid_batch_id * batch_size:(prev_valid_batch_id + 1) * batch_size, :,
                                           :, :]
                        batch_valid_labels = train_labels[
                                             prev_valid_batch_id * batch_size:(prev_valid_batch_id + 1) * batch_size, :]

                        feed_valid_dict = {tf_valid_dataset: batch_valid_data, tf_valid_labels: batch_valid_labels}
                        #_ = session.run(tf_valid_logits, feed_dict=feed_valid_dict)
                        prev_valid_predictions = session.run(tf_valid_predictions,feed_dict=feed_valid_dict)

                        prev_valid_accuracy_after = accuracy(prev_valid_predictions, batch_valid_labels)

                        logger.info('Updating the Policy for Layer %s', cnn_ops[current_state[0]])
                        logger.info('\tState: %s', str(current_state))
                        logger.info('\tAction: %s', str(current_action))
                        logger.info('\tValid accuracy Gain: %.2f (Unseen) %.2f (seen)',
                                    (next_valid_accuracy_after - next_valid_accuracy),
                                    (prev_valid_accuracy_after-prev_valid_accuracy)
                                    )
                        adapter.update_policy({'states': current_state, 'actions': current_action,
                                               'next_accuracy': next_valid_accuracy_after - next_valid_accuracy,
                                               'prev_accuracy': prev_valid_accuracy_after-prev_valid_accuracy})

                    #reset rolling activation means because network changed
                    rolling_ativation_means = {}
                    for op in cnn_ops:
                        if 'conv' in op:
                            logger.debug("Resetting activation means to zero for %s with %d",op,cnn_hyperparameters[op]['weights'][3])
                            rolling_ativation_means[op]=np.zeros([cnn_hyperparameters[op]['weights'][3]])

                    logger.debug('Resetting both data distribution means')

                    prev_mean_distribution = mean_data_distribution
                    for li in range(num_labels):
                        rolling_data_distribution[li]=0.0
                        mean_data_distribution[li]=0.0

                if batch_id>0 and batch_id%interval_parameters['history_dump_interval']==0:
                    with open(output_dir + os.sep + 'state_actions_' + str(batch_id)+'.pickle', 'wb') as f:
                        pickle.dump(state_action_history, f, pickle.HIGHEST_PROTOCOL)
                        state_action_history = {}

                    with open(output_dir + os.sep + 'Q_' + str(batch_id)+'.pickle', 'wb') as f:
                        pickle.dump(adapter.get_Q(), f, pickle.HIGHEST_PROTOCOL)

                    pool_dist_string = ''
                    for val in hard_pool.get_class_distribution():
                        pool_dist_string += str(val)+','

                    pool_dist_logger.info(pool_dist_string)

            t1 = time.clock()
            op_count = len(graph.get_operations())
            var_count = len(tf.global_variables()) + len(tf.local_variables()) + len(tf.model_variables())
            perf_logger.info('%d,%.5f,%.5f,%d,%d',batch_id,t1-t0,t1_train-t0_train,op_count,var_count)
