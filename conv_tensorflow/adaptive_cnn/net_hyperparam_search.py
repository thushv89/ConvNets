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
import queue
from multiprocessing import Pool as MPPool


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
# stationary 0.1 non-stationary 0.01
start_lr = 0.01
min_learning_rate = 0.0001
decay_learning_rate = True
decay_rate = 0.5
dropout_rate = 0.25
use_dropout = True
use_loc_res_norm = True
#keep beta small (0.2 is too much >0.002 seems to be fine)
include_l2_loss = True
beta = 1e-5
check_early_stopping_from = 5
accuracy_drop_cap = 3
iterations_per_batch = 1
epochs = 1
final_2d_width = 3

lrn_radius = 4
lrn_alpha = 0.001/9.0
lrn_beta = 0.75

TOWER_NAME = 'tower'

TF_LOSS_VEC_STR = 'loss_vector'

TF_WEIGHTS,TF_BIAS = 'weights','bias'
TF_ACTIVAIONS_STR = 'Activations'
TF_CONV_WEIGHT_SHAPE_STR = 'ShapeWeights'
TF_FC_WEIGHT_IN_STR = 'InWeight'
TF_FC_WEIGHT_OUT_STR = 'OutWeight'

TF_TRAIN_MOMENTUM = 'TrainMomentum'
TF_POOL_MOMENTUM = 'PoolMomentum'

TF_SCOPE_DIVIDER = '/'
TF_ADAPTATION_NAME_SCOPE = 'Adapt'
TF_GLOBAL_SCOPE = 'AdaCNN'


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def initialize_cnn_with_ops(cnn_ops, cnn_hyps):
    var_list = []
    logger.info('CNN Hyperparameters')
    logger.info('%s\n',cnn_hyps)

    logger.info('Initializing the iConvNet (conv_global,pool_global,classifier)...\n')
    for op in cnn_ops:

        if 'conv' in op:
            with tf.variable_scope(op) as scope:
                var_list.append(tf.get_variable(
                    name=TF_WEIGHTS,
                    initializer=tf.truncated_normal(cnn_hyps[op]['weights'],
                                                    stddev=2./max(100,cnn_hyps[op]['weights'][0]*cnn_hyps[op]['weights'][1])
                                                    ),
                    validate_shape = False, dtype=tf.float32))
                var_list.append(tf.get_variable(
                    name=TF_BIAS,
                    initializer = tf.constant(np.random.random()*0.001,shape=[cnn_hyps[op]['weights'][3]]),
                    validate_shape = False, dtype=tf.float32))

                var_list.append(tf.get_variable(
                    name=TF_ACTIVAIONS_STR,
                    initializer=tf.zeros(shape=[cnn_hyps[op]['weights'][3]],
                                         dtype=tf.float32, name = scope.name+'_activations'),
                    validate_shape=False,trainable=False))

                logger.debug('Weights for %s initialized with size %s',op,str(cnn_hyps[op]['weights']))
                logger.debug('Biases for %s initialized with size %d',op,cnn_hyps[op]['weights'][3])

        if 'fulcon' in op:
            with tf.variable_scope(op) as scope:
                var_list.append(tf.get_variable(
                    name=TF_WEIGHTS,
                    initializer = tf.truncated_normal([cnn_hyps[op]['in'],cnn_hyps[op]['out']],
                                                     stddev=2./cnn_hyps[op]['in']),
                    validate_shape = False, dtype=tf.float32))
                var_list.append(tf.get_variable(
                    name=TF_BIAS,
                    initializer=tf.constant(np.random.random()*0.001,shape=[cnn_hyps[op]['out']]),
                    validate_shape = False, dtype=tf.float32))

                logger.debug('Weights for %s initialized with size %d,%d',
                    op,cnn_hyps[op]['in'],cnn_hyps[op]['out'])
                logger.debug('Biases for %s initialized with size %d',op,cnn_hyps[op]['out'])

    return var_list


def inference(dataset,tf_cnn_hyperparameters,training):
    global cnn_ops

    first_fc = 'fulcon_out' if 'fulcon_0' not in cnn_ops else 'fulcon_0'

    last_conv_id = ''
    for op in cnn_ops:
        if 'conv' in op:
            last_conv_id = op

    logger.debug('Defining the logit calculation ...')
    logger.debug('\tCurrent set of operations: %s'%cnn_ops)
    activation_ops = []

    x = dataset
    if research_parameters['whiten_images']:
        mu,var = tf.nn.moments(x,axes=[1,2,3])
        tr_x = tf.transpose(x,[1,2,3,0])
        tr_x = (tr_x-mu)/tf.maximum(tf.sqrt(var),1.0/(image_size*image_size*num_channels))
        x = tf.transpose(tr_x,[3,0,1,2])

    logger.debug('\tReceived data for X(%s)...'%x.get_shape().as_list())

    #need to calculate the output according to the layers we have
    for op in cnn_ops:
        if 'conv' in op:
            with tf.variable_scope(op,reuse=True) as scope:
                logger.debug('\tConvolving (%s) With Weights:%s Stride:%s'%(op,cnn_hyperparameters[op]['weights'],cnn_hyperparameters[op]['stride']))
                logger.debug('\t\tWeights: %s',tf.shape(tf.get_variable(TF_WEIGHTS)).eval())
                w,b = tf.get_variable(TF_WEIGHTS), tf.get_variable(TF_BIAS)

                x = tf.nn.conv2d(x, w, cnn_hyperparameters[op]['stride'],
                                 padding=cnn_hyperparameters[op]['padding'])
                x = lrelu(x + b, name=scope.name+'/top')

                activation_ops.append(tf.assign(tf.get_variable(TF_ACTIVAIONS_STR),tf.reduce_max(x,[0,1,2]),validate_shape=False))

                if use_loc_res_norm and op==last_conv_id:
                    x = tf.nn.local_response_normalization(x,depth_radius=lrn_radius,alpha=lrn_alpha, beta=lrn_beta) # hyperparameters from tensorflow cifar10 tutorial

        if 'pool' in op:
            logger.debug('\tPooling (%s) with Kernel:%s Stride:%s'%(op,cnn_hyperparameters[op]['kernel'],cnn_hyperparameters[op]['stride']))
            if cnn_hyperparameters[op]['type'] is 'max':
                x = tf.nn.max_pool(x,ksize=cnn_hyperparameters[op]['kernel'],
                                   strides=cnn_hyperparameters[op]['stride'],
                                   padding=cnn_hyperparameters[op]['padding'])
            elif cnn_hyperparameters[op]['type'] is 'avg':
                x = tf.nn.avg_pool(x,ksize=cnn_hyperparameters[op]['kernel'],
                                   strides=cnn_hyperparameters[op]['stride'],
                                   padding=cnn_hyperparameters[op]['padding'])
            if use_loc_res_norm and 'pool_global'!= op:
                x = tf.nn.local_response_normalization(x,depth_radius=lrn_radius,alpha=lrn_alpha, beta=lrn_beta)

        if 'fulcon' in op:
            with tf.variable_scope(op,reuse=True) as scope:
                w,b = tf.get_variable(TF_WEIGHTS),tf.get_variable(TF_BIAS)

                if first_fc==op:
                    # we need to reshape the output of last subsampling layer to
                    # convert 4D output to a 2D input to the hidden layer
                    # e.g subsample layer output [batch_size,width,height,depth] -> [batch_size,width*height*depth]

                    logger.debug('Input size of fulcon_out : %d', cnn_hyperparameters[op]['in'])
                    # Transpose x (b,h,w,d) to (b,d,w,h)
                    # This help us to do adaptations more easily
                    x = tf.transpose(x,[0,3,1,2])
                    x = tf.reshape(x, [batch_size, tf_cnn_hyperparameters[op][TF_FC_WEIGHT_IN_STR]])
                    x = lrelu(tf.matmul(x, w)+b,name=scope.name+'/top')
                    if training and use_dropout:
                        x = tf.nn.dropout(x,keep_prob= 1.0-dropout_rate,name='dropout')

                elif 'fulcon_out' == op:
                    x = tf.matmul(x, w)+ b

                else:
                    x = lrelu(tf.matmul(x, w)+ b,name = scope.name+'/top')
                    if training and use_dropout:
                        x = tf.nn.dropout(x,keep_prob= 1.0-dropout_rate,name='dropout')

    return x,activation_ops


def tower_loss(dataset,labels,weighted,tf_data_weights, tf_cnn_hyperparameters):
    global cnn_ops
    logits,_ = inference(dataset,tf_cnn_hyperparameters,True)
    # use weighted loss
    if weighted:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels) * tf_data_weights)
    else:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))

    if include_l2_loss:
        fulcons = []
        for op in cnn_ops:
            if 'fulcon' in op and op != 'fulcon_out':
                fulcons.append(op)
        fc_weights = []
        for op in fulcons:
            with tf.variable_scope(op):
                fc_weights.append(tf.get_variable(TF_WEIGHTS))

        loss = tf.reduce_sum([loss,beta*tf.reduce_sum([tf.nn.l2_loss(w) for w in fc_weights])])

    total_loss = loss

    return total_loss


def calc_loss_vector(scope,dataset,labels,tf_cnn_hyperparameters):
    logits,_ = inference(dataset,tf_cnn_hyperparameters,True)
    return tf.nn.softmax_cross_entropy_with_logits(logits, labels,name=TF_LOSS_VEC_STR)


def gradients(optimizer, loss, global_step, learning_rate):

    # grad_and_vars [(grads_w,w),(grads_b,b)]
    grad_and_vars = optimizer.compute_gradients(loss, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=TF_GLOBAL_SCOPE))
    return grad_and_vars


def update_train_momentum_velocity(grads_and_vars):
    vel_update_ops = []
    # update velocity vector

    for (g,v) in grads_and_vars:
        var_name = v.name.split(':')[0]

        with tf.variable_scope(var_name,reuse=True) as scope:
            vel = tf.get_variable(TF_TRAIN_MOMENTUM)

            vel_update_ops.append(
                tf.assign(vel,
                          research_parameters['momentum']*vel + g)
            )

    return vel_update_ops



def apply_gradient_with_momentum(optimizer,learning_rate,global_step):
    grads_and_vars = []
    # for each trainable variable
    if decay_learning_rate:
        learning_rate = tf.maximum(min_learning_rate,tf.train.exponential_decay(learning_rate,global_step,decay_steps=1,decay_rate=decay_rate,staircase=True))
    for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=TF_GLOBAL_SCOPE):
        with tf.variable_scope(v.name.split(':')[0],reuse=True):
            vel = tf.get_variable(TF_TRAIN_MOMENTUM)
            grads_and_vars.append((vel*learning_rate,v))

    return optimizer.apply_gradients(grads_and_vars)

def apply_gradient_with_pool_momentum(optimizer,learning_rate,global_step):
    grads_and_vars = []
    # for each trainable variable
    if decay_learning_rate:
        learning_rate = tf.maximum(min_learning_rate,tf.train.exponential_decay(learning_rate,global_step,decay_steps=1,decay_rate=decay_rate,staircase=True))
    for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=TF_GLOBAL_SCOPE):
        with tf.variable_scope(v.name.split(':')[0],reuse=True):
            vel = tf.get_variable(TF_POOL_MOMENTUM)
            grads_and_vars.append((vel*learning_rate,v))

    return optimizer.apply_gradients(grads_and_vars)

def optimize_with_momentum(optimizer, loss, global_step, learning_rate):
    vel_update_ops, grad_update_ops = [],[]

    if decay_learning_rate:
        learning_rate = tf.maximum(min_learning_rate,tf.train.exponential_decay(learning_rate,global_step,decay_steps=1,decay_rate=decay_rate,staircase=True))

    for op in cnn_ops:
        if 'conv' in op and 'fulcon' in op:
            with tf.variable_scope(op) as scope:
                w,b = tf.get_variable(TF_WEIGHTS),tf.get_variable(TF_BIAS)
                [(grads_w,w),(grads_b,b)] = optimizer.compute_gradients(loss, [w,b])

                # update velocity vector
                with tf.variable_scope(TF_WEIGHTS) as child_scope:
                    w_vel = tf.get_variable(TF_POOL_MOMENTUM)
                    vel_update_ops.append(
                        tf.assign(w_vel,
                                  research_parameters['momentum']*w_vel + grads_w)
                    )
                with tf.variable_scope(TF_BIAS) as child_scope:
                    b_vel = tf.get_variable(TF_POOL_MOMENTUM)
                    vel_update_ops.append(
                        tf.assign(b_vel,
                                  research_parameters['momentum']*b_vel + grads_b)
                    )
                grad_update_ops.append([(w_vel*learning_rate,w),(b_vel*learning_rate,b)])

    return grad_update_ops,vel_update_ops


def average_gradients(tower_grads):
    # tower_grads => [((grads0gpu0,var0gpu0),...,(grads0gpuN,var0gpuN)),((grads1gpu0,var1gpu0),...,(grads1gpuN,var1gpuN))]
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, v in grad_and_vars:
            expanded_g = tf.expand_dims(g,0)
            grads.append(expanded_g)
        grad = tf.concat(0,values=grads)
        grad = tf.reduce_mean(grad,axis=0)
        grad_and_var = (grad,v)
        average_grads.append(grad_and_var)

    return average_grads


def concat_loss_vector_towers(tower_loss_vectors):
    concat_loss_vec = None
    for loss_vec in tower_loss_vectors:
        if concat_loss_vec is None:
            concat_loss_vec = tf.identity(loss_vec)
        else:
            concat_loss_vec = tf.concat(1,loss_vec)

    return concat_loss_vec


def mean_tower_activations(tower_activations):
    mean_activations = []
    for a_i in zip(*tower_activations):
        stacked_activations = None
        for a in a_i:
            if stacked_activations is None:
                stacked_activations = tf.identity(a)
            else:
                stacked_activations = tf.stack([stacked_activations,a],axis=0)

        mean_activations.append(tf.reduce_mean(stacked_activations,[0]))
    return mean_activations


def inc_global_step(global_step):
    return global_step+1


def predict_with_logits(logits):
    # Predictions for the training, validation, and test data.
    prediction = tf.nn.softmax(logits)
    return prediction


def predict_with_dataset(dataset,tf_cnn_hyperparameters):
    logits,_ = inference(dataset,tf_cnn_hyperparameters,False)
    prediction = tf.nn.softmax(logits)
    return prediction


def accuracy(predictions, labels):
    assert predictions.shape[0]==labels.shape[0]
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])

tf_distort_data_batch,distorted_imgs = None,None

def load_data_from_memmap(dataset_info,dataset_filename,label_filename,start_idx,size,randomize):
    global logger,tf_distort_data_batch,distorted_imgs

    num_labels = dataset_info['num_labels']
    col_count = (dataset_info['image_w'],dataset_info['image_w'],dataset_info['num_channels'])
    print(col_count)
    logger.info('Processing files %s,%s'%(dataset_filename,label_filename))
    fp1 = np.memmap(dataset_filename,dtype=np.float32,mode='r',
                    offset=np.dtype('float32').itemsize*col_count[0]*col_count[1]*col_count[2]*start_idx,shape=(size,col_count[0],col_count[1],col_count[2]))

    fp2 = np.memmap(label_filename,dtype=np.int32,mode='r',
                    offset=np.dtype('int32').itemsize*1*start_idx,shape=(size,1))

    train_dataset = fp1[:,:,:,:]
    rand_train_dataset = None
    if randomize:
        try:
            pool = MPPool(processes=10)
            distorted_imgs = pool.map(distort_img,train_dataset)
            train_dataset = np.asarray(distorted_imgs)
            pool.close()
            pool.join()
        except Exception:
           raise AssertionError

        #if tf_distort_data_batch is None:
        #    tf_distort_data_batch = tf.placeholder(shape=[batch_size,image_size,image_size,num_channels],dtype=tf.float32)
        #with tf.device('/gpu:0'):
        #    if distorted_imgs is None:
        #        distorted_imgs = distort_img_with_tf(tf_distort_data_batch)
        #    for dist_i in range((size//batch_size)):
        #        dist_databatch = session.run(distorted_imgs,feed_dict={tf_distort_data_batch:train_dataset[dist_i*batch_size:(dist_i+1)*batch_size,:,:,:]})
        #        if rand_train_dataset is None:
        #            rand_train_dataset = dist_databatch
        #        else:
        #            rand_train_dataset = np.append(rand_train_dataset,dist_databatch,axis=0)

        #assert train_dataset.shape[0]==rand_train_dataset.shape[0]
        #train_dataset = rand_train_dataset

    # labels is nx1 shape
    train_labels = fp2[:]

    train_ohe_labels = (np.arange(num_labels) == train_labels[:]).astype(np.float32)
    del fp1,fp2

    assert np.all(np.argmax(train_ohe_labels[:10],axis=1).flatten()==train_labels[:10].flatten())
    return train_dataset,train_ohe_labels

def distort_img(img):
    if np.random.random()<0.4:
        img = np.fliplr(img)
    if np.random.random()<0.4:
        brightness = np.random.random()*1.5 - 0.6
        img += brightness
    if np.random.random()<0.4:
        contrast = np.random.random()*0.8 + 0.4
        img *= contrast

    return img


def init_tf_hyperparameters():
    global cnn_ops, cnn_hyperparameters
    tf_hyp_list = {}
    for op in cnn_ops:
        if 'conv' in op:
            with tf.variable_scope(op) as scope:
                tf_hyp_list[op]={TF_CONV_WEIGHT_SHAPE_STR:
                    tf.get_variable(name=TF_CONV_WEIGHT_SHAPE_STR,
                                    initializer=cnn_hyperparameters[op]['weights'],
                                    dtype=tf.int32,trainable=False)
                }

        if 'fulcon' in op:
            with tf.variable_scope(op) as scope:
                tf_hyp_list[op]={TF_FC_WEIGHT_IN_STR:tf.get_variable(name=TF_FC_WEIGHT_IN_STR,
                                          initializer=cnn_hyperparameters[op]['in'],
                                          dtype=tf.int32,trainable=False)
                                 ,TF_FC_WEIGHT_OUT_STR: tf.get_variable(name=TF_FC_WEIGHT_OUT_STR,
                                           initializer=cnn_hyperparameters[op]['out'],
                                           dtype=tf.int32,trainable=False)
                                 }
    return tf_hyp_list


def update_tf_hyperparameters(op,tf_weight_shape,tf_in_size):
    global cnn_ops, cnn_hyperparameters
    update_ops = []
    if 'conv' in op:
        with tf.variable_scope(op,reuse=True):
            update_ops.append(tf.assign(tf.get_variable(TF_CONV_WEIGHT_SHAPE_STR,dtype=tf.int32),tf_weight_shape))
    if 'fulcon' in op:
        with tf.variable_scope(op,reuse=True):
            update_ops.append(tf.assign(tf.get_variable(TF_FC_WEIGHT_IN_STR,dtype=tf.int32),tf_in_size))

    return update_ops


def define_velocity_vectors(main_scope):
    # if using momentum
        vel_var_list = []
        if research_parameters['optimizer']=='Momentum':
            for tmp_op in cnn_ops:
                op_scope = tmp_op
                if 'conv' in tmp_op:
                    for v in  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=main_scope.name+TF_SCOPE_DIVIDER+op_scope):
                        if TF_WEIGHTS in v.name:
                            with tf.variable_scope(op_scope+TF_SCOPE_DIVIDER+TF_WEIGHTS) as scope:
                                vel_var_list.append(
                                    tf.get_variable(name=TF_TRAIN_MOMENTUM,
                                                    initializer=tf.zeros(shape=cnn_hyperparameters[tmp_op]['weights'], dtype=tf.float32),
                                                    dtype=tf.float32,trainable=False))
                                vel_var_list.append(
                                        tf.get_variable(name=TF_POOL_MOMENTUM,
                                                        initializer=tf.zeros(shape=cnn_hyperparameters[tmp_op]['weights'], dtype=tf.float32),
                                                        dtype=tf.float32,trainable=False))
                        elif TF_BIAS in v.name:
                            with tf.variable_scope(op_scope+TF_SCOPE_DIVIDER+TF_BIAS) as scope:
                                vel_var_list.append(tf.get_variable(name=TF_TRAIN_MOMENTUM,
                                    initializer =tf.zeros(shape=cnn_hyperparameters[tmp_op]['weights'][3], dtype=tf.float32),
                                    dtype=tf.float32,trainable=False))

                                vel_var_list.append(tf.get_variable(name=TF_POOL_MOMENTUM,
                                    initializer= tf.zeros(shape=[cnn_hyperparameters[tmp_op]['weights'][3]], dtype=tf.float32),
                                    dtype=tf.float32,trainable=False))

                elif 'fulcon' in tmp_op:
                    for v in  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=main_scope.name+TF_SCOPE_DIVIDER+op_scope):
                        if TF_WEIGHTS in v.name:
                            with tf.variable_scope(op_scope+TF_SCOPE_DIVIDER+TF_WEIGHTS) as scope:
                                vel_var_list.append(tf.get_variable(name=TF_TRAIN_MOMENTUM,
                                    initializer=tf.zeros(shape=[cnn_hyperparameters[tmp_op]['in'], cnn_hyperparameters[tmp_op]['out']], dtype=tf.float32),
                                    dtype=tf.float32,trainable=False))
                                vel_var_list.append(tf.get_variable(name=TF_POOL_MOMENTUM,
                                                                    initializer=tf.zeros(shape=[cnn_hyperparameters[tmp_op]['in'], cnn_hyperparameters[tmp_op]['out']], dtype=tf.float32),
                                                                    dtype=tf.float32,trainable=False))
                        elif TF_BIAS in v.name:
                            with tf.variable_scope(op_scope+TF_SCOPE_DIVIDER+TF_BIAS) as scope:
                                vel_var_list.append(tf.get_variable(name=TF_TRAIN_MOMENTUM,
                                    initializer=tf.zeros(shape=[cnn_hyperparameters[tmp_op]['out']], dtype=tf.float32),
                                    dtype=tf.float32,trainable=False))

                                vel_var_list.append(tf.get_variable(name=TF_POOL_MOMENTUM,
                                    initializer=tf.zeros(shape=[cnn_hyperparameters[tmp_op]['out']], dtype=tf.float32),
                                    dtype=tf.float32,trainable=False))

        return vel_var_list

def get_activation_dictionary(activation_list,cnn_ops,conv_op_ids):
    current_activations = {}
    for act_i, layer_act in enumerate(current_activations_list):
        current_activations[cnn_ops[conv_op_ids[act_i]]] = layer_act
    return current_activations

research_parameters = {
    'save_train_test_images':False,
    'log_class_distribution':True,'log_distribution_every':128,
    'adapt_structure' : False,
    'hard_pool_acceptance_rate':0.1,
    'replace_op_train_rate':0.8, # amount of batches from hard_pool selected to train
    'optimizer':'Momentum','momentum':0.9,'pool_momentum':0.0,
    'use_custom_momentum_opt':True,
    'remove_filters_by':'Activation',
    'optimize_end_to_end':True, # if true functions such as add and finetune will optimize the network from starting layer to end (fulcon_out)
    'loss_diff_threshold':0.02,
    'debugging':True if logging_level==logging.DEBUG else False,
    'stop_training_at':11000,
    'train_min_activation':False,
    'use_weighted_loss':True,
    'whiten_images':True,
    'finetune_rate': 0.5,
    'pool_randomize':True,
    'pool_randomize_rate':0.25,
}

interval_parameters = {
    'history_dump_interval':500,
    'policy_interval' : 20, #number of batches to process for each policy iteration
    'test_interval' : 100
}

state_action_history = []

cnn_ops, cnn_hyperparameters = None,None
num_gpus = -1

if __name__=='__main__':
    global cnn_ops,cnn_hyperparameters

    try:
        opts,args = getopt.getopt(
            sys.argv[1:],"",["output_dir=","num_gpus="])
    except getopt.GetoptError as err:
        print('<filename>.py --output_dir= --num_gpus=')

    if len(opts)!=0:
        for opt,arg in opts:
            if opt == '--output_dir':
                output_dir = arg
            if opt == '--num_gpus':
                num_gpus = int(arg)

    assert interval_parameters['test_interval']%num_gpus==0

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #type of data training
    datatype = 'cifar-10'
    behavior = 'non-stationary'

    if behavior=='non-stationary':
        start_lr = 0.01
    if research_parameters['adapt_structure']:
        use_dropout = False

    dataset_info = {'dataset_type':datatype,'behavior':behavior}
    dataset_filename,label_filename = None,None
    test_dataset,test_labels = None,None

    if datatype=='cifar-10':
        image_size = 24
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

        pool_size = batch_size * 10 * num_labels
        test_size=10000
        test_dataset_filename='data_non_station'+os.sep+'cifar-10-test-dataset.pkl'
        test_label_filename = 'data_non_station'+os.sep+'cifar-10-test-labels.pkl'

        if not research_parameters['adapt_structure']:
            cnn_string = "C,5,1,64#P,3,2,0#C,5,1,128#P,3,2,0#C,3,1,256#Terminate,0,0,0"
        else:
            cnn_string = "C,5,1,32#P,3,2,0#C,5,1,32#P,3,2,0#C,3,1,32#Terminate,0,0,0"
            filter_upper_bound, filter_lower_bound = 256, 64

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

        pool_size = batch_size * 10 * num_labels
        test_size = 26032
        test_dataset_filename = 'data_non_station' + os.sep + 'svhn-10-nonstation-test-dataset.pkl'
        test_label_filename = 'data_non_station' + os.sep + 'svhn-10-nonstation-test-labels.pkl'

        if not research_parameters['adapt_structure']:
            cnn_string = "C,5,1,64#P,3,2,0#C,5,1,128#Terminate,0,0,0"
        else:
            cnn_string = "C,5,1,32#P,3,2,0#C,5,1,32#Terminate,0,0,0"
            filter_upper_bound, filter_lower_bound = 128, 64

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

    # Loading test data
    train_dataset,train_labels = None,None

    test_dataset,test_labels = load_data_from_memmap(dataset_info,test_dataset_filename,test_label_filename,0,test_size,False)

    assert chunk_size%batch_size==0
    batches_in_chunk = chunk_size//batch_size

    logger.info('='*80)
    logger.info('\tTrain Size: %d'%dataset_size)
    logger.info('='*80)

    #cnn_string = "C,5,1,256#P,3,2,0#C,5,1,512#C,3,1,128#FC,2048,0,0#Terminate,0,0,0"

    #cnn_string = "C,3,1,128#P,5,2,0#C,5,1,128#C,3,1,512#C,5,1,128#C,5,1,256#P,2,2,0#C,5,1,64#Terminate,0,0,0"
    #cnn_string = "C,3,4,128#P,5,2,0#Terminate,0,0,0"

    cnn_ops,cnn_hyperparameters = utils.get_ops_hyps_from_string(dataset_info,cnn_string,final_2d_width)

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

    # RL testing parameters
    learning_rates = [0.01,0.1,0.001]
    l2_betas = [None,1e-4,1e-2,1e-3]
    lrn_rads = [None,4,3,5]
    lrn_alphas = [0.0001,0.001,0.00001]
    lrn_betas = [0.75,0.5,0.9]

    # Structure inspired by the paper
    # STRIVING FOR SIMPLICITY: THE ALL CONVOLUTIONAL NET

    best_lr,best_l2beta,best_lrn_rad,best_lrn_alpha,best_lrn_beta = learning_rates[0],l2_betas[0],lrn_rads[0],lrn_alphas[0],lrn_betas[0]

    besthyp_logger = logging.getLogger('best_hyp_logger')
    besthyp_logger.setLevel(logging.INFO)
    bestHandler = logging.FileHandler(output_dir + os.sep + 'best_hyps.log', mode='w')
    bestHandler.setFormatter(logging.Formatter('%(message)s'))
    besthyp_logger.handlers = []
    besthyp_logger.addHandler(bestHandler)
    besthyp_logger.info('#BEST HYPERPARAMETERS')

    glb_max_test_accuracy = 0
    for hp_i,tuning_set in enumerate([learning_rates,l2_betas,lrn_rads,lrn_alphas,lrn_betas]):
        if hp_i==0:
            hyparam='learning_rate'
        elif hp_i==1:
            hyparam='l2_beta'
        elif hp_i==2:
            hyparam='lrn_rads'
        elif hp_i==3:
            hyparam='lrn_alpha'
            if best_lrn_rad is None:
                continue
        elif hp_i==4:
            hyparam='lrn_beta'
            if best_lrn_rad is None:
                continue

        prim_sub_output_dir = os.path.join(output_dir, 'testing_' + hyparam)
        if not os.path.exists(prim_sub_output_dir):
            os.makedirs(prim_sub_output_dir)

        if hyparam=='learning_rate':
            trunc_tuning_set = tuning_set
        elif hyparam=='l2_beta':
            trunc_tuning_set = tuning_set[1:]
        elif hyparam=='lrn_rads':
            trunc_tuning_set = tuning_set[1:]
        else:
            trunc_tuning_set = tuning_set

        for value in trunc_tuning_set:

            sub_dir_string = ''
            if hyparam == 'learning_rate':
                sub_dir_string += 'lr_%.4f' % value
                start_lr = value
            else:
                sub_dir_string += 'lr_%.4f' % best_lr
                start_lr = best_lr

            if hyparam == 'l2_beta':
                if value is None:
                    sub_dir_string += 'l2_NONE'
                    include_l2_loss = False
                else:
                    sub_dir_string += 'l2_%.6f' % value
                    include_l2_loss = True
                    beta = value
            else:
                if best_l2beta is None:
                    sub_dir_string += 'l2_NONE'
                    include_l2_loss = False
                else:
                    sub_dir_string += 'l2_%.6f' % best_l2beta
                    include_l2_loss = True
                    beta = best_l2beta
            if hyparam == 'lrn_rad':
                if value is None:
                    sub_dir_string += 'lrn_rad_NONE'
                    use_loc_res_norm = False
                else:
                    sub_dir_string += 'lrn_rad_%d' % value
                    use_loc_res_norm = True
                    lrn_radius = value
            else:
                if best_lrn_rad is None:
                    sub_dir_string += 'lrn_rad_NONE'
                    use_loc_res_norm = False
                else:
                    sub_dir_string += 'lrn_rad_%d' % best_lrn_rad
                    use_loc_res_norm = True
                    lrn_radius = best_lrn_rad

            if hyparam == 'lrn_alpha':
                sub_dir_string += 'lrn_alp_%.5f' % value
                lrn_alpha = value
            else:
                sub_dir_string += 'lrn_alp_%.5f' % best_lrn_alpha
                lrn_alpha = best_lrn_alpha

            if hyparam == 'lrn_beta':
                sub_dir_string += 'lrn_beta_%.2f' % value
                lrn_beta = value
            else:
                sub_dir_string += 'lrn_beta_%.2f' % best_lrn_beta
                lrn_beta = best_lrn_beta

            sub_output_dir = os.path.join(prim_sub_output_dir, sub_dir_string)
            if not os.path.exists(sub_output_dir):
                os.makedirs(sub_output_dir)

            hyp_logger = logging.getLogger('hyperparameter_logger')
            hyp_logger.setLevel(logging.INFO)
            hypHandler = logging.FileHandler(sub_output_dir + os.sep + 'Hyperparameter.log', mode='w')
            hypHandler.setFormatter(logging.Formatter('%(message)s'))
            hyp_logger.addHandler(hypHandler)
            hyp_logger.info('#Starting hyperparameters')
            hyp_logger.info(cnn_hyperparameters)
            hyp_logger.info('#Dataset info')
            hyp_logger.info(dataset_info)
            hyp_logger.info('Learning rate: %.5f', start_lr)
            hyp_logger.info('Batch size: %d', batch_size)
            hyp_logger.info('L2Loss: %s, %.8f',include_l2_loss,beta)
            hyp_logger.info('LRN: %s, %d (radius), %.5f (alpha), %.5f (beta)',use_loc_res_norm,lrn_radius,lrn_alpha,lrn_beta)

            hyp_logger.info('#Research parameters')
            hyp_logger.info(research_parameters)
            hyp_logger.info('#Interval parameters')
            hyp_logger.info(interval_parameters)

            with tf.Graph().as_default() as graph, tf.device('/cpu:0'):

                hardness = 0.5
                hard_pool = Pool(size=pool_size,batch_size=batch_size,image_size=image_size,num_channels=num_channels,num_labels=num_labels,assert_test=False)

                first_fc = 'fulcon_out' if 'fulcon_0' not in cnn_ops else 'fulcon_0'
                # -1 is because we don't want to count pool_global
                layer_count = len([op for op in cnn_ops if 'conv' in op or 'pool' in op])-1

                # ids of the convolution ops
                convolution_op_ids = []
                for op_i, op in enumerate(cnn_ops):
                    if 'conv' in op:
                        convolution_op_ids.append(op_i)

                # Test data (Global)
                tf_test_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels),name='TestDataset')
                tf_test_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels),name='TestLabels')

                #GLOBAL:
                with tf.variable_scope(TF_GLOBAL_SCOPE) as scope:
                    global_step = tf.get_variable(initializer=0, dtype=tf.int32,trainable=False,name='global_step')
                    tf_cnn_hyperparameters = init_tf_hyperparameters()
                    _ = initialize_cnn_with_ops(cnn_ops,cnn_hyperparameters)
                    _ = define_velocity_vectors(scope)

                logger.debug('CNN_HYPERPARAMETERS 2')
                logger.debug('\t%s\n',tf_cnn_hyperparameters)

                error_logger = logging.getLogger('error_logger')
                error_logger.setLevel(logging.INFO)
                errHandler = logging.FileHandler(sub_output_dir + os.sep + 'Error.log', mode='w')
                errHandler.setFormatter(logging.Formatter('%(message)s'))
                error_logger.handlers = []
                error_logger.addHandler(errHandler)
                error_logger.info('#Batch_ID,Loss(Train),Valid(Unseen),Test Accuracy')

                perf_logger = logging.getLogger('time_logger')
                perf_logger.setLevel(logging.INFO)
                perfHandler = logging.FileHandler(sub_output_dir + os.sep + 'time.log', mode='w')
                perfHandler.setFormatter(logging.Formatter('%(message)s'))
                perf_logger.handlers = []
                perf_logger.addHandler(perfHandler)
                perf_logger.info('#Batch_ID,Time(Full),Time(Train),Op count, Var count')

                if research_parameters['log_class_distribution']:
                    class_dist_logger = logging.getLogger('class_dist_logger')
                    class_dist_logger.setLevel(logging.INFO)
                    class_distHandler = logging.FileHandler(sub_output_dir + os.sep + 'class_distribution.log', mode='w')
                    class_distHandler.setFormatter(logging.Formatter('%(message)s'))
                    class_dist_logger.handlers = []
                    class_dist_logger.addHandler(class_distHandler)

                session = tf.InteractiveSession(config=config)

                # custom momentum optimizing
                # apply_gradient([g,v]) does the following v -= eta*g
                # eta is learning_rate
                # Since what we need is
                # v(t+1) = mu*v(t) - eta*g
                # theta(t+1) = theta(t) + v(t+1) --- (2)
                # we form (2) in the form v(t+1) = mu*v(t) + eta*g
                # theta(t+1) = theta(t) - v(t+1)
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)

                init_op = tf.global_variables_initializer()
                _ = session.run(init_op)

                tf_train_data_batch,tf_train_label_batch,tf_data_weights = [],[],[]
                tf_pool_data_batch,tf_pool_label_batch = [],[]
                tf_valid_data_batch,tf_valid_label_batch = None,None

                # tower_grads will contain
                # [[(grad0gpu0,var0gpu0),...,(gradNgpu0,varNgpu0)],...,[(grad0gpuD,var0gpuD),...,(gradNgpuD,varNgpuD)]]
                tower_grads,tower_loss_vectors,tower_losses,tower_activation_update_ops,tower_predictions = [],[],[],[],[]
                tower_pool_grads,tower_pool_losses,tower_pool_activation_update_ops = [],[],[]
                tower_logits = []

                with tf.variable_scope(TF_GLOBAL_SCOPE):
                    for gpu_id in range(num_gpus):
                        with tf.device('/gpu:%d'%gpu_id):
                            with tf.name_scope('%s_%d' %(TOWER_NAME,gpu_id)) as scope:

                                tf.get_variable_scope().reuse_variables()
                                # Input train data
                                tf_train_data_batch.append(tf.placeholder(tf.float32,
                                                            shape=(batch_size, image_size, image_size, num_channels),
                                                            name='TrainDataset'))
                                tf_train_label_batch.append(tf.placeholder(tf.float32, shape=(batch_size, num_labels), name='TrainLabels'))
                                tf_data_weights.append(tf.placeholder(tf.float32, shape=(batch_size), name='TrainWeights'))

                                # Training data opearations
                                tower_logit_op,tower_tf_activation_ops = inference(tf_train_data_batch[-1],tf_cnn_hyperparameters,True)
                                tower_logits.append(tower_logit_op)
                                tower_activation_update_ops.append(tower_tf_activation_ops)

                                tf_tower_loss = tower_loss(tf_train_data_batch[-1],tf_train_label_batch[-1],True,tf_data_weights[-1],tf_cnn_hyperparameters)
                                print(session.run(tf_cnn_hyperparameters))
                                tower_losses.append(tf_tower_loss)
                                tf_tower_loss_vec = calc_loss_vector(scope,tf_train_data_batch[-1],tf_train_label_batch[-1],tf_cnn_hyperparameters)
                                tower_loss_vectors.append(tf_tower_loss_vec)

                                tower_grad = gradients(optimizer,tf_tower_loss, global_step, tf.constant(start_lr,dtype=tf.float32))
                                tower_grads.append(tower_grad)

                                tower_pred = predict_with_dataset(tf_train_data_batch[-1],tf_cnn_hyperparameters)
                                tower_predictions.append(tower_pred)

                                # Pooling data operations
                                tf_pool_data_batch.append(tf.placeholder(tf.float32,
                                                            shape=(batch_size, image_size, image_size, num_channels),
                                                            name='PoolDataset'))
                                tf_pool_label_batch.append(tf.placeholder(tf.float32, shape=(batch_size, num_labels), name='PoolLabels'))

                                with tf.name_scope('pool') as scope:
                                    single_pool_logit_op, single_activation_update_op = inference(tf_pool_data_batch[-1],tf_cnn_hyperparameters,True)
                                    tower_pool_activation_update_ops.append(single_activation_update_op)

                                    single_pool_loss = tower_loss(tf_pool_data_batch[-1],tf_pool_label_batch[-1],False,None,tf_cnn_hyperparameters)
                                    tower_pool_losses.append(single_pool_loss)
                                    single_pool_grad = gradients(optimizer,single_pool_loss,global_step,start_lr)
                                    tower_pool_grads.append(single_pool_grad)

                logger.info('GLOBAL_VARIABLES (all)')
                logger.info('\t%s\n',[v.name for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)])
                logger.info('GLOBAL_VARIABLES')
                logger.info('\t%s\n', [v.name for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=TF_GLOBAL_SCOPE)])
                logger.info('TRAINABLE_VARIABLES')
                logger.info('\t%s\n', [v.name for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=TF_GLOBAL_SCOPE)])

                with tf.device('/gpu:0'):
                    # Train data operations
                    # avg_grad_and_vars = [(avggrad0,var0),(avggrad1,var1),...]
                    tf_avg_grad_and_vars = average_gradients(tower_grads)
                    apply_grads_op = apply_gradient_with_momentum(optimizer,start_lr,global_step)
                    concat_loss_vec_op = concat_loss_vector_towers(tower_loss_vectors)
                    update_train_velocity_op = update_train_momentum_velocity(tf_avg_grad_and_vars)
                    tf_mean_activation = mean_tower_activations(tower_activation_update_ops)
                    mean_loss_op = tf.reduce_mean(tower_losses)

                    # Pool data operations
                    tf_pool_avg_gradvars = average_gradients(tower_pool_grads)

                    tf_mean_pool_activations = mean_tower_activations(tower_pool_activation_update_ops)
                    mean_pool_loss = tf.reduce_mean(tower_pool_losses)

                with tf.variable_scope(TF_GLOBAL_SCOPE) as main_scope,tf.device('/gpu:0'):

                    increment_global_step_op = tf.assign(global_step,global_step+1)

                    # GLOBAL: Tensorflow operations for hard_pool
                    with tf.name_scope('pool') as scope:
                        tf.get_variable_scope().reuse_variables()
                        pool_pred = predict_with_dataset(tf_pool_data_batch[0],tf_cnn_hyperparameters)

                    # TODO: Scope?
                    # GLOBAL: Tensorflow operations for test data
                    # Valid data (Next train batch) Unseen
                    tf_valid_data_batch = tf.placeholder(tf.float32,
                                                      shape=(batch_size, image_size, image_size, num_channels),
                                                      name='ValidDataset')
                    tf_valid_label_batch = tf.placeholder(tf.float32, shape=(batch_size, num_labels), name='ValidLabels')
                    # Tensorflow operations for validation data
                    valid_loss_op = tower_loss(tf_valid_data_batch,tf_valid_label_batch,False,None,tf_cnn_hyperparameters)
                    valid_predictions_op = predict_with_dataset(tf_valid_data_batch,tf_cnn_hyperparameters)

                    test_predicitons_op = predict_with_dataset(tf_test_dataset,tf_cnn_hyperparameters)

                init_op = tf.global_variables_initializer()
                _ = session.run(init_op)

                logger.debug('TRAINABLE_VARIABLES')
                logger.debug('\t%s\n',[v.name for v in tf.trainable_variables()])

                logger.info('Variables initialized...')

                train_losses = []
                mean_train_loss = 0
                prev_test_accuracy = 0 # used to calculate test accuracy drop

                rolling_ativation_means = {}
                for op in cnn_ops:
                    if 'conv' in op:
                        logger.debug('\tDefining rolling activation mean for %s',op)
                        rolling_ativation_means[op]=np.zeros([cnn_hyperparameters[op]['weights'][3]])
                dist_decay = 0.9 # was 0.5
                act_decay = 0.9
                current_state,current_action = None,None
                next_valid_accuracy = 0,0

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
                start_adapting = False

                batch_id_multiplier = (dataset_size//batch_size) - interval_parameters['test_interval']

                assert batches_in_chunk%num_gpus ==0
                assert num_gpus>0

                all_test_accuracies = []
                memmap_idx = 0
                for batch_id in range(0,ceil(dataset_size//batch_size)-num_gpus,num_gpus):

                    if batch_id+1>=research_parameters['stop_training_at']:
                        break

                    t0 = time.clock() # starting time for a batch

                    logger.debug('='*80)
                    logger.debug('tf op count: %d',len(graph.get_operations()))
                    logger.debug('=' * 80)

                    logger.debug('\tTraining with batch %d',batch_id)

                    chunk_batch_id = batch_id%batches_in_chunk

                    if chunk_batch_id==0:
                        # We load 1 extra batch (chunk_size+1) because we always make the valid batch the batch_id+1
                        logger.info('\tCurrent memmap start index: %d', memmap_idx)
                        if memmap_idx+chunk_size+batch_size<dataset_size:
                            train_dataset,train_labels = load_data_from_memmap(dataset_info,dataset_filename,label_filename,memmap_idx,chunk_size+batch_size,True)
                        else:
                            train_dataset, train_labels = load_data_from_memmap(dataset_info, dataset_filename, label_filename,
                                                                                memmap_idx, chunk_size,True)
                        memmap_idx += chunk_size
                        logger.info('Loading dataset chunk of size (chunk size + batch size): %d',train_dataset.shape[0])
                        logger.info('\tDataset shape: %s',train_dataset.shape)
                        logger.info('\tLabels shape: %s',train_labels.shape)
                        logger.info('\tNext memmap start index: %d',memmap_idx)

                    # Feed dicitonary with placeholders for each tower
                    batch_data,batch_labels,batch_weights = [],[],[]
                    train_feed_dict = {}
                    for gpu_id in range(num_gpus):

                        batch_data.append(train_dataset[(chunk_batch_id+gpu_id)*batch_size:(chunk_batch_id+gpu_id+1)*batch_size, :, :, :])
                        batch_labels.append(train_labels[(chunk_batch_id+gpu_id)*batch_size:(chunk_batch_id+gpu_id+1)*batch_size, :])

                        cnt = Counter(np.argmax(batch_labels[-1], axis=1))
                        if behavior=='non-stationary':
                            batch_w = np.zeros((batch_size,))
                            batch_labels_int = np.argmax(batch_labels[-1], axis=1)

                            for li in range(num_labels):
                                batch_w[np.where(batch_labels_int==li)[0]] = max(1.0 - (cnt[li]*1.0/batch_size),1.0/num_labels)
                            batch_weights.append(batch_w)

                        elif behavior=='stationary':
                            batch_weights.append(np.ones((batch_size,)))
                        else:
                            raise NotImplementedError

                        train_feed_dict.update({
                            tf_train_data_batch[gpu_id] : batch_data[-1], tf_train_label_batch[gpu_id] : batch_labels[-1],
                            tf_data_weights[gpu_id]:batch_weights[-1]
                        })

                    t0_train = time.clock()
                    for _ in range(iterations_per_batch):
                        if research_parameters['optimizer']=='Momentum':
                            _, _,l, super_loss_vec, current_activations_list,train_predictions = session.run(
                                [apply_grads_op, update_train_velocity_op,mean_loss_op, concat_loss_vec_op, tf_mean_activation,tower_predictions], feed_dict=train_feed_dict
                            )
                        else:
                            raise NotImplementedError

                    t1_train = time.clock()

                    current_activations = get_activation_dictionary(current_activations_list,cnn_ops,convolution_op_ids)

                    # this snippet logs the normalized class distribution every specified interval
                    for bl in batch_labels:
                        if chunk_batch_id%research_parameters['log_distribution_every']==0:
                            dist_cnt = Counter(np.argmax(bl, axis=1))
                            norm_dist = ''
                            for li in range(num_labels):
                                norm_dist += '%.5f,'%(dist_cnt[li]*1.0/batch_size)
                            class_dist_logger.info(norm_dist)

                    if np.isnan(l):
                        logger.critical('Diverged (NaN detected) (batchID) %d (last Cost) %.3f',chunk_batch_id,train_losses[-1])
                        error_logger.info('NaN detected')
                        all_test_accuracies = [0.0]
                        break

                    # rolling activation mean update
                    for op,op_activations in current_activations.items():
                        logger.debug('checking %s',op)
                        logger.debug('\tRolling size (%s): %s',op,rolling_ativation_means[op].shape)
                        logger.debug('\tCurrent size (%s): %s', op, op_activations.shape)
                    for op, op_activations in current_activations.items():
                        assert current_activations[op].size == cnn_hyperparameters[op]['weights'][3]
                        rolling_ativation_means[op]=act_decay * rolling_ativation_means[op] + current_activations[op]

                    train_losses.append(l)

                    # validation batch (Unseen)
                    batch_valid_data = train_dataset[(chunk_batch_id+num_gpus)*batch_size:(chunk_batch_id+num_gpus+1)*batch_size, :, :, :]
                    batch_valid_labels = train_labels[(chunk_batch_id+num_gpus)*batch_size:(chunk_batch_id+num_gpus+1)*batch_size, :]

                    feed_valid_dict = {tf_valid_data_batch:batch_valid_data, tf_valid_label_batch:batch_valid_labels}
                    next_valid_predictions = session.run(valid_predictions_op,feed_dict=feed_valid_dict)
                    next_valid_accuracy = accuracy(next_valid_predictions, batch_valid_labels)

                    # concat all the tower data to a single array
                    single_iteration_batch_data,single_iteration_batch_labels = None,None
                    for gpu_id in range(num_gpus):
                        if single_iteration_batch_data is None and single_iteration_batch_labels is None:
                            single_iteration_batch_data,single_iteration_batch_labels = batch_data[gpu_id],batch_labels[gpu_id]
                        else:
                            single_iteration_batch_data = np.append(single_iteration_batch_data,batch_data[gpu_id],axis=0)
                            single_iteration_batch_labels = np.append(single_iteration_batch_labels,batch_labels[gpu_id],axis=0)

                    if np.random.random()<research_parameters['hard_pool_acceptance_rate']:
                        train_accuracy = np.mean([accuracy(train_predictions[gid],batch_labels[gid]) for gid in range(num_gpus)])/100.0
                        hard_pool.add_hard_examples(single_iteration_batch_data,single_iteration_batch_labels,super_loss_vec,1.0-train_accuracy)
                        logger.debug('Pooling data summary')
                        logger.debug('\tData batch size %d',single_iteration_batch_data.shape[0])
                        logger.debug('\tAccuracy %.3f', train_accuracy)
                        logger.debug('\tPool size (after): %d',hard_pool.get_size())
                        assert hard_pool.get_size()>1

                    if batch_id%interval_parameters['test_interval']==0:
                        mean_train_loss = np.mean(train_losses)
                        logger.info('='*60)
                        logger.info('\tBatch ID: %d'%batch_id)
                        if decay_learning_rate:
                            logger.info('\tLearning rate: %.5f'%session.run(tf.train.exponential_decay(start_lr,global_step,decay_steps=1,decay_rate=decay_rate,staircase=True)))
                        else:
                            logger.info('\tLearning rate: %.5f' %start_lr)

                        logger.info('\tMinibatch Mean Loss: %.3f'%mean_train_loss)
                        logger.info('\tValidation Accuracy (Unseen): %.3f'%next_valid_accuracy)

                        test_accuracies = []
                        for test_batch_id in range(test_size//batch_size):
                            batch_test_data = test_dataset[test_batch_id*batch_size:(test_batch_id+1)*batch_size, :, :, :]
                            batch_test_labels = test_labels[test_batch_id*batch_size:(test_batch_id+1)*batch_size, :]

                            feed_test_dict = {tf_test_dataset:batch_test_data, tf_test_labels:batch_test_labels}
                            test_predictions = session.run(test_predicitons_op,feed_dict=feed_test_dict)
                            test_accuracies.append(accuracy(test_predictions, batch_test_labels))
                            if test_batch_id<10:
                                logger.debug('='*80)
                                logger.debug('Actual Test Labels %d',test_batch_id)
                                logger.debug(np.argmax(batch_test_labels, axis=1).flatten()[:5])
                                logger.debug('Predicted Test Labels %d',test_batch_id)
                                logger.debug(np.argmax(test_predictions,axis=1).flatten()[:5])
                                logger.debug('Test: %d, %.3f',test_batch_id,accuracy(test_predictions, batch_test_labels))
                                logger.debug('=' * 80)

                        current_test_accuracy = np.mean(test_accuracies)
                        all_test_accuracies.append(current_test_accuracy)

                        logger.info('\tTest Accuracy: %.3f'%current_test_accuracy)
                        logger.info('='*60)
                        logger.info('')

                        if research_parameters['adapt_structure'] and abs(current_test_accuracy-prev_test_accuracy)>20:
                            accuracy_drop_logger.info('%s,%d',state_action_history,current_test_accuracy-prev_test_accuracy)

                        prev_test_accuracy = current_test_accuracy
                        error_logger.info('%d,%.3f,%.3f,%.3f',
                                          batch_id,mean_train_loss,
                                          next_valid_accuracy,np.mean(test_accuracies)
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

                    t1 = time.clock()
                    op_count = len(graph.get_operations())
                    var_count = len(tf.global_variables()) + len(tf.local_variables()) + len(tf.model_variables())
                    perf_logger.info('%d,%.5f,%.5f,%d,%d',batch_id,t1-t0,(t1_train-t0_train)/num_gpus,op_count,var_count)

                session.run(increment_global_step_op)
                session.close()

            exp_mean_test_accuracy = np.mean(all_test_accuracies[-10:])
            if exp_mean_test_accuracy>glb_max_test_accuracy:
                glb_max_test_accuracy = exp_mean_test_accuracy
                if hyparam == 'learning_rate':
                    best_lr = value
                    besthyp_logger.info('Best learning rate: %.6f',best_lr)
                    if best_l2beta is not None:
                        besthyp_logger.info('Best L2Beta: %.6f ', best_l2beta)
                    else:
                        besthyp_logger.info('Best L2Beta: None')

                    if best_lrn_rad is not None:
                        besthyp_logger.info('Best LRN Radius: %d', best_lrn_rad)
                    else:
                        besthyp_logger.info('Best LRN Radius: None')

                    besthyp_logger.info('Best LRN alpha: %.6f', best_lrn_alpha)
                    besthyp_logger.info('Best LRN beta: %.6f', best_lrn_beta)
                    besthyp_logger.info('Test Accuracy: %.3f\n',exp_mean_test_accuracy)
                elif hyparam == 'l2_beta':
                    best_l2beta = value
                    if best_l2beta is not None:
                        besthyp_logger.info('Best L2Beta: %.6f', best_l2beta)
                    else:
                        besthyp_logger.info('Best L2Beta: None')
                    besthyp_logger.info('Best learning rate: %.6f', best_lr)
                    if best_lrn_rad is not None:
                        besthyp_logger.info('Best LRN Radius: %d', best_lrn_rad)
                    else:
                        besthyp_logger.info('Best LRN Radius: None')
                    besthyp_logger.info('Best LRN alpha: %.6f', best_lrn_alpha)
                    besthyp_logger.info('Best LRN beta: %.6f', best_lrn_beta)
                    besthyp_logger.info('Test Accuracy: %.3f\n', exp_mean_test_accuracy)
                elif hyparam == 'lrn_rads':
                    best_lrn_rad = value
                    if best_lrn_rad is not None:
                        besthyp_logger.info('Best LRN radius: %.6f',best_lrn_rad)
                    else:
                        besthyp_logger.info('Best LRN Radius: None')
                    besthyp_logger.info('Best learning rate: %.6f', best_lr)
                    if best_l2beta is not None:
                        besthyp_logger.info('Best L2Beta: %.6f ', best_l2beta)
                    else:
                        besthyp_logger.info('Best L2Beta: None')
                    besthyp_logger.info('Best LRN alpha: %.6f', best_lrn_alpha)
                    besthyp_logger.info('Best LRN beta: %.6f', best_lrn_beta)
                    besthyp_logger.info('Test Accuracy: %.3f\n', exp_mean_test_accuracy)

                elif hyparam == 'lrn_alpha':
                    best_lrn_alpha = value
                    besthyp_logger.info('Best LRN alpha: %.6f', best_lrn_alpha)
                    besthyp_logger.info('Best learning rate: %.6f', best_lr)
                    if best_l2beta is not None:
                        besthyp_logger.info('Best L2Beta: %.6f ', best_l2beta)
                    else:
                        besthyp_logger.info('Best L2Beta: None')
                    if best_lrn_rad is not None:
                        besthyp_logger.info('Best LRN Radius: %d', best_lrn_rad)
                    else:
                        besthyp_logger.info('Best LRN Radius: None')
                    besthyp_logger.info('Best LRN beta: %.6f', best_lrn_beta)
                    besthyp_logger.info('Test Accuracy: %.3f\n', exp_mean_test_accuracy)

                elif hyparam == 'lrn_beta':
                    best_lrn_beta = value
                    besthyp_logger.info('Best LRN Beta: %.6f (Test Accuracy: %.3f)', best_lrn_beta,exp_mean_test_accuracy)
                    besthyp_logger.info('Best learning rate: %.6f', best_lr)
                    if best_l2beta is not None:
                        besthyp_logger.info('Best L2Beta: %.6f ', best_l2beta)
                    else:
                        besthyp_logger.info('Best L2Beta: None')
                    if best_lrn_rad is not None:
                        besthyp_logger.info('Best LRN Radius: %d', best_lrn_rad)
                    else:
                        besthyp_logger.info('Best LRN Radius: None')
                    besthyp_logger.info('Best LRN alpha: %.6f', best_lrn_alpha)
                    besthyp_logger.info('Test Accuracy: %.3f\n', exp_mean_test_accuracy)