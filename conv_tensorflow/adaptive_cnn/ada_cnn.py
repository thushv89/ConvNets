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

batch_size = 128 # number of datapoints in a single batch

start_lr = 0.1
decay_learning_rate = True

#dropout seems to be making it impossible to learn
#maybe only works for large nets
dropout_rate = 0.25
use_dropout = True

include_l2_loss = True
#keep beta small (0.2 is too much >0.002 seems to be fine)
beta = 1e-5

construct_policy_interval = 1 #number of epochs an action taken during net constructution
assert_true = True

train_dataset, train_labels = None,None
valid_dataset, valid_labels = None,None
test_dataset, test_labels = None,None

layer_count = 0 #ordering of layers
time_stamp = 0 #use this to indicate when the layer was added

feature_map_upper_bound = 2048
#pool_hyperparameters = {'pool_size':10000,'hardness':0.5,'valid_pool_size':1600}

iconv_ops = ['conv_global','pool_global','fulcon_out'] #ops will give the order of layers
conv_depths = {} # store the in and out depth of each convolutional layer
conv_order  = {} # layer order (in terms of convolutional layer) in the full structure

conv_global_hyp = {'kernel':[1,1,num_channels,2048],'stride':[1,1,1,1],'padding':'SAME'}
pool_global_hyp = {'type':'avg','kernel':[1,image_size,image_size,1],'stride':[1,1,1,1],'padding':'VALID'}
hyparams = {
    'conv_global':conv_global_hyp,'pool_global':pool_global_hyp,
    'fulcon_out':{'in':1*1*conv_depths['conv_global'], 'out':num_labels,
                  'whd':[1,1,num_channels]
                  }
}

weights,biases = {},{}
final_x = (image_size,image_size)

def init_iconvnet():

    print('Initializing the iConvNet (conv_global,pool_global,classifier)...')
    for op in iconv_ops:
        if 'conv' in op:
            weights[op] = tf.Variable(tf.truncated_normal(
                hyparams[op]['kernel'],
                stddev=2./(hyparams[op]['kernel'][0]*hyparams[op]['kernel'][1])
            ))
            biases[op] = tf.Variable(tf.constant(
                np.random.random()*0.01,shape=[hyparams[op]['kernel'][3]]
            ))

            print('Weights for %s initialized with size %s'%(op,str(hyparams[op]['kernel'])))
            print('Biases for %s initialized with size %d'%(op,hyparams[op]['kernel'][3]))

        if 'fulcon' in op:
            weights['fulcon_out'] = tf.Variable(tf.truncated_normal(
                [hyparams[op]['in'],hyparams[op]['out']],
                stddev=2./hyparams[op]['in']
            ))
            biases[op] = tf.Variable(tf.constant(
                np.random.random()*0.01,shape=[hyparams[op]['out']]
            ))

            print('Weights for %s initialized with size %d,%d'%(
                op,hyparams[op]['in'],hyparams[op]['out']
            ))
            print('Biases for %s initialized with size %d'%(
                op,hyparams[op]['out']
            ))


def update_conv(conv_id,to_d):
    '''
    If a convolutional layer (intermediate) is removed. This function will
     Correct the dimention mismatch either by adding more weights or removing
    :param to_d: the depth to be updated to
    :return:
    '''
    from_d = conv_depths[get_layer_id(conv_id)]['in']
    conv_weights = hyparams[get_layer_id(conv_id)]['weights']

    logger.info("Need to update the layer %s in_depth from %d to %d"%(conv_id,from_d,to_d))
    hyparams[conv_id]['weights'] =[conv_weights[0],conv_weights[1],to_d,conv_weights[3]]
    if to_d<from_d:
        logger.info("\tRequired to remove %d depth layers"%(from_d-to_d))
        # update hyperparameters
        logger.debug('\tNew weights should be: %s'%hyparams[get_layer_id(conv_id)]['weights'])
        # update weights
        weights[conv_id] = tf.slice(weights[conv_id],
                                                  [0,0,0,0],[conv_weights[0],conv_weights[1],to_d,conv_weights[3]]
                                                  )
        # no need to update biase
    elif to_d>from_d:
        logger.info("\tRequired to add %d depth layers"%(to_d-from_d))

        conv_new_weights = tf.Variable(tf.truncated_normal(
            [conv_weights[0],conv_weights[1],to_d-from_d,conv_weights[3]],
            stddev=2./(conv_weights[0]*conv_weights[1])
        ))
        tf.initialize_variables([conv_new_weights]).run()
        weights[get_layer_id(conv_id)] = tf.concat(2,[weights[get_layer_id(conv_id)],conv_new_weights])
    else:
        logger.info('\tNo modifications done...')
        return

    new_shape = weights[get_layer_id(conv_id)].get_shape().as_list()
    logger.info('Shape of new weights after the modification: %s'%new_shape)

    assert np.all(np.asarray(new_shape)>0)

    #find the convolutional layer before the addition
    prev_conv_id = None

    for op in reversed(iconv_ops[:iconv_ops.index(conv_id)+1]):
        if 'conv' in op and conv_id!=op:
            prev_conv_id = op
            break
    if prev_conv_id is not None:
        logger.debug("Prev_conv_id, conv_id %s,%s"%(prev_conv_id,conv_id))
        assert weights[prev_conv_id].get_shape().as_list()[3]==new_shape[2]
    else:
        assert new_shape[2]==num_channels

    assert new_shape == hyparams[conv_id]['weights']


def add_conv_layer(w,stride,conv_id):
    '''
    Specify the tensor variables and hyperparameters for a convolutional neuron layer. And update conv_ops list
    :param w: Weights of the receptive field
    :param stride: Stried of the receptive field
    :return: None
    '''
    global layer_count,weights,biases,hyparams



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

    out_id = iconv_ops.pop(-1)
    iconv_ops.append(pool_id)
    iconv_ops.append(out_id)

def get_conv_layer_with_hyparameters(w,stride,get_specific=True):

    global weights

    mod_op_array = iconv_ops if not get_specific else reversed(iconv_ops)
    for op in mod_op_array:
        if 'conv' in op:
            if hyparams[op]['weights'][:2]==w[:2] and hyparams[op]['stride']==stride:
                logger.info("Removing (Convolution) layer %s with weights %s"%(op,hyparams[op]['weights']))
                return op
    return None

def get_logits(dataset):

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
    rows = shape[0]

    fin_xw,fin_xh = get_final_x(None,None)
    logger.debug("My calculation, TensorFlow calculation: (%d,%d), (%d,%d)"%(fin_xw,fin_xh,shape[1],shape[2]))
    assert fin_xw == shape[1] and fin_xh == shape[2]

    print('Unwrapping last convolution layer %s to %s hidden layer'%(shape,(rows,hyparams['fulcon_out']['in'])))
    x = tf.reshape(x, [rows,hyparams['fulcon_out']['in']])

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

def get_final_x(new_op,hyp):
    '''
    Takes a new operations and concat with existing set of operations
    then output what the final output size will be
    :param op_list: list of operations
    :param hyp: Hyperparameters fo the operation
    :return: a tuple (width,height) of x
    '''

    xw,xh = image_size,image_size
    for op in iconv_ops:
        hyp_op = hyparams[op]
        if 'conv' in op:
            fw,fh = hyp_op['weights'][0], hyp_op['weights'][1]
            sw,sh = hyp_op['stride'][1], hyp_op['stride'][2]

            xw,xh = ceil(float(xw)/float(sw)),ceil(float(xh)/float(sh))
        elif 'pool' in op:
            fw,fh = hyp_op['kernel'][1], hyp_op['kernel'][2]
            sw,sh = hyp_op['stride'][1], hyp_op['stride'][2]
            xw,xh = ceil(float(xw)/float(sw)),ceil(float(xh)/float(sh))

    if new_op is not None:
        if 'conv' in new_op:
            fw,fh = hyp['weights'][0],hyp['weights'][1]
        elif 'pool' in new_op:
            fw,fh = hyp['kernel'][1],hyp['kernel'][2]
        sw,sh = hyp['stride'][1],hyp['stride'][2]

        xw,xh = ceil(float(xw)/float(sw)),ceil(float(xh)/float(sh))

    return xw,xh

def get_op_from_action(action):
    '''
    converts a given action (add,C_K5x5x64_S1x1)
    to a tuple of operation,hyperparameters
    :param action: action to be converted to an op
    :return: a Tuple of (op_type, kernel, stride)
    '''


def get_time_stamp_for_layer(layer_id):
    return int(layer_id.split('_')[1])

def get_layer_count_with_type():

def execute_action(action,policy):
    '''
    Execute an action and output if it was successful. Include 3 main actions
    Add,Type_Hyperparameters => Add a layer of given type with the given hyperparameters
    Remove,Type_Hyperparameters => Removes a layer furthest from output layer having the given hyperparameters
    Finetune => Finetune the network with a pool of data
    :param action: action as a string
    :param policy: if executing action to learn policy or picking action
    :return: Whether successful or not
    '''


if __name__=='__main__':
    global logger

    logger = logging.getLogger('main_logger')
    logger.setLevel(logging_level)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(logging_format))
    console.setLevel(logging_level)
    logger.addHandler(console)

    # Value logger will log info used to calculate policies
    value_logger = logging.getLogger('state_action_logger')
    value_logger.setLevel(logging.INFO)
    value_fileHandler = logging.FileHandler('state_action_log.log', mode='w')
    value_fileHandler.setFormatter(logging.Formatter('%(message)s'))
    value_logger.addHandler(value_fileHandler)

    # Value logger will log info used to calculate policies
    test_logger = logging.getLogger('test_logger')
    test_logger.setLevel(logging.INFO)
    fileHandler = logging.FileHandler('test_log', mode='w')
    fileHandler.setFormatter(logging.Formatter('%(message)s'))
    test_logger.addHandler(fileHandler)

    load_data.load_and_save_data_cifar10()
    (train_dataset,train_labels),(valid_dataset,valid_labels),(test_dataset,test_labels) = load_data.reformat_data_cifar10()
    train_size,valid_size,test_size = train_dataset.shape[0],valid_dataset.shape[0],test_dataset.shape[0]

    graph = tf.Graph()

    policy_learner = qlearner.ContinuousState(
        learning_rate=0.5,discount_rate=0.9,policy_interval=policy_interval,gp_interval=5,eta_1=10,eta_2=30,epsilon=0.5
    )
    action_picker = learn_best_actions.ActionPicker(learning_rate=0.5,discount_rate=0.9)
    hard_pool = Pool(image_size=image_size,num_channels=num_channels,num_labels=num_labels,
                     size=pool_hyperparameters['pool_size'],batch_size=batch_size,assert_test=assert_true)
    valid_pool = Pool(image_size=image_size,num_channels=num_channels,num_labels=num_labels,
                     size=pool_hyperparameters['valid_pool_size'],batch_size=batch_size,assert_test=assert_true)

    valid_accuracy_log = []
    time_log = []
    param_log = []
    previous_action_success = True
    previous_action = None
    previous_pool_duration = None

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

        pool_logits = get_logits(tf_pool_dataset)
        pool_loss = calc_loss(pool_logits,tf_pool_labels)
        pool_optimize = optimize_with_fixed_lr_func(pool_loss,tf.constant(0.05))

        # valid predict function
        pred_valid = predict_with_dataset(tf_valid_dataset)
        pred_test = predict_with_dataset(tf_test_dataset)

        tf.initialize_all_variables().run()

        start_time = time.clock()
        for _ in range(10):
            for batch_id in range(ceil(train_size//batch_size)-1):

                batch_data = train_dataset[batch_id*batch_size:(batch_id+1)*batch_size, :, :, :]
                batch_labels = train_labels[batch_id*batch_size:(batch_id+1)*batch_size, :]

                # validation batch
                batch_valid_data = train_dataset[(batch_id+1)*batch_size:(batch_id+2)*batch_size, :, :, :]
                batch_valid_labels = train_labels[(batch_id+1)*batch_size:(batch_id+2)*batch_size, :]

                if research_parameters['use_valid_pool']:
                    if np.random.np.random.random_sample()<=0.05 or np.random.random_sample()>=0.95:
                        valid_pool.add_all_from_ndarray(batch_valid_data,batch_valid_labels)

                batch_start_time = time.clock()
                feed_dict = {tf_dataset : batch_data, tf_labels : batch_labels}
                _, l, l_vec, (_,updated_lr), predictions,_ = session.run(
                        [logits,loss,loss_vector,optimize,pred,inc_gstep], feed_dict=feed_dict
                )

                batch_end_time = time.clock()

                assert not np.isnan(l)

                hard_fraction = 1.0 - float(accuracy(predictions, batch_labels))/100.0

                if hard_fraction > pool_hyperparameters['hardness']:
                    '''logger.debug("Hard pool (pos,size) before adding %d points having %.1f hard examples:(%d,%d)"
                             %(batch_labels.shape[0],hard_fraction,hard_pool.get_position(),hard_pool.get_size())
                             )'''
                    hard_pool.add_hard_examples(batch_data,batch_labels,l_vec,hard_fraction)
                    #logger.debug("\tHard pool (pos,size) after (%s,%s):"%(hard_pool.get_position(),hard_pool.get_size()))

                if not research_parameters['use_valid_pool']:
                    valid_feed_dict = {tf_valid_dataset:batch_valid_data,tf_valid_labels:batch_valid_labels}
                    valid_predictions = session.run([pred_valid],feed_dict=valid_feed_dict)
                    valid_accuracy_log.append(accuracy(valid_predictions[0],batch_valid_labels))
                else:
                    vpool_data = valid_pool.get_pool_data()
                    for vpool_batchid in range(valid_pool.get_size()//batch_size):
                        vpool_batch_data = vpool_data['pool_dataset'][pbatch_id*batch_size:(pbatch_id+1)*batch_size,:,:,:]
                        vpool_batch_labels = vpool_data['pool_labels'][pbatch_id*batch_size:(pbatch_id+1)*batch_size,:]

                        feed_pool_dict = {tf_valid_dataset : vpool_batch_data, tf_valid_labels : vpool_batch_labels}
                        vpool_predictions = session.run([pred_valid], feed_dict=feed_pool_dict)
                        valid_accuracy_log.append(accuracy(vpool_predictions[0],vpool_batch_labels))

                time_log.append(batch_end_time-batch_start_time)

                # for batch [1,2,3,...,500,10000,10001,...,10500,...]
                if 0 <= time_stamp % interval_dict['action_update_interval'] <= 500:

                    #reset previous action
                    if time_stamp % interval_dict['action_update_interval'] == 0:
                        previous_action = None
                        original_net_size = len(iconv_ops)

                    # remove the layer added by the previous action
                    if previous_action is not None and previous_action_success and \
                                            time_stamp % interval_dict['action_test_interval'] == 0:
                        # first remove the previously added layer
                        rm_action = 'remove_last'
                        logger.debug("Removing  with %s",rm_action)
                        execute_action(rm_action,policy=False)
                        logger.debug("Ops after removal %s"%iconv_ops)
                        assert original_net_size == len(iconv_ops)

                    #for batch [500,10500,20500,30500,...]
                    if time_stamp > 0 and time_stamp % interval_dict['action_update_interval'] == 500:

                        logger.info('Action Picker Finished ...')
                        predicted_actions = action_picker.get_best_actions(5,research_parameters['seperate_cp_best_actions'])
                        logger.info('\t Got following best actions %s'%predicted_actions)
                        # update the policy action space
                        policy_learner.update_action_space(predicted_actions)
                        action_picker.reset_all()

                        assert original_net_size == len(iconv_ops)

                    #for batch [10,20,30,...]
                    elif time_stamp % interval_dict['action_test_interval'] == 0:

                        logger.info("====================== Executing action for time stamp %d ======================"%time_stamp)
                        # as long as it's not the first time of we run the action picker
                        # [20,30,...,500],[5020,5030,...]

                        mean_valid_accuracy = np.mean(valid_accuracy_log[np.max([0,len(valid_accuracy_log)-(interval_dict['action_test_interval']//2)]):])
                        param_count = get_parameter_count()
                        duration = np.sum(time_log[np.max([0,len(time_log)-(interval_dict['action_test_interval']//2)]):])

                        if previous_action is not None:
                            prev_op_info = get_op_from_action(previous_action)

                            param_cost = prev_op_info[1][0]*prev_op_info[1][1]*prev_op_info[1][3]
                            stride_cost = (prev_op_info[2][1]-1)*(prev_op_info[2][2]-1)
                            logger.debug('\tWith data (Error:%.3f,Time:%.3f,Params:%d,Stride:%.3f'%
                                         (data['error_t'],data['time_cost'],param_cost,stride_cost)
                            )
                        else:
                            param_cost = None
                            stride_cost = None

                        data = {
                            'error_t':100-mean_valid_accuracy,'time_cost':duration,
                            'param_cost': param_cost,
                            'stride_cost': stride_cost,
                        }

                        logger.debug('Action Picker running for time stamp %d'%time_stamp)

                        # add the new layer
                        value_logger.info('%d,%s,%s'%(time_stamp,data,previous_action))
                        action = action_picker.update_policy(time_stamp,data,previous_action_success)
                        logger.info('Executing action %s'%action)
                        previous_action_success = execute_action(action,policy=False)
                        previous_action = action

                # for batch [100,200,300,400,...]
                elif time_stamp > 0 and time_stamp % policy_interval == 0:

                    logger.info('\n==================== Time Stamp: %d ====================='%time_stamp)
                    logger.debug('\tGlobal step: %d'%global_step.eval())
                    logger.debug('\tCurrent Ops: %s'%iconv_ops)
                    logger.info('\tMinibatch loss at step %d: %f' % (batch_id, l))
                    logger.debug('\tLearning rate: %.3f'%updated_lr)
                    logger.info('\tMinibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
                    mean_valid_accuracy = np.mean(valid_accuracy_log[np.max([0,len(valid_accuracy_log)-(policy_interval//2)]):])
                    logger.debug('\t%d Mean Valid accuracy (Latter half)[%d:%d]: %.1f%%' %
                          (time_stamp,np.max([0,len(valid_accuracy_log)-(policy_interval//2)]),len(valid_accuracy_log), mean_valid_accuracy)
                          )

                    layer_type_count = get_layer_count_with_type()
                    param_count = get_parameter_count()
                    logger.debug('\t%d Parameter count: %d'%(time_stamp,param_count))
                    duration = np.sum(time_log[np.max([0,len(time_log)-(policy_interval//2)]):])
                    logger.debug('\t%d Time taken (Latter half)[%d:%d]: %.3f'%
                          (time_stamp,np.max([0,len(time_log)-(policy_interval//2)]),len(time_log),duration)
                    )

                    if previous_action is not None:
                        if 'P_' in previous_action or 'C_' in previous_action:
                            prev_op_info = get_op_from_action(previous_action)
                            param_cost = prev_op_info[1][0]*prev_op_info[1][1]*prev_op_info[1][3]
                            stride_cost = (prev_op_info[2][1]-1)*(prev_op_info[2][2]-1)
                            if 'P_' in previous_action:
                                complex_cost = get_layer_count_with_type()['pool']
                            elif 'C_' in previous_action:
                                complex_cost = get_layer_count_with_type()['conv']

                            if 'remove' in previous_action:
                                param_cost *= 0.
                                stride_cost *= 0.
                        # actions remove_last, finetune, train_with_batch
                        else:
                            param_cost = 0.
                            stride_cost = 0.
                            complex_cost = 0.
                            # was chosen depending on the propotion between
                            # data processed within policy interval and the_data in the finetune pool
                            if previous_action == 'finetune':
                                duration = previous_pool_duration/6.0
                        logger.info('%d Data for this episode: Error:%.3f, Time:%.3f, Param Cost:%d, Stride:%d'%
                                (time_stamp,data['error_t'],data['time_cost'],param_cost,stride_cost)
                                )
                    else:
                        param_cost,stride_cost,complex_cost = None,None,None

                    data = {
                        'error_t':100-mean_valid_accuracy,'time_cost':duration,
                        'param_cost': param_cost, 'layer_rank':layer_count,'complexity_cost':complex_cost,
                        'stride_cost': stride_cost,'all_params':param_count,
                        'conv_layer_count':layer_type_count['conv'],'pool_layer_count':layer_type_count['pool']
                    }

                    value_logger.info('%d,%s,%s'%(time_stamp,data,previous_action))

                    logger.info('=============================================================\n')
                    action = policy_learner.update_policy(time_stamp,data,previous_action_success)
                    logger.info('\tReceived action %s'%action)

                    if action != 'finetune':
                        previous_action_success = execute_action(action,policy=True)
                    else:

                        pool_start_time = time.clock()
                        p_data = hard_pool.get_pool_data()

                        logger.debug("Finetuning with pool of size %d"%hard_pool.get_size())
                        for pbatch_id in range(hard_pool.get_size()//batch_size):
                            pool_batch_data = p_data['pool_dataset'][pbatch_id*batch_size:(pbatch_id+1)*batch_size,:,:,:]
                            pool_batch_labels = p_data['pool_labels'][pbatch_id*batch_size:(pbatch_id+1)*batch_size,:]

                            feed_pool_dict = {tf_pool_dataset : pool_batch_data, tf_pool_labels : pool_batch_labels}

                            _, pool_l, _ = \
                                session.run([pool_logits,pool_loss,pool_optimize], feed_dict=feed_pool_dict)

                            if assert_true:
                                assert not np.isnan(pool_l)
                        pool_end_time = time.clock()
                        previous_pool_duration = pool_end_time-pool_start_time
                        previous_action_success = True

                    previous_action = action
                    logits = get_logits(tf_dataset)

                if time_stamp>0 and time_stamp % interval_dict['test_interval'] == 0:
                    # test batches
                    test_accuracies = []
                    for batch_id in range(ceil(test_size//batch_size)):
                        batch_test_data = test_dataset[(batch_id)*batch_size:(batch_id+1)*batch_size, :, :, :]
                        batch_test_labels = test_labels[(batch_id)*batch_size:(batch_id+1)*batch_size, :]

                        test_feed_dict = {tf_test_dataset:batch_test_data,tf_test_labels:batch_test_labels}
                        test_predictions = session.run([pred_test],feed_dict=test_feed_dict)
                        test_accuracies.append(accuracy(valid_predictions[0],batch_valid_labels))

                    logger.info("==============================================")
                    logger.info("Test Accuracy: %.1f"%np.mean(test_accuracies))
                    logger.info("==============================================\n")
                    test_logger.info("%d,%.3f"%(time_stamp,np.mean(test_accuracies)))
                time_stamp += 1

        '''if batch_id == 101:
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
            break'''
