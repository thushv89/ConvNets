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
logging_level = logging.DEBUG
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
conv_depths = {}
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

    if to_d<from_d:
        logger.info("\tRequired to remove %d depth layers"%(from_d-to_d))
        # update hyperparameters
        hyparams[get_layer_id(conv_id)]['weights'] =[conv_weights[0],conv_weights[1],to_d,conv_weights[3]]
        # update weights
        weights[get_layer_id(conv_id)] = tf.slice(weights[get_layer_id(conv_id)],
                                                  [0,0,0,0],[0,0,to_d,0]
                                                  )
        # no need to update biase
    elif to_d>from_d:
        logger.info("\tRequired to add %d depth layers"%to_d-from_d)
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


def add_conv_layer(w,stride,conv_id):
    '''
    Specify the tensor variables and hyperparameters for a convolutional neuron layer. And update conv_ops list
    :param w: Weights of the receptive field
    :param stride: Stried of the receptive field
    :return: None
    '''
    global layer_count,weights,biases,hyparams


    hyparams[conv_id]={'weights':w,'stride':stride,'padding':'SAME'}
    weights[conv_id]= tf.Variable(tf.truncated_normal(w,stddev=2./(w[0]*w[1])),name='w_'+conv_id)
    biases[conv_id] = tf.Variable(tf.constant(np.random.random()*0.01,shape=[w[3]]),name='b_'+conv_id)

    tf.initialize_variables([weights[conv_id],biases[conv_id]]).run()

    out_id = iconv_ops.pop(-1)
    iconv_ops.append(conv_id)
    iconv_ops.append(out_id)



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


def remove_conv_layer(w,stride):
    '''
    Removes the first layer (furthest to fulcon_out) having the given w and stride
    We only compare the width and height of w and the full stride
    We do not remove the hyperparameters or weights immediately. If the removed ones
    are not used after a certain threshold, they will be removed
    :return: id of the removed layer
    '''
    global weights
    out_id = iconv_ops.pop(-1)
    rm_idx = 0
    for op in iconv_ops:
        if 'conv' in op:
            if hyparams[op]['weights'][:2]==w[:2] and hyparams[op]['stride']==stride:
                logger.info("Removing (Convolution) layer %s"%op)
                iconv_ops.remove(op)
                break
        rm_idx += 1
    iconv_ops.append(out_id)
    return op,rm_idx

def remove_pool_layer(kernel,stride,pool_type):
    '''
    Removes the first layer (furthest to fulcon_out) having the given w and stride
    We compare the full kernel and stride
    We do not remove the hyperparameters or weights immediately. If the removed ones
    are not used after a certain threshold, they will be removed
    :return: id of the removed layer
    '''
    global weights
    out_id = iconv_ops.pop(-1)
    for op in iconv_ops:
        if 'pool' in op:
            if hyparams[op]['kernel']==kernel and hyparams[op]['stride']==stride and hyparams[op]['type']==pool_type:
                logger.info("Removing (Pool) layer %s"%op)
                iconv_ops.remove(op)
                break

    iconv_ops.append(out_id)
    return op

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

def get_stride_cost():
    '''
    This is included in the reward to peanalize the algorithm for using high strides
    :return:
    '''
    global weights
    total_stride = 0
    for op in iconv_ops:
        if 'conv' in op or 'pool' in op:
            stride = hyparams[op]['stride']
            total_stride += np.prod(stride)
    return total_stride
def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])

def get_op_from_action(action):
    '''
    converts a given action (add,C_K5x5x64_S1x1)
    to a tuple of operation,hyperparameters
    :param action: action to be converted to an op
    :return: a Tuple of (op_type, kernel, stride)
    '''
    op_token = action.split(',')[1]
    op = ''
    if 'C_' in action:
        #add,C_K5x5x64_S1x1
        conv_tokens = op_token.split('_')
        for tok in conv_tokens:
            if 'K' == tok[0]:
                kernel=[int(k) for k in tok[1:].split('x')]
                kernel.insert(2,-1)
            elif 'S' == tok[0]:
                stride=[int(k) for k in tok[1:].split('x')]
                stride.insert(0,1)
                stride.append(1)

        op = ('conv',kernel,stride)
    elif 'P_' in action:
        #add,P_K5x5_S3x3
        pool_tokens = op_token.split('_')
        for tok in pool_tokens:
            if 'K' == tok[0]:
                kernel=[int(k) for k in tok[1:].split('x')]
                kernel.insert(0,1)
                kernel.append(1)
            elif 'S' == tok[0]:
                stride=[int(k) for k in tok[1:].split('x')]
                stride.insert(0,1)
                stride.append(1)

        op = ('pool',kernel,stride,pool_tokens[-1])

    return op

def execute_action(action):
    global layer_count
    if 'remove' in action:
        logger.info("Removing layer %s",action)
        op_info = get_op_from_action(action)
        if op_info[0]=='conv':
            #remove convolution layer
            op_type,weights,stride = op_info
            logger.info('\tRemoving convolution layer')
            rm_op,rm_idx = remove_conv_layer(op_info[1],op_info[2])

            # if the layer we removed is not eh last convovolutional layer
            # if it's the last convo layer, next_rm_op will be the fulcon_out
            next_rm_op,last_conv_layer = -1,True
            for rm_idx in range(rm_idx,len(iconv_ops)):
                if 'conv' in iconv_ops[rm_idx]:
                    next_rm_op = iconv_ops[rm_idx]
                    last_conv_layer = False

            if not last_conv_layer:
                update_conv(next_rm_op,conv_depths[rm_op]['in'])
                conv_depths[next_rm_op]['in']=conv_depths[rm_op]['in']
                logger.info("\tDepth of %s after update: %s"%(next_rm_op,conv_depths[next_rm_op]['in']))

        elif op_info[0]=='pool':
            #remove pool layer
            op_type,kernel,stride,pool_type = op_info
            logger.info('\tRemoving pool layer with kernel:%s,stride:%s,type:%s'%(kernel,stride,pool_type))
            remove_pool_layer(kernel,stride,pool_type)
        logger.info("\tRemoved layer: %s"%action)
        return

    if action=='finetune':
        #TODO
        logger.info("To be impelemented...")
        return

    # Convolution: 'op',[filter_w,filter_h,out_depth],[stride_w,stride_h]
    # Pool: 'op',[filter_w,filter_h],[stride_w,stride_h]
    op_info = get_op_from_action(action)

    if op_info[0]=='conv':
        conv_id = get_layer_id('conv_'+str(layer_count))
        layer_count += 1
        initial_conv_layer = np.all([True if 'conv' not in op else False for op in iconv_ops])
        print('Ops %s, Initial conv %s'%(iconv_ops,initial_conv_layer))
        if initial_conv_layer:
            conv_depths[conv_id]={'in':num_channels,'out':op_info[1][3]}
            logger.debug('(Initial) Depth for the new Conv Layer: in:%d, out:%d'%(conv_depths[conv_id]['in'],conv_depths[conv_id]['out']))
        else:
            #find the last conv_id currently in conv_ops
            prev_conv_id = ''
            for op in iconv_ops:
                # if iteration come to current conv_op, keep the last previous and break the loop
                if op==conv_id:
                    break
                if 'conv' in op:
                    prev_conv_id = op

            conv_depths[conv_id]={'in':conv_depths[prev_conv_id]['out'],'out':op_info[1][3]}
            logger.debug('(Following) Depth for the new Conv Layer: in:%d, out:%d'%(conv_depths[conv_id]['in'],conv_depths[conv_id]['out']))

        w = [op_info[1][0],op_info[1][1],conv_depths[conv_id]['in'],conv_depths[conv_id]['out']]
        s = op_info[2]
        logger.info('Adding Convolutional Layer %s'%conv_id)
        logger.debug('\tWith Weights:%s,Stride:%s'%(w,s))
        add_conv_layer(w,s,conv_id)

    elif op_info[0]=='pool':
        pool_id = get_layer_id('pool_'+str(layer_count))

        k,s,pool_type = op_info[1],op_info[2],op_info[3]

        logger.info('Adding Pooling Layer %s'%pool_id)
        logger.debug('\tWith Kernel:%s,Stride:%s,Type:%s'%(k,s,pool_type))
        add_pool_layer(k,s,pool_type,pool_id)

        layer_count += 1
    else:
        raise NotImplementedError

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
        learning_rate=0.5,discount_rate=0.9,policy_interval=policy_interval,gp_interval=5,eta_1=10,eta_2=20
    )
    action_picker = learn_best_actions.ActionPicker(learning_rate=0.5,discount_rate=0.9)

    valid_accuracy_log = []
    time_log = []
    param_log = []

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
        for _ in range(10):
            for batch_id in range(ceil(train_size//batch_size)-1):

                time_stamp += 1
                batch_data = train_dataset[batch_id*batch_size:(batch_id+1)*batch_size, :, :, :]
                batch_labels = train_labels[batch_id*batch_size:(batch_id+1)*batch_size, :]

                # validation batch
                batch_valid_data = train_dataset[(batch_id+1)*batch_size:(batch_id+2)*batch_size, :, :, :]
                batch_valid_labels = train_labels[(batch_id+1)*batch_size:(batch_id+2)*batch_size, :]

                start_time = time.clock()
                feed_dict = {tf_dataset : batch_data, tf_labels : batch_labels}
                _, l, (_,updated_lr), predictions,_ = session.run([logits,loss,optimize,pred,inc_gstep], feed_dict=feed_dict)
                end_time = time.clock()


                valid_feed_dict = {tf_valid_dataset:batch_valid_data,tf_valid_labels:batch_valid_labels}
                valid_predictions = session.run([pred_valid],feed_dict=valid_feed_dict)
                valid_accuracy_log.append(accuracy(valid_predictions[0],batch_valid_labels))
                time_log.append(end_time-start_time)


                # for batch [1,2,3,...,500,10000,10001,...,10500,...]
                if batch_id>0 and 0 < batch_id % (100*policy_interval) <= 500:

                    #for batch [500,10500,20500,30500,...]
                    if batch_id % (100*policy_interval) == 500:
                        logger.info('Action Picker Finished ...')
                        predicted_actions = action_picker.get_best_actions(5)
                        logger.info('\t Got following best actions %s'%predicted_actions)
                        # update the policy action space
                        policy_learner.update_action_space(predicted_actions)

                    #for batch [5,10,15,...]
                    elif batch_id % 5 == 0:

                        logger.info("====================== Executin action for time stamp %d ======================"%time_stamp)
                        if batch_id!=5:
                            rm_action = 'remove,'+action.split(',')[1]
                            logger.debug("Removing  with %s",rm_action)
                            execute_action(rm_action)

                        mean_valid_accuracy = np.mean(valid_accuracy_log[np.max([0,len(valid_accuracy_log)-5]):])
                        param_count = get_parameter_count()
                        stride_cost = get_stride_cost()
                        duration = np.sum(time_log[np.max([0,len(time_log)-5]):])
                        data = {
                            'error_t':100-mean_valid_accuracy,'time_cost':duration,
                            'param_cost':param_count,'num_layers':len(iconv_ops),
                            'stride_cost': stride_cost
                        }
                        logger.debug('Action Picker running for batch %d'%batch_id)
                        logger.debug('\tWith data (Error:%.3f,Time:%.3f,Params:%d,Layers:%d,Stride:%.3f'%
                                     (data['error_t'],data['time_cost'],data['param_cost'],data['num_layers'],data['stride_cost'])
                                     )
                        value_logger.info('%d,%s'%(time_stamp,data))
                        action = action_picker.update_policy(time_stamp,data)
                        logger.info('Executing action %s'%action)
                        execute_action(action)

                # for batch [100,200,300,400,...]
                elif batch_id>0 and batch_id % policy_interval == 0:

                    logger.info('\n==================== Time Stamp: %d ====================='%time_stamp)
                    logger.debug('Global step: %d'%global_step.eval())
                    logger.info('Minibatch loss at step %d: %f' % (batch_id, l))
                    logger.debug('Learning rate: %.3f'%updated_lr)
                    logger.info('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
                    mean_valid_accuracy = np.mean(valid_accuracy_log[np.max([0,len(valid_accuracy_log)-(policy_interval//2)]):])
                    logger.debug('%d Mean Valid accuracy (Latter half)[%d:%d]: %.1f%%' %
                          (time_stamp,np.max([0,len(valid_accuracy_log)-(policy_interval//2)]),len(valid_accuracy_log), mean_valid_accuracy)
                          )
                    param_count = get_parameter_count()
                    stride_cost = get_stride_cost()
                    logger.debug('%d Parameter count: %d'%(time_stamp,param_count))
                    duration = np.sum(time_log[np.max([0,len(time_log)-(policy_interval//2)]):])
                    logger.debug('%d Time taken (Latter half)[%d:%d]: %.3f'%
                          (time_stamp,np.max([0,len(time_log)-(policy_interval//2)]),len(time_log),duration)
                    )

                    data = {
                        'error_t':100-mean_valid_accuracy,'time_cost':duration,
                        'param_cost':param_count,'num_layers':len(iconv_ops),
                        'stride_cost': stride_cost
                    }
                    logger.info('%d Data for this episode: Error:%.3f, Time:%.3f, Param Count:%d, Layers:%d, Stride:%d'%
                                (time_stamp,data['error_t'],data['time_cost'],data['param_cost'],data['num_layers'],data['stride_cost'])
                                )
                    value_logger.info('%d,%s'%(time_stamp,data))
                    logger.info('=============================================================\n')
                    action = policy_learner.update_policy(time_stamp,data)
                    logger.info('\tReceived action %s'%action)

                    execute_action(action)
                    logits = get_logits(tf_dataset)


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
