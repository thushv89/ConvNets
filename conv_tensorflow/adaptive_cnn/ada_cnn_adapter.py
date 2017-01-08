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

##################################################################
# AdaCNN Adapter
# ===============================================================
# AdaCNN Adapter will adapt the final model proposed by AdaCNN Constructor
# Then runs e-greedy Q-Learn for all the layer to decide of more filters required
# or should remove some. But keep in mind number of layers will be fixed
##################################################################

logger = None
logging_level = logging.DEBUG
logging_format = '[%(funcName)s] %(message)s'

batch_size = 128 # number of datapoints in a single batch
start_lr = 0.0001
decay_learning_rate = True
dropout_rate = 0.25
use_dropout = False
#keep beta small (0.2 is too much >0.002 seems to be fine)
include_l2_loss = False
beta = 1e-5
check_early_stopping_from = 5
accuracy_drop_cap = 3


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

    print('Initializing the iConvNet (conv_global,pool_global,classifier)...')
    for op in cnn_ops:
        if 'conv' in op:
            weights[op] = tf.Variable(tf.truncated_normal(
                cnn_hyps[op]['weights'],
                stddev=2./max(20,cnn_hyps[op]['weights'][0]*cnn_hyps[op]['weights'][1])
            ),name=op+'_weights')
            biases[op] = tf.Variable(tf.constant(
                np.random.random()*0.01,shape=[cnn_hyps[op]['weights'][3]]
            ),name=op+'_bias')

            logger.debug('Weights for %s initialized with size %s',op,str(cnn_hyps[op]['weights']))
            logger.debug('Biases for %s initialized with size %d',op,cnn_hyps[op]['weights'][3])

        if 'fulcon' in op:
            weights['fulcon_out'] = tf.Variable(tf.truncated_normal(
                [cnn_hyps[op]['in'],cnn_hyps[op]['out']],
                stddev=2./cnn_hyps[op]['in']
            ),name=op+'_weights')
            biases[op] = tf.Variable(tf.constant(
                np.random.random()*0.01,shape=[cnn_hyps[op]['out']]
            ),name=op+'_bias')

            logger.debug('Weights for %s initialized with size %d,%d',
                op,cnn_hyps[op]['in'],cnn_hyps[op]['out'])
            logger.debug('Biases for %s initialized with size %d',op,cnn_hyps[op]['out'])

    return weights,biases


def get_logits_with_ops(dataset,cnn_ops,cnn_hyps,weights,biases):
    logger.info('Current set of operations: %s'%cnn_ops)
    act_means = {}
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
            act_means[op]=tf.reduce_mean(x,[0,1,2])

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

    fin_x = get_final_x(cnn_ops,cnn_hyps)
    logger.debug("My calculation, TensorFlow calculation: %d, %d"%(fin_x,shape[1]))
    assert fin_x == shape[1]

    print('Unwrapping last convolution layer %s to %s hidden layer'%(shape,(rows,cnn_hyps['fulcon_out']['in'])))
    x = tf.reshape(x, [rows,cnn_hyps['fulcon_out']['in']])

    return tf.matmul(x, weights['fulcon_out']) + biases['fulcon_out'],act_means


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
    logits,_ = get_logits_with_ops(dataset,cnn_ops,cnn_hyps,weights,biases)
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


def adapt_cnn(cnn_ops,cnn_hyps,states,actions,rolling_activation_means):

    # find the last convolutional layer
    last_conv_id = ''
    for tmp_op in reversed(cnn_ops):
        if 'conv' in tmp_op:
            last_conv_id = tmp_op
            break

    for si,ai in zip(states,actions):
        op = cnn_ops[si[0]]
        logger.debug('Running action %s for op %s',str(ai),op)
        # Dont need to change anything
        if ai[0]!='add' and ai[0]!='remove':
            continue
        # add depth
        elif ai[0]=='add':
            amount_to_add = ai[1]
            logger.debug('Adding %d filter for %s action recieved',amount_to_add,op)
            assert 'conv' in op

            # change both weights and biase in the current op
            op_new_weights = tf.Variable(
                tf.truncated_normal([cnn_hyps[op]['weights'][0],cnn_hyps[op]['weights'][1],
                                     cnn_hyps[op]['weights'][2],amount_to_add],stddev=0.02)
            )
            op_new_bias = tf.Variable(
                tf.truncated_normal([amount_to_add],stddev=0.001)
            )
            tf.initialize_variables([op_new_weights,op_new_bias]).run()

            weights[op]= tf.concat(3,[weights[op],op_new_weights])
            biases[op]= tf.concat(0,[biases[op],op_new_bias])

            # change out hyperparameter of op
            cnn_hyperparameters[op]['weights'][3]+=amount_to_add

            # ================ Changes to next_op ===============
            # Very last convolutional layer
            # this is different from other layers
            # as a change in this require changes to FC layer
            if op==last_conv_id:
                # change FC layer
                fc_new_weights = tf.Variable(
                    tf.truncated_normal([amount_to_add,num_labels],stddev=0.02)
                )
                tf.initialize_variables([fc_new_weights]).run()
                weights['fulcon_out'] = tf.concat(0,[weights['fulcon_out'],fc_new_weights])

                cnn_hyps['fulcon_out']['in']+=amount_to_add

            else:

                # change in hyperparameter of next conv op
                next_conv_op = [tmp_op for tmp_op in cnn_ops[cnn_ops.index(op)+1:] if 'conv' in tmp_op][0]
                assert op!=next_conv_op

                # change only the weights in next conv_op
                next_op_new_weights = tf.Variable(
                    tf.truncated_normal([cnn_hyps[next_conv_op][0],cnn_hyps[next_conv_op][1],amount_to_add,cnn_hyps[next_conv_op][3]],stddev=0.02)
                )
                tf.initialize_variables([next_op_new_weights]).run()
                weights[next_conv_op] = tf.concat(3,[weights[next_conv_op],next_op_new_weights])

                cnn_hyperparameters[next_conv_op]['weights'][2]+=amount_to_add

        elif ai[0]=='remove':

            # this is trickier than adding weights
            # We remove the given number of filters
            # which have the least rolling mean activation averaged over whole map
            amount_to_rmv = ai[1]
            assert 'conv' in op

            indices_of_filters_keep = (np.argsort(rolling_ativation_means[op]).flatten()[amount_to_rmv:]).astype('int32')

            # currently no way to generally slice using gather
            # need to do a transoformation to do this.
            # change both weights and biase in the current op
            w = tf.identity(weights[op])
            w = tf.transpose(w,[3,0,1,2])
            w = tf.gather_nd(w,[[idx] for idx in indices_of_filters_keep])
            w = tf.transpose(w,[1,2,3,0])
            logger.debug('Size after feature map reduction: %s,%s',op,str(w.get_shape().as_list()))
            weights[op]= w
            biases[op]= tf.gather(biases[op],indices_of_filters_keep)

            # change out hyperparameter of op
            cnn_hyperparameters[op]['weights'][3]-=amount_to_rmv
            assert w.get_shape().as_list()==cnn_hyperparameters[op]['weights']

            if op==last_conv_id:
                w = tf.identity(weights['fulcon_out'])
                #w = tf.transpose(w,[1,0])
                w = tf.gather_nd(w,[[idx] for idx in indices_of_filters_keep])
                #w = tf.transpose(w,[1,0])
                weights['fulcon_out'] = w
                cnn_hyps['fulcon_out']['in']-=amount_to_rmv
                logger.debug('Size after feature map reduction: fulcon_out,%s',str(w.get_shape().as_list()))
            else:
                # change in hyperparameter of next conv op
                next_conv_op = [tmp_op for tmp_op in cnn_ops[cnn_ops.index(op)+1:] if 'conv' in tmp_op][0]
                assert op!=next_conv_op

                # change only the weights in next conv_op
                w = tf.identity(weights[next_conv_op])
                w = tf.transpose(w,[2,0,1,3])
                w = tf.gather_nd(w,[[idx] for idx in indices_of_filters_keep])
                w = tf.transpose(w,[1,2,0,3])
                logger.debug('Size after feature map reduction: %s,%s',next_conv_op,str(w.get_shape().as_list()))
                weights[next_conv_op] = w

                cnn_hyperparameters[next_conv_op]['weights'][2]-=amount_to_rmv
                assert w.get_shape().as_list()==cnn_hyperparameters[op]['weights']


def load_data_from_memmap(dataset_info,dataset_filename,label_filename,start_idx,size):
    global logger
    col_count = (dataset_info['image_size'],dataset_info['image_size'],dataset_info['num_channels'])
    logger.info('Processing files %s,%s'%(dataset_filename,label_filename))
    fp1 = np.memmap(dataset_filename,dtype=np.float32,mode='r',
                    offset=np.dtype('float32').itemsize*col_count[0]*col_count[1]*col_count[2]*start_idx,shape=(size,col_count[0],col_count[1],col_count[2]))

    fp2 = np.memmap(label_filename,dtype=np.int32,mode='r',
                    offset=np.dtype('int32').itemsize*1*size,shape=(size,1))

    train_dataset = fp1[:,:,:,:]
    train_labels = fp2[:]

    del fp1,fp2
    return train_dataset,train_labels

if __name__=='__main__':

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
            dataset_filename='data_non_station'+os.sep+'cifar-10-non-station-dataset.pkl'
            label_filename='data_non_station'+os.sep+'cifar-10-non-station-label.pkl'
            dataset_size = 1280000
            chunk_size = 51200

        test_size=10000
        test_dataset_filename='data_non_station'+os.sep+'cifar-10-non-station-test-dataset.pkl'
        test_label_filename = 'data_non_station'+os.sep+'cifar-10-non-station-test-label.pkl'
        fp1 = np.memmap(dataset_filename,dtype=np.float32,mode='r',
                        offset=np.dtype('float32').itemsize*image_size*image_size*num_channels*0,
                        shape=(test_size,image_size,image_size,num_channels))
        fp2 = np.memmap(label_filename,dtype=np.int32,mode='r',
                        offset=np.dtype('int32').itemsize*0,shape=(test_size,1))
        test_dataset = fp1[:,:,:,:]
        test_labels = fp2[:]

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
        fp1 = np.memmap(dataset_filename,dtype=np.float32,mode='r',
                        offset=np.dtype('float32').itemsize*image_size*image_size*num_channels*0,
                        shape=(test_size,image_size,image_size,num_channels))

        fp2 = np.memmap(label_filename,dtype=np.int32,mode='r',
                        offset=np.dtype('int32').itemsize*0,shape=(test_size,1))
        test_dataset = fp1[:,:,:,:]
        test_labels = fp2[:]

    del fp1,fp2

    dataset_info['image_size']=image_size
    dataset_info['num_labels']=num_labels
    dataset_info['num_channels']=num_channels
    dataset_info['dataset_size']=dataset_size
    dataset_info['chunk_size']=chunk_size

    logger = logging.getLogger('main_logger')
    logger.setLevel(logging_level)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(logging_format))
    console.setLevel(logging_level)
    logger.addHandler(console)

    # Loading data
    memmap_idx = 0
    train_dataset,train_labels = None,None

    assert chunk_size%batch_size==0
    batches_in_chunk = chunk_size//batch_size

    train_size = train_dataset.shape[0]
    logger.info('='*80)
    logger.info('\tTrain Size: %d'%train_size)
    logger.info('='*80)

    policy_interval = 10 #number of batches to process for each policy iteration

    #cnn_string = "C,5,1,256#P,3,2,0#P,5,2,0#C,3,1,128#C,5,1,512#C,3,1,64#C,3,1,128#C,5,1,128#Terminate,0,0,0"
    cnn_string = "C,5,1,256#P,5,2,0#Terminate,0,0,0"

    # Resetting this every policy interval
    rolling_data_distribution = {}
    mean_data_distribution = {}
    prev_mean_distribution = {}
    for li in range(num_labels):
        rolling_data_distribution[li]=0.0
        mean_data_distribution[li]=1.0/float(num_labels)
        prev_mean_distribution[li]=1.0/float(num_labels)

    graph = tf.Graph()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.65)
    with tf.Session(graph=graph,
                    config=tf.ConfigProto(allow_soft_placement=True,gpu_options=gpu_options)) as session, \
            tf.device('/gpu:0'):

        cnn_ops,cnn_hyperparameters = get_ops_hyps_from_string(cnn_string)
        weights,biases = initialize_cnn_with_ops(cnn_ops,cnn_hyperparameters)

        # -1 is because we don't want to count pool_global
        layer_count = len([op for op in cnn_ops if 'conv' in op or 'pool' in op])-1

        # Adapting Policy Learner
        adapter = qlearner.AdaCNNAdaptingQLearner(learning_rate=0.1,
                                                  discount_rate=0.99,
                                                  fit_interval = 5,
                                                  filter_upper_bound=1024,
                                                  net_depth=layer_count,
                                                  upper_bound=10,
                                                  epsilon=0.99)

        global_step = tf.Variable(0, trainable=False)

        logger.info('Input data defined...\n')
        # Input train data
        tf_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels),name='TrainDataset')
        tf_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels),name='TrainLabels')

        # Valid data (Next train batch) Unseen
        tf_valid_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels),name='ValidDataset')
        tf_valid_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels),name='ValidLabels')

        logits,act_means = get_logits_with_ops(tf_dataset,cnn_ops,cnn_hyperparameters,weights,biases)
        loss = calc_loss(logits,tf_labels)
        pred = predict_with_logits(logits)
        optimize = optimize_func(loss,global_step)
        inc_gstep = inc_global_step(global_step)

        # valid predict function
        pred_valid = predict_with_dataset(tf_valid_dataset,cnn_ops,cnn_hyperparameters,weights,biases)

        tf.initialize_all_variables().run()

        train_losses = []
        mean_train_loss = 0
        rolling_next_accuracy = 0
        rolling_prev_accuracy = 0

        rolling_ativation_means = {}
        for op in cnn_ops:
            if 'conv' in op:
                rolling_ativation_means[op]=np.zeros([cnn_hyperparameters[op]['weights'][3]])

        decay = 0.9
        act_decay = 0.5
        current_states,current_actions = None,None
        prev_valid_accuracy,next_valid_accuracy = 0,0
        #TODO: Think about decaying learning rate (should or shouldn't)
        for batch_id in range(ceil(dataset_size//batch_size)-1):
            chunk_batch_id = batch_id%batches_in_chunk

            if chunk_batch_id==0:
                train_dataset,train_labels = load_data_from_memmap(dataset_info,dataset_filename,label_filename,memmap_idx,chunk_size+1)
                memmap_idx += chunk_size

            batch_data = train_dataset[chunk_batch_id*batch_size:(chunk_batch_id+1)*batch_size, :, :, :]
            batch_labels = train_labels[chunk_batch_id*batch_size:(chunk_batch_id+1)*batch_size, :]

            feed_dict = {tf_dataset : batch_data, tf_labels : batch_labels}
            _, activation_means, l, (_,updated_lr), predictions = session.run(
                    [logits,act_means,loss,optimize,pred], feed_dict=feed_dict
            )
            if np.isnan(l):
                logger.critical('NaN detected: (batchID)'%chunk_batch_id)

            assert not np.isnan(l)

            # rolling activation mean update
            for op,op_activations in activation_means.items():
                rolling_ativation_means[op]=(1-act_decay)*rolling_ativation_means[op] + decay*activation_means[op]

            train_losses.append(l)

            # validation batch (Unseen)
            batch_valid_data = train_dataset[(chunk_batch_id+1)*batch_size:(chunk_batch_id+2)*batch_size, :, :, :]
            batch_valid_labels = train_labels[(chunk_batch_id+1)*batch_size:(chunk_batch_id+2)*batch_size, :]

            feed_valid_dict = {tf_valid_dataset:batch_valid_data, tf_valid_labels:batch_valid_labels}
            next_valid_predictions = session.run([pred_valid],feed_dict=feed_valid_dict)
            next_valid_accuracy = accuracy(next_valid_predictions[0], batch_valid_labels)
            rolling_next_accuracy = (1-decay)*rolling_next_accuracy + decay*next_valid_accuracy
            # validation batch (Seen) random from last 50 batches
            if batch_id>0:
                prev_valid_batch_id = np.random.randint(max(0,batch_id-50),batch_id)
            else:
                prev_valid_batch_id = 0
            logger.debug('Chose %d as previous validation batch_id',prev_valid_batch_id)
            batch_valid_data = train_dataset[prev_valid_batch_id*batch_size:(prev_valid_batch_id+1)*batch_size, :, :, :]
            batch_valid_labels = train_labels[prev_valid_batch_id*batch_size:(prev_valid_batch_id+1)*batch_size, :]

            feed_valid_dict = {tf_valid_dataset:batch_valid_data, tf_valid_labels:batch_valid_labels}
            prev_valid_predictions = session.run([pred_valid],feed_dict=feed_valid_dict)
            prev_valid_accuracy = accuracy(prev_valid_predictions[0], batch_valid_labels)
            rolling_prev_accuracy = (1-decay)*rolling_prev_accuracy + decay*prev_valid_accuracy

            if batch_id%25==0:
                mean_train_loss = np.mean(train_losses)
                logger.info('='*40)
                logger.info('\tBatch ID: %d'%batch_id)
                logger.info('\tLearning rate: %.5f'%updated_lr)
                logger.info('\tMinibatch Mean Loss: %.3f'%mean_train_loss)
                logger.info('\tValidation Accuracy (Unseen): %.3f'%next_valid_accuracy)
                logger.info('\tValidation Accuracy (Seen): %.3f'%prev_valid_accuracy)

                # reset variables
                mean_train_loss = 0.0
                train_losses = []

            cnt = Counter(np.argmax(batch_labels,axis=1))

            # calculate rolling mean of data distribution (amounts of different classes)
            for li in range(num_labels):
                rolling_data_distribution[li]=(1-decay)*rolling_data_distribution[li] + decay*float(cnt[li])/float(batch_size)
                mean_data_distribution[li] += float(cnt[li])/(float(batch_size)*float(policy_interval))

            # ====================== Policy update and output ======================
            if batch_id>0 and batch_id%policy_interval==0:
                #Update and Get a new policy
                if current_states is not None and current_actions is not None:
                    logger.info('Updating the Policy for')
                    logger.info('\tStates: %s',str(current_states))
                    logger.info('\tActions: %s',str(current_actions))
                    logger.info('\tValide accuracy: %.2f (Unseen) %.2f (seen)',next_valid_accuracy,prev_valid_accuracy)
                    adapter.update_policy({'states':current_states,'actions':current_actions,
                                           'next_accuracy':next_valid_accuracy,'prev_accuracy':prev_valid_accuracy})
                distMSE = 0.0
                for li in range(num_labels):
                    distMSE += (prev_mean_distribution[li]-rolling_data_distribution[li])**2

                filter_dict = {}
                for op_i,op in enumerate(cnn_ops):
                    if 'conv' in op:
                        filter_dict[op_i]=cnn_hyperparameters[op]['weights'][3]
                    elif 'pool' in op:
                        filter_dict[op_i]=0

                current_states,current_actions = adapter.output_trajectory({'distMSE':distMSE,'filter_counts':filter_dict})
                # where all magic happens
                adapt_cnn(cnn_ops,cnn_hyperparameters,current_states,current_actions,rolling_ativation_means)
                # calculating all the things again
                logits,act_means = get_logits_with_ops(tf_dataset,cnn_ops,cnn_hyperparameters,weights,biases)

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

