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
start_lr = 0.005
decay_learning_rate = False
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


def get_logits_with_ops(dataset,cnn_ops,cnn_hyps,weights,biases):
    logger.debug('Current set of operations: %s'%cnn_ops)
    act_means = {}
    x = dataset
    logger.debug('Received data for X(%s)...'%x.get_shape().as_list())

    logger.debug('Adjusting the logit calculation ...')

    #need to calculate the output according to the layers we have
    for op in cnn_ops:
        if 'conv' in op:
            logger.debug('\tConvolving (%s) With Weights:%s Stride:%s'%(op,cnn_hyps[op]['weights'],cnn_hyps[op]['stride']))
            #logger.debug('\t\tX before convolution:%s'%(x.get_shape().as_list()))
            logger.debug('\t\tWeights: %s',tf.shape(weights[op]).eval())
            x = tf.nn.conv2d(x, weights[op], cnn_hyps[op]['stride'], padding=cnn_hyps[op]['padding'])
            #logger.debug('\t\t Relu with x(%s) and b(%s)'%(tf.shape(x).eval(),tf.shape(biases[op]).eval()))
            x = tf.nn.relu(x + biases[op])
            #logger.debug('\t\tX after %s:%s'%tf.shape(weights[op]).eval())
            act_means[op]=tf.reduce_mean(x,[0,1,2])

        if 'pool' in op:
            logger.debug('\tPooling (%s) with Kernel:%s Stride:%s'%(op,cnn_hyps[op]['kernel'],cnn_hyps[op]['stride']))
            if cnn_hyps[op]['type'] is 'max':
                x = tf.nn.max_pool(x,ksize=cnn_hyps[op]['kernel'],strides=cnn_hyps[op]['stride'],padding=cnn_hyps[op]['padding'])
            elif cnn_hyps[op]['type'] is 'avg':
                x = tf.nn.avg_pool(x,ksize=cnn_hyps[op]['kernel'],strides=cnn_hyps[op]['stride'],padding=cnn_hyps[op]['padding'])

            #logger.debug('\t\tX after %s:%s'%(op,tf.shape(x).eval()))

        if 'fulcon' in op:
            break

    # we need to reshape the output of last subsampling layer to
    # convert 4D output to a 2D input to the hidden layer
    # e.g subsample layer output [batch_size,width,height,depth] -> [batch_size,width*height*depth]


    fin_x = get_final_x(cnn_ops,cnn_hyps)

    #print('Unwrapping last convolution layer %s to %s hidden layer'%(shape,(batch_size,cnn_hyps['fulcon_out']['in'])))
    print('fulcon in : %d'%cnn_hyps['fulcon_out']['in'])
    x = tf.reshape(x, [batch_size,cnn_hyps['fulcon_out']['in']])

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
        learning_rate = tf.constant(start_lr,dtype=tf.float32,name='learning_rate')

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=start_lr).minimize(loss)
    return optimizer,learning_rate


def optimize_with_variable_func(loss,global_step,variables):
    # Optimizer.
    if decay_learning_rate:
        learning_rate = tf.train.exponential_decay(start_lr, global_step,decay_steps=1,decay_rate=0.95)
    else:
        learning_rate = tf.constant(start_lr,dtype=tf.float32,name='learning_rate')

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss,var_list=variables)
    return optimizer,learning_rate


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


def adapt_cnn_with_tf_ops(cnn_ops,cnn_hyps,si,ai,layer_activations):
    global logger,weights,biases

    update_ops = []
    tf_new_weights = {} # tensors to add for each 'add' action
    tf_new_biases = {}
    changed_variables = []
    changed_ops = []
    # find the last convolutional layer
    last_conv_id = ''
    for tmp_op in reversed(cnn_ops):
        if 'conv' in tmp_op:
            last_conv_id = tmp_op
            break

    op = cnn_ops[si[0]]
    logger.debug('Running action %s for op %s',str(ai),op)
    # Dont need to change anything
    if ai[0]!='add' and ai[0]!='remove':
        return
    # add depth
    elif ai[0]=='add':
        amount_to_add = ai[1]
        logger.debug('Adding %d filter for %s action recieved',amount_to_add,op)
        assert 'conv' in op

        # change both weights and biase in the current op
        logger.debug('Concatting %s(old) and %s(new)',str(tf.shape(weights[op]).eval()),
                     str([cnn_hyps[op]['weights'][0],cnn_hyps[op]['weights'][1],cnn_hyps[op]['weights'][2],amount_to_add]))
        tf_new_weights[op] = tf.concat(3,[weights[op],
                                            tf.truncated_normal([cnn_hyps[op]['weights'][0],
                                                                 cnn_hyps[op]['weights'][1],cnn_hyps[op]['weights'][2],amount_to_add],stddev=0.02)
                                            ])
        tf_new_biases[op]= tf.concat(0,[biases[op],tf.truncated_normal([amount_to_add],stddev=0.001)])

        logger.debug('Summary of changes to weights of %s ...',op)
        logger.debug('\tNew Weights: %s',str(tf.shape(tf_new_weights[op]).eval()))

        update_ops.append(tf.assign(weights[op], tf_new_weights[op], validate_shape=False))
        update_ops.append(tf.assign(biases[op], tf_new_biases[op], validate_shape = False))

        changed_variables.extend([weights[op],biases[op]])
        changed_ops.append(op)
        # change out hyperparameter of op
        cnn_hyps[op]['weights'][3]+=amount_to_add
        #weights[op].set_shape(cnn_hyps[op]['weights'])
        #biases[op].set_shape([cnn_hyps[op]['weights'][3]])
        assert cnn_hyps[op]['weights'][2]==tf_new_weights[op].get_shape().as_list()[2]

        # ================ Changes to next_op ===============
        # Very last convolutional layer
        # this is different from other layers
        # as a change in this require changes to FC layer
        if op==last_conv_id:
            # change FC layer
            tf_new_weights['fulcon_out'] = tf.concat(0,[weights['fulcon_out'],tf.truncated_normal([amount_to_add,num_labels],stddev=0.02)])

            logger.debug('Summary of changes to weights of fulcon_out')
            logger.debug('\tCurrent Weights: %s',str(tf.shape(weights['fulcon_out']).eval()))
            logger.debug('\tNew Weights: %s',str(tf.shape(tf_new_weights['fulcon_out']).eval()))

            update_ops.append(tf.assign(weights['fulcon_out'], tf_new_weights['fulcon_out'], validate_shape= False))

            changed_variables.append(weights['fulcon_out'])
            changed_ops.append('fulcon_out')

            cnn_hyps['fulcon_out']['in']+=amount_to_add
            #weights['fulcon_out'].set_shape(cnn_hyps['fulcon_out']['in'])
            logger.debug('%s in: %d','fulcon_out',cnn_hyperparameters['fulcon_out']['in'])
        else:

            # change in hyperparameter of next conv op
            next_conv_op = [tmp_op for tmp_op in cnn_ops[cnn_ops.index(op)+1:] if 'conv' in tmp_op][0]
            assert op!=next_conv_op

            # change only the weights in next conv_op
            tf_new_weights[next_conv_op]=tf.concat(2,[weights[next_conv_op],
                                          tf.truncated_normal([cnn_hyps[next_conv_op]['weights'][0],cnn_hyps[next_conv_op]['weights'][1],
                                                               amount_to_add,cnn_hyps[next_conv_op]['weights'][3]],stddev=0.02)])
            logger.debug('Summary of changes to weights of %s',next_conv_op)
            logger.debug('\tCurrent Weights: %s',str(tf.shape(weights[next_conv_op]).eval()))
            logger.debug('\tNew Weights: %s',str(tf.shape(tf_new_weights[next_conv_op]).eval()))

            update_ops.append(tf.assign(weights[next_conv_op], tf_new_weights[next_conv_op], validate_shape=False))

            changed_variables.append(weights[next_conv_op])
            changed_ops.append(next_conv_op)

            cnn_hyps[next_conv_op]['weights'][2]+=amount_to_add
            #weights[next_conv_op].set_shape(cnn_hyps[next_conv_op]['weights'])
            assert cnn_hyps[next_conv_op]['weights'][2]==tf.shape(tf_new_weights[next_conv_op]).eval()[2]

    elif ai[0]=='remove':

        # this is trickier than adding weights
        # We remove the given number of filters
        # which have the least rolling mean activation averaged over whole map
        amount_to_rmv = ai[1]
        assert 'conv' in op

        indices_of_filters_keep = (np.argsort(layer_activations).flatten()[amount_to_rmv:]).astype('int32')

        # currently no way to generally slice using gather
        # need to do a transoformation to do this.
        # change both weights and biase in the current op

        tf_new_weights[op]=tf.transpose(weights[op],[3,0,1,2])
        tf_new_weights[op] = tf.gather_nd(tf_new_weights[op],[[idx] for idx in indices_of_filters_keep])
        tf_new_weights[op] = tf.transpose(tf_new_weights[op],[1,2,3,0])
        logger.debug('Size after feature map reduction: %s,%s',op,tf.shape(tf_new_weights[op]).eval())
        update_ops.append(tf.assign(weights[op],tf_new_weights[op],validate_shape=False))

        tf_new_biases[op]=tf.gather(biases[op],indices_of_filters_keep)
        update_ops.append(tf.assign(biases[op],tf_new_biases[op],validate_shape=False))

        changed_variables.extend([weights[op],biases[op]])
        changed_ops.append(op)

        # change out hyperparameter of op
        cnn_hyps[op]['weights'][3]-=amount_to_rmv

        #weights[op].set_shape(cnn_hyps[op]['weights'])
        #biases[op].set_shape([cnn_hyps[op]['weights'][3]])
        assert tf.shape(tf_new_weights[op]).eval()[3]==cnn_hyperparameters[op]['weights'][3]

        if op==last_conv_id:

            tf_new_weights['fulcon_out'] = tf.gather_nd(weights['fulcon_out'],[[idx] for idx in indices_of_filters_keep])

            update_ops.append(tf.assign(weights['fulcon_out'],tf_new_weights['fulcon_out'],validate_shape=False))

            changed_variables.append(weights['fulcon_out'])
            changed_ops.append('fulcon_out')

            cnn_hyps['fulcon_out']['in']-=amount_to_rmv
            logger.debug('Size after feature map reduction: fulcon_out,%s',str(tf.shape(tf_new_weights['fulcon_out']).eval()))

        else:
            # change in hyperparameter of next conv op
            next_conv_op = [tmp_op for tmp_op in cnn_ops[cnn_ops.index(op)+1:] if 'conv' in tmp_op][0]
            assert op!=next_conv_op

            # change only the weights in next conv_op

            tf_new_weights[next_conv_op] = tf.transpose(weights[next_conv_op],[2,0,1,3])
            tf_new_weights[next_conv_op] = tf.gather_nd(tf_new_weights[next_conv_op],[[idx] for idx in indices_of_filters_keep])
            tf_new_weights[next_conv_op] = tf.transpose(tf_new_weights[next_conv_op],[1,2,0,3])

            logger.debug('Size after feature map reduction: %s,%s',next_conv_op,str(tf.shape(tf_new_weights[next_conv_op]).eval()))
            update_ops.append(tf.assign(weights[next_conv_op],tf_new_weights[next_conv_op],validate_shape=False))
            changed_variables.append(weights[next_conv_op])
            changed_ops.append(next_conv_op)

            cnn_hyps[next_conv_op]['weights'][2]-=amount_to_rmv
            logger.debug('')
            assert tf.shape(tf_new_weights[next_conv_op]).eval()[2] ==cnn_hyperparameters[next_conv_op]['weights'][2]

    return update_ops,changed_variables,changed_ops

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

weights,biases = None,None

research_parameters = \
    {'save_train_test_images':False,
     'log_class_distribution':True,'log_distribution_every':128,
     'adapt_structure' : True
     }

if __name__=='__main__':
    global weights,biases
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

    if research_parameters['log_class_distribution']:
        class_dist_logger = logging.getLogger('class_dist_logger')
        class_dist_logger.setLevel(logging.INFO)
        class_distHandler = logging.FileHandler('class_distribution.log', mode='w')
        class_distHandler.setFormatter(logging.Formatter('%(message)s'))
        class_dist_logger.addHandler(class_distHandler)

    # Loading data
    memmap_idx = 0
    train_dataset,train_labels = None,None

    test_dataset,test_labels = load_data_from_memmap(dataset_info,test_dataset_filename,test_label_filename,0,test_size)

    assert chunk_size%batch_size==0
    batches_in_chunk = chunk_size//batch_size

    logger.info('='*80)
    logger.info('\tTrain Size: %d'%dataset_size)
    logger.info('='*80)

    policy_interval = 10 #number of batches to process for each policy iteration

    #cnn_string = "C,5,1,256#P,3,2,0#P,5,2,0#C,3,1,128#C,5,1,512#C,3,1,64#C,3,1,128#C,5,1,128#Terminate,0,0,0"
    cnn_string = "C,3,1,128#P,5,2,0#C,5,1,128#C,3,1,512#C,5,1,128#C,5,1,256#P,2,2,0#C,5,1,64#Terminate,0,0,0"
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

        global_step = tf.Variable(0, dtype=tf.int32,trainable=False,name='global_step')

        logger.info('Input data defined...\n')
        # Input train data
        tf_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels),name='TrainDataset')
        tf_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels),name='TrainLabels')

        # Pool data
        tf_pool_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels),name='PoolDataset')
        tf_pool_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels),name='PoolLabels')

        # Valid data (Next train batch) Unseen
        tf_valid_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels),name='ValidDataset')
        tf_valid_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels),name='ValidLabels')

        # Test data (Global)
        tf_test_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels),name='TestDataset')
        tf_test_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels),name='TestLabels')

        init_op = tf.global_variables_initializer()
        _ = session.run(init_op)

        logits,act_means = get_logits_with_ops(tf_dataset,cnn_ops,cnn_hyperparameters,weights,biases)
        loss = calc_loss(logits,tf_labels)
        loss_vec = calc_loss_vector(logits,tf_labels)
        pred = predict_with_logits(logits)
        optimize = optimize_func(loss,global_step)
        inc_gstep = inc_global_step(global_step)

        init_op = tf.global_variables_initializer()
        _ = session.run(init_op)

        # valid predict function
        pred_valid = predict_with_dataset(tf_valid_dataset,cnn_ops,cnn_hyperparameters,weights,biases)
        pred_test = predict_with_dataset(tf_test_dataset,cnn_ops,cnn_hyperparameters,weights,biases)

        logger.info('Tensorflow functions defined')
        logger.info('Variables initialized')

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

        logger.info('Starting Training Phase')
        #TODO: Think about decaying learning rate (should or shouldn't)
        for batch_id in range(ceil(dataset_size//batch_size)-3):

            logger.debug('\tTraining with batch %d',batch_id)
            for op in cnn_ops:
                if 'pool' not in op:
                    assert weights[op].name in [v.name for v in tf.trainable_variables()]

            chunk_batch_id = batch_id%batches_in_chunk

            if chunk_batch_id==0:
                # We load 1 extra batch (chunk_size+1) because we always make the valid batch the batch_id+1
                train_dataset,train_labels = load_data_from_memmap(dataset_info,dataset_filename,label_filename,memmap_idx,chunk_size+batch_size)
                memmap_idx += chunk_size
                logger.info('Loading dataset chunk of size: %d',train_dataset.shape[0])
                logger.info('\tNext memmap start index: %d',memmap_idx)
            batch_data = train_dataset[chunk_batch_id*batch_size:(chunk_batch_id+1)*batch_size, :, :, :]
            batch_labels = train_labels[chunk_batch_id*batch_size:(chunk_batch_id+1)*batch_size, :]

            feed_dict = {tf_dataset : batch_data, tf_labels : batch_labels}
            _, activation_means, l,l_vec, (_,updated_lr), predictions = session.run(
                    [logits,act_means,loss,loss_vec,optimize,pred], feed_dict=feed_dict
            )

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
            if np.random.random()<0.1:
                hard_pool.add_hard_examples(batch_data,batch_labels,l_vec,hardness)
            assert not np.isnan(l)

            # rolling activation mean update
            for op,op_activations in activation_means.items():
                rolling_ativation_means[op]=(1-act_decay)*rolling_ativation_means[op] + decay*activation_means[op]

            train_losses.append(l)

            # validation batch (Unseen)
            batch_valid_data = train_dataset[(chunk_batch_id+1)*batch_size:(chunk_batch_id+2)*batch_size, :, :, :]
            batch_valid_labels = train_labels[(chunk_batch_id+1)*batch_size:(chunk_batch_id+2)*batch_size, :]

            feed_valid_dict = {tf_valid_dataset:batch_valid_data, tf_valid_labels:batch_valid_labels}
            next_valid_predictions = session.run(pred_valid,feed_dict=feed_valid_dict)
            next_valid_accuracy = accuracy(next_valid_predictions, batch_valid_labels)
            rolling_next_accuracy = (1-decay)*rolling_next_accuracy + decay*next_valid_accuracy
            # validation batch (Seen) random from last 50 batches
            if chunk_batch_id>0:
                prev_valid_batch_id = np.random.randint(max(0,chunk_batch_id-50),chunk_batch_id)
            else:
                prev_valid_batch_id = 0

            batch_valid_data = train_dataset[prev_valid_batch_id*batch_size:(prev_valid_batch_id+1)*batch_size, :, :, :]
            batch_valid_labels = train_labels[prev_valid_batch_id*batch_size:(prev_valid_batch_id+1)*batch_size, :]

            feed_valid_dict = {tf_valid_dataset:batch_valid_data, tf_valid_labels:batch_valid_labels}
            prev_valid_predictions = session.run(pred_valid,feed_dict=feed_valid_dict)
            prev_valid_accuracy = accuracy(prev_valid_predictions, batch_valid_labels)
            rolling_prev_accuracy = (1-decay)*rolling_prev_accuracy + decay*prev_valid_accuracy

            if batch_id%25==0:
                mean_train_loss = np.mean(train_losses)
                logger.info('='*60)
                logger.info('\tBatch ID: %d'%batch_id)
                logger.info('\tLearning rate: %.5f'%updated_lr)
                logger.info('\tMinibatch Mean Loss: %.3f'%mean_train_loss)
                logger.info('\tValidation Accuracy (Unseen): %.3f'%next_valid_accuracy)
                logger.info('\tValidation Accuracy (Seen): %.3f'%prev_valid_accuracy)

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
                        logger.debug('='*80)
                        logger.debug('Predicted Test Labels %d',test_batch_id)
                        logger.debug(np.argmax(test_predictions,axis=1).flatten()[:5])
                        logger.debug('='*80)
                        logger.debug('Test: %d, %.3f',test_batch_id,accuracy(test_predictions, batch_test_labels))

                logger.info('\tTest Accuracy: %.3f'%np.mean(test_accuracies))
                logger.info('='*60)
                logger.info('')

                if research_parameters['save_train_test_images']:
                    local_dir = 'saved_images'+str(batch_id)
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

            if research_parameters['adapt_structure']:
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
                        logger.info('\tValid accuracy: %.2f (Unseen) %.2f (seen)',next_valid_accuracy,prev_valid_accuracy)
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
                    # where all magic happens (adding and removing filters)
                    for si,ai in zip(current_states,current_actions):
                        current_op = cnn_ops[si[0]]
                        if 'conv' in current_op and (ai[0]=='add' or ai[0]=='remove'):

                            update_ops, changed_vars,chaged_ops = adapt_cnn_with_tf_ops(cnn_ops, cnn_hyperparameters,
                                                                        si, ai, rolling_ativation_means[current_op])
                            _ = session.run(update_ops)

                            changed_var_names = [v.name for v in changed_vars]

                            logger.debug('All variable names: %s', str([v.name for v in tf.all_variables()]))
                            logger.debug('Changed var names: %s', str(changed_var_names))
                            changed_var_values = session.run(changed_vars)
                            logger.debug('Changed Variable Values')
                            for n, v in zip(changed_var_names,changed_var_values):
                                logger.debug('\t%s,%s',n,str(v.shape))

                            # redefine tensorflow ops as they changed sizes
                            logger.info('Redefining Tensorflow ops (logits, loss, optimize,...')
                            logits, act_means = get_logits_with_ops(tf_dataset, cnn_ops,
                                                            cnn_hyperparameters, weights, biases)
                            loss = calc_loss(logits, tf_labels)
                            loss_vec = calc_loss_vector(logits, tf_labels)
                            pred = predict_with_logits(logits)
                            optimize = optimize_func(loss, global_step)
                            pred_valid = predict_with_dataset(tf_valid_dataset, cnn_ops,
                                                              cnn_hyperparameters, weights, biases)
                            pred_test = predict_with_dataset(tf_test_dataset, cnn_ops,
                                                             cnn_hyperparameters, weights, biases)

                            logger.debug('')

                    # pooling takes place here
                    pool_logits,_ = get_logits_with_ops(tf_pool_dataset,cnn_ops,cnn_hyperparameters,weights,biases)
                    pool_loss = calc_loss(pool_logits,tf_pool_labels)
                    optimize_with_variable,upd_lr = optimize_with_variable_func(pool_loss,global_step,[weights[op],biases[op]])

                    pool_dataset,pool_labels = hard_pool.get_pool_data()['pool_dataset'],hard_pool.get_pool_data()['pool_labels']
                    for si,ai in zip(current_states,current_actions):
                        if ai[0]=='finetune':
                            op = cnn_ops[si[0]]
                            logger.debug('Only tuning following variables...')
                            logger.debug('\t%s,%s',weights[op].name,str(weights[op]))
                            logger.debug('\t%s,%s',biases[op].name,str(biases[op]))
                            assert weights[op].name in [v.name for v in tf.trainable_variables()]
                            assert biases[op].name in [v.name for v in tf.trainable_variables()]

                            for pool_id in range((hard_pool.get_size()//batch_size)-1):
                                pbatch_data = pool_dataset[pool_id*batch_size:(pool_id+1)*batch_size, :, :, :]
                                pbatch_labels = pool_labels[pool_id*batch_size:(pool_id+1)*batch_size, :]
                                pool_feed_dict = {tf_pool_dataset:pbatch_data,tf_pool_labels:pbatch_labels}
                                _, _, _ = session.run([pool_logits,pool_loss,optimize_with_variable],feed_dict=pool_feed_dict)

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

