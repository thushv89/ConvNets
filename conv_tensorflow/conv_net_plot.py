__author__ = 'Thushan Ganegedara'
# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import numpy as np
import sys
import logging
import getopt
import os
import load_data
from math import floor,ceil
'''=========================================================================
Batch size: 16
Depths:  {'conv_3': 32, 'conv_2': 64, 'iconv_1x1': 16, 'conv_1': 32, 'iconv_3x3': 16, 'iconv_5x5': 16}
Num Steps:  10001
Decay Learning Rate:  True ,  0.1
Dropout:  False ,  0.25
Early Stopping:  False
Include L2, Beta:  True ,  1e-10

So far worked (Normal CNN)
'conv_1'      'pool_1'     'conv_2'     'pool_1'    'conv_3'     'pool_2'     'fulcon_hidden_1','fulcon_out'
conv (3x3) > pool (2x2) > conv (3x3) > pool (2x2) > conv (3x3) > pool (2x2) (Subsampling layers)
1024->512->10 (hidden layers)

Inception CNN
'conv_1','pool_1','conv_2','pool_1','incept_1','pool_2','fulcon_hidden_1','fulcon_out'
=========================================================================='''

data_filename = ''
log_suffix = ''
datatype = 'cifar-10'
if datatype=='cifar-10':
    image_size = 32
    num_labels = 10
    num_channels = 3 # rgb
elif datatype=='notMNIST':
    image_size = 28
    num_labels = 10
    num_channels = 1 # grayscale

batch_size = 16
patch_size = 3

num_epochs = 1000

start_lr = 0.005
decay_learning_rate = True

#dropout seems to be making it impossible to learn
#maybe only works for large nets
dropout_rate = 0.25
in_dropout_rate = 0.2
use_dropout = True

early_stopping = False
accuracy_drops_cap = 10

total_iterations = 0

#seems having SQRT(2/n_in) as the stddev gives better weight initialization
#I used constant of 0.1 and tried weight_init_factor=0.01 but non of them worked
#lowering this makes the CNN impossible to learn
weight_init_factor = 1

include_l2_loss = True
#1e-4 : 1e-5 for 2 conv layers
#1e-7 : 1e-8 for 3 conv layers
beta = 5e-8

#making bias small seems to be helpful (pref 0)

#--------------------- SUBSAMPLING OPERATIONS and THERE PARAMETERS -------------------------------------------------#
#conv_ops = ['conv_1','pool_1','conv_2','pool_2','conv_3','pool_2','incept_1','pool_3','fulcon_hidden_1','fulcon_hidden_2','fulcon_out']
conv_ops = [
    'conv_1','pool_1',
    'conv_2','pool_1',
    'conv_3','pool_1',
    'conv_4',
    'conv_5','pool_1',
    'conv_6',
    'conv_7',
    'pool_global','fulcon_out'
            ]

#number of feature maps for each convolution layer
depth_conv = {'conv_1':64,'conv_2':128,'conv_3':256,'conv_4':512,'conv_5':1024,'conv_6':1024,'conv_7':1024}
incept_orders = {'incept_1':['ipool_2x2','iconv_1x1','iconv_3x3','iconv_5x5']}

maxout,maxout_bank_size = False,1

#weights (conv): [width,height,in_depth,out_depth]
#kernel (pool): [_,width,height,_]
conv_1_hyparams = {'weights':[7,7,num_channels,depth_conv['conv_1']],'stride':[1,2,2,1],'padding':'SAME'}
conv_2_hyparams = {'weights':[3,3,depth_conv['conv_1'],depth_conv['conv_2']],'stride':[1,1,1,1],'padding':'SAME'}
conv_3_hyparams = {'weights':[3,3,depth_conv['conv_2'],depth_conv['conv_3']],'stride':[1,1,1,1],'padding':'SAME'}
conv_4_hyparams = {'weights':[3,3,depth_conv['conv_3'],depth_conv['conv_4']],'stride':[1,1,1,1],'padding':'SAME'}
conv_5_hyparams = {'weights':[3,3,depth_conv['conv_4'],depth_conv['conv_5']],'stride':[1,1,1,1],'padding':'SAME'}
conv_6_hyparams = {'weights':[3,3,depth_conv['conv_5'],depth_conv['conv_6']],'stride':[1,1,1,1],'padding':'SAME'}
conv_7_hyparams = {'weights':[3,3,depth_conv['conv_6'],depth_conv['conv_7']],'stride':[1,1,1,1],'padding':'SAME'}
pool_1_hyparams = {'type':'max','kernel':[1,3,3,1],'stride':[1,2,2,1],'padding':'SAME'}
pool_2_hyparams = {'type':'max','kernel':[1,3,3,1],'stride':[1,2,2,1],'padding':'SAME'}
pool_3_hyparams = {'type':'avg','kernel':[1,3,3,1],'stride':[1,2,2,1],'padding':'SAME'}
pool_global_hyparams = {'type':'avg','kernel':[1,7,7,1],'stride':[1,1,1,1],'padding':'SAME'}



# fully connected layer hyperparameters
hidden_1_hyparams = {'in':0,'out':1024}
hidden_2_hyparams = {'in':1024,'out':512}
out_hyparams = {'in':1024,'out':100}

hyparams = {'conv_1': conv_1_hyparams, 'conv_2': conv_2_hyparams, 'conv_3':conv_3_hyparams,'conv_4':conv_4_hyparams,'conv_5':conv_5_hyparams,'conv_6':conv_6_hyparams,'conv_7':conv_7_hyparams,
           'pool_1': pool_1_hyparams, 'pool_2':pool_2_hyparams, 'pool_3':pool_3_hyparams,'pool_global':pool_global_hyparams,
           'fulcon_hidden_1':hidden_1_hyparams,'fulcon_hidden_2': hidden_2_hyparams, 'fulcon_out':out_hyparams}


#=====================================================================================================================#

def pretty_print_convnet():
    structure = ''
    for op in conv_ops:
        if 'conv' in op:
            structure += 'Convolution Layer: ' + op + '\n'
            structure += '\tWeights shape: ' + str(hyparams[op]['weights']) + '\n'
            structure += '\tStride: ' + str(hyparams[op]['stride']) + '\n'
            structure += '\tPadding: ' + str(hyparams[op]['padding']) + '\n'
        elif 'pool' in op:
            structure += 'Pooling Layer: ' + op + '\n'
            structure += '\tType: ' + str(hyparams[op]['type']) + '\n'
            structure += '\tKernel shape: ' + str(hyparams[op]['kernel']) + '\n'
            structure += '\tStride: ' + str(hyparams[op]['stride']) + '\n'
            structure += '\tPadding: ' + str(hyparams[op]['padding']) + '\n'
        elif 'loc_res_norm' in op:
            structure += 'Local Response Normalization: ' + op + '\n'
        elif 'fulcon' in op:
            structure += 'Fully Connected Layer: ' + op + '\n'
            structure += '\tIn: ' + str(hyparams[op]['in']) + '\n'
            structure += '\tOut: ' + str(hyparams[op]['out']) + '\n'

    return structure

train_dataset, train_labels = None,None
valid_dataset, valid_labels = None,None
test_dataset, test_labels = None,None

tf_dataset = None
tf_labels = None

weights,biases = {},{}

valid_size,train_size,test_size = 0,0,0

def lrelu(x, leak=0.2, name="lrelu"):
     with tf.variable_scope(name):
         f1 = 0.5 * (1 + leak)
         f2 = 0.5 * (1 - leak)
         return f1 * x + f2 * abs(x)

def accuracy(predictions, labels):
    assert predictions.shape[0]==labels.shape[0]
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


def create_subsample_layers():
    print('Defining parameters ...')

    for op in conv_ops:
        if 'fulcon' in op:
            #we don't create weights biases for fully connected layers because we need to calc the
            #fan_out of the last convolution/pooling (subsampling) layer
            #as that's gonna be fan_in for the 1st hidden layer
            break
        if 'conv' in op:
            print('\tDefining weights and biases for %s (weights:%s)'%(op,hyparams[op]['weights']))
            print('\t\tWeights:%s'%hyparams[op]['weights'])
            print('\t\tBias:%d'%hyparams[op]['weights'][3])
            weights[op]=tf.Variable(
                tf.truncated_normal(hyparams[op]['weights'],
                                    stddev=2./min(50,10*hyparams[op]['weights'][0]*hyparams[op]['weights'][1])
                                    )
            )
            biases[op] = tf.Variable(tf.constant(np.random.random()*1e-5,shape=[hyparams[op]['weights'][3]]))
        if 'incept' in op:
            print('\n\tDefining the weights and biases for the Incept Module')
            inc_hyparams = hyparams[op]
            for k,v in inc_hyparams.items():
                if 'conv' in k:
                    w_key = op+'_'+k
                    print('\t\tParameters for %s'%w_key)
                    print('\t\t\tWeights:%s'%inc_hyparams[k]['weights'])
                    print('\t\t\tBias:%d'%inc_hyparams[k]['weights'][3])
                    weights[w_key] = tf.Variable(
                        tf.truncated_normal(inc_hyparams[k]['weights'],
                                            stddev=2./min(50,inc_hyparams[k]['weights'][0]*inc_hyparams[k]['weights'][1])
                                            )
                    )
                    biases[w_key] = tf.Variable(tf.constant(np.random.random()*0.0,shape=[inc_hyparams[k]['weights'][3]]))

def create_fulcon_layers(fan_in):
    for op in conv_ops:
        if 'fulcon' in op:
            hyparams[op]['in'] = fan_in
            break

    for op in conv_ops:
            if 'fulcon' not in op:
                continue
            else:
                if op in weights and op in biases:
                    break

                weights[op] = tf.Variable(
                    tf.truncated_normal(
                        [hyparams[op]['in'],hyparams[op]['out']],stddev=2./hyparams[op]['in']
                    )
                )

                biases[op] = tf.Variable(tf.constant(np.random.random()*1e-5,shape=[hyparams[op]['out']]))


def get_logits(dataset,is_train):

    # Variables.
    if use_dropout and is_train:
        x = tf.nn.dropout(dataset,1.0 - in_dropout_rate,seed=tf.set_random_seed(98765))
    else:
        x = dataset

    print('Calculating inputs for data X(%s)...'%x.get_shape().as_list())
    for op in conv_ops:
        if 'conv' in op:
            print('\tCovolving data (%s)'%op)
            x = tf.nn.conv2d(x, weights[op], hyparams[op]['stride'], padding=hyparams[op]['padding'])

            if maxout:
                tf_maxout_x = None
                for max_i in range(depth_conv[op]//maxout_bank_size):
                    tmp_x = tf.reduce_max(x[:,:,:,max_i*maxout_bank_size:(max_i+1)*maxout_bank_size], reduction_indices=[3], keep_dims=True, name=None)
                    if tf_maxout_x is None:
                        tf_maxout_x = tf.identity(tmp_x)
                    else:
                        tf_maxout_x = tf.concat(3,[tf_maxout_x,tmp_x])

                x = tf_maxout_x
                print('\t\tX after maxout :%s'%(tf_maxout_x.get_shape().as_list()))
            else:
                x = lrelu(x + biases[op])
            print('\t\tX after %s:%s'%(op,x.get_shape().as_list()))
        if 'pool' in op:
            print('\tPooling data')
            if hyparams[op]['type']=='max':
                x = tf.nn.max_pool(x,ksize=hyparams[op]['kernel'],strides=hyparams[op]['stride'],padding=hyparams[op]['padding'])
            elif hyparams[op]['type']=='avg':
                x = tf.nn.avg_pool(x,ksize=hyparams[op]['kernel'],strides=hyparams[op]['stride'],padding=hyparams[op]['padding'])
            print('\t\tX after %s:%s'%(op,x.get_shape().as_list()))
        if op=='loc_res_norm':
            print('\tLocal Response Normalization')
            x = tf.nn.local_response_normalization(x, depth_radius=3, bias=None, alpha=1e-2, beta=0.75)
        if 'incept' in op:
            print('\tInception for data ...')

            tf_incept_out = None

            conv_1x1_id = op + '_' + 'iconv_1x1'
            for inc_op in incept_orders[op]:
                inc_op_id = op + '_' + inc_op

                if 'pool' in inc_op:
                    print('\t\tPooling %s'%inc_op_id)
                    # pooling followed by 1x1 convolution
                    tmp_x = tf.nn.avg_pool(x,
                                           ksize=hyparams[op][inc_op]['kernel'],
                                           strides=hyparams[op][inc_op]['stride'],
                                           padding=hyparams[op][inc_op]['padding']
                                           )

                    #1x1 convolution with iconv_1x1

                    tmp_x = tf.nn.conv2d(
                        tmp_x,weights[conv_1x1_id],
                        hyparams[op]['iconv_1x1']['stride'],
                        padding=hyparams[op]['iconv_1x1']['padding']
                    )
                    # relu activation
                    tmp_x = lrelu(tmp_x + biases[conv_1x1_id])

                    if tf_incept_out is None:
                        tf_incept_out = tf.identity(tmp_x)
                    else:
                        tf_incept_out = tf.concat(3,[tf_incept_out,tmp_x])

                    print('\n\t\tStacked input of Inception module after %s, %s'%(inc_op,tf_incept_out.get_shape().as_list()))
                if 'conv' in inc_op:
                    print('\t\tConvolving %s'%inc_op_id)

                    # no following convolution after 1x1 convolution
                    if inc_op=='iconv_1x1':
                        tmp_x = tf.nn.conv2d(x,
                                             weights[conv_1x1_id],
                                             hyparams[op]['iconv_1x1']['stride'],
                                             padding=hyparams[op]['iconv_1x1']['padding']
                                             )
                        #relu activation
                        tmp_x = lrelu(tmp_x + biases[conv_1x1_id])

                        if tf_incept_out is None:
                            tf_incept_out = tf.identity(tmp_x)
                        else:
                            tf_incept_out = tf.concat(3,[tf_incept_out,tmp_x])
                        print('\n\t\tStacked input of Inception module after %s, %s'%(inc_op,tf_incept_out.get_shape().as_list()))

                    else:
                        # 1x1 convolution
                        tmp_x = tf.nn.conv2d(x,
                                             weights[conv_1x1_id],
                                             hyparams[op]['iconv_1x1']['stride'],
                                             padding=hyparams[op]['iconv_1x1']['padding']
                                             )
                        #relu activation
                        tmp_x = lrelu(tmp_x + biases[conv_1x1_id])

                        #5x5 or 3x3 convolution
                        tmp_x = tf.nn.conv2d(tmp_x,
                                             weights[inc_op_id],
                                             hyparams[op][inc_op]['stride'],
                                             padding=hyparams[op][inc_op]['padding']
                        )
                        #relu activation
                        tmp_x = lrelu(tmp_x + biases[inc_op_id])

                        if tf_incept_out is None:
                            tf_incept_out = tf.identity(tmp_x)
                        else:
                            tf_incept_out = tf.concat(3,[tf_incept_out,tmp_x])
                        print('\n\t\tStacked input of Inception module after %s, %s'%(inc_op,tf_incept_out.get_shape().as_list()))

            print('\n\t\tFinal stacked input of Inception module, %s'%tf_incept_out.get_shape().as_list())
            x=tf_incept_out

        if 'fulcon' in op:
            break

    # we need to reshape the output of last subsampling layer to
    # convert 4D output to a 2D input to the hidden layer
    # e.g subsample layer output [batch_size,width,height,depth] -> [batch_size,width*height*depth]
    shape = x.get_shape().as_list()
    rows = shape[0]
    create_fulcon_layers(shape[1] * shape[2] * shape[3])
    for op in conv_ops:
        if 'fulcon' in op:
            print('Unwrapping last convolution layer %s to %s hidden layer'%(shape,(rows,hyparams[op]['in'])))
            x = tf.reshape(x, [rows,hyparams[op]['in']])
            break

    for op in conv_ops:
        if 'fulcon_hidden' not in op:
            continue
        else:
            if is_train and use_dropout:
                x = tf.nn.dropout(lrelu(tf.matmul(x,weights[op])+biases[op]),keep_prob=1.-dropout_rate,seed=tf.set_random_seed(12321))
            else:
                x = lrelu(tf.matmul(x,weights[op])+biases[op])

    if use_dropout and is_train:
        x = tf.nn.dropout(x,1.0-dropout_rate,seed=tf.set_random_seed(98765))

    return tf.matmul(x, weights['fulcon_out']) + biases['fulcon_out']

def calc_loss(logits,labels):
    # Training computation.
    if include_l2_loss:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels)) + \
               (beta/2)*tf.reduce_sum([tf.nn.l2_loss(w) if 'fulcon' in kw or 'conv' in kw else 0 for kw,w in weights.items()])
    else:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))

    return loss

def optimize_func(loss,global_step,decay_step):
    # Optimizer.
    if decay_learning_rate:
        learning_rate = tf.train.exponential_decay(start_lr, global_step,decay_steps=decay_step,decay_rate=0.99)
        learning_rate = tf.maximum(learning_rate,start_lr*0.05)
    else:
        learning_rate = start_lr

    #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    optimizer = tf.train.MomentumOptimizer(learning_rate,momentum=0.9).minimize(loss)
    return optimizer,learning_rate


def predict_with_logits(logits):
    # Predictions for the training, validation, and test data.
    prediction = tf.nn.softmax(logits)
    return prediction

def predict_with_dataset(dataset):
    prediction = tf.nn.softmax(get_logits(dataset,False))
    return prediction

def get_conv_net_structure():
    return conv_ops,hyparams

logits,loss,pred,optimize,inc_gstep,test_pred = None,None,None,None,None,None
tf_dataset,tf_labels,tf_test_dataset,tf_test_labels = None,None,None,None
global_step,start_lr = None,None

def inc_global_step(global_step):
    return global_step.assign(global_step+1)


def initialize_conv_net(dataset_type,hyparams):
    global batch_size,num_epochs,start_lr,decay_learning_rate,dropout_rate,in_dropout_rate,use_dropout,early_stopping,accuracy_drops_cap,include_l2_loss,beta,check_early_stop_from,decay_step
    global image_size,num_labels,num_channels
    global total_iterations
    global logits,loss,pred,optimize,inc_gstep,test_pred
    global tf_dataset,tf_labels,tf_test_dataset,tf_test_labels
    global global_step,start_lr

    # hyperparameters
    if dataset_type=='cifar-10':
        image_size = 32
        num_labels = 10
        num_channels = 3 # rgb
    elif dataset_type=='imagenet':
        image_size = 224
        num_labels = 100
        num_channels = 3 # rgb
    else:
        raise NotImplementedError

    batch_size = hyparams['batch_size']

    num_epochs = hyparams['num_epochs']
    decay_step = hyparams['decay_step']

    start_lr = hyparams['start_lr']
    decay_learning_rate = hyparams['use_decay_lr']

    #dropout seems to be making it impossible to learn
    #maybe only works for large nets
    dropout_rate = hyparams['dropout_rate']
    in_dropout_rate = hyparams['in_dropout_rate']
    use_dropout = hyparams['use_dropout']

    early_stopping = hyparams['use_early_stop']
    accuracy_drops_cap = hyparams['accuracy_drop_cap']
    check_early_stop_from = hyparams['check_early_stop_from']

    include_l2_loss = hyparams['include_l2loss']
    #1e-4 : 1e-5 for 2 conv layers
    #1e-7 : 1e-8 for 3 conv layers
    beta = hyparams['beta']

    global_step = tf.Variable(0, trainable=False)
    start_lr = tf.Variable(start_lr)
    create_subsample_layers()

    # Input data.
    print('Input data defined...\n')
    tf_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

    tf_test_dataset = tf.placeholder(tf.float32, shape=(batch_size,image_size,image_size,num_channels))
    tf_test_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

    print('================ Training ==================\n')
    logits = get_logits(tf_dataset,True)
    loss = calc_loss(logits,tf_labels)
    pred = predict_with_logits(logits)
    optimize = optimize_func(loss,global_step,decay_step)
    inc_gstep = inc_global_step(global_step)
    print('==============================================\n')

    print('================ Testing ==================\n')
    test_pred = predict_with_dataset(tf_test_dataset)
    print('==============================================\n')

    tf.initialize_all_variables().run()

    print('Initialized...')
    print('\tBatch size:',batch_size)
    print('\tDepths: ',depth_conv)
    print('\tNum Epochs: ',num_epochs)
    print('\tDecay Learning Rate: ',decay_learning_rate,', ',hyparams['start_lr'])
    print('\tDropout: ',use_dropout,', ',dropout_rate)
    print('\tEarly Stopping: ',early_stopping)
    print('\tInclude L2, Beta: ',include_l2_loss,', ',beta)
    print('\tDecay step %d'%decay_step)
    print('==================================================\n')

def train_conv_net(session,datasets):
    global train_size,valid_size,test_size
    global start_lr
    global logits,loss,pred,optimize,inc_gstep
    global image_size,num_labels,num_channels
    global tf_dataset,tf_labels,tf_test_dataset,tf_test_labels
    global global_step,start_lr

    total_iterations = 0

    train_dataset,train_labels = datasets['train_dataset'],datasets['train_labels']
    valid_dataset,valid_labels = datasets['valid_dataset'],datasets['valid_labels']
    test_dataset,test_labels = datasets['test_dataset'],datasets['test_labels']

    train_size,valid_size,test_size = train_dataset.shape[0],valid_dataset.shape[0],test_dataset.shape[0]
    print('Dataset sizes')
    print('\tTrain: %d'%train_size)
    print('\tValid: %d'%valid_size)
    print('\tTest: %d'%test_size)
    # if hyperparameters change, graph must be reset

    accuracy_drop = 0 # used for early stopping
    max_test_accuracy = 0

    for epoch in range(num_epochs):
        for iteration in range(floor(float(train_size)/batch_size)):
            offset = iteration * batch_size
            assert offset < train_size
            batch_data = train_dataset[offset:offset + batch_size, :, :, :]
            batch_labels = train_labels[offset:offset + batch_size, :]

            feed_dict = {tf_dataset : batch_data, tf_labels : batch_labels}
            _, l, (_,updated_lr), predictions = session.run([logits,loss,optimize,pred], feed_dict=feed_dict)

            if total_iterations % 50 == 0:
                print('Minibatch loss at epoch,iteration %d,%d: %f' % (epoch,iteration, l))
                print('Learning rate: %.5f'%updated_lr)
                print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))

            total_iterations += 1

        ts_acc_arr = None
        for test_batch_id in range(floor(float(test_size)/batch_size)):

            batch_test_data = test_dataset[test_batch_id*batch_size:(test_batch_id+1)*batch_size,:,:,:]
            batch_test_labels = test_labels[test_batch_id*batch_size:(test_batch_id+1)*batch_size,:]

            feed_test_dict = {tf_test_dataset:batch_test_data, tf_test_labels:batch_test_labels}
            test_predictions = session.run([test_pred],feed_dict=feed_test_dict)

            if ts_acc_arr is None:
                ts_acc_arr = np.asarray(test_predictions[0],dtype=np.float32)
            else:
                ts_acc_arr = np.append(ts_acc_arr,test_predictions[0],axis=0)

            assert test_predictions[0].shape[0]==batch_test_labels.shape[0]

        test_accuracy = accuracy(ts_acc_arr, test_labels[:ts_acc_arr.shape[0],:])
        print('Test accuracy for epoch %d: (Now) %.1f%% (Max) %.1f%%' %(epoch,test_accuracy,max_test_accuracy))
        _ = session.run([inc_gstep])
        print('Global step: %d'%global_step.eval())

        if test_accuracy > max_test_accuracy:
            max_test_accuracy = test_accuracy
            accuracy_drop = 0
        else:
            accuracy_drop += 1

        if epoch>check_early_stop_from and accuracy_drop>accuracy_drops_cap:
            print("Test accuracy saturated...")
            return max_test_accuracy
            break

    return max_test_accuracy

def train_conv_net_once(session,datasets,g_step):
    global train_size,valid_size,test_size
    global start_lr
    global logits,loss,pred,optimize,inc_gstep
    global image_size,num_labels,num_channels
    global tf_dataset,tf_labels,tf_test_dataset,tf_test_labels
    global global_step,start_lr

    total_iterations = 1

    update_gstep = global_step.assign(g_step)
    _ = session.run([update_gstep])

    train_dataset,train_labels = datasets['train_dataset'],datasets['train_labels']
    valid_dataset,valid_labels = datasets['valid_dataset'],datasets['valid_labels']
    test_dataset,test_labels = datasets['test_dataset'],datasets['test_labels']

    train_size,valid_size,test_size = train_dataset.shape[0],valid_dataset.shape[0],test_dataset.shape[0]
    print('Dataset sizes')
    print('\tTrain: %d'%train_size)
    print('\tValid: %d'%valid_size)
    print('\tTest: %d'%test_size)
    # if hyperparameters change, graph must be reset

    max_test_accuracy = 0

    for iteration in range(floor(float(train_size)/batch_size)):
        offset = iteration * batch_size
        assert offset < train_size
        batch_data = train_dataset[offset:offset + batch_size, :, :, :]
        batch_labels = train_labels[offset:offset + batch_size, :]

        feed_dict = {tf_dataset : batch_data, tf_labels : batch_labels}
        _, l, (_,updated_lr), predictions = session.run([logits,loss,optimize,pred], feed_dict=feed_dict)
        #print('\tMinibatch loss at iteration %d: %f' % (iteration, l))
        if total_iterations % 50 == 0:
            print('\tMinibatch loss at iteration %d: %f' % (iteration, l))
            print('\tLearning rate: %.5f'%updated_lr)
            print('\tMinibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            print('\tGlobal step: %d'%global_step.eval())
            assert not np.isnan(l)
        total_iterations += 1

    ts_acc_arr = None
    for test_batch_id in range(floor(float(test_size)/batch_size)):

        batch_test_data = test_dataset[test_batch_id*batch_size:(test_batch_id+1)*batch_size,:,:,:]
        batch_test_labels = test_labels[test_batch_id*batch_size:(test_batch_id+1)*batch_size,:]

        feed_test_dict = {tf_test_dataset:batch_test_data, tf_test_labels:batch_test_labels}
        test_predictions = session.run([test_pred],feed_dict=feed_test_dict)

        if ts_acc_arr is None:
            ts_acc_arr = np.asarray(test_predictions[0],dtype=np.float32)
        else:
            ts_acc_arr = np.append(ts_acc_arr,test_predictions[0],axis=0)

        assert test_predictions[0].shape[0]==batch_test_labels.shape[0]

    test_accuracy = accuracy(ts_acc_arr, test_labels[:ts_acc_arr.shape[0],:])
    print('\tTest accuracy: %.1f%%' %test_accuracy)

    return test_accuracy

if __name__=='__main__':

    global train_size,valid_size,test_size
    global log_suffix,data_filename
    global total_iterations

    total_iterations = 0

    dataset_type = 'imagenet'
    # hyperparameters
    if dataset_type=='cifar-10':
        image_size = 32
        num_labels = 10
        num_channels = 3 # rgb
    elif dataset_type=='imagenet':
        image_size = 224
        num_labels = 100
        num_channels = 3 # grayscale
    else:
        raise NotImplementedError

    batch_size = 16

    num_epochs = 250
    decay_step = 1

    start_lr = 0.2
    decay_learning_rate = True

    #dropout seems to be making it impossible to learn
    #maybe only works for large nets
    dropout_rate = 0.1
    in_dropout_rate = 0.2
    use_dropout = True

    early_stopping = True
    accuracy_drops_cap = 10
    check_early_stop_from = 100

    include_l2_loss = True
    beta = 1e-3

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
    if dataset_type=='cifar-10':
        (full_train_dataset,full_train_labels),(valid_dataset,valid_labels),(test_dataset,test_labels)=load_data.reformat_data_cifar10(data_filename)

    graph = tf.Graph()

    # Value logger will log info used to calculate policies
    test_logger = logging.getLogger('test_logger_'+log_suffix)
    test_logger.setLevel(logging.INFO)
    fileHandler = logging.FileHandler('test_logger_'+log_suffix, mode='w')
    fileHandler.setFormatter(logging.Formatter('%(message)s'))
    test_logger.addHandler(fileHandler)

    test_accuracies = []

    with tf.Session(graph=graph) as session:
        #tf.global_variables_initializer().run()
        # Input data.

        print('Input data defined...\n')
        tf_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
        tf_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

        tf_test_dataset = tf.placeholder(tf.float32, shape=(batch_size,image_size,image_size,num_channels))
        tf_test_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

        data_percentages = []
        data_percentages.extend(list(np.arange(0.001,0.010,0.001)))
        data_percentages.extend(list(np.arange(0.01,0.10,0.01)))
        data_percentages.extend(list(np.arange(0.1,1.1,0.1)))
        for data_percentage in data_percentages:

            if data_percentage < 0.01:
                check_early_stop_from = 100
            elif 0.01 <= data_percentage < 0.1:
                check_early_stop_from = 50
            elif 0.1 <= data_percentage < 1.1:
                check_early_stop_from = 25

            train_dataset = full_train_dataset[:(full_train_dataset.shape[0]*data_percentage),:,:,:]
            train_labels = full_train_labels[:(full_train_dataset.shape[0]*data_percentage),:]

            train_size,valid_size,test_size = train_dataset.shape[0],valid_dataset.shape[0],test_dataset.shape[0]

            # Input data.
            print('Running with %d data points and early stop after %d...\n'%(train_size,check_early_stop_from))
            tf_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
            tf_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

            tf_test_dataset = tf.placeholder(tf.float32, shape=(batch_size,image_size,image_size,num_channels))
            tf_test_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

            global_step = tf.Variable(0, trainable=False)
            #start_lr = tf.Variable(start_lr)
            create_subsample_layers()

            print('================ Training ==================\n')
            logits = get_logits(tf_dataset,True)
            loss = calc_loss(logits,tf_labels)
            pred = predict_with_dataset(tf_dataset)
            optimize = optimize_func(loss,global_step,decay_step)
            inc_gstep = inc_global_step(global_step)
            print('==============================================\n')

            print('================ Testing ==================\n')
            test_pred = predict_with_dataset(tf_test_dataset)
            print('==============================================\n')

            tf.initialize_all_variables().run()

            print('Initialized...')
            print('\tBatch size:',batch_size)
            print('\tDepths: ',depth_conv)
            print('\tNum Epochs: ',num_epochs)
            print('\tDecay Learning Rate: ',decay_learning_rate,', ',start_lr)
            print('\tDropout: ',use_dropout,', ',dropout_rate)
            print('\tEarly Stopping: ',early_stopping)
            print('\tInclude L2, Beta: ',include_l2_loss,', ',beta)
            print('\tDecay step %d'%decay_step)
            print('==================================================\n')

            accuracy_drop = 0 # used for early stopping
            max_test_accuracy = 0

            for epoch in range(num_epochs):
                for iteration in range(floor(float(train_size)/batch_size)):
                    offset = iteration * batch_size
                    assert offset < train_size
                    batch_data = train_dataset[offset:offset + batch_size, :, :, :]
                    batch_labels = train_labels[offset:offset + batch_size, :]

                    feed_dict = {tf_dataset : batch_data, tf_labels : batch_labels}
                    _, l, (_,updated_lr), predictions = session.run([logits,loss,optimize,pred], feed_dict=feed_dict)

                    if total_iterations % 50 == 0:
                        print('Minibatch loss at epoch,iteration %d,%d: %f' % (epoch,iteration, l))
                        print('Learning rate: %.5f'%updated_lr)
                        print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))

                    total_iterations += 1

                ts_acc_arr = None
                for test_batch_id in range(floor(float(test_size)/batch_size)):

                    batch_test_data = test_dataset[test_batch_id*batch_size:(test_batch_id+1)*batch_size,:,:,:]
                    batch_test_labels = test_labels[test_batch_id*batch_size:(test_batch_id+1)*batch_size,:]

                    feed_test_dict = {tf_test_dataset:batch_test_data, tf_test_labels:batch_test_labels}
                    test_predictions = session.run([test_pred],feed_dict=feed_test_dict)

                    if ts_acc_arr is None:
                        ts_acc_arr = np.asarray(test_predictions[0],dtype=np.float32)
                    else:
                        ts_acc_arr = np.append(ts_acc_arr,test_predictions[0],axis=0)

                    assert test_predictions[0].shape[0]==batch_test_labels.shape[0]

                test_accuracy = accuracy(ts_acc_arr, test_labels[:ts_acc_arr.shape[0],:])
                print('Test accuracy for epoch %d: (Now) %.1f%% (Max) %.1f%%' %(epoch,test_accuracy,max_test_accuracy))
                _ = session.run([inc_gstep])
                print('Global step: %d'%global_step.eval())

                if test_accuracy > max_test_accuracy:
                    max_test_accuracy = test_accuracy
                    accuracy_drop = 0
                else:
                    accuracy_drop += 1

                if epoch>check_early_stop_from and accuracy_drop>accuracy_drops_cap:
                    print("Test accuracy saturated...")
                    test_logger.info("%.3f,%.3f"%(data_percentage,max_test_accuracy))
                    break



