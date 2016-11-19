__author__ = 'Thushan Ganegedara'

import conv_net_plot
import sys
import getopt
import load_data
import tensorflow as tf
import logging
if __name__=='__main__':

    num_epochs = 100
    batch_size = [16,64,128]
    start_learning_rates = [0.1,0.01,0.001]
    decay_learning_rate = [False,True]
    include_l2loss = [False,True]
    betas = [1e-3,1e-5,1e-7]
    in_dropouts = [0.1,0.2,0.5]
    dropouts = 0.2
    use_dropouts = [False,True]
    accuracy_drop_cap = 10
    use_early_stop = True

    try:
        opts,args = getopt.getopt(
            sys.argv[1:],"",['data=','log_suffix='])
    except getopt.GetoptError as err:
        print('<filename>.py --restore_model=<filename> --restore_pools=<filename> --restore_logs=<filename> --train=<0or1> --persist_window=<int> --bump_window=<int>')


    if len(opts)!=0:
        for opt,arg in opts:
            if opt == '--data':
                data_filename = arg
            if opt == '--log_suffix':
                log_suffix = arg

    hyperparam_logger = logging.getLogger('hyperparam_logger'+log_suffix)
    hyperparam_logger.setLevel(logging.INFO)
    fileHandler = logging.FileHandler('hyperparam_logger'+log_suffix, mode='w')
    fileHandler.setFormatter(logging.Formatter('%(message)s'))
    hyperparam_logger.addHandler(fileHandler)

    (train_dataset,train_labels),(valid_dataset,valid_labels),(test_dataset,test_labels)=load_data.reformat_data_cifar10(data_filename)
    train_size,valid_size,test_size = train_dataset.shape[0],valid_dataset.shape[0],test_dataset.shape[0]
    datasets = {'train_dataset':train_dataset[:int(train_size*0.2),:],'train_labels':train_labels[:int(train_size*0.2),:],
                'valid_dataset':valid_dataset,'valid_labels':valid_labels,
                'test_dataset':test_dataset,'test_labels':test_labels}

    res_batch_size = {}
    res_start_lr = {}
    ops,op_params = conv_net_plot.get_conv_net_structure()
    hyperparam_logger.info('================= Structural Info ====================')
    for op in ops:
        if 'conv' in op:
            hyperparam_logger.info('Convolution')
            hyperparam_logger.info('\tFilter: %s'%op_params[op]['weights'])
            hyperparam_logger.info('\tStride: %s'%op_params[op]['stride'])
            hyperparam_logger.info('\tPadding: %s'%op_params[op]['padding'])
        elif 'pool' in op:
            hyperparam_logger.info('Pooling')
            hyperparam_logger.info('\tType: %s'%op_params[op]['type'])
            hyperparam_logger.info('\tFilter: %s'%op_params[op]['kernel'])
            hyperparam_logger.info('\tStride: %s'%op_params[op]['stride'])
            hyperparam_logger.info('\tPadding: %s'%op_params[op]['padding'])
        elif 'fulcon' in op:
            hyperparam_logger.info('Fully connected')
            hyperparam_logger.info('\tIn: %s'%op_params[op]['in'])
            hyperparam_logger.info('\tOut: %s'%op_params[op]['out'])

    graph = tf.Graph()
    with tf.Session(graph=graph) as session:

        print("Testing batch size...")
        for bs in batch_size:
            hyperparams = {'batch_size':bs,'start_lr':start_learning_rates[0],'num_epochs':num_epochs,
                           'use_decay_lr':decay_learning_rate[0],'dropout_rate':dropouts,'in_dropout_rate':in_dropouts[0],
                           'use_dropout':use_dropouts[0],'accuracy_drop_cap':accuracy_drop_cap,'include_l2loss':include_l2loss[0],'beta':betas[0],
                           'use_early_stop':use_early_stop}

            max_test_accuracy = conv_net_plot.train_conv_net(session,'cifar-10',datasets,hyperparams)
            res_batch_size[bs] = max_test_accuracy

        best_batch_size = list(sorted(res_batch_size,reverse=True))[0]
        hyperparam_logger.info('Batch size: %d'%best_batch_size)

        print("Testing Learning rate")
        for start_lr in start_learning_rates:
            hyperparams = {'batch_size':best_batch_size,'start_lr':start_lr,'num_epochs':num_epochs,
                           'use_decay_lr':decay_learning_rate[0],'dropout_rate':dropouts,'in_dropout_rate':in_dropouts[0],
                           'use_dropout':use_dropouts[0],'accuracy_drop_cap':accuracy_drop_cap,'include_l2loss':include_l2loss[0],'beta':betas[0],
                           'use_early_stop':use_early_stop}

            max_test_accuracy = conv_net_plot.train_conv_net(session,'cifar-10',datasets,hyperparams)
            res_start_lr[start_lr] = max_test_accuracy

        hyperparam_logger.info('Start Learning rate: %d'%list(sorted(res_start_lr,reverse=True))[0])

