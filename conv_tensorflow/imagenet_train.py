__author__ = 'Thushan Ganegedara'

import conv_net_plot
import sys
import getopt
import load_data
import tensorflow as tf
import logging
import operator
import numpy as np

if __name__=='__main__':

    try:
        opts,args = getopt.getopt(
            sys.argv[1:],"",['log_suffix='])
    except getopt.GetoptError as err:
        print('')


    if len(opts)!=0:
        for opt,arg in opts:
            if opt == '--log_suffix':
                log_suffix = arg

    logger = logging.getLogger('logger_'+log_suffix)
    logger.setLevel(logging.INFO)
    fileHandler = logging.FileHandler('logger_'+log_suffix, mode='w')
    fileHandler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(fileHandler)

    data_save_directory = "imagenet_sm/"
    train_dataset_filename = 'imagenet_small_train_dataset'
    train_label_filename = 'imagenet_small_train_labels'

    valid_dataset_fname,valid_label_fname = 'imagenet_small_valid_dataset','imagenet_small_valid_labels'

    train_size = 127000

    data_percentages = []
    #data_percentages.extend(list(np.arange(0.001,0.010,0.001)))
    #data_percentages.extend(list(np.arange(0.01,0.10,0.01)))
    #data_percentages.extend(list(np.arange(0.1,1.1,0.1)))
    data_percentages.append(0.1)
    valid_dataset,valid_labels = load_data.reformat_data_imagenet_with_memmap((valid_dataset_fname,valid_label_fname))
    print('Valid data processed ...')

    batch_size = 16
    num_epochs = 10

    decay_step = 1
    start_lr = 0.1
    decay_learning_rate = True

    #dropout seems to be making it impossible to learn
    #maybe only works for large nets
    dropout_rate = 0.1
    in_dropout_rate = 0.2
    use_dropout = True

    early_stopping = True
    accuracy_drop_cap = 10
    check_early_stop_from = 100

    include_l2loss = True
    beta = 1e-3

    graph = tf.Graph()
    with tf.Session(graph=graph) as session:
        for data_percentage in data_percentages:

            if data_percentage < 0.01:
                check_early_stop_from = 100
            elif 0.01 <= data_percentage < 0.1:
                check_early_stop_from = 50
            elif 0.1 <= data_percentage < 1.1:
                check_early_stop_from = 25

            train_size_clipped = train_size*data_percentage


            train_dataset,train_labels = None,None
            max_test_accuracy = 0.0

            for epoch in range(num_epochs):
                print('Running epoch %d'%epoch)
                filled_size = 0
                chunk_size = batch_size*100
                test_accuracies = []
                while filled_size<train_size_clipped:

                    start_memmap,end_memmap,dataset_sizes = load_data.get_next_memmap_indices(train_dataset_filename,chunk_size)

                    col_count = 224**2 * 3
                    # Loading data from memmap
                    #reading from same memmap

                    print('Processing files %s,%s'%(train_dataset_filename,train_label_filename))
                    print('\tOffset: %d%d'%(start_memmap,end_memmap))
                    fp1 = np.memmap(train_dataset_filename,dtype=np.float32,mode='r',
                                    offset=np.dtype('float32').itemsize*col_count*start_memmap[1],shape=(end_memmap[1]-start_memmap[1],224,224,3))

                    fp2 = np.memmap(train_label_filename,dtype=np.float32,mode='r',
                                    offset=np.dtype('float32').itemsize*col_count*start_memmap[1],shape=(end_memmap[1]-start_memmap[1],224,224,3))

                    train_dataset = fp1[:,:,:,:]
                    train_labels = fp2[:,0]
                    filled_size += train_dataset.shape[0]
                    print('\tCurrent size of train data: %d'%filled_size)
                    del fp1,fp2

                    train_size,valid_size = train_dataset.shape[0],valid_dataset.shape[0]
                    print('Size (train,valid) %d,%d'%(train_size,valid_size))

                    hyperparams = {
                        'batch_size':batch_size,'start_lr':start_lr,'num_epochs':1,
                        'use_decay_lr':decay_learning_rate,'dropout_rate':dropout_rate,'in_dropout_rate':in_dropout_rate,
                        'use_dropout':use_dropout,'accuracy_drop_cap':accuracy_drop_cap,'include_l2loss':include_l2loss,'beta':beta,
                        'use_early_stop':False,'check_early_stop_from':check_early_stop_from,'decay_step':decay_step
                                   }

                    test_accuracy = conv_net_plot.train_conv_net(session,'imagenet',
                                                 {'train_dataset':train_dataset,'train_labels':train_labels,
                                                  'valid_dataset':valid_dataset,'valid_labels':valid_labels,
                                                  'test_dataset':valid_dataset,'test_labels':valid_labels},hyperparams)
                    test_accuracy.append(test_accuracy)

                mean_test_accuracy = np.mean(test_accuracies)
                test_accuracies = []
                print('Test accuracy for epoch %d: (Now) %.1f%% (Max) %.1f%%' %(epoch,mean_test_accuracy,max_test_accuracy))

                if mean_test_accuracy > max_test_accuracy:
                    max_test_accuracy = mean_test_accuracy
                    accuracy_drop = 0
                else:
                    accuracy_drop += 1

                if epoch>check_early_stop_from and accuracy_drop>accuracy_drop_cap:
                    print("Test accuracy saturated...")
                    logger.info('%.3f,%.3f',data_percentage,max_test_accuracy)
