__author__ = 'Thushan Ganegedara'

import conv_net_plot
import sys
import getopt
import load_data
import tensorflow as tf
import logging
import operator
import numpy as np
from six.moves import cPickle as pickle

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

    data_save_directory = "imagenet_small/"
    train_dataset_filename = 'imagenet_small_train_dataset'
    train_label_filename = 'imagenet_small_train_labels'

    valid_dataset_fname,valid_label_fname = 'imagenet_small_valid_dataset','imagenet_small_valid_labels'


    data_percentages = []
    #data_percentages.extend(list(np.arange(0.001,0.010,0.001)))
    #data_percentages.extend(list(np.arange(0.01,0.10,0.01)))
    #data_percentages.extend(list(np.arange(0.1,1.1,0.1)))
    data_percentages.append(0.1)

    col_count = 224**2 * 3
    with open(data_save_directory+'dataset_sizes.pickle','rb') as f:
        dataset_sizes = pickle.load(f)
        train_size = dataset_sizes[data_save_directory+train_dataset_filename]
        valid_size = dataset_sizes[valid_dataset_fname+'.h5']
        dataset_sizes.pop(data_save_directory+train_dataset_filename)
        dataset_sizes.pop(valid_dataset_fname+'.h5')
        dataset_sizes[train_dataset_filename]=train_size
        dataset_sizes[valid_dataset_fname] = valid_size

        print(dataset_sizes)
    train_size,valid_size = dataset_sizes[train_dataset_filename],dataset_sizes[valid_dataset_fname]

    fp1 = np.memmap(data_save_directory+valid_dataset_fname, dtype=np.float32,mode='r',
                                    offset=np.dtype('float32').itemsize*0,shape=(valid_size,col_count))
    fp2 = np.memmap(data_save_directory+valid_label_fname, dtype=np.float32,mode='r',
                    offset=np.dtype('float32').itemsize*0,shape=(valid_size,1))
    vdataset = fp1[:,:]
    vlabels = fp2[:]
    valid_dataset,valid_labels = load_data.reformat_data_imagenet_with_memmap_array(vdataset,vlabels)

    print('Valid data processed ...')

    batch_size = 64
    num_epochs = 10

    decay_step = 1
    start_lr = 0.01
    decay_learning_rate = True

    #dropout seems to be making it impossible to learn
    #maybe only works for large nets
    dropout_rate = 0.1
    in_dropout_rate = 0.2
    use_dropout = True

    early_stopping = True
    accuracy_drop_cap = 10
    check_early_stop_from = 100

    include_l2loss = False
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

                    start_memmap,end_memmap,dataset_sizes = load_data.get_next_memmap_indices((train_dataset_filename,train_label_filename),chunk_size)


                    # Loading data from memmap
                    #reading from same memmap

                    print('Processing files %s,%s'%(train_dataset_filename,train_label_filename))
                    print('\tOffset: %d, %d'%(start_memmap,end_memmap))
                    fp1 = np.memmap(data_save_directory+train_dataset_filename,dtype=np.float32,mode='r',
                                    offset=np.dtype('float32').itemsize*col_count*start_memmap,shape=(end_memmap-start_memmap,col_count))

                    fp2 = np.memmap(data_save_directory+train_label_filename,dtype=np.float32,mode='r',
                                    offset=np.dtype('int32').itemsize*1*start_memmap,shape=(end_memmap-start_memmap,1))

                    train_dataset = fp1[:,:]
                    train_labels = fp2[:]
                    filled_size += train_dataset.shape[0]
                    assert train_dataset.shape[0]==train_labels.shape[0]
                    print('\tCurrent size of filled data: %d'%filled_size)
                    print('\tTrain data: %d'%train_dataset.shape[0])
                    print('\tTrain labels: %d'%train_labels.shape[0])
                    del fp1,fp2

                    train_dataset,train_labels = load_data.reformat_data_imagenet_with_memmap_array(train_dataset,train_labels)
                    assert train_dataset.shape[0]==train_labels.shape[0]
                    assert valid_dataset.shape[0]==valid_labels.shape[0]
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
                    test_accuracies.append(test_accuracy)

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
