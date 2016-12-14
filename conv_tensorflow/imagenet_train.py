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

    train_dataset_prefix = 'imagenet_small_train_dataset_'
    train_labels_prefix = 'imagenet_small_train_labels_'
    train_dataset_filenames = ['imagenet_small_train_dataset_1.npy','imagenet_small_train_dataset_2.npy',
                               'imagenet_small_train_dataset_3.npy','imagenet_small_train_dataset_4.npy',
                               'imagenet_small_train_dataset_5.npy']

    train_label_filenames = ['imagenet_small_train_labels_1.npy','imagenet_small_train_labels_2.npy',
                               'imagenet_small_train_labels_3.npy','imagenet_small_train_labels_4.npy',
                               'imagenet_small_train_labels_5.npy']

    valid_dataset_fname,valid_label_fname = 'imagenet_small_valid_dataset.npy','imagenet_small_valid_labels.npy'

    train_size = 127000

    data_percentages = []
    #data_percentages.extend(list(np.arange(0.001,0.010,0.001)))
    #data_percentages.extend(list(np.arange(0.01,0.10,0.01)))
    #data_percentages.extend(list(np.arange(0.1,1.1,0.1)))
    data_percentages.append(0.1)
    valid_dataset,valid_labels = load_data.reformat_data_imagenet_with_memmap((valid_dataset_fname,valid_label_fname))
    print('Valid data processed ...')

    batch_size = 16
    num_epochs = 250

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
            filled_size = 0

            train_dataset,train_labels = None,None

            chunk_size = batch_size*100
            while filled_size<train_size_clipped:
                
                start_memmap,end_memmap,dataset_sizes = load_data.get_next_memmap_indices(train_dataset_filenames,chunk_size)

                col_count = 224**2 * 3
                # Loading data from memmap
                #reading from same memmap
                if start_memmap[0]==end_memmap[0]:
                    print('Processing files %s,%s'%(train_dataset_prefix+str(start_memmap[0]),train_labels_prefix+str(start_memmap[0])))
                    print('\tOffset: %s'%(str(start_memmap),str(end_memmap)))
                    fp1 = np.memmap(train_dataset_prefix+str(start_memmap[0]),dtype=np.float32,mode='r',
                                    offset=np.dtype('float32').itemsize*col_count*start_memmap[1],shape=(end_memmap[1]-start_memmap[1],224,224,3))

                    fp2 = np.memmap(train_labels_prefix+str(start_memmap[0]),dtype=np.float32,mode='r',
                                    offset=np.dtype('float32').itemsize*col_count*start_memmap[1],shape=(end_memmap[1]-start_memmap[1],224,224,3))

                    train_dataset = fp1[:,:,:,:]
                    train_labels = fp2[:,0]
                    filled_size += train_dataset.shape[0]
                    print('\tCurrent size of train data: %d'%filled_size)

                    del fp1,fp2
                #reading from 2 different memmaps
                else:
                    print('Processing 2 files...')
                    print('\t%s,%s'%(train_dataset_prefix+str(start_memmap[0]),train_labels_prefix+str(start_memmap[0])))
                    print('\t%s,%s'%(train_dataset_prefix+str(end_memmap[0]),train_labels_prefix+str(end_memmap[0])))
                    print('\tOffset: %s'%(str(start_memmap),str(end_memmap)))

                    fp1_start = np.memmap(train_dataset_prefix+str(start_memmap[0]),dtype=np.float32,mode='r',
                                    offset=np.dtype('float32').itemsize*col_count*start_memmap[1],
                                    shape=(dataset_sizes[train_dataset_prefix+str(start_memmap[0])]-start_memmap[1],224,224,3))

                    fp2_start = np.memmap(train_labels_prefix+str(start_memmap[0]),dtype=np.float32,mode='r',
                                    offset=np.dtype('float32').itemsize*col_count*start_memmap[1],
                                    shape=(dataset_sizes[train_dataset_prefix+str(start_memmap[0])]-start_memmap[1],224,224,3))

                    fp1_end = np.memmap(train_dataset_prefix+str(start_memmap[0]),dtype=np.float32,mode='r',
                                    offset=np.dtype('float32').itemsize*0,shape=(end_memmap[1],224,224,3))

                    fp2_end = np.memmap(train_labels_prefix+str(start_memmap[0]),dtype=np.float32,mode='r',
                                    offset=np.dtype('float32').itemsize*0,shape=(end_memmap[1],224,224,3))

                    train_dataset = fp1_start[:,:,:,:]
                    train_labels = fp2_start[:,0]
                    train_dataset = np.append(train_dataset,fp1_end[:,:,:,:],axis=0)
                    train_labels = np.append(train_labels,fp2_end[:,0],axis=0)
                    filled_size += train_dataset.shape[0]
                    print('\tCurrent size of train data: %d'%filled_size)
                    del fp1_start,fp2_start,fp1_end,fp2_end

                train_size,valid_size = train_dataset.shape[0],valid_dataset.shape[0]
                print('Size (train,valid) %d,%d'%(train_size,valid_size))

                hyperparams = {
                    'batch_size':batch_size,'start_lr':start_lr,'num_epochs':1,
                    'use_decay_lr':decay_learning_rate,'dropout_rate':dropout_rate,'in_dropout_rate':in_dropout_rate,
                    'use_dropout':use_dropout,'accuracy_drop_cap':accuracy_drop_cap,'include_l2loss':include_l2loss,'beta':beta,
                    'use_early_stop':early_stopping,'check_early_stop_from':check_early_stop_from,'decay_step':decay_step
                               }

                max_test_accuracy = conv_net_plot.train_conv_net(session,'imagenet',
                                             {'train_dataset':train_dataset,'train_labels':train_labels,
                                              'valid_dataset':valid_dataset,'valid_labels':valid_labels,
                                              'test_dataset':valid_dataset,'test_labels':valid_labels},hyperparams)

                logger.info('%.3f,%.3f',data_percentage,max_test_accuracy)