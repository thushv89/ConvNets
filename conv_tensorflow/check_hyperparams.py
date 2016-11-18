__author__ = 'Thushan Ganegedara'

import conv_net_plot
import sys
import getopt
import load_data

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
            sys.argv[1:],"",['data='])
    except getopt.GetoptError as err:
        print('<filename>.py --restore_model=<filename> --restore_pools=<filename> --restore_logs=<filename> --train=<0or1> --persist_window=<int> --bump_window=<int>')


    if len(opts)!=0:
        for opt,arg in opts:
            if opt == '--data':
                data_filename = arg

    (train_dataset,train_labels),(valid_dataset,valid_labels),(test_dataset,test_labels)=load_data.reformat_data_cifar10(data_filename)
    train_size,valid_size,test_size = train_dataset.shape[0],valid_dataset.shape[0],test_dataset.shape[0]
    datasets = {'train_dataset':train_dataset[:int(train_size*0.1),:],'train_labels':train_labels[:int(train_size*0.1),:],
                'valid_dataset':valid_dataset,'valid_labels':valid_labels,
                'test_dataset':test_dataset,'test_labels':test_labels}

    test_accuracy_batch_size = {}
    for bs in batch_size:
        hyperparams = {'batch_size':bs,'start_lr':start_learning_rates[0],'num_epochs':num_epochs,
                       'use_decay_lr':decay_learning_rate[0],'dropout_rate':dropouts,'in_dropout_rate':in_dropouts[0],
                       'use_dropout':use_dropouts[0],'accuracy_drop_cap':accuracy_drop_cap,'include_l2loss':include_l2loss[0],'beta':betas[0],
                       'use_early_stop':use_early_stop}

        max_test_accuracy = conv_net_plot.train_conv_net('cifar-10',datasets,hyperparams)
        test_accuracy_batch_size[bs] = max_test_accuracy

    print(test_accuracy_batch_size)