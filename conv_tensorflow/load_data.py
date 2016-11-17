__author__ = 'Thushan Ganegedara'

from six.moves import cPickle as pickle
from six.moves import range
import numpy as np
import os

def load_and_save_data_cifar10(filename,**params):

    valid_size_required = 10000
    cifar_file = '..'+os.sep+'data'+os.sep+filename

    if os.path.exists(cifar_file):
        return

    train_pickle_file = '..'+os.sep+'data'+os.sep+'cifar_10_data_batch_'
    test_pickle_file = '..'+os.sep+'data' + os.sep + 'cifar_10_test_batch'
    train_raw = None
    test_dataset = None
    train_raw_labels = None
    test_labels = None

    #train data
    for i in range(1,5+1):
        with open(train_pickle_file+str(i),'rb') as f:
            save = pickle.load(f,encoding="latin1")

            if train_raw is None:
                train_raw = np.asarray(save['data'],dtype=np.float32)
                train_raw_labels = np.asarray(save['labels'],dtype=np.int16)
            else:

                train_raw = np.append(train_raw,save['data'],axis=0)
                train_raw_labels = np.append(train_raw_labels,save['labels'],axis=0)

    #test file
    with open(test_pickle_file,'rb') as f:
        save = pickle.load(f,encoding="latin1")
        test_dataset = np.asarray(save['data'],dtype=np.float32)
        test_labels = np.asarray(save['labels'],dtype=np.int16)


    valid_rand_idx = np.random.randint(0,train_raw.shape[0]-valid_size_required)
    valid_perm = np.random.permutation(train_raw.shape[0])[valid_rand_idx:valid_rand_idx+valid_size_required]

    valid_dataset = np.asarray(train_raw[valid_perm,:],dtype=np.float32)
    valid_labels = np.asarray(train_raw_labels[valid_perm],dtype=np.int16)
    print('Shape of valid dataset (%s) and labels (%s)'%(valid_dataset.shape,valid_labels.shape))

    train_dataset = np.delete(train_raw,valid_perm,axis=0)
    train_labels = np.delete(train_raw_labels,valid_perm,axis=0)
    print('Shape of train dataset (%s) and labels (%s)'%(train_dataset.shape,train_labels.shape))

    print('Per image whitening ...')
    pixel_depth = 255 if np.max(train_dataset[0,:])>1.1 else 1
    print('\tDectected pixel depth: %d'%pixel_depth)
    print('\tZero mean and Unit variance')

    train_dataset[:,:1024] = np.subtract(train_dataset[:,:1024],np.mean(train_dataset[:,:1024],axis=1).reshape((-1,1)))/pixel_depth
    train_dataset[:,1024:2048] = np.subtract(train_dataset[:,1024:2048],np.mean(train_dataset[:,1024:2048],axis=1).reshape((-1,1)))/pixel_depth
    train_dataset[:,2048:] = np.subtract(train_dataset[:,2048:3072],np.mean(train_dataset[:,2048:3072],axis=1).reshape((-1,1)))/pixel_depth

    valid_dataset[:,:1024] = np.subtract(valid_dataset[:,:1024],np.mean(valid_dataset[:,:1024],axis=1).reshape((-1,1)))/pixel_depth
    valid_dataset[:,1024:2048] = np.subtract(valid_dataset[:,1024:2048],np.mean(valid_dataset[:,1024:2048],axis=1).reshape((-1,1)))/pixel_depth
    valid_dataset[:,2048:] = np.subtract(valid_dataset[:,2048:],np.mean(valid_dataset[:,2048:],axis=1).reshape((-1,1)))/pixel_depth

    test_dataset[:,:1024] = np.subtract(test_dataset[:,:1024],np.mean(test_dataset[:,:1024],axis=1).reshape((-1,1)))/pixel_depth
    test_dataset[:,1024:2048] = np.subtract(test_dataset[:,1024:2048],np.mean(test_dataset[:,1024:2048],axis=1).reshape((-1,1)))/pixel_depth
    test_dataset[:,2048:] = np.subtract(test_dataset[:,2048:],np.mean(test_dataset[:,2048:],axis=1).reshape((-1,1)))/pixel_depth

    print('\tTrain Mean/Variance:%.2f,%.2f'%(
        np.mean(np.mean(train_dataset[:,:1024],axis=1),axis=0),
        np.mean(np.std(train_dataset[:,:1024],axis=1),axis=0)**2)
          )
    print('\tValid Mean/Variance:%.2f,%.2f'%(
        np.mean(np.mean(valid_dataset[:,:1024],axis=1),axis=0),
        np.mean(np.std(valid_dataset[:,:1024],axis=1),axis=0)**2)
          )
    print('\tTest Mean/Variance:%.2f,%.2f'%(
        np.mean(np.mean(test_dataset[:,:1024],axis=1),axis=0),
        np.mean(np.std(test_dataset[:,:1024],axis=1),axis=0)**2)
          )
    print('Successfully whitened data ...\n')

    if len(params)>0 and params['zca_whiten']:
        datasets = [train_dataset,valid_dataset,test_dataset]

        for d_i,dataset in enumerate(datasets):
            red = zca_whiten(dataset[:,:1024])
            whiten_dataset = red.reshape(-1,1024)
            green = zca_whiten(dataset[:,1024:2048])
            whiten_dataset = np.append(whiten_dataset,green.reshape(-1,1024),axis=1)
            blue = zca_whiten(dataset[:,2048:3072])
            whiten_dataset = np.append(whiten_dataset,blue.reshape(-1,1024),axis=1)
            print("Whiten data shape: ",whiten_dataset.shape)

            if d_i==0:
                train_dataset = whiten_dataset
            elif d_i == 1:
                valid_dataset = whiten_dataset
            elif d_i ==2:
                test_dataset = whiten_dataset
            else:
                raise NotImplementedError

    print('\nDumping processed data')
    cifar_data = {'train_dataset':train_dataset,'train_labels':train_labels,
                  'valid_dataset':valid_dataset,'valid_labels':valid_labels,
                  'test_dataset':test_dataset,'test_labels':test_labels
                  }
    try:
        with open(cifar_file, 'wb') as f:
            pickle.dump(cifar_data, f, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save cifar_data:', e)

def zca_whiten(x):
    print('ZCA Whitening')
    print(np.max(np.mean(x[1,:].flatten())))
    print(np.min(np.mean(x[1,:].flatten())))
    assert np.all(np.abs(np.mean(np.mean(x[1,:].flatten())))<0.01)
    assert np.all(np.std(np.mean(x[1,:].flatten()))<1.1)
    print('Data is already zero mean unit variance')

    x = x.T
    x_perm = np.random.permutation(x.shape[1])
    x_sample = x[:,np.r_[x_perm[:5000]]]
    print(x_sample.shape)
    cov_x = np.cov(x_sample)
    print("Cov min: %s"%np.min(cov_x))
    print("Cov max: %s"%np.max(cov_x))

    eig_vals,eig_vecs = np.linalg.eigh(cov_x)
    print('Eig val shape: %s'%eig_vals.shape)
    print('Eig vec shape: %s,%s'%(eig_vecs.shape[0],eig_vecs.shape[1]))

    var_total,stop_idx = 0,0
    for e_val in eig_vals[::-1]:
        var_total += np.asarray(e_val/np.sum(eig_vals))
        stop_idx += 1
        if var_total>0.99:
            break
    print("Taking only %s eigen vectors"%stop_idx)

    eig_vecs = eig_vecs[:,-stop_idx:] #e.g. 1024x400
    eig_vals = eig_vals[-stop_idx:]

    assert np.all(eig_vals>0.0)

    x_rot = np.dot(eig_vecs.T,x)
    #unit covariance
    x_rot_norm = x_rot/np.reshape(np.sqrt(eig_vals+1e-6),(-1,1))
    assert np.abs(x_rot[0,1]/np.asscalar(np.sqrt(eig_vals[0]+1e-6)) - x_rot_norm[0,1])<1e3

    zca_x = np.dot(eig_vecs,x_rot_norm)

    print('ZCA whitened data shape:',zca_x.shape)

    return zca_x.T

def reformat_data_cifar10(filename,**params):

    image_size = 32
    num_labels = 10
    num_channels = 3 # rgb

    print("Reformatting data ...")
    cifar10_file = '..'+os.sep+'data'+os.sep+filename
    with open(cifar10_file,'rb') as f:
        save = pickle.load(f)
        train_dataset, train_labels = save['train_dataset'],save['train_labels']
        valid_dataset, valid_labels = save['valid_dataset'],save['valid_labels']
        test_dataset, test_labels = save['test_dataset'],save['test_labels']

        train_dataset = train_dataset.reshape((-1,image_size,image_size,num_channels),order='F').astype(np.float32)
        valid_dataset = valid_dataset.reshape((-1,image_size,image_size,num_channels),order='F').astype(np.float32)
        test_dataset = test_dataset.reshape((-1,image_size,image_size,num_channels),order='F').astype(np.float32)

        print('\tFinal shape (train):%s',train_dataset.shape)
        print('\tFinal shape (valid):%s',valid_dataset.shape)
        print('\tFinal shape (test):%s',test_dataset.shape)

        train_labels = (np.arange(num_labels) == train_labels[:,None]).astype(np.float32)
        valid_labels = (np.arange(num_labels) == valid_labels[:,None]).astype(np.float32)
        test_labels = (np.arange(num_labels) == test_labels[:,None]).astype(np.float32)

        print('\tFinal shape (train) labels:%s',train_labels.shape)
        print('\tFinal shape (valid) labels:%s',valid_labels.shape)
        print('\tFinal shape (test) labels:%s',test_labels.shape)

        #valid_dataset = zca_whiten(valid_dataset)
        #valid_dataset = valid_dataset.reshape(valid_dataset.shape[0],image_size,image_size,num_channels)
        #test_dataset = zca_whiten(test_dataset)
        #test_dataset = test_dataset.reshape(test_dataset.shape[0],image_size,image_size,num_channels)

    return (train_dataset,train_labels),(valid_dataset,valid_labels),(test_dataset,test_labels)
