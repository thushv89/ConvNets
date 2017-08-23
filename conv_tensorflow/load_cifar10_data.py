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

    train_dataset = (train_dataset-np.mean(train_dataset,axis=1).reshape(-1,1))/np.std(train_dataset,axis=1).reshape(-1,1)

    valid_dataset = (valid_dataset-np.mean(valid_dataset,axis=1).reshape(-1,1))/np.std(valid_dataset,axis=1).reshape(-1,1)

    test_dataset = (test_dataset-np.mean(test_dataset,axis=1).reshape(-1,1))/np.std(test_dataset,axis=1).reshape(-1,1)


    print('\tTrain Mean/Variance:%.2f,%.2f'%(
        np.mean(np.mean(train_dataset,axis=1),axis=0),
        np.mean(np.std(train_dataset,axis=1),axis=0)**2)
          )
    print('\tValid Mean/Variance:%.2f,%.2f'%(
        np.mean(np.mean(valid_dataset,axis=1),axis=0),
        np.mean(np.std(valid_dataset,axis=1),axis=0)**2)
          )
    print('\tTest Mean/Variance:%.2f,%.2f'%(
        np.mean(np.mean(test_dataset,axis=1),axis=0),
        np.mean(np.std(test_dataset,axis=1),axis=0)**2)
          )
    print('Successfully whitened data ...\n')

    if len(params)>0 and params['zca_whiten']:
        datasets = [train_dataset,valid_dataset,test_dataset]

        for d_i,dataset in enumerate(datasets):
            if params['separate_rgb']:
                red = zca_whiten(dataset[:,:1024])
                whiten_dataset = red.reshape(-1,1024)
                green = zca_whiten(dataset[:,1024:2048])
                whiten_dataset = np.append(whiten_dataset,green.reshape(-1,1024),axis=1)
                blue = zca_whiten(dataset[:,2048:3072])
                whiten_dataset = np.append(whiten_dataset,blue.reshape(-1,1024),axis=1)
            else:
                whiten_dataset = zca_whiten(dataset)

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
        print(e)


def zca_whiten(x,gcn=False,variance_cut=False):
    print('ZCA Whitening')
    print('\tMax, Min data:',np.max(x[1,:].flatten()),',',np.min(x[1,:].flatten()))
    print('\tMax, Min mean of data:',np.max(np.mean(x[1,:].flatten())),',',np.min(np.mean(x[1,:].flatten())))

    assert np.all(np.abs(np.mean(np.mean(x[1,:].flatten())))<0.05)
    #assert np.all(np.std(np.mean(x[1,:].flatten()))<1.1)
    print('\tData is already zero')

    original_x = np.asarray(x)
    x = x.T #features x samples (3072 X 10000)

    if gcn:
        x = x/np.std(x,axis=0)
        print('\tMin, Max data:',np.max(x[1,:].flatten()),',',np.min(x[1,:].flatten()))
        print('\tMin max variance of x: ',np.max(np.std(x,axis=0)),', ',np.min(np.std(x,axis=0)))
        assert np.std(x[:,1])<1.1 and np.std(x[:,1]) > 0.9
        print('\tData is unit variance')

    #x_perm = np.random.permutation(x.shape[1])
    #x_sample = x[:,np.r_[x_perm[:min(10000,x.shape[1])]]]
    #print(x_sample.shape)
    sigma = np.dot(x,x.T)/x.shape[1]
    print("\tCov min: %s"%np.min(sigma))
    print("\tCov max: %s"%np.max(sigma))

    # since cov matrix is symmetrice SVD is numerically more stable
    U,S,V = np.linalg.svd(sigma)
    print('\tEig val shape: %s'%S.shape)
    print('\tEig vec shape: %s,%s'%(U.shape[0],U.shape[1]))

    if variance_cut:
        var_total,stop_idx = 0,0
        for e_val in S[::-1]:
            var_total += np.asarray(e_val/np.sum(S))
            stop_idx += 1
            if var_total>0.99:
                break
        print("\tTaking only %s eigen vectors"%stop_idx)

        U = U[:,-stop_idx:] #e.g. 1024x400
        S = S[-stop_idx:]

    assert np.all(S>0.0)

    # unit covariance
    zcaMat = np.dot(np.dot(U, np.diag(1.0/np.sqrt(S+5e-2))),U.T)
    zcaOut = np.dot(zcaMat,x).T
    print('ZCA whitened data shape:',zcaOut.shape)

    return 0.5 *zcaOut + original_x * 0.5

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