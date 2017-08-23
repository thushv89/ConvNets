__author__ = 'Thushan Ganegedara'

import imagenet_load_data
from scipy.misc import imsave
import numpy as np
import os
from six.moves import cPickle as pickle

def load_and_test_imagenet_data(train_image_fname, train_label_fname, test_image_fname, test_label_fname,datasize_fname,save_data_to_dir):

    with open(datasize_fname,'rb') as f:
        dataset_sizes = pickle.load(f)

    train_size = dataset_sizes['train_dataset']
    valid_size = dataset_sizes['valid_dataset']
    fp1 = np.memmap(data_save_directory+train_dataset_filename,dtype=np.float32,mode='r',
                    offset=np.dtype('float32').itemsize*col_count[0]*col_count[1]*col_count[2]*0,shape=(train_size//4,col_count[0],col_count[1],col_count[2]))
    fp2 = np.memmap(data_save_directory+train_label_filename,dtype=np.int32,mode='r',
                    offset=np.dtype('int32').itemsize*1*0,shape=(train_size//4,1))

    trdataset = fp1[:,:,:,:]
    trlabels = fp2[:,0]

    fp1 = np.memmap(data_save_directory + valid_dataset_fname, dtype=np.float32, mode='r',
                    offset=np.dtype('float32').itemsize * 0,
                    shape=(valid_size // 2, col_count[0], col_count[1], col_count[2]))
    fp2 = np.memmap(data_save_directory + valid_label_fname, dtype=np.int32, mode='r',
                    offset=np.dtype('int32').itemsize * 0, shape=(valid_size // 2, 1))

    vdataset = fp1[:, :, :, :]
    vlabels = fp2[:, 0]

    if not os.path.exists(save_data_to_dir)
        os.mkdir(save_data_to_dir)

    for img,lbl in zip(trdataset,trlabels):



if __name__=='__main__':

    '''load_data.load_and_save_data_cifar10(filename='cifar-10-white.pickle',zca_whiten=True,return_original=True,separate_rgb=False)
    (tr_white_dataset,tr_labels),(v_white_dataset,v_labels),(ts_white_dataset,ts_labels) = load_data.reformat_data_cifar10(filename='cifar-10-white.pickle')
    #(tr_dataset,tr_labels),(v_dataset,v_labels),(ts_dataset,ts_labels) = load_data.reformat_data_cifar10(filename='cifar-10.pickle')
    for i in range(10):
        rand_idx = np.random.randint(0,5000)
        #imsave('test_img.png', tr_dataset[rand_idx,:,:,:])
        imsave('test_img_whitened_'+str(i)+'.png', tr_white_dataset[rand_idx,:,:,:])'''

    # Testing if imagenet data (train and test belong to same class)
    test_dir = 'test_data'
    test_test_dir = test_dir + os.sep + 'test'
    test_train_dir = test_dir + os.sep + 'train'

    data_save_directory = "imagenet_small/"
    train_dataset_filename = 'imagenet_small_train_dataset'
    train_label_filename = 'imagenet_small_train_labels'

    valid_dataset_fname,valid_label_fname = 'imagenet_small_valid_dataset','imagenet_small_valid_labels'

    col_count = (224,224,3)
    with open(data_save_directory+'dataset_sizes.pickle','rb') as f:
        dataset_sizes = pickle.load(f)

    train_size,valid_size = dataset_sizes[train_dataset_filename],dataset_sizes[valid_dataset_fname]

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    fp1 = np.memmap(data_save_directory+valid_dataset_fname, dtype=np.float32,mode='r',
                    offset=np.dtype('float32').itemsize*0,shape=(valid_size//2,col_count[0],col_count[1],col_count[2]))
    fp2 = np.memmap(data_save_directory+valid_label_fname, dtype=np.int32,mode='r',
                    offset=np.dtype('int32').itemsize*0,shape=(valid_size//2,1))
    vdataset = fp1[:,:,:,:]
    vlabels = fp2[:,0]

    if not os.path.exists(test_test_dir):
        os.makedirs(test_test_dir)

    test_img_indices = {}
    for img,lbl in zip(vdataset,vlabels):
        lbl_path = test_test_dir+os.sep+str(lbl)
        if not os.path.exists(lbl_path):
            print('Creating directory '+lbl_path)
            os.makedirs(lbl_path)
            test_img_indices[lbl] = 0

        imsave(lbl_path + os.sep + 'test_'+str(test_img_indices[lbl])+'.png', img)
        test_img_indices[lbl] += 1

    del fp1,fp2

    fp1 = np.memmap(data_save_directory+train_dataset_filename,dtype=np.float32,mode='r',
                    offset=np.dtype('float32').itemsize*col_count[0]*col_count[1]*col_count[2]*0,shape=(train_size//4,col_count[0],col_count[1],col_count[2]))
    fp2 = np.memmap(data_save_directory+train_label_filename,dtype=np.int32,mode='r',
                    offset=np.dtype('int32').itemsize*1*0,shape=(train_size//4,1))

    trdataset = fp1[:,:,:,:]
    trlabels = fp2[:,0]

    del fp1,fp2

    train_img_indices = {}
    for img,lbl in zip(trdataset,trlabels):
        lbl_path = test_train_dir+os.sep+str(lbl)
        if not os.path.exists(lbl_path):
            print('Creating directory '+lbl_path)
            os.makedirs(lbl_path)
            train_img_indices[lbl] = 0

        imsave(lbl_path + os.sep + 'train_'+str(test_img_indices[lbl])+'.png', img)
        train_img_indices[lbl] += 1

