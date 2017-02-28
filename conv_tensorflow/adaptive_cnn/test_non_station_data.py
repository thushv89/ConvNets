from math import ceil,floor
import numpy as np
import os
from scipy.misc import imsave



def load_data_from_memmap(dataset_info,dataset_filename,label_filename,start_idx,size):
    global logger
    num_labels = dataset_info['num_labels']
    col_count = (dataset_info['image_size'],dataset_info['image_size'],dataset_info['num_channels'])
    print('Processing files %s,%s'%(dataset_filename,label_filename))
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

datatype = 'cifar-10'
batch_size = 128

if datatype == 'cifar-10':
    image_size = 32
    num_labels = 10
    num_channels = 3  # rgb

    dataset_filename = 'data_non_station' + os.sep + 'cifar-10-nonstation-dataset.pkl'
    label_filename = 'data_non_station' + os.sep + 'cifar-10-nonstation-labels.pkl'
    dataset_size = 1280000
    chunk_size = 51200

batches_in_chunk = chunk_size//batch_size

dataset_info = {}
dataset_info['image_size'] = image_size
dataset_info['num_labels'] = num_labels
dataset_info['num_channels'] = num_channels
dataset_info['dataset_size'] = dataset_size
dataset_info['chunk_size'] = chunk_size
dataset_info['num_labels'] = num_labels

memmap_idx = 0

persist_dir = 'test_data_cifar-10'
if not os.path.exists(persist_dir):
    os.makedirs(persist_dir)

for batch_id in range(ceil(dataset_size // batch_size) - batches_in_chunk+1):

    chunk_batch_id = batch_id % batches_in_chunk
    if chunk_batch_id == 0:

        print('Memmap Idx: %d, Size: %d, Batch ID: %d'%(memmap_idx,chunk_size+batch_size,batch_id))
        train_dataset, train_labels = load_data_from_memmap(dataset_info, dataset_filename, label_filename, memmap_idx,
                                                            chunk_size + batch_size)
        memmap_idx += chunk_size

    batch_data = train_dataset[chunk_batch_id * batch_size:(chunk_batch_id + 1) * batch_size, :, :, :]
    batch_labels = train_labels[chunk_batch_id * batch_size:(chunk_batch_id + 1) * batch_size, :]
    batch_labels_int = np.argmax(batch_labels,axis=1).flatten()

    if np.random.random()<0.01:
        batch_dir = persist_dir + os.sep + str(batch_id)
        if not os.path.exists(batch_dir):
            os.makedirs(batch_dir)
        save_count_data = {}

        for img_id in np.random.randint(0,batch_size,(50)):
            lbl = batch_labels_int[img_id]
            lbl_dir = batch_dir+os.sep+str(lbl)
            if not os.path.exists(lbl_dir):
                save_count_data[lbl] = 0
                os.makedirs(lbl_dir)
            imsave(lbl_dir + os.sep + 'image_' + str(save_count_data[lbl]) + '.png', batch_data[img_id, :, :, :])
            save_count_data[lbl] += 1