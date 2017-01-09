__author__ = 'Thushan Ganegedara'

import sys, argparse
import pickle
import struct
import random
import math
import numpy as np
import os
from collections import defaultdict
import csv
from PIL import Image
from scipy.misc import imsave
import logging
from collections import Counter

logging_level = logging.DEBUG
logging_format = '[%(funcName)s] %(message)s'

# Produce a covariance matrix
def kernel(a, b):
    """ Squared exponential kernel """
    sqdist = np.sum(a ** 2, 1).reshape(-1, 1) + np.sum(b ** 2, 1) - 2 * np.dot(a, b.T)
    return np.exp(-0.5 * sqdist)

# generate 'size' samples for a given distribution
def sample_from_distribution(dist,size):
    global logger
    dist_cumsum = np.cumsum(dist)
    label_sequence = []
    label_found = False
    # take each element of the dist
    # generate a random number and
    # if number < cumulative sum of distribution at i th point
    # add i as a label other wise don't
    # if a label is not found we add the last label
    logger.debug('Sampling %d from the given distribution of size(%s)',size,str(dist.shape))
    for i in range(size):
        for j in range(dist.size):
            r = np.random.random()
            if r<dist[j]:
                label_sequence.append(j)
                label_found = True
                break
            label_found = False
        if not label_found:
            # if any label not found we add the last label
            label_sequence.append(dist.size-1)

    assert len(label_sequence)==size
    np.random.shuffle(label_sequence)
    return label_sequence

def get_augmented_sample_for_label(dataset_info,dataset,label):
    # dataset_info => ['type']['image_size']['resize_to']['num_channels']
    dataset_type,image_size = dataset_info['dataset_type'],dataset_info['image_size']
    num_channels = dataset_info['num_channels']

    if dataset_type=='imagenet-100':
        resize_to = dataset_info['resize_to']
        ops = ['original','rotate','noise','crop','flip']
        choice_probs = [0.05,0.2,0.4,0.25,0.1]
    elif dataset_type=='cifar-10':
        ops = ['original','rotate','noise','flip']
        choice_probs = [0.2,0.2,0.4,0.2]

    image_index = np.random.choice(list(np.where(dataset['labels']==label)[0].flatten()))
    selected_op = np.random.choice(ops,p=choice_probs)

    unnorm_image = dataset['dataset'][image_index,:,:,:]-np.min(dataset['dataset'][image_index,:,:,:])
    unnorm_image /= np.max(unnorm_image)
    unnorm_image *= 255.0
    im = Image.fromarray(np.uint8(unnorm_image))

    assert np.max(unnorm_image)>127 and np.max(unnorm_image)<256
    assert np.min(unnorm_image)>=0

    if selected_op=='original':
        if dataset_type=='imagenet-100':
            im.thumbnail((resize_to,resize_to), Image.ANTIALIAS)
        sample_img = np.array(im)
    elif selected_op=='rotate':
        if dataset_type=='imagenet-100':
            im.thumbnail((resize_to,resize_to), Image.ANTIALIAS)
        angle = np.random.choice([90,180,270])
        im = im.rotate(angle)
        sample_img = np.array(im)
    elif selected_op=='noise':
        noise_amount = min(0.5,np.random.random())
        if dataset_type=='imagenet-100':
            im.thumbnail((resize_to,resize_to), Image.ANTIALIAS)
            sample_img = (1-noise_amount)*np.array(im) + noise_amount*np.random.random_sample((resize_to,resize_to,num_channels))*255.0
        elif dataset_type=='cifar-10':
            sample_img = (1-noise_amount)*np.array(im) + noise_amount*np.random.random_sample((image_size,image_size,num_channels))*255.0
    elif selected_op=='crop':
        x,y = np.random.randint(24,image_size-resize_to),np.random.randint(24,image_size-resize_to)
        im = im.crop((x,y,x+resize_to,y+resize_to))
        sample_img = np.array(im)
    elif selected_op=='flip':
        if dataset_type=='imagenet-100':
            im.thumbnail((resize_to,resize_to), Image.ANTIALIAS)
        if np.random.random()<0.5:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            im = im.transpose(Image.FLIP_TOP_BOTTOM)
        sample_img = (np.array(im).astype('float32')-np.mean(np.array(im)))/np.std(np.array(im))

    if dataset_type =='imagenet-100':
        assert sample_img.shape[0]==resize_to
    elif dataset_type=='cifar-10':
        assert sample_img.shape[0]==image_size

    return image_index,sample_img

# generate gaussian priors
def generate_gaussian_priors_for_labels(batch_size,elements,chunk_size,num_labels):
    chunk_count = elements/chunk_size

    x = np.linspace(0, 100,chunk_count).reshape(-1, 1)
    # 1e-6 * is for numerical stibility
    L = np.linalg.cholesky(kernel(x, x) + 1e-6 * np.eye(chunk_count))

    # single chunk represetn a single point on x axis of the gaussian curve
    # f_prior represent how the data distribution looks like at each chunk
    # f_prior [chunk_count,num_labels] size
    # e.g. f_prior[0] is class distribution at first chunk of elements
    f_prior = np.dot(L, np.random.normal(size=(chunk_count, num_labels)))
    # normalization
    f_prior -= f_prior.min()
    f_prior **= math.ceil(math.sqrt(num_labels))
    f_prior /= np.sum(f_prior, axis=1).reshape(-1, 1)

    return f_prior

def sample_cifar_10_with_gauss(dataset_info,data_filename,f_prior,save_directory):
    global image_use_counter,logger,class_distribution_logger
    # dataset_info => ['elements']['chunk_size']['type']['image_size']['resize_dim']['num_channels']
    elements,chunk_size = dataset_info['elements'],dataset_info['chunk_size']
    num_chunks = elements/chunk_size
    num_slices = dataset_info['dataset_size']/dataset_info['chunk_size']
    dataset_type,image_size = dataset_info['dataset_type'],dataset_info['image_size']
    num_channels = dataset_info['num_channels']

    fp1 = np.memmap(filename=save_directory+os.sep+'cifar-10-nonstation-dataset.pkl', dtype='float32', mode='w+', shape=(elements,image_size,image_size,num_channels))
    fp2 = np.memmap(filename=save_directory+os.sep+'cifar-10-nonstation-labels.pkl', dtype='int32', mode='w+', shape=(elements,1))

    memmap_idx = 0
    for i,dist in enumerate(f_prior):
        dataset = load_slice_from_cifar_10(dataset_info,data_filename,int(i%num_slices)*chunk_size,
                                           min(int((i%num_slices)+1)*chunk_size,dataset_info['dataset_size']))
        label_sequence = sample_from_distribution(dist,chunk_size)

        for label in label_sequence:
            sidx,sample = get_augmented_sample_for_label(dataset_info,dataset,label)
            image_use_counter[sidx] = image_use_counter[sidx]+1 if sidx in image_use_counter else 1
            fp1[memmap_idx,:,:,:] = sample
            fp2[memmap_idx,0] = label
            memmap_idx += 1

        logger.info('Sampling finished for %d/%d points of the curve',i,f_prior.shape[0])
        logger.info('\t%d/%d Samples from this slice were used',len(image_use_counter),chunk_size)
        logger.info('\tEach Sample was used %.2f on average',np.mean(list(image_use_counter.values())))

        cnt = Counter(label_sequence)
        dist_str = ''
        for li in range(num_labels):
            dist_str += str(cnt[li]) if li in cnt else str(0) + ','
            class_distribution_logger.info('%d,%s',i,dist_str)


def sample_imagenet_with_gauss(dataset_info,dataset_filename,label_filename,f_prior,save_directory):
    global image_use_counter,logger,class_distribution_logger
    # dataset_info => ['elements']['chunk_size']['type']['image_size']['resize_dim']['num_channels']
    elements,chunk_size = dataset_info['elements'],dataset_info['chunk_size']
    num_chunks = elements/chunk_size
    num_slices = dataset_info['dataset_size']/dataset_info['data_in_memory']
    dataset_type,image_size,resize_dim = dataset_info['dataset_type'],dataset_info['image_size'],dataset_info['resize_to']
    num_channels = dataset_info['num_channels']

    fp1 = np.memmap(filename=save_directory+os.sep+'imagenet-100-nonstation-dataset.pkl', dtype='float32', mode='w+', shape=(elements,resize_dim,resize_dim,num_channels))
    fp2 = np.memmap(filename=save_directory+os.sep+'imagenet-100-nonstation-labels.pkl', dtype='int32', mode='w+', shape=(elements,1))

    memmap_idx = 0
    for i,dist in enumerate(f_prior):
        dataset = load_slice_from_imagenet(dataset_info,dataset_filename,label_filename,int(i%num_slices)*chunk_size,
                                           min(int((i%num_slices)+1)*chunk_size,dataset_info['dataset_size']))
        label_sequence = sample_from_distribution(dist,chunk_size)

        for label in label_sequence:
            sidx,sample = get_augmented_sample_for_label(dataset_info,dataset,label)
            image_use_counter[sidx] = image_use_counter[sidx]+1 if sidx in image_use_counter else 1
            fp1[memmap_idx,:,:,:] = sample
            fp2[memmap_idx,0] = label
            memmap_idx += 1

        logger.info('Sampling finished for %d/%d points of the curve',i,f_prior.shape[0])
        logger.info('\t%d/%d Samples from this slice were used',len(image_use_counter),chunk_size)
        logger.info('\tEach Sample was used %.2f on average',np.mean(list(image_use_counter.values())))

        cnt = Counter(label_sequence)
        dist_str = ''
        for li in range(num_labels):
            dist_str += str(cnt[li]) if li in cnt else str(0) + ','
            class_distribution_logger.info('%d,%s',i,dist_str)

'''
def sample_cifar_with_gauss(dataset_info,dataset,f_prior):
    for i,dist in enumerate(f_prior):
        label_sequence = sample_from_distribution(dist,chunk_size)
        for label in label_sequence:'''


def load_slice_from_imagenet(dataset_info,dataset_filename,label_filename,start_idx,end_idx):
    global image_use_counter,logger

    image_use_counter = {}
    col_count = (dataset_info['image_size'],dataset_info['image_size'],dataset_info['num_channels'])

    # Loading data from memmap
    logger.info('Processing files %s,%s'%(dataset_filename,label_filename))
    logger.info('Reading data from: %d to %d',start_idx,end_idx)
    fp1 = np.memmap(dataset_filename,dtype=np.float32,mode='r',
                    offset=np.dtype('float32').itemsize*col_count[0]*col_count[1]*col_count[2]*start_idx,shape=(end_idx-start_idx,col_count[0],col_count[1],col_count[2]))
    fp2 = np.memmap(label_filename,dtype=np.int32,mode='r',
                    offset=np.dtype('int32').itemsize*1*start_idx,shape=(end_idx-start_idx,1))
    train_dataset = fp1[:,:,:,:]
    train_labels = fp2[:]

    del fp1,fp2
    return {'dataset':train_dataset,'labels':train_labels.flatten()}

train_dataset,train_labels = None,None
def load_slice_from_cifar_10(dataset_info,data_filename,start_idx,end_idx):
    global image_use_counter,logger
    global train_dataset,train_labels

    image_use_counter = {}
    col_count = (dataset_info['image_size'],dataset_info['image_size'],dataset_info['num_channels'])

    # Loading data from memmap
    logger.info('Processing files %s',data_filename)
    logger.info('Reading data from: %d to %d',start_idx,end_idx)

    if train_labels is None and train_dataset is None:
        with open(data_filename,'rb') as f:
            save = pickle.load(f)
            train_dataset, train_labels = save['train_dataset'],save['train_labels']
            valid_dataset, valid_labels = save['valid_dataset'],save['valid_labels']

            train_dataset = train_dataset.reshape((-1,col_count[0],col_count[1],col_count[2]),order='F').astype(np.float32)
            valid_dataset = valid_dataset.reshape((-1,col_count[0],col_count[1],col_count[2]),order='F').astype(np.float32)
            train_dataset = np.append(train_dataset,valid_dataset,axis=0)
            del valid_dataset

            train_labels = np.append(train_labels[:,None],valid_labels[:,None],axis=0)

    return {'dataset':train_dataset[start_idx:end_idx+1,:,:,:],'labels':train_labels[start_idx:end_idx+1,None]}

def generate_imagenet_test_data(dataset_filename,label_filename,save_directory):

    resize_dim,test_size = dataset_info['resize_to'], dataset_info['test_size']
    num_channels = dataset_info['num_channels']
    col_count = (dataset_info['image_size'],dataset_info['image_size'],dataset_info['num_channels'])
    fp1 = np.memmap(dataset_filename,dtype=np.float32,mode='r',
                    offset=np.dtype('float32').itemsize*col_count[0]*col_count[1]*col_count[2]*0,shape=(test_size,col_count[0],col_count[1],col_count[2]))
    fp2 = np.memmap(label_filename,dtype=np.int32,mode='r',
                    offset=np.dtype('int32').itemsize*1*test_size,shape=(test_size,1))

    test_dataset = fp1[:,:,:,:]
    test_labels = fp2[:]

    del fp1,fp2

    fp1 = np.memmap(filename=save_directory+os.sep+'imagenet-100-nonstation-test-dataset.pkl', dtype='float32', mode='w+', shape=(test_size,resize_dim,resize_dim,num_channels))
    fp2 = np.memmap(filename=save_directory+os.sep+'imagenet-100-nonstation-test-labels.pkl', dtype='int32', mode='w+', shape=(test_size,1))

    idx = 0
    for img,lbl in zip(test_dataset[:],test_labels[:]):
        unnorm_image = img-np.min(img)
        unnorm_image /= np.max(unnorm_image)
        unnorm_image *= 255.0
        im = Image.fromarray(np.uint8(unnorm_image))
        im.thumbnail((resize_to,resize_to), Image.ANTIALIAS)
        sample_img = np.array(im)
        sample_img = (sample_img - np.mean(sample_img))/np.std(sample_img)
        fp1[idx,:,:,:]=sample_img
        fp2[idx,0]=lbl
        idx+=1

    del fp1,fp2

def generate_cifar_test_data(data_filename,save_directory):
    image_size = dataset_info['image_size']
    test_size = dataset_info['test_size']
    num_channels = dataset_info['num_channels']
    col_count = (dataset_info['image_size'],dataset_info['image_size'],dataset_info['num_channels'])

    with open(data_filename,'rb') as f:
        save = pickle.load(f)
        test_dataset, test_labels = save['test_dataset'],save['test_labels']

    fp1 = np.memmap(filename=save_directory+os.sep+'cifar-10-nonstation-test-dataset.pkl', dtype='float32', mode='w+', shape=(test_size,image_size,image_size,num_channels))
    fp2 = np.memmap(filename=save_directory+os.sep+'cifar-10-nonstation-test-labels.pkl', dtype='int32', mode='w+', shape=(test_size,1))

    idx = 0
    for img,lbl in zip(test_dataset[:],test_labels[:]):
        img = img.reshape((-1,col_count[0],col_count[1],col_count[2]),order='F').astype(np.float32)
        img = (img - np.mean(img))/np.std(img)
        fp1[idx,:,:,:]=img
        fp2[idx,0]=lbl
        idx+=1

    del fp1,fp2

def test_generated_data(dataset_info,persist_dir,dataset_filename,label_filename):
    global logger

    dataset_type = dataset_info['dataset_type']
    logger.info('Persisting a random slice of generated data')
    if dataset_type=='imagenet-100':
        col_count = (dataset_info['resize_to'],dataset_info['resize_to'],dataset_info['num_channels'])
    elif dataset_type=='cifar-10':
        col_count = (dataset_info['image_size'],dataset_info['image_size'],dataset_info['num_channels'])

    size = batch_size*10
    start_idx = np.random.randint(0,elements-size-1)
    logger.info('\tRandom slice: (Start,Size) %d,%d',start_idx,size)
    fp1 = np.memmap(dataset_filename,dtype=np.float32,mode='r',
                    offset=np.dtype('float32').itemsize*col_count[0]*col_count[1]*col_count[2]*start_idx,shape=(size,col_count[0],col_count[1],col_count[2]))
    fp2 = np.memmap(label_filename,dtype=np.int32,mode='r',
                    offset=np.dtype('int32').itemsize*1*start_idx,shape=(size,1))

    random_slice_dataset = fp1[:,:,:,:]
    random_slice_labels = fp2[:]

    del fp1,fp2

    save_count_data = {}
    for li,lbl in enumerate(random_slice_labels.flatten()):
        lbl_dir = persist_dir+os.sep+str(lbl)
        if not os.path.exists(lbl_dir):
            save_count_data[lbl] = 0
            os.makedirs(lbl_dir)
        imsave(lbl_dir + os.sep + 'image_'+str(save_count_data[lbl])+'.png',random_slice_dataset[li,:,:,:])
        save_count_data[lbl]+=1

image_use_counter = None
logger = None
class_distribution_logger = None
if __name__ == '__main__':

    global logger
    logger = logging.getLogger('main_logger')
    logger.setLevel(logging_level)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(logging_format))
    console.setLevel(logging_level)
    logger.addHandler(console)

    persist_dir = 'data_generator_dir' # various things we persist related to ConstructorRL

    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)

    class_distribution_logger = logging.getLogger('class_distribution_logger')
    class_distribution_logger.setLevel(logging.INFO)
    cdfileHandler = logging.FileHandler(persist_dir+os.sep+'class_distribution_log', mode='w')
    cdfileHandler.setFormatter(logging.Formatter('%(message)s'))
    class_distribution_logger.addHandler(cdfileHandler)

    batch_size = 128
    elements = batch_size*10000 # number of elements in the whole dataset
    # there are elements/chunk_size points in the gaussian curve for each class
    chunk_size = batch_size*10 # number of samples sampled for each instance of the gaussian curve

    dataset_type = 'cifar-10' #'cifar-10 imagenet-100

    if dataset_type == 'cifar-10':
        data_save_directory = 'data_non_station'
        if not os.path.exists(data_save_directory):
            os.makedirs(data_save_directory)
        image_size = 32
        num_labels = 10
        num_channels = 3
        dataset_size = 50000
        test_size = 10000
        image_use_counter = {}
        dataset_info = {'dataset_type':dataset_type,'elements':elements,'chunk_size':chunk_size,'image_size':image_size,
                        'num_channels':num_channels,'num_labels':num_labels,'dataset_size':dataset_size,'test_size':test_size}

    elif dataset_type == 'imagenet-100':
        data_save_directory = '..'+os.sep+'imagenet_small'
        if not os.path.exists(data_save_directory):
            os.makedirs(data_save_directory)
        image_size = 224
        num_labels = 100
        num_channels = 3
        dataset_size = 128000
        data_in_memory = chunk_size
        resize_to = 128
        test_size = 5000
        image_use_counter = {}
        dataset_info = {'dataset_type':dataset_type,'elements':elements,'chunk_size':chunk_size,'image_size':image_size,'num_labels':num_labels,'dataset_size':dataset_size,
                        'num_channels':num_channels,'data_in_memory':chunk_size,'resize_to':resize_to,'test_size':test_size}

    logger.info('='*60)
    logger.info('Dataset Information')
    logger.info('\t%s',str(dataset_info))
    logger.info('='*60)

    logger.info('Generating gaussian priors')
    priors = generate_gaussian_priors_for_labels(batch_size,elements,chunk_size,num_labels)
    logger.info('\tGenerated priors of size: %d,%d',priors.shape[0],priors.shape[1])

    if dataset_type == 'imagenet-100':
        dataset_filename = '..'+os.sep+'imagenet_small'+os.sep+'imagenet_small_train_dataset'
        label_filename = '..'+os.sep+'imagenet_small'+os.sep+'imagenet_small_train_labels'
        new_dataset_filename = data_save_directory+os.sep+'imagenet-100-nonstation-dataset.pkl'
        new_labels_filename = data_save_directory+os.sep+'imagenet-100-nonstation-labels.pkl'
        #sample_imagenet_with_gauss(dataset_info,dataset_filename,label_filename,priors,data_save_directory)
        generate_imagenet_test_data(dataset_filename,label_filename,data_save_directory)
    elif dataset_type == 'cifar-10':
        data_filename = '..'+os.sep+'..'+os.sep+'data'+os.sep+'cifar-10.pickle'
        new_dataset_filename = data_save_directory+os.sep+'cifar-10-nonstation-dataset.pkl'
        new_labels_filename = data_save_directory+os.sep+'cifar-10-nonstation-labels.pkl'
        generate_cifar_test_data(data_filename,data_save_directory)
        #sample_cifar_10_with_gauss(dataset_info,data_filename,priors,data_save_directory)

    #test_generated_data(dataset_info,persist_dir+os.sep+'test_data',new_dataset_filename,new_labels_filename)
    ''' =============== Quick Test =====================
    x = np.linspace(0, 10, sample_size).reshape(-1, 1)
    #x = np.random.random(size=[1,sample_size])

    # 1e-6 * is for numerical stibility
    L = np.linalg.cholesky(kernel(x, x) + 1e-6 * np.eye(sample_size))

    # massage the data to get a good distribution
    # len(data) is number of labels
    # f_prior are the samples
    f_prior = np.dot(L, np.random.normal(size=(sample_size, num_labels)))
    # make minimum zero
    f_prior -= f_prior.min()
    f_prior = f_prior ** math.ceil(math.sqrt(num_labels))
    f_prior /= np.sum(f_prior, axis=1).reshape(-1, 1)

    x_axis = np.arange(sample_size*2)

    import matplotlib.pyplot as plt
    for i in range(num_labels):
        plt.plot(x_axis,np.append(f_prior[:,i],f_prior[:,i],axis=0))

    #print(np.sum(f_prior,axis=1).shape)
    #plt.plot(x_axis,np.sum(f_prior,axis=1))
    plt.show()'''

