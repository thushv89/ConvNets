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
from PIL import Image,ImageEnhance
from scipy.misc import imsave
import logging
from collections import Counter
import scipy.io


logging_level = logging.INFO
logging_format = '[%(funcName)s] %(message)s'

# Produce a covariance matrix
def kernel(a, b):
    """ Squared exponential kernel """
    sqdist = np.sum(a ** 2, 1).reshape(-1, 1) + np.sum(b ** 2, 1) - 2 * np.dot(a, b.T)
    return np.exp(-0.5 * sqdist)

# generate 'size' samples for a given distribution
def sample_from_distribution(dist,size):
    global logger

    def get_label_sequence(dist,size):
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
            r = np.random.random()
            for j in range(dist.size):
                if r<dist_cumsum[j]:
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

    label_sequence = get_label_sequence(dist,size)
    cnt = Counter(label_sequence)

    euc_distance = 0
    euc_threshold = (0.02**2)*dist.size
    for li in range(dist.size):
        if li in cnt:
            euc_distance += ((cnt[li]*1.0/size)-dist[li])**2
        else:
            euc_distance += dist[li]**2

    if euc_distance>euc_threshold:
        logger.debug('Distribution:')
        logger.debug(dist)
        logger.debug('='*80)
        logger.debug('Label Sequence Counts')
        norm_counts = []
        for li in range(dist.size):
            if li in cnt:
                norm_counts.append(cnt[li]*1.0/size)
            else:
                norm_counts.append(0)
        logger.debug(norm_counts)
        logger.debug('='*80)
        logger.debug('')

        logger.debug('Regenerating Label Sequence ...')
        label_sequence = get_label_sequence(dist,size)
        for li in range(dist.size):
            if li in cnt:
                euc_distance += ((cnt[li] * 1.0 / size) - dist[li]) ** 2
            else:
                euc_distance += dist[li] ** 2

    assert euc_distance<euc_threshold
    return label_sequence

def get_augmented_sample_for_label(dataset_info,dataset,label,use_counter):
    global image_abuse_threshold
    global save_gen_data,gen_save_count,gen_perist_dir
    # dataset_info => ['type']['image_size']['resize_to']['num_channels']
    dataset_type,image_size = dataset_info['dataset_type'],dataset_info['image_size']

    num_channels = dataset_info['num_channels']

    resize_to = dataset_info['resize_to']

    image_index = np.random.choice(list(np.where(dataset['labels']==label)[0].flatten()))
    # don't use single image too much. It seems lot of data is going unused
    # May be we shouldn't impose uniform use. That could lead to lot of duplicates in training set

    uint8_image = dataset['dataset'][image_index,:,:,:]
    im = Image.fromarray(uint8_image)

    #assert np.max(uint8_image)>30 and np.max(uint8_image)<256
    #assert np.min(uint8_image)>=0
    if dataset_type=='cifar-10' or dataset_type=='cifar-100':
        x, y = 4,4
        im = im.crop((x, y, x + resize_to, y + resize_to))

    if dataset_type=='imagenet-100':
        im.thumbnail((resize_to+50,resize_to+50), Image.ANTIALIAS)
    if dataset_type=='imagenet-100' and np.random.random()<0.25:
        center = image_size//2
        x,y = np.random.randint(center-(resize_to//2)-20,center-(resize_to//2)+20),np.random.randint(center-(resize_to//2)-20,center-(resize_to//2)+20)
        im = im.crop((x,y,x+resize_to,y+resize_to))
    elif dataset_type=='imagenet-100':
        im.thumbnail((resize_to, resize_to), Image.ANTIALIAS)

    if np.random.random()<0.01:
        angle = np.random.choice([15, 30, 345])
        im = im.rotate(angle)

    if np.random.random()<0.4:
        bright_amount = np.random.random()*1.4+0.25
        bri_enhancer = ImageEnhance.Brightness(im)
        im = bri_enhancer.enhance(bright_amount)

    if np.random.random()<0.4:
        cont_amount = np.random.random()*1.4 + 0.25
        cont_enhancer = ImageEnhance.Contrast(im)
        im = cont_enhancer.enhance(cont_amount)

    if dataset_type!= 'svhn-10' and np.random.random()<0.4:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)

    if save_gen_data and np.random.random() < 0.1:
        if not os.path.exists(gen_perist_dir):
            os.makedirs(gen_perist_dir)

        lbl_dir = gen_perist_dir + os.sep + str(label)
        if not os.path.exists(lbl_dir):
            gen_save_count[label] = 0
            os.makedirs(lbl_dir)
        imsave(lbl_dir + os.sep + dataset_type + '_image_'
               + str(gen_save_count[label]) + '_' + '.png',
               im)
        gen_save_count[label] += 1

    #logger.info("UnNormalized: %.3f,%.3f", np.max(sample_img), np.min(sample_img))
    #logger.info('selected op: %s', selected_op)

    # normalization
    sample_img = normalize_img(np.asarray(im,dtype=np.uint8))

    #logger.info("Normalized: %.3f,%.3f",np.max(sample_img),np.min(sample_img))
    #logger.info('selected op: %s', selected_op)

    if np.all(np.isnan(sample_img)):
        logger.info("%.3f,%.3f", np.max(sample_img), np.min(sample_img))
    assert not np.all(np.isnan(sample_img))

    if dataset_type =='imagenet-100':
        assert sample_img.shape[0]==resize_to
    elif dataset_type=='cifar-10' or dataset_type=='svhn-10'or dataset_type=='cifar-100':
        assert sample_img.shape[0]==resize_to

    return image_index,sample_img

def generate_step_priors(batch_size,elements,chunk_size,num_labels,step_length):
    chunk_count = int(elements / chunk_size)
    raise NotImplementedError


# generate gaussian priors
def generate_gaussian_priors_for_labels(batch_size,elements,chunk_size,num_labels):
    chunk_count = int(elements/chunk_size)

    # the upper bound of x defines the number of peaks
    # smaller bound => less peaks
    # larger bound => more peaks
    x = np.linspace(0,max(num_labels,100), chunk_count).reshape(-1, 1)
    # 1e-6 * is for numerical stibility
    L = np.linalg.cholesky(kernel(x, x) + 1e-6 * np.eye(chunk_count))

    # single chunk represetn a single point on x axis of the gaussian curve
    # f_prior represent how the data distribution looks like at each chunk
    # f_prior [chunk_count,num_labels] size
    # e.g. f_prior[0] is class distribution at first chunk of elements
    f_prior = np.dot(L, np.random.normal(size=(chunk_count, num_labels)))
    # normalization
    f_prior -= f_prior.min()
    # use below line to control the scale of gaussians higher the power, higher the fluctuations are
    # original was sqrt(num_labels)
    f_prior = f_prior ** math.ceil(num_labels)
    f_prior /= np.sum(f_prior, axis=1).reshape(-1, 1)

    return f_prior

save_gen_data = False
gen_save_count = {}
gen_perist_dir = 'gen_data_test'

def sample_cifar_10_with_distribution(dataset_info, data_dir, f_prior,
                                      save_directory, new_dataset_filename, new_labels_filename):
    global image_use_counter,logger,class_distribution_logger,save_gen_data

    # dataset_info => ['elements']['chunk_size']['type']['image_size']['resize_dim']['num_channels']
    elements,chunk_size = dataset_info['elements'],dataset_info['chunk_size']
    num_chunks = elements/chunk_size
    num_slices = dataset_info['dataset_size']/dataset_info['chunk_size']
    resize_to = dataset_info['resize_to']
    dataset_type,image_size = dataset_info['dataset_type'],dataset_info['image_size']
    num_channels = dataset_info['num_channels']

    fp1 = np.memmap(filename=new_dataset_filename, dtype='float32', mode='w+', shape=(elements,resize_to,resize_to,num_channels))
    fp2 = np.memmap(filename=new_labels_filename, dtype='int32', mode='w+', shape=(elements,1))

    memmap_idx = 0
    for i,dist in enumerate(f_prior):
        dataset = load_slice_from_cifar_10(dataset_info,data_dir,int(i%num_slices)*chunk_size,
                                           min(int((i%num_slices)+1)*chunk_size,dataset_info['dataset_size']))
        label_sequence = sample_from_distribution(dist,chunk_size)

        for label in label_sequence:
            sidx,sample = get_augmented_sample_for_label(dataset_info,dataset,label,image_use_counter)
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
            dist_str += str(cnt[li]/len(label_sequence)) + ',' if li in cnt else str(0) + ','
        class_distribution_logger.info('%d,%s',i,dist_str)


def sample_cifar_100_with_distribution(dataset_info, data_filename, f_prior,
                                      save_directory, new_dataset_filename, new_labels_filename):
    global image_use_counter,logger,class_distribution_logger,save_gen_data

    # dataset_info => ['elements']['chunk_size']['type']['image_size']['resize_dim']['num_channels']
    elements,chunk_size = dataset_info['elements'],dataset_info['chunk_size']
    num_chunks = elements/chunk_size
    num_slices = dataset_info['dataset_size']/dataset_info['chunk_size']
    dataset_type,image_size = dataset_info['dataset_type'],dataset_info['image_size']
    resize_to = dataset_info['resize_to']
    num_channels = dataset_info['num_channels']

    fp1 = np.memmap(filename=new_dataset_filename, dtype='float32', mode='w+', shape=(elements,resize_to,resize_to,num_channels))
    fp2 = np.memmap(filename=new_labels_filename, dtype='int32', mode='w+', shape=(elements,1))

    memmap_idx = 0
    for i,dist in enumerate(f_prior):
        start_slice = int(i%num_slices)*chunk_size
        end_slice = min(int((i%num_slices)+1)*chunk_size,dataset_info['dataset_size'])
        # rare occasions where slice is too small to have data for all classes
        if end_slice - start_slice < num_labels * 2:
            rand_idx = np.random.randint(0,num_slices-1)
            start_slice = int(rand_idx % num_slices) * chunk_size
            end_slice = min(int((rand_idx % num_slices) + 1) * chunk_size, dataset_info['dataset_size'])
        dataset = load_slice_from_cifar_100(dataset_info,data_filename,start_slice,end_slice)
        label_sequence = sample_from_distribution(dist,chunk_size)

        for label in label_sequence:
            sidx,sample = get_augmented_sample_for_label(dataset_info,dataset,label,image_use_counter)
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
            dist_str += str(cnt[li]/len(label_sequence)) + ',' if li in cnt else str(0) + ','
        class_distribution_logger.info('%d,%s',i,dist_str)


def sample_svhn_10_with_distribution(dataset_info, data_filename, f_prior,
                                      save_directory, new_dataset_filename, new_labels_filename):
    global image_use_counter,logger,class_distribution_logger
    # dataset_info => ['elements']['chunk_size']['type']['image_size']['resize_dim']['num_channels']
    elements,chunk_size = dataset_info['elements'],dataset_info['chunk_size']
    num_chunks = elements/chunk_size
    num_slices = dataset_info['dataset_size']/dataset_info['chunk_size']
    dataset_type,image_size = dataset_info['dataset_type'],dataset_info['image_size']
    num_channels = dataset_info['num_channels']

    fp1 = np.memmap(filename=new_dataset_filename, dtype='float32', mode='w+', shape=(elements,image_size,image_size,num_channels))
    fp2 = np.memmap(filename=new_labels_filename, dtype='int32', mode='w+', shape=(elements,1))

    memmap_idx = 0
    for i,dist in enumerate(f_prior):
        dataset = load_slice_from_svhn_10(dataset_info,data_filename,int(i%num_slices)*chunk_size,
                                           min(int((i%num_slices)+1)*chunk_size,dataset_info['dataset_size']))
        label_sequence = sample_from_distribution(dist,chunk_size)

        for label in label_sequence:
            sidx,sample = get_augmented_sample_for_label(dataset_info,dataset,label,image_use_counter)
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
            dist_str += str(cnt[li]/len(label_sequence)) + ',' if li in cnt else str(0) + ','
        class_distribution_logger.info('%d,%s',i,dist_str)


def sample_imagenet_with_distribution(dataset_info, dataset_filename, label_filename, f_prior, new_dataset_fname, new_labels_fname):
    global image_use_counter,logger,class_distribution_logger
    # dataset_info => ['elements']['chunk_size']['type']['image_size']['resize_dim']['num_channels']
    elements,chunk_size = dataset_info['elements'],dataset_info['chunk_size']
    num_chunks = elements/chunk_size
    num_slices = dataset_info['dataset_size']/dataset_info['data_in_memory']
    dataset_type,image_size,resize_dim = dataset_info['dataset_type'],dataset_info['image_size'],dataset_info['resize_to']
    num_channels = dataset_info['num_channels']

    fp1 = np.memmap(filename=new_dataset_fname, dtype='float32', mode='w+', shape=(elements,resize_dim,resize_dim,num_channels))
    fp2 = np.memmap(filename=new_labels_fname, dtype='int32', mode='w+', shape=(elements,1))

    memmap_idx = 0
    for i,dist in enumerate(f_prior):
        dataset = load_slice_from_imagenet(dataset_info,dataset_filename,label_filename,int(i%num_slices)*chunk_size,
                                           min(int((i%num_slices)+1)*chunk_size,dataset_info['dataset_size']))
        label_sequence = sample_from_distribution(dist,chunk_size)

        for label in label_sequence:
            sidx,sample = get_augmented_sample_for_label(dataset_info,dataset,label,image_use_counter)
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
            dist_str += str(cnt[li]) + ',' if li in cnt else str(0) + ','
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


def load_slice_from_cifar_10(dataset_info,data_dir,start_idx,end_idx):
    global image_use_counter,logger
    global train_dataset,train_labels

    image_use_counter = {}
    col_count = (dataset_info['image_size'],dataset_info['image_size'],dataset_info['num_channels'])

    # Loading data from memmap
    logger.info('Processing files in %s',data_dir)
    logger.info('Reading data from: %d to %d',start_idx,end_idx)

    if train_labels is None and train_dataset is None:
        filenames = [os.path.join(data_dir,'cifar_10_data_batch_%d'%i) for i in range(1,6)]
        for fname in filenames:
            with open(fname,'rb') as f:
                save = pickle.load(f,encoding='latin1')
                dataset, labels = save['data'],save['labels']
                dataset = dataset.reshape((-1,col_count[0],col_count[1],col_count[2]),order='F').astype(np.uint8)
                dataset = np.transpose(dataset,[0,2,1,3])
                if train_dataset is None:
                    train_dataset = dataset
                    train_labels = labels
                else:
                    train_dataset = np.append(train_dataset,dataset,axis=0)
                    train_labels = np.append(train_labels,labels,axis=0)

    return {'dataset':train_dataset[start_idx:end_idx+1,:,:,:],'labels':train_labels[start_idx:end_idx+1,None]}

def load_slice_from_cifar_100(dataset_info,data_filename,start_idx,end_idx):
    global image_use_counter,logger
    global train_dataset,train_labels

    image_use_counter = {}
    col_count = (dataset_info['image_size'],dataset_info['image_size'],dataset_info['num_channels'])

    # Loading data from memmap
    logger.info('Processing files %s',data_filename)
    logger.info('Reading data from: %d to %d',start_idx,end_idx)

    if train_labels is None and train_dataset is None:
        # test file
        with open(data_filename, 'rb') as f:
            save = pickle.load(f, encoding="latin1")
            train_dataset, train_labels = np.asarray(save['data']),np.asarray(save['fine_labels'])
            train_dataset = train_dataset.reshape((-1,col_count[0],col_count[1],col_count[2]),order='F').astype(np.uint8)
            train_dataset = np.transpose(train_dataset,[0,2,1,3])
            train_labels = train_labels.reshape(-1,)

    return {'dataset':train_dataset[start_idx:end_idx+1,:,:,:],'labels':train_labels[start_idx:end_idx+1,None]}


def load_slice_from_svhn_10(dataset_info,data_filename,start_idx,end_idx):
    global image_use_counter,logger
    global train_dataset,train_labels

    image_use_counter = {}
    col_count = (dataset_info['image_size'],dataset_info['image_size'],dataset_info['num_channels'])

    # Loading data from memmap
    logger.info('Processing files %s',data_filename)
    logger.info('Reading data from: %d to %d',start_idx,end_idx)

    if train_labels is None and train_dataset is None:
        data_dict = scipy.io.loadmat(data_filename)
        X,y = data_dict['X'],data_dict['y']

        train_dataset = np.asarray(X.transpose(3,0,1,2),dtype=np.uint8)

        assert np.max(train_dataset[:10, :, :, :]) <= 255 and np.max(train_dataset[:10, :, :, :]) > 80
        train_labels = np.asarray(y,dtype=np.int32).reshape(-1,)
        train_labels = train_labels - 1 # labels go from 1 to 10

        logger.info('Shape of X: %s',train_dataset.shape)
        logger.info('Shape of y: %s',train_labels.shape)

        logger.info('y: %s',train_labels[:10])
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

    fp1 = np.memmap(filename=save_directory+os.sep+'imagenet-100-test-dataset.pkl', dtype='float32', mode='w+', shape=(test_size,resize_dim,resize_dim,num_channels))
    fp2 = np.memmap(filename=save_directory+os.sep+'imagenet-100-test-labels.pkl', dtype='int32', mode='w+', shape=(test_size,1))

    idx = 0
    for img,lbl in zip(test_dataset[:],test_labels[:]):
        unnorm_image = img-np.min(img)
        unnorm_image /= np.max(unnorm_image)
        unnorm_image *= 255.0
        im = Image.fromarray(np.uint8(unnorm_image))
        im.thumbnail((resize_to,resize_to), Image.ANTIALIAS)
        sample_img = np.array(im)
        sample_img = normalize_img(sample_img)
        #assert np.min(sample_img)<-0.2 and np.max(sample_img)>0.2
        fp1[idx,:,:,:]=sample_img
        fp2[idx,0]=lbl
        idx+=1

    del fp1,fp2

def generate_cifar_test_data(data_dir,save_directory):
    image_size = dataset_info['image_size']
    resize_to = dataset_info['resize_to']
    test_size = dataset_info['test_size']
    num_channels = dataset_info['num_channels']
    col_count = (dataset_info['image_size'],dataset_info['image_size'],dataset_info['num_channels'])

    with open(os.path.join(data_dir,'cifar_10_test_batch'),'rb') as f:
        save = pickle.load(f,encoding='latin1')
        test_dataset, test_labels = save['data'],save['labels']

    fp1 = np.memmap(filename=save_directory+os.sep+'cifar-10-test-dataset.pkl', dtype='float32', mode='w+', shape=(test_size,resize_to,resize_to,num_channels))
    fp2 = np.memmap(filename=save_directory+os.sep+'cifar-10-test-labels.pkl', dtype='int32', mode='w+', shape=(test_size,1))

    idx = 0
    for img,lbl in zip(test_dataset[:],test_labels[:]):
        img = img.reshape((col_count[0],col_count[1],col_count[2]),order='F').astype(np.uint8)
        img = np.transpose(img,[1,0,2])

        im = Image.fromarray(img)
        x, y = 4, 4
        im = im.crop((x, y, x + resize_to, y + resize_to))
        img = normalize_img(np.asarray(im))

        fp1[idx,:,:,:]=img
        fp2[idx,0]=lbl
        idx+=1

    del fp1,fp2


def generate_cifar_100_test_data(data_filename,save_directory):
    image_size = dataset_info['image_size']
    test_size = dataset_info['test_size']
    num_channels = dataset_info['num_channels']
    col_count = (dataset_info['image_size'],dataset_info['image_size'],dataset_info['num_channels'])

    with open(data_filename,'rb') as f:
        save = pickle.load(f,encoding="latin1")
        test_dataset, test_labels = np.asarray(save['data']),np.asarray(save['fine_labels']).reshape(-1,)
    logger.info('Test data summary: %s, %d',test_dataset.shape,test_labels.shape[0])

    fp1 = np.memmap(filename=save_directory+os.sep+'cifar-100-test-dataset.pkl', dtype='float32', mode='w+', shape=(test_size,image_size,image_size,num_channels))
    fp2 = np.memmap(filename=save_directory+os.sep+'cifar-100-test-labels.pkl', dtype='int32', mode='w+', shape=(test_size,1))

    idx = 0
    for img,lbl in zip(test_dataset[:],test_labels[:]):
        img = img.reshape((col_count[0],col_count[1],col_count[2]),order='F').astype(np.uint8)
        img = np.transpose(img,[1,0,2])

        im = Image.fromarray(img)
        x, y = 4, 4
        im = im.crop((x, y, x + resize_to, y + resize_to))
        img = normalize_img(np.asarray(im))

        fp1[idx,:,:,:]=img
        fp2[idx,0]=lbl
        idx+=1

    del fp1,fp2


def generate_svhn_test_data(dataset_info, data_filename,save_directory):
    image_size = dataset_info['image_size']
    test_size = dataset_info['test_size']
    num_channels = dataset_info['num_channels']
    col_count = (dataset_info['image_size'],dataset_info['image_size'],dataset_info['num_channels'])

    data_dict = scipy.io.loadmat(data_filename)
    X, y = data_dict['X'], data_dict['y']

    test_dataset = np.asarray(X.transpose(3, 0, 1, 2), dtype=np.uint8)
    assert np.max(test_dataset[:10,:,:,:])<=255 and np.max(test_dataset[:10,:,:,:])>80

    test_labels = np.asarray(y, dtype=np.int32).reshape(-1,)
    test_labels = test_labels - 1

    fp1 = np.memmap(filename=save_directory+os.sep+'svhn-10-test-dataset.pkl', dtype='float32', mode='w+', shape=(test_size,image_size,image_size,num_channels))
    fp2 = np.memmap(filename=save_directory+os.sep+'svhn-10-test-labels.pkl', dtype='int32', mode='w+', shape=(test_size,1))

    idx = 0
    for img,lbl in zip(test_dataset,test_labels[:]):
        img = normalize_img(img)
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
    elif dataset_type=='cifar-10' or dataset_type=='cifar-100':
        col_count = (dataset_info['image_size'],dataset_info['image_size'],dataset_info['num_channels'])
    elif dataset_type == 'svhn-10':
        col_count = (dataset_info['image_size'], dataset_info['image_size'], dataset_info['num_channels'])

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
        imsave(lbl_dir + os.sep + dataset_type+'_image_'+str(save_count_data[lbl])+'.png',random_slice_dataset[li,:,:,:])
        save_count_data[lbl]+=1


def normalize_img(img_uint8):
    img = img_uint8.astype('float32')
    return (img-np.mean(img))/max(np.std(img),1.0/img.size)

image_use_counter = None
logger = None
class_distribution_logger = None
image_abuse_threshold = 3
if __name__ == '__main__':

    global logger
    logger = logging.getLogger('main_logger')
    logger.setLevel(logging_level)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(logging_format))
    console.setLevel(logging_level)
    logger.addHandler(console)

    persist_dir = 'data_generator_dir' # various things we persist related to ConstructorRL

    dataset_type = 'cifar-100'  # 'cifar-10 imagenet-100
    distribution_type = 'stationary'
    distribution_type2 = 'gauss'  # gauss or step
    data_save_directory = 'data_non_station'

    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)

    batch_size = 128
    elements = int(batch_size*10000) # number of elements in the whole dataset
    # there are elements/chunk_size points in the gaussian curve for each class
    chunk_size = int(batch_size*10) # number of samples sampled for each instance of the gaussian curve

    class_distribution_logger = logging.getLogger('class_distribution_logger')
    class_distribution_logger.setLevel(logging.INFO)
    cdfileHandler = logging.FileHandler(persist_dir + os.sep + 'class_distribution_log' + dataset_type + '.log',
                                        mode='w')
    cdfileHandler.setFormatter(logging.Formatter('%(message)s'))
    class_distribution_logger.addHandler(cdfileHandler)

    if not os.path.exists(data_save_directory):
        os.makedirs(data_save_directory)

    if dataset_type == 'cifar-10':
        image_size = 32
        num_labels = 10
        num_channels = 3
        resize_to = 24
        dataset_size = 50000
        test_size = 10000
        image_use_counter = {}
        dataset_info = {'dataset_type':dataset_type,'elements':elements,'chunk_size':chunk_size,'image_size':image_size,'resize_to':resize_to,
                        'num_channels':num_channels,'num_labels':num_labels,'dataset_size':dataset_size,'test_size':test_size}

    if dataset_type == 'cifar-100':
        image_size = 32
        num_labels = 100
        num_channels = 3
        resize_to = 24
        dataset_size = 50000
        test_size = 10000
        image_use_counter = {}
        dataset_info = {'dataset_type':dataset_type,'elements':elements,'chunk_size':chunk_size,'image_size':image_size,'resize_to':resize_to,
                        'num_channels':num_channels,'num_labels':num_labels,'dataset_size':dataset_size,'test_size':test_size}

    elif dataset_type == 'svhn-10':
        image_size = 32
        resize_to = 32
        num_labels = 10
        num_channels = 3
        dataset_size = 73257
        test_size = 26032
        image_use_counter = {}
        dataset_info = {'dataset_type': dataset_type, 'elements': elements, 'chunk_size': chunk_size,
                        'image_size': image_size,'resize_to':resize_to,
                        'num_channels': num_channels, 'num_labels': num_labels, 'dataset_size': dataset_size,
                        'test_size': test_size}

    elif dataset_type == 'imagenet-100':

        image_size = 224
        num_labels = 100
        num_channels = 3
        dataset_size = 128000
        data_in_memory = chunk_size
        resize_to = 64
        test_size = 5000
        image_use_counter = {}
        dataset_info = {'dataset_type':dataset_type,'elements':elements,'chunk_size':chunk_size,'image_size':image_size,'num_labels':num_labels,'dataset_size':dataset_size,
                        'num_channels':num_channels,'data_in_memory':chunk_size,'resize_to':resize_to,'test_size':test_size}

    logger.info('='*60)
    logger.info('Dataset Information')
    logger.info('\t%s',str(dataset_info))
    logger.info('='*60)

    logger.info('Generating gaussian priors')
    if distribution_type=='non-stationary':
        if distribution_type2=='gauss':
            priors = generate_gaussian_priors_for_labels(batch_size,elements,chunk_size,num_labels)
        elif distribution_type2=='step':
            step_length=25
            priors = generate_step_priors(batch_size,elements,chunk_size,num_labels,step_length)
    elif distribution_type=='stationary':
        priors = np.ones((elements//chunk_size,num_labels))*(1.0/num_labels)
    else:
        raise NotImplementedError

    logger.info('\tGenerated priors of size: %d,%d',priors.shape[0],priors.shape[1])

    if dataset_type == 'imagenet-100':
        dataset_filename = '..'+os.sep+'imagenet_small'+os.sep+'imagenet_small_train_dataset'
        label_filename = '..'+os.sep+'imagenet_small'+os.sep+'imagenet_small_train_labels'
        if distribution_type == 'non-stationary':
            new_dataset_filename = data_save_directory+os.sep+'imagenet-100-nonstation-dataset.pkl'
            new_labels_filename = data_save_directory+os.sep+'imagenet-100-nonstation-labels.pkl'
        elif distribution_type == 'stationary':
            new_dataset_filename = data_save_directory + os.sep + 'imagenet-100-station-dataset.pkl'
            new_labels_filename = data_save_directory + os.sep + 'imagenet-100-station-labels.pkl'
        else:
            raise NotImplementedError

        print(new_dataset_filename)
        sample_imagenet_with_distribution(dataset_info, dataset_filename, label_filename, priors, new_dataset_filename, new_labels_filename)
        #generate_imagenet_test_data(dataset_filename,label_filename,data_save_directory)
    elif dataset_type == 'svhn-10':
        test_data_filename = '..' +os.sep +'svhn' + os.sep + 'test_32x32.mat'
        train_data_filename = '..' + os.sep + 'svhn' + os.sep + 'train_32x32.mat'

        dataset_filename = '..' + os.sep + 'svhn' + os.sep + ''
        label_filename = '..' + os.sep + 'svhn' + os.sep + ''
        if distribution_type == 'non-stationary':
            new_dataset_filename = data_save_directory + os.sep + 'svhn-10-nonstation-dataset.pkl'
            new_labels_filename = data_save_directory + os.sep + 'svhn-10-nonstation-labels.pkl'
        elif distribution_type == 'stationary':
            new_dataset_filename = data_save_directory + os.sep + 'svhn-10-station-dataset.pkl'
            new_labels_filename = data_save_directory + os.sep + 'svhn-10-station-labels.pkl'
        else:
            raise NotImplementedError

        print(new_dataset_filename)
        sample_svhn_10_with_distribution(dataset_info, train_data_filename, priors, data_save_directory, new_dataset_filename, new_labels_filename)
        #generate_svhn_test_data(dataset_info,test_data_filename,data_save_directory)

    elif dataset_type == 'cifar-10':
        data_filename = '..'+os.sep+'..'+os.sep+'data'+os.sep+'cifar-10.pickle'
        if distribution_type=='non-stationary':
            new_dataset_filename = data_save_directory+os.sep+'cifar-10-nonstation-dataset.pkl'
            new_labels_filename = data_save_directory+os.sep+'cifar-10-nonstation-labels.pkl'
        elif distribution_type == 'stationary':
            new_dataset_filename = data_save_directory + os.sep + 'cifar-10-station-dataset.pkl'
            new_labels_filename = data_save_directory + os.sep + 'cifar-10-station-labels.pkl'
        else:
            raise NotImplementedError

        print(new_dataset_filename)
        sample_cifar_10_with_distribution(dataset_info, 'data'+os.sep+'cifar-10-pickle', priors, data_save_directory, new_dataset_filename, new_labels_filename)
        #generate_cifar_test_data('data'+os.sep+'cifar-10-pickle', data_save_directory)

    elif dataset_type == 'cifar-100':
        data_filename = '..'+os.sep+'..'+os.sep+'data'+os.sep+'cifar-100-python'+os.sep+'train'
        test_data_filename = '..' + os.sep + '..' + os.sep + 'data' + os.sep + 'cifar-100-python' + os.sep + 'test'
        if distribution_type=='non-stationary':
            new_dataset_filename = data_save_directory+os.sep+'cifar-100-nonstation-dataset.pkl'
            new_labels_filename = data_save_directory+os.sep+'cifar-100-nonstation-labels.pkl'
        elif distribution_type == 'stationary':
            new_dataset_filename = data_save_directory + os.sep + 'cifar-100-station-dataset.pkl'
            new_labels_filename = data_save_directory + os.sep + 'cifar-100-station-labels.pkl'
        else:
            raise NotImplementedError

        print(new_dataset_filename)
        #generate_cifar_100_test_data(test_data_filename,data_save_directory)
        sample_cifar_100_with_distribution(dataset_info, data_filename, priors, data_save_directory, new_dataset_filename, new_labels_filename)

    #test_generated_data(dataset_info,persist_dir+os.sep+'test_data_'+dataset_info['dataset_type'],new_dataset_filename,new_labels_filename)

    # =============== Quick Test =====================
    '''sample_size = 1000
    num_labels = 10
    x = np.linspace(0, 50, sample_size).reshape(-1, 1)
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

    x_axis = np.arange(sample_size)

    import matplotlib.pyplot as plt
    for i in range(num_labels):
        plt.plot(x_axis,f_prior[:,i])

    #print(np.sum(f_prior,axis=1).shape)
    #plt.plot(x_axis,np.sum(f_prior,axis=1))
    plt.show()'''

