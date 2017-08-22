__author__ = 'Thushan Ganegedara'

from six.moves import cPickle as pickle
from six.moves import range
import numpy as np
import os
from scipy import ndimage
from PIL import Image
import xml.etree.ElementTree as ET
from math import ceil,floor
from scipy.misc import imsave

def save_imagenet_as_memmaps(train_dir,valid_dir, valid_annotation_dir, gloss_fname, n_nat_classes, n_art_classes, resize_images_to, save_dir):
    '''
    Retrieves images of imagenet data and store them in a memmap
    :param train_dir: Train data dir (e.g. .../Data/CLS-LOC/train/)
    :param valid_dir: Valid data dir (e.g. .../Data/CLS-LOC/val/)
    :param valid_annotation_dir: Annotation dir (e.g. .../ILSVRC2015/Annotations/CLS-LOC/val/)
    :param n_nat_classes: How many natural object classes in the data
    :param n_art_classes: How many artificial object classes in the data
    :return:
    '''

    dataset_filename = save_dir + os.sep +  'imagenet_250_train_dataset'
    label_filename = save_dir + os.sep + 'imagenet_250_train_labels'
    valid_dataset_filename = save_dir + os.sep+ 'imagenet_250_valid_dataset'
    valid_label_filename = save_dir + os.sep+ 'imagenet_250_valid_labels'
    gloss_cls_loc_fname = save_dir + os.sep+ 'gloss-cls-loc.txt' # the gloss.txt has way more categories than the 1000 imagenet ones. This files saves the 1000

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    num_channels = 3

    # label map is needed because I have to create my own class labels
    # my label -> actual label
    new_to_old_label_map = dict()

    # data looks like below
    # n01440764/n01440764_10026 1
    # n01440764/n01440764_10027 2
    # n01440764/n01440764_10029 3


    valid_map = build_or_retrieve_valid_mapping_from_filename_to_synset_id(valid_annotation_dir,save_dir)


    # ignoring the 0th element because it is just a space
    train_subdirectories = [os.path.split(x[0])[1] for x in os.walk(train_dir)][1:]
    print('Subdirectories: %s\n' % train_subdirectories[:5])
    print('Num subdir: ',len(train_subdirectories))

    synset_id_to_desc_map = write_art_nat_ordered_class_descriptions(train_subdirectories,gloss_fname, gloss_cls_loc_fname)

    selected_natural_synsets, selected_artificial_synsets, id_to_label_map = sample_n_natural_and_artificial_classes(n_nat_classes,n_art_classes, gloss_cls_loc_fname)

    print('Summary of selected synsets ...')
    print('\tNatural (%d): %s' % (len(selected_natural_synsets), selected_natural_synsets[:5]))
    print('\tArtificial (%d): %s' % (len(selected_artificial_synsets), selected_artificial_synsets[:5]))

    # check if the class synsets we chose are actually in the training data
    for _ in range(10):
        rand_nat = np.random.choice(selected_natural_synsets)
        rand_art = np.random.choice(selected_artificial_synsets)

        assert rand_nat in train_subdirectories
        assert rand_art in train_subdirectories

    write_selected_art_nat_synset_ids_and_descriptions(selected_natural_synsets,selected_artificial_synsets,
                                                       synset_id_to_desc_map,'selected-gloss.txt',save_dir)

    # we use a shuffled train set when storing to avoid any order
    n_train = 0
    train_filenames = []
    train_synset_ids = []
    for subdir in train_subdirectories:
        if subdir in selected_natural_synsets or subdir in selected_artificial_synsets:
            file_count = len([file for file in os.listdir(train_dir+os.sep+subdir) if file.endswith('.JPEG')])
            train_filenames.extend([train_dir+os.sep+subdir+os.sep+file for file in os.listdir(train_dir+os.sep+subdir) if file.endswith('.JPEG')])
            train_synset_ids.extend([subdir for _ in range(file_count)])
            n_train += file_count

    train_filenames, train_synset_ids = shuffle_in_unison_scary(train_filenames, train_synset_ids)

    print('Found %d training samples in %d subdirectories...\n' % (n_train, len(train_subdirectories)))
    assert n_train > 0, 'No training samples found'

    # Create memmaps for saving new resized subset of data

    print('Creating train dataset ...')

    if not os.path.exists(dataset_filename):
        fp1 = np.memmap(filename=dataset_filename, dtype='float32', mode='w+',
                        shape=(n_train, resize_images_to, resize_images_to, num_channels))
        fp2 = np.memmap(filename=label_filename, dtype='int32', mode='w+', shape=(n_train, 1))

        filesize_dictionary = dict()
        print("\tMemory allocated for (%d items)..." % n_train)
        filesize_dictionary['train_dataset'] = n_train

        save_train_data_in_filenames(train_filenames,train_synset_ids,fp1,fp2, resize_images_to, num_channels, id_to_label_map)

        with open(data_save_directory + os.sep + 'synset_id_to_my_label.pkl','wb') as f:
            pickle.dump(id_to_label_map, f, pickle.HIGHEST_PROTOCOL)

    else:
        print('Training data exists ...')

        with open(data_save_directory + os.sep + 'synset_id_to_my_label.pkl','rb') as f:
            id_to_label_map = pickle.load(f)

    write_selected_art_nat_synset_ids_and_descriptions(selected_natural_synsets,selected_artificial_synsets,
                                                       valid_map,'gloss-cls-loc-with-class.txt',save_dir,id_to_label_map)

    del fp1, fp2 # delete to save memory

    assert len(id_to_label_map) == n_nat_classes + n_art_classes, \
        'Label map ([synset_id] -> my class label) size does not math class counts'

    # --------------------------------------------------------------------------------
    # Saving validation data
    # --------------------------------------------------------------------------------

    valid_filenames, valid_classes = zip(*valid_map.items())
    print('\tValid filenames:', list(valid_map.keys())[:5])
    print('\tValid classes:', list(valid_map.values())[:5])

    # only get the valid data points related to the classes I picked
    selected_valid_files = []
    selected_valid_synset_ids = []
    for f, c in zip(valid_filenames, valid_classes):
        # only select the ones in the classes I picked
        if c in list(id_to_label_map.keys()):
            fname = f.rstrip('.xml') + '.JPEG'
            selected_valid_files.append(valid_dir+os.sep+fname)
            selected_valid_synset_ids.append(c)

    print('Found %d matching valid files...' % len(selected_valid_files))

    if not os.path.exists(save_dir + os.sep+ valid_dataset_filename):

        fp1 = np.memmap(filename=save_dir + valid_dataset_filename, dtype='float32', mode='w+',
                        shape=(len(selected_valid_files), resize_images_to, resize_images_to, num_channels))
        fp2 = np.memmap(filename=save_dir + valid_label_filename, dtype='int32', mode='w+',
                        shape=(len(selected_valid_files), 1))

        save_train_data_in_filenames(selected_valid_files,selected_valid_synset_ids,fp1,fp2, resize_images_to, num_channels,id_to_label_map)

        filesize_dictionary['valid_dataset'] = len(selected_valid_files)
        del fp1, fp2
        print('Created tha valid file with %d entries' %filesize_dictionary['valid_dataset'])
    else:
        print('Valid data exists.')

    with open(save_dir + 'dataset_sizes.pickle', 'wb') as f:
        pickle.dump(filesize_dictionary, f, pickle.HIGHEST_PROTOCOL)
    with open(save_dir + 'dataset_sizes.txt', 'w') as f:
        f.write(filesize_dictionary)


def shuffle_in_unison_scary(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

    return a,b

def build_or_retrieve_valid_mapping_from_filename_to_synset_id(valid_annotation_dir, data_save_dir):

    print('Building a mapping from valid file name to synset id (also the folder names in the training folder)')
    valid_map_fname = data_save_dir + os.sep + 'valid_fname_to_synset_id_map.pickle'
    # valid_map contains a dictionary mapping filename to synset id (e.g. n01440764 is a synset id)
    if not os.path.exists(valid_map_fname):
        valid_map = dict()
        for file in os.listdir(valid_annotation_dir):
            if file.endswith(".xml"):
                head, tail = os.path.split(file)
                tree = ET.parse(valid_annotation_dir + os.sep + file)
                root = tree.getroot()
                for name in root.iter('name'):
                    valid_map[tail] = str(name.text)
                    break
        print('Created a dictionary (valid image file name > synset id )')

        with open(valid_map_fname, 'wb') as f:
            pickle.dump(valid_map, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(valid_map_fname, 'rb') as f:
            valid_map = pickle.load(f)

    return valid_map


def write_art_nat_ordered_class_descriptions(train_subdirectories,gloss_file_name,gloss_cls_loc_fname):
    '''
    Creates an ordered gloss.txt by the synset ID
    This is important because the first 398 synset IDs are natural items
    and rest is artificial
    :param train_subdirectories:
    :param gloss_file_name: Original gloss.txt
    :param save_dir: Data saving directory
    :return:
    '''

    train_class_descriptions = []
    synset_id_to_description = {}
    with open(gloss_file_name, 'r') as f:
        for line in f:
            subdir_str = line.split('\t')[0]
            if subdir_str in train_subdirectories:
                synset_id_to_description[subdir_str] = line.split('\t')[1]
                train_class_descriptions.append(line)

    with open(gloss_cls_loc_fname, 'w') as f:
        for desc_str in train_class_descriptions:
            f.write(desc_str)

    return synset_id_to_description


def sample_n_natural_and_artificial_classes(n_nat,n_art,gloss_fname):
    '''
    Sample n_nat natural classes and n_art artificial classes from imagenet
    :param n_nat:
    :param n_art:
    :return:
    '''
    natural_synsets, artificial_synsets = [], []
    with open(gloss_fname, 'r') as f:
        for index, line in enumerate(f):
            subdir_str = line.split('\t')[0]
            if index < 398:
                natural_synsets.append(subdir_str)
            else:
                artificial_synsets.append(subdir_str)

    natural_synsets = np.random.permutation(natural_synsets)
    selected_natural_synsets = list(natural_synsets[:n_nat])

    artificial_synsets = np.random.permutation(artificial_synsets)
    selected_artificial_synsets = list(artificial_synsets[:n_art])

    all_selected = selected_natural_synsets + selected_artificial_synsets
    class_labels = range(0,n_nat+n_art)
    synset_id_to_label_map = dict(all_selected,class_labels)
    return selected_natural_synsets,selected_artificial_synsets,synset_id_to_label_map


def write_selected_art_nat_synset_ids_and_descriptions(sel_nat_synsets, sel_art_synsets, synset_id_description_map, filename,save_dir,label_map=None):
    '''
    Write the selected synset ids to a file
    Good for verifying that there are needed amount of natural and artificlal classes
    :param sel_nat_synsets:
    :param sel_art_synsets:
    :param synset_id_description_map:
    :param filename:
    :param save_dir:
    :return:
    '''

    with open(save_dir + os.sep + filename, 'w') as f:
        for nat_id in sel_nat_synsets:
            if not label_map:
                current_str = str(nat_id) + '\t' + synset_id_description_map[nat_id]
            else:
                current_str = str(nat_id) +'\t' + label_map[nat_id] + '\t' + synset_id_description_map[nat_id]

            f.write(current_str)

        for art_id in sel_art_synsets:
            if not label_map:
                current_str = str(art_id) + '\t' + synset_id_description_map[art_id]
            else:
                current_str = str(art_id) + '\t' + label_map[art_id] + '\t' + synset_id_description_map[art_id]
            f.write(current_str)


def resize_image(fname,resize_to,n_channels):
    '''
    resize image
    if the resize size is more than the actual size, we pad with zeros
    if the image is black and white, we create 3 channels of same data
    :param fname:
    :return:
    '''
    im = Image.open(fname)
    im.thumbnail((resize_to,resize_to), Image.ANTIALIAS)
    resized_img = np.array(im)

    if resized_img.ndim<3:
        resized_img = resized_img.reshape((resized_img.shape[0],resized_img.shape[1],1))
        resized_img = np.repeat(resized_img,3,axis=2)
        assert resized_img.shape[2]==n_channels

    if resized_img.shape[0]<resize_to:
        diff = resize_to - resized_img.shape[0]
        lpad,rpad = floor(float(diff)/2.0),ceil(float(diff)/2.0)
        #print('\tshape of resized img before padding %s'%str(resized_img.shape))
        resized_img = np.pad(resized_img,((lpad,rpad),(0,0),(0,0)),'constant',constant_values=((0,0),(0,0),(0,0)))
        #print('\tshape of resized img after padding %s'%str(resized_img.shape))
    if resized_img.shape[1]<resize_to:
        diff = resize_to - resized_img.shape[1]
        lpad,rpad = floor(float(diff)/2.0),ceil(float(diff)/2.0)
        #print('\tshape of resized img before padding %s'%str(resized_img.shape))
        resized_img = np.pad(resized_img,((0,0),(lpad,rpad),(0,0)),'constant',constant_values=((0,0),(0,0),(0,0)))
        #print('\tshape of resized img after padding %s'%str(resized_img.shape))
    assert resized_img.shape[0]==resize_to
    assert resized_img.shape[1]==resize_to
    assert resized_img.shape[2]==n_channels
    return resized_img


def save_train_data_in_filenames(train_filenames, train_synset_ids, memmap_img, memmap_labels, resize_to, n_channels, synset_to_label_map = None, n_threads=1):
    '''
    Save a batch of training data
    :param train_dir:
    :param train_filenames:
    :param memmap_img:
    :param memmap_labels:
    :param train_offset:
    :param n_channels:
    :return:
    '''

    pixel_depth = -1
    train_offset = 0

    for fidx, file in enumerate(train_filenames):

        synset_id = train_synset_ids[fidx]

        print('Processing file %s (%d)' % (file, train_offset))
        if train_offset % int(len(train_filenames) * 0.05) == 0:
            print('\t%d%% complete' % (train_offset // int(len(train_filenames) * 100.0)))

        resized_img = resize_image(file,resize_to,n_channels)

        # probably has an alpha layer, ignore these kind of images
        if resized_img.ndim == 3 and resized_img.shape[2] > n_channels:
            print('Ignoring image %s of size %s' % (file, str(resized_img.shape)))
            continue
        if pixel_depth == -1:
            pixel_depth = 255.0 if np.max(resized_img) > 128 else 1.0
            print('\tFound pixel depth %.1f' % pixel_depth)

        # standardizing the image
        resized_img = (resized_img - np.mean(resized_img))
        resized_img /= np.std(resized_img)

        if train_offset < 5:
            assert -.1 < np.mean(resized_img) < .1, 'Mean is not zero'
            assert 0.9 < np.std(resized_img) < 1.1, 'Standard deviation is not one'

        memmap_img[train_offset, :, :, :] = resized_img

        memmap_labels[train_offset, 0] = synset_to_label_map[synset_id]
        train_offset += 1


def save_data_chunk_with_offset(train_offset, train_filenames, train_synset_ids, id_to_label_map):


def reformat_data_imagenet_with_npy(train_filenames,**params):

    image_size = 224
    num_labels = 100
    num_channels = 3 # rgb
    #col_count = image_size**2 * num_channels

    dataset = np.load(train_filenames[0],'r')
    labels = np.load(train_filenames[1],'r')

    #fp1 = np.memmap(train_filenames[0],dtype=np.float32,mode='r',offset=np.dtype('float32').itemsize*memmap_offset,shape=(row_count,col_count))
    #fp2 = np.memmap(train_filenames[1],dtype=np.float32,mode='r',offset=np.dtype('float32').itemsize*memmap_offset,shape=(row_count,col_count))
    #dataset = np.empty((row_count,col_count),dtype=np.float32)
    #labels = np.empty((row_count,),dtype=np.int32)
    #dataset[:] = fp1[:]
    #labels[:,None] = fp2[:,0]
    #del fp1,fp2

    print("Reformatting data ...")
    dataset = dataset.reshape((-1,image_size,image_size,num_channels),order='C').astype(np.float32)
    labels = labels.flatten()

    if 'test_images' in params and params['test_images']:
        for i in range(10):
            rand_idx = np.random.randint(0,dataset.shape[0])
            imsave('test_img_'+str(i)+'.png', dataset[rand_idx,:,:,:])

    print('\tFinal shape (train):%s',dataset.shape)
    ohe_labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    print('\tFinal shape (train) labels:%s',labels.shape)

    assert np.all(labels==np.argmax(ohe_labels,axis=1))
    return dataset,ohe_labels

memmap_offset=-1


def get_next_memmap_indices(filenames,chunk_size,dataset_size):
    global memmap_offset

    if memmap_offset == -1:
        memmap_offset = 0

    if memmap_offset>=dataset_size:
        print('Resetting memmap offset...\n')
        memmap_offset = 0

    # e.g if dataset_size=10, offset=4 chunk_size=5
    if memmap_offset+chunk_size<=dataset_size-1:
        prev_offset = memmap_offset
        memmap_offset = memmap_offset+chunk_size
        return int(prev_offset),int(memmap_offset)

    # e.g. if dataset_size = 10 offset=7 chunk_size=5
    # data from last => (10-1) - 7
    else:
        prev_offset = memmap_offset
        memmap_offset = dataset_size
        return int(prev_offset),int(memmap_offset)

def reformat_data_imagenet_with_memmap_array(dataset,labels,**params):

    image_size = 224
    num_labels = 100
    num_channels = 3 # rgb

    labels = labels.flatten().astype(np.int32)
    if 'silent' not in params or ('silent' in params and not params['silent']):
        print("Reformatted data ...")
        print('\tlabels shape: ',labels.shape)

    if 'test_images' in params and params['test_images']:
        for i in range(10):
            rand_idx = np.random.randint(0,dataset.shape[0])
            imsave('test_img_'+str(i)+'.png', dataset[rand_idx,:,:,:])

    if 'silent' not in params or ('silent' in params and not params['silent']):
        print('\tFinal shape:%s',dataset.shape)
    ohe_labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    if 'silent' not in params or ('silent' in params and not params['silent']):
        print('\tFinal shape labels:',ohe_labels.shape)

    assert np.all(labels==np.argmax(ohe_labels,axis=1))
    return dataset,ohe_labels

#reformat_data_imagenet_with_memmap(('imagenet_small_train_dataset_2.npy','imagenet_small_train_labels_2.npy'),test_images=True)

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

if __name__ == '__main__':

    train_directory = "/home/tgan4199/imagenet/ILSVRC2015/Data/CLS-LOC/train/"
    valid_directory = "/home/tgan4199/imagenet/ILSVRC2015/Data/CLS-LOC/val/"
    valid_annotation_directory = "/home/tgan4199/imagenet/ILSVRC2015/Annotations/CLS-LOC/val/"
    data_info_directory = "/home/tgan4199/imagenet/ILSVRC2015/ImageSets/"
    data_save_directory = "imagenet_small/"
    gloss_fname = '/home/tgan4199/imagenet/ILSVRC2015/gloss_cls-loc.txt'

    save_imagenet_as_memmaps(train_directory,valid_directory,valid_annotation_directory,gloss_fname,125,125,128,data_save_directory)