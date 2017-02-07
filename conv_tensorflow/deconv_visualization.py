import tensorflow as tf
import numpy as np
import os
import getopt
import sys
import logging
from six.moves import cPickle as pickle
import load_data
from scipy.misc import imsave
from skimage.transform import rotate

__author__ = 'Thushan Ganegedara'

''' #################################################################################
#                                                                                   #
#              Visualizing and Understanding Convolutional Networks                 #
#                                                                                   #
#                      https://arxiv.org/pdf/1311.2901v3.pdf                        #
################################################################################# '''

# We define the things you need to define in order to use this code
# ======================================================================================
# conv_ops : The operations in your convolution network ORDERED as a python list.
#            Each operation should have a unique ID
# conv_depths: Dictionary of output filtermap depth for each convolution operation in conv_ops
# conv_hyperparameters: They hyperparameters for each operation as a dictionary
#                       Convolution: Dictionary with keys: weights, stride and padding
#                       Pooling: Dictionary with keys: type, kernel, stride and padding
#                       Fully Connected: Dictionary with keys: in and out
# weights,biases: Persisted weights and biases after training the network
# validation_dataset: Dataset we use to find the maximum activations for each feature map

# Meaning of notation used in comments
# b - batch size
# w, h and d - width, height and depth

number_of_featuremaps_per_layer = 20
examples_per_featuremap = 10

batch_size = 128

dataset_type = 'imagenet-100' # cifar-10 or imagenet-100

if dataset_type=='cifar-10':
    image_size = 32
    num_labels = 10
    num_channels = 3
    valid_size = 10000
elif dataset_type == 'imagenet-100':
    image_size = 224
    num_labels = 100
    num_channels = 3
    valid_size = 5000

output_dir = 'imagenet-100-inc'

iconv_ops = None
hyparams = None
depth_conv = None

weights,biases = {},{}

valid_dataset_filename = None
valid_dataset = None


def load_valid_dataset(dataset_type):
    if dataset_type=='cifar-10':
        valid_dataset_filename = '..' + os.sep + 'data' + os.sep + 'cifar-10.pickle'

        (train_dataset, train_labels), \
        (valid_dataset, valid_labels), \
        (test_dataset, test_labels) = load_data.reformat_data_cifar10(valid_dataset_filename)

        del train_dataset,train_labels,test_dataset,test_labels

    elif dataset_type=='imagenet-100':
        valid_dataset_fname = 'imagenet_small' + os.sep + 'imagenet_small_valid_dataset'
        valid_label_fname = 'imagenet_small' + os.sep + 'imagenet_small_valid_labels'

        fp1 = np.memmap(valid_dataset_fname, dtype=np.float32, mode='r',
                        offset=np.dtype('float32').itemsize * 0,
                        shape=(valid_size, image_size, image_size, num_channels))
        fp2 = np.memmap(valid_label_fname, dtype=np.int32, mode='r',
                        offset=np.dtype('int32').itemsize * 0, shape=(valid_size, 1))
        valid_dataset = fp1[:, :, :, :]
        valid_labels = fp2[:]

        valid_dataset, valid_labels = load_data.reformat_data_imagenet_with_memmap_array(
            valid_dataset, valid_labels, silent=True
        )

    del valid_labels
    return valid_dataset

def restore_weights_and_biases_and_hyperparameters(
        session,weights_filename,biases_filename):
    global iconv_ops,hyparams,depth_conv
    global weights,biases

    '''with open(output_dir + os.sep + 'ops_hyperparameters.pickle', 'rb') as f:
        param_dict = pickle.load(f)
    iconv_ops = param_dict['op_list']
    hyparams = param_dict['hyperparameters']'''
    iconv_ops = ['conv_1', 'pool_1', 'conv_2', 'pool_2',
                 'conv_3', 'pool_3', 'conv_4',
                 'conv_5', 'pool_5', 'conv_6',
                 'pool_global', 'fulcon_out']

    depth_conv = {'conv_1': 256, 'conv_2': 512, 'conv_3': 1024, 'conv_4': 1024, 'conv_5': 1024, 'conv_6': 2048,
                  'conv_7': 2048}

    final_2d_output = (1, 1)

    conv_1_hyparams = {'weights': [7, 7, num_channels, depth_conv['conv_1']], 'stride': [1, 2, 2, 1], 'padding': 'SAME'}
    conv_2_hyparams = {'weights': [3, 3, depth_conv['conv_1'], depth_conv['conv_2']], 'stride': [1, 1, 1, 1],
                       'padding': 'SAME'}
    conv_3_hyparams = {'weights': [3, 3, depth_conv['conv_2'], depth_conv['conv_3']], 'stride': [1, 1, 1, 1],
                       'padding': 'SAME'}
    conv_4_hyparams = {'weights': [3, 3, depth_conv['conv_3'], depth_conv['conv_4']], 'stride': [1, 1, 1, 1],
                       'padding': 'SAME'}
    conv_5_hyparams = {'weights': [3, 3, depth_conv['conv_4'], depth_conv['conv_5']], 'stride': [1, 1, 1, 1],
                       'padding': 'SAME'}
    conv_6_hyparams = {'weights': [3, 3, depth_conv['conv_5'], depth_conv['conv_6']], 'stride': [1, 1, 1, 1],
                       'padding': 'SAME'}
    conv_7_hyparams = {'weights': [3, 3, depth_conv['conv_6'], depth_conv['conv_7']], 'stride': [1, 1, 1, 1],
                       'padding': 'SAME'}

    pool_1_hyparams = {'type': 'max', 'kernel': [1, 3, 3, 1], 'stride': [1, 2, 2, 1], 'padding': 'SAME'}

    pool_global_hyparams = {'type': 'avg', 'kernel': [1, 7, 7, 1], 'stride': [1, 1, 1, 1], 'padding': 'VALID'}
    out_hyparams = {'in': final_2d_output[0] * final_2d_output[1] * depth_conv['conv_7'], 'out': num_labels}

    hyparams = {
        'conv_1': conv_1_hyparams, 'conv_2': conv_2_hyparams,
        'conv_3': conv_3_hyparams, 'conv_4': conv_4_hyparams,
        'conv_5': conv_5_hyparams, 'conv_6': conv_6_hyparams,
        'conv_7': conv_7_hyparams,
        'pool_1': pool_1_hyparams, 'pool_2': pool_1_hyparams,
        'pool_3': pool_1_hyparams, 'pool_4': pool_1_hyparams,
        'pool_5': pool_1_hyparams,
        'pool_global': pool_global_hyparams,
        'fulcon_out': out_hyparams
    }

    # restoring weights and biases
    logger.info('Restoring weights %s',weights_filename)
    weight_saver = tf.train.import_meta_graph(weights_filename+'.meta')
    weight_saver.restore(sess=session,save_path=weights_filename)

    logger.info('Restoring biases %s', biases_filename)
    biases_saver = tf.train.import_meta_graph(biases_filename + '.meta')
    biases_saver.restore(sess=session,save_path=biases_filename)

    op_variables = [[v.name,v] for v in tf.trainable_variables()]

    # create weights and biases dictionaries
    for op in iconv_ops:
        if 'conv' in op or 'fulcon' in op:
            for vname, v in op_variables:
                if vname.startswith('w_'+op):
                    weights[op] = v
                elif vname.startswith('b_'+op):
                    biases[op] = v

    logger.info('Restored Variables:')
    logger.info('=' * 60)
    logger.info('Weights')
    logger.info(weights)
    logger.info('=' * 60)
    logger.info('Biases')
    logger.info(biases)
    logger.info('=' * 60)

def get_logits(dataset):
    '''
    This method returns the outputs of each layer of the network specified in conv_ops
    :param dataset: Batch of data as a tensorflow placeholder
    :return: Output tensors list [input, outputs of each op in conv_ops in order]
    '''

    global final_2d_output,current_final_2d_output
    outputs = []
    logger.info('Current set of operations: %s'%iconv_ops)
    outputs.append(dataset)
    logger.debug('Received data for X(%s)...'%outputs[-1].get_shape().as_list())

    logger.info('Performing the specified operations ...')

    #need to calculate the output according to the layers we have
    for op in iconv_ops:
        if 'conv' in op:
            logger.debug('\tConvolving (%s) With Weights:%s Stride:%s'%(op,hyparams[op]['weights'],hyparams[op]['stride']))
            logger.debug('\t\tX before convolution:%s'%(outputs[-1].get_shape().as_list()))
            logger.debug('\t\tWeights: %s',weights[op].get_shape().as_list())
            outputs.append(tf.nn.conv2d(outputs[-1], weights[op], hyparams[op]['stride'], padding=hyparams[op]['padding']))
            logger.debug('\t\t Relu with x(%s) and b(%s)'%(outputs[-1].get_shape().as_list(),biases[op].get_shape().as_list()))
            outputs[-1] = tf.nn.relu(outputs[-1] + biases[op])
            logger.debug('\t\tX after %s:%s'%(op,outputs[-1].get_shape().as_list()))

        if 'pool' in op:
            logger.debug('\tPooling (%s) with Kernel:%s Stride:%s'%(op,hyparams[op]['kernel'],hyparams[op]['stride']))

            outputs.append(tf.nn.max_pool(outputs[-1],ksize=hyparams[op]['kernel'],strides=hyparams[op]['stride'],padding=hyparams[op]['padding']))
            logger.debug('\t\tX after %s:%s'%(op,outputs[-1].get_shape().as_list()))

        if op=='loc_res_norm':
            print('\tLocal Response Normalization')
            outputs.append(tf.nn.local_response_normalization(outputs[-1], depth_radius=3, bias=None, alpha=1e-2, beta=0.75))

        if 'fulcon' in op:
            break


    # we need to reshape the output of last subsampling layer to
    # convert 4D output to a 2D input to the hidden layer
    # e.g subsample layer output [batch_size,width,height,depth] -> [batch_size,width*height*depth]
    shape = outputs[-1].get_shape().as_list()
    current_final_2d_output = (shape[1],shape[2])

    rows = shape[0]

    print('Unwrapping last convolution layer %s to %s hidden layer'%(shape,(rows,hyparams['fulcon_out']['in'])))
    reshaped_output = tf.reshape(outputs[-1], [rows,hyparams['fulcon_out']['in']])

    outputs.append(tf.matmul(reshaped_output, weights['fulcon_out']) + biases['fulcon_out'])

    return outputs


def find_max_activations_for_layer(activations):
    # this should return the id of the image that had max activation for all feature maps
    # this matrix of size b x d
    max_activations_images = tf.reduce_max(activations, axis=[1, 2])
    img_id_with_max_activation = tf.argmax(max_activations_images, axis=0)
    return max_activations_images, img_id_with_max_activation


def deconv_featuremap_with_data(layer_id, featuremap_id, tf_selected_dataset, guided_backprop=False):
    global weights, biases

    pool_switches = {}
    activation_masks = {}  # used for guided_backprop
    outputs_fwd = []
    logger.info('Current set of operations: %s' % iconv_ops)
    outputs_fwd.append(tf_selected_dataset)
    logger.debug('Received data for X(%s)...' % outputs_fwd[-1].get_shape().as_list())

    logger.info('Performing the forward pass ...')

    # need to calculate the output according to the layers we have
    for op in iconv_ops:
        if 'conv' in op:
            logger.debug(
                '\tConvolving (%s) With Weights:%s Stride:%s' % (op, hyparams[op]['weights'], hyparams[op]['stride']))
            logger.debug('\t\tX before convolution:%s' % (outputs_fwd[-1].get_shape().as_list()))
            logger.debug('\t\tWeights: %s', weights[op].get_shape().as_list())
            outputs_fwd.append(
                tf.nn.conv2d(outputs_fwd[-1], weights[op], hyparams[op]['stride'], padding=hyparams[op]['padding']))
            logger.debug('\t\t Relu with x(%s) and b(%s)' % (
            outputs_fwd[-1].get_shape().as_list(), biases[op].get_shape().as_list()))
            outputs_fwd[-1] = tf.nn.relu(outputs_fwd[-1] + biases[op])
            logger.debug('\t\tX after %s:%s' % (op, outputs_fwd[-1].get_shape().as_list()))

            if guided_backprop:
                activation_masks[op] = tf.not_equal(outputs_fwd[-1], tf.constant(0, dtype=tf.float32))
                assert activation_masks[op].get_shape().as_list() == outputs_fwd[-1].get_shape().as_list()

            if op == layer_id:
                break

        if 'pool' in op:
            logger.debug(
                '\tPooling (%s) with Kernel:%s Stride:%s' % (op, hyparams[op]['kernel'], hyparams[op]['stride']))
            pool_out, switch = tf.nn.max_pool_with_argmax(outputs_fwd[-1], ksize=hyparams[op]['kernel'],
                                                          strides=hyparams[op]['stride'],
                                                          padding=hyparams[op]['padding'])
            outputs_fwd.append(pool_out)
            pool_switches[op] = switch
            logger.debug('\t\tX after %s:%s' % (op, outputs_fwd[-1].get_shape().as_list()))

        if 'fulcon' in op:
            break

    logger.info('Performing the backward pass ...\n')
    logger.debug('\tInput Size (Non-Zeroed): %s', str(outputs_fwd[-1].get_shape().as_list()))
    # b - batch size, h - height, w - width, d - number of filters
    b, h, w, d = outputs_fwd[-1].get_shape().as_list()

    # outputs[-1] will have the required activation
    # will be of size b x 1 x 1

    # we create a tensor from the activations of the layer which only has non-zeros
    # for the selected feature map (layer_activations_2)
    layer_activations = tf.transpose(outputs_fwd[-1], [3, 0, 1, 2])
    layer_indices = tf.constant([[featuremap_id]])
    layer_updates = tf.expand_dims(layer_activations[featuremap_id, :, :, :], 0)
    layer_activations_2 = tf.scatter_nd(layer_indices, layer_updates,
                                        tf.constant(layer_activations.get_shape().as_list()))
    layer_activations_2 = tf.transpose(layer_activations_2, [1, 2, 3, 0])
    assert outputs_fwd[-1].get_shape().as_list() == layer_activations_2.get_shape().as_list()

    # single out only the maximally activated neuron and set the zeros
    argmax_indices = tf.argmax(tf.reshape(layer_activations_2, [b, h * w * d]), axis=1)
    batch_range = tf.range(b, dtype=tf.int64)
    nonzero_indices = tf.stack([batch_range, argmax_indices], axis=1)
    updates = tf.gather_nd(tf.reshape(layer_activations_2, [b, h * w * d]), nonzero_indices)
    logger.debug('\tNon-zero indices shape: %s', nonzero_indices.get_shape().as_list())
    logger.debug('\tNon-zero updates shape: %s', updates.get_shape().as_list())
    # Creating the new activations (of size: b x w x h x d)
    # with only the highest activation of given feature map ID non-zero and rest set to zero
    zeroed_activations = tf.scatter_nd(nonzero_indices, updates, tf.constant([b, h * w * d], dtype=tf.int64))
    zeroed_activations = tf.reshape(zeroed_activations, [b, h, w, d])

    outputs_bckwd = [zeroed_activations]
    op_index = iconv_ops.index(layer_id)

    logger.debug('Input Size (Zeroed): %s', str(outputs_bckwd[-1].get_shape().as_list()))

    for op in reversed(iconv_ops[:op_index + 1]):
        if 'conv' in op:
            # Deconvolution

            logger.debug('\tDeConvolving (%s) With Weights:%s Stride:%s' % (
            op, weights[op].get_shape().as_list(), hyparams[op]['stride']))
            logger.debug('\t\tX before deconvolution:%s' % (outputs_bckwd[-1].get_shape().as_list()))
            logger.debug('\t\tWeights: %s', weights[op].get_shape().as_list())

            output_shape = outputs_bckwd[-1].get_shape().as_list()
            output_shape[1] *= hyparams[op]['stride'][1]
            output_shape[2] *= hyparams[op]['stride'][2]
            output_shape[3] = hyparams[op]['weights'][2]
            logger.debug('\t\tExpected output shape: %s', output_shape)
            outputs_bckwd.append(
                tf.nn.conv2d_transpose(outputs_bckwd[-1], filter=weights[op], strides=hyparams[op]['stride'],
                                       padding=hyparams[op]['padding'], output_shape=tf.constant(output_shape))
            )

            logger.debug('\t\tX after %s:%s' % (op, outputs_bckwd[-1].get_shape().as_list()))

        if 'pool' in op:

            # find previous conv_op
            previous_conv_op = None
            for before_op in reversed(iconv_ops[:iconv_ops.index(op) + 1]):
                if 'conv' in before_op:
                    previous_conv_op = before_op
                    break

            logger.debug('Detected previous conv op %s', previous_conv_op)

            # Unpooling operation and Rectification
            logger.debug(
                '\tUnPooling (%s) with Kernel:%s Stride:%s' % (op, hyparams[op]['kernel'], hyparams[op]['stride']))
            logger.debug('\t\tInput shape: %s', outputs_bckwd[-1].get_shape().as_list())

            output_shape = outputs_bckwd[-1].get_shape().as_list()
            output_shape[1] *= hyparams[op]['stride'][1]
            output_shape[2] *= hyparams[op]['stride'][2]

            logger.debug('\t\tExpected output shape: %s', output_shape)
            # Unpooling

            # Switch variable returns an array of size b x h x w x d. But only provide flattened indices
            # Meaning that if you have an output of size 4x4 it will flatten it to a 16 element long array

            # we're goin go make a batch_range which is like [0,0,...,0,1,1,...,1,...]
            # so each unique number will have (h/stride * w/stride * d) elements
            # first it will be of shape b x h/stride x w/stride x d
            # but then we reshape it to b x (h/stride * w/stride * d)
            tf_switches = pool_switches[op]
            tf_batch_range = tf.reshape(tf.range(b, dtype=tf.int64), [b, 1, 1, 1])
            tf_ones_mask = tf.ones_like(tf_switches)
            tf_multi_batch_range = tf_ones_mask * tf_batch_range

            # here we have indices that looks like b*(h/stride)*(w/stride) x 2
            tf_indices = tf.stack([tf.reshape(tf_multi_batch_range, [-1]), tf.reshape(tf_switches, [-1])], axis=1)

            updates = tf.reshape(outputs_bckwd[-1], [-1])

            ref = tf.Variable(tf.zeros([b, output_shape[1] * output_shape[2] * output_shape[3]], dtype=tf.float32),
                              dtype=tf.float32, name='ref_' + op)

            session.run(tf.variables_initializer([ref]))

            updated_unpool = tf.scatter_nd(tf.to_int32(tf_indices), updates,
                                           tf.constant([b, output_shape[1] * output_shape[2] * output_shape[3]]),
                                           name='updated_unpool_' + op)

            outputs_bckwd.append(tf.reshape(updated_unpool, [b, output_shape[1], output_shape[2], output_shape[3]]))

            outputs_bckwd[-1] = tf.nn.relu(outputs_bckwd[-1])

            if guided_backprop and previous_conv_op is not None:
                logger.info('Output-bckwd: %s', outputs_bckwd[-1].get_shape().as_list())
                logger.info('Activation mask %s', activation_masks[previous_conv_op].get_shape().as_list())
                assert outputs_bckwd[-1].get_shape().as_list() == activation_masks[
                    previous_conv_op].get_shape().as_list()
                outputs_bckwd[-1] = outputs_bckwd[-1] * tf.to_float(activation_masks[previous_conv_op])

            logger.debug('\t\tX after %s:%s' % (op, outputs_bckwd[-1].get_shape().as_list()))

    return outputs_fwd, outputs_bckwd

def visualize_with_deconv(session,layer_id,all_x,guided_backprop=False):
    global logger, weights, biases
    '''
    DECONV works the following way.
    # Pick a layer
    # Pick a subset of feature maps or all the feature maps in the layer (if small)
    # For each feature map
    #     Pick the n images that maximally activate that feature map
    #     For each image
    #          Do back propagation for the given activations from that layer until the pixel input layer
    '''

    selected_featuremap_ids = list(np.random.randint(0,depth_conv[layer_id],(number_of_featuremaps_per_layer,)))

    images_for_featuremap = {} # a dictionary with featuremmap_id : an ndarray with size num_of_images_per_featuremap x image_size
    mean_activations_for_featuremap = {} # this is a dictionary containing featuremap_id : [list of mean activations for each image in order]

    layer_index = iconv_ops.index(layer_id)
    tf_deconv_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_selected_image = tf.placeholder(tf.float32, shape=(examples_per_featuremap, image_size, image_size, num_channels))

    tf_activations = get_logits(tf_deconv_dataset)
    # layer_index+1 because we have input as 0th index
    tf_max_activations, tf_img_ids = find_max_activations_for_layer(tf_activations[layer_index+1])

    # activations for batch of data for a layer of size w(width),h(height),d(depth) will be = b x w x h x d
    # reduce this to b x d by using reduce sum
    image_shape = all_x.shape[1:]
    for batch_id in range(all_x.shape[0]//batch_size):

        batch_data = all_x[batch_id*batch_size:(batch_id+1)*batch_size, :, :, :]
        feed_dict = {tf_deconv_dataset:batch_data}

        # max_activations b x d, img_ids_for_max will be 1 x d
        max_activations,img_ids_for_max = session.run([tf_max_activations,tf_img_ids],feed_dict=feed_dict)
        if batch_id==0:
            logger.debug('Max activations for batch %d size: %s',batch_id,str(max_activations.shape))
            logger.debug('Image ID  for batch %d size: %s',batch_id,str(img_ids_for_max.shape))

        for d_i in range(img_ids_for_max.shape[0]):

            # we only run this for selected set of featurmaps
            if d_i not in selected_featuremap_ids:
                continue

            img_id = np.asscalar(img_ids_for_max[d_i])

            if d_i==selected_featuremap_ids[1]:
                logger.debug('Found %d image id for depth %d',img_id,d_i)

            if d_i not in mean_activations_for_featuremap:
                mean_activations_for_featuremap[d_i] = [np.asscalar(max_activations[img_id,d_i])]
                images_for_featuremap[d_i] = np.reshape(batch_data[img_id,:,:,:],(1,image_shape[0],image_shape[1],image_shape[2]))
            else:
                if len(mean_activations_for_featuremap[d_i])>= examples_per_featuremap:
                    # delete the minimum

                    min_idx = np.asscalar(np.argmin(np.asarray(mean_activations_for_featuremap[d_i])))
                    if d_i==selected_featuremap_ids[1]:
                        logger.debug('Mean activations: %s',mean_activations_for_featuremap[d_i])
                        logger.debug('\tFound minimum activation with %.2f at index %d',np.min(mean_activations_for_featuremap[d_i]),min_idx)

                    del mean_activations_for_featuremap[d_i][min_idx]
                    images_for_featuremap[d_i] = np.delete(images_for_featuremap[d_i],[min_idx],axis=0)
                reshp_img = np.reshape(batch_data[img_id,:,:,:],(1,image_shape[0],image_shape[1],image_shape[2]))
                images_for_featuremap[d_i] = np.append(images_for_featuremap[d_i],reshp_img,axis=0)
                mean_activations_for_featuremap[d_i].append(np.asscalar(max_activations[img_id,d_i]))

    logger.info('Size of image set for feature map: %s',str(len(mean_activations_for_featuremap[selected_featuremap_ids[0]])))

    all_deconv_outputs = []
    all_images = []
    for d_i in selected_featuremap_ids:
        tf_fwd_outputs, tf_bck_outputs = deconv_featuremap_with_data(layer_id,d_i,tf_selected_image,guided_backprop)
        fwd_outputs, deconv_outputs = session.run([tf_fwd_outputs,tf_bck_outputs],
                                                  feed_dict={tf_selected_image:images_for_featuremap[d_i]})
        all_deconv_outputs.append(deconv_outputs[-1])
        all_images.append(images_for_featuremap[d_i])
    return all_deconv_outputs, all_images

backprop_feature_dir = None

if __name__=='__main__':

    try:
        opts,args = getopt.getopt(
            sys.argv[1:],"",["backprop_dir="])
    except getopt.GetoptError as err:
        print('<filename>.py --backprop_dir=')

    if len(opts)!=0:
        for opt,arg in opts:
            if opt == '--backprop_dir':
                backprop_feature_dir = arg

    logger = logging.getLogger('deconv_logger')
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter('[%(funcName)s] %(message)s'))
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    graph = tf.Graph()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True,gpu_options=gpu_options)) as session:

        restore_weights_and_biases_and_hyperparameters(
            session,output_dir + os.sep + 'cnn-weights-7',output_dir + os.sep + 'cnn-biases-7')
        layer_ids = [op for op in iconv_ops if 'conv' in op and op != 'conv_1']

        if not os.path.exists(output_dir + os.sep + backprop_feature_dir):
            os.makedirs(output_dir + os.sep + backprop_feature_dir)
            backprop_feature_dir = output_dir + os.sep + backprop_feature_dir

        rotate_required = True if dataset_type == 'cifar-10' else False  # cifar-10 training set has rotated images so we rotate them to original orientation

        for lid in layer_ids:
            all_deconvs, all_images = visualize_with_deconv(session, lid, valid_dataset, True)
            d_i = 0
            for deconv_di, images_di in zip(all_deconvs, all_images):
                local_dir = backprop_feature_dir + os.sep + lid + '_' + str(d_i)
                if not os.path.exists(local_dir):
                    os.mkdir(local_dir)

                # saving deconv images
                for img_i in range(deconv_di.shape[0]):
                    if rotate_required:
                        local_img = (deconv_di[img_i, :, :, :] - np.min(deconv_di[img_i, :, :, :])).astype('uint16')
                        local_img = rotate(local_img, 270)
                    else:
                        local_img = deconv_di[img_i, :, :, :]
                    imsave(local_dir + os.sep + 'deconv_' + lid + '_' + str(d_i) + '_' + str(img_i) + '.png', local_img)

                # saving original images
                for img_i in range(images_di.shape[0]):
                    if rotate_required:
                        images_di[img_i, :, :, :] = images_di[img_i, :, :, :] - np.min(images_di[img_i, :, :, :])
                        images_di[img_i, :, :, :] = images_di[img_i, :, :, :] * 255.0 / np.max(
                            images_di[img_i, :, :, :])
                        local_img = images_di[img_i, :, :, :].astype('uint16')

                        local_img = rotate(local_img, 270)
                    else:
                        local_img = images_di[img_i, :, :, :]
                    imsave(local_dir + os.sep + 'image_' + lid + '_' + str(d_i) + '_' + str(img_i) + '.png', local_img)

                d_i += 1
