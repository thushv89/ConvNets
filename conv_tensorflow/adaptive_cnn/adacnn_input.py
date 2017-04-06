import tensorflow as tf
import os

def save_svhn10_bin(filenames):
    '''
    :param filenames: Dictionary with keys: train and test and lists of filenames as values
    :return:
    '''



def read_cifar100(filename_queue):
    '''
    A row in bin file: <1 x coarse label><1 x fine label><3072 x pixel>
    :param filename_queue:
    :return:
    '''

def read_cifar10(filename_queue):
    '''
    Read cifar10 binary file (1 byte = 1 uint8)
    A row in bin file: <1 x label><3072 x pixel>
    :param filename_queue:
    :return:
    '''
    label_bytes = 1
    width,height,channels = 32,32,3
    image_bytes = width*height*channels
    record_bytes = label_bytes + image_bytes

    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    # produce next record (key=>scalar tensor,value=>scalar tensor)
    key, value = reader.read(filename_queue)

    record_bytes = tf.decode_raw(value, tf.uint8)

    # get first byte of the record, which is the label and convert to int32
    result_label = tf.cast(tf.strided_slice(record_bytes,[0],[label_bytes]))

    # reshape the 1D vector to depth,height,width
    uint8_img = tf.reshape(tf.strided_slice(record_bytes,[label_bytes],[label_bytes+image_bytes]),[channels,height,width])
    # [depth,height,width] => [height,width,depth]
    uint8_img = tf.transpose(uint8_img,[1,2,0])

    return {'key':key,'image':uint8_img}

def generate_cifar10_batch(image,label, min_queue_examples,batch_size, shuffle, num_threads)

    num_dequeue = 3

    if shuffle:
        # returns a batch of images and batch of labels
        # takes a single example at a time (enqueue_many=False) default
        images, label_batch = tf.train.shuffle_batch(
            [image,label],
            batch_size=batch_size,
            num_threads=num_threads,
            capacity = min_queue_examples + num_dequeue*batch_size,
            min_after_dequeue = min_queue_examples # this property define how many examples to "always" retain in the queue
        )
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=min_queue_examples + num_dequeue * batch_size,
            min_after_dequeue=min_queue_examples
            # this property define how many examples to "always" retain in the queue
        )

    return images, tf.reshape(label_batch,[batch_size])

def distorted_cifar10_inputs(dataset_info, data_dir, batch_size):

    width,height,channels = dataset_info['width'],dataset_info['height'],dataset_info['channels']
    crop_width,crop_height = dataset_info['crop_width'],dataset_info['crop_height']

    filenames = [os.path.join(data_dir,'data_batch_%d.bin'%i) for i in range(1,6)]

    for fname in filenames:
        if not tf.gfile.Exists(fname):
            raise ValueError('Failed to find %s'%fname)

    filename_queue = tf.train.string_input_producer(filenames)

    read_input = read_cifar10(filename_queue)
    rgb_image = tf.cast(read_input['image'],tf.float32)

    distorted_img = tf.random_crop(rgb_image,[crop_height,crop_width,channels])
    distorted_img = tf.image.random_flip_left_right(distorted_img)
    distorted_img = tf.image.random_contrast(distorted_img,lower=0.2,upper=1.8)

    std_image = tf.image.per_image_standardization(distorted_img)

    std_image.set_shape([height,width,channels])
    read_input['label'].set_shape([1])

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int