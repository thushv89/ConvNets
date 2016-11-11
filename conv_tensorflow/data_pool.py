
import tensorflow as tf
import numpy as np

class Pool(object):
    '''
    This class is used to maintain a pool of data which will be used
    from time to time for finetuning the network
    '''
    def __init__(self,**params):

        self.size = params['size']
        self.position = 0
        self.dataset = None
        self.labels = None
        self.batch_size = params['batch_size']
        self.filled_size = 0

    def add_all(self,data,labels):
        '''
        Add all the data to the pool
        :param data:
        :param labels:
        :return:
        '''
        raise NotImplementedError

    def add_hard_examples(self,data,labels,loss,fraction):
        '''
        This method will add only the hard examples to the pool
        Hard examples are the ones that were not correctly classified
        :param data: full data batch
        :param labels: full label batch
        :param loss: loss vector for the batch
        :param fraction: fraction of data we want
        :return: None
        '''

        _,hard_indices = tf.nn.top_k(loss,k=int(fraction*self.batch_size),sorted=False,name='top_hard')

        add_size = hard_indices.get_shape().as_list()[0]

        if self.dataset is None and self.labels is None:
            self.dataset = tf.constant(data)
            self.labels = tf.constant(labels)

        else:
            # if position has more space for all the hard_examples
            if self.position <= self.size - add_size:
                self.dataset[self.position:self.position+add_size,:,:,:] = data[hard_indices,:,:,:]
                self.labels[self.position:self.position+add_size,:] = labels[hard_indices,:]
            else:
                overflow = (self.position + add_size) % self.size
                end_chunk_size = self.size - self.position

                # Adding till the end
                self.dataset[self.position:,:,:,:] = data[hard_indices[self.position:end_chunk_size],:,:,:]
                self.labels[self.position:,:] = labels[hard_indices[self.position:end_chunk_size],:]

                # Starting from the beginning for the remaining
                self.dataset[:overflow,:,:,:] = data[hard_indices[end_chunk_size:],:,:,:]
                self.labels[:overflow,:] = labels[hard_indices[end_chunk_size:],:]

        self.position = (self.position+add_size)%self.size

    def get_position(self):
        return self.position

    def get_size(self):
        if self.dataset is None:
            return 0
        else:
            return self.dataset.get_shape().as_list()[0]