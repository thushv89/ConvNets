
import tensorflow as tf
import numpy as np

class Pool(object):
    '''
    This class is used to maintain a pool of data which will be used
    from time to time for finetuning the network
    '''
    def __init__(self,**params):

        self.assert_test = params['assert_test']
        self.size = params['size']
        self.position = 0
        self.dataset = np.empty((self.size,params['image_size'],params['image_size'],params['num_channels']),dtype=np.float32)
        self.labels = np.empty((self.size,params['num_labels']),dtype=np.float32)
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

        _,tf_hard_indices = tf.nn.top_k(loss,k=int(fraction*self.batch_size),sorted=False,name='top_hard')
        add_size = tf_hard_indices.get_shape().as_list()[0]
        hard_indices = tf_hard_indices.eval()

        # if position has more space for all the hard_examples
        if self.position + add_size <= self.size - 1:
            self.dataset[self.position:self.position+add_size,:,:,:] = data[hard_indices,:,:,:]
            self.labels[self.position:self.position+add_size,:] = labels[hard_indices,:]
        else:
            overflow = (self.position + add_size) % (self.size-1)
            end_chunk_size = self.size - (self.position + 1)

            assert overflow + end_chunk_size == add_size
            # Adding till the end
            self.dataset[self.position:,:,:,:] = data[hard_indices[:end_chunk_size+1],:,:,:]
            self.labels[self.position:,:] = labels[hard_indices[:end_chunk_size+1],:]

            # Starting from the beginning for the remaining
            self.dataset[:overflow,:,:,:] = data[hard_indices[end_chunk_size:],:,:,:]
            self.labels[:overflow,:] = labels[hard_indices[end_chunk_size:],:]


        if self.assert_test:
            assert np.all(self.dataset[self.position,:,:,:].flatten()== data[hard_indices[0],:,:,:].flatten())

        if self.filled_size != self.size:
            self.filled_size = min(self.position+add_size+1,self.size)

        self.position = (self.position+add_size)%self.size

    def get_position(self):
        return self.position

    def get_size(self):
        return self.filled_size

    def get_pool_data(self):
        return {'pool_dataset':self.dataset,'pool_labels':self.labels}