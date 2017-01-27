import tensorflow as tf
import numpy as np
__author__ = 'Thushan Ganegedara'

if __name__ == '__main__':

    graph  = tf.Graph()
    with tf.Session(graph=graph,config = tf.ConfigProto(allow_soft_placement=True)) as session,tf.device('/gpu:0'):


        b,w,h,d = 2,4,4,2

        # Here I'm defining a 2x4x4x2 matrix
        input = tf.constant(
            [[[[1,5],[0,0],[0,0],[2,6]],
              [[0,0],[0,0],[0,0],[0,0]],
              [[0,0],[0,0],[0,0],[0,0]],
              [[3,7],[0,0],[0,0],[4,8]]],
             [[[3,7],[0,0],[0,0],[4,8]],
              [[0,0],[0,0],[0,0],[0,0]],
              [[0,0],[0,0],[0,0],[0,0]],
              [[1,5],[0,0],[0,0],[2,6]]]
             ],dtype=tf.float32)
        print("Input shape: ",input.get_shape().as_list())
        tf_pooled_input,tf_switches = tf.nn.max_pool_with_argmax(input,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

        # switch shape: b x h/stride x w/stride x d
        print("Switch shape: ",tf_switches.get_shape().as_list())
        # index of flattend output
        pin,switch = session.run([tf_pooled_input,tf_switches])
        # b x 2 x 2 x 1
        print("Switch values: ",switch.reshape(b,-1))

        # we're goin go make a batch_range which is like [0,0,...,0,1,1,...,1,...]
        # so each unique number will have (h/stride * w/stride * d) elements
        # first it will be of shape b x h/stride x w/stride x d
        # but then we reshape it to b x (h/stride * w/stride * d)

        tf_batch_range = tf.reshape(tf.range(b,dtype=tf.int64),[b,1,1,1])
        tf_ones_mask = tf.ones_like(tf_switches)
        tf_multi_batch_range = tf_ones_mask * tf_batch_range

        # here we have indices that looks like b*(h/stride)*(w/stride) x 2
        tf_indices = tf.stack([tf.reshape(tf_multi_batch_range,[-1]),tf.reshape(tf_switches,[-1])],axis=1)
        print(tf_indices.get_shape().as_list())
        # x,y
        updates = tf.to_double(tf.gather_nd(tf.reshape(input,[b,w*h*d]),tf_indices),name='updates')

        print("Update shape: ",updates.get_shape().as_list())
        ref = tf.Variable(tf.zeros_like([b,w*h*d],dtype=tf.float64),dtype=tf.float64,name='ref')

        session.run(tf.variable_initializer([ref]))

        print('Unpooled input shape: ',ref.get_shape().as_list())
        # this is now of size b x (w*h*d)
        #updated_unpool = tf.scatter_nd_update(ref,tf.to_int64(tf_indices),updates,name='updated_unpool')
        updated_unpool = tf.scatter_nd(tf.to_int32(tf_indices),updates,tf.constant([b,w*h*d]),name='updated_unpool')
        updated_unpool = tf.reshape(updated_unpool,[b,h,w,d])
        unpool_input,_,unpool_out = session.run([input,updates,updated_unpool])

        assert np.array_equal(unpool_input,unpool_out)
        print("="*60)
        print('SUCCESSFUL')
        print("="*60)
