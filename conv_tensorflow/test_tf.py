import tensorflow as tf
import numpy as np


if __name__=='__main__':

    graph = tf.Graph()

    with tf.Session(graph=graph) as session:
        tf_data = tf.constant([[[[1,1],[1,1]],[[3,3],[3,3]]],[[[2,2],[2,2]],[[4,4],[4,4]]]],dtype=tf.float32)
        print("Before padding",tf_data.get_shape().as_list())
        print(tf_data.eval()[0,:,:,0])
        tf_pad_data = tf.pad(tf_data,[[0,0],[1,1],[1,1],[0,0]],mode="CONSTANT")
        print("After padding",tf_data.get_shape().as_list())
        print(tf_pad_data.eval()[0][:][:][0])
        #tf_dialated = tf.nn.atrous_conv2d(tf_data,filters=tf.ones([2,2,2,4]),rate=3,padding="VALID")
        #tf_dialated = tf.nn.conv2d(tf_data,tf.zeros([1,1,2,4]),strides=[1,1,1,1],padding="SAME")
        tf_dialated = tf.nn.dilation2d(tf_pad_data, tf.zeros([2,2,2]), [1,1,1,1], [1,3,3,1], padding="SAME")
        print("After dialated",tf_dialated.get_shape().as_list())
        print(tf_dialated.eval()[0,:,:,0])
