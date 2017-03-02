__author__ = 'Thushan Ganegedara'

import tensorflow as tf
import numpy as np

def update_v():
    ops = []
    global v1
    update_values = {}
    update_values['v2'] = tf.concat(0,[v1['first'],tf.constant([1])])
    ops.append(tf.assign(v1['first'],update_values['v2'],validate_shape=False))

    update_values['v3'] = tf.concat(0,[v1['second'],tf.constant([2])])
    ops.append(tf.assign(v1['second'],update_values['v3'],validate_shape=False))
    return ops


v1 = {}
if __name__ == '__main__':
    graph = tf.Graph()
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.65)
    session = tf.InteractiveSession(graph=graph,
                        config=tf.ConfigProto(allow_soft_placement=True))
    _ = tf.device('/gpu:0')

    ind = tf.placeholder(tf.int32)
    a = tf.placeholder(dtype=tf.float32,shape=(10,10),name='a')
    b = tf.slice(a,[0,0],[10,ind])
    print(session.run(b,feed_dict={a:np.random.random((10,10)),ind:2}))
    print(session.run(b, feed_dict={a: np.random.random((10, 10)), ind: 5}))
    session.close()
