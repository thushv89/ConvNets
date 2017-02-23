__author__ = 'Thushan Ganegedara'

import tensorflow as tf

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
    b  = tf.nn.top_k(tf.reshape(tf.random_normal([3,3]),shape=[-1]))
    print(session.run(b))
    session.close()
