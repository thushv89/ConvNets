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
    v1['first'] = tf.Variable([0],name='v1')
    v1['second'] = tf.Variable([1],name='v2')
    tf.initialize_all_variables().run()
    for _ in range(5):
        update_op = update_v()
        _ = session.run(update_op)

    v_val = session.run(v1)
    print(v1['first'].get_shape().as_list())
    print(v1['second'].get_shape().as_list())
    print(v_val)

    session.close()
