import tensorflow as tf
import numpy as np
import logging
import sys

logging_level = logging.INFO
logging_format = '[%(funcName)s] %(message)s'

class OptimizerOpsFacade(object):

    def __init__(self, **params):
        self.basic_hyperparameters = params['basic_hyperparameters']
        self.learning_rate = params['learning_rate']

        self.batch_size = self.basic_hyperparameters['batch_size']
        self.research_parameters = params['research_parameters']
        self.cnn_hyperparameters = params['cnn_hyperparameters']
        self.tf_weights = params['tf_weights']
        self.tf_biases = params['tf_biases']
        self.tf_weight_vels = params['tf_weight_vels']
        self.tf_bias_vels = params['tf_bias_vels']
        self.tf_pool_w_vels = params['tf_pool_w_vels']
        self.tf_pool_b_vels = params['tf_pool_b_vels']

        self.logger = logging.getLogger('optimizer_logger')
        self.logger.setLevel(logging_level)
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter(logging_format))
        console.setLevel(logging_level)
        self.logger.addHandler(console)

    def optimize_func(self, loss, global_step, learning_rate):
        optimize_ops = []
        # Optimizer.
        optimize_ops.append(tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss))

        return optimize_ops, learning_rate

    def optimize_with_momenutm_func(self, loss, global_step, learning_rate):
        vel_update_ops, optimize_ops = [], []

        if self.research_parameters['adapt_structure'] or self.research_parameters['use_custom_momentum_opt']:
            # custom momentum optimizing
            # apply_gradient([g,v]) does the following v -= eta*g
            # eta is learning_rate
            # Since what we need is
            # v(t+1) = mu*v(t) - eta*g
            # theta(t+1) = theta(t) + v(t+1) --- (2)
            # we form (2) in the form v(t+1) = mu*v(t) + eta*g
            # theta(t+1) = theta(t) - v(t+1)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)

            for op in self.tf_weight_vels.keys():
                [(grads_w, w), (grads_b, b)] = optimizer.compute_gradients(loss, [self.weights[op], self.biases[op]])

                # update velocity vector
                vel_update_ops.append(
                    tf.assign(self.tf_weight_vels[op], self.research_parameters['momentum'] * self.tf_weight_vels[op] + grads_w))
                vel_update_ops.append(
                    tf.assign(self.tf_bias_vels[op], self.research_parameters['momentum'] * self.tf_bias_vels[op] + grads_b))

                optimize_ops.append(optimizer.apply_gradients(
                    [(self.tf_weight_vels[op] * learning_rate, self.weights[op]), (self.tf_bias_vels[op] * learning_rate, self.biases[op])]
                ))
        else:
            optimize_ops.append(
                tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                           momentum=self.research_parameters['momentum']).minimize(loss))

        return optimize_ops, vel_update_ops, learning_rate

    def optimize_with_momenutm_func_pool(self, loss, global_step, learning_rate):

        vel_update_ops, optimize_ops = [], []

        if self.research_parameters['adapt_structure'] or self.research_parameters['use_custom_momentum_opt']:
            # custom momentum optimizing
            # apply_gradient([g,v]) does the following v -= eta*g
            # eta is learning_rate
            # Since what we need is
            # v(t+1) = mu*v(t) - eta*g
            # theta(t+1) = theta(t) + v(t+1) --- (2)
            # we form (2) in the form v(t+1) = mu*v(t) + eta*g
            # theta(t+1) = theta(t) - v(t+1)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)

            for op in self.tf_pool_w_vels.keys():
                [(grads_w, w), (grads_b, b)] = optimizer.compute_gradients(loss, [self.weights[op], self.biases[op]])

                # update velocity vector
                vel_update_ops.append(
                    tf.assign(self.tf_pool_w_vels[op], self.research_parameters['momentum'] * self.tf_pool_w_vels[op] + grads_w))
                vel_update_ops.append(
                    tf.assign(self.tf_pool_b_vels[op], self.research_parameters['momentum'] * self.tf_pool_b_vels[op] + grads_b))

                optimize_ops.append(optimizer.apply_gradients(
                    [(self.tf_pool_w_vels[op] * learning_rate, self.weights[op]),
                     (self.tf_pool_b_vels[op] * learning_rate, self.biases[op])]
                ))
        else:
            optimize_ops.append(
                tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                           momentum=self.research_parameters['momentum']).minimize(loss))

        return optimize_ops, vel_update_ops, learning_rate

    def optimize_with_tensor_slice_func(self, loss, filter_indices_to_replace, op, w, b):

        learning_rate = tf.constant(self.learning_rate, dtype=tf.float32, name='learning_rate')
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        grads_wb = optimizer.compute_gradients(loss, [w, b])

        (grads_w, w), (grads_b, b) = grads_wb[0], grads_wb[1]

        curr_weight_shape = self.tf_cnn_hyperparameters[op]['weights'].eval()
        transposed_shape = [curr_weight_shape[3], curr_weight_shape[0], curr_weight_shape[1], curr_weight_shape[2]]

        replace_amnt = filter_indices_to_replace.size

        self.logger.debug('Applying gradients for %s', op)
        self.logger.debug('\tAnd filter IDs: %s', filter_indices_to_replace)

        mask_grads_w = tf.scatter_nd(
            filter_indices_to_replace.reshape(-1, 1),
            tf.ones(shape=[replace_amnt, transposed_shape[1], transposed_shape[2], transposed_shape[3]],
                    dtype=tf.float32),
            shape=transposed_shape
        )

        mask_grads_w = tf.transpose(mask_grads_w, [1, 2, 3, 0])

        mask_grads_b = tf.scatter_nd(
            filter_indices_to_replace.reshape(-1, 1),
            tf.ones_like(filter_indices_to_replace, dtype=tf.float32),
            shape=[curr_weight_shape[3]]
        )

        new_grads_w = grads_w * mask_grads_w
        new_grads_b = grads_b * mask_grads_b

        grad_apply_op = optimizer.apply_gradients([(new_grads_w, w), (new_grads_b, b)])

        return grad_apply_op

    def optimize_with_min_indices_mom(self, loss, opt_ind_dict, indices_size, learning_rate):

        grad_ops, vel_update_ops = [], []

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        for op in self.cnn_ops:

            [(grads_w, w), (grads_b, b)] = optimizer.compute_gradients(loss, [self.weights[op], self.biases[op]])

            vel_update_ops.append(
                tf.assign(self.tf_weight_vels[op], self.research_parameters['momentum'] * self.tf_weight_vels[op] + grads_w))
            vel_update_ops.append(
                tf.assign(self.tf_bias_vels[op], self.research_parameters['momentum'] * self.tf_bias_vels[op] + grads_b))

            if 'conv' in op:

                transposed_shape = [self.tf_cnn_hyperparameters[op]['weights'][3], self.tf_cnn_hyperparameters[op]['weights'][0],
                                    self.tf_cnn_hyperparameters[op]['weights'][1], self.tf_cnn_hyperparameters[op]['weights'][2]]

                self.logger.debug('Applying gradients for %s', op)

                mask_grads_w = tf.scatter_nd(
                    tf.reshape(opt_ind_dict[op], [-1, 1]),
                    tf.ones(shape=[indices_size, transposed_shape[1], transposed_shape[2], transposed_shape[3]],
                            dtype=tf.float32),
                    shape=transposed_shape
                )

                mask_grads_w = tf.transpose(mask_grads_w, [1, 2, 3, 0])

                mask_grads_b = tf.scatter_nd(
                    tf.reshape(opt_ind_dict[op], [-1, 1]),
                    tf.ones_like(opt_ind_dict[op], dtype=tf.float32),
                    shape=[self.tf_cnn_hyperparameters[op]['weights'][3]]
                )

                grads_w = self.tf_weight_vels[op] * learning_rate * mask_grads_w
                grads_b = self.tf_bias_vels[op] * learning_rate * mask_grads_b
                grad_ops.append(optimizer.apply_gradients([(grads_w, w), (grads_b, b)]))

            elif 'fulcon' in op:

                # use dropout (random) for this layer because we can't just train
                # min activations of the last layer (classification)
                self.logger.debug('Applying gradients for %s', op)

                mask_grads_w = tf.scatter_nd(
                    tf.reshape(opt_ind_dict[op], [-1, 1]),
                    tf.ones(shape=[indices_size, self.tf_cnn_hyperparameters[op]['out']],
                            dtype=tf.float32),
                    shape=[self.tf_cnn_hyperparameters[op]['in'], self.tf_cnn_hyperparameters[op]['out']]
                )
                mask_grads_b = tf.scatter_nd(
                    tf.reshape(opt_ind_dict[op], [-1, 1]),
                    tf.ones_like(opt_ind_dict[op], dtype=tf.float32),
                    shape=[self.tf_cnn_hyperparameters[op]['out']]
                )

                grads_w = self.tf_weight_vels[op] * learning_rate * mask_grads_w
                grads_b = self.tf_bias_vels[op] * learning_rate * mask_grads_b

                grad_ops.append(
                    optimizer.apply_gradients([(grads_w, self.weights[op]), (grads_b, self.biases[op])]))

    def optimize_all_affected_with_indices(self, loss, filter_indices_to_replace, op, w, b, indices_size):
        '''
        Any adaptation of a convolutional layer would result in a change in the following layer.
        This optimization optimize the filters/weights responsible in both those layer
        :param loss:
        :param filter_indices_to_replace:
        :param op:
        :param w:
        :param b:
        :param cnn_hyps:
        :param cnn_ops:
        :return:
        '''

        vel_update_ops = []
        grad_ops = []
        grads_w, grads_b = {}, {}
        mask_grads_w, mask_grads_b = {}, {}
        learning_rate = tf.constant(self.learning_rate, dtype=tf.float32, name='learning_rate')
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)

        replace_amnt = indices_size

        if 'conv' in op:
            [(grads_w[op], w), (grads_b[op], b)] = optimizer.compute_gradients(loss, [w, b])

            transposed_shape = [self.tf_cnn_hyperparameters[op]['weights'][3], self.tf_cnn_hyperparameters[op]['weights'][0],
                                self.tf_cnn_hyperparameters[op]['weights'][1], self.tf_cnn_hyperparameters[op]['weights'][2]]

            self.logger.debug('Applying gradients for %s', op)
            self.logger.debug('\tAnd filter IDs: %s', filter_indices_to_replace)

            mask_grads_w[op] = tf.scatter_nd(
                tf.reshape(filter_indices_to_replace, [-1, 1]),
                tf.ones(shape=[replace_amnt, transposed_shape[1], transposed_shape[2], transposed_shape[3]],
                        dtype=tf.float32),
                shape=transposed_shape
            )

            mask_grads_w[op] = tf.transpose(mask_grads_w[op], [1, 2, 3, 0])

            mask_grads_b[op] = tf.scatter_nd(
                tf.reshape(filter_indices_to_replace, [-1, 1]),
                tf.ones_like(filter_indices_to_replace, dtype=tf.float32),
                shape=[self.tf_cnn_hyperparameters[op]['weights'][3]]
            )

            grads_w[op] = grads_w[op] * mask_grads_w[op]
            grads_b[op] = grads_b[op] * mask_grads_b[op]

            vel_update_ops.append(
                tf.assign(self.tf_pool_w_vels[op], self.research_parameters['momentum'] * self.tf_pool_w_vels[op] + grads_w[op]))
            vel_update_ops.append(
                tf.assign(self.tf_pool_b_vels[op], self.research_parameters['momentum'] * self.tf_pool_b_vels[op] + grads_b[op]))

            grad_ops.append(optimizer.apply_gradients(
                [(self.tf_pool_w_vels[op] * learning_rate, w), (self.tf_pool_b_vels[op] * learning_rate, b)]))

        next_op = None
        for tmp_op in self.cnn_ops[self.cnn_ops.index(op) + 1:]:
            if 'conv' in tmp_op or 'fulcon' in tmp_op:
                next_op = tmp_op
                break
        self.logger.debug('Next conv op: %s', next_op)
        [(grads_w[next_op], w)] = optimizer.compute_gradients(loss, [self.weights[next_op]])

        if 'conv' in next_op:

            transposed_shape = [self.tf_cnn_hyperparameters[next_op]['weights'][2],
                                self.tf_cnn_hyperparameters[next_op]['weights'][0],
                                self.tf_cnn_hyperparameters[next_op]['weights'][1],
                                self.tf_cnn_hyperparameters[next_op]['weights'][3]]

            self.logger.debug('Applying gradients for %s', next_op)
            self.logger.debug('\tAnd filter IDs: %s', filter_indices_to_replace)

            mask_grads_w[next_op] = tf.scatter_nd(
                tf.reshape(filter_indices_to_replace, [-1, 1]),
                tf.ones(shape=[replace_amnt, transposed_shape[1], transposed_shape[2], transposed_shape[3]],
                        dtype=tf.float32),
                shape=transposed_shape
            )

            mask_grads_w[next_op] = tf.transpose(mask_grads_w[next_op], [1, 2, 0, 3])
            grads_w[next_op] = grads_w[next_op] * mask_grads_w[next_op]

            vel_update_ops.append(
                tf.assign(self.tf_pool_w_vels[next_op],
                          self.research_parameters['momentum'] * self.tf_pool_w_vels[next_op] + grads_w[next_op]))

            grad_ops.append(optimizer.apply_gradients([(self.tf_pool_w_vels[next_op] * learning_rate, self.weights[next_op])]))

        elif 'fulcon' in next_op:

            self.logger.debug('Applying gradients for %s', next_op)
            self.logger.debug('\tAnd filter IDs: %s', filter_indices_to_replace)

            mask_grads_w[next_op] = tf.scatter_nd(
                tf.reshape(filter_indices_to_replace, [-1, 1]),
                tf.ones(shape=[replace_amnt, self.tf_cnn_hyperparameters[next_op]['out']],
                        dtype=tf.float32),
                shape=[self.tf_cnn_hyperparameters[next_op]['in'], self.tf_cnn_hyperparameters[next_op]['out']]
            )

            grads_w[next_op] = grads_w[next_op] * mask_grads_w[next_op]

            vel_update_ops.append(
                tf.assign(self.tf_pool_w_vels[next_op],
                          self.research_parameters['momentum'] * self.tf_pool_w_vels[next_op] + grads_w[next_op]))

            grad_ops.append(optimizer.apply_gradients([(self.tf_pool_w_vels[next_op] * learning_rate, self.weights[next_op])]))

        return grad_ops, vel_update_ops

