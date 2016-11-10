__author__ = 'Thushan Ganegedara'

from enum import IntEnum
from collections import defaultdict
from sklearn.gaussian_process import GaussianProcess
import numpy as np
import json
import random
import logging
import sys

logging_level = logging.INFO
logging_format = '[%(name)s] [%(funcName)s] %(message)s'


class ContinuousState(object):

    def __init__(self, **params):
        self.params = params
        self.learning_rate = params['learning_rate']
        self.discount_rate = params['discount_rate']

        self.q = {} #store Q values
        self.gps = {} #store gaussian curves
        self.rl_logger = logging.getLogger('Policy Logger')
        self.rl_logger.setLevel(logging_level)
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter(logging_format))
        console.setLevel(logging_level)
        self.rl_logger.addHandler(console)

        self.actions = [
            'add,C_K5x5x16_S3x3','add,C_K5x5x64_S3x3','add,C_K5x5x16_S1x1','add,C_K5x5x64_S1x1',
            'add,C_K3x3x16_S3x3','add,C_K3x3x64_S3x3','add,C_K3x3x16_S1x1','add,C_K3x3x64_S1x1',
            'add,C_K1x1x16_S3x3','add,C_K1x1x64_S3x3','add,C_K1x1x16_S1x1','add,C_K1x1x64_S1x1',
            'add_P_K5x5_S3x3','add_P_K5x5_S1x1','add_P_K2x2_S2x2','add_P_K2x2_S1x1','finetune','remove'
        ]

    def restore_policy(self,**restore_data):
        # use this to restore from saved data
        self.prev_state = restore_data['prev_state']
        self.prev_action = restore_data['prev_action']
        self.q = restore_data['q']
        self.gps = restore_data['gps']

    def update_policy(self, global_step, data):

        # Errors (current and previous) used to calculate reward
        # err_t should be calculated on something the CNN hasn't seen yet
        # err_t-1 should be calculated on something the CNN has seen (e.g mean of last 10 batches)
        err_t = data['error_t']
        err_t_minus_1 = data['error_t-1']
        time_cost = data['avg_time_cost']
        param_cost = data['param_cost']




        err_diff = err_t - err_t_minus_1
        curr_err = err_t

        if global_step > self.eta_1 and global_step <= self.eta_2:
            # TODO: Evenly collect Q values for each action
            # We will have a predefined set of actions
            # A1 = [{'add'},'remove','finetune']

        # since we have a continuous state space, we need a regression technique to get the
        # Q-value for prev state and action for a discrete state space, this can be done by
        # using a hashtable Q(s,a) -> value


        # self.q is like this there are 3 main items Action.pool, Action.reduce, Action.increment
        # each action has (s, q) value pairs
        # use this (s, q) pairs to predict the q value for a new state
        if global_step % self.gp_interval==0:
            for a, value_dict in self.q.items():

                x, y = zip(*value_dict.items())

                gp = GaussianProcess(theta0=0.1, thetaL=0.001, thetaU=1, nugget=0.1)
                gp.fit(np.array(x), np.array(y))
                self.gps[a] = gp

        if self.prev_state or self.prev_action:

            reward = -(curr_err + 0.01*time_cost + 0.001*param_cost)

            #sample = reward + self.discount_rate * max(self.q[state, a] for a in self.actions)
            # len(gps) == 0 in the first time move() is called
            if len(self.gps) == 0:
                sample = reward
            else:
                sample = reward + self.discount_rate * max((np.asscalar(gp.predict([self.prev_state])[0])) for gp in self.gps.values())

            if self.prev_state in self.q[self.prev_action]:
                self.q[self.prev_action][self.prev_state] = (1 - self.learning_rate) * self.q[self.prev_action][self.prev_state] + self.learning_rate * sample
            else:
                self.q[self.prev_action][self.prev_state] = sample

        #action = list(self.Action)[i % len(self.Action)]
        #the first time move() is called
        #Evenly chosing actions is important because with this algorithm will learn q values for all actions
        if len(self.gps) == 0 or i <= even_chose_thresh:
            action = list(Action)[i % len(Action)]
        else:
            # determine best action by sampling the GPs
            if random.random() <= 0.1:
                action = list(Action)[i % len(Action)]
                self.rl_logger.info("Explore action: %s", action)
            else:
                action = max((np.asscalar(gp.predict(state)[0]), action) for action, gp in self.gps.items())[1]
                self.rl_logger.info("Chose with Q: %s", action)

            for a, gp in self.gps.items():
                self.rl_logger.debug("Approximated Q values for (above State,%s) pair: %10.3f",a, np.asscalar(gp.predict(state)[0]))

        self.undesired_state = False
        #to_move = (data['initial_size'] * 0.1) / (data['initial_size'] * data['neuron_balance'])
        #to_move = 0.25 * np.exp(-(data['neuron_balance']-1.)**2/2.) * (1. + err_diff) * (1. + curr_err)
        # newer to_move eqn

        #to_move = (50.0/data['initial_size'][layer_idx])*np.exp(-(data['neuron_balance'][layer_idx]-(.5*balance_thresh))**2/balance_thresh) * np.abs(err_diff)

        # skewed normal distribution
        #mu,sigma,alpha_skew = 1.0,5.0,4.0
        #nb = (data['neuron_balance'][layer_idx] - mu)/sigma**2
        #nb_alpha = alpha_skew*nb/2**0.5
        #to_move = (50.0/data['initial_size'][layer_idx]) * (1/(2*np.pi)**0.5) * np.exp(-nb**2/2.0) *\
        #          (1 + np.sign(nb_alpha) * np.sqrt(1-np.exp(-nb_alpha**2*((4/np.pi)+0.14*nb_alpha**2)/(1+0.14*nb_alpha**2)))) * \
        #          (1+np.abs(err_diff))
        #to_move = (100.0/data['initial_size'][layer_idx]) * np.abs(err_diff)


        to_move = 5.0/data['initial_size'][layer_idx]
        self.rl_logger.info('Nodes to add/remove: %10.3f', to_move*data['initial_size'][layer_idx])

        if to_move<0.1/data['initial_size'][layer_idx]:
            self.rl_logger.debug('Nodes to add/remove negligable')
            if pooling:
                funcs['pool'](1)
                action = Action.pool

        else:
            if data['neuron_balance'][layer_idx] < 10.0/data['initial_size'][layer_idx] and action == Action.reduce:
                self.rl_logger.debug('Undesired State. Neuron balance too low (%10.3f)... Executing pool operation',data['neuron_balance'][layer_idx])
                funcs['pool'](1)
                self.undesired_state = True


            elif pooling and action == Action.pool:
                funcs['pool_finetune'](1)
                self.add_idx,self.rem_idx = [],[]
            elif action == Action.reduce:
                # method signature: amount, to_merge, to_inc
                # introducing division by two to reduce the impact
                to_move /= 2.
                self.add_idx,self.rem_idx = funcs['merge_increment_pool'](1, to_move, 0,layer_idx)
            elif action == Action.increment:
                self.add_idx,self.rem_idx = funcs['merge_increment_pool'](1, 0, to_move,layer_idx)

        self.prev_action = action
        self.prev_state = state
        self.prev_change = int(to_move*data['initial_size'][layer_idx])
        self.rl_logger.info('\n')
    def get_current_action_change(self):
        return str(self.prev_action),self.prev_change

    def get_policy_info(self):
        return self.q,self.gps,self.prev_state,self.prev_action

    def end(self):
        return [{'name': 'q_state.json', 'json': json.dumps({str(k):{str(tup): value for tup, value in v.items()} for k,v in self.q.items()})}]