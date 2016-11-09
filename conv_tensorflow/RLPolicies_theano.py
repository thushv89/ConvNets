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
        self.eta_1 = params['eta_1'] # Q collection starts after this
        self.eta_2 = params['eta_2'] # Evenly perform actions until this
        self.gp_interval = params['gp_interval'] #GP calculating interval (10)
        self.policy_intervel = params['policy_interval']
        self.q = {} #store Q values
        self.gps = {} #store gaussian curves
        self.rl_logger = logging.getLogger('Policy Logger')
        self.rl_logger.setLevel(logging_level)
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter(logging_format))
        console.setLevel(logging_level)
        self.rl_logger.addHandler(console)

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

        # If we haven't completed eta_1 iterations, keep pooling
        if global_step <=self.eta_1:
            # do pool operation
            return

        # Calculate state
        # architecture_similarity is calculated as follows.
        # say you have two architectures
        # C_k3x3x3x32_s1x1, P_k2x2_s1x1, C_k3x3x32x16_s2x2, P_k2x2_s2x2, F_1024, F_10
        # C_k2x2x3x16_s1x1, P_k3x3_s1x1, F_1024, F_512, F_10
        # Break C,P together and F separate. This gives
        # [C_k3x3x3x32_s1x1, P_k2x2_s1x1, C_k3x3x32x16_s2x2, P_k2x2_s2x2]
        # [C_k2x2x3x16_s1x1, P_k3x3_s1x1]
        # [F_1024, F_10]
        # [F_1024, F_512, F_10]
        # Pad 0s C,P vector, so they have same length
        # [C_k3x3x3x32_s1x1, P_k2x2_s1x1, C_k3x3x32x16_s2x2, P_k2x2_s2x2]
        # [C_k2x2x3x16_s1x1, P_k3x3_s1x1, C_k0x0x0x0_s0x0, P_k0x0_s0x0]
        # convert these to vectors following way
        # [0.6*sqrt(3*3*3*32)-0.4(1*1),0.6*(3*3)-0.4*(1*1),...] <- For 1st CP Vector
        # Convert this to a sum using moving avg (higher levels high weight)
        # Currently focus only on CP vector ignore F
        # Pad 0s, to end of F vector, so they have same length
        # [F_1024, F_10, F_0]
        # [F_1024, F_512, F_10]
        # TODO: Convert to Vector

        state = (data['error_t'], data['num_layers'], data['architecture_similarity'])

        err_diff = err_t - err_t_minus_1
        curr_err = err_t

        if global_step > self.eta_1 and global_step <= self.eta_2:
            # TODO: Evenly collect Q values for each action
            # We will have a predefined set of actions
            # A1 = ['add','remove','finetune']
            # if A1 is 'add' > conv [
        # since we have a continuous state space, we need a regression technique to get the
        # Q-value for prev state and action for a discrete state space, this can be done by
        # using a hashtable Q(s,a) -> value


        # self.q is like this there are 3 main items Action.pool, Action.reduce, Action.increment
        # each action has (s, q) value pairs
        # use this (s, q) pairs to predict the q value for a new state
        if global_step%self.gp_interval==0:
            for a, value_dict in self.q.items():

                x, y = zip(*value_dict.items())

                gp = GaussianProcess(theta0=0.1, thetaL=0.001, thetaU=1, nugget=0.1)
                gp.fit(np.array(x), np.array(y))
                self.gps[a] = gp

        if self.prev_state or self.prev_action:

            #reward = - data['error_log'][-1]

            #reward = (1 - err_diff)*(-curr_err)
            if self.undesired_state:
                reward = -10
            else:
                reward = -curr_err

            neuron_penalty = 0

            if data['neuron_balance'][layer_idx] > balance_thresh or data['neuron_balance'][layer_idx] < 1:
                # the coeff was 2.0 before
                # all tests were done with 1.5
                neuron_penalty = .5 * abs(1.5 - data['neuron_balance'][layer_idx])
            reward -= neuron_penalty

            if verbose:
                self.rl_logger.debug('Reward: %10.3f, Penalty: %10.3f', reward, neuron_penalty)            #sample = reward
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
            if verbose:
                self.rl_logger.info("Evenly chose action: %s", action)
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