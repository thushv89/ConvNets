__author__ = 'Thushan Ganegedara'

from enum import IntEnum
from collections import defaultdict
from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np
import json
import random
import logging
import sys

logging_level = logging.DEBUG
logging_format = '[%(name)s] [%(funcName)s] %(message)s'


class ContinuousState(object):

    def __init__(self, **params):

        self.learning_rate = params['learning_rate']
        self.discount_rate = params['discount_rate']
        self.eta_1 = params['eta_1'] # Q collection starts after this
        self.eta_2 = params['eta_2'] # Evenly perform actions until this
        self.gp_interval = params['gp_interval'] #GP calculating interval (10)
        self.policy_intervel = params['policy_interval']
        self.q = {} #store Q values
        self.gps = {} #store gaussian curves
        self.epsilon = params['epsilon']
        self.rl_logger = logging.getLogger('Policy Logger')
        self.rl_logger.setLevel(logging_level)
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter(logging_format))
        console.setLevel(logging_level)
        self.rl_logger.addHandler(console)

        self.local_time_stamp = 0
        self.actions = None

        self.prev_action = None
        self.prev_state = None

    def update_action_space(self,actions):

        self.actions = []
        for a in actions:
            a_op = a.split(',')[1]
            self.actions.append(a)
            self.actions.append('remove,'+a_op)

        self.actions.append('finetune')

    def restore_policy(self,**restore_data):
        # use this to restore from saved data
        self.prev_state = restore_data['prev_state']
        self.prev_action = restore_data['prev_action']
        self.q = restore_data['q']
        self.gps = restore_data['gps']

    def update_policy(self, global_time_stamp, data,success=True):

        if not success and self.prev_action is not None:
            self.rl_logger.warn('\n')
            self.rl_logger.warn("Previous action %s wasn't successful ...\n"%self.prev_action)

        self.rl_logger.info('\n============== Policy Update for %d (Global: %d) ==============='%(self.local_time_stamp,global_time_stamp))
        # Errors (current and previous) used to calculate reward
        # err_t should be calculated on something the CNN hasn't seen yet
        # err_t-1 should be calculated on something the CNN has seen (e.g mean of last 10 batches)
        err_t = data['error_t']
        time_cost = data['time_cost']
        param_cost = data['param_cost']
        stride_cost = data['stride_cost']
        success_cost = 0 if success else -100

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

        state = (data['error_t'], data['num_layers'],data['param_cost'])

        curr_err = err_t

        if self.local_time_stamp <= self.eta_1:
            # TODO: Evenly collect Q values for each action
            # We will have a predefined set of actions
            action = self.actions[-1]
            self.rl_logger.info('\tNot enough data running action:%s'%action)
        # since we have a continuous state space, we need a regression technique to get the
        # Q-value for prev state and action for a discrete state space, this can be done by
        # using a hashtable Q(s,a) -> value


        # self.q is like this there are 3 main items Action.pool, Action.reduce, Action.increment
        # each action has (s, q) value pairs
        # use this (s, q) pairs to predict the q value for a new state
        elif self.local_time_stamp % self.gp_interval==0:
            for a, value_dict in self.q.items():
                if len(value_dict) < 3:
                    break
                x, y = zip(*value_dict.items())

                gp = GaussianProcessRegressor()
                if np.asarray(x).shape[0]==1:
                    gp.fit(np.asarray(x).reshape(1,-1),np.asarray(y))
                else:
                    gp.fit(np.asarray(x), np.asarray(y))

                self.gps[a] = gp

        if self.prev_state or self.prev_action:

            reward = -(curr_err + 0.01*time_cost + 1e-5*param_cost +
                       2**(stride_cost//3) + 2**(data['num_layers']//2)) + success_cost
            self.rl_logger.info("Reward for action: %s: %.3f"%(self.prev_action,reward))
            # sample = reward + self.discount_rate * max(self.q[state, a] for a in self.actions)
            # len(gps) == 0 in the first time move() is called
            if self.prev_action not in self.gps or len(self.q[self.prev_action]) < 3:
                self.rl_logger.debug('\tNot enough data to predict with a GP. Using the reward as the sample ...')
                sample = reward
            else:
                self.rl_logger.debug('\tPredicting with a GP. Using the gp prediction + reward as the sample ...')
                sample = reward + self.discount_rate * max((np.asscalar(gp.predict([np.asarray(self.prev_state).reshape((1,-1))])[0])) for gp in self.gps.values())

            self.rl_logger.info('\tUpdating the Q values ...')

            if self.prev_action not in self.q:
                self.rl_logger.debug('\tPrevious action not found in Q values. Creating a new entry ...')
                self.q[self.prev_action]={self.prev_state: sample}
            else:
                if self.prev_state in self.q[self.prev_action]:
                    self.rl_logger.debug('\tPrevious state found in Q values. Updating it ...')
                    self.q[self.prev_action][self.prev_state] = (1 - self.learning_rate) * self.q[self.prev_action][self.prev_state] + \
                                                                self.learning_rate * sample
                else:
                    self.rl_logger.debug('\tPrevious state not found in Q values. Creating a new entry ...')
                    self.q[self.prev_action][self.prev_state] = sample


            if len(self.gps) == 0 or self.local_time_stamp <= self.eta_2:
                action = self.actions[self.local_time_stamp % len(self.actions)]
                self.rl_logger.info("\tEvenly chose action: %s"%action)
            else:
                # determine best action by sampling the GPs
                if random.random() <= self.epsilon:
                    action = self.actions[self.local_time_stamp % len(self.actions)]
                    self.rl_logger.info("\tExplore action: %s", action)
                else:
                    action = max((np.asscalar(gp.predict(state)[0]), action) for action, gp in self.gps.items())[1]
                    self.rl_logger.info("\tChose with Q: %s", action)

                for a, gp in self.gps.items():
                    self.rl_logger.debug("\tApproximated Q values for (above State,%s) pair: %10.3f",a, np.asscalar(gp.predict(np.asarray(state).reshape((1,-1)))[0]))

        if self.local_time_stamp % 10 == 0:
            self.rl_logger.debug("Q: %s"%self.q)
            
        # decay epsilon
        if self.local_time_stamp % 10==0:
            self.epsilon = max(self.epsilon*0.9,0.05)

        self.local_time_stamp += 1
        self.prev_action = action
        self.prev_state = state

        self.rl_logger.info('===========================================================================\n')
        return action

    def get_policy_info(self):
        return self.q,self.gps,self.prev_state,self.prev_action
