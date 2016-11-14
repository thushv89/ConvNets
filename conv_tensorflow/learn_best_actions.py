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


class ActionPicker(object):
    '''
    This class will be used to learn the set of best actions at a given time
    Motivation for this class is that, number of hyperparameters for subsampling layers is very high
    This can result in too many actions in the action space
    This class will determine roughly the best actions from the complete actions space and return those actions
    for actual Q learning algorithm
    '''
    def __init__(self, **params):
        self.params = params
        self.learning_rate = params['learning_rate']
        self.discount_rate = params['discount_rate']

        self.q = {} #store Q values mapping from action to Q in this case

        self.local_time_stamp = 0

        self.action_logger = logging.getLogger('Action picker Logger')
        self.action_logger.setLevel(logging_level)
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter(logging_format))
        console.setLevel(logging_level)
        self.action_logger.addHandler(console)

        self.actions = [
            'add,C_K5x5x16_S2x2','add,C_K5x5x64_S2x2','add,C_K5x5x16_S1x1','add,C_K5x5x64_S1x1',
            'add,C_K3x3x16_S2x2','add,C_K3x3x64_S2x2','add,C_K3x3x16_S1x1','add,C_K3x3x64_S1x1',
            'add,C_K1x1x16_S2x2','add,C_K1x1x64_S2x2','add,C_K1x1x16_S1x1','add,C_K1x1x64_S1x1',
            'add,P_K5x5_S2x2_avg','add,P_K5x5_S1x1_avg','add,P_K2x2_S2x2_avg','add,P_K2x2_S1x1_avg',
            'add,P_K5x5_S2x2_max','add,P_K5x5_S1x1_max','add,P_K2x2_S2x2_max','add,P_K2x2_S1x1_max'
        ]
        self.prev_action = None

    def restore_policy(self,**restore_data):
        # use this to restore from saved data
        self.prev_state = restore_data['prev_state']
        self.prev_action = restore_data['prev_action']
        self.q = restore_data['q']
        self.gps = restore_data['gps']

    def reset_all(self):
        self.prev_action = None
        self.prev_state = None
        self.q = {}

    def update_policy(self, global_time_stamp, data, success):
        #we do not consider states in calculating Q values here
        # to speed up computations

        a = self.actions[self.local_time_stamp%len(self.actions)]
        error_fraction = float(data['error_t'])/100.0

        if self.prev_action is not None:
            # Errors (current and previous) used to calculate reward
            # err_t should be calculated on something the CNN hasn't seen yet
            # err_t-1 should be calculated on something the CNN has seen (e.g mean of last 10 batches)


            # this error cost captures costs and rewards as follows
            # e(t-1) -> e(t) = high rew / high cost / low rew / low cost
            #   0.3  -> 0.5  = high cost
            #   0.6  -> 0.8  = low cost
            #   0.5  -> 0.3  = high reward
            #   0.8  -> 0.6  = low reward
            error_cost = (1.0 - ((error_fraction+self.prev_state[0])/2.0))*(error_fraction-self.prev_state[0])
            stride_cost = data['stride_cost']
            success_cost = 0 if success else -100
            reward = -(10*error_cost + 1e-3*stride_cost) + success_cost

            print('\tActionPicker - Running action: %s'%a)
            if self.prev_action not in self.q:
                self.q[self.prev_action] = reward
            else:
                self.q[self.prev_action] = (1-self.learning_rate)*self.q[self.prev_action] + self.learning_rate*reward
            print('\tActionPicker - Updated Q[%s]=%.3f'%(self.prev_action,self.q[self.prev_action]))

        self.prev_action = a
        self.prev_state = [error_fraction]
        self.local_time_stamp += 1

        return a

    def get_best_actions(self,count):
        print('Q: %s'%self.q)
        sorted_actions = list(sorted(self.q,reverse=True))
        return sorted_actions[0:count]

    def get_policy_info(self):
        return self.q,self.gps,self.prev_state,self.prev_action
