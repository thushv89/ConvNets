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

        self.action_logger = logging.getLogger('Policy Logger')
        self.action_logger.setLevel(logging_level)
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter(logging_format))
        console.setLevel(logging_level)
        self.action_logger.addHandler(console)

        self.actions = [
            'add,C_K5x5x16_S3x3','add,C_K5x5x64_S3x3','add,C_K5x5x16_S1x1','add,C_K5x5x64_S1x1',
            'add,C_K3x3x16_S3x3','add,C_K3x3x64_S3x3','add,C_K3x3x16_S1x1','add,C_K3x3x64_S1x1',
            'add,C_K1x1x16_S3x3','add,C_K1x1x64_S3x3','add,C_K1x1x16_S1x1','add,C_K1x1x64_S1x1',
            'add,P_K5x5_S3x3_avg','add,P_K5x5_S1x1_avg','add,P_K2x2_S2x2_avg','add,P_K2x2_S1x1_avg',
            'add,P_K5x5_S3x3_max','add,P_K5x5_S1x1_max','add,P_K2x2_S2x2_max','add,P_K2x2_S1x1_max'
        ]
        self.prev_action = None

    def restore_policy(self,**restore_data):
        # use this to restore from saved data
        self.prev_state = restore_data['prev_state']
        self.prev_action = restore_data['prev_action']
        self.q = restore_data['q']
        self.gps = restore_data['gps']

    def update_policy(self, global_time_stamp, data):
        #we do not consider states in calculating Q values here
        # to speed up computations

        a = self.actions[self.local_time_stamp%len(self.actions)]

        if self.prev_action is not None:
            # Errors (current and previous) used to calculate reward
            # err_t should be calculated on something the CNN hasn't seen yet
            # err_t-1 should be calculated on something the CNN has seen (e.g mean of last 10 batches)

            time_cost = data['time_cost']
            param_cost = data['param_cost']
            stride_cost = data['stride_cost']
            reward = -(data['error_t'] + 0.05*stride_cost)

            print('\tActionPicker - Running action: %s'%a)
            if self.prev_action not in self.q:
                self.q[self.prev_action] = reward
            else:
                self.q[self.prev_action] = (1-self.learning_rate)*self.q[self.prev_action] + self.learning_rate*reward
            print('\tActionPicker - Updated Q[%s]=%.3f'%(self.prev_action,self.q[self.prev_action]))

        self.prev_action = a
        self.local_time_stamp += 1

        return a

    def get_best_actions(self,count):
        sorted_actions = list(sorted(self.q))
        sorted_actions.reverse()
        return sorted(self.q)[0:count]

    def get_policy_info(self):
        return self.q,self.gps,self.prev_state,self.prev_action
