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
            'add,P_K5x5_S3x3','add,P_K5x5_S1x1','add,P_K2x2_S2x2','add,P_K2x2_S1x1'
        ]

    def restore_policy(self,**restore_data):
        # use this to restore from saved data
        self.prev_state = restore_data['prev_state']
        self.prev_action = restore_data['prev_action']
        self.q = restore_data['q']
        self.gps = restore_data['gps']

    def update_policy(self, global_time_stamp, data):



        #we do not consider states in calculating Q values here
        # to speed up computations

        if self.prev_action is not None:
            # Errors (current and previous) used to calculate reward
            # err_t should be calculated on something the CNN hasn't seen yet
            # err_t-1 should be calculated on something the CNN has seen (e.g mean of last 10 batches)
            err_t = data['error_t']
            err_t_minus_1 = data['error_t-1']
            time_cost = data['avg_time_cost']
            param_cost = data['param_cost']

            reward = -(data['error_t'] + 0.1*time_cost + 1e-6*param_cost)

            a = self.actions[self.local_time_stamp%len(self.actions)]
            if a not in self.q:
                self.q[a] = reward
            else:
                self.q[a] = (1-self.learning_rate)*self.q[a] + self.learning_rate*reward

        self.local_time_stamp += 1
        self.rl_logger.info('\n')

    def get_best_actions(self,count):
        sorted_actions = sorted(self.q).reverse()
        return sorted_actions[0:count]

    def get_policy_info(self):
        return self.q,self.gps,self.prev_state,self.prev_action
