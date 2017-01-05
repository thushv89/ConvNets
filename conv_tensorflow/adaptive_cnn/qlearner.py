__author__ = 'Thushan Ganegedara'

from enum import IntEnum
from collections import defaultdict
from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np
import json
import random
import logging
import sys
from math import ceil
logging_level = logging.DEBUG
logging_format = '[%(name)s] [%(funcName)s] %(message)s'

class AdaCNNConstructionQLearner(object):

    def __init__(self, **params):

        self.learning_rate = params['learning_rate']
        self.discount_rate = params['discount_rate']
        self.image_size = params['image_size']
        self.upper_bound = params['upper_bound']
        self.q = {} #store Q values
        self.epsilon = params['epsilon']
        self.experiance_dict = {}
        self.rl_logger = logging.getLogger('Discreet Policy Logger')
        self.rl_logger.setLevel(logging_level)
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter(logging_format))
        console.setLevel(logging_level)
        self.rl_logger.addHandler(console)

        self.local_time_stamp = 0
        # all the available actions
        # 'C',r,s,d => rxr convolution with sxs stride with d feature maps
        # 'P',r,s => max pooling with rxr window size and sxs stride
        self.actions = [
            ('C',1,1,64),('C',1,1,128),('C',1,1,256),
            ('C',3,1,64),('C',3,1,128),('C',3,1,256),
            ('C',5,1,64),('C',5,1,128),('C',5,1,256),
            ('P',2,2),('P',3,2),('P',5,2),('Terminate',0,0,0)
        ]
        self.init_state = (-1,('Init',0,0,0),self.image_size)


    def restore_policy(self,**restore_data):
        # use this to restore from saved data
        #self.prev_state = restore_data['prev_state']
        #self.prev_action = restore_data['prev_action']
        self.q = restore_data['q']

    def get_output_size(self,in_size,op):
        '''
        Takes a new operations and concat with existing set of operations
        then output what the final output size will be
        :param op_list: list of operations
        :param hyp: Hyperparameters fo the operation
        :return: a tuple (width,height) of x
        '''

        x = in_size
        if op[0] == 'C' or op[0] == 'P':
            s = op[2]
            x = ceil(float(x)/float(s))

        return x

    def output_trajectory(self):

        prev_actions = [] # all the actions belonging to a output trajectory
        prev_states = [] # all the states belonging to a output trajectory

        # State => Layer_Depth,Operation,Output_Size

        # Errors (current and previous) used to calculate reward
        # err_t should be calculated on something the CNN hasn't seen yet
        # err_t-1 should be calculated on something the CNN has seen (e.g mean of last 10 batches)

        layer_depth = self.init_state[0]
        output_size = self.image_size

        while layer_depth<self.upper_bound and output_size>1:
            # If we're just starting
            if len(prev_states) == 0 or len(prev_actions) ==0:
                state = self.init_state
            else:
                # update output size. This is the output of the current state
                output_size = self.get_output_size(output_size,prev_actions[-1])
                state = (layer_depth,prev_actions[-1],output_size)
                self.rl_logger.info('Data for (Depth,Current Op,Output Size) %s'%str(state))

            # initializing q values to zero
            if state not in self.q:
                act_dict = {}
                for a in self.actions:
                    act_dict[a] = 0
                self.q[state]=act_dict

            # deterministic selection (if epsilon is not 1 or q is not empty)
            if np.random.random()>self.epsilon:
                self.rl_logger.debug('Choosing action deterministic...')
                copy_actions = list(self.actions)
                np.random.shuffle(copy_actions)
                action_idx = np.asscalar(np.argmax([self.q[state][a] for a in self.actions]))
                action = copy_actions[action_idx]
                self.rl_logger.debug('\tChose: %s'%str(action))
                # If action Terminate received, we set it to None. Coz at the end we add terminate anyway
                if action[0] == 'Terminate':
                    action = None
                    self.rl_logger.debug('\tTerminate action selected. Terminating Loop')
                    break
                # ================= Look ahead 1 step (Validate Predicted Action) =========================
                # make sure predicted action stride is not larger than resulting output.
                # make sure predicted kernel size is not larger than the resulting output
                # To avoid such scenarios we create a restricted action space if this happens and chose from that
                restricted_action_space = list(self.actions)
                predicted_kernel_size = action[1]
                predicted_stride = action[2]
                next_output_size =self.get_output_size(output_size,action)

                while next_output_size<predicted_stride or next_output_size<predicted_kernel_size:
                    self.rl_logger.debug('\tAction %s is not valid (Predicted output size: %d). '%(str(action),next_output_size))
                    restricted_action_space.remove(action)
                    # if we do not have any more possible actions
                    if len(restricted_action_space)==0:
                        action = None
                        break
                    action_idx = np.asscalar(np.argmax([self.q[state][a] for a in restricted_action_space]))
                    action = restricted_action_space[action_idx]
                    # update kernel,stride,output_size accordingly
                    predicted_kernel_size = action[1]
                    predicted_stride = action[2]
                    next_output_size =self.get_output_size(output_size,action)

            # random selection
            else:
                self.rl_logger.debug('Choosing action stochastic...')
                action = self.actions[np.random.randint(0,len(self.actions))]
                self.rl_logger.debug('\tChose: %s'%str(action))
                # If action Terminate received, we set it to None. Coz at the end we add terminate anyway
                if action[0] == 'Terminate':
                    action = None
                    self.rl_logger.debug('\tTerminate action selected. Terminating Loop')
                    break

                # ================= Look ahead 1 step (Validate Predicted Action) =========================
                # make sure predicted action stride is not larger than resulting output.
                # make sure predicted kernel size is not larger than the resulting output
                # To avoid such scenarios we create a restricted action space if this happens
                restricted_action_space = list(self.actions)
                predicted_kernel_size = action[1]
                predicted_stride = action[2]
                next_output_size =self.get_output_size(output_size,action)

                while next_output_size<predicted_stride or next_output_size<predicted_kernel_size:
                    self.rl_logger.debug('\tAction %s is not valid (Predicted output size: %d). '%(str(action),next_output_size))
                    restricted_action_space.remove(action)
                    # if we do not have any more possible actions
                    if len(restricted_action_space)==0:
                        action = None
                        break

                    action_idx = np.asscalar(np.argmax([self.q[state][a] for a in restricted_action_space]))
                    action = restricted_action_space[action_idx]
                    # update kernel,stride,output_size accordingly
                    predicted_kernel_size = action[1]
                    predicted_stride = action[2]
                    next_output_size =self.get_output_size(output_size,action)

            self.rl_logger.debug('Finally Selected action: %s'%str(action))
            # if a valid state is found
            if action is not None:
                prev_actions.append(action)
                prev_states.append(state)

                # update layer depth
                layer_depth += 1
            # if valid state not found
            else:
                break

        # Terminal State
        terminal_state = (layer_depth,self.actions[-1],output_size)
        prev_states.append(terminal_state)
        # this q value represents terminating network at different depths
        if terminal_state not in self.q:
            self.q[terminal_state]={self.actions[-1]:0}
            self.rl_logger.debug('Added terminal state %s to Q with value (%s)'%(str(terminal_state),str(self.actions[-1])))

        # decay epsilon
        self.epsilon = max(self.epsilon*0.9,0.1)
        self.rl_logger.debug('='*60)
        self.rl_logger.debug('States')
        self.rl_logger.debug(prev_states)
        self.rl_logger.debug('='*60)
        return prev_states

    def update_policy(self,data):
        '''
        Update the policy
        :param data: ['accuracy']['trajectory']
        :return:
        '''

        acc = float(data['accuracy'])/100.0
        reward = acc

        self.rl_logger.info("Reward for trajectory: %.3f"%reward)

        for t_i in range(len(data['trajectory'])-1):
            state = data['trajectory'][t_i]
            next_state = data['trajectory'][t_i+1]
            action = next_state[1]
            self.rl_logger.debug('Updating Q for ...')
            self.rl_logger.debug('\tCurrent State: %s'%str(state))
            self.rl_logger.debug('\tAction Taken: %s'%str(action))
            self.rl_logger.debug('\tNext State: %s'%str(next_state))
            # Bellman's equation (iterative)
            # si-prev state, a-action_taken, sj-next state
            # Q(t+1)(si,a) = (1-alp)*Q(t)(si,a) + alp*[r(t)+gam*max(Q(t)(sj,a'))]
            self.q[state][action] = (1-self.learning_rate)*self.q[state][action] +\
                self.learning_rate*(reward+self.discount_rate*np.max([self.q[next_state][a] for a in self.q[next_state].keys()]))

        self.rl_logger.info('\tUpdated the Q values ...')

    def get_policy_info(self):
        return self.q,self.gps,self.prev_state,self.prev_action


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
        self.actions = []

        self.prev_action = None
        self.prev_state = None
        self.finetune_action = None

    def restore_policy(self,**restore_data):
        # use this to restore from saved data
        self.prev_state = restore_data['prev_state']
        self.prev_action = restore_data['prev_action']
        self.q = restore_data['q']
        self.gps = restore_data['gps']

    def update_policy(self, global_time_stamp, data,success=True):

        self.rl_logger.info('\n============== Policy Update for %d (Global: %d) ==============='%(self.local_time_stamp,global_time_stamp))
        # Errors (current and previous) used to calculate reward
        # err_t should be calculated on something the CNN hasn't seen yet
        # err_t-1 should be calculated on something the CNN has seen (e.g mean of last 10 batches)

        err_state = float(data['error_t'])/100.0
        conv_state = float(data['conv_layer_count'])/10.0
        pool_state = float(data['pool_layer_count'])/10.0
        param_state = float(data['all_params'])/100000.0

        state = (err_state,conv_state,pool_state,param_state)

        if self.local_time_stamp <= self.eta_1:
            # If enough time has not been spend collecting data, finetune
            action = self.finetune_action
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
                # we do not worry about x having only 1 input instance because we won't fit a curve unless there are more than
                # a given number of inputs
                gp.fit(np.asarray(x), np.asarray(y))

                self.gps[a] = gp

        if self.prev_state or self.prev_action:

            err_t = float(data['error_t'])/100.0
            error_cost = (1.0 - ((err_t+self.prev_state[0])/2.0))*(err_t-self.prev_state[0])
            time_cost = float(data['time_cost'])/10.0
            param_cost = float(data['param_cost'])/10000.0
            stride_cost = float(data['stride_cost'])/50.0
            layer_rank_cost = 2**(1-float(data['layer_rank']))
            complexity_cost = min(float(2**data['complexity_cost'])/2**8,0.5)
            success_cost = 0 if success else 10

            self.rl_logger.info('Data for %s: E %.2f, T %.2f, P %.2f, S %.2f, Rnk %.2f, Cpx %.2f Suc %.2f'%(self.prev_action,err_t,time_cost,param_cost,stride_cost,layer_rank_cost,complexity_cost,success_cost))
            if 'remove' in self.prev_action:
                reward = -(error_cost*20.0 + time_cost + param_cost + stride_cost + complexity_cost + success_cost + layer_rank_cost)
            else:
                reward = -(error_cost*20.0 + time_cost + param_cost + stride_cost + complexity_cost + success_cost)

            self.rl_logger.info("Reward for action: %s: %.3f"%(self.prev_action,reward))
            # sample = reward + self.discount_rate * max(self.q[state, a] for a in self.actions)
            # len(gps) == 0 in the first time move() is called
            if self.prev_action not in self.gps or len(self.q[self.prev_action]) < 3:
                self.rl_logger.debug('\tNot enough data to predict with a GP. Using the reward as the sample ...')
                sample = reward
            else:
                self.rl_logger.debug('\tPredicting with a GP. Using the gp prediction + reward as the sample ...')
                sample = reward + self.discount_rate * max((np.asscalar(gp.predict(np.asarray(self.prev_state).reshape((1,-1)))[0])) for gp in self.gps.values())

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
                    action = max((np.asscalar(gp.predict(np.asarray(state).reshape((1,-1)))[0]), action) for action, gp in self.gps.items())[1]
                    self.rl_logger.info("\tChose with Q: %s", action)

                for a, gp in self.gps.items():
                    self.rl_logger.debug("\tApproximated Q values for (above State,%s) pair: %10.3f",a, np.asscalar(gp.predict(np.asarray(state).reshape((1,-1)))[0]))

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
