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
from six.moves import cPickle as pickle
import os

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
        self.experiance_tuples = []
        self.rl_logger = logging.getLogger('Discreet Policy Logger')
        self.rl_logger.setLevel(logging_level)
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter(logging_format))
        console.setLevel(logging_level)
        self.rl_logger.addHandler(console)

        self.persist_dir = 'constructor_rl' # various things we persist related to ConstructorRL
        if not os.path.exists(self.persist_dir):
            os.makedirs(self.persist_dir)

        # Log the output every few epochs
        self.best_policy_logger = logging.getLogger('best_policy_logger')
        self.best_policy_logger.setLevel(logging.INFO)
        bpfileHandler = logging.FileHandler(self.persist_dir+os.sep+'best_policy_log', mode='w')
        bpfileHandler.setFormatter(logging.Formatter('%(message)s'))
        self.best_policy_logger.addHandler(bpfileHandler)

        self.local_time_stamp = 0
        # all the available actions
        # 'C',r,s,d => rxr convolution with sxs stride with d feature maps
        # 'P',r,s => max pooling with rxr window size and sxs stride
        self.actions = [
            ('C',1,1,64),('C',1,1,128),('C',1,1,256),
            ('C',3,1,64),('C',3,1,128),('C',3,1,256),
            ('C',5,1,64),('C',5,1,128),('C',5,1,256),
            ('P',2,2,0),('P',3,2,0),('P',5,2,0),
            ('Terminate',0,0,0)
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

                    action = restricted_action_space[np.random.randint(0,len(restricted_action_space))]
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
        self.experiance_tuples.append((data['trajectory'],data['accuracy']))

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

        # Replay experiance
        if np.random.random()<0.1:
            self.rl_logger.debug('Replaying Experience')
            for traj,acc in self.experiance_tuples:
                exp_reward = acc
                for t_i in range(len(traj)-1):
                    state = traj[t_i]
                    next_state = traj[t_i+1]
                    action = next_state[1]
                    # Bellman's equation (iterative)
                    # si-prev state, a-action_taken, sj-next state
                    # Q(t+1)(si,a) = (1-alp)*Q(t)(si,a) + alp*[r(t)+gam*max(Q(t)(sj,a'))]
                    self.q[state][action] = (1-self.learning_rate)*self.q[state][action] +\
                        self.learning_rate*(exp_reward+self.discount_rate*np.max([self.q[next_state][a] for a in self.q[next_state].keys()]))

        self.rl_logger.info('\tUpdated the Q values ...')
        self.local_time_stamp += 1

        if self.local_time_stamp%10==0:
            with open(self.persist_dir+os.sep+'Q_'+str(self.local_time_stamp)+'.pickle','wb') as f:
                pickle.dump(self.q, f, pickle.HIGHEST_PROTOCOL)
        if self.local_time_stamp%25==0:
            with open(self.persist_dir+os.sep+'experience_'+str(self.local_time_stamp)+'.pickle','wb') as f:
                pickle.dump(self.experiance_tuples, f, pickle.HIGHEST_PROTOCOL)

        # log every structure proposed
        net_string = ''
        for s in data['trajectory']:
            net_string += '#'+s[1][0]+','+str(s[1][1])+','+str(s[1][2])+','+str(s[1][3])
        self.best_policy_logger.info("%d,%s,%.3f",self.local_time_stamp,net_string,data['accuracy'])

    def get_policy_info(self):
        return self.q,self.gps,self.prev_state,self.prev_action


class AdaCNNAdaptingQLearner(object):

    def __init__(self, **params):

        self.learning_rate = params['learning_rate']
        self.discount_rate = params['discount_rate']
        self.filter_upper_bound = params['filter_upper_bound']

        self.gp_interval = params['gp_interval'] #GP calculating interval (10)
        self.q = {} #store Q values
        self.gps = {} #store gaussian curves

        self.epsilon = params['epsilon']
        self.net_depth = params['net_depth']

        self.rl_logger = logging.getLogger('Adapting Policy Logger')
        self.rl_logger.setLevel(logging_level)
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter(logging_format))
        console.setLevel(logging_level)
        self.rl_logger.addHandler(console)

        self.local_time_stamp = 0
        self.actions = [
            ('add',32),('add',64),('add',128),
            ('remove',32),('remove',64),('remove',128),
            ('finetune',0),('do_nothing',0)
        ]

        self.init_state = (-1,0,0)

    def restore_policy(self,**restore_data):
        # use this to restore from saved data
        self.q = restore_data['q']
        self.gps = restore_data['gps']

    def output_action(self,data):
        prev_actions = []
        prev_states = []

        # data => ['distMSE']['filter_counts']
        # ['filter_counts'] => depth_index : filter_count
        # State => Layer_Depth (w.r.t net), dist_MSE, number of filters in layer
        for ni in range(self.net_depth):
            if len(prev_states) == 0 or len(prev_actions) ==0:
                state = self.init_state
            else:
                state = (ni,data['distMSE'],data['filter_counts'][ni])
                self.rl_logger.info('Data for (Depth Index,DistMSE,Filter Count) %s'%str(state))

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
                q_for_actions = []
                for a in copy_actions:
                    if a in self.gps and ni in self.gps[a]:
                        q_for_actions.append(self.gps[a][ni].predict(state))
                    else:
                        q_for_actions = 0.0

                action_idx = np.asscalar(np.argmax(q_for_actions))
                action = copy_actions[action_idx]
                self.rl_logger.debug('\tChose: %s'%str(action))

                # ================= Look ahead 1 step (Validate Predicted Action) =========================
                # make sure predicted action stride is not larger than resulting output.
                # make sure predicted kernel size is not larger than the resulting output
                # To avoid such scenarios we create a restricted action space if this happens and chose from that
                restricted_action_space = list(copy_actions)
                if action[0]=='add':
                    next_filter_count =data['filter_counts'][ni]+action[1]
                elif action[0]=='remove':
                    next_filter_count =data['filter_counts'][ni]-action[1]

                while next_filter_count<0 or next_filter_count>self.filter_upper_bound:
                    self.rl_logger.debug('\tAction %s is not valid (Next Filter Count: %d). '%(str(action),next_filter_count))
                    del q_for_actions[restricted_action_space.index(action)]
                    restricted_action_space.remove(action)

                    # if we do not have any more possible actions
                    if len(restricted_action_space)==0:
                        action = self.actions[-1]
                        break

                    action_idx = np.asscalar(np.argmax(q_for_actions))
                    action = restricted_action_space[action_idx]
                    # update FILTER count
                    if action[0]=='add':
                        next_filter_count =data['filter_counts'][ni]+action[1]
                    elif action[0]=='remove':
                        next_filter_count =data['filter_counts'][ni]-action[1]

            # random selection
            else:
                self.rl_logger.debug('Choosing action stochastic...')
                action = self.actions[np.random.randint(0,len(self.actions))]
                self.rl_logger.debug('\tChose: %s'%str(action))

                # ================= Look ahead 1 step (Validate Predicted Action) =========================
                # make sure predicted action stride is not larger than resulting output.
                # make sure predicted kernel size is not larger than the resulting output
                # To avoid such scenarios we create a restricted action space if this happens
                restricted_action_space = list(self.actions)
                if action[0]=='add':
                    next_filter_count =data['filter_counts'][ni]+action[1]
                elif action[0]=='remove':
                    next_filter_count =data['filter_counts'][ni]-action[1]

                while next_filter_count<0 or next_filter_count>self.filter_upper_bound:
                    self.rl_logger.debug('\tAction %s is not valid (Next Filter Count: %d). '%(str(action),next_filter_count))
                    restricted_action_space.remove(action)

                    # if we do not have any more possible actions
                    if len(restricted_action_space)==0:
                        action = self.actions[-1]
                        break

                    action = restricted_action_space[np.random.randint(0,len(restricted_action_space))]

                    # update FILTER count
                    if action[0]=='add':
                        next_filter_count =data['filter_counts'][ni]+action[1]
                    elif action[0]=='remove':
                        next_filter_count =data['filter_counts'][ni]-action[1]

            self.rl_logger.debug('Finally Selected action: %s'%str(action))

        prev_actions.append(action)
        prev_states.append(state)

        # decay epsilon
        self.epsilon = max(self.epsilon*0.9,0.1)
        self.rl_logger.debug('='*60)
        self.rl_logger.debug('States')
        self.rl_logger.debug(prev_states)
        self.rl_logger.debug('='*60)

        return (prev_states,prev_actions)

    def update_policy(self, data):
        # data['states'] => list of states
        # data['actions'] => list of actions
        # data['accuracy'] => validation accuracy
        reward = data['accuracy']

        for si,ai in zip(data['states'],data['actions']):
            sj = si
            if ai[0]=='add':
                sj[2]=si[2]+ai[1]
            elif ai[0]=='remove':
                sj[2]=si[2]-ai[1]

            # gp_data[a][layer_index][(state,q)]
            if ai not in self.q:
                self.q[ai]={}
            else:
                if si[0] not in self.gp_data[ai]:
                    self.q[ai]={si[0]:[(si,reward)]}
                else:
                    self.q[ai][si[0]].append((si,reward))

            self.q[ai][si] = (1-self.learning_rate)*self.q[ai][si] +\
                        self.learning_rate*(reward+self.discount_rate*np.max([self.q[next_state][a] for a in self.q[next_state].keys()]))

        self.local_time_stamp += 1
    def get_policy_info(self):
        return self.q,self.gps,self.prev_state,self.prev_action
