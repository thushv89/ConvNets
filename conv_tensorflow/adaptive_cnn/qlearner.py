__author__ = 'Thushan Ganegedara'

from enum import IntEnum
from collections import defaultdict
#from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np
import json
import random
import logging
import sys
from math import ceil
from six.moves import cPickle as pickle
import os
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.base import clone
import tensorflow as tf

from collections import OrderedDict

logging_level = logging.DEBUG
logging_format = '[%(name)s] [%(funcName)s] %(message)s'

class AdaCNNConstructionQLearner(object):

    def __init__(self, **params):

        self.learning_rate = params['learning_rate']
        self.discount_rate = params['discount_rate']
        self.image_size = params['image_size']
        self.upper_bound = params['upper_bound']
        self.num_episodes = params['num_episodes']

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
            ('C',1,1,64),
            ('C',3,1,64),('C',3,1,128),('C',3,1,256),('C',3,1,512),
            ('C',5,1,64),('C',5,1,128),('C',5,1,256),('C',5,1,512),
            ('P',2,2,0),('P',3,2,0),('P',5,2,0),
            ('Terminate',0,0,0)
        ]
        self.init_state = (-1,('Init',0,0,0),self.image_size)

        self.best_model_string = ''
        self.best_model_accuracy = 0.0

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
            action = None
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
                action_idx = np.asscalar(np.argmax([self.q[state][a] for a in copy_actions]))
                action = copy_actions[action_idx]
                self.rl_logger.debug('\tChose: %s'%str(action))

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

            # If action Terminate received, we set it to None. Coz at the end we add terminate anyway
            if action[0] == 'Terminate':
                action = None
                self.rl_logger.debug('\tTerminate action selected.')

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

        if data['accuracy']>self.best_model_accuracy:
            self.best_model_accuracy = data['accuracy']
            self.best_model_string = net_string

        if self.num_episodes==self.local_time_stamp:
            self.best_policy_logger.info()
            self.best_policy_logger.info("Best:%s,%.3f",net_string,data['accuracy'])


    def get_policy_info(self):
        return self.q,self.gps,self.prev_state,self.prev_action


class AdaCNNAdaptingQLearner(object):

    def __init__(self, **params):


        self.discount_rate = params['discount_rate']
        self.filter_upper_bound = params['filter_upper_bound']
        self.filter_min_bound = params['filter_min_bound']

        self.fit_interval = params['fit_interval'] #GP calculating interval (10)
        self.target_update_rate = params['target_update_rate']

        self.persit_dir = params['persist_dir']
        self.q_logger = logging.getLogger('pred_q_logger')
        self.q_logger.setLevel(logging.INFO)
        q_distHandler = logging.FileHandler(self.persit_dir + os.sep + 'predicted_q.log', mode='w')
        q_distHandler.setFormatter(logging.Formatter('%(message)s'))
        self.q_logger.addHandler(q_distHandler)

        self.reward_logger = logging.getLogger('reward_logger')
        self.reward_logger.setLevel(logging.INFO)
        rewarddistHandler = logging.FileHandler(self.persit_dir + os.sep + 'reward.log', mode='w')
        rewarddistHandler.setFormatter(logging.Formatter('%(message)s'))
        self.reward_logger.addHandler(rewarddistHandler)

        self.explore_tries = params['exploratory_tries']
        self.explore_interval = params['exploratory_interval']

        self.batch_size = params['batch_size']

        self.regressor = MLPRegressor(activation='relu', batch_size=self.batch_size,
                                      hidden_layer_sizes=(128, 64, 32), learning_rate='constant',
                                      learning_rate_init=0.001, max_iter=10,
                                      random_state=1, shuffle=True,
                                      solver='sgd', momentum=0.95)

        self.target_network = None
        self.net_depth = params['net_depth']
        self.n_conv = params['n_conv'] # number of convolutional layers
        self.conv_ids = params['conv_ids']

        self.filter_bound_vec = self.make_filter_bound_vector(self.filter_upper_bound, self.net_depth,self.conv_ids)

        self.rl_logger = logging.getLogger('Adapting Policy Logger')
        self.rl_logger.setLevel(logging_level)
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter(logging_format))
        console.setLevel(logging_level)
        self.rl_logger.addHandler(console)
        self.rl_logger.debug('Conv ids: %s',self.conv_ids)
        self.rl_logger.debug('Filter vector %s',self.filter_bound_vec)
        self.local_time_stamp = 0
        self.global_time_stamp = 0

        self.actions = [
            ('do_nothing', 0),('finetune', 0),
            ('add',16),('remove',8)
        ]
        self.q_logger.info('#%s',self.actions)

        self.q_length = 25 * len(self.actions)

        self.past_mean_accuracy = 0

        self.prev_action,self.prev_state = None,None
        self.same_action_count = [0 for _ in range(self.net_depth)]
        self.epsilons = [params['epsilon'] for _ in range(self.net_depth)]

        self.same_action_threshold = 10

        # Of format {s1,a1,s2,a2,s3,a3} NOTE that this doesnt hold the current state
        self.state_history_length = 4

        self.session = params['session']

        self.input_size = 48

        self.layer_info = [self.calculate_input_size(),128, 64, 32,len(self.actions)]
        print(self.layer_info)
        self.current_state_history = []
        # Format of {phi(s_t),a_t,r_t,phi(s_t+1)}
        self.experience = []

        self.tf_weights,self.tf_bias = [],[]
        self.tf_target_weights,self.tf_target_biase = [],[]

        self.momentum = 0.9
        self.learning_rate = 0.001

        self.tf_init_mlp()
        self.tf_state_input = tf.placeholder(tf.float32, shape=(None, self.input_size),name='InputDataset')
        self.tf_q_targets = tf.placeholder(tf.float32, shape=(None,len(self.actions)),name='TargetDataset')

        self.tf_out_op = self.tf_calc_output(self.tf_state_input)
        self.tf_out_target_op = self.tf_calc_output_target(self.tf_state_input)
        self.tf_loss_op = self.tf_sqr_loss(self.tf_out_op,self.tf_q_targets)
        self.tf_optimize_op = self.tf_momentum_optimize(self.tf_loss_op)

        self.tf_target_update_ops = self.tf_target_weight_copy_op()

        all_variables = []
        for w,b,wt,bt in zip(self.tf_weights,self.tf_bias,self.tf_target_weights,self.tf_target_biase):
            all_variables.extend([w,b,wt,bt])
        init_op = tf.variables_initializer(all_variables)
        _ = self.session.run(init_op)

        self.state_history_collector = []
        self.state_history_dumped = False

    def calculate_input_size(self):
        dummy_state = (0,0,0)
        dummy_action = ('add',0)
        dummy_history = []
        for _ in range(self.state_history_length-1):
            dummy_history.append([dummy_state,dummy_action])
        dummy_history.append([dummy_state])

        return 48

    def tf_init_mlp(self):
         for li in range(len(self.layer_info)-1):
             self.tf_weights.append(tf.Variable(tf.truncated_normal([self.layer_info[li],self.layer_info[li+1]],
                                                                    stddev= 2./self.layer_info[li]),
                                                name='weights_'+str(li)+'_'+str(li+1)))
             self.tf_target_weights.append(tf.Variable(tf.truncated_normal([self.layer_info[li], self.layer_info[li + 1]],
                                                                    stddev=2. / self.layer_info[li]),
                                                name='target_weights_' + str(li) + '_' + str(li + 1)))
             self.tf_bias.append(tf.Variable(tf.zeros([self.layer_info[li+1]]),name = 'bias_'+str(li)+'_'+str(li+1)))
             self.tf_target_biase.append(
                 tf.Variable(tf.zeros([self.layer_info[li + 1]]), name='target_bias_' + str(li) + '_' + str(li + 1)))

    def tf_calc_output(self,tf_state_input):
        x = tf_state_input
        for li,(w,b) in enumerate(zip(self.tf_weights[:-1],self.tf_bias[:-1])):
            x = tf.nn.relu(tf.matmul(x,w) + b)

        return tf.matmul(x,self.tf_weights[-1])+self.tf_bias[-1]

    def tf_calc_output_target(self,tf_state_input):
        x = tf_state_input
        for li,(w,b) in enumerate(zip(self.tf_target_weights[:-1],self.tf_target_biase[:-1])):
            x = tf.nn.relu(tf.matmul(x,w) + b)

        return tf.matmul(x,self.tf_weights[-1])+self.tf_bias[-1]

    def tf_sqr_loss(self,tf_output,tf_targets):
        return tf.reduce_mean((tf_output-tf_targets)**2)

    def tf_momentum_optimize(self,loss):
        optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate,
                                   momentum=self.momentum).minimize(loss)
        return optimizer

    def tf_target_weight_copy_op(self):
        update_ops = []
        for li,(w,b) in enumerate(zip(self.tf_weights,self.tf_bias)):
            update_ops.append(tf.assign(self.tf_target_weights[li],w))
            update_ops.append(tf.assign(self.tf_target_biase[li], b))

        return update_ops

    def clean_experience(self):

        if len(self.experience)>self.q_length:
            for _ in range(len(self.experience)-self.q_length):
                del self.experience[0]

    def make_filter_bound_vector(self, n_fil,n_layer,conv_ids):
        fil_vec = []
        curr_fil = n_fil/2**(len(conv_ids)-1)
        for li in range(n_layer):
            if li in conv_ids:
                fil_vec.append(max(self.filter_min_bound,int(curr_fil)))
                curr_fil *= 2
            else:
                fil_vec.append(0)

        assert len(fil_vec)==n_layer
        return fil_vec

    def restore_policy(self,**restore_data):
        # use this to restore from saved data
        self.q = restore_data['q']
        self.regressor = restore_data['lrs']

    def clean_Q(self):
        self.rl_logger.debug('Cleaning Q values (removing old ones)')
        self.rl_logger.debug('\tSize of Q before: %d',len(self.q))
        if len(self.q)>self.q_length:
            for _ in range(len(self.q)-self.q_length):
                self.q.popitem(last=True)
            self.rl_logger.debug('\tSize of Q after: %d', len(self.q))

    def output_action(self,data,ni):
        action = None
        invalid_actions = []
        # data => ['distMSE']['filter_counts']
        # ['filter_counts'] => depth_index : filter_count
        # State => Layer_Depth (w.r.t net), dist_MSE, number of filters in layer

        state = (ni,data['distMSE'],data['filter_counts'][ni])
        self.rl_logger.info('Data for (Depth Index,DistMSE,Filter Count) %s\n'%str(state))
        phi_t_plus_1 = list(self.current_state_history)
        phi_t_plus_1.append([state])

        self.rl_logger.debug('Current state history: %s\n', self.current_state_history)
        self.rl_logger.debug('phi_t+1:%s\n',phi_t_plus_1)
        self.rl_logger.debug('Epsilons: %s\n',self.epsilons)
        # pooling operation always action='do nothing'
        if data['filter_counts'][ni]==0:
            return state, self.actions[0]

        # we try actions evenly otherwise cannot have the approximator
        if (self.global_time_stamp%self.explore_interval)<self.explore_tries:
            self.rl_logger.debug('(Exploratory Mode) Choosing action exploratory...')
            action = self.actions[np.random.randint(0,len(self.actions))]

            if action[0]=='add':
                next_filter_count =data['filter_counts'][ni]+action[1]
            elif action[0]=='remove':
                next_filter_count =data['filter_counts'][ni]-action[1]
            else:
                next_filter_count = data['filter_counts'][ni]

            if next_filter_count<=0 or next_filter_count>self.filter_bound_vec[ni]:
                action = self.actions[0]

        # deterministic selection (if epsilon is not 1 or q is not empty)
        elif np.random.random()>self.epsilons[ni]:
            self.rl_logger.debug('Choosing action deterministic...')
            # we create this copy_actions in case we need to change the order the actions processed
            # without changing the original action space (self.actions)
            copy_actions = list(self.actions)

            curr_x = np.asarray(self.get_ohe_state_history(phi_t_plus_1)).reshape(1,-1)
            q_for_actions = self.session.run(self.tf_out_target_op,feed_dict={self.tf_state_input:curr_x})
            q_for_actions = q_for_actions.flatten().tolist()

            q_value_strings = ''
            for q_val in q_for_actions:
                q_value_strings += '%.5f'%q_val+','
            self.q_logger.info("%d,%s",self.local_time_stamp,q_value_strings)
            self.rl_logger.debug('\tActions: %s',self.actions)
            self.rl_logger.debug('\tPredicted Q: %s',q_for_actions)

            argsort_q_for_actions = np.argsort(q_for_actions).flatten()
            #if abs(np.asscalar(q_for_actions[argsort_q_for_actions[-1]]) -
            #               np.asscalar(q_for_actions[argsort_q_for_actions[-2]]))>0.001:
            action_idx = np.asscalar(np.argmax(q_for_actions))
            #else:
            #    self.rl_logger.debug('Choosing stochastic out of best two actions')
            #    action_idx = np.random.choice([np.asscalar(q_for_actions[argsort_q_for_actions[-1]]),
            #                                   np.asscalar(q_for_actions[argsort_q_for_actions[-2]])])

            action = copy_actions[int(action_idx)]
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
            else:
                next_filter_count = data['filter_counts'][ni]

            while next_filter_count<=0 or next_filter_count>self.filter_bound_vec[ni]:
                self.rl_logger.debug('\tAction %s is not valid (Next Filter Count: %d). '%(str(action),next_filter_count))
                #TODO: introduce penalty for trying in invalid action
                del q_for_actions[restricted_action_space.index(action)]
                restricted_action_space.remove(action)
                invalid_actions.append(action)

                # if we do not have any more possible actions
                if len(restricted_action_space)==0:
                    action = self.actions[0]

                action_idx = np.asscalar(np.argmax(q_for_actions))
                action = restricted_action_space[action_idx]
                # update FILTER count
                if action[0]=='add':
                    next_filter_count = data['filter_counts'][ni]+action[1]
                elif action[0]=='remove':
                    next_filter_count = data['filter_counts'][ni]-action[1]
                else:
                    next_filter_count = data['filter_counts'][ni]

        # random selection
        else:
            self.rl_logger.debug('Choosing action stochastic...')
            action = self.actions[np.random.randint(0,len(self.actions))]
            self.rl_logger.debug('\tChose: %s'%str(action))

            # ================= Look ahead 1 step (Validate Predicted Action) =========================
            # make sure predicted action stride is not larger than resulting output.
            # make sure predicted kernel size is not larger than the resulting output
            # To avoid such scenarios we create a
            #  action space if this happens
            restricted_action_space = list(self.actions)
            if action[0]=='add':
                next_filter_count =data['filter_counts'][ni]+action[1]
            elif action[0]=='remove':
                next_filter_count =data['filter_counts'][ni]-action[1]
            else:
                next_filter_count = data['filter_counts'][ni]
            while next_filter_count<=0 or next_filter_count>self.filter_bound_vec[ni]:
                self.rl_logger.debug('\tAction %s is not valid (Next Filter Count: %d). '%(str(action),next_filter_count))
                restricted_action_space.remove(action)
                invalid_actions.append(action)
                #TODO: introduce penalty for trying in invalid action

                # if we do not have any more possible actions
                if len(restricted_action_space)==0:
                    action = self.actions[0]
                    break

                action = restricted_action_space[np.random.randint(0,len(restricted_action_space))]

                # update FILTER count
                if action[0]=='add':
                    next_filter_count =data['filter_counts'][ni]+action[1]
                elif action[0]=='remove':
                    next_filter_count =data['filter_counts'][ni]-action[1]
                else:
                    next_filter_count = data['filter_counts'][ni]

        self.rl_logger.debug('Finally Selected action: %s'%str(action))

        # decay epsilon
        if self.global_time_stamp>self.explore_tries:
            self.epsilons[ni] = max(self.epsilons[ni] * 0.9, 0.1)

            # this is to reduce taking the same action over and over again
            if self.same_action_count[ni] >= self.same_action_threshold:
                self.epsilons[ni] = min(self.epsilons[ni]*2,1.0)

        self.rl_logger.debug('='*60)
        self.rl_logger.debug('State')
        self.rl_logger.debug(state)
        self.rl_logger.debug('='*60)

        if action == self.prev_action:
            self.same_action_count[ni] += 1
        else:
            self.same_action_count[ni] = 0

        self.prev_action = action
        self.prev_state = state

        return state,action,invalid_actions

    def get_ohe_state(self,s):
        ohe_state = np.zeros((1, self.net_depth + len(s) - 1))
        ohe_state[0, -1] = s[2] * 1.0 / self.filter_bound_vec[int(s[0])]
        ohe_state[0, -2] = s[1]
        ohe_state[0, int(s[0])] = 1.0

        return tuple(ohe_state.flatten())

    def get_ohe_state_ndarray(self,s):
        return np.asarray(self.get_ohe_state(s)).reshape(1,-1)

    def get_ohe_state_history(self,sh):
        ohe_state_hist_list = []
        for hist_item in range(self.state_history_length-1):
            temp_arr = list(self.get_ohe_state(sh[hist_item][0]))
            temp_arr.extend(sh[hist_item][1])
            ohe_state_hist_list.extend(temp_arr)

        ohe_state_hist_list.extend(list(self.get_ohe_state(sh[-1][0])))
        return tuple(ohe_state_hist_list)

    def get_xy_with_experince(self, experience_slice):

        x,y,rewards,sj = None,None,None,None

        for [phi_t,ai,reward,phi_t_plus_1] in experience_slice:
            # phi_t, ai, reward, phi_t_plus_1
            if x is None:
                x = np.asarray(self.get_ohe_state_history(phi_t)).reshape((1,-1))
            else:
                x = np.append(x,np.asarray(self.get_ohe_state_history(phi_t)).reshape((1,-1)),axis=0)

            ohe_a = [1 if self.actions.index(ai)==act else 0 for act in range(len(self.actions))]
            if y is None:
                y = np.asarray(ohe_a).reshape(1,-1)
            else:
                y = np.append(y,np.asarray(ohe_a).reshape(1,-1),axis=0)

            if rewards is None:
                rewards = np.asarray(reward).reshape(1,-1)
            else:
                rewards = np.append(rewards,np.asarray(reward).reshape(1,-1),axis=0)

            if sj is None:
                sj = np.asarray(self.get_ohe_state_history(phi_t_plus_1)).reshape(1,-1)
            else:
                sj = np.append(sj,np.asarray(self.get_ohe_state_history(phi_t_plus_1)).reshape(1,-1),axis=0)

        return x,y,rewards,sj

    def update_policy(self, data):
        # data['states'] => list of states
        # data['actions'] => list of actions
        # data['next_accuracy'] => validation accuracy (unseen)
        # data['prev_accuracy'] => validation accuracy (seen)

        if self.global_time_stamp>0 and len(self.experience)>0 and self.global_time_stamp%self.fit_interval==0:
            self.rl_logger.info('Training the Q Approximator with Experience...')
            self.rl_logger.debug('(Q) Total experience data: %d', len(self.experience))

            if len(self.experience)>self.batch_size:
                exp_indices = np.random.randint(0,len(self.experience),(self.batch_size,))
                self.rl_logger.debug('Experience indices: %s',exp_indices)
                x,y,r,next_state = self.get_xy_with_experince([self.experience[ei] for ei in exp_indices])
            else:
                x,y,r,next_state = self.get_xy_with_experince(self.experience)

            if self.global_time_stamp<5:
                assert np.max(x)<=1.0 and np.max(x)>=-1.0 and np.max(y)<=1.0 and np.max(y)>=-1.0

            self.rl_logger.debug('Summary of Structured Experience data')
            self.rl_logger.debug('\tX:%s',x.shape)
            self.rl_logger.debug('\tY:%s', y.shape)
            self.rl_logger.debug('\tR:%s', r.shape)
            self.rl_logger.debug('\tNextState:%s', next_state.shape)
            if self.target_network is not None:
                pred_q = self.session.run(self.tf_out_target_op,feed_dict={self.tf_state_input:x})
                self.rl_logger.debug('\tPredicted %s:',pred_q.shape)
                target_q = r.flatten() + self.discount_rate * np.max(pred_q,axis=1).flatten()
            else:
                target_q = r

            self.rl_logger.debug('\tTarget Q %s:', target_q.shape)
            self.rl_logger.debug('\tTarget Q Values %s:', target_q[:5])
            assert target_q.size <= self.batch_size

            y = np.multiply(y,target_q.reshape(-1,1))

            # since the state contain layer id, let us make the layer id one-hot encoded
            self.rl_logger.debug('X (shape): %s, Y (shape): %s', x.shape, y.shape)
            self.rl_logger.debug('X: \n%s, Y: \n%s',str(x[:3,:]),str(y[:3]))

            # self.regressor.partial_fit(x, y)
            _ = self.session.run([self.tf_loss_op, self.tf_optimize_op], feed_dict={
                self.tf_state_input: x, self.tf_q_targets: y
            })

            if self.target_network is None or self.global_time_stamp%self.target_update_rate==0:
                self.rl_logger.info('Coppying the Q approximator as the Target Network')
                #self.target_network = self.regressor.partial_fit(x, y)
                if self.local_time_stamp%self.n_conv==0:
                    _ = self.session.run([self.tf_target_update_ops])

            self.clean_experience()

        mean_accuracy = (data['pool_accuracy']-data['prev_pool_accuracy'])/100.0

        si,ai,sj = data['prev_state'],data['prev_action'],data['curr_state']
        self.rl_logger.debug('Si,Ai,Sj: %s,%s,%s',si,ai,sj)
        # if si[2] (layer_depth) ==0 means a pooling operation in CNN
        # we don't do changes to pooling ops
        # so ignore them
        if si is None or si[2]==0:
            return

        new_filter_size = sj[2]
        if ai[0]=='add':
            assert sj[2] == si[2]+ai[1]
            reward = mean_accuracy
        elif ai[0]=='remove':
            assert sj[2] == si[2]-ai[1]
            reward = mean_accuracy - (0.05*(self.filter_bound_vec[si[0]]-new_filter_size) / self.filter_bound_vec[si[0]])
        elif ai[0]=='replace':
            reward = mean_accuracy - (0.05*(self.filter_bound_vec[si[0]]-new_filter_size) / self.filter_bound_vec[si[0]])
        elif ai[0]=='finetune' or ai[0]=='do_nothing':
            new_filter_size = si[2]
            reward = mean_accuracy - (0.01 * (self.filter_bound_vec[si[0]]-new_filter_size) / self.filter_bound_vec[si[0]])
            if ai[0]=='do_nothing' and np.random.random()<0.1:
                reward = -0.5
        else:
            reward = mean_accuracy

        self.reward_logger.info("%d,%.5f",self.local_time_stamp,reward)
        # how the update on state_history looks like
        # t=5 (s2,a2),(s3,a3),(s4,a4)
        # t=6 (s3,a3),(s4,a4),(s5,a5)
        # add previous action (the action we know the reward for) i.e not the current action
        # as a one-hot vector

        #phi_t (s_t-3,a_t-3),(s_t-2,a_t-2),(s_t-1,a_t-1),(s_t,a_t)
        phi_t = list(self.current_state_history)
        phi_t.append([si])

        #current_state_h
        self.current_state_history.append([si])
        self.current_state_history[-1].append([1 if self.actions.index(ai)==act else 0 for act in range(len(self.actions))])

        if len(self.current_state_history)>self.state_history_length:
            del self.current_state_history[0]
            assert len(self.current_state_history) == self.state_history_length

        phi_t_plus_1 = list(self.current_state_history)
        phi_t_plus_1.append([sj])

        if not self.state_history_dumped and np.random.random()<0.15:
            self.state_history_collector.append(phi_t_plus_1)
            self.rl_logger.debug('State added: size %d\n',len(self.state_history_collector))

        if len(self.state_history_collector)==128:
            self.rl_logger.debug('Persisting Random State Collection')
            with open(self.persit_dir + os.sep + 'QMetricStates.pickle', 'wb') as f:
                pickle.dump(self.state_history_collector, f, pickle.HIGHEST_PROTOCOL)
                self.state_history_collector = []
                self.state_history_dumped = True

        # update experience
        if len(phi_t)>=self.state_history_length+1:
            self.experience.append([phi_t,ai,reward,phi_t_plus_1])
            for invalid_a in data['invalid_actions']:
                self.rl_logger.debug('Adding the invalid action %s to experience',invalid_a)
                self.experience.append([phi_t,invalid_a,-0.5,phi_t_plus_1])

            if self.global_time_stamp<3:
                self.rl_logger.debug('Latest Experience: ')
                self.rl_logger.debug('\t%s\n',self.experience[-1])

        self.rl_logger.debug('Update Summary ')
        self.rl_logger.debug('\tState: %s',si)
        self.rl_logger.debug('\tAction: %s',ai)
        self.rl_logger.debug('\tReward: %.3f',reward)

        self.local_time_stamp += 1
        if self.local_time_stamp%self.n_conv == 0:
            self.global_time_stamp += 1

        self.rl_logger.debug('Global/Local time step: %d/%d\n',self.global_time_stamp,self.local_time_stamp)


    def get_Q(self):
        raise NotImplementedError
