__author__ = 'Thushan Ganegedara'

import os
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np

constructor_dir = 'constructor_rl'

Q = {}
for i in range(10,21,10):
    with open(constructor_dir+os.sep+'Q_'+str(i)+'.pickle','rb') as f:
        Q[i] = pickle.load(f)

if __name__ == '__main__':

    # find recorded max depth
    max_depth = 0
    for ep,q in Q.items():
        for state in q.keys():
            depth = state[0]
            if depth>max_depth:
                max_depth = depth
    print('Found max depth: %d'%max_depth)
    fig, axarr = plt.subplots(1,1)
    ax1 = axarr
    #ax2 = axarr[1]
    #ax3 = axarr[2]

    x_axis = np.arange(max_depth)
    q_10 =  Q[20]
    # {action => {layer_depth => list_of_q_values}}
    list_q_for_action_by_depth = {}
    for state,action_dict in q_10.items():
        # depth for action in a given state
        layer_depth = state[0]+1
        for a,q in action_dict.items():
            # map doesnt have action a in it
            if a not in list_q_for_action_by_depth:
                list_q_for_action_by_depth[a]= {layer_depth:[q]}
            # map has action a
            else:
                if layer_depth not in list_q_for_action_by_depth[a]:
                    list_q_for_action_by_depth[a][layer_depth] = [q]
                else:
                    list_q_for_action_by_depth[a][layer_depth].append(q)

    for a,depth_dict in list_q_for_action_by_depth.items():
        print('Action %s'%str(a))
        print('\t',depth_dict)

    # average out the q values for given action and given depth
    mean_q_for_action_by_depth = {}
    for a,depth_dict in list_q_for_action_by_depth.items():
        mean_q_for_action_by_depth[a]={}
        for depth,list_q in depth_dict.items():
            if np.sum(list_q)>0:
                mean_q_for_action_by_depth[a][depth] = np.sum(list_q)/np.count_nonzero(list_q)
            else:
                mean_q_for_action_by_depth[a][depth] = 0.0

    order_q_for_action_by_depth = {}
    for a,depth_dict in mean_q_for_action_by_depth.items():
        order_q_for_action_by_depth[a] = []
        for d_i in range(max_depth):
            if d_i in depth_dict:
                order_q_for_action_by_depth[a].append(depth_dict[d_i])
            else:
                order_q_for_action_by_depth[a].append(0.0)

    print()
    for a,q_list in order_q_for_action_by_depth.items():
        if np.random.random()>0.67:
            ax1.plot(x_axis, q_list, label=a)
        elif np.random.random()<0.33:
            ax1.plot(x_axis, q_list,'--', label=a)
        else:
            ax1.plot(x_axis, q_list,'-*', label=a)
        print('Action %s'%str(a))
        print('\t',q_list)

    ax1.legend(fontsize=12)
    #fig.subplots_adjust(wspace=.2,hspace=0.4,bottom=0.2)
    plt.tight_layout()
    plt.show()

