import matplotlib.pyplot as plt
import numpy as np
import os


log_dir = 'logs_to_plot' + os.sep + 'cifar-10-rl-plots'

fig, axarr = plt.subplots(1,2)
ax1 = axarr[0]
ax2 = axarr[1]

# ======================================================================
#                       Q Metric Plot
# ======================================================================
file_name = os.path.join(log_dir, 'QMetric.log')
q_y_data = []
q_x_axis = []

decay = 0.9
with open(file_name,'r') as f:
    reward_data = []
    running_reward = 0.0
    for line in f:
        if line.startswith('#'):
            continue
        else:
            line_tokens = line.split(',')
            if int(line_tokens[0])<10000:
                continue
            q_x_axis.append(int(line_tokens[0]))
            q_y_data.append(float(line_tokens[-1]))

ax2.plot(q_x_axis, q_y_data)
ax2.set_xlabel('Time (t)')
ax2.set_ylabel('Average of Maximum Q on $S_{rand}$')
ax2.set_title('Behavior of $Q_{rand}^{eval}$ over time $t$')


# ======================================================================
#                       Q Per Action Plot
# ======================================================================

file_name = os.path.join(log_dir, 'reward.log')
rew_data = {}
decay = 0.8
prev_rolling_rew = 0
with open(file_name,'r') as f:
    for line in f:
        if line.startswith('#'):
            continue
        else:
            line_tokens = line.split(':')
            prev_rolling_rew = (1-decay) * float(line_tokens[3]) + decay*prev_rolling_rew
            rew_data[int(line_tokens[0])*50] = prev_rolling_rew

file_name = os.path.join(log_dir, 'predicted_q.log')
q_act_y_data = [[] for _ in range(10)]
q_act_x_axis = []
q_act_labels = ['Remove,1','Remove,2','Remove,3','Remove,4','Add,1','Add,2','Add,3','Add,4','DoNothing','Finetune']

with open(file_name,'r') as f:
    for line in f:
        if line.startswith('#'):
            continue
        else:
            line_tokens = line.split(',')
            q_act_x_axis.append(int(line_tokens[0])*50)
            for l_i in range(10):
                q_act_y_data[l_i].append(float(line_tokens[l_i+1]))

ax3 = ax1.twinx()
plot_rew_data = []
for x in q_act_x_axis:
    plot_rew_data.append(rew_data[x])
ax3.plot(q_act_x_axis,plot_rew_data,label='Accuracy of $B_{pool}$',color='black',linestyle='--')
ax3.set_ylim([-60,60])
ax3.legend(fontsize=12,loc=1)
for i in [0,3,4,7,9]:
    ax1.plot(q_act_x_axis, q_act_y_data[i],label=q_act_labels[i])
ax1.set_xlabel('Time (t)')
ax1.set_ylabel('$Q(s_t,a)$')
ax3.set_ylabel('Accuracy$_{valid}$(%)')
ax1.set_title('Behavior of Q value for each action w.r.t $S_t$')
ax1.legend(fontsize=12,loc=3)


# zero line
ax1.plot(q_act_x_axis,[0 for _ in range(len(q_act_x_axis))],color='gray',linestyle=':')

plt.show()

