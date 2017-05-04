import matplotlib.pyplot as plt
import numpy as np
import os


log_dir = 'logs_to_plot' + os.sep + 'cifar-10-rl-plots'

fontsize_legend = 18
fontsize_title = 18
fontsize_ticks = 16
fontsize_label = 16
fontsize_legend_small = 14
fontsize_annotations = 16

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
ax2.set_xlabel('Time (t)',fontsize=fontsize_label)
ax2.set_ylabel('$Q_{rand}^{eval}$ (Average of Maximum Q on $S_{rand}$)',fontsize=fontsize_label)
ax2.set_title('Behavior of $Q_{rand}^{eval}$ over time $t$ (During $\pi_{greedy}$)',fontsize=fontsize_title)
ax2.tick_params(axis='both', which='major', labelsize=fontsize_ticks)

# ======================================================================
#                       Q Per Action Plot
# ======================================================================

file_name = os.path.join(log_dir, 'reward.log')
rew_data = {}
decay = 0.
prev_rolling_rew = 0
with open(file_name,'r') as f:
    for line in f:
        if line.startswith('#'):
            continue
        else:
            line_tokens = line.split(':')
            # ===========================================================================
            # Be mindful about the -1 we put for invalid actions in the reward.log
            # ===========================================================================
            if float(line_tokens[3])==-1.0:
                continue
            prev_rolling_rew = (1-decay) * float(line_tokens[3]) + decay*prev_rolling_rew
            rew_data[int(line_tokens[0])*50] = prev_rolling_rew

file_name = os.path.join(log_dir, 'predicted_q.log')
q_act_y_data = [[] for _ in range(10)]
q_act_x_axis = []
q_act_labels = ['(1,Remove,4)','(2,Remove,4)','(3,Remove,4)','(4,Remove,4)','(1,Add,8)','(2,Add,8)','(3,Add,8)','(4,Add,8)','DoNothing','Finetune']

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
ax3.plot(q_act_x_axis,plot_rew_data,label='Accuracy$_{valid}$',color='black',linestyle=':',markersize=4) #marker argument (for marker style)
ax3.set_ylim([-60,60])
ax3.legend(fontsize=12,loc=1)
for i in [0,3,4,7,9]:
    ax1.plot(q_act_x_axis, q_act_y_data[i],label=q_act_labels[i],linewidth=2)
ax1.set_xlabel('Time (t)',fontsize=fontsize_label)
ax1.set_ylabel('$Q(s_t,a)$',fontsize=fontsize_label)
ax3.set_ylabel('Accuracy$_{valid}$(%)',fontsize=fontsize_label)
ax1.set_title('Behavior of Q value for each action w.r.t $S_t$',fontsize=fontsize_title)
ax1.legend(fontsize=fontsize_legend_small,loc=3)
ax1.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
ax3.tick_params(axis='both', which='major', labelsize=fontsize_ticks)

# zero line
ax1.plot(q_act_x_axis,[0 for _ in range(len(q_act_x_axis))],color='gray',linestyle=':')
# deviding
ax1.annotate('$\mathbf{\pi_{explore}}$', xy=(8000,0.009),xytext=(8000, 0.009),fontsize=fontsize_annotations)
ax1.annotate('$\mathbf{\pi_{greedy}}$', xy=(10400,0.009),xytext=(10400, 0.009),fontsize=fontsize_annotations)
ax1.annotate('1', xy=(2000,0.009),xytext=(2000, 0.009),fontsize=fontsize_annotations)
ax1.annotate('2', xy=(13000,0.006),xytext=(13000, 0.006),fontsize=fontsize_annotations)

ax3.plot((10000,10000),(-60,60),'k--')
fig.subplots_adjust(wspace=0.25,hspace=0.15,bottom=0.1,top=0.92,right=0.97,left=0.07)
plt.show()

