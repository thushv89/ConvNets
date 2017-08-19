import matplotlib.pyplot as plt
import numpy as np
import os


log_dir = 'logs_to_plot' + os.sep + 'cifar-10-rl-plots'

fontsize_legend = 18
fontsize_title = 20
fontsize_ticks = 16
fontsize_label = 16
fontsize_legend_small = 14
fontsize_annotations = 18

fig, axarr = plt.subplots(1,3)
ax1 = axarr[0]
ax4 = axarr[1]
ax2 = axarr[2]

# ======================================================================
#                       Q Metric Plot
# ======================================================================
file_name = os.path.join(log_dir, 'QMetric-nonstationary.log')
ns_q_y_data = []
ns_q_x_axis = []

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
            ns_q_x_axis.append(int(line_tokens[0]))
            ns_q_y_data.append(float(line_tokens[-1]))

file_name = os.path.join(log_dir, 'QMetric-stationary.log')
s_q_y_data = []
s_q_x_axis = []

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
            s_q_x_axis.append(int(line_tokens[0]))
            s_q_y_data.append(float(line_tokens[-1]))

ax2.plot(ns_q_x_axis, ns_q_y_data,label='Non-stationary',linewidth=2)
ax2.plot(s_q_x_axis, s_q_y_data,label='Stationary',linewidth=2)
ax2.set_xlabel('Time (t)',fontsize=fontsize_label)
ax2.set_ylabel('$Q_{rand}^{eval}$ (Average of Maximum Q on $S_{rand}$)',fontsize=fontsize_label)
ax2.set_title('Behavior of $Q_{rand}^{eval}$ over time $t$ (During $\pi_{greedy}$)',fontsize=fontsize_title,y=1.05)
ax2.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
ax2.set_xlim([10000,20500])
ax2.legend(fontsize=fontsize_legend)
ax2.annotate('Adaptations\n Stopped', xy=(16500,0.0048),xytext=(17000, 0.006),fontsize=fontsize_annotations,arrowprops=dict(facecolor='black', shrink=0.05))
ax2.annotate('', xy=(20500,0.004),xytext=(19000, 0.006),fontsize=fontsize_annotations,arrowprops=dict(facecolor='black', shrink=0.05))
# ======================================================================
#                       Q Per Action Plot
# ======================================================================

file_name = os.path.join(log_dir, 'reward-nonstationary.log')
ns_index_map = {}
ns_rew_data = {}
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
            ns_rew_data[int(line_tokens[1])] = prev_rolling_rew
            ns_index_map[int(line_tokens[0])] = int(line_tokens[1])

file_name = os.path.join(log_dir, 'reward-stationary.log')
s_index_map = {}
s_rew_data = {}
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
            s_rew_data[int(line_tokens[1])] = prev_rolling_rew
            s_index_map[int(line_tokens[0])] = int(line_tokens[1])


file_name = os.path.join(log_dir, 'predicted_q-nonstationary.log')
q_act_y_data = [[] for _ in range(10)]
q_act_x_axis = []
q_act_labels = ['(1,Remove,4)','(2,Remove,4)','(3,Remove,4)','(4,Remove,4)','(1,Add,8)','(2,Add,8)','(3,Add,8)','(4,Add,8)','DoNothing','Finetune']

with open(file_name,'r') as f:
    for line in f:
        if line.startswith('#'):
            continue
        else:
            line_tokens = line.split(',')
            q_act_x_axis.append(ns_index_map[int(line_tokens[0])])
            for l_i in range(10):
                q_act_y_data[l_i].append(float(line_tokens[l_i+1]))

ax3 = ax1.twinx()
plot_rew_data = []

for x in q_act_x_axis:
    plot_rew_data.append(ns_rew_data[x])

ax3.plot(q_act_x_axis,plot_rew_data,label='Accuracy$_{valid}$',color='black',linestyle=':',linewidth=2,markersize=4) #marker argument (for marker style)
ax3.set_ylim([-60,60])
ax3.set_xlim([0,20500])
ax3.legend(fontsize=12,loc=1)
for i in [0,3,4,7,9]:
    ax1.plot(q_act_x_axis, q_act_y_data[i],label=q_act_labels[i],linewidth=2)
ax1.set_xlabel('Time (t)',fontsize=fontsize_label)
ax1.set_ylabel('$Q(s_t,a)$',fontsize=fontsize_label)
ax1.set_ylim([-0.01,0.01])
ax3.set_ylabel('Accuracy$_{valid}$(%)',fontsize=fontsize_label)
ax1.set_title('Behavior of Q value w.r.t $S_t$ (Non-Stationary)',fontsize=fontsize_title,y=1.05)
ax1.legend(fontsize=fontsize_legend_small,loc=3)
ax1.set_xlim([0,20500])
ax1.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
ax3.tick_params(axis='both', which='major', labelsize=fontsize_ticks)

# zero line
ax1.plot(q_act_x_axis,[0 for _ in range(len(q_act_x_axis))],color='gray',linestyle=':')
# deviding
ax1.annotate('$\mathbf{\pi_{explore}}$', xy=(6600,0.009),xytext=(7500, 0.009),fontsize=fontsize_annotations)
ax1.annotate('$\mathbf{\pi_{greedy}}$', xy=(10900,0.009),xytext=(10400, 0.009),fontsize=fontsize_annotations)
#ax1.annotate('1', xy=(2000,0.009),xytext=(2000, 0.009),fontsize=fontsize_annotations)
#ax1.annotate('2', xy=(13000,0.006),xytext=(13000, 0.006),fontsize=fontsize_annotations)

ax3.plot((10000,10000),(-60,60),'k--')


file_name = os.path.join(log_dir, 'predicted_q-stationary.log')
s_q_act_y_data = [[] for _ in range(10)]
s_q_act_x_axis = []
s_q_act_labels = ['(1,Remove,4)','(2,Remove,4)','(3,Remove,4)','(4,Remove,4)','(1,Add,8)','(2,Add,8)','(3,Add,8)','(4,Add,8)','DoNothing','Finetune']

with open(file_name,'r') as f:
    for line in f:
        if line.startswith('#'):
            continue
        else:
            line_tokens = line.split(',')
            s_q_act_x_axis.append(ns_index_map[int(line_tokens[0])])
            for l_i in range(10):
                s_q_act_y_data[l_i].append(float(line_tokens[l_i+1]))

ax5 = ax4.twinx()
s_plot_rew_data = []

for x in s_q_act_x_axis:
    s_plot_rew_data.append(s_rew_data[x])

ax5.plot(s_q_act_x_axis,s_plot_rew_data,label='Accuracy$_{valid}$',color='black',linestyle=':',linewidth=2,markersize=4) #marker argument (for marker style)
ax5.set_ylim([-80,80])
ax5.set_xlim([0,20500])
ax5.legend(fontsize=12,loc=2)
for i in [0,3,4,7,9]:
    ax4.plot(s_q_act_x_axis, s_q_act_y_data[i],label=s_q_act_labels[i],linewidth=2)
ax4.set_xlabel('Time (t)',fontsize=fontsize_label)
ax4.set_ylabel('$Q(s_t,a)$',fontsize=fontsize_label)
ax4.set_ylim([-0.03,0.03])
ax5.set_ylabel('Accuracy$_{valid}$(%)',fontsize=fontsize_label)
ax4.set_title('Behavior of Q value w.r.t $S_t$ (Stationary)',fontsize=fontsize_title,y=1.05)
ax4.legend(fontsize=fontsize_legend_small,loc=3)
ax4.set_xlim([0,17000])
ax4.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
ax5.tick_params(axis='both', which='major', labelsize=fontsize_ticks)

# zero line
ax5.plot(q_act_x_axis,[0 for _ in range(len(q_act_x_axis))],color='gray',linestyle=':')
# deviding
ax4.annotate('$\mathbf{\pi_{explore}}$', xy=(7000,-0.025),xytext=(8000, -0.025),fontsize=fontsize_annotations)
ax4.annotate('$\mathbf{\pi_{greedy}}$', xy=(10500,-0.025),xytext=(10200, -0.025),fontsize=fontsize_annotations)
#ax4.annotate('1', xy=(2000,0.009),xytext=(2000, 0.009),fontsize=fontsize_annotations)
#ax4.annotate('2', xy=(13000,0.006),xytext=(13000, 0.006),fontsize=fontsize_annotations)

ax5.plot((10000,10000),(-80,80),'k--')

ax1.annotate('(a)', xy=(9500,-0.0),xytext=(9700, -0.0143),fontsize=fontsize_annotations)
ax4.annotate('(b)', xy=(8200,-0.0),xytext=(8200, -0.043),fontsize=fontsize_annotations)
ax2.annotate('(c)', xy=(15500,0.002),xytext=(15500, -0.0002),fontsize=fontsize_annotations)

fig.subplots_adjust(wspace=0.35,hspace=0.15,bottom=0.2,top=0.88,right=0.99,left=0.04)
plt.show()

