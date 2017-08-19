import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Rectangle

log_dir = 'logs_to_plot_icml' + os.sep + 'sup-pool'

fontsize_legend = 18
fontsize_title = 22
fontsize_ticks = 16
fontsize_label = 18
fontsize_legend_small = 14
fontsize_annotations = 18
colors = ['red','blue','green','yellow','black','cyan','magenta','gray','orange','pink']
fig, axarr = plt.subplots(1,2)
ax1 = axarr[0]
ax2 = axarr[1]

# ======================================================================
#                       Q Metric Plot
# ======================================================================

file_name = os.path.join(log_dir, 'class_distribution_logcifar-10.log')
s10_train_y_data = [[] for _ in range(10)]
s10_train_x_axis = []

x_val = 0
rolling_s10_train = [0 for _ in range(10)]
decay = 0.95
with open(file_name,'r') as f:

    for line in f:
        if line.startswith('#'):
            continue
        else:

            line_tokens = line.split(',')
            if int(line_tokens[0])%5!=0:
                continue
            else:
                s10_train_x_axis.append(int(line_tokens[0])*10)
                for t_i,token in enumerate(line_tokens[1:-1]):
                    #rolling_s10_train[t_i] = (1-decay) * float(token) + decay * rolling_s10_train[t_i]
                    prev_value = s10_train_y_data[t_i - 1][-1] if t_i > 0 else 0
                    s10_train_y_data[t_i].append(prev_value + float(token))


#for i in range(10):
#    print(s10_train_y_data[:][i])

'''for r_i in range(len(s10_train_y_data[0])):
    #print('before')
    #print(s10_train_y_data[:][r_i])
    # normalization step
    # get sum
    sum_s10 = 0
    for t_i in range(10):
        sum_s10 += s10_train_y_data[t_i][r_i]

    if sum_s10!=1.0:
        # normalize
        for t_i in range(10):
            s10_train_y_data[t_i][r_i] = s10_train_y_data[t_i][r_i] * 1.0 / sum_s10

        # cumulate
        cum_value = 0
        for t_i in range(10):
            s10_train_y_data[t_i][r_i] += cum_value
            cum_value = s10_train_y_data[t_i][r_i]

    #print('after')
    ##print(s10_train_y_data[:][r_i])'''

file_name = os.path.join(log_dir, 'pool_distribution.log')
s10_pool_y_data = [[] for _ in range(10)]
s10_pool_x_axis = []

x_val = 0
with open(file_name,'r') as f:
    for line in f:
        if line.startswith('#'):
            continue
        else:
            if x_val*500>=10000:
                line_tokens = line.split(',')[:-1]
                for t_i,token in enumerate(line_tokens):
                    print(t_i)
                    prev_value = s10_pool_y_data[t_i - 1][-1] if t_i > 0 else 0
                    s10_pool_y_data[t_i].append(prev_value+float(token))

                s10_pool_x_axis.append(x_val*500)
            x_val += 1

class_id = 9
for c_i,col in enumerate(reversed(s10_train_y_data)):
    ax1.plot(s10_train_x_axis, col,linewidth=3,color=colors[c_i])
    ax1.fill_between(s10_train_x_axis, 0, col, facecolor=colors[c_i])
    class_id -= 1

ax1.plot([5000,5000],[0,1],'k--')
ax1.set_xlim([0,s10_train_x_axis[-1]])
ax1.set_ylim([0,1.0])
ax1.set_xlabel('Time (t)',fontsize=fontsize_label)
ax1.set_title('Training Data Distribution (1 Epoch)',fontsize=fontsize_title,y=1.05)
ax1.set_ylabel('Proportion of instances of each class',fontsize=fontsize_label)
ax1.annotate('Class split', xy=(5000, 0.8),xytext=(3000,0.92),fontsize=fontsize_annotations,arrowprops=dict(facecolor='black', shrink=0.05),bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="black", lw=2))

s10_plots=[]
label_list = []
class_id = 9
for c_i,col in enumerate(reversed(s10_pool_y_data)):
    label_list.append('Class '+str(class_id))
    col_plot= ax2.plot(s10_pool_x_axis, col, linewidth=3,color=colors[c_i])
    ax2.fill_between(s10_pool_x_axis, 0, col, facecolor=colors[c_i])
    s10_plots.append(col_plot)
    class_id -= 1
#s10_plots = ax2.stackplot(s10_pool_x_axis, s10_pool_y_data,linewidth=1)
proxy_rects = [Rectangle((0, 0), 1, 1, fc=colors[c_i]) for c_i,pc in enumerate(s10_plots)]

ax2.set_xlim([10000,s10_pool_x_axis[-1]])
ax2.set_ylim([0,1.0])
ax2.set_xlabel('Time (t)',fontsize=fontsize_label)
ax2.set_title('Data distribution of $B_{pool}$ (Last 2 Epochs)',fontsize=fontsize_title,y=1.05)
ax2.set_ylabel('Proportion of instances of each class',fontsize=fontsize_label)
ax2.legend(proxy_rects,label_list,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=fontsize_legend)
ax2.plot((20000,20000),(0,1.0),'k--',linewidth=2)
ax2.annotate('$2^{nd}$ Epoch', xy=(17000,0.025),xytext=(17000, 0.05),fontsize=fontsize_annotations)
ax2.annotate('$3^{rd}$ Epoch', xy=(21000,0.025),xytext=(21000, 0.05),fontsize=fontsize_annotations)

fig.subplots_adjust(wspace=0.2,hspace=0.15,bottom=0.1,top=0.88,right=0.9,left=0.04)
plt.show()

#ax2.set_xlabel('Time (t)',fontsize=fontsize_label)
#ax2.set_ylabel('$Q_{rand}^{eval}$ (Average of Maximum Q on $S_{rand}$)',fontsize=fontsize_label)
#ax2.set_title('Behavior of $Q_{rand}^{eval}$ over time $t$ (During $\pi_{greedy}$)',fontsize=fontsize_title,y=1.05)
#ax2.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
#ax2.set_xlim([10000,20500])
#ax2.legend(fontsize=fontsize_legend)
print(s10_pool_x_axis[-1])