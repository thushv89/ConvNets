import matplotlib.pyplot as plt
import numpy as np
import os

log_dir = 'logs_to_plot' + os.sep + 'cifar-10-time-struct'
cnn_layers = 4
cnn_layer_ids = [0,1,2,4]
cnn_layer_legends = ['Layer 1','Layer 2','Layer 3','Layer 4']
fig, axarr = plt.subplots(1,3)
ax1 = axarr[0]
ax2 = axarr[1]
ax3 = axarr[2]

fontsize_legend = 14
fontsize_title = 16
fontsize_ticks = 14
fontsize_label = 14
fontsize_legend_small = 10
# ======================================================================
#                       Structure growth AdaCNN
# ======================================================================
file_name = os.path.join(log_dir, 'cnn_structure.log')
struct_y_data = [[] for _ in range(cnn_layers)]
struct_x_axis = []

with open(file_name,'r') as f:
    for line in f:
        if line.startswith('#'):
            continue
        else:
            line_tokens = line.split(':')
            if int(line_tokens[0])<10000:
                continue
            struct_x_axis.append(int(line_tokens[0]))
            cnn_struct = line_tokens[1][1:-1].split(',')
            for idx,id in enumerate(cnn_layer_ids):
                struct_y_data[idx].append(cnn_struct[id])

for id in range(cnn_layers):
    ax2.plot(struct_x_axis,struct_y_data[id],label=cnn_layer_legends[id])
ax2.set_xlabel('Time ($t$)',fontsize=fontsize_label)
ax2.set_ylabel('Number of kernels',fontsize=fontsize_label)
ax2.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
ax2.set_title('Structure adaptation over time ($t$)',fontsize=fontsize_title)
ax2.legend(fontsize=fontsize_legend,loc=2)

# ======================================================================
#           Time consumption Rigid-CNN, Rigid-CNN-B, AdaCNN
# ======================================================================
time_filenames = ['time_rigid.log','time_rigidb.log','time_adacnn.log']
num_algo = 3
time_train_data = [[] for _ in range(num_algo)]
time_full_data = [[] for _ in range(num_algo)]


for fi,filename in enumerate(time_filenames):
    file_name = os.path.join(log_dir, filename)
    with open(file_name,'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            else:
                line_tokens = line.split(',')
                if int(line_tokens[0])<10:
                    continue
                time_train_data[fi].append(float(line_tokens[2]))
                time_full_data[fi].append(float(line_tokens[1]))

time_train_mean = [np.mean(time_train_data[i]) for i in range(num_algo)]
time_train_std = [np.std(time_train_data[i]) for i in range(num_algo)]

time_full_mean = [np.mean(time_full_data[i]) for i in range(num_algo)]
time_full_std = [np.std(time_full_data[i]) for i in range(num_algo)]

print(time_train_mean)
print(time_train_std)
print(time_full_mean)
print(time_full_std)

N=1
indices = np.arange(N)
width = 0.1

#ax1.bar(indices,(time_train_mean[0]),width, color='r', yerr=(time_train_std[0]))
#ax1.bar(indices+width,(time_train_mean[1]),width, color='y', yerr=(time_train_std[1]))
#ax1.bar(indices+2*width,(time_train_mean[2]),width, color='b', yerr=(time_train_std[2]))

ax1.bar(indices+0*width+0.1,(time_full_mean[0]),width, color='r', yerr=(time_full_std[0]))
ax1.bar(indices+1*width+0.1,(time_full_mean[1]),width, color='y', yerr=(time_full_std[1]))
ax1.bar(indices+2*width+0.1,(time_full_mean[2]),width, color='b', yerr=(time_full_std[2]))

ax1.set_ylim([0,3])
ax1.set_ylabel('Time per Iteration (s)',fontsize=fontsize_label)
ax1.set_xticks([])
ax1.legend(('Rigid-CNN', 'Rigid-CNN-B','AdaCNN'),loc=1,fontsize=fontsize_legend)
ax1.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
ax1.set_title('Time consumption per Iteration',fontsize=fontsize_title)

# ======================================================================
#                       Data distribution
# ======================================================================
file_name = os.path.join(log_dir, 'class_distribution_cifar10.log')
num_classes = 10
dist_y_data = [[] for _ in range(num_classes)]
dist_x_axis = []
dist_legends = ['Class %d'%i for i in range(num_classes)]
with open(file_name,'r') as f:
    for line in f:
        if line.startswith('#'):
            continue
        else:
            line_tokens = line.split(',')
            if int(line_tokens[0])>100:
                break
            dist_x_axis.append(int(line_tokens[0])*10)
            for dist_i in range(num_classes):
                dist_y_data[dist_i].append(float(line_tokens[dist_i+1]))

for id in range(num_classes):
    ax3.plot(dist_x_axis,dist_y_data[id],label=dist_legends[id])

ax3.set_xlabel('Time ($t$)',fontsize=fontsize_label)
ax3.set_ylabel('Proportion of instances of each class',fontsize=fontsize_label)
ax3.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
ax3.set_title('Class distribution over time ($t$)',fontsize=fontsize_title)
ax3.legend(fontsize=fontsize_legend_small,loc=1)

plt.show()