import matplotlib.pyplot as plt
import numpy as np
import os

log_dir = 'logs_to_plot' + os.sep + 'cifar10-cifar100-struct-dist'
cifar10_cnn_layers = 3
cifar10_cnn_layer_ids = [0,2,4]
cifar10_ns_cnn_layer_legends = ['Layer 1 (NS)','Layer 3 (NS)','Layer 4 (NS)']
cifar10_s_cnn_layer_legends = ['Layer 1 (S)','Layer 3 (S)','Layer 4 (S)']
cifar10_layer_colors = ['red','blue','green','cyan']

cifar100_cnn_layers = 3
cifar100_cnn_layer_ids = [1,4,5]
cifar100_ns_cnn_layer_legends = ['Layer 2 (NS)','Layer 4 (NS)','Layer 5 (NS)']
cifar100_s_cnn_layer_legends = ['Layer 2 (S)','Layer 4 (S)','Layer 5 (S)']
cifar100_layer_colors = ['red','blue','green','cyan','magenta']

fig, axarr = plt.subplots(1,3)
ax1 = axarr[0]
ax2 = axarr[1]
ax3 = axarr[2]

fontsize_legend = 14
fontsize_title = 20
fontsize_ticks = 14
fontsize_label = 14
fontsize_legend_small = 10
# ======================================================================
#                       Structure growth AdaCNN (CIFAR 10)
# ======================================================================
cifar10_ns_file_name = os.path.join(log_dir, 'cifar10-nonstationary-cnn_structure.log')
cifar10_s_file_name = os.path.join(log_dir, 'cifar10-stationary-cnn_structure.log')
cifar10_ns_struct_y_data = [[] for _ in range(cifar10_cnn_layers)]
cifar10_s_struct_y_data = [[] for _ in range(cifar10_cnn_layers)]
cifar10_ns_struct_x_axis = []
cifar10_s_struct_x_axis = []

with open(cifar10_ns_file_name,'r') as f:
    for line in f:
        if line.startswith('#'):
            continue
        else:
            line_tokens = line.split(':')
            if int(line_tokens[0])<10000:
                continue
            cifar10_ns_struct_x_axis.append(int(line_tokens[0]))
            cnn_struct = line_tokens[1][1:-1].split(',')
            for idx,id in enumerate(cifar10_cnn_layer_ids):
                cifar10_ns_struct_y_data[idx].append(cnn_struct[id])

with open(cifar10_s_file_name,'r') as f:
    for line in f:
        if line.startswith('#'):
            continue
        else:
            line_tokens = line.split(':')
            if int(line_tokens[0])<10000:
                continue
            cifar10_s_struct_x_axis.append(int(line_tokens[0]))
            cnn_struct = line_tokens[1][1:-1].split(',')
            for idx,id in enumerate(cifar10_cnn_layer_ids):
                cifar10_s_struct_y_data[idx].append(cnn_struct[id])

for id in range(cifar10_cnn_layers):
    ax1.plot(cifar10_ns_struct_x_axis,cifar10_ns_struct_y_data[id],label=cifar10_ns_cnn_layer_legends[id],color=cifar10_layer_colors[id],linewidth=2)
for id in range(cifar10_cnn_layers):
    ax1.plot(cifar10_s_struct_x_axis, cifar10_s_struct_y_data[id], label=cifar10_s_cnn_layer_legends[id],linestyle='--',color=cifar10_layer_colors[id],linewidth=2)

ax1.set_xlabel('Time ($t$)',fontsize=fontsize_label)
ax1.set_ylabel('Number of kernels',fontsize=fontsize_label)
ax1.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
ax1.set_title('Structure adaptation for CIFAR10 over time ($t$)',fontsize=fontsize_title,y=1.05)
ax1.legend(fontsize=fontsize_legend,loc=2)

# ======================================================================
#                       Structure growth AdaCNN (CIFAR 100)
# ======================================================================
cifar100_ns_file_name = os.path.join(log_dir, 'cifar100-nonstationary-cnn_structure.log')
cifar100_s_file_name = os.path.join(log_dir, 'cifar100-stationary-cnn_structure.log')
cifar100_ns_struct_y_data = [[] for _ in range(cifar100_cnn_layers)]
cifar100_s_struct_y_data = [[] for _ in range(cifar100_cnn_layers)]
cifar100_ns_struct_x_axis = []
cifar100_s_struct_x_axis = []

with open(cifar100_ns_file_name,'r') as f:
    for line in f:
        if line.startswith('#'):
            continue
        else:
            line_tokens = line.split(':')
            if int(line_tokens[0])<10000:
                continue
            cifar100_ns_struct_x_axis.append(int(line_tokens[0]))
            cnn_struct = line_tokens[1][1:-1].split(',')
            for idx,id in enumerate(cifar100_cnn_layer_ids):
                cifar100_ns_struct_y_data[idx].append(cnn_struct[id])

with open(cifar100_s_file_name,'r') as f:
    for line in f:
        if line.startswith('#'):
            continue
        else:
            line_tokens = line.split(':')
            if int(line_tokens[0])<10000:
                continue
            cifar100_s_struct_x_axis.append(int(line_tokens[0]))
            cnn_struct = line_tokens[1][1:-1].split(',')
            for idx,id in enumerate(cifar100_cnn_layer_ids):
                cifar100_s_struct_y_data[idx].append(cnn_struct[id])

for id in range(cifar100_cnn_layers):
    ax2.plot(cifar100_ns_struct_x_axis,cifar100_ns_struct_y_data[id],label=cifar100_ns_cnn_layer_legends[id],color=cifar100_layer_colors[id],linewidth=2)
for id in range(cifar100_cnn_layers):
    ax2.plot(cifar100_s_struct_x_axis, cifar100_s_struct_y_data[id], label=cifar100_s_cnn_layer_legends[id],linestyle='--',color=cifar100_layer_colors[id],linewidth=2)

ax2.set_xlabel('Time ($t$)',fontsize=fontsize_label)
ax2.set_ylabel('Number of kernels',fontsize=fontsize_label)
ax2.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
ax2.set_title('Structure adaptation for CIFAR100 over time ($t$)',fontsize=fontsize_title,y=1.05)
ax2.legend(fontsize=fontsize_legend,loc=2)


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
ax3.set_title('Class distribution over time ($t$)',fontsize=fontsize_title,y=1.05)
ax3.legend(fontsize=fontsize_legend_small,loc=1)

fig.subplots_adjust(wspace=0.15,hspace=0.15,bottom=0.1,top=0.88,right=0.97,left=0.07)

plt.show()