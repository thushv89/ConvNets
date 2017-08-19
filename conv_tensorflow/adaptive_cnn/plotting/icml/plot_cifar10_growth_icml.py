import matplotlib.pyplot as plt
import numpy as np
import os

log_dir = 'logs_to_plot_icml' + os.sep + 'cifar10-growth'
cifar10_cnn_layers = 3
cifar10_cnn_layer_ids = [0,1,4]
cifar10_ns_cnn_layer_legends = ['Layer 1','Layer 2','Layer 4']
cifar10_layer_colors = ['red','blue','green','cyan']

cifar100_cnn_layers = 3
cifar100_cnn_layer_ids = [1,4,5]
cifar100_ns_cnn_layer_legends = ['Layer 2 (NS)','Layer 4 (NS)','Layer 5 (NS)']
cifar100_s_cnn_layer_legends = ['Layer 2 (S)','Layer 4 (S)','Layer 5 (S)']
cifar100_layer_colors = ['red','blue','green','cyan','magenta']

fig, ax1 = plt.subplots(1,1)



fontsize_legend = 14
fontsize_title = 22
fontsize_ticks = 14
fontsize_label = 16
fontsize_legend_small = 10
fontsize_annotations = 18
# ======================================================================
#                       Structure growth AdaCNN (CIFAR 10)
# ======================================================================
cifar10_ns_file_name = os.path.join(log_dir, 'cnn_structure_2.log')
cifar10_ns_struct_y_data = [[] for _ in range(cifar10_cnn_layers)]
cifar10_ns_struct_x_axis = []

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

for id in range(cifar10_cnn_layers):
    ax1.plot(cifar10_ns_struct_x_axis,cifar10_ns_struct_y_data[id],label=cifar10_ns_cnn_layer_legends[id],color=cifar10_layer_colors[id],linewidth=2)

ax1.set_xlim([10000,20000])
ax1.plot([15000,15000],[20.01,140],'k--')
ax1.set_xlabel('Time ($t$)',fontsize=fontsize_label)
ax1.set_ylabel('Number of kernels',fontsize=fontsize_label)
ax1.tick_params(axis='both', which='major', labelsize=fontsize_ticks)
ax1.set_title('Structure adaptation for CIFAR10 over time ($t$) (During $\pi_{greedy}$)',fontsize=fontsize_title,y=1.05)
ax1.annotate('Class Split', xy=(15000, 125),xytext=(13500,125),fontsize=fontsize_annotations,arrowprops=dict(facecolor='black', shrink=0.05))
ax1.legend(fontsize=fontsize_legend,loc=2)


fig.subplots_adjust(wspace=0.15,hspace=0.15,bottom=0.1,top=0.88,right=0.97,left=0.07)

plt.show()