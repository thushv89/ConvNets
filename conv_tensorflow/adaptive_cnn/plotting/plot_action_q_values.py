import matplotlib.pyplot as plt
import numpy as np
import os

operations = ['cifar10_nonstationary','cifar10_stationary']

log_dir = 'logs_to_plot'


fig, axarr = plt.subplots(2,2)

if 'cifar10_nonstationary' in operations:
    file_names = [os.path.join(log_dir, 'predicted_q_cifar10_inc_nonstationary.log'),
                  os.path.join(log_dir, 'predicted_q_cifar10_inc_stationary.log')]

    legends = ['cifar10-inc','cifar10-noninc']
    actions_list = ['rnn','nrn','nnr','ann','nan','nna','nnn','fff']
    data = []
    for f_i,fname in enumerate(file_names):
        with open(fname,'r') as f:
            error_data = []
            running_error = 0.0
            for line in f:
                if line.startswith('#'):
                    continue
                else:
                    running_error = decay * float(line.split(',')[3])
                    error_data.append(running_error)

            data.append(error_data)

    x_axis = np.arange(0,len(data[0]))

    ax1 = axarr[0][0]

    for i in range(len(file_names)):
        ax1.plot(x_axis, data[i], label=legends[i])
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Rolling Error')

    ax1.legend(fontsize=12,loc=2)

if 'cifar100_nonstationary' in operations:
    file_names = [os.path.join(log_dir, 'Error_cifar100_inc_nonstationary.log'),
                  os.path.join(log_dir, 'Error_cifar100_noninc_nonstationary.log')]

    legends = ['cifar100-inc','cifar100-noninc']
    data = []
    for f_i,fname in enumerate(file_names):
        with open(fname,'r') as f:
            error_data = []
            running_error = 0.0
            for line in f:
                if line.startswith('#'):
                    continue
                else:
                    running_error = decay * float(line.split(',')[3]) + (1.0-decay) * running_error
                    error_data.append(running_error)

            data.append(error_data)

    x_axis = np.arange(0,len(data[0]))

    ax1 = axarr[1][0]

    for i in range(len(file_names)):
        ax1.plot(x_axis, data[i], label=legends[i])
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Rolling Error')

    ax1.legend(fontsize=12,loc=2)


plt.show()