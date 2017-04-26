# Read multiple reward logs and calculate moving average of the reward and plot it over time
import matplotlib.pyplot as plt
import numpy as np
import os

operations = ['rewards']

log_dir = 'logs_to_plot'
decay = 0.95

fig, axarr = plt.subplots(1,2)

if 'rewards' in operations:
    file_names = [os.path.join(log_dir, 'reward_d5.log'), os.path.join(log_dir, 'reward_d7.log'),
                  os.path.join(log_dir, 'reward_d9.log')]

    legends = ['discount 0.5', 'discount 0.7', 'discount 0.9']
    data = []
    for f_i,fname in enumerate(file_names):
        with open(fname,'r') as f:
            reward_data = []
            running_reward = 0.0
            for line in f:
                if line.startswith('#'):
                    continue
                else:
                    running_reward = float(line.split(',')[1]) + decay * running_reward
                    reward_data.append(running_reward)

            data.append(reward_data)

    x_axis = np.arange(0,len(data[0]))

    ax1 = axarr[0]

    for i in range(len(file_names)):
        ax1.plot(x_axis, data[i], label=legends[i])
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Rolling Reward Mean')
        print('Reward (%s): %.3f'%(legends[i],data[i][-1]))
    ax1.legend(fontsize=12,loc=2)


if 'q' in operations:
    file_names = [os.path.join(log_dir, 'QMetric_128.log'), os.path.join(log_dir, 'QMetric_128_64.log'),
                  os.path.join(log_dir, 'QMetric_128_64_32.log')]

    legends = ['128', '128,64', '128,64,32']
    data = []
    for f_i, fname in enumerate(file_names):
        with open(fname, 'r') as f:
            q_data = []
            running_reward = 0.0
            for line in f:
                if line.startswith('#'):
                    continue
                else:
                    running_reward = float(line.split(',')[1]) + decay * running_reward
                    q_data.append(running_reward)

            data.append(q_data)

    x_axis = np.arange(0, len(data[0]))
    ax2 = axarr[1]

    for i in range(len(file_names)):
        ax2.plot(x_axis, data[i], label=legends[i])
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Rolling Q Mean')
        print('Q (%s): %.3f'%(legends[i], data[i][-1]))
    ax2.legend(fontsize=12, loc=2)

plt.show()
