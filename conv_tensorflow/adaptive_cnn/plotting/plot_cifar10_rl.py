import matplotlib.pyplot as plt
import numpy as np
import os


log_dir = 'logs_to_plot'

fig, axarr = plt.subplots(1,2)


file_name = os.path.join(log_dir, 'reward.log')
ft_reward_data = []
rolling_ft_rewards = []
rolling_ft_r = 0

a2_reward_data = []
rolling_a2_rewards = []
rolling_a2_r = 0

r2_reward_data = []
rolling_r2_rewards = []
rolling_r2_r = 0

ft_x_axis = []
a2_x_axis = []
r2_x_axis = []

decay = 0.9
with open(file_name,'r') as f:
    reward_data = []
    running_reward = 0.0
    for line in f:
        if line.startswith('#'):
            continue
        else:
            line_tokens = line.split(':')

            if 'finetune' in line_tokens[2]:
                ft_x_axis.append(int(line_tokens[1]))
                reward = float(line_tokens[-1].split(',')[-1])
                ft_reward_data.append(reward)
                rolling_ft_r = reward + decay * rolling_ft_r
                rolling_ft_rewards.append(rolling_ft_r)
            if 'add' in line_tokens[2].split(',')[3]:
                a2_x_axis.append(int(line_tokens[1]))
                reward = float(line_tokens[-1].split(',')[-1])
                a2_reward_data.append(reward)
                rolling_a2_r = reward + decay * rolling_a2_r
                rolling_a2_rewards.append(rolling_a2_r)
            if 'remove' in line_tokens[2].split(',')[3]:
                r2_x_axis.append(int(line_tokens[1]))
                reward = float(line_tokens[-1].split(',')[-1])
                r2_reward_data.append(reward)
                rolling_r2_r = reward + decay * rolling_r2_r
                rolling_r2_rewards.append(rolling_r2_r)


ax1 = axarr[0]
ax2 = axarr[1]

ax1.plot(ft_x_axis, rolling_ft_rewards,label='ft')
ax1.plot(a2_x_axis, rolling_a2_rewards,label='a2')
ax1.plot(r2_x_axis, rolling_r2_rewards,label='r2')
ax1.set_xlabel('Time')
ax1.set_ylabel('Rolling Reward Mean')
ax1.legend(fontsize=12,loc=2)

ax2.plot(ft_x_axis, ft_reward_data)
ax2.plot(a2_x_axis, a2_reward_data)
ax2.plot(r2_x_axis, r2_reward_data)
ax2.legend(fontsize=12,loc=2)
plt.show()

