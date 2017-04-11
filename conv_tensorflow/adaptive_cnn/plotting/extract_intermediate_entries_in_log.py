import os

filtered_lines = []

with open('Error.log','r') as f:
    for line in f:
        if line.startswith('#'):
            filtered_lines.append(line)
            print(line,end='')
        else:
            if len(line)<=0:
                continue
            batch_id = int(line.split(',')[0])
            if (batch_id+1)%100==0:
                filtered_lines.append(line)
                print(line,end='')
