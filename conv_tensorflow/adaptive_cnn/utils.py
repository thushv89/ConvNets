from math import ceil,floor

def get_final_x(dataset_info,cnn_ops,cnn_hyps):
    '''
    Takes a new operations and concat with existing set of operations
    then output what the final output size will be
    :param op_list: list of operations
    :param hyp: Hyperparameters fo the operation
    :return: a tuple (width,height) of x
    '''

    x = dataset_info['image_w']
    for op in cnn_ops:
        hyp_op = cnn_hyps[op]
        if 'conv' in op:
            f = hyp_op['weights'][0]
            s = hyp_op['stride'][1]

            x = ceil(float(x)/float(s))
        elif 'pool' in op:
            if hyp_op['padding']=='SAME':
                f = hyp_op['kernel'][1]
                s = hyp_op['stride'][1]
                x = ceil(float(x)/float(s))
            elif hyp_op['padding']=='VALID':
                f = hyp_op['kernel'][1]
                s = hyp_op['stride'][1]
                x = ceil(float(x - f + 1)/float(s))

    return x


def get_ops_hyps_from_string(dataset_info,net_string,final_2d_width=1):
    # E.g. String
    # Init,0,0,0#C,1,1,64#C,5,1,64#C,5,1,128#P,5,2,0#C,1,1,64#P,2,2,0#Terminate,0,0,0

    num_channels = dataset_info['num_channels']
    image_size = dataset_info['image_w']
    num_labels = dataset_info['num_labels']

    cnn_ops = []
    cnn_hyperparameters = {}
    prev_conv_hyp = None

    op_tokens = net_string.split('#')
    depth_index = 0
    fulcon_depth_index = 0  # makes implementations easy

    last_feature_map_depth = 3  # need this to calculate the fulcon layer in size
    last_fc_out = 0

    for token in op_tokens:
        # state (layer_depth,op=(type,kernel,stride,depth),out_size)
        token_tokens = token.split(',')
        # op => type,kernel,stride,depth
        op = (token_tokens[0], int(token_tokens[1]), int(token_tokens[2]), int(token_tokens[3]))
        if op[0] == 'C':
            op_id = 'conv_' + str(depth_index)
            if prev_conv_hyp is None:
                hyps = {'weights': [op[1], op[1], num_channels, op[3]], 'stride': [1, op[2], op[2], 1],
                        'padding': 'SAME'}
            else:
                hyps = {'weights': [op[1], op[1], prev_conv_hyp['weights'][3], op[3]], 'stride': [1, op[2], op[2], 1],
                        'padding': 'SAME'}

            cnn_ops.append(op_id)
            cnn_hyperparameters[op_id] = hyps
            prev_conv_hyp = hyps  # need this to set the input depth for a conv layer
            last_feature_map_depth = op[3]
            depth_index += 1

        elif op[0] == 'P':
            op_id = 'pool_' + str(depth_index)
            hyps = {'type': 'max', 'kernel': [1, op[1], op[1], 1], 'stride': [1, op[2], op[2], 1], 'padding': 'SAME'}
            cnn_ops.append(op_id)
            cnn_hyperparameters[op_id] = hyps
            depth_index += 1

        elif op[0] == 'FC':
            if fulcon_depth_index == 0:
                if len(op_tokens) > 2:
                    output_size = get_final_x(dataset_info,cnn_ops, cnn_hyperparameters)
                # this could happen if we get terminal state without any other states in trajectory
                else:
                    output_size = image_size

                if output_size > final_2d_width:
                    cnn_ops.append('pool_global')

                    #k_size = ceil(output_size // final_2d_width) + floor(ceil(output_size // final_2d_width) // 2)
                    #s_size = k_size - floor(output_size // final_2d_width)
                    k_size,s_size = output_size // final_2d_width,output_size // final_2d_width
                    assert output_size % final_2d_width == 0
                    pg_hyps = {'type': 'avg',
                               'kernel': [1, k_size, k_size, 1],
                               'stride': [1, s_size, s_size, 1],
                               'padding': 'VALID'}
                    cnn_hyperparameters['pool_global'] = pg_hyps

            op_id = 'fulcon_' + str(fulcon_depth_index)
            # for the first fulcon layer size comes from the last convolutional layer
            if fulcon_depth_index==0:
                hyps = {'in': final_2d_width * final_2d_width * last_feature_map_depth, 'out': op[1]}
            # all the other fulcon layers the size comes from the previous fulcon layer
            else:
                hyps = {'in': cnn_hyperparameters['fulcon_'+str(fulcon_depth_index-1)]['out'], 'out': op[1]}
            cnn_ops.append(op_id)
            cnn_hyperparameters[op_id] = hyps
            fulcon_depth_index += 1
            last_fc_out = op[1]

        elif op[0] == 'Terminate':

            # if no FCs are present
            if fulcon_depth_index == 0:
                if len(op_tokens) > 2:
                    output_size = get_final_x(dataset_info, cnn_ops, cnn_hyperparameters)
                # this could happen if we get terminal state without any other states in trajectory
                else:
                    output_size = image_size

                if fulcon_depth_index==0 and output_size > final_2d_width:
                    cnn_ops.append('pool_global')
                    #k_size = ceil(output_size//final_2d_width)+floor(ceil(output_size//final_2d_width)//2)
                    #s_size = k_size - floor(output_size//final_2d_width)
                    k_size, s_size = output_size // final_2d_width, output_size // final_2d_width
                    assert output_size%final_2d_width==0
                    pg_hyps = {'type': 'avg',
                               'kernel': [1, k_size,k_size, 1],
                               'stride': [1, s_size,s_size, 1],
                               'padding': 'VALID'}
                    cnn_hyperparameters['pool_global'] = pg_hyps

                op_id = 'fulcon_out'
                if fulcon_depth_index==0:
                    hyps = {'in': final_2d_width * final_2d_width * last_feature_map_depth, 'out': num_labels}
                else:
                    hyps = {'in':cnn_hyperparameters['fulcon_'+str(depth_index-1)]['out'],'out':num_labels}

                cnn_ops.append(op_id)
                cnn_hyperparameters[op_id] = hyps

            else:
                op_id = 'fulcon_out'
                hyps = {'in': last_fc_out, 'out': num_labels}
                cnn_ops.append(op_id)
                cnn_hyperparameters[op_id] = hyps

        elif op[0] == 'Init':
            continue
        else:
            print('=' * 40)
            print(op[0])
            print('=' * 40)
            raise NotImplementedError

    return cnn_ops, cnn_hyperparameters


def get_cnn_string_from_ops(cnn_ops, cnn_hyps):
    current_cnn_string = ''
    for op in cnn_ops:
        if 'conv' in op:
            current_cnn_string += '#C,' + str(cnn_hyps[op]['weights'][0]) + ',' + str(
                cnn_hyps[op]['stride'][1]) + ',' + str(cnn_hyps[op]['weights'][3])
        elif 'pool' in op:
            current_cnn_string += '#P,' + str(cnn_hyps[op]['kernel'][0]) + ',' + str(
                cnn_hyps[op]['stride'][1]) + ',' + str(0)
        elif 'fulcon_out' in op:
            current_cnn_string += '#Terminate,0,0,0'
        elif 'fulcon' in op:
            current_cnn_string += '#FC,' + str(cnn_hyps[op]['in'])

    return current_cnn_string