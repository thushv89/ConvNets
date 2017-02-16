# ConvNets

This Repository contains the research I did relavent to CNNs. And below I will list down the important scripts and how to run them

##conv_theano
CNN implemented in Theano (Obsolete)

##conv_tensorflow
Tensorflow implementation of a CNN

conv_net_plot.py
* Contains a simple CNN. It will start a CNN of specified architecture and run it on cifar-10 dataset. The CNN architecture can be changed via setting `hyparams`,`conv_ops`,`depth_conv` parameters in the code

deconv_visualization.py
* Contains the code for visualizing high level convolutional filters using deconv [paper](https://arxiv.org/pdf/1311.2901.pdf)

experiment_inc_initialization.py (Obsolete)
* An experimental script used to test if adding layers incrementally help

experiment_inc_initialization_v2.py
* Made several improvements over the v1 script. Instead of having `conv_balance` and `pool_balance`, got rid of `conv_balance`
* Can run this using `python3 experiment_inc_initialization_v2.py --output_dir=<dir_to_save_all_logs_and_data>`
* Use CUDA_VISIBLE_DEVICES=x,y to restrict GPU usage

##conv_tensorflow/adaptive_cnn

