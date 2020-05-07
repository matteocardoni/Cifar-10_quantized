###Trainig and network parameters

num_epochs = 1 #number of epochs for the training

#The network is structured with 6 convolutional layers and one hidden layer after the convolution layers.

kernel1 = 32  #number of kernels in the 1st convolutional layer
kernel2 = 32  #number of kernels in the 2nd convolutional layer
kernel3 = 64  #number of kernels in the 3rd convolutional layer
kernel4 = 64  #number of kernels in the 4th convolutional layer
kernel5 = 128  #number of kernels in the 5th convolutional layer
kernel6 = 128  #number of kernels in the 6th convolutional layer
hidden_layer_nodes = 128  #numero of nodes in the hidden layer

kernel_size = [3, 3]  #Size for the kernel in every convolution layer

quant_delay = 2000000000000 #quantization delay: number of steps after which weights and activations are quantized during training.
                          #cannot be declared with exponential, because it has to be of type int
learning_rate = 0.0001

num_calibration_steps = 10000
