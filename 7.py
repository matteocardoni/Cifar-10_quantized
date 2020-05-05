###Network definition

def cnn_model_fn(features, labels, mode):
    # definition of the input layer
    with tf.name_scope('model_input') as scope:
        input_layer = tf.reshape(features, [-1, 32, 32, 3], name="input")

    # definition of the 1st convolution layer, whith a number of kernels corresponding to the variable kernel1
    # and a kernel size corresponding to the variable kernel_size
    with tf.name_scope('model_conv1') as scope:
        quant1_conv1 = tf.quantization.fake_quant_with_min_max_args(inputs=input_layer, min=0, max=1, num_bits=8)
        conv1 = tf.layers.conv2d(inputs=quant1_conv1, filters=kernel1, kernel_size=kernel_size,
                                 padding="same", activation=tf.nn.relu,
                                 trainable=mode == tf.estimator.ModeKeys.TRAIN)
        quant2_conv1 = tf.quantization.fake_quant_with_min_max_args(inputs=conv1, min=0, max=6, num_bits=8)

    # definition of the 2nd convolution layer, whith a number of kernels corresponding to the variable kernel2
    # and a kernel size corresponding to the variable kernel_size
    with tf.name_scope('model_conv2') as scope:
        conv2 = tf.layers.conv2d(inputs=quant2_conv1, filters=kernel2, kernel_size=kernel_size,
                                 padding="same", activation=tf.nn.relu,
                                 trainable=mode == tf.estimator.ModeKeys.TRAIN)
        quant1_conv2 = tf.quantization.fake_quant_with_min_max_args(inputs=conv2, min=0, max=6, num_bits=8)
        # Max pooling with a (2x2) filter
        pool1 = tf.layers.max_pooling2d(inputs=quant1_conv2, pool_size=[2, 2], strides=(2, 2))
        # Dropout with a rate of 0.2
        drop1 = tf.layers.dropout(inputs=pool1, rate=0.2)

    # definition of the 3rd convolution layer, whith a number of kernels corresponding to the variable kernel3
    # and a kernel size corresponding to the variable kernel_size
    with tf.name_scope('model_conv3') as scope:
        conv3 = tf.layers.conv2d(inputs=drop1, filters=kernel3, kernel_size=kernel_size,
                                 padding="same", activation=tf.nn.relu,
                                 trainable=mode == tf.estimator.ModeKeys.TRAIN)
        quant1_conv3 = tf.quantization.fake_quant_with_min_max_args(inputs=conv3, min=0, max=6, num_bits=8)

    # definition of the 4th convolution layer, whith a number of kernels corresponding to the variable kernel4
    # and a kernel size corresponding to the variable kernel_size
    with tf.name_scope('model_conv4') as scope:
        conv4 = tf.layers.conv2d(inputs=quant1_conv3, filters=kernel4, kernel_size=kernel_size,
                                 padding="same", activation=tf.nn.relu,
                                 trainable=mode == tf.estimator.ModeKeys.TRAIN)
        quant1_conv4 = tf.quantization.fake_quant_with_min_max_args(inputs=conv4, min=0, max=6, num_bits=8)
        # Max pooling with a (2x2) filter
        pool2 = tf.layers.max_pooling2d(inputs=quant1_conv4, pool_size=[2, 2], strides=(2, 2))
        # Dropout with a rate of 0.3
        drop2 = tf.layers.dropout(inputs=pool2, rate=0.3)

    # definition of the 5th convolution layer, whith a number of kernels corresponding to the variable kernel5
    # and a kernel size corresponding to the variable kernel_size
    with tf.name_scope('model_conv5') as scope:
        conv5 = tf.layers.conv2d(inputs=drop2, filters=kernel5, kernel_size=kernel_size,
                                 padding="same", activation=tf.nn.relu,
                                 trainable=mode == tf.estimator.ModeKeys.TRAIN)
        quant1_conv5 = tf.quantization.fake_quant_with_min_max_args(inputs=conv5, min=0, max=6, num_bits=8)

    # definition of the 6th convolution layer, whith a number of kernels corresponding to the variable kernel6
    # and a kernel size corresponding to the variable kernel_size
    with tf.name_scope('model_conv6') as scope:
        conv6 = tf.layers.conv2d(inputs=quant1_conv5, filters=kernel6, kernel_size=kernel_size,
                                 padding="same", activation=tf.nn.relu,
                                 trainable=mode == tf.estimator.ModeKeys.TRAIN)
        quant1_conv6 = tf.quantization.fake_quant_with_min_max_args(inputs=conv6, min=0, max=6, num_bits=8)
        # Max pooling with a (2x2) filter
        pool3 = tf.layers.max_pooling2d(inputs=quant1_conv6, pool_size=[2, 2], strides=(2, 2))
        # Dropout with a rate of 0.4
        drop3 = tf.layers.dropout(inputs=pool3, rate=0.4)

    with tf.name_scope('model_dense') as scope:
        # definition of the flatten layere.
        # Every image is now a matrix of dimension (4x4), so the vector will be of lenght 4*4 times the number
        # of kernels in the last convolution layer, that corresponds to the variable kernel6
        flat = tf.reshape(drop3, [-1, 4 * 4 * kernel6])

        # definition of the hidden layer, with a number of node corresponding to the variable hiddel_layer
        dense = tf.layers.dense(inputs=flat, units=hidden_layer_nodes,
                                activation=tf.nn.relu,
                                trainable=mode == tf.estimator.ModeKeys.TRAIN)

        quant_dense = tf.quantization.fake_quant_with_min_max_args(inputs=dense, min=0, max=6, num_bits=8)

        ##Dropout with a rate of 0.5
        drop4 = tf.layers.dropout(inputs=quant_dense, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # output layer definition
    with tf.name_scope('model_output') as scope:
        logits = tf.layers.dense(inputs=drop4, units=10, trainable=mode == tf.estimator.ModeKeys.TRAIN)
        quant_logits = tf.quantization.fake_quant_with_min_max_args(inputs=logits, min=0, max=6, num_bits=8)

    predictions = {
        "classes": tf.argmax(input=quant_logits, axis=1),
        "probabilities": tf.nn.softmax(quant_logits, name="softmax_tensor")
    }

    # PREDICT mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        export_outputs = {
            'predict_output': tf.estimator.export.PredictOutput(predictions)
        }

        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=quant_logits)

    # TRAIN mode
    if mode == tf.estimator.ModeKeys.TRAIN:
        g = tf.get_default_graph()
        # function for the quantization aware training in train mode
        tf.contrib.quantize.create_training_graph(input_graph=g, quant_delay=quant_delay)
        # definition of the optimizer and the learning rate
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # EVAL mode
    g = tf.get_default_graph()
    # function for the quantization aware training in evaluation mode
    tf.contrib.quantize.create_eval_graph(input_graph=g)
    labels = tf.argmax(labels, axis=1)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)