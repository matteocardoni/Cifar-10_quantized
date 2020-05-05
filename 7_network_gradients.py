###Training and saving in pb format

run_cfg = tf.estimator.RunConfig(
    model_dir=savedir,
    tf_random_seed=2,
    save_summary_steps=2,
    # session_config = sess_config,
    save_checkpoints_steps=100,
    keep_checkpoint_max=1)

# Instantiate the Estimator
cifar10_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, config=run_cfg)

# Set up logging for predictions
tensors_to_log = {"probabilities": "softmax_tensor"}

#function to print the softmax tensor evey 1000 steps 
logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=1000)


#core function for the training. Executes the training for a number of epochs corresponding to xEpochs
def fit_all_batches(xEpochs):
    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=train_features,
        y=train_labels,
        batch_size=32,
        num_epochs=xEpochs,
        shuffle=False)

    # train one step and display the probabilties
    cifar10_classifier.train(
        input_fn=train_input_fn,
        steps=None,
        hooks=[logging_hook])

#calls the function fit_all_batches for a number of epochs corresponding to num_epochs (variable declarated in a window above)
fit_all_batches(num_epochs)

defaultGraph = tf.get_default_graph()


#The evaluation function is setted for only one epoch
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x=valid_features,
    y=valid_labels,
    num_epochs=1,
    shuffle=False)

#calling of the function eval_input_fn, to obtain accuracy and loss results
eval_results = cifar10_classifier.evaluate(input_fn=eval_input_fn,
                                           hooks=[logging_hook])
#print the accuracy and loss results
print(eval_results)


#Save the model
def serving_input_receiver_fn():
    feature_tensor = tf.placeholder(tf.float32, [None, 32, 32, 3])
    return tf.estimator.export.TensorServingInputReceiver(feature_tensor, {'input': feature_tensor})


input_receiver_fn_map = {
    tf.estimator.ModeKeys.PREDICT: serving_input_receiver_fn,
}

#this function saves the model in a timestemp folder
cifar10_classifier.experimental_export_all_saved_models(
    savedir,
    input_receiver_fn_map)