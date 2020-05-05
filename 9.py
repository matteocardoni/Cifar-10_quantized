#prediction of the image numbered nfeature in the processed dataset
def predictions(nfeature):
    pred_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=valid_features[nfeature:nfeature + 1],
        y=valid_labels[nfeature:nfeature + 1],
        num_epochs=1,
        shuffle=False)

    pred_results = cifar10_classifier.predict(
        input_fn=pred_input_fn,
        hooks=[logging_hook])

    results = pred_results
    results = list(results)

    results = results[0]
    
    y_pred_flat = results['probabilities']

    y_pred = np.column_stack((label_names, y_pred_flat))
    print(y_pred)
    plt.imshow(valid_features[nfeature])

predictions(6)

[print(n.name) for n in defaultGraph.as_graph_def().node]