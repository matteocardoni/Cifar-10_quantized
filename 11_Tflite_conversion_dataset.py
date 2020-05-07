###Conversion of the Frozen Graph into a Tflite model

import tensorflow as tf

num_calibration_steps = 10000

def representative_dataset_gen():
  for i in range(1,num_calibration_steps - 1):
    image = train_features[i:i + 1]
    # Get sample input data as a numpy array in a method of your choosing.
    yield [image]


localpb = savedir + '/' + frozen_graph_name
tflite_file = tflite_file_name

print("{} -> {}".format(localpb, tflite_file))


converter = tf.lite.TFLiteConverter.from_frozen_graph(
    localpb, 
    ['model_input/input'], 
    ['softmax_tensor'],
)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.representative_dataset = images_batch

converter.representative_dataset = representative_dataset_gen

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8


tflite_model = converter.convert()

open(tflite_file,'wb').write(tflite_model)

interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
