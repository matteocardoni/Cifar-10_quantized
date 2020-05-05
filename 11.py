###Conversion of the Frozen Graph into a Tflite model

import tensorflow as tf

localpb = savedir + '/' + frozen_graph_name
tflite_file = tflite_file_name

print("{} -> {}".format(localpb, tflite_file))


converter = tf.lite.TFLiteConverter.from_frozen_graph(
    localpb, 
    ['model_input/input'], 
    ['softmax_tensor'],
)


converter.inference_input_type = tf.uint8 #Declare that the input inference is of type uint8
converter.quantized_input_stats = {"model_input/input":(128,127)} #Mean (first value) and standar deviation (second value) for the input tensor.
                                                                  #These values are necessary if inference_input_type is QUANTIZED_UINT8
converter.inference_type = tf.uint8 #Declare that the output inference is of type uint8


tflite_model = converter.convert()

open(tflite_file,'wb').write(tflite_model)

interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()