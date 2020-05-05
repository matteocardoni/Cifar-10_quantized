###Frozen Graph conversion

# This function returns the path of the saved model, named saved_model.pb, that is in a timestamp folder.
def find_saved_model_path(folder):
# traverse root directory, and list directories as dirs and files as files
    for root, dirs, files in os.walk("."):
        path = root.split(os.sep)
        for file in files:
            if file == 'saved_model.pb':  # to check if there's a file named saved_model.pb
                path_length = len(path)  # length of the path corresponding to the file saved_model.pb
                folder_name = folder + '/' + path[
                    path_length - 1]  # the name of the folder is the element of number (path_length -1)
    return (folder_name)

folder_name = find_saved_model_path(savedir) #path for the saved model

sess = tf.InteractiveSession()

tf.saved_model.loader.load(sess,["serve"],folder_name)

graph = tf.get_default_graph()

from tensorflow.python.tools import freeze_graph
from tensorflow.python.saved_model import tag_constants

input_graph_filename = None
input_saved_model_dir = folder_name
output_node_names = "softmax_tensor"
output_graph_filename = os.path.join(savedir, frozen_graph_name)
input_binary = False
input_saver_def_path = False
restore_op_name = None
filename_tensor_name = None
clear_devices = True
input_meta_graph = False
checkpoint_path = None

saved_model_tags = tag_constants.SERVING

freeze_graph.freeze_graph(input_graph_filename, input_saver_def_path,
                            input_binary, checkpoint_path, output_node_names,
                              restore_op_name, filename_tensor_name,
                              output_graph_filename, clear_devices, "", "", "",
                              input_meta_graph, input_saved_model_dir,
                            saved_model_tags)