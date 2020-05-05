###Download, directories and model saving settings 

url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz' #Url for Cifar-10 download (Python version)

work_directory = '/content'
download_directory = work_directory #directory for the download of the zipped dataset
Datasets_folder = work_directory  #directory for the download of the unzipped dataset

zipped_dataset_name = 'dataset.tar.gz'
unzipped_dataset_name = 'dataset'

#Directory place the dataset
Cifar_10_dir_path = download_directory + unzipped_dataset_name + '/cifar-10-batches-py'

#Directory to save the model
savedir = work_directory + '/Quantized_model'

#Frozen Graph name
frozen_graph_name = 'frozen_graph.pb'

#Name of the Tflite saved model
tflite_file_name = savedir + '/cifar10_quantized_min_max.tflite'