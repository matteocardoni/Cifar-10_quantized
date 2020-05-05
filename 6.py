###Dataset processing

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

#cifar10_1 = unpickle(Cifar_10_dir_path + '/data_batch_1')

cifar10_dataset_folder_path = Cifar_10_dir_path

cifar10_Label_names = unpickle(Cifar_10_dir_path + '/batches.meta')
cifar10_Label_names[b'label_names']

label_names = []
for x in cifar10_Label_names[b'label_names']:
    label = x.decode()
    label_names.append(label)

label_names


def load_cfar10_batch(cifar10_dataset_folder_path, batch_id):
    with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')
    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']
    return features, labels


def normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x - min_val) / (max_val - min_val)
    x = x.astype('float32')
    return x


def one_hot_encode(x):
    encoded = np.zeros((len(x), 10))
    for idx, val in enumerate(x):
        encoded[idx][val] = 1
    return encoded


def _preprocess_and_save(normalize, one_hot_encode, features, labels, filename):
    features = normalize(features)
    labels = one_hot_encode(labels)
    pickle.dump((features, labels), open(filename, 'wb'))


def preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode):
    n_batches = 5
    valid_features = []
    valid_labels = []
    all_features = []
    all_labels = []

    for batch_i in range(1, n_batches + 1):
        features, labels = load_cfar10_batch(cifar10_dataset_folder_path, batch_i)

        # find index to be the point as validation data in the whole dataset of the batch (10%)
        index_of_validation = int(len(features) * 0.1)

        valid_features.extend(features[-index_of_validation:])
        valid_labels.extend(labels[-index_of_validation:])
        all_features.extend(features[:-index_of_validation])
        all_labels.extend(labels[:-index_of_validation])

    # preprocess the all stacked validation dataset
    _preprocess_and_save(normalize, one_hot_encode,
                         np.array(valid_features), np.array(valid_labels),
                         Datasets_folder + '/preprocess_validation.p')

    _preprocess_and_save(normalize, one_hot_encode,
                         np.array(all_features), np.array(all_labels),
                         Datasets_folder + '/preprocess_all.p')


preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode)

#Print information about the processed dataset
valid_features, valid_labels = pickle.load(open(Datasets_folder + '/preprocess_validation.p', mode='rb'))
print("Format of the test images: " + str(valid_features.dtype))
print("Number and shape of test images: " + str(valid_features.shape))

train_features, train_labels = pickle.load(open(Datasets_folder + '/preprocess_all.p', mode='rb'))
print("Fromat of the train images: " + str(train_features.dtype))
print("Number and shape of the train images: " + str(train_features.shape))